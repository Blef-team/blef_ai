"""Instrumented OS trainer that logs every visit to a target infoset.

Built to investigate WHY OS converges to "bluff A-exists" on (1,1) hand=Q
instead of the obviously dominant "Q-exists" action, even when our analytic
per-iteration drift says action 3 should win.

Records, per visit to the target infoset:
  - which player is active, whether it's the traverser this iter
  - σ at this node (regret-matched, before update)
  - which action a* was sampled (after ε-on-policy mixing)
  - the IS-corrected utility from the recursion
  - coef_sampled / coef_unsampled if regret was updated
  - regrets for a few interesting actions before/after the update

Output: a per-visit dataframe printed to stdout; plus a summary that tracks
how the regrets of selected actions evolve over the course of training.
"""
from __future__ import annotations

import collections
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

from cfr_ai.information_set import InformationSet, make_key, get_hand_abstraction
from cfr_ai.game import Game


ACTION_LABELS = {
    0: "9-exists", 1: "10-exists", 2: "J-exists", 3: "Q-exists",
    4: "K-exists", 5: "A-exists",
    6: "pair-9", 7: "pair-10", 8: "pair-J", 9: "pair-Q",
    10: "pair-K", 11: "pair-A",
    87: "GSF-spades",
}
WATCH_ACTIONS = [0, 3, 4, 5, 87]


class TracingOSTrainer:
    def __init__(self, hand_sizes, min_bet, target_key, eps=0.6):
        self.infoset_map: Dict[str, InformationSet] = {}
        self.hand_sizes = hand_sizes
        self.min_bet = min_bet
        self.exploration = eps
        self.target_key = target_key
        self.traces: List[dict] = []
        self.nodes_touched = 0

    def _walk(self, hands, hand_abstractions, history, active, traverser,
              my_reach, opp_reach, sample_reach, existence_array, iter_num):
        if Game.check_finish(history):
            u_active = 1.0 if existence_array[history[-2]] else -1.0
            u_traverser = u_active if active == traverser else -u_active
            return u_traverser / sample_reach, 1.0, u_traverser

        key = make_key(hands[active], hand_abstractions[active], history, self.min_bet)
        if key not in self.infoset_map:
            self.infoset_map[key] = InformationSet(history, iter_num, self.min_bet)
        info_set = self.infoset_map[key]

        actions = info_set.possible_actions
        n = len(actions)
        sigma = info_set.get_strategy(0.0)

        if active == traverser:
            sigma_tilde = (1.0 - self.exploration) * sigma + self.exploration / n
        else:
            sigma_tilde = sigma

        a_idx = random.choices(range(n), weights=sigma_tilde, k=1)[0]
        action = actions[a_idx]

        is_target = key == self.target_key
        pre_state = None
        if is_target:
            pre_state = {
                "iter": iter_num,
                "active": active,
                "traverser": traverser,
                "a_idx": a_idx,
                "sigma_a_sampled": float(sigma[a_idx]),
                "sigma_tilde_a_sampled": float(sigma_tilde[a_idx]),
                "sample_reach_in": sample_reach,
                "opp_reach": opp_reach,
                **{f"sigma_{a}": float(sigma[a]) for a in WATCH_ACTIONS},
                **{f"r_{a}_before": float(info_set.regrets[a]) for a in WATCH_ACTIONS},
            }

        opp = (active + 1) % 2
        if active == traverser:
            new_my_reach = my_reach * sigma[a_idx]
            new_opp_reach = opp_reach
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * sigma[a_idx]
        new_sample_reach = sample_reach * sigma_tilde[a_idx]

        util, tail, raw_u = self._walk(
            hands, hand_abstractions, history + [action], opp, traverser,
            new_my_reach, new_opp_reach, new_sample_reach,
            existence_array, iter_num,
        )

        if active == traverser:
            scale = opp_reach * tail
            coef_sampled = (1.0 - sigma[a_idx]) * util * scale
            coef_unsampled = -sigma[a_idx] * util * scale
            for i in range(n):
                info_set.regrets[i] += coef_sampled if i == a_idx else coef_unsampled
            info_set.strategy_sum += (my_reach * sigma).astype(np.float32)
            new_tail = tail * sigma[a_idx]
        else:
            info_set.strategy_sum += (opp_reach * sigma).astype(np.float32)
            new_tail = tail

        if is_target and pre_state is not None:
            self.traces.append({
                **pre_state,
                "util": float(util),
                "raw_u": float(raw_u),
                "tail_returned": float(tail),
                "scale": float(opp_reach * tail),
                "coef_sampled": float(coef_sampled) if active == traverser else 0.0,
                "coef_unsampled": float(coef_unsampled) if active == traverser else 0.0,
                **{f"r_{a}_after": float(info_set.regrets[a]) for a in WATCH_ACTIONS},
            })

        info_set.times_touched += 1
        info_set.last_touched = iter_num
        self.nodes_touched += 1
        return util, new_tail, raw_u

    def train(self, num_iter):
        for i in range(num_iter):
            traverser = (i // 2) % 2
            starting_player = i % 2
            hands = Game.deal_cards(self.hand_sizes)
            existence_array = Game.precompute_set_existence(hands)
            hand_abstractions = [get_hand_abstraction(h, self.hand_sizes) for h in hands]
            self._walk(
                hands, hand_abstractions, [], starting_player, traverser,
                1.0, 1.0, 1.0, existence_array, i,
            )


def fmt_action(a):
    return f"{a}({ACTION_LABELS.get(a, '?')})"


def main():
    import psutil
    psutil.Process().nice(psutil.NORMAL_PRIORITY_CLASS)

    NUM_ITER = 100_000
    TARGET = "1-88-3"  # root infoset, hand=Q

    t0 = time.time()
    tr = TracingOSTrainer([1, 1], 0, TARGET)
    tr.train(NUM_ITER)
    dt = time.time() - t0
    print(f"Trained {NUM_ITER:,} iter in {dt:.0f}s "
          f"({NUM_ITER/dt:.0f} it/s); traces at {TARGET}: {len(tr.traces)}",
          flush=True)

    # Split traces by traverser vs opp visit
    tv = [x for x in tr.traces if x["active"] == x["traverser"]]
    print(f"Traverser visits (regret updates here): {len(tv)}", flush=True)

    # 1. Sampling distribution at this infoset (traverser visits only)
    counter = collections.Counter(x["a_idx"] for x in tv)
    print(f"\n[1] What actions did the traverser sample at this infoset?")
    print(f"     (under eps-on-policy mixing with sigma_traverser)")
    print(f"  {'action':<22} {'count':>7}  {'pct':>5}")
    for a, c in counter.most_common(10):
        print(f"  {fmt_action(a):<22} {c:>7}  {100*c/len(tv):>4.1f}%")

    # 2. Mean util when each watched action was sampled
    print(f"\n[2] Mean util (= u_traverser / sample_reach_at_term) per sampled action:")
    print(f"     (high util magnitude = sampled action drove deep into unlikely subtree)")
    print(f"  {'action':<22} {'count':>7}  {'mean util':>11}  {'mean raw_u':>11}")
    for a in WATCH_ACTIONS:
        utils = [x["util"] for x in tv if x["a_idx"] == a]
        raws = [x["raw_u"] for x in tv if x["a_idx"] == a]
        if utils:
            print(f"  {fmt_action(a):<22} {len(utils):>7}  "
                  f"{np.mean(utils):>+11.2f}  {np.mean(raws):>+11.3f}")
        else:
            print(f"  {fmt_action(a):<22} {0:>7}  {'n/a':>11}")

    # 3. Mean coef contribution to each watched action
    print(f"\n[3] Per-visit mean dRegret contribution to action k from each a*:")
    print(f"     (coef_sampled if k==a*, else coef_unsampled = -sigma(a*)*util*scale)")
    print(f"  {'a* sampled':<22}", end="")
    for k in WATCH_ACTIONS:
        print(f"  {'dr('+str(k)+')':>9}", end="")
    print()
    for a_star in WATCH_ACTIONS + sorted({x["a_idx"] for x in tv if x["a_idx"] not in WATCH_ACTIONS})[:5]:
        rows = [x for x in tv if x["a_idx"] == a_star]
        if not rows:
            continue
        print(f"  {fmt_action(a_star):<22}", end="")
        for k in WATCH_ACTIONS:
            if k == a_star:
                vals = [x["coef_sampled"] for x in rows]
            else:
                vals = [x["coef_unsampled"] for x in rows]
            print(f"  {np.mean(vals):>+9.2f}", end="")
        print()

    # 4. Regret trajectory over visits -- sample 25 points
    print(f"\n[4] Regret trajectory over traverser visits at {TARGET} (Hand=Q root):")
    print(f"  {'visit':>6} {'iter':>7}", end="")
    for a in WATCH_ACTIONS:
        print(f"  {'r('+str(a)+')':>10}", end="")
    print(f"  {'sigma(3)':>9} {'sigma(5)':>9}")
    sample_indices = list(range(0, len(tv), max(1, len(tv) // 25))) + [len(tv) - 1]
    for i in sample_indices:
        if i >= len(tv):
            continue
        x = tv[i]
        print(f"  {i:>6} {x['iter']:>7}", end="")
        for a in WATCH_ACTIONS:
            print(f"  {x[f'r_{a}_after']:>10.1f}", end="")
        print(f"  {x['sigma_3']:>9.3f} {x['sigma_5']:>9.3f}")

    # 5. When did action 5 first get positive regret? action 3?
    print(f"\n[5] First visit at which each action's regret crossed above 0:")
    for a in WATCH_ACTIONS:
        first_pos = next((i for i, x in enumerate(tv) if x[f"r_{a}_after"] > 0), None)
        print(f"  action {fmt_action(a):<22}: visit {first_pos if first_pos is not None else 'never positive'}")


if __name__ == "__main__":
    main()
