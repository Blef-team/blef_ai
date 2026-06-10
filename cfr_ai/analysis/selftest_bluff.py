"""MC self-test: macro-ON vs macro-OFF, SAME bluff model.

Both agents read the same `strategy.bluff.npz`. One resolves the macro to the
hand's bluffiest legal bet b* (= argmin p-g); the other ignores the macro and
renormalises the concrete probs. So the ONLY difference is the macro — identical
training, identical concrete policy, identical (un)cleaning. This isolates the
macro's marginal value with zero confounds and needs no external baseline.

Monte-Carlo playouts (sample one action per node) -> fast and robust to dense
(un-clear_lows'd) strategies, unlike exact full-EV which explodes on them.

Per deal we play macro-ON at seat 0 vs macro-OFF at seat 1, AND the swap, then
score the macro-ON agent's payoff averaged over both seats (so the result is
symmetric over mover order). advantage in [-1,1]; + => the macro helps.
"""
import argparse
import os
import random
from functools import lru_cache

import numpy as np

from cfr_ai.game import Game
import cfr_ai.information_set as ISM
from cfr_ai.analysis.head_to_head import (
    make_key_wrapper, get_possible_actions_wrapper,
)
from cfr_ai.abstraction.probs import g_vector
from cfr_ai.abstraction.probs_jit import p_vector_fast
from cfr_ai.encoding import clear_lows
from cfr_ai.keys import (
    LAST_BET_SHIFT, H_M1_SHIFT, H_M2_SHIFT, ABS_ID_SHIFT, ABSENT_CODE,
    _HISTORY_CODE_STRS,
)


def load_bluff(folder, setup):
    d = np.load(os.path.join(folder, "outputs", setup, "strategy.bluff.npz"),
                allow_pickle=True)
    keys, probs, macro = d["keys"], d["probs"], d["macro"]
    table = {str(keys[i]): (probs[i], float(macro[i])) for i in range(len(keys))}
    macro_kind = str(d["macro_kind"]) if "macro_kind" in d.files else "bluff"
    return table, int(d["min_bet"]), [int(x) for x in d["hand_sizes"]], macro_kind


def load_standard(setup_dir):
    """Load a standard (e.g. V2.1) strategy.npz as {key_str: (lo, hi, probs)}.
    Mirrors head_to_head.preload_all_strategies' key reconstruction."""
    from cfr_ai.strategy_io import load_strategy
    fs = load_strategy(setup_dir)
    id_to_abs = {v: k for k, v in fs.abs_str_to_id.items()}
    table = {}
    for k_int, row in fs.key_to_row.items():
        k, row = int(k_int), int(row)
        hs = k & 0xF
        lb = (k >> LAST_BET_SHIFT) & 0xFF
        h1 = (k >> H_M1_SHIFT) & 0xFF
        h2 = (k >> H_M2_SHIFT) & 0xFF
        abs_id = k >> ABS_ID_SHIFT
        ks = f"{hs}-{lb}-"
        if h1 != ABSENT_CODE:
            ks += _HISTORY_CODE_STRS[h1] + "-"
            if h2 != ABSENT_CODE:
                ks += _HISTORY_CODE_STRS[h2] + "-"
        ks += id_to_abs[abs_id]
        lo, hi = int(fs.lower_action[row]), int(fs.upper_action[row])
        table[ks] = (lo, hi, fs.strategy[row, lo:hi + 1].astype(np.float64))
    return table


def std_dist(table, my_hand, hand_sizes, history, min_bet):
    """Distribution for a standard model (cleaned, no macro)."""
    possible = get_possible_actions_wrapper(ISM, history, min_bet)
    abst = ISM.get_hand_abstraction(my_hand, hand_sizes)
    key = make_key_wrapper(ISM, my_hand, abst, history, min_bet)
    num = len(possible)
    e = table.get(key)
    if e is None:
        s = np.zeros(num); s[-1] = 1.0; return possible, s
    lo, _hi, probs = e
    out = np.zeros(num)
    for i, a in enumerate(possible):
        j = a - lo
        if 0 <= j < len(probs):
            out[i] = probs[j]
    tot = out.sum()
    if tot <= 0.0:
        out = np.zeros(num); out[-1] = 1.0
    else:
        out = out / tot
    return possible, out


@lru_cache(maxsize=None)
def _score(hand_tuple, total, kind):
    """argmax'd to pick the macro's target bet (matches training). value => p
    (most probable); difftruthy => p-g (most over-supported); bluff => g-p
    (argmax = argmin(p-g) = bluffiest)."""
    pv = p_vector_fast(list(hand_tuple), total - len(hand_tuple))
    if kind == "value":
        return pv
    if kind == "difftruthy":
        return pv - g_vector(total)
    return g_vector(total) - pv


def node_dist(table, my_hand, hand_sizes, history, min_bet, total, macro_on,
              macro_kind="bluff", clean=False):
    possible = get_possible_actions_wrapper(ISM, history, min_bet)
    abst = ISM.get_hand_abstraction(my_hand, hand_sizes)
    key = make_key_wrapper(ISM, my_hand, abst, history, min_bet)
    e = table.get(key)
    num = len(possible)
    if e is None:
        s = np.zeros(num); s[-1] = 1.0; return possible, s
    probs_full, macro = e
    out = np.array([probs_full[a] for a in possible], dtype=np.float64)
    if macro_on and macro > 0.0:
        score = _score(tuple(my_hand), total, macro_kind)
        b_star, best, ties = -1, -1e18, 0   # argmax with RANDOM tie-break
        for a in possible:
            if a <= 87:
                s = score[a]
                if s > best + 1e-9:
                    best, b_star, ties = s, a, 1
                elif s > best - 1e-9:
                    ties += 1
                    if random.random() * ties < 1.0:
                        b_star = a
        if b_star >= 0:
            out[possible.index(b_star)] += macro
    tot = out.sum()
    if tot <= 0.0:
        out = np.zeros(num); out[-1] = 1.0
        return possible, out
    out = out / tot
    if clean:                       # match V2.1's deployment cleaning
        out = clear_lows(out)
    return possible, out


def playout(policies, hands, existence):
    """policies[seat](my_hand, history) -> (possible, dist). Returns payoff to
    seat 0 (+1/-1)."""
    history, active = [], 0
    while not Game.check_finish(history):
        possible, d = policies[active](hands[active], history)
        a = random.choices(possible, weights=d, k=1)[0]
        history.append(a); active = (active + 1) % 2
    pc = 1 if existence[history[-2]] else -1   # payoff to the checked (last-bet) player
    return pc if active == 0 else -pc


def run_setup(folder, hand_sizes, num_deals, baseline_dir=None):
    setup = "_".join(str(x) for x in hand_sizes)
    table, min_bet, _, macro_kind = load_bluff(folder, setup)
    total = sum(hand_sizes)
    n_macro = sum(1 for _, m in table.values() if m > 0)

    def bluff_p(macro_on, clean):
        return lambda h, hist: node_dist(table, h, hand_sizes, hist, min_bet,
                                         total, macro_on, macro_kind=macro_kind,
                                         clean=clean)

    if baseline_dir is None:
        # Self-test: macro-ON vs macro-OFF (same model, uncleaned, fair).
        A, B = bluff_p(True, False), bluff_p(False, False)
        label = f"{macro_kind} macro-ON vs OFF"
    else:
        # macro-ON (cleaned to match deployment) vs V2.1 (standard).
        std_table = load_standard(os.path.join(baseline_dir, setup))
        A = bluff_p(True, True)
        B = lambda h, hist: std_dist(std_table, h, hand_sizes, hist, min_bet)
        label = f"{macro_kind}(macro-ON) vs V2.1"

    adv = 0.0
    for _ in range(num_deals):
        hands = Game.deal_cards(hand_sizes)
        existence = Game.precompute_set_existence(hands)
        v_a0 = playout([A, B], hands, existence)   # A (macro model) at seat 0
        v_a1 = playout([B, A], hands, existence)   # A (macro model) at seat 1
        adv += (v_a0 - v_a1) / 2.0
    advantage = adv / num_deals
    se = (1.0 / num_deals) ** 0.5
    print(f"{setup}: {label} advantage = {advantage:+.4f} (~SE {se:.4f}) "
          f"over {num_deals} deals | min_bet={min_bet} | "
          f"{len(table):,} rows, {n_macro:,} with macro", flush=True)
    return advantage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bluff-folder", required=True)
    ap.add_argument("--baseline-dir", default=None,
                    help="If set, run bluff(macro-ON) vs the standard strategy "
                         "at <baseline-dir>/<setup>/strategy.npz (e.g. V2.1). "
                         "Otherwise run the macro-ON vs macro-OFF self-test.")
    ap.add_argument("--num-deals", type=int, default=10000)
    ap.add_argument("--setups", nargs="*", default=["9_11", "5_11", "4_5", "6_6", "5_7", "3_9"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    p_vector_fast([0, 1], 3)  # warm JIT
    for s in args.setups:
        hs = [int(x) for x in s.split("_")]
        run_setup(args.bluff_folder, hs, args.num_deals, baseline_dir=args.baseline_dir)
    print("=== SELFTEST-DONE ===", flush=True)


if __name__ == "__main__":
    main()
