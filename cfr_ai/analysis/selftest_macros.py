"""H2H for the multi-macro model. Reuses selftest_bluff's standard-model loader,
keys, MC playout, and clean. Resolves an ENABLED SUBSET of macros per node.

Modes:
  --baseline-dir DIR : enabled-macros (cleaned) vs the standard model at DIR/<setup>.
  (default)          : enabled-macros-ON vs all-macros-OFF (same model).
--enable lists which macro kinds to resolve (default: all in the strategy).
"""
import argparse
import os
import random
from functools import lru_cache

import numpy as np

from cfr_ai.game import Game
import cfr_ai.information_set as ISM
from cfr_ai.analysis.head_to_head import make_key_wrapper, get_possible_actions_wrapper
from cfr_ai.abstraction.probs import g_vector
from cfr_ai.abstraction.probs_jit import p_vector_fast
from cfr_ai.encoding import clear_lows
from cfr_ai.analysis.selftest_bluff import load_standard, std_dist


def load_macros(folder, setup):
    """Read the unified macro strategy.npz and rebuild the {string_key:
    (concrete probs[89], masses[n_macros])} table node_dist expects,
    reconstructing the string key from the int64 composite (inverse of save)."""
    import json
    from cfr_ai.keys import (LAST_BET_SHIFT, H_M1_SHIFT, H_M2_SHIFT, ABS_ID_SHIFT,
                             ABSENT_CODE, _HISTORY_CODE_STRS)
    sd = os.path.join(folder, "outputs", setup)
    d = np.load(os.path.join(sd, "strategy.npz"), allow_pickle=True)
    keys, probs, masses = d["keys"], d["probs"], d["masses"]
    kinds = [str(x) for x in d["kinds"]]
    with open(os.path.join(sd, "strategy.abs.json"), encoding="utf-8") as f:
        id_to_abs = {v: k for k, v in json.load(f).items()}
    table = {}
    for i in range(len(keys)):
        k = int(keys[i])
        ks = f"{k & 0xF}-{(k >> LAST_BET_SHIFT) & 0xFF}-"
        h1, h2 = (k >> H_M1_SHIFT) & 0xFF, (k >> H_M2_SHIFT) & 0xFF
        if h1 != ABSENT_CODE:
            ks += _HISTORY_CODE_STRS[h1] + "-"
            if h2 != ABSENT_CODE:
                ks += _HISTORY_CODE_STRS[h2] + "-"
        ks += id_to_abs[k >> ABS_ID_SHIFT]
        table[ks] = (probs[i], masses[i])
    return table, int(d["min_bet"]), [int(x) for x in setup.split("_")], kinds


@lru_cache(maxsize=None)
def _score(hand_tuple, total, kind):
    pv = p_vector_fast(list(hand_tuple), total - len(hand_tuple))
    if kind == "value":
        return pv
    if kind == "difftruthy":
        return pv - g_vector(total)
    return g_vector(total) - pv


def _bstar(hand_tuple, total, kind, possible):
    score = _score(hand_tuple, total, kind)
    bs, best, ties = -1, -1e18, 0
    for a in possible:
        if a <= 87:
            s = score[a]
            if s > best + 1e-9:
                best, bs, ties = s, a, 1
            elif s > best - 1e-9:
                ties += 1
                if random.random() * ties < 1.0:
                    bs = a
    return bs


def node_dist(table, kinds, enabled, my_hand, hand_sizes, history, min_bet, total, clean):
    possible = get_possible_actions_wrapper(ISM, history, min_bet)
    abst = ISM.get_hand_abstraction(my_hand, hand_sizes)
    key = make_key_wrapper(ISM, my_hand, abst, history, min_bet)
    num = len(possible)
    e = table.get(key)
    if e is None:
        s = np.zeros(num); s[-1] = 1.0; return possible, s
    probs_full, macro_probs = e
    out = np.array([probs_full[a] for a in possible], dtype=np.float64)
    for k, kind in enumerate(kinds):
        if enabled[k] and macro_probs[k] > 0.0:
            bs = _bstar(tuple(my_hand), total, kind, possible)
            if bs >= 0:
                out[possible.index(bs)] += macro_probs[k]
    tot = out.sum()
    if tot <= 0.0:
        out = np.zeros(num); out[-1] = 1.0
        return possible, out
    out = out / tot
    if clean:
        out = clear_lows(out)
    return possible, out


def playout(policies, hands, existence):
    history, active = [], 0
    while not Game.check_finish(history):
        possible, d = policies[active](hands[active], history)
        a = random.choices(possible, weights=d, k=1)[0]
        history.append(a); active = (active + 1) % 2
    pc = 1 if existence[history[-2]] else -1
    return pc if active == 0 else -pc


def run_setup(folder, hand_sizes, num_deals, enable, baseline_dir):
    setup = "_".join(str(x) for x in hand_sizes)
    table, min_bet, _, kinds = load_macros(folder, setup)
    total = sum(hand_sizes)
    n = len(kinds)
    en = [(k in enable) for k in kinds] if enable else [True] * n
    en_label = "+".join(kinds[i] for i in range(n) if en[i]) or "none"
    n_macro_rows = sum(1 for _, m in table.values() if m.sum() > 0)

    def macro_p(enabled, clean):
        return lambda h, hist: node_dist(table, kinds, enabled, h, hand_sizes, hist, min_bet, total, clean)

    if baseline_dir is None:
        A, B = macro_p(en, False), macro_p([False] * n, False)
        label = f"[{en_label}]-ON vs OFF"
    else:
        std = load_standard(os.path.join(baseline_dir, setup))
        A = macro_p(en, True)
        B = lambda h, hist: std_dist(std, h, hand_sizes, hist, min_bet)
        label = f"[{en_label}] vs V2.1"

    adv = 0.0
    for _ in range(num_deals):
        hands = Game.deal_cards(hand_sizes)
        existence = Game.precompute_set_existence(hands)
        adv += (playout([A, B], hands, existence) - playout([B, A], hands, existence)) / 2.0
    advantage = adv / num_deals
    se = (1.0 / num_deals) ** 0.5
    print(f"{setup}: {label} advantage = {advantage:+.4f} (~SE {se:.4f}) over {num_deals} deals "
          f"| min_bet={min_bet} | kinds={kinds} | {len(table):,} rows, {n_macro_rows:,} w/macro", flush=True)
    return advantage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--macros-folder", required=True)
    ap.add_argument("--baseline-dir", default=None)
    ap.add_argument("--enable", nargs="*", default=None,
                    help="Subset of macro kinds to resolve (default: all). "
                         "e.g. --enable difftruthy")
    ap.add_argument("--num-deals", type=int, default=10000)
    ap.add_argument("--setups", nargs="*", default=["9_11", "5_11", "4_5", "6_6", "5_7", "3_9"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    p_vector_fast([0, 1], 3)
    for s in args.setups:
        hs = [int(x) for x in s.split("_")]
        run_setup(args.macros_folder, hs, args.num_deals, args.enable, args.baseline_dir)
    print("=== setups-complete ===", flush=True)


if __name__ == "__main__":
    main()
