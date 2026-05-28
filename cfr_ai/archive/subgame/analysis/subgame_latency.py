"""Per-setup, per-state-depth latency breakdown of the subgame solver.

For each setup, measure wall time of `subgame_pick_action` at multiple
states (empty history, 1 action, 2 actions, ...). The empty-history call
is usually the worst because every action is legal and the rollout tree
is at its deepest.

Run: python -m cfr_ai.analysis.subgame_latency --hand-sizes 1 3 --max-hist 4
"""

import argparse
import itertools
import os
import sys
import time
from typing import List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cfr_ai.game import Game
from cfr_ai.lbr import load_flat_strategy
from cfr_ai.subgame import subgame_pick_action, make_context


def _walk_with_blueprint(fs, hand_sizes, deal, rng, max_steps: int):
    """Sample a blueprint-driven history of length up to `max_steps`."""
    from cfr_ai.information_set import make_key, get_hand_abstraction
    from cfr_ai.lbr import _parse_key_to_composite
    history = []
    for step in range(max_steps):
        seat = step % 2
        hand = deal[seat]
        hand_size = hand_sizes[seat]
        hand_abs = get_hand_abstraction(hand, hand_sizes)
        key = make_key(hand, hand_abs, history, fs.min_bet)
        parts = key.split("-")
        last_bet = int(parts[1])
        suffix = "-".join(parts[2:])
        if suffix.split("-")[-1] not in fs.abs_str_to_id:
            return history  # quietly skip; current state still valid
        comp_key = _parse_key_to_composite(
            suffix, hand_size, last_bet, dict(fs.abs_str_to_id))
        row = fs.key_to_row.get(np.int64(comp_key))
        if row is None:
            return history
        lo = int(fs.lower_action[row])
        hi = int(fs.upper_action[row])
        probs = np.array(fs.strategy[row, lo:hi + 1], dtype=np.float64)
        if probs.sum() <= 0:
            return history
        probs /= probs.sum()
        chosen_offset = int(rng.choice(len(probs), p=probs))
        a = lo + chosen_offset
        history.append(a)
        if a == 88:
            history.pop()  # back off; subgame should not be called at terminal
            return history
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hand-sizes", nargs=2, type=int, required=True)
    p.add_argument("--max-hist", type=int, default=4,
                   help="Try states with history lengths 0, 1, ..., max-hist-1.")
    p.add_argument("--n-trials", type=int, default=10,
                   help="Trials per (history-length, my_seat) bucket.")
    p.add_argument("--n-belief-samples", type=int, default=None)
    args = p.parse_args()

    hand_sizes = sorted(args.hand_sizes)
    print(f"[subgame_latency] setup={hand_sizes} "
          f"n_belief={args.n_belief_samples}", flush=True)

    t0 = time.time()
    fs = load_flat_strategy(hand_sizes)
    print(f"  loaded {len(fs.key_to_row)} entries in {time.time()-t0:.1f}s; "
          f"min_bet={fs.min_bet}", flush=True)

    if args.n_belief_samples is not None:
        max_n_opp = args.n_belief_samples
    else:
        max_n_opp = max(
            len(list(itertools.combinations(range(24 - hand_sizes[0]), hand_sizes[1]))),
            len(list(itertools.combinations(range(24 - hand_sizes[1]), hand_sizes[0]))),
        )
    ctx = make_context(hand_sizes, fs, max_n_opp)
    print(f"  context buffers: {ctx.estimated_mb():.1f} MB "
          f"(max_n_opp={max_n_opp})", flush=True)

    rng = np.random.default_rng(7)

    # Warmup once.
    deal = Game.deal_cards(hand_sizes)
    subgame_pick_action(deal[0], hand_sizes, 0, [], fs,
                       n_belief_samples=args.n_belief_samples, context=ctx)

    seats = [0] if hand_sizes[0] == hand_sizes[1] else [0, 1]
    print(f"\n  {'hist_len':>9}  {'seat':>4}  {'n':>4}  "
          f"{'mean':>7}  {'p50':>7}  {'p95':>7}  {'max':>7}",
          flush=True)
    for hist_len in range(args.max_hist):
        for seat in seats:
            times = []
            for _ in range(args.n_trials):
                deal = Game.deal_cards(hand_sizes)
                # Walk forward until we have hist_len actions AND the next
                # actor is `seat` (history length parity == seat).
                history = _walk_with_blueprint(fs, hand_sizes, deal, rng, max_steps=hist_len)
                # If we accidentally walked further or shorter, just use what we got
                if len(history) != hist_len:
                    continue
                # parity check: position to act now is len(history) % 2
                if len(history) % 2 != seat:
                    continue
                t0 = time.time()
                subgame_pick_action(
                    deal[seat], hand_sizes, seat, history, fs,
                    n_belief_samples=args.n_belief_samples, context=ctx,
                )
                times.append(time.time() - t0)
            if not times:
                print(f"  {hist_len:>9d}  {seat:>4d}  {'-':>4}  "
                      f"{'-':>7}  {'-':>7}  {'-':>7}  {'-':>7}", flush=True)
                continue
            arr = np.array(times)
            print(f"  {hist_len:>9d}  {seat:>4d}  {len(arr):>4d}  "
                  f"{arr.mean()*1000:>6.0f}ms "
                  f"{np.median(arr)*1000:>6.0f}ms "
                  f"{np.quantile(arr, 0.95)*1000:>6.0f}ms "
                  f"{arr.max()*1000:>6.0f}ms", flush=True)


if __name__ == "__main__":
    main()
