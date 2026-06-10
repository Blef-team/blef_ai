"""Head-to-head: subgame solver(blueprint A) vs. subgame solver(blueprint B).

Both seats play via subgame solver, but with different underlying
blueprints (e.g. v0_baseline vs v_numba_3_7_test). The purpose is to
compare blueprint *quality under subgame solving*, which can differ from
their direct head-to-head comparison.

Reports per-seat normalised advantage:
  advantage_P0 = (B_starts_P0 - A_starts_P0) / num_deals / 2
where A = subgame(blueprint A), B = subgame(blueprint B).

Run:
  python -m cfr_ai.analysis.subgame_vs_subgame --hand-sizes 1 3 \\
      --model-a current --model-b v_numba_3_7_test --num-deals 500 \\
      --n-belief-samples 300
"""

import argparse
import csv
import itertools
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cfr_ai.game import Game
from cfr_ai.lbr import load_flat_strategy
from cfr_ai.subgame import subgame_pick_action, make_context, warmup


def _resolve_strategy_dir(tag_or_folder: str, hand_sizes: List[int]) -> str:
    """Map a tag (e.g. 'current' or 'v_numba_3_7_test') or explicit folder
    path to the strategy directory for this setup."""
    setup = "_".join(str(x) for x in hand_sizes)
    if os.path.isdir(tag_or_folder):
        # Assume it's a folder containing outputs/<setup>/
        cand = os.path.join(tag_or_folder, "outputs", setup)
        if os.path.isdir(cand):
            return cand
        # Or might be the leaf setup dir directly.
        if os.path.isdir(os.path.join(tag_or_folder, str(hand_sizes[0]))):
            return tag_or_folder
        raise ValueError(f"Folder {tag_or_folder} doesn't look like a model dir")
    if tag_or_folder in ("current", "."):
        return os.path.join("cfr_ai", "outputs", setup)
    return os.path.join("cfr_ai", "archive", tag_or_folder, "outputs", setup)


def _play_one(
    hands: Tuple[List[int], List[int]],
    hand_sizes: List[int],
    seat0_blueprint, seat0_ctx,
    seat1_blueprint, seat1_ctx,
    n_belief_samples: Optional[int],
    action_topk: Optional[int],
    action_prob_threshold: float,
    rng: np.random.Generator,
    starting_seat_idx: int,
    existence: np.ndarray,
) -> int:
    """Play one MC game; return payoff for seat 0."""
    history: List[int] = []
    active_seat = starting_seat_idx
    while not Game.check_finish(history):
        my_hand = hands[active_seat]
        if active_seat == 0:
            bp, ctx = seat0_blueprint, seat0_ctx
        else:
            bp, ctx = seat1_blueprint, seat1_ctx
        a, _ = subgame_pick_action(
            my_hand, hand_sizes, active_seat, history, bp,
            n_belief_samples=n_belief_samples,
            context=ctx,
            action_topk=action_topk,
            action_prob_threshold=action_prob_threshold,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        history.append(int(a))
        active_seat = 1 - active_seat
    payoff_for_checked = 1 if existence[history[-2]] else -1
    return payoff_for_checked if active_seat == 0 else -payoff_for_checked


def run_h2h(
    hand_sizes: List[int],
    bp_A, bp_B,
    num_deals: int,
    n_belief_samples: Optional[int],
    action_topk: Optional[int],
    action_prob_threshold: float,
    seed: int,
    show_progress: bool = True,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    max_n_opp_full = max(
        len(list(itertools.combinations(range(24 - hand_sizes[0]), hand_sizes[1]))),
        len(list(itertools.combinations(range(24 - hand_sizes[1]), hand_sizes[0]))),
    )
    if n_belief_samples is not None:
        max_n_opp = min(n_belief_samples, max_n_opp_full)
    else:
        max_n_opp = max_n_opp_full
    ctx_A = make_context(hand_sizes, bp_A, max_n_opp)
    ctx_B = make_context(hand_sizes, bp_B, max_n_opp)
    warmup(hand_sizes, bp_A, ctx_A, n_belief_samples=n_belief_samples)
    warmup(hand_sizes, bp_B, ctx_B, n_belief_samples=n_belief_samples)

    starts = [0] if hand_sizes[0] == hand_sizes[1] else [0, 1]
    payoffs_AB = {s: 0.0 for s in starts}  # seat0=A, seat1=B
    payoffs_BA = {s: 0.0 for s in starts}  # seat0=B, seat1=A

    t0 = time.time()
    iterator = range(num_deals)
    if show_progress:
        iterator = tqdm(iterator, desc=f"H2H {hand_sizes} sub(A) vs sub(B)")
    for _ in iterator:
        hands = Game.deal_cards(hand_sizes)
        existence = Game.precompute_set_existence(hands)
        for s in starts:
            p_seat0 = _play_one(
                hands, hand_sizes, bp_A, ctx_A, bp_B, ctx_B,
                n_belief_samples, action_topk, action_prob_threshold,
                rng, s, existence,
            )
            payoffs_AB[s] += p_seat0
            p_seat0 = _play_one(
                hands, hand_sizes, bp_B, ctx_B, bp_A, ctx_A,
                n_belief_samples, action_topk, action_prob_threshold,
                rng, s, existence,
            )
            payoffs_BA[s] += p_seat0
    wall = time.time() - t0

    out = {"deals": num_deals, "wall_s": wall}
    # B's edge (subgame with blueprint B over subgame with blueprint A).
    for s in starts:
        adv = (payoffs_BA[s] - payoffs_AB[s]) / num_deals / 2.0
        out[f"B_adv_start_seat{s}"] = adv
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hand-sizes", nargs=2, type=int, required=True)
    p.add_argument("--model-a", type=str, required=True,
                   help="Archive tag (e.g. 'current', 'v0_baseline') or folder.")
    p.add_argument("--model-b", type=str, required=True,
                   help="Archive tag or folder.")
    p.add_argument("--num-deals", type=int, default=500)
    p.add_argument("--n-belief-samples", type=int, default=300)
    p.add_argument("--action-topk", type=int, default=None)
    p.add_argument("--action-prob-threshold", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--update-summary", action="store_true")
    args = p.parse_args()

    hand_sizes = sorted(args.hand_sizes)
    print(f"[subgame_vs_subgame] setup={hand_sizes} "
          f"A={args.model_a} B={args.model_b} deals={args.num_deals} "
          f"belief={args.n_belief_samples}", flush=True)

    dir_A = _resolve_strategy_dir(args.model_a, hand_sizes)
    dir_B = _resolve_strategy_dir(args.model_b, hand_sizes)
    print(f"  loading A from {dir_A}", flush=True)
    t0 = time.time()
    bp_A = load_flat_strategy(hand_sizes, setup_dir=dir_A)
    print(f"    {len(bp_A.key_to_row)} entries, {time.time()-t0:.1f}s", flush=True)
    print(f"  loading B from {dir_B}", flush=True)
    t0 = time.time()
    bp_B = load_flat_strategy(hand_sizes, setup_dir=dir_B)
    print(f"    {len(bp_B.key_to_row)} entries, {time.time()-t0:.1f}s", flush=True)

    res = run_h2h(
        hand_sizes, bp_A, bp_B, args.num_deals,
        args.n_belief_samples, args.action_topk, args.action_prob_threshold,
        args.seed, show_progress=not args.no_progress,
    )
    print()
    print(f"  Subgame(B) edge over subgame(A) per game:", flush=True)
    for k, v in res.items():
        if k.startswith("B_adv_"):
            print(f"    {k}: {v:+.4f}", flush=True)
    print(f"  wall = {res['wall_s']:.1f}s", flush=True)

    if args.update_summary:
        out_csv = os.path.join("cfr_ai", "outputs", "subgame_vs_subgame.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        write_header = not os.path.exists(out_csv)
        fields = ["setup", "model_a", "model_b", "num_deals", "n_belief",
                  "action_topk", "threshold", "seed",
                  "B_adv_start_seat0", "B_adv_start_seat1", "wall_s"]
        row = {
            "setup": ",".join(str(x) for x in hand_sizes),
            "model_a": args.model_a, "model_b": args.model_b,
            "num_deals": args.num_deals, "n_belief": args.n_belief_samples,
            "action_topk": args.action_topk if args.action_topk is not None else "",
            "threshold": args.action_prob_threshold,
            "seed": args.seed,
            "B_adv_start_seat0": res.get("B_adv_start_seat0", ""),
            "B_adv_start_seat1": res.get("B_adv_start_seat1", ""),
            "wall_s": round(res["wall_s"], 1),
        }
        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"  appended to {out_csv}", flush=True)


if __name__ == "__main__":
    main()
