"""Head-to-head: subgame solver vs. blueprint sampling.

For a given setup, simulate `--num-deals` Monte-Carlo games where one
seat plays via `subgame_pick_action` (using the same blueprint as the
opponent) and the other plays the blueprint directly via sampling.
Mirrors the structure of `analysis/head_to_head.py` but the per-decision
logic differs by seat role.

Reports per-seat normalised advantage in the same units as head_to_head.py:
  advantage_P0 = (B_starts_P0 - A_starts_P0) / num_deals / 2
where A = blueprint, B = subgame.

Optional --modes flag chooses which seat-role assignments to evaluate:
  - both: both directions (A=blueprint vs B=subgame, swapped seats too).
  - one-side: just one role assignment (faster, useful for big setups).

Run:
  python -m cfr_ai.analysis.subgame_h2h --hand-sizes 1 3 \\
      --num-deals 500 --n-belief-samples 300

CSV summary written to cfr_ai/outputs/subgame_h2h.csv with columns:
  setup, num_deals, n_belief, action_topk, threshold, seed,
  advantage_P0, advantage_P1, wall_s

Both advantage values are the SUBGAME's edge over BLUEPRINT, in payoff
units per game (so 0.020 means the subgame seat wins on average 0.02
units more than the blueprint seat over a 50-50 mix of starts).
"""

import argparse
import csv
import itertools
import os
import random
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
from cfr_ai.information_set import (
    get_possible_actions, get_hand_abstraction, make_key,
)
from cfr_ai.lbr import load_flat_strategy, _parse_key_to_composite
from cfr_ai.subgame import subgame_pick_action, make_context, warmup


# ---------------------------------------------------------------------------
# Blueprint sampling driver (matches agent.py / head_to_head.py semantics)
# ---------------------------------------------------------------------------

def _blueprint_sample(flat_strategy, hand_sizes, my_hand, history, rng):
    """Sample an action from the blueprint at this state. Falls back to
    check (or argmax of the round-start policy if last_bet==88) when the
    abstraction is unknown to the trained policy."""
    min_bet = flat_strategy.min_bet
    hand_size = len(my_hand)
    hand_abs = get_hand_abstraction(my_hand, hand_sizes)
    key = make_key(my_hand, hand_abs, history, min_bet)
    parts = key.split("-")
    last_bet = int(parts[1])
    suffix = "-".join(parts[2:])
    possible_actions = get_possible_actions(history, min_bet)
    abs_str_to_id = flat_strategy.abs_str_to_id
    # If abstraction unknown, default to check (or fallback at round start).
    from cfr_ai.lbr import _split_suffix
    _, _, abs_str = _split_suffix(suffix)
    if abs_str not in abs_str_to_id:
        if len(history) == 0:
            return 87  # great straight flush spades (matches agent.py fallback)
        return 88  # check
    comp_key = _parse_key_to_composite(
        suffix, hand_size, last_bet, dict(abs_str_to_id))
    row = flat_strategy.key_to_row.get(np.int64(comp_key))
    if row is None:
        if len(history) == 0:
            return 87
        return 88
    lo = int(flat_strategy.lower_action[row])
    hi = int(flat_strategy.upper_action[row])
    probs = np.array(flat_strategy.strategy[row, lo:hi + 1], dtype=np.float64)
    if probs.sum() <= 0:
        return 88
    probs /= probs.sum()
    # Align probs to possible_actions (just in case widths differ).
    if len(probs) != len(possible_actions):
        # Trim or pad
        out = np.zeros(len(possible_actions))
        n = min(len(probs), len(possible_actions))
        out[:n] = probs[:n]
        if out.sum() <= 0:
            out[-1] = 1.0
        out /= out.sum()
        probs = out
    chosen_offset = int(rng.choice(len(probs), p=probs))
    return possible_actions[chosen_offset]


def _play_one(
    hands: Tuple[List[int], List[int]],
    hand_sizes: List[int],
    seat0_role: str,  # 'blueprint' or 'subgame'
    seat1_role: str,
    flat_strategy,
    context,
    n_belief_samples: Optional[int],
    action_topk: Optional[int],
    action_prob_threshold: float,
    rng: np.random.Generator,
    starting_seat_idx: int,  # which seat acts first (0 or 1)
    existence: np.ndarray,
) -> int:
    """Play one MC game. Returns payoff for SEAT 0 in {+1, -1}."""
    history: List[int] = []
    active_seat = starting_seat_idx
    while not Game.check_finish(history):
        my_hand = hands[active_seat]
        role = seat0_role if active_seat == 0 else seat1_role
        if role == 'blueprint':
            a = _blueprint_sample(flat_strategy, hand_sizes, my_hand, history, rng)
        else:  # subgame
            a, _ = subgame_pick_action(
                my_hand, hand_sizes, active_seat, history, flat_strategy,
                n_belief_samples=n_belief_samples,
                context=context,
                action_topk=action_topk,
                action_prob_threshold=action_prob_threshold,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
        history.append(int(a))
        active_seat = 1 - active_seat
    # Terminal: last action was 88 (check). The checked-on bet is history[-2].
    # The "checked player" = active player at this moment (whoever was about
    # to act when their opponent checked is actually... wait the semantics
    # are: the player who CHECKED was just-active. After they check, the
    # "checked player" (who made the bet) is on the spot. Re-read head_to_head.
    #
    # In `run_mc_playout` of head_to_head.py:
    #   payoff_for_checked_player = 1 if existence[history[-2]] else -1
    #   if active_player_idx == 0:  return payoff_for_checked_player
    #   else: return -payoff_for_checked_player
    # active_player_idx swaps each loop iteration; after the loop the
    # "active player" is the one whose turn would be NEXT (i.e., the one
    # who got checked). So if active=0, the checker was 1, and the bet
    # holder was 0; payoff_for_0 = 1 if exists else -1. Matches.
    payoff_for_checked_player = 1 if existence[history[-2]] else -1
    return payoff_for_checked_player if active_seat == 0 else -payoff_for_checked_player


def run_h2h(
    hand_sizes: List[int],
    flat_strategy,
    num_deals: int,
    n_belief_samples: Optional[int],
    action_topk: Optional[int],
    action_prob_threshold: float,
    seed: int,
    show_progress: bool = True,
) -> Dict[str, float]:
    """Returns a dict with keys subgame_adv_seat0_start, subgame_adv_seat1_start,
    wall_s, deals."""
    rng = np.random.default_rng(seed)

    # Pre-allocate context for subgame
    max_n_opp_full = max(
        len(list(itertools.combinations(range(24 - hand_sizes[0]), hand_sizes[1]))),
        len(list(itertools.combinations(range(24 - hand_sizes[1]), hand_sizes[0]))),
    )
    if n_belief_samples is not None:
        max_n_opp = min(n_belief_samples, max_n_opp_full)
    else:
        max_n_opp = max_n_opp_full
    ctx = make_context(hand_sizes, flat_strategy, max_n_opp)
    warmup(hand_sizes, flat_strategy, ctx, n_belief_samples=n_belief_samples)

    # Accumulators. For each starting-seat assignment and each role-assignment.
    # Role assignments: (seat0='blueprint', seat1='subgame') and
    #                   (seat0='subgame', seat1='blueprint')
    # Starting seats:   0, 1 (only one for symmetric setups).
    starts = [0] if hand_sizes[0] == hand_sizes[1] else [0, 1]

    # We'll track per-starter:
    #   payoffs_BPsub[s] -> sum payoff for seat 0 when (seat0=BP, seat1=sub), starter=s
    #   payoffs_subBP[s] -> sum payoff for seat 0 when (seat0=sub, seat1=BP), starter=s
    payoffs_BPsub = {s: 0.0 for s in starts}
    payoffs_subBP = {s: 0.0 for s in starts}

    t0 = time.time()
    iterator = range(num_deals)
    if show_progress:
        iterator = tqdm(iterator, desc=f"H2H {hand_sizes} subgame vs BP")
    for _ in iterator:
        hands = Game.deal_cards(hand_sizes)
        existence = Game.precompute_set_existence(hands)
        for s in starts:
            # Role assignment 1: seat0=BP, seat1=sub
            p_seat0 = _play_one(
                hands, hand_sizes, 'blueprint', 'subgame',
                flat_strategy, ctx, n_belief_samples,
                action_topk, action_prob_threshold, rng, s, existence,
            )
            payoffs_BPsub[s] += p_seat0
            # Role assignment 2: seat0=sub, seat1=BP
            p_seat0 = _play_one(
                hands, hand_sizes, 'subgame', 'blueprint',
                flat_strategy, ctx, n_belief_samples,
                action_topk, action_prob_threshold, rng, s, existence,
            )
            payoffs_subBP[s] += p_seat0
    wall = time.time() - t0

    # Subgame's advantage:
    # When subgame plays seat-0-card-count: it's in the role assignments where
    # subgame is at seat 0. payoff for seat 0 is the subgame's payoff.
    # When subgame plays seat-1-card-count: it sits at seat 1; seat 0 payoff
    # is the BLUEPRINT's payoff. Subgame's payoff = -seat0_payoff.
    out = {"deals": num_deals, "wall_s": wall}
    for s in starts:
        adv = (payoffs_subBP[s] - payoffs_BPsub[s]) / num_deals / 2.0
        out[f"subgame_adv_start_seat{s}"] = adv
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hand-sizes", nargs=2, type=int, required=True)
    p.add_argument("--num-deals", type=int, default=500)
    p.add_argument("--n-belief-samples", type=int, default=300)
    p.add_argument("--action-topk", type=int, default=None)
    p.add_argument("--action-prob-threshold", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--update-summary", action="store_true",
                   help="Append the result to cfr_ai/outputs/subgame_h2h.csv.")
    args = p.parse_args()

    hand_sizes = sorted(args.hand_sizes)
    print(f"[subgame_h2h] setup={hand_sizes} deals={args.num_deals} "
          f"belief={args.n_belief_samples} topk={args.action_topk} "
          f"thr={args.action_prob_threshold}", flush=True)

    t0 = time.time()
    fs = load_flat_strategy(hand_sizes)
    print(f"  loaded {len(fs.key_to_row)} non-checking entries in "
          f"{time.time()-t0:.1f}s; min_bet={fs.min_bet}", flush=True)

    res = run_h2h(
        hand_sizes, fs, args.num_deals,
        args.n_belief_samples, args.action_topk, args.action_prob_threshold,
        args.seed, show_progress=not args.no_progress,
    )
    print()
    print(f"  Subgame's edge over blueprint (payoff per game):", flush=True)
    for k, v in res.items():
        if k.startswith("subgame_adv_"):
            print(f"    {k}: {v:+.4f}", flush=True)
    print(f"  wall = {res['wall_s']:.1f}s "
          f"({res['wall_s']/(2*max(1,res['deals'])):.2f}s/game-pair)", flush=True)

    if args.update_summary:
        out_csv = os.path.join("cfr_ai", "outputs", "subgame_h2h.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        write_header = not os.path.exists(out_csv)
        fields = ["setup", "num_deals", "n_belief", "action_topk", "threshold",
                  "seed", "advantage_start_seat0", "advantage_start_seat1",
                  "wall_s"]
        row = {
            "setup": ",".join(str(x) for x in hand_sizes),
            "num_deals": args.num_deals,
            "n_belief": args.n_belief_samples,
            "action_topk": args.action_topk if args.action_topk is not None else "",
            "threshold": args.action_prob_threshold,
            "seed": args.seed,
            "advantage_start_seat0": res.get("subgame_adv_start_seat0", ""),
            "advantage_start_seat1": res.get("subgame_adv_start_seat1", ""),
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
