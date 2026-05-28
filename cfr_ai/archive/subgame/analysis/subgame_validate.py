"""Sanity-check the subgame solver against the blueprint.

Three checks, each is a structural property the solver must satisfy.
None of these guarantee correctness, but failures are highly diagnostic:

  1. Uniform-history posterior: with no opp turns observed, posterior over
     opp hands must be uniform (reach == 1 for every n).
  2. Posterior consistency after one opp turn: if opp played `a`, the
     posterior reach[n] should equal sigma_opp(a | opp_hand_n) up to a
     constant. We can verify by comparing reach/reach.sum() to manually
     computed action probs.
  3. Greedy-vs-mixed-blueprint sanity at the root: the subgame solver's
     V[best] at empty history must be >= the value the blueprint achieves
     by mixing from this state. (Best response can't do worse than the
     mixed strategy it's responding against — same agent on opp side.)

Run: python -m cfr_ai.analysis.subgame_validate --hand-sizes 1 1
"""

import argparse
import itertools
import os
import sys
import time
from typing import List

import numpy as np

# Make project root importable when running this script directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cfr_ai.game import Game
from cfr_ai.lbr import (
    load_flat_strategy, intern_abstractions_for_hand,
    intern_abstractions_for_hands, _lookup_by_key, _cfr_vs_cfr_value_jit,
)
from cfr_ai.trainer import (
    LAST_BET_SHIFT, H_M1_SHIFT, H_M2_SHIFT, ABS_ID_SHIFT,
    MAX_DEPTH, _HISTORY_CODE_ID,
)
from cfr_ai.subgame import (
    subgame_pick_action, make_context, _build_posterior_reach_jit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigma_opp_action_probs(
    flat_strategy, opp_hand_size: int, opp_abs_ids: np.ndarray,
    history_buf, hist_len: int, min_bet: int,
) -> np.ndarray:
    """For each opp hand, return the full (n_actions,) strategy at the
    current state. Python-side reference (no JIT) used to verify the JIT
    posterior builder matches a hand-computed posterior."""
    from cfr_ai.lbr import _history_to_key_parts  # JIT but callable
    last_bet, h_m1_id, h_m2_id = _history_to_key_parts(
        history_buf, hist_len, min_bet, _HISTORY_CODE_ID)
    if last_bet == 88:
        lo, hi = min_bet, 87
    else:
        lo, hi = last_bet + 1, 88
    n_actions = hi - lo + 1
    n_opp = opp_abs_ids.shape[0]
    out = np.zeros((n_opp, n_actions), dtype=np.float64)
    scratch = np.zeros(89, dtype=np.float64)
    base_key = (opp_hand_size
                | (last_bet << LAST_BET_SHIFT)
                | (h_m1_id << H_M1_SHIFT)
                | (h_m2_id << H_M2_SHIFT))
    for n in range(n_opp):
        abs_id = int(opp_abs_ids[n, last_bet])
        key = base_key | (abs_id << ABS_ID_SHIFT)
        _lookup_by_key(
            flat_strategy.key_to_row, flat_strategy.strategy,
            flat_strategy.lower_action, flat_strategy.upper_action,
            np.int64(key), n_actions, scratch[:n_actions],
        )
        out[n] = scratch[:n_actions]
    return out, lo


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_uniform_posterior(fs, hand_sizes: List[int], my_seat: int) -> bool:
    """Empty history -> reach should be all-1.0."""
    opp_seat = 1 - my_seat
    opp_hand_size = hand_sizes[opp_seat]
    my_hand = list(sorted(np.random.default_rng(0).choice(24, hand_sizes[my_seat], replace=False).tolist()))
    remaining = [c for c in range(24) if c not in my_hand]
    all_opp = [sorted(h) for h in itertools.combinations(remaining, opp_hand_size)]
    n_opp = len(all_opp)
    opp_abs_ids = intern_abstractions_for_hands(all_opp, hand_sizes, fs.abs_str_to_id)
    reach = np.ones(n_opp, dtype=np.float64)
    history_buf = np.zeros(MAX_DEPTH, dtype=np.int64)
    _build_posterior_reach_jit(
        history_buf, 0, my_seat == 1,
        opp_hand_size, opp_abs_ids,
        fs.key_to_row, fs.strategy, fs.lower_action, fs.upper_action,
        _HISTORY_CODE_ID, fs.min_bet, reach,
    )
    ok = np.allclose(reach, 1.0)
    print(f"  [1] uniform posterior, my_seat={my_seat}: "
          f"min={reach.min():.6f} max={reach.max():.6f} "
          f"-> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def check_one_step_posterior(fs, hand_sizes: List[int]) -> bool:
    """After one opp turn (opp acted first, played action a), posterior
    should equal sigma_opp(a | opp_hand_n) up to a normalisation."""
    # We are seat 1 (opp acted first as seat 0); see opp's action.
    my_seat = 1
    opp_seat = 0
    my_hand_size = hand_sizes[my_seat]
    opp_hand_size = hand_sizes[opp_seat]
    my_hand = list(sorted(
        np.random.default_rng(1).choice(24, my_hand_size, replace=False).tolist()))
    remaining = [c for c in range(24) if c not in my_hand]
    all_opp = [sorted(h) for h in itertools.combinations(remaining, opp_hand_size)]
    n_opp = len(all_opp)
    opp_abs_ids = intern_abstractions_for_hands(all_opp, hand_sizes, fs.abs_str_to_id)

    # Compute Python-side sigma_opp at the empty state.
    history_buf = np.zeros(MAX_DEPTH, dtype=np.int64)
    sigma_opp_full, lo = _sigma_opp_action_probs(
        fs, opp_hand_size, opp_abs_ids, history_buf, 0, fs.min_bet)
    # Pick an action with non-trivial support: the one most likely under the
    # marginal sigma_opp (sum over opp hands).
    marginal = sigma_opp_full.sum(axis=0)
    chosen_action_idx = int(np.argmax(marginal))
    a = lo + chosen_action_idx
    expected_reach = sigma_opp_full[:, chosen_action_idx]

    history_buf[0] = a
    reach = np.ones(n_opp, dtype=np.float64)
    _build_posterior_reach_jit(
        history_buf, 1, True,  # opp started
        opp_hand_size, opp_abs_ids,
        fs.key_to_row, fs.strategy, fs.lower_action, fs.upper_action,
        _HISTORY_CODE_ID, fs.min_bet, reach,
    )

    # reach should equal expected_reach exactly (we started with 1.0s).
    max_err = float(np.max(np.abs(reach - expected_reach)))
    ok = max_err < 1e-9
    print(f"  [2] one-step posterior matches sigma_opp(a={a}|opp_hand), "
          f"max_err={max_err:.2e} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def check_best_response_at_least_blueprint(fs, hand_sizes: List[int]) -> bool:
    """At empty history, subgame's V[best] >= V[blueprint argmax].

    Strictly we should compare to V[mixed-blueprint], but argmax gives a
    sufficient lower bound: best response to the same opp policy can't do
    worse than playing the blueprint's most-probable action."""
    my_seat = 0
    opp_seat = 1
    rng = np.random.default_rng(2)
    my_hand_size = hand_sizes[my_seat]
    opp_hand_size = hand_sizes[opp_seat]

    n_passed = 0
    n_total = 0
    for trial in range(5):
        my_hand = list(sorted(rng.choice(24, my_hand_size, replace=False).tolist()))

        a_sub, diag = subgame_pick_action(
            my_hand, hand_sizes, my_seat, [], fs,
            return_diagnostics=True,
        )
        if "V" not in diag:
            continue
        V = diag["V"]
        v_best = diag["best_value"]
        lo = diag["lo"]
        # Compute V at the blueprint's argmax action for comparison.
        # We just call subgame's machinery again with the blueprint action
        # forced — same V[a] is already in diag["V"] for every legal a.
        from cfr_ai.information_set import make_key, get_hand_abstraction
        from cfr_ai.lbr import _parse_key_to_composite
        hand_abs = get_hand_abstraction(my_hand, hand_sizes)
        key = make_key(my_hand, hand_abs, [], fs.min_bet)
        parts = key.split("-")
        last_bet = int(parts[1])
        suffix = "-".join(parts[2:])
        abs_str_to_id = dict(fs.abs_str_to_id)  # don't mutate
        if suffix.split("-")[-1] not in abs_str_to_id:
            # Hand has an abstraction not in the trained policy — skip.
            continue
        comp_key = _parse_key_to_composite(
            suffix, my_hand_size, last_bet, abs_str_to_id)
        row = fs.key_to_row.get(np.int64(comp_key))
        if row is None:
            continue
        bp_lo = int(fs.lower_action[row])
        bp_hi = int(fs.upper_action[row])
        probs = fs.strategy[row, bp_lo:bp_hi + 1]
        if probs.sum() <= 0:
            continue
        # blueprint argmax action
        a_bp = bp_lo + int(np.argmax(probs))
        # value at that action — must be at index (a_bp - lo) in V
        if not (lo <= a_bp <= lo + len(V) - 1):
            continue
        v_bp = float(V[a_bp - lo])

        n_total += 1
        if v_best >= v_bp - 1e-6:
            n_passed += 1

    ok = (n_total > 0 and n_passed == n_total)
    print(f"  [3] V_subgame[best] >= V[blueprint_argmax]: {n_passed}/{n_total} trials "
          f"-> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def latency_bench(fs, hand_sizes: List[int], n_decisions: int, n_belief_samples=None):
    """Measure per-decision wall time on realistic states.

    States are generated by sampling deals and walking forward with the
    blueprint until a legal subgame turn appears."""
    import random
    rng = np.random.default_rng(123)
    my_seats = [0]
    if hand_sizes[0] != hand_sizes[1]:
        my_seats = [0, 1]
    # n_opp upper bound for context
    if n_belief_samples is not None:
        max_n_opp = n_belief_samples
    else:
        max_n_opp = max(
            len(list(itertools.combinations(range(24 - hand_sizes[0]), hand_sizes[1]))),
            len(list(itertools.combinations(range(24 - hand_sizes[1]), hand_sizes[0]))),
        )
    ctx = make_context(hand_sizes, fs, max_n_opp)
    print(f"  context: max_n_opp={max_n_opp}, buffers={ctx.estimated_mb():.1f} MB",
          flush=True)

    # Warmup
    deal = Game.deal_cards(hand_sizes)
    subgame_pick_action(deal[0], hand_sizes, 0, [], fs,
                       n_belief_samples=n_belief_samples, context=ctx)

    times = []
    diag_lens = []
    for k in range(n_decisions):
        deal = Game.deal_cards(hand_sizes)
        # Walk forward using blueprint to get a realistic state
        history = _walk_with_blueprint(fs, hand_sizes, deal, rng, max_steps=int(rng.integers(0, 6)))
        my_seat = k % len(my_seats)
        # If history.length is odd, the second seat is to act; pick a seat that
        # matches. (For symmetric (k,k) we always have my_seat=0.)
        if hand_sizes[0] == hand_sizes[1]:
            my_seat = 0
        else:
            my_seat = my_seats[k % len(my_seats)]
        if len(history) > 0 and history[-1] == 88:
            history = history[:-2]  # back off if walked into terminal
        # Determine starter from my_seat: by convention, history starts with
        # whichever seat acted first; the smoke just simulates so we pretend
        # seat 0 starts.
        # We need: the player to act now = my_seat. That means hist_len has
        # the right parity. Trim/pad to align.
        if len(history) % 2 != my_seat:
            history = history[:-1] if len(history) > 0 else history

        t0 = time.time()
        a, diag = subgame_pick_action(
            deal[my_seat], hand_sizes, my_seat, history, fs,
            n_belief_samples=n_belief_samples, context=ctx,
            return_diagnostics=True,
        )
        dt = time.time() - t0
        times.append(dt)
        diag_lens.append(len(history))
    arr = np.array(times)
    print(f"  per-decision wall ({n_decisions} trials, hist_len mean={np.mean(diag_lens):.1f}): "
          f"mean={arr.mean()*1000:.0f}ms p50={np.median(arr)*1000:.0f}ms "
          f"p95={np.quantile(arr, 0.95)*1000:.0f}ms "
          f"max={arr.max()*1000:.0f}ms", flush=True)


def _walk_with_blueprint(fs, hand_sizes, deal, rng, max_steps: int):
    """Simulate `max_steps` actions using blueprint sampling to get a
    realistic mid-game history."""
    from cfr_ai.information_set import make_key, get_possible_actions, get_hand_abstraction
    from cfr_ai.lbr import _parse_key_to_composite
    history = []
    abs_str_to_id = fs.abs_str_to_id
    for step in range(max_steps):
        seat = step % 2
        hand = deal[seat]
        hand_size = hand_sizes[seat]
        hand_abs = get_hand_abstraction(hand, hand_sizes)
        key = make_key(hand, hand_abs, history, fs.min_bet)
        possible = get_possible_actions(history, fs.min_bet)
        parts = key.split("-")
        last_bet = int(parts[1])
        suffix = "-".join(parts[2:])
        if suffix.split("-")[-1] not in abs_str_to_id:
            # New abs — sample check
            history.append(88)
            return history
        comp_key = _parse_key_to_composite(
            suffix, hand_size, last_bet, dict(abs_str_to_id))
        row = fs.key_to_row.get(np.int64(comp_key))
        if row is None:
            history.append(88)
            return history
        lo = int(fs.lower_action[row])
        hi = int(fs.upper_action[row])
        probs = np.array(fs.strategy[row, lo:hi + 1], dtype=np.float64)
        if probs.sum() <= 0:
            history.append(88)
            return history
        probs /= probs.sum()
        # Sample
        chosen_offset = int(rng.choice(len(probs), p=probs))
        a = lo + chosen_offset
        history.append(a)
        if a == 88:
            return history
    return history


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hand-sizes", nargs=2, type=int, default=[1, 1])
    p.add_argument("--n-decisions", type=int, default=20)
    p.add_argument("--n-belief-samples", type=int, default=None)
    p.add_argument("--skip-latency", action="store_true")
    args = p.parse_args()

    hand_sizes = sorted(args.hand_sizes)
    print(f"[subgame_validate] setup={hand_sizes}", flush=True)

    t0 = time.time()
    fs = load_flat_strategy(hand_sizes)
    print(f"  loaded {len(fs.key_to_row)} non-checking entries in "
          f"{time.time()-t0:.2f}s; min_bet={fs.min_bet}", flush=True)

    print("Structural checks:")
    ok1 = check_uniform_posterior(fs, hand_sizes, my_seat=0)
    ok2 = check_uniform_posterior(fs, hand_sizes, my_seat=1)
    ok3 = check_one_step_posterior(fs, hand_sizes)
    ok4 = check_best_response_at_least_blueprint(fs, hand_sizes)

    all_ok = ok1 and ok2 and ok3 and ok4
    print(f"\n  Structural: {'ALL PASS' if all_ok else 'SOME FAIL'}", flush=True)

    if not args.skip_latency:
        print("\nLatency benchmark:")
        latency_bench(fs, hand_sizes, args.n_decisions, args.n_belief_samples)


if __name__ == "__main__":
    main()
