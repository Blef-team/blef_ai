"""Depth-1 subgame solver for the Blef CFR AI.

At a live decision point, the deployed agent already has more information
than the blueprint's abstraction encodes:

  - the exact bet history (not just the abstracted last-two codes)
  - the exact opp action sequence (which constrains opp's hand under sigma_opp)

This module turns that extra information into action choice:

  1. Enumerate (or sample) all opp hands consistent with our cards.
  2. Walk the observed history, multiplying each opp hand's reach by
     sigma_opp(action_taken | opp_hand) at each opp turn. The product is
     the unnormalised Bayesian posterior P(opp_hand | history, sigma_opp).
  3. For each legal action `a` we might play, evaluate the value of the
     resulting state via `_cfr_vs_cfr_value_jit` (from lbr), passing
     the posterior reach. Pick argmax_a V(a).

Conceptually this is one-step best response against the blueprint, with
the live history substituted for a sampled one. Compute is dominated by
the per-action CFR-vs-CFR rollout. For setups where the opp-hand
population is too large for full enumeration we sample N hands uniformly
(unbiased; the posterior in expectation matches the full enumeration).

The heavy machinery is reused from `lbr.py`:
  - composite-int64 key, `FlatStrategy` loader, `_lookup_by_key`
  - `_opp_turn_lookups` (grouped lookups amortised over abstractions)
  - `_cfr_vs_cfr_value_jit` (CFR-vs-CFR rollout with reach vector)

So this file is small: a posterior builder plus a Python wrapper that
allocates buffers, evaluates each legal action, and returns argmax.
"""

import itertools
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit, types
from numba.typed import Dict as NbDict

from cfr_ai.game import Game
from cfr_ai.lbr import (
    FlatStrategy,
    load_flat_strategy,
    intern_abstractions_for_hand,
    intern_abstractions_for_hands,
    _lookup_by_key,
    _history_to_key_parts,
    _cfr_vs_cfr_value_jit,
)
from cfr_ai.trainer import (
    LAST_BET_SHIFT, H_M1_SHIFT, H_M2_SHIFT, ABS_ID_SHIFT,
    MAX_DEPTH, _HISTORY_CODE_ID,
)


# ---------------------------------------------------------------------------
# Posterior builder
# ---------------------------------------------------------------------------

@njit(cache=False)
def _build_posterior_reach_jit(
    history_buf, hist_len, opp_starts,
    opp_hand_size, opp_abs_ids,
    key_to_row, strategy, lower_action, upper_action,
    history_code_id, cfr_min_bet,
    reach,
):
    """Walk history; multiply reach[n] by sigma_opp(action_taken | opp_hand_n)
    at each opp turn. `opp_starts=True` iff opp played history[0].

    Mutates `reach` in place. `reach` should be pre-initialised to 1.0;
    we don't reset it here so callers can do a hot-start (e.g. successive
    decisions in the same game where only the new opp turn added one factor).

    Opp hands sharing the same abstraction at this `last_bet` share the
    same sigma_opp lookup. We group them via a typed.Dict to avoid N main-
    dict hits per opp turn.
    """
    n_opp = reach.shape[0]
    scratch = np.empty(89, dtype=np.float64)

    for i in range(hist_len):
        # Whose turn was it at history[i]?
        if opp_starts:
            is_opp_turn = (i % 2 == 0)
        else:
            is_opp_turn = (i % 2 == 1)
        if not is_opp_turn:
            continue

        # Compute (last_bet, h_m1, h_m2) at the state JUST BEFORE the i-th action.
        last_bet, h_m1_id, h_m2_id = _history_to_key_parts(
            history_buf, i, cfr_min_bet, history_code_id)

        action_taken = history_buf[i]
        if last_bet == 88:
            lo = cfr_min_bet
            hi = 87
        else:
            lo = last_bet + 1
            hi = 88
        n_actions = hi - lo + 1

        # An action outside the legal range should never happen on a real
        # game log; if it does, treat it as zero-information (skip).
        if action_taken < lo or action_taken > hi:
            continue
        action_idx = action_taken - lo

        base_key = (opp_hand_size
                    | (last_bet << LAST_BET_SHIFT)
                    | (h_m1_id << H_M1_SHIFT)
                    | (h_m2_id << H_M2_SHIFT))

        # Per-abs cache: abs_id -> sigma_opp(action_taken | this abs).
        abs_to_p = NbDict.empty(key_type=types.int64, value_type=types.float64)

        for n in range(n_opp):
            if reach[n] <= 0.0:
                continue
            abs_id = opp_abs_ids[n, last_bet]
            a64 = np.int64(abs_id)
            if a64 in abs_to_p:
                p = abs_to_p[a64]
            else:
                key = base_key | (abs_id << ABS_ID_SHIFT)
                _lookup_by_key(
                    key_to_row, strategy, lower_action, upper_action,
                    key, n_actions, scratch[:n_actions],
                )
                p = scratch[action_idx]
                abs_to_p[a64] = p
            reach[n] *= p


# ---------------------------------------------------------------------------
# Python wrapper / driver
# ---------------------------------------------------------------------------

@dataclass
class SubgameContext:
    """Pre-allocated buffers + bookkeeping for repeated subgame queries with
    a fixed `(hand_sizes, flat_strategy, n_belief_samples)`. Reusing one
    context across decisions avoids re-allocating ~tens-to-hundreds of MB
    of scratch per call.

    `max_n_opp` caps n_opp (after enumeration / sampling). Buffers are sized
    for max_n_opp; actual calls may use fewer rows.
    """
    hand_sizes: Tuple[int, int]
    flat_strategy: FlatStrategy
    max_n_opp: int

    # Scratch reused across decisions
    history_buf: np.ndarray         # int64[MAX_DEPTH]
    reach: np.ndarray               # float64[max_n_opp]
    pd_buf: np.ndarray              # float64[MAX_DEPTH, max_n_opp, 89]
    marg_buf: np.ndarray            # float64[MAX_DEPTH, 89]
    nr_buf: np.ndarray              # float64[MAX_DEPTH, max_n_opp]

    def estimated_mb(self) -> float:
        return (self.pd_buf.nbytes + self.marg_buf.nbytes + self.nr_buf.nbytes
                + self.reach.nbytes + self.history_buf.nbytes) / (1024 * 1024)


def make_context(
    hand_sizes: List[int],
    flat_strategy: FlatStrategy,
    max_n_opp: int,
) -> SubgameContext:
    history_buf = np.zeros(MAX_DEPTH, dtype=np.int64)
    reach = np.zeros(max_n_opp, dtype=np.float64)
    pd_buf = np.zeros((MAX_DEPTH, max_n_opp, 89), dtype=np.float64)
    marg_buf = np.zeros((MAX_DEPTH, 89), dtype=np.float64)
    nr_buf = np.zeros((MAX_DEPTH, max_n_opp), dtype=np.float64)
    return SubgameContext(
        hand_sizes=tuple(hand_sizes),
        flat_strategy=flat_strategy,
        max_n_opp=max_n_opp,
        history_buf=history_buf,
        reach=reach,
        pd_buf=pd_buf,
        marg_buf=marg_buf,
        nr_buf=nr_buf,
    )


def warmup(
    hand_sizes: List[int],
    flat_strategy: FlatStrategy,
    context: SubgameContext,
    n_belief_samples: Optional[int] = None,
) -> None:
    """Pre-trigger JIT compilation of all code paths the solver uses.

    Numba specialises recursive functions per code path the first time
    each is exercised. Both the empty-history (posterior loop runs zero
    times) and the non-empty-history path need to be hit. Without this,
    the very first real call can take ~3 seconds on a cold container —
    fine for AWS Lambda cold-starts, but disruptive in interactive tests.

    Two calls cover the two paths:
      - empty history (no opp turns -> posterior builder JIT-compiles its
        outer structure but skips the inner branch)
      - history of length 1 from opp's seat (forces the inner branch)
    """
    # Both paths require sensible hands. Sample two arbitrary deals.
    deal = Game.deal_cards(hand_sizes)
    subgame_pick_action(
        deal[0], hand_sizes, 0, [], flat_strategy,
        n_belief_samples=n_belief_samples, context=context,
    )
    # Second call: seat 1 with hist_len=1 to exercise the posterior loop's
    # main branch.
    deal2 = Game.deal_cards(hand_sizes)
    subgame_pick_action(
        deal2[1], hand_sizes, 1, [0], flat_strategy,
        n_belief_samples=n_belief_samples, context=context,
    )


def _enumerate_or_sample_opp_hands(
    my_hand: List[int],
    opp_hand_size: int,
    n_belief_samples: Optional[int],
    rng: np.random.Generator,
) -> List[List[int]]:
    remaining = [c for c in range(24) if c not in my_hand]
    all_opp = list(itertools.combinations(remaining, opp_hand_size))
    pop = len(all_opp)
    if n_belief_samples is None or n_belief_samples >= pop:
        return [sorted(h) for h in all_opp]
    idx = rng.choice(pop, size=n_belief_samples, replace=False)
    return [sorted(all_opp[i]) for i in idx]


def subgame_pick_action(
    my_hand: List[int],
    hand_sizes: List[int],
    my_seat: int,
    history: List[int],
    flat_strategy: FlatStrategy,
    n_belief_samples: Optional[int] = None,
    seed: int = 0,
    context: Optional[SubgameContext] = None,
    return_diagnostics: bool = False,
    action_topk: Optional[int] = None,
    action_prob_threshold: float = 0.0,
) -> Tuple[int, Optional[Dict]]:
    """Pick the best action from the current state.

    Args:
        my_hand: cards we hold (0..23).
        hand_sizes: full setup `[hand_size_seat0, hand_size_seat1]`.
        my_seat: 0 or 1 — our seat in the setup.
        history: observed actions (bets + final check) so far.
        flat_strategy: blueprint loaded via `load_flat_strategy`.
        n_belief_samples: if not None and smaller than total opp-hand
            population, sample this many opp hands uniformly. The primary
            latency knob — total work is roughly proportional to
            n_opp x tree_size. For large setups (rounds 6+) this should
            generally be set to 200-500.
        seed: RNG seed for belief sampling.
        context: pre-allocated SubgameContext. Strongly recommended for
            repeated calls (e.g. across a live game).
        return_diagnostics: if True, returns extra info in the second slot.
        action_topk: if set, evaluate only the top-K actions by blueprint
            sigma_me probability. Cuts compute proportionally. Safe when
            the blueprint is reasonably well-trained — the optimal subgame
            action is almost always in sigma_me's main support. Set to
            None to disable (evaluate all legal actions).
        action_prob_threshold: actions with blueprint sigma_me probability
            below this threshold are skipped. Combine with action_topk for
            two filters in sequence.

    Returns:
        (best_action, diagnostics_or_none).

    Notes:
        - Legal actions respect the deployed agent's min_bet (i.e., we
          pick among the actions the agent would actually be allowed to
          play). This differs from LBR, which is unrestricted.
        - When `n_belief_samples` is None the posterior is exact (modulo
          fp32 noise in the stored strategy). Sampling introduces variance
          O(1/sqrt(n_belief_samples)).
    """
    opp_seat = 1 - my_seat
    my_hand_size = hand_sizes[my_seat]
    opp_hand_size = hand_sizes[opp_seat]
    min_bet = flat_strategy.min_bet

    rng = np.random.default_rng(seed)
    opp_hands = _enumerate_or_sample_opp_hands(
        my_hand, opp_hand_size, n_belief_samples, rng)
    n_opp = len(opp_hands)

    if context is not None:
        if n_opp > context.max_n_opp:
            raise ValueError(
                f"context.max_n_opp={context.max_n_opp} too small for "
                f"n_opp={n_opp}; rebuild context.")
        history_buf = context.history_buf
        reach = context.reach[:n_opp]
        pd_buf = context.pd_buf[:, :n_opp, :]
        marg_buf = context.marg_buf
        nr_buf = context.nr_buf[:, :n_opp]
    else:
        history_buf = np.zeros(MAX_DEPTH, dtype=np.int64)
        reach = np.zeros(n_opp, dtype=np.float64)
        pd_buf = np.zeros((MAX_DEPTH, n_opp, 89), dtype=np.float64)
        marg_buf = np.zeros((MAX_DEPTH, 89), dtype=np.float64)
        nr_buf = np.zeros((MAX_DEPTH, n_opp), dtype=np.float64)

    # Reset reach to 1.0; copy history.
    for n in range(n_opp):
        reach[n] = 1.0
    hist_len = len(history)
    for i, a in enumerate(history):
        history_buf[i] = a

    # Build abstraction id arrays
    my_abs_ids = intern_abstractions_for_hand(
        my_hand, hand_sizes, flat_strategy.abs_str_to_id)
    opp_abs_ids = intern_abstractions_for_hands(
        opp_hands, hand_sizes, flat_strategy.abs_str_to_id)

    # Existence array per opp hand (orientation matters: position-0 hand
    # vs position-1 hand for the set-existence calculation).
    exist = np.zeros((n_opp, 88), dtype=np.bool_)
    for i, oh in enumerate(opp_hands):
        if my_seat == 0:
            exist[i] = Game.precompute_set_existence([list(my_hand), list(oh)])
        else:
            exist[i] = Game.precompute_set_existence([list(oh), list(my_hand)])

    # Posterior reach: walk history, multiply by sigma_opp per opp turn.
    #
    # The "starting player" is fixed by the game's seating choice but isn't
    # passed into this function explicitly. We infer it from history parity:
    # if we (sub) are about to act and hist_len is even, we are the starter
    # (and opp acted at the odd indices); if hist_len is odd, opp is the
    # starter (and acted at the even indices). Either way:
    #   opp_starts <=> hist_len is odd
    opp_starts = (hist_len % 2 == 1)
    _build_posterior_reach_jit(
        history_buf, hist_len, opp_starts,
        opp_hand_size, opp_abs_ids,
        flat_strategy.key_to_row, flat_strategy.strategy,
        flat_strategy.lower_action, flat_strategy.upper_action,
        _HISTORY_CODE_ID, min_bet,
        reach,
    )

    posterior_mass = float(reach.sum())

    # Legal actions from this state, respecting min_bet (deployed semantics).
    if hist_len == 0 or history_buf[hist_len - 1] < min_bet:
        # Round-start node: agent can open with any bet >= min_bet.
        lo, hi = min_bet, 87
    else:
        lo, hi = int(history_buf[hist_len - 1]) + 1, 88
    n_actions = hi - lo + 1

    if posterior_mass <= 0.0:
        # The history is impossible under sigma_opp for every enumerated
        # opp hand: e.g. opp played an action with zero probability in
        # the blueprint for all hands we considered. Fall back to the
        # blueprint's own choice from this state.
        return _blueprint_fallback(
            my_hand, hand_sizes, my_seat, history, flat_strategy
        ), ({"reason": "zero_posterior"} if return_diagnostics else None)

    # Optional: prune candidate actions by blueprint sigma_me. This is a
    # big latency win when n_actions is large (e.g. opp opened with a low
    # bet -> we have ~80 legal continuations). Best response within
    # blueprint's support is still a meaningful improvement over the
    # blueprint's mixed sampling.
    candidate_indices = list(range(n_actions))
    sigma_me_probs = None
    if (action_topk is not None or action_prob_threshold > 0.0):
        sigma_me_probs = _lookup_my_sigma(
            flat_strategy, my_hand, my_hand_size, my_abs_ids, hand_sizes,
            history_buf, hist_len, min_bet,
            lo, hi,
        )
        # Ensure check is always considered (rare cases where the blueprint
        # has the agent never check — keep the safety net).
        check_idx = 88 - lo if hi == 88 else None
        if sigma_me_probs is not None:
            # Filter by threshold first
            mask = sigma_me_probs > action_prob_threshold
            if check_idx is not None and 0 <= check_idx < n_actions:
                mask[check_idx] = True
            # Then top-K within the masked set
            if action_topk is not None and action_topk < int(mask.sum()):
                # Rank kept actions by probability, take top-K
                rank = np.argsort(-sigma_me_probs)
                top_set = set()
                for idx in rank:
                    if mask[idx]:
                        top_set.add(int(idx))
                        if len(top_set) >= action_topk:
                            break
                if check_idx is not None and 0 <= check_idx < n_actions:
                    top_set.add(int(check_idx))
                candidate_indices = sorted(top_set)
            else:
                candidate_indices = [i for i in range(n_actions) if mask[i]]
            if not candidate_indices:
                candidate_indices = [int(np.argmax(sigma_me_probs))]

    # Evaluate V[a] for each candidate action.
    V = np.full(n_actions, -np.inf, dtype=np.float64)
    for i in candidate_indices:
        a = lo + i
        history_buf[hist_len] = a
        # After we act, opp is active in the rollout. From our perspective
        # (we are the "lbr" player here), lbr_is_active=False at the new node.
        V[i] = _cfr_vs_cfr_value_jit(
            history_buf, hist_len + 1,
            my_hand_size, my_abs_ids,
            opp_hand_size, opp_abs_ids, reach, exist,
            flat_strategy.key_to_row, flat_strategy.strategy,
            flat_strategy.lower_action, flat_strategy.upper_action,
            _HISTORY_CODE_ID, min_bet, False,
            pd_buf, marg_buf, nr_buf,
        )

    best_i = int(np.argmax(V))
    best_action = lo + best_i

    diag = None
    if return_diagnostics:
        diag = {
            "n_opp": n_opp,
            "posterior_mass": posterior_mass,
            "posterior_effective_size": float(
                (reach.sum() ** 2) / (reach * reach).sum()
            ) if posterior_mass > 0 else 0.0,
            "V": V.copy(),
            "lo": lo,
            "hi": hi,
            "best_action": best_action,
            "best_value": float(V[best_i]),
            "n_evaluated": len(candidate_indices),
            "n_legal": n_actions,
            "sigma_me_probs": sigma_me_probs.copy() if sigma_me_probs is not None else None,
        }
    return best_action, diag


def _lookup_my_sigma(
    flat_strategy, my_hand, my_hand_size, my_abs_ids, hand_sizes,
    history_buf, hist_len, min_bet, lo, hi,
):
    """Return the blueprint sigma_me as a (hi-lo+1,) array aligned with
    the action range [lo, hi]. Used by the action-filtering paths."""
    from cfr_ai.lbr import _history_to_key_parts, _composite_key
    last_bet, h_m1_id, h_m2_id = _history_to_key_parts(
        history_buf, hist_len, min_bet, _HISTORY_CODE_ID)
    abs_id = int(my_abs_ids[last_bet])
    key = _composite_key(my_hand_size, last_bet, h_m1_id, h_m2_id, abs_id)
    n_actions = hi - lo + 1
    out = np.zeros(n_actions, dtype=np.float64)
    # _lookup_by_key writes into out[:n_actions].
    _lookup_by_key(
        flat_strategy.key_to_row, flat_strategy.strategy,
        flat_strategy.lower_action, flat_strategy.upper_action,
        np.int64(key), n_actions, out,
    )
    return out


def _blueprint_fallback(
    my_hand: List[int],
    hand_sizes: List[int],
    my_seat: int,
    history: List[int],
    flat_strategy: FlatStrategy,
) -> int:
    """If the posterior collapses to zero (no opp hand can explain the
    observed history under sigma_opp), default to the blueprint's argmax.
    Rare on production policies (every action gets at least a small
    smoothing probability), but a safe fallback.
    """
    from cfr_ai.information_set import make_key, get_possible_actions
    from cfr_ai.information_set import get_hand_abstraction

    min_bet = flat_strategy.min_bet
    hand_abs = get_hand_abstraction(my_hand, hand_sizes)
    key = make_key(my_hand, hand_abs, history, min_bet)
    relevant = get_possible_actions(history, min_bet)
    parts = key.split("-")
    hand_size = int(parts[0])
    last_bet = int(parts[1])
    suffix = "-".join(parts[2:])

    # Walk through the FlatStrategy's row index by reconstructing the
    # composite key. If we can't find it, default to check.
    from cfr_ai.lbr import _parse_key_to_composite, _STR_TO_CODE_ID
    abs_str_to_id = flat_strategy.abs_str_to_id
    if suffix.split("-")[-1] not in abs_str_to_id:
        return 88  # check
    comp_key = _parse_key_to_composite(
        suffix, hand_size, last_bet, abs_str_to_id)
    row = flat_strategy.key_to_row.get(np.int64(comp_key))
    if row is None:
        return 88
    lo = int(flat_strategy.lower_action[row])
    hi = int(flat_strategy.upper_action[row])
    probs = flat_strategy.strategy[row, lo:hi + 1]
    if probs.sum() <= 0.0:
        return 88
    # Pick argmax (deterministic) rather than sample — fallback should be
    # reproducible.
    return lo + int(np.argmax(probs))


# ---------------------------------------------------------------------------
# CLI: smoke-test the solver on a couple of small setups
# ---------------------------------------------------------------------------

def _smoke():
    import argparse
    p = argparse.ArgumentParser(
        description="Quick sanity / latency smoke for the subgame solver.")
    p.add_argument("--hand-sizes", nargs=2, type=int, default=[1, 1])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-decisions", type=int, default=10,
                   help="Number of random states to time.")
    p.add_argument("--n-belief-samples", type=int, default=None,
                   help="If set, sample this many opp hands instead of full enum.")
    args = p.parse_args()

    hand_sizes = sorted(args.hand_sizes)
    print(f"[subgame smoke] setup={hand_sizes} "
          f"n_belief_samples={args.n_belief_samples}", flush=True)

    t0 = time.time()
    fs = load_flat_strategy(hand_sizes)
    print(f"  loaded {len(fs.key_to_row)} non-checking entries in "
          f"{time.time() - t0:.2f}s; min_bet={fs.min_bet}", flush=True)

    rng = np.random.default_rng(args.seed)

    # n_opp upper bound for context sizing.
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

    # Warmup
    deal = Game.deal_cards(hand_sizes)
    t0 = time.time()
    a, diag = subgame_pick_action(
        deal[0], hand_sizes, 0, [], fs,
        n_belief_samples=args.n_belief_samples,
        context=ctx, return_diagnostics=True,
    )
    print(f"  warmup: action={a}, n_opp={diag['n_opp']}, "
          f"posterior_mass={diag['posterior_mass']:.4f}, "
          f"wall={time.time() - t0:.3f}s", flush=True)

    times: List[float] = []
    for k in range(args.n_decisions):
        deal = Game.deal_cards(hand_sizes)
        # Random history: empty (round start) or first-action chosen
        if k % 2 == 1:
            history = [int(rng.integers(0, 88))]
        else:
            history = []
        my_seat = k % 2
        t0 = time.time()
        a, diag = subgame_pick_action(
            deal[my_seat], hand_sizes, my_seat, history, fs,
            n_belief_samples=args.n_belief_samples,
            context=ctx, return_diagnostics=True,
        )
        dt = time.time() - t0
        times.append(dt)
        if "best_value" in diag:
            print(f"  k={k} my_seat={my_seat} hist_len={len(history)} -> "
                  f"action={a:2d} V_best={diag['best_value']:+.4f} "
                  f"n_opp={diag['n_opp']} wall={dt:.3f}s", flush=True)
        else:
            print(f"  k={k} my_seat={my_seat} hist_len={len(history)} -> "
                  f"action={a:2d} (fallback: {diag.get('reason','?')}) "
                  f"wall={dt:.3f}s", flush=True)

    arr = np.array(times)
    print(f"  per-decision: mean={arr.mean()*1000:.0f}ms "
          f"p50={np.median(arr)*1000:.0f}ms "
          f"p95={np.quantile(arr, 0.95)*1000:.0f}ms "
          f"max={arr.max()*1000:.0f}ms", flush=True)


if __name__ == "__main__":
    _smoke()
