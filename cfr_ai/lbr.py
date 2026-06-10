"""LBR (Local Best Response) exploitability for the Blef CFR AI.

The CFR strategy is loaded as a flat-array `FlatStrategy` keyed by a
composite int64 (same encoding as `trainer.py`); the recursive value
functions are JIT-compiled with numba. The Python wrapper handles disk
I/O, LBR-hand enumeration, and belief sampling; the JIT handles the
per-(LBR-hand, S1, S2) game-tree walk.

Composite key layout (same as `trainer.py`):
    [abs_id : 36][h_m2 : 8][h_m1 : 8][last_bet : 8][hand_size : 4]
"""

import itertools
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit, types
from numba.typed import Dict as NbDict
from tqdm import tqdm

from cfr_ai.game import Game
from cfr_ai.information_set import get_hand_abstraction, history_codes
from cfr_ai.keys import (
    LAST_BET_SHIFT, H_M1_SHIFT, H_M2_SHIFT, ABS_ID_SHIFT, ABSENT_CODE,
    MAX_DEPTH, _HISTORY_CODE_ID, _HISTORY_CODE_STRS, _STR_TO_CODE_ID,
    _split_suffix,
)


# Sentinel for "as deep as you can recurse" — past this many LBR-active
# decisions LBR equals exact best response (game depth is far smaller). The
# JIT recursion handles this fine; the only practical limit on depth is the
# 88^depth combinatorial explosion at each LBR-active node, which makes
# anything past ~3 infeasible on round-4+ setups.
INF_DEPTH = 10**6


@dataclass
class FlatStrategy:
    """Read-only CFR strategy in flat-array form. Only non-checking infosets
    are stored; missing keys default to check-100% via `_default_strategy`."""
    key_to_row: object  # numba.typed.Dict[int64, int64]
    strategy: np.ndarray         # float32[n_rows, 89] — padded to width 89
    lower_action: np.ndarray     # int16[n_rows]
    upper_action: np.ndarray     # int16[n_rows]
    abs_str_to_id: Dict[str, int]  # for matching new opp hands' abstractions
    min_bet: int
    # Augmenting macros (V3+): present only for macro setups. `masses[row, k]` is
    # macro k's probability mass (m_k/T), on the SAME scale as the row's concrete
    # `strategy` slice (c/T) — concrete + masses sum to 1. The macro-aware LBR in
    # `lbr_macro.py` folds each macro onto its per-hand b* exactly like the agent.
    # Left None here; the non-macro JIT path never reads them.
    masses: Optional[np.ndarray] = None        # float64[N, n_macros]
    macro_kinds: Optional[List[str]] = None    # e.g. ["value", "difftruthy", "bluff"]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
# `_split_suffix` moved to `cfr_ai/keys.py` so the deployed agent can use it
# without pulling numba via this module's @njit decorators.


def _parse_key_to_composite(
    key_str: str,
    hand_size: int,
    last_bet: int,
    abs_str_to_id: Dict[str, int],
) -> int:
    """Parse a strategy-file key suffix (everything after 'hs-lb-') into
    a composite int64 key. See `_split_suffix` for the parse rule."""
    h_m1_id, h_m2_id, abs_str = _split_suffix(key_str)
    abs_id = abs_str_to_id.get(abs_str)
    if abs_id is None:
        abs_id = len(abs_str_to_id)
        abs_str_to_id[abs_str] = abs_id
    return (hand_size
            | (last_bet << LAST_BET_SHIFT)
            | (h_m1_id << H_M1_SHIFT)
            | (h_m2_id << H_M2_SHIFT)
            | (abs_id << ABS_ID_SHIFT))


def load_flat_strategy(
    hand_sizes: List[int],
    setup_dir: str = None,
) -> FlatStrategy:
    """Load the CFR strategy for `hand_sizes` from disk into a `FlatStrategy`.
    Default path is `cfr_ai/outputs/<setup>/strategy.npz`."""
    from cfr_ai.strategy_io import load_strategy
    if setup_dir is None:
        setup_dir = os.path.join("cfr_ai", "outputs",
                                 "_".join(str(x) for x in hand_sizes))
    return load_strategy(setup_dir)




def intern_abstractions_for_hand(
    hand: List[int],
    hand_sizes: List[int],
    abs_str_to_id: Dict[str, int],
) -> np.ndarray:
    """Build the (89,) int64 array of abstraction ids for a given hand.
    Unknown abstraction strings get assigned a new id — but we mark them
    with a high sentinel so the JIT knows the corresponding composite key
    can never match any stored row (i.e., it's a check-100% infoset)."""
    strings = get_hand_abstraction(hand, hand_sizes)
    out = np.empty(89, dtype=np.int64)
    a2i = abs_str_to_id
    for lb in range(89):
        s = strings[lb]
        i = a2i.get(s)
        if i is None:
            # New abstraction not in the trained strategy — assign a fresh
            # high id. JIT lookups for this key will miss → default policy.
            i = len(a2i)
            a2i[s] = i
        out[lb] = i
    return out


def intern_abstractions_for_hands(
    hands: List[List[int]],
    hand_sizes: List[int],
    abs_str_to_id: Dict[str, int],
) -> np.ndarray:
    """Vectorised: build (N, 89) int64 array of abstraction ids."""
    n = len(hands)
    out = np.empty((n, 89), dtype=np.int64)
    for i, h in enumerate(hands):
        out[i] = intern_abstractions_for_hand(h, hand_sizes, abs_str_to_id)
    return out


# ---------------------------------------------------------------------------
# JIT helpers
# ---------------------------------------------------------------------------

@njit(cache=False)
def _composite_key(hand_size, last_bet, h_m1_id, h_m2_id, abs_id):
    return (hand_size
            | (last_bet << LAST_BET_SHIFT)
            | (h_m1_id << H_M1_SHIFT)
            | (h_m2_id << H_M2_SHIFT)
            | (abs_id << ABS_ID_SHIFT))


@njit(cache=False)
def _history_to_key_parts(history_buf, hist_len, min_bet, history_code_id):
    """Return (last_bet, h_m1_id, h_m2_id) for the current history."""
    if hist_len == 0 or history_buf[hist_len - 1] < min_bet:
        return 88, ABSENT_CODE, ABSENT_CODE
    last_bet = history_buf[hist_len - 1]
    if hist_len > 1 and history_buf[hist_len - 2] >= min_bet:
        h_m1_id = history_code_id[last_bet, history_buf[hist_len - 2]]
        if hist_len > 2 and history_buf[hist_len - 3] >= min_bet:
            h_m2_id = history_code_id[last_bet, history_buf[hist_len - 3]]
        else:
            h_m2_id = ABSENT_CODE
    else:
        h_m1_id = ABSENT_CODE
        h_m2_id = ABSENT_CODE
    return last_bet, h_m1_id, h_m2_id


@njit(cache=False)
def _lookup_by_key(
    key_to_row, strategy, lower_action, upper_action,
    key, n_legal_actions, out,
):
    """Write the strategy for one infoset (by composite key) into
    out[0:n_legal_actions]. Stored strategies are normalised at load time
    (incl. the all-zero -> check-100% remap), so no per-lookup renormalise.
    If the key is missing, writes the check-100% default."""
    k64 = np.int64(key)
    if k64 not in key_to_row:
        for k in range(n_legal_actions - 1):
            out[k] = 0.0
        out[n_legal_actions - 1] = 1.0
        return
    row = key_to_row[k64]
    lo = lower_action[row]
    hi = upper_action[row]
    stored_w = hi - lo + 1
    w = stored_w if stored_w < n_legal_actions else n_legal_actions
    for k in range(w):
        out[k] = strategy[row, lo + k]
    for k in range(w, n_legal_actions):
        out[k] = 0.0


@njit(cache=False)
def _lookup_one_strategy(
    key_to_row, strategy, lower_action, upper_action,
    hand_size, last_bet, h_m1_id, h_m2_id, abs_id,
    min_bet, n_legal_actions, out,
):
    """Compose the key and delegate to _lookup_by_key. Used at LBR-active
    nodes where there's exactly one lookup per call (no base_key reuse)."""
    key = _composite_key(hand_size, last_bet, h_m1_id, h_m2_id, abs_id)
    _lookup_by_key(
        key_to_row, strategy, lower_action, upper_action,
        key, n_legal_actions, out,
    )


@njit(cache=False, inline='always')
def _opp_turn_lookups(
    pd, opp_abs_ids, reach, last_bet, n_opp, n_actions,
    base_key,
    key_to_row, strategy, lower_action, upper_action,
):
    """Compute pd[n, :n_actions] = reach[n] * strategy(opp_n) for each opp
    n, grouped by abstraction at this `last_bet`.

    Many opp hands share the same abstraction at this node, so they share
    a composite key, so they share a stored strategy. We do ONE main-dict
    lookup per group instead of N. Within a group, subsequent active opps
    are scaled from the first active opp in the group:
        pd[first_n] holds reach[first_n] * strat,
        pd[n]      = pd[first_n] * (reach[n] / reach[first_n])
                   = reach[n] * strat.

    Zero-reach opps are zeroed and don't contribute to the grouping.
    Returns total_reach = sum of positive reaches.
    """
    first_with_abs = NbDict.empty(key_type=types.int64, value_type=types.int64)
    total_reach = 0.0
    for n in range(n_opp):
        r = reach[n]
        if r <= 0.0:
            for k in range(n_actions):
                pd[n, k] = 0.0
            continue
        total_reach += r
        abs_id = opp_abs_ids[n, last_bet]
        a64 = np.int64(abs_id)
        if a64 in first_with_abs:
            first_n = first_with_abs[a64]
            inv = r / reach[first_n]
            for k in range(n_actions):
                pd[n, k] = pd[first_n, k] * inv
        else:
            first_with_abs[a64] = np.int64(n)
            key = base_key | (abs_id << ABS_ID_SHIFT)
            _lookup_by_key(
                key_to_row, strategy, lower_action, upper_action,
                key, n_actions, pd[n, :n_actions],
            )
            for k in range(n_actions):
                pd[n, k] *= r
    return total_reach


# ---------------------------------------------------------------------------
# JIT recursion: CFR-vs-CFR rollout
# ---------------------------------------------------------------------------

@njit(cache=False)
def _cfr_vs_cfr_value_jit(
    history_buf, hist_len,
    lbr_hand_size, lbr_abs_ids,
    opp_hand_size, opp_abs_ids, reach, exist,
    key_to_row, strategy, lower_action, upper_action,
    history_code_id, cfr_min_bet, lbr_is_active,
    per_dists_buf, marginal_buf, new_reach_buf,
):
    """E[LBR's payoff | both sides play CFR from this state]."""
    n_opp = reach.shape[0]

    if hist_len > 0 and history_buf[hist_len - 1] == 88:
        bet = history_buf[hist_len - 2]
        sign = 1.0 if lbr_is_active else -1.0
        total = 0.0
        denom = 0.0
        for n in range(n_opp):
            if reach[n] > 0.0:
                v = 1.0 if exist[n, bet] else -1.0
                total += v * reach[n]
                denom += reach[n]
        if denom <= 0.0:
            return 0.0
        return sign * total / denom

    last_bet, h_m1_id, h_m2_id = _history_to_key_parts(
        history_buf, hist_len, cfr_min_bet, history_code_id)

    if last_bet == 88:
        lo, hi = cfr_min_bet, 87
    else:
        lo, hi = last_bet + 1, 88
    n_actions = hi - lo + 1

    if lbr_is_active:
        abs_id = lbr_abs_ids[last_bet]
        lbr_dist = np.empty(n_actions, dtype=np.float64)
        _lookup_one_strategy(
            key_to_row, strategy, lower_action, upper_action,
            lbr_hand_size, last_bet, h_m1_id, h_m2_id, abs_id,
            cfr_min_bet, n_actions, lbr_dist,
        )
        total = 0.0
        for i in range(n_actions):
            if lbr_dist[i] > 0.0:
                history_buf[hist_len] = lo + i
                v = _cfr_vs_cfr_value_jit(
                    history_buf, hist_len + 1,
                    lbr_hand_size, lbr_abs_ids,
                    opp_hand_size, opp_abs_ids, reach, exist,
                    key_to_row, strategy, lower_action, upper_action,
                    history_code_id, cfr_min_bet, False,
                    per_dists_buf, marginal_buf, new_reach_buf,
                )
                total += lbr_dist[i] * v
        return total

    # Opp turn: marginalise over opp's action, with strategy lookups
    # grouped by abstraction-at-this-last_bet to amortise dict hits.
    pd = per_dists_buf[hist_len]
    base_key = (opp_hand_size
                | (last_bet << LAST_BET_SHIFT)
                | (h_m1_id << H_M1_SHIFT)
                | (h_m2_id << H_M2_SHIFT))
    total_reach = _opp_turn_lookups(
        pd, opp_abs_ids, reach, last_bet, n_opp, n_actions, base_key,
        key_to_row, strategy, lower_action, upper_action,
    )

    # Marginal into pre-allocated buffer (zeroed in-place to avoid alloc).
    marginal = marginal_buf[hist_len]
    for k in range(n_actions):
        marginal[k] = 0.0
    for n in range(n_opp):
        for k in range(n_actions):
            marginal[k] += pd[n, k]

    total = 0.0
    new_reach = new_reach_buf[hist_len]
    for k in range(n_actions):
        if marginal[k] <= 0.0:
            continue
        p_action = marginal[k] / total_reach
        inv = total_reach / marginal[k]
        for n in range(n_opp):
            new_reach[n] = pd[n, k] * inv
        history_buf[hist_len] = lo + k
        v = _cfr_vs_cfr_value_jit(
            history_buf, hist_len + 1,
            lbr_hand_size, lbr_abs_ids,
            opp_hand_size, opp_abs_ids, new_reach[:n_opp], exist,
            key_to_row, strategy, lower_action, upper_action,
            history_code_id, cfr_min_bet, True,
            per_dists_buf, marginal_buf, new_reach_buf,
        )
        total += p_action * v
    return total


# ---------------------------------------------------------------------------
# JIT recursion: single-sample LBR (action selection pass of DS)
# ---------------------------------------------------------------------------

@njit(cache=False)
def _lbr_value_jit(
    history_buf, hist_len,
    lbr_hand_size, lbr_abs_ids,
    opp_hand_size, opp_abs_ids, reach, exist,
    key_to_row, strategy, lower_action, upper_action,
    history_code_id, cfr_min_bet, lbr_is_active, depth,
    per_dists_buf, marginal_buf, new_reach_buf,
):
    n_opp = reach.shape[0]
    if hist_len > 0 and history_buf[hist_len - 1] == 88:
        bet = history_buf[hist_len - 2]
        sign = 1.0 if lbr_is_active else -1.0
        total = 0.0
        denom = 0.0
        for n in range(n_opp):
            if reach[n] > 0.0:
                v = 1.0 if exist[n, bet] else -1.0
                total += v * reach[n]
                denom += reach[n]
        if denom <= 0.0:
            return 0.0
        return sign * total / denom

    last_bet, h_m1_id, h_m2_id = _history_to_key_parts(
        history_buf, hist_len, cfr_min_bet, history_code_id)

    if lbr_is_active:
        if depth <= 0:
            return _cfr_vs_cfr_value_jit(
                history_buf, hist_len,
                lbr_hand_size, lbr_abs_ids,
                opp_hand_size, opp_abs_ids, reach, exist,
                key_to_row, strategy, lower_action, upper_action,
                history_code_id, cfr_min_bet, True,
                per_dists_buf, marginal_buf, new_reach_buf,
            )
        # LBR is unrestricted: min_bet=0 for its own choices.
        if last_bet == 88:
            lo, hi = 0, 87
        else:
            lo, hi = last_bet + 1, 88
        n_actions = hi - lo + 1
        best = -1e18
        for i in range(n_actions):
            history_buf[hist_len] = lo + i
            v = _lbr_value_jit(
                history_buf, hist_len + 1,
                lbr_hand_size, lbr_abs_ids,
                opp_hand_size, opp_abs_ids, reach, exist,
                key_to_row, strategy, lower_action, upper_action,
                history_code_id, cfr_min_bet, False, depth - 1,
                per_dists_buf, marginal_buf, new_reach_buf,
            )
            if v > best:
                best = v
        return best

    if last_bet == 88:
        lo, hi = cfr_min_bet, 87
    else:
        lo, hi = last_bet + 1, 88
    n_actions = hi - lo + 1

    # Opp turn: grouped lookups (see _opp_turn_lookups).
    pd = per_dists_buf[hist_len]
    base_key = (opp_hand_size
                | (last_bet << LAST_BET_SHIFT)
                | (h_m1_id << H_M1_SHIFT)
                | (h_m2_id << H_M2_SHIFT))
    total_reach = _opp_turn_lookups(
        pd, opp_abs_ids, reach, last_bet, n_opp, n_actions, base_key,
        key_to_row, strategy, lower_action, upper_action,
    )

    marginal = marginal_buf[hist_len]
    for k in range(n_actions):
        marginal[k] = 0.0
    for n in range(n_opp):
        for k in range(n_actions):
            marginal[k] += pd[n, k]

    total = 0.0
    new_reach = new_reach_buf[hist_len]
    for k in range(n_actions):
        if marginal[k] <= 0.0:
            continue
        p_action = marginal[k] / total_reach
        inv = total_reach / marginal[k]
        for n in range(n_opp):
            new_reach[n] = pd[n, k] * inv
        history_buf[hist_len] = lo + k
        v = _lbr_value_jit(
            history_buf, hist_len + 1,
            lbr_hand_size, lbr_abs_ids,
            opp_hand_size, opp_abs_ids, new_reach[:n_opp], exist,
            key_to_row, strategy, lower_action, upper_action,
            history_code_id, cfr_min_bet, True, depth,
            per_dists_buf, marginal_buf, new_reach_buf,
        )
        total += p_action * v
    return total


# ---------------------------------------------------------------------------
# JIT recursion: double-sampled LBR
# ---------------------------------------------------------------------------

@njit(cache=False)
def _lbr_value_ds_jit(
    history_buf, hist_len,
    lbr_hand_size, lbr_abs_ids,
    opp_hand_size,
    opp_abs_ids_S1, reach_S1, exist_S1,
    opp_abs_ids_S2, reach_S2, exist_S2,
    key_to_row, strategy, lower_action, upper_action,
    history_code_id, cfr_min_bet, lbr_is_active, depth,
    per_dists_buf_S1, per_dists_buf_S2,
    marginal_buf_S1, marginal_buf_S2,
    new_reach_buf_S1, new_reach_buf_S2,
):
    n_S1 = reach_S1.shape[0]
    n_S2 = reach_S2.shape[0]
    if hist_len > 0 and history_buf[hist_len - 1] == 88:
        bet = history_buf[hist_len - 2]
        sign = 1.0 if lbr_is_active else -1.0
        total = 0.0
        denom = 0.0
        for n in range(n_S2):
            if reach_S2[n] > 0.0:
                v = 1.0 if exist_S2[n, bet] else -1.0
                total += v * reach_S2[n]
                denom += reach_S2[n]
        if denom <= 0.0:
            return 0.0
        return sign * total / denom

    last_bet, h_m1_id, h_m2_id = _history_to_key_parts(
        history_buf, hist_len, cfr_min_bet, history_code_id)

    if lbr_is_active:
        if depth <= 0:
            return _cfr_vs_cfr_value_jit(
                history_buf, hist_len,
                lbr_hand_size, lbr_abs_ids,
                opp_hand_size, opp_abs_ids_S2, reach_S2, exist_S2,
                key_to_row, strategy, lower_action, upper_action,
                history_code_id, cfr_min_bet, True,
                per_dists_buf_S2, marginal_buf_S2, new_reach_buf_S2,
            )
        if last_bet == 88:
            lo, hi = 0, 87
        else:
            lo, hi = last_bet + 1, 88
        n_actions = hi - lo + 1
        best_a = lo
        best_v = -1e18
        for i in range(n_actions):
            history_buf[hist_len] = lo + i
            v = _lbr_value_jit(
                history_buf, hist_len + 1,
                lbr_hand_size, lbr_abs_ids,
                opp_hand_size, opp_abs_ids_S1, reach_S1, exist_S1,
                key_to_row, strategy, lower_action, upper_action,
                history_code_id, cfr_min_bet, False, depth - 1,
                per_dists_buf_S1, marginal_buf_S1, new_reach_buf_S1,
            )
            if v > best_v:
                best_v = v
                best_a = lo + i
        history_buf[hist_len] = best_a
        return _lbr_value_ds_jit(
            history_buf, hist_len + 1,
            lbr_hand_size, lbr_abs_ids,
            opp_hand_size,
            opp_abs_ids_S1, reach_S1, exist_S1,
            opp_abs_ids_S2, reach_S2, exist_S2,
            key_to_row, strategy, lower_action, upper_action,
            history_code_id, cfr_min_bet, False, depth - 1,
            per_dists_buf_S1, per_dists_buf_S2,
            marginal_buf_S1, marginal_buf_S2,
            new_reach_buf_S1, new_reach_buf_S2,
        )

    if last_bet == 88:
        lo, hi = cfr_min_bet, 87
    else:
        lo, hi = last_bet + 1, 88
    n_actions = hi - lo + 1

    # Opp turn: grouped lookups, separately for S1 and S2.
    pd_S1 = per_dists_buf_S1[hist_len]
    pd_S2 = per_dists_buf_S2[hist_len]
    base_key = (opp_hand_size
                | (last_bet << LAST_BET_SHIFT)
                | (h_m1_id << H_M1_SHIFT)
                | (h_m2_id << H_M2_SHIFT))
    total_S1 = _opp_turn_lookups(
        pd_S1, opp_abs_ids_S1, reach_S1, last_bet, n_S1, n_actions, base_key,
        key_to_row, strategy, lower_action, upper_action,
    )
    total_S2 = _opp_turn_lookups(
        pd_S2, opp_abs_ids_S2, reach_S2, last_bet, n_S2, n_actions, base_key,
        key_to_row, strategy, lower_action, upper_action,
    )

    marginal_S1 = marginal_buf_S1[hist_len]
    marginal_S2 = marginal_buf_S2[hist_len]
    for k in range(n_actions):
        marginal_S1[k] = 0.0
        marginal_S2[k] = 0.0
    for n in range(n_S1):
        for k in range(n_actions):
            marginal_S1[k] += pd_S1[n, k]
    for n in range(n_S2):
        for k in range(n_actions):
            marginal_S2[k] += pd_S2[n, k]

    total = 0.0
    new_S1 = new_reach_buf_S1[hist_len]
    new_S2 = new_reach_buf_S2[hist_len]
    for k in range(n_actions):
        if marginal_S2[k] <= 0.0:
            continue
        p_action = marginal_S2[k] / total_S2
        inv_S2 = total_S2 / marginal_S2[k]
        for n in range(n_S2):
            new_S2[n] = pd_S2[n, k] * inv_S2
        if marginal_S1[k] > 0.0:
            inv_S1 = total_S1 / marginal_S1[k]
            for n in range(n_S1):
                new_S1[n] = pd_S1[n, k] * inv_S1
        else:
            for n in range(n_S1):
                new_S1[n] = reach_S1[n]
        history_buf[hist_len] = lo + k
        v = _lbr_value_ds_jit(
            history_buf, hist_len + 1,
            lbr_hand_size, lbr_abs_ids,
            opp_hand_size,
            opp_abs_ids_S1, new_S1[:n_S1], exist_S1,
            opp_abs_ids_S2, new_S2[:n_S2], exist_S2,
            key_to_row, strategy, lower_action, upper_action,
            history_code_id, cfr_min_bet, True, depth,
            per_dists_buf_S1, per_dists_buf_S2,
            marginal_buf_S1, marginal_buf_S2,
            new_reach_buf_S1, new_reach_buf_S2,
        )
        total += p_action * v
    return total


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def lbr_exploitability(
    hand_sizes: List[int],
    starting_player: int,
    flat_strategy: FlatStrategy,
    depth: int,
    n_belief_samples: int = 300,
    n_lbr_hand_samples: int = 500,
    seed: int = 42,
    show_progress: bool = True,
) -> Dict[str, float]:
    """JIT-backed drop-in for `lbr.lbr_exploitability`. Returns
    {expl, se_worst, K_lbr_hand}."""

    def run_seat(lbr_player, seat_seed):
        opp_player = 1 - lbr_player
        lbr_hand_size = hand_sizes[lbr_player]
        opp_hand_size = hand_sizes[opp_player]

        rng = np.random.default_rng(seat_seed)
        all_lbr_hands = list(itertools.combinations(range(24), lbr_hand_size))
        lbr_hand_population = len(all_lbr_hands)
        if n_lbr_hand_samples is not None and n_lbr_hand_samples < lbr_hand_population:
            idx = rng.choice(lbr_hand_population, size=n_lbr_hand_samples, replace=False)
            all_lbr_hands = [all_lbr_hands[i] for i in idx]

        pd_S1 = np.zeros((MAX_DEPTH, n_belief_samples, 89), dtype=np.float64)
        pd_S2 = np.zeros((MAX_DEPTH, n_belief_samples, 89), dtype=np.float64)
        # Pre-allocated marginal / new_reach buffers, slotted by depth so
        # children at deeper hist_len don't clobber parent's values.
        marg_S1 = np.zeros((MAX_DEPTH, 89), dtype=np.float64)
        marg_S2 = np.zeros((MAX_DEPTH, 89), dtype=np.float64)
        nr_S1 = np.zeros((MAX_DEPTH, n_belief_samples), dtype=np.float64)
        nr_S2 = np.zeros((MAX_DEPTH, n_belief_samples), dtype=np.float64)
        history_buf = np.zeros(MAX_DEPTH, dtype=np.int64)

        per_hand_values: List[float] = []
        iterator = tqdm(
            all_lbr_hands, desc=f"LBR-NB seat={lbr_player}, start={starting_player}"
        ) if show_progress else all_lbr_hands
        for lbr_hand_t in iterator:
            lbr_hand = list(lbr_hand_t)
            lbr_abs_ids = intern_abstractions_for_hand(
                lbr_hand, hand_sizes, flat_strategy.abs_str_to_id)
            remaining = [c for c in range(24) if c not in lbr_hand_t]
            all_opp_hands = list(itertools.combinations(remaining, opp_hand_size))
            pop = len(all_opp_hands)

            def sample_belief():
                if n_belief_samples >= pop:
                    hands = [sorted(h) for h in all_opp_hands]
                else:
                    idx = rng.choice(pop, size=n_belief_samples, replace=False)
                    hands = [sorted(all_opp_hands[i]) for i in idx]
                opp_abs = intern_abstractions_for_hands(
                    hands, hand_sizes, flat_strategy.abs_str_to_id)
                exist = np.zeros((len(hands), 88), dtype=np.bool_)
                for i, oh in enumerate(hands):
                    if lbr_player == 0:
                        exist[i] = Game.precompute_set_existence([lbr_hand, oh])
                    else:
                        exist[i] = Game.precompute_set_existence([oh, lbr_hand])
                return opp_abs, exist

            opp_abs_S1, exist_S1 = sample_belief()
            opp_abs_S2, exist_S2 = sample_belief()
            reach_S1 = np.ones(opp_abs_S1.shape[0], dtype=np.float64)
            reach_S2 = np.ones(opp_abs_S2.shape[0], dtype=np.float64)
            lbr_is_active = (lbr_player == starting_player)

            v = _lbr_value_ds_jit(
                history_buf, 0,
                lbr_hand_size, lbr_abs_ids,
                opp_hand_size,
                opp_abs_S1, reach_S1, exist_S1,
                opp_abs_S2, reach_S2, exist_S2,
                flat_strategy.key_to_row, flat_strategy.strategy,
                flat_strategy.lower_action, flat_strategy.upper_action,
                _HISTORY_CODE_ID, flat_strategy.min_bet,
                lbr_is_active, depth,
                pd_S1, pd_S2,
                marg_S1, marg_S2,
                nr_S1, nr_S2,
            )
            per_hand_values.append(v)

        arr = np.array(per_hand_values)
        K = len(arr)
        mean = float(arr.mean())
        if K > 1:
            sample_std = float(arr.std(ddof=1))
            fpc = max(0.0, (lbr_hand_population - K) / (lbr_hand_population - 1))
            std_err = sample_std / (K ** 0.5) * (fpc ** 0.5)
        else:
            std_err = 0.0
        return mean, std_err, K, lbr_hand_population

    # Match production seed convention: starting_player uses `seed`, other uses `seed + 1`.
    v_sp, se_sp, K_sp, _ = run_seat(starting_player, seed)
    v_other, se_other, K_other, _ = run_seat(1 - starting_player, seed + 1)
    mean = (v_sp + v_other) / 2.0
    se = (se_sp ** 2 + se_other ** 2) ** 0.5 / 2.0
    K = max(K_sp, K_other)
    rel = se_relative_ci_upper(K)
    se_worst = se * (1.0 + rel)
    return {
        "expl": mean,
        "se_worst": se_worst,
        "K_lbr_hand": K,
    }


# ---------------------------------------------------------------------------
# CLI — drop-in replacement for `python -m cfr_ai.lbr`
# ---------------------------------------------------------------------------

def se_relative_ci_upper(K: int, alpha: float = 0.05) -> float:
    """Approx 95% upper-CI relative width on the sample std dev given K samples.
    Normal approx to chi-squared; good for K >= 20."""
    if K <= 1:
        return 0.0
    z = 1.96 if alpha == 0.05 else 2.576
    return z / (2 * (K - 1)) ** 0.5


def _sampling_label(n_belief: int, n_lbr_hand: int) -> str:
    lbr_str = str(n_lbr_hand) if n_lbr_hand is not None else "all"
    opp_str = str(n_belief) if n_belief is not None else "all"
    return f"({lbr_str}, {opp_str})"


def _expl_cell(r) -> str:
    """Format a per-sp result as the "+X.XXX% [+/- Y.YYYpp]" cell content.

    We use the ASCII "+/-" rather than the U+00B1 glyph so the summary and
    metadata CSVs stay pure ASCII — UTF-8 "±" renders as mojibake ("Â±")
    when these files are opened in Excel/cp1252 tools on Windows."""
    if r["se_worst"] == 0:
        return f"{r['expl']*100:+.3f}%"
    return f"{r['expl']*100:+.3f}% +/- {r['se_worst']*100:.3f}pp"


def _depth_label(depth: int) -> str:
    """Display/column-label form: 'inf' for the BR sentinel, str(depth) else."""
    return "inf" if depth >= INF_DEPTH else str(depth)


def _update_summary(hand_sizes, depth, per_sp_results, sampling_labels):
    """Write this setup's LBR-<depth> columns into the unified summary CSV
    AND into the setup's metadata.csv (preserves training cols + other depth
    rows already there).

    `per_sp_results` is `{starting_player: (result_dict, duration_s)}`. When
    both starting players are run, the cell value joins them with " | ".
    """
    from cfr_ai import summary as summary_mod

    sps_sorted = sorted(per_sp_results.keys())
    expl_str = " | ".join(_expl_cell(per_sp_results[sp][0]) for sp in sps_sorted)
    duration_str = " | ".join(f"{per_sp_results[sp][1]:.0f}" for sp in sps_sorted)
    sampling = sampling_labels[sps_sorted[0]]
    depth_label = _depth_label(depth)

    summary_mod.update_lbr_row(hand_sizes, depth_label,
                               expl_str, duration_str, sampling)

    setup_dir = os.path.join(
        "cfr_ai", "outputs", "_".join(str(x) for x in sorted(hand_sizes)))
    summary_mod.update_metadata_lbr(setup_dir, depth_label,
                                    expl_str, duration_str, sampling)

    print(
        f"\nWrote {summary_mod.setup_key(hand_sizes)} (LBR-{depth_label}) "
        f"to {summary_mod.SUMMARY_PATH} and {setup_dir}/metadata.csv",
        flush=True,
    )


def main():
    """Single front door for LBR. Delegates to the macro-aware dispatcher in
    `lbr_macro`, which auto-routes by strategy type: a macro strategy (carries
    `masses`, i.e. the augmented-action setups) goes to the per-hand folding
    engine; a concrete strategy goes to `lbr_exploitability` (defined here). So
    a macro setup can NEVER be silently evaluated by the concrete engine (which
    would feed it a sub-stochastic strategy). `cfr_ai.lbr` and `cfr_ai.lbr_macro`
    are the same front door.

    The lazy import is deliberate: `lbr_macro` imports this module at load time,
    so importing it at top level here would be circular."""
    from cfr_ai.lbr_macro import main as _dispatch
    _dispatch()


if __name__ == "__main__":
    main()
