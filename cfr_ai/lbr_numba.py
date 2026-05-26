"""Numba-JIT'd LBR (Local Best Response) exploitability for the Blef CFR AI.

Mirrors `lbr.py` semantically but converts the CFR strategy from
`Dict[str, np.ndarray]` to a flat-array form keyed by composite int64
(same encoding as trainer_numba.py), and JIT-compiles the recursive value
functions. Speedup on production-scale setups is expected to be 5-10x,
turning some multi-day LBR runs into overnight ones.

The Python wrapper handles disk I/O, LBR-hand enumeration, and belief
sampling; the JIT handles the per-(LBR-hand, S1, S2) game-tree walk.

Composite key layout (same as trainer_numba.py):
    [abs_id : 36][h_m2 : 8][h_m1 : 8][last_bet : 8][hand_size : 4]
"""

import csv
import itertools
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from numba import njit, types
from numba.typed import Dict as NbDict
from tqdm import tqdm

from cfr_ai.game import Game
from cfr_ai.encoding import decode_probabilities
from cfr_ai.information_set import get_hand_abstraction, history_codes
from cfr_ai.trainer_numba import (
    LAST_BET_SHIFT, H_M1_SHIFT, H_M2_SHIFT, ABS_ID_SHIFT, ABSENT_CODE,
    MAX_DEPTH, _HISTORY_CODE_ID, _HISTORY_CODE_STRS,
)


# Inverse of _HISTORY_CODE_STRS: code-string -> uint8 id. Built once.
_STR_TO_CODE_ID: Dict[str, int] = {s: i for i, s in enumerate(_HISTORY_CODE_STRS)}


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


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _parse_key_to_composite(
    key_str: str,
    hand_size: int,
    last_bet: int,
    abs_str_to_id: Dict[str, int],
) -> int:
    """Parse a strategy-file key suffix (everything after 'hs-lb-') into
    a composite int64 key. Suffix can be:
      - "abs"              → 0 history codes (`hs-lb-abs`)
      - "c1-abs"           → 1 history code  (`hs-lb-c1-abs`)
      - "c1-c2-abs"        → 2 history codes (`hs-lb-c1-c2-abs`)
    where c1/c2 are entries from `_HISTORY_CODE_STRS` and `abs` is the
    hand-abstraction string."""
    parts = key_str.split("-")
    abs_str = parts[-1]
    abs_id = abs_str_to_id.get(abs_str)
    if abs_id is None:
        abs_id = len(abs_str_to_id)
        abs_str_to_id[abs_str] = abs_id

    if len(parts) == 1:
        h_m1_id = ABSENT_CODE
        h_m2_id = ABSENT_CODE
    elif len(parts) == 2:
        h_m1_id = _STR_TO_CODE_ID[parts[0]]
        h_m2_id = ABSENT_CODE
    else:  # 3 parts: c1, c2, abs
        h_m1_id = _STR_TO_CODE_ID[parts[0]]
        h_m2_id = _STR_TO_CODE_ID[parts[1]]

    return (hand_size
            | (last_bet << LAST_BET_SHIFT)
            | (h_m1_id << H_M1_SHIFT)
            | (h_m2_id << H_M2_SHIFT)
            | (abs_id << ABS_ID_SHIFT))


def load_flat_strategy(
    hand_sizes: List[int],
    setup_dir: str = None,
) -> FlatStrategy:
    """Load the CFR strategy for `hand_sizes` from disk and convert to
    `FlatStrategy`. Strategy CSVs live in `setup_dir/{hand_size}/{last_bet}.csv`;
    if `setup_dir` is None, defaults to `cfr_ai/outputs/<setup>`.

    Only non-checking entries are stored (matching the on-disk format);
    missing-key lookups in the JIT default to check-100%."""
    if setup_dir is None:
        setup_dir = os.path.join("cfr_ai", "outputs",
                                 "_".join(str(x) for x in hand_sizes))
    if not os.path.isdir(setup_dir):
        raise FileNotFoundError(f"No strategy directory at {setup_dir}")

    min_bet = 0
    md = os.path.join(setup_dir, "metadata.csv")
    if os.path.exists(md):
        with open(md, "r") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[0].strip() == "Minimum bet":
                    try:
                        min_bet = int(row[1].strip())
                    except ValueError:
                        pass

    abs_str_to_id: Dict[str, int] = {}
    # First pass: count rows so we can pre-size arrays.
    rows = []
    for hand_size in set(hand_sizes):
        size_dir = os.path.join(setup_dir, str(hand_size))
        if not os.path.isdir(size_dir):
            raise FileNotFoundError(f"No strategy folder at {size_dir}")
        for fname in os.listdir(size_dir):
            if not fname.endswith(".csv"):
                continue
            last_bet = int(fname[:-4])
            with open(os.path.join(size_dir, fname), "r") as f:
                rdr = csv.reader(f)
                next(rdr, None)
                for row in rdr:
                    if not row or not row[0]:
                        continue
                    rows.append((hand_size, last_bet, row[0], row[1]))

    n_rows = len(rows)
    strategy = np.zeros((n_rows, 89), dtype=np.float32)
    lower_action = np.zeros(n_rows, dtype=np.int16)
    upper_action = np.zeros(n_rows, dtype=np.int16)
    key_to_row = NbDict.empty(key_type=types.int64, value_type=types.int64)

    for i, (hand_size, last_bet, suffix, encoded) in enumerate(rows):
        arr = decode_probabilities(encoded).astype(np.float32)

        if last_bet == 88 or last_bet < min_bet:
            lo, hi = min_bet, 87
        else:
            lo, hi = last_bet + 1, 88
        width = hi - lo + 1

        # Trim or pad to width
        if len(arr) > width:
            arr = arr[:width]
        if len(arr) < width:
            padded = np.zeros(width, dtype=np.float32)
            padded[:len(arr)] = arr
            arr = padded

        # Normalise so per-lookup we can trust sum==1 and skip the divide.
        # Force a check-100% default (last legal action) on the rare all-zero
        # entry — saves the runtime branch in `_lookup_one_strategy`.
        s = float(arr.sum())
        if s > 0.0:
            arr = arr / s
        else:
            arr = np.zeros(width, dtype=np.float32)
            arr[-1] = 1.0

        strategy[i, lo:hi + 1] = arr
        lower_action[i] = lo
        upper_action[i] = hi
        key = _parse_key_to_composite(suffix, hand_size, last_bet, abs_str_to_id)
        key_to_row[np.int64(key)] = np.int64(i)

    return FlatStrategy(
        key_to_row=key_to_row,
        strategy=strategy,
        lower_action=lower_action,
        upper_action=upper_action,
        abs_str_to_id=abs_str_to_id,
        min_bet=int(min_bet),
    )


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

def lbr_exploitability_numba(
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
    from cfr_ai.lbr import se_relative_ci_upper

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

def main():
    """CLI mirroring `cfr_ai/lbr.py` but routed through the JIT path.
    Reuses lbr.py's `_update_summary` so the on-disk CSV format is identical."""
    import argparse, time
    from cfr_ai.lbr import _update_summary, _sampling_label

    p = argparse.ArgumentParser(
        description="LBR-based exploitability (JIT version) for the Blef CFR AI.")
    p.add_argument("--hand-sizes", nargs=2, type=int, required=True)
    p.add_argument("--depth", type=int, default=1,
                   help="Number of LBR-optimised decisions per game. "
                        "1 = LBR-1 (default), 2 = LBR-2, etc. "
                        "The JIT path does not support exact best response "
                        "(use cfr_ai.lbr for that on small setups).")
    p.add_argument("--starting-player", type=int, default=None, choices=[0, 1])
    p.add_argument("--n-belief-samples", type=int, default=300)
    p.add_argument("--n-lbr-hand-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--update-summary", action="store_true",
                   help="Append/update this setup's row in outputs/lbr_summary.csv.")
    args = p.parse_args()

    hand_sizes = sorted(args.hand_sizes)
    print(f"Loading flat CFR strategy for {hand_sizes}...", flush=True)
    t0 = time.time()
    fs = load_flat_strategy(hand_sizes)
    print(f"  loaded {len(fs.key_to_row)} non-checking entries in "
          f"{time.time() - t0:.1f}s; min_bet={fs.min_bet}", flush=True)

    if args.starting_player is not None:
        sps = [args.starting_player]
    elif hand_sizes[0] == hand_sizes[1]:
        sps = [0]
    else:
        sps = [0, 1]

    print(f"Depth: {args.depth}", flush=True)

    per_sp_results = {}
    sampling_labels = {}
    for sp in sps:
        t0 = time.time()
        result = lbr_exploitability_numba(
            hand_sizes, sp, fs, depth=args.depth,
            n_belief_samples=args.n_belief_samples,
            n_lbr_hand_samples=args.n_lbr_hand_samples,
            seed=args.seed,
        )
        dt = time.time() - t0
        per_sp_results[sp] = (result, dt)
        sampling_labels[sp] = _sampling_label(args.n_belief_samples, args.n_lbr_hand_samples)
        if result["se_worst"] == 0:
            print(f"  starting_player={sp}: exploitability={result['expl']*100:+.3f}% (exact)  ({dt:.1f}s)", flush=True)
        else:
            print(f"  starting_player={sp}: exploitability={result['expl']*100:+.3f}% "
                  f"± {result['se_worst']*100:.3f}pp (K={result['K_lbr_hand']})  ({dt:.1f}s)", flush=True)

    if args.update_summary:
        _update_summary(hand_sizes, args.depth, per_sp_results, sampling_labels)


if __name__ == "__main__":
    main()
