"""Macro-aware Local Best Response for V3 augmenting-macro strategies.

Plain `lbr.py` loads only the concrete `probs`; for macro setups (sum>=7) those
are SUB-STOCHASTIC (concrete c/T + macro masses m/T sum to 1), so plain LBR
evaluates a broken strategy. This module folds each macro's mass onto its
per-hand b* over the node's legal concrete bets, EQUAL-SPLIT among argmax ties
(= the agent's random tie-break in expectation), then `clear_lows` — matching
the served agent (spec validated in scratch/smoke_macro_resolve.py vs the agent;
JIT validated in scratch/smoke_macro_lbr.py vs an independent reference).

    value      => b* = argmax(p)        difftruthy => argmax(p - g)
    bluff      => argmin(p - g) = argmax(g - p)
    p = p_vector(hand, total - |hand|),  g = g_vector(total)

Reuses lbr.py's key/lookup JIT helpers; adds macro variants of the three
recursions threaded with masses[n_rows,n_macros] and per-hand score arrays
[*, n_macros, 88]. For sums<=6 (no masses) it transparently delegates to lbr.py.
"""
import itertools
import os
import time
from typing import Dict, List

import numpy as np
from numba import njit
from tqdm import tqdm

from cfr_ai.game import Game
from cfr_ai.keys import (
    LAST_BET_SHIFT, H_M1_SHIFT, H_M2_SHIFT, ABS_ID_SHIFT, MAX_DEPTH,
    ABSENT_CODE, _HISTORY_CODE_ID,
)
from cfr_ai import lbr as _lbr
from cfr_ai.lbr import (
    FlatStrategy, INF_DEPTH, load_flat_strategy, lbr_exploitability,
    intern_abstractions_for_hand, intern_abstractions_for_hands,
    _lookup_by_key, _history_to_key_parts, _composite_key,
    se_relative_ci_upper, _sampling_label, _update_summary,
)
from cfr_ai.abstraction.probs import g_vector, p_vector_factored


CLIP = 0.01  # clear_lows threshold (matches encoding.clear_lows / the agent)
_KIND_CODE = {"value": 0, "difftruthy": 1, "bluff": 2}


# ---------------------------------------------------------------------------
# Loader + per-hand score precompute (Python)
# ---------------------------------------------------------------------------

def load_macro_flat_strategy(hand_sizes: List[int], setup_dir: str = None) -> FlatStrategy:
    """`load_flat_strategy` + attach `masses`/`macro_kinds` from strategy.npz.
    Row order is the npz file order for probs AND masses AND keys, so they stay
    aligned with the FlatStrategy's key_to_row (row i = file index i)."""
    fs = load_flat_strategy(hand_sizes, setup_dir=setup_dir)
    if setup_dir is None:
        setup_dir = os.path.join("cfr_ai", "outputs",
                                 "_".join(str(x) for x in hand_sizes))
    data = np.load(os.path.join(setup_dir, "strategy.npz"), allow_pickle=True)
    if "masses" in data.files and "kinds" in data.files:
        fs.masses = np.asarray(data["masses"], dtype=np.float64)
        fs.macro_kinds = [str(x) for x in data["kinds"]]
    return fs


def _scores_for_hands(hands, total_cards: int, kind_codes: np.ndarray) -> np.ndarray:
    """[len(hands), n_macros, 88] score per macro per hand. p_vector_factored is
    lru-cached by count-signature, so repeated signatures are cheap."""
    g = g_vector(total_cards)
    nm = len(kind_codes)
    out = np.empty((len(hands), nm, 88), dtype=np.float64)
    for i, hand in enumerate(hands):
        pv = p_vector_factored(list(hand), total_cards - len(hand))
        for k in range(nm):
            code = kind_codes[k]
            if code == 0:
                out[i, k] = pv
            elif code == 1:
                out[i, k] = pv - g
            else:
                out[i, k] = g - pv
    return out


def _scores_for_hand(hand, total_cards: int, kind_codes: np.ndarray) -> np.ndarray:
    """[n_macros, 88] for a single hand."""
    return _scores_for_hands([hand], total_cards, kind_codes)[0]


# ---------------------------------------------------------------------------
# JIT fold + macro-aware lookups
# ---------------------------------------------------------------------------

@njit(cache=False)
def _fold_clear_m(out, n_actions, lo, hi, masses_row, scores2d, n_macros):
    """In place on out[0:n_actions] (the concrete c/T over bets [lo..hi]): fold
    each macro's mass onto its argmax legal-concrete bet (equal-split among
    ties within 1e-9), then clear_lows (zero <1%, renormalise). Legal concrete
    bets are [lo..min(hi,87)] (88=check excluded from macro targeting)."""
    top = hi if hi <= 87 else 87
    for k in range(n_macros):
        m = masses_row[k]
        if m <= 0.0:
            continue
        best = -1.0e18
        for b in range(lo, top + 1):
            sv = scores2d[k, b]
            if sv > best:
                best = sv
        cnt = 0
        for b in range(lo, top + 1):
            if scores2d[k, b] >= best - 1e-9:
                cnt += 1
        if cnt > 0:
            share = m / cnt
            for b in range(lo, top + 1):
                if scores2d[k, b] >= best - 1e-9:
                    out[b - lo] += share
    # clear_lows over out[0:n_actions] (out sums to ~1: concrete + masses)
    s = 0.0
    for i in range(n_actions):
        s += out[i]
    if s > 0.0:
        thr = CLIP * s
        for i in range(n_actions):
            if out[i] < thr:
                out[i] = 0.0
        s2 = 0.0
        for i in range(n_actions):
            s2 += out[i]
        if s2 > 0.0:
            inv = 1.0 / s2
            for i in range(n_actions):
                out[i] *= inv


@njit(cache=False)
def _lookup_macro(key_to_row, strategy, lower_action, upper_action,
                  key, n_actions, out, masses, n_macros, scores2d):
    """Concrete lookup then per-hand macro fold+clear into out[0:n_actions].
    Missing key -> check-100% default, no fold (absent / check-only infoset)."""
    _lookup_by_key(key_to_row, strategy, lower_action, upper_action,
                   key, n_actions, out)
    k64 = np.int64(key)
    if k64 in key_to_row:
        row = key_to_row[k64]
        _fold_clear_m(out, n_actions, lower_action[row], upper_action[row],
                      masses[row], scores2d, n_macros)


@njit(cache=False)
def _opp_turn_lookups_m(pd, opp_abs_ids, reach, last_bet, n_opp, n_actions, base_key,
                        key_to_row, strategy, lower_action, upper_action,
                        masses, n_macros, opp_scores):
    """pd[n,:n_actions] = reach[n] * served_strategy(opp n). Per-hand (no
    abstraction grouping: the macro fold differs per concrete hand)."""
    total_reach = 0.0
    for n in range(n_opp):
        r = reach[n]
        if r <= 0.0:
            for k in range(n_actions):
                pd[n, k] = 0.0
            continue
        total_reach += r
        abs_id = opp_abs_ids[n, last_bet]
        key = base_key | (abs_id << ABS_ID_SHIFT)
        _lookup_by_key(key_to_row, strategy, lower_action, upper_action,
                       key, n_actions, pd[n, :n_actions])
        k64 = np.int64(key)
        if k64 in key_to_row:
            row = key_to_row[k64]
            _fold_clear_m(pd[n], n_actions, lower_action[row], upper_action[row],
                          masses[row], opp_scores[n], n_macros)
        for k in range(n_actions):
            pd[n, k] *= r
    return total_reach


@njit(cache=False)
def _cfr_vs_cfr_value_jit_m(
    history_buf, hist_len,
    lbr_hand_size, lbr_abs_ids,
    opp_hand_size, opp_abs_ids, reach, exist,
    key_to_row, strategy, lower_action, upper_action,
    history_code_id, cfr_min_bet, lbr_is_active,
    per_dists_buf, marginal_buf, new_reach_buf,
    masses, n_macros, lbr_scores, opp_scores,
):
    """E[LBR payoff | both play the served (macro-resolved) CFR strategy]."""
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
        key = _composite_key(lbr_hand_size, last_bet, h_m1_id, h_m2_id, abs_id)
        _lookup_macro(key_to_row, strategy, lower_action, upper_action,
                      key, n_actions, lbr_dist, masses, n_macros, lbr_scores)
        total = 0.0
        for i in range(n_actions):
            if lbr_dist[i] > 0.0:
                history_buf[hist_len] = lo + i
                v = _cfr_vs_cfr_value_jit_m(
                    history_buf, hist_len + 1,
                    lbr_hand_size, lbr_abs_ids,
                    opp_hand_size, opp_abs_ids, reach, exist,
                    key_to_row, strategy, lower_action, upper_action,
                    history_code_id, cfr_min_bet, False,
                    per_dists_buf, marginal_buf, new_reach_buf,
                    masses, n_macros, lbr_scores, opp_scores,
                )
                total += lbr_dist[i] * v
        return total

    pd = per_dists_buf[hist_len]
    base_key = (opp_hand_size
                | (last_bet << LAST_BET_SHIFT)
                | (h_m1_id << H_M1_SHIFT)
                | (h_m2_id << H_M2_SHIFT))
    total_reach = _opp_turn_lookups_m(
        pd, opp_abs_ids, reach, last_bet, n_opp, n_actions, base_key,
        key_to_row, strategy, lower_action, upper_action,
        masses, n_macros, opp_scores,
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
        v = _cfr_vs_cfr_value_jit_m(
            history_buf, hist_len + 1,
            lbr_hand_size, lbr_abs_ids,
            opp_hand_size, opp_abs_ids, new_reach[:n_opp], exist,
            key_to_row, strategy, lower_action, upper_action,
            history_code_id, cfr_min_bet, True,
            per_dists_buf, marginal_buf, new_reach_buf,
            masses, n_macros, lbr_scores, opp_scores,
        )
        total += p_action * v
    return total


@njit(cache=False)
def _lbr_value_jit_m(
    history_buf, hist_len,
    lbr_hand_size, lbr_abs_ids,
    opp_hand_size, opp_abs_ids, reach, exist,
    key_to_row, strategy, lower_action, upper_action,
    history_code_id, cfr_min_bet, lbr_is_active, depth,
    per_dists_buf, marginal_buf, new_reach_buf,
    masses, n_macros, lbr_scores, opp_scores,
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
            return _cfr_vs_cfr_value_jit_m(
                history_buf, hist_len,
                lbr_hand_size, lbr_abs_ids,
                opp_hand_size, opp_abs_ids, reach, exist,
                key_to_row, strategy, lower_action, upper_action,
                history_code_id, cfr_min_bet, True,
                per_dists_buf, marginal_buf, new_reach_buf,
                masses, n_macros, lbr_scores, opp_scores,
            )
        if last_bet == 88:
            lo, hi = 0, 87
        else:
            lo, hi = last_bet + 1, 88
        n_actions = hi - lo + 1
        best = -1e18
        for i in range(n_actions):
            history_buf[hist_len] = lo + i
            v = _lbr_value_jit_m(
                history_buf, hist_len + 1,
                lbr_hand_size, lbr_abs_ids,
                opp_hand_size, opp_abs_ids, reach, exist,
                key_to_row, strategy, lower_action, upper_action,
                history_code_id, cfr_min_bet, False, depth - 1,
                per_dists_buf, marginal_buf, new_reach_buf,
                masses, n_macros, lbr_scores, opp_scores,
            )
            if v > best:
                best = v
        return best

    if last_bet == 88:
        lo, hi = cfr_min_bet, 87
    else:
        lo, hi = last_bet + 1, 88
    n_actions = hi - lo + 1

    pd = per_dists_buf[hist_len]
    base_key = (opp_hand_size
                | (last_bet << LAST_BET_SHIFT)
                | (h_m1_id << H_M1_SHIFT)
                | (h_m2_id << H_M2_SHIFT))
    total_reach = _opp_turn_lookups_m(
        pd, opp_abs_ids, reach, last_bet, n_opp, n_actions, base_key,
        key_to_row, strategy, lower_action, upper_action,
        masses, n_macros, opp_scores,
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
        v = _lbr_value_jit_m(
            history_buf, hist_len + 1,
            lbr_hand_size, lbr_abs_ids,
            opp_hand_size, opp_abs_ids, new_reach[:n_opp], exist,
            key_to_row, strategy, lower_action, upper_action,
            history_code_id, cfr_min_bet, True, depth,
            per_dists_buf, marginal_buf, new_reach_buf,
            masses, n_macros, lbr_scores, opp_scores,
        )
        total += p_action * v
    return total


@njit(cache=False)
def _lbr_value_ds_jit_m(
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
    masses, n_macros, lbr_scores, opp_scores_S1, opp_scores_S2,
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
            return _cfr_vs_cfr_value_jit_m(
                history_buf, hist_len,
                lbr_hand_size, lbr_abs_ids,
                opp_hand_size, opp_abs_ids_S2, reach_S2, exist_S2,
                key_to_row, strategy, lower_action, upper_action,
                history_code_id, cfr_min_bet, True,
                per_dists_buf_S2, marginal_buf_S2, new_reach_buf_S2,
                masses, n_macros, lbr_scores, opp_scores_S2,
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
            v = _lbr_value_jit_m(
                history_buf, hist_len + 1,
                lbr_hand_size, lbr_abs_ids,
                opp_hand_size, opp_abs_ids_S1, reach_S1, exist_S1,
                key_to_row, strategy, lower_action, upper_action,
                history_code_id, cfr_min_bet, False, depth - 1,
                per_dists_buf_S1, marginal_buf_S1, new_reach_buf_S1,
                masses, n_macros, lbr_scores, opp_scores_S1,
            )
            if v > best_v:
                best_v = v
                best_a = lo + i
        history_buf[hist_len] = best_a
        return _lbr_value_ds_jit_m(
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
            masses, n_macros, lbr_scores, opp_scores_S1, opp_scores_S2,
        )

    if last_bet == 88:
        lo, hi = cfr_min_bet, 87
    else:
        lo, hi = last_bet + 1, 88
    n_actions = hi - lo + 1

    pd_S1 = per_dists_buf_S1[hist_len]
    pd_S2 = per_dists_buf_S2[hist_len]
    base_key = (opp_hand_size
                | (last_bet << LAST_BET_SHIFT)
                | (h_m1_id << H_M1_SHIFT)
                | (h_m2_id << H_M2_SHIFT))
    total_S1 = _opp_turn_lookups_m(
        pd_S1, opp_abs_ids_S1, reach_S1, last_bet, n_S1, n_actions, base_key,
        key_to_row, strategy, lower_action, upper_action,
        masses, n_macros, opp_scores_S1,
    )
    total_S2 = _opp_turn_lookups_m(
        pd_S2, opp_abs_ids_S2, reach_S2, last_bet, n_S2, n_actions, base_key,
        key_to_row, strategy, lower_action, upper_action,
        masses, n_macros, opp_scores_S2,
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
        v = _lbr_value_ds_jit_m(
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
            masses, n_macros, lbr_scores, opp_scores_S1, opp_scores_S2,
        )
        total += p_action * v
    return total


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def lbr_exploitability_macro(
    hand_sizes: List[int],
    starting_player: int,
    flat_strategy: FlatStrategy,
    depth: int,
    n_belief_samples: int = 300,
    n_lbr_hand_samples: int = 500,
    seed: int = 42,
    show_progress: bool = True,
) -> Dict[str, float]:
    """Macro-aware drop-in for lbr.lbr_exploitability. Requires
    `flat_strategy.masses`/`macro_kinds`. Returns {expl, se_worst, K_lbr_hand}."""
    assert flat_strategy.masses is not None and flat_strategy.macro_kinds, \
        "lbr_exploitability_macro requires a macro strategy (masses/kinds)"
    kind_codes = np.array([_KIND_CODE[k] for k in flat_strategy.macro_kinds],
                          dtype=np.int64)
    n_macros = int(len(kind_codes))
    masses = np.ascontiguousarray(flat_strategy.masses, dtype=np.float64)
    total_cards = int(sum(hand_sizes))

    # Stacked history-code table (see lbr._history_to_key_parts): layer 1 (h_m2)
    # is ABSENT-filled for a depth-2 strategy so composed keys match storage.
    if flat_strategy.history_depth >= 3:
        hm2_layer = _HISTORY_CODE_ID
    else:
        hm2_layer = np.full_like(_HISTORY_CODE_ID, ABSENT_CODE)
    history_code_id = np.ascontiguousarray(np.stack((_HISTORY_CODE_ID, hm2_layer)))

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
        marg_S1 = np.zeros((MAX_DEPTH, 89), dtype=np.float64)
        marg_S2 = np.zeros((MAX_DEPTH, 89), dtype=np.float64)
        nr_S1 = np.zeros((MAX_DEPTH, n_belief_samples), dtype=np.float64)
        nr_S2 = np.zeros((MAX_DEPTH, n_belief_samples), dtype=np.float64)
        history_buf = np.zeros(MAX_DEPTH, dtype=np.int64)

        per_hand_values: List[float] = []
        iterator = tqdm(
            all_lbr_hands, desc=f"LBR-macro seat={lbr_player}, start={starting_player}"
        ) if show_progress else all_lbr_hands
        for lbr_hand_t in iterator:
            lbr_hand = list(lbr_hand_t)
            lbr_abs_ids = intern_abstractions_for_hand(
                lbr_hand, hand_sizes, flat_strategy.abs_str_to_id)
            lbr_scores = np.ascontiguousarray(
                _scores_for_hand(lbr_hand, total_cards, kind_codes))
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
                scores = np.ascontiguousarray(
                    _scores_for_hands(hands, total_cards, kind_codes))
                return opp_abs, exist, scores

            opp_abs_S1, exist_S1, scores_S1 = sample_belief()
            opp_abs_S2, exist_S2, scores_S2 = sample_belief()
            reach_S1 = np.ones(opp_abs_S1.shape[0], dtype=np.float64)
            reach_S2 = np.ones(opp_abs_S2.shape[0], dtype=np.float64)
            lbr_is_active = (lbr_player == starting_player)

            v = _lbr_value_ds_jit_m(
                history_buf, 0,
                lbr_hand_size, lbr_abs_ids,
                opp_hand_size,
                opp_abs_S1, reach_S1, exist_S1,
                opp_abs_S2, reach_S2, exist_S2,
                flat_strategy.key_to_row, flat_strategy.strategy,
                flat_strategy.lower_action, flat_strategy.upper_action,
                history_code_id, flat_strategy.min_bet,
                lbr_is_active, depth,
                pd_S1, pd_S2, marg_S1, marg_S2, nr_S1, nr_S2,
                masses, n_macros, lbr_scores, scores_S1, scores_S2,
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

    v_sp, se_sp, K_sp, _ = run_seat(starting_player, seed)
    v_other, se_other, K_other, _ = run_seat(1 - starting_player, seed + 1)
    mean = (v_sp + v_other) / 2.0
    se = (se_sp ** 2 + se_other ** 2) ** 0.5 / 2.0
    K = max(K_sp, K_other)
    rel = se_relative_ci_upper(K)
    se_worst = se * (1.0 + rel)
    return {"expl": mean, "se_worst": se_worst, "K_lbr_hand": K}


# ---------------------------------------------------------------------------
# CLI — macro-aware drop-in for `python -m cfr_ai.lbr_macro`
# ---------------------------------------------------------------------------

def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Macro-aware LBR exploitability for V3 (folds per-hand b*).")
    p.add_argument("--hand-sizes", nargs=2, type=int, required=True)
    p.add_argument("--setup-dir", type=str, default=None)
    p.add_argument("--depth", type=int, default=INF_DEPTH)
    p.add_argument("--starting-player", type=int, default=None, choices=[0, 1])
    p.add_argument("--n-belief-samples", type=int, default=300)
    p.add_argument("--n-lbr-hand-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--update-summary", action="store_true")
    args = p.parse_args()

    hand_sizes = sorted(args.hand_sizes)
    print(f"Loading macro CFR strategy for {hand_sizes}"
          f"{' from ' + args.setup_dir if args.setup_dir else ''}...", flush=True)
    t0 = time.time()
    fs = load_macro_flat_strategy(hand_sizes, setup_dir=args.setup_dir)
    is_macro = fs.masses is not None and fs.macro_kinds
    print(f"  loaded {len(fs.key_to_row)} entries in {time.time() - t0:.1f}s; "
          f"min_bet={fs.min_bet}; macro={'yes ' + str(fs.macro_kinds) if is_macro else 'NO (delegating to plain LBR)'}",
          flush=True)

    if args.starting_player is not None:
        sps = [args.starting_player]
    elif hand_sizes[0] == hand_sizes[1]:
        sps = [0]
    else:
        sps = [0, 1]

    depth_display = "inf (= BR)" if args.depth >= INF_DEPTH else str(args.depth)
    print(f"Depth: {depth_display}", flush=True)

    runner = lbr_exploitability_macro if is_macro else lbr_exploitability
    per_sp_results = {}
    sampling_labels = {}
    for sp in sps:
        t0 = time.time()
        result = runner(
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
                  f"+/- {result['se_worst']*100:.3f}pp (K={result['K_lbr_hand']})  ({dt:.1f}s)", flush=True)

    if args.update_summary:
        _update_summary(hand_sizes, args.depth, per_sp_results, sampling_labels)


if __name__ == "__main__":
    main()
