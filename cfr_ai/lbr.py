"""Local Best Response (LBR) exploitability for the Blef CFR AI.

LBR is a depth-bounded best response: at each LBR decision point, LBR considers
every action and picks the one with highest expected value, evaluating the
continuation by recursing LBR-vs-CFR up to `depth` more LBR decisions and then
falling back to CFR-vs-CFR rollout. With depth -> infinity (in practice, a number
larger than any possible round depth), LBR equals the exact best response, which
gives us a verification handle against `exploitability.py`.

Per-LBR-hand expected value is computed analytically by tracking a reach-probability
vector over the opponent's concrete hands. The CFR strategy is looked up via the existing abstraction,
but LBR itself is unrestricted.
"""

import argparse
import csv
import itertools
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from cfr_ai.game import Game, BlefCards
from cfr_ai.information_set import (
    make_key, get_hand_abstraction, get_possible_actions
)
CFRStrategy = Dict[str, np.ndarray]
INF_DEPTH = 10**6


def load_cfr_strategy(hand_sizes: List[int]) -> Tuple[CFRStrategy, int]:
    """Load saved strategies from disk for a setup and renormalise each array
    to sum to 1.

    Reads `cfr_ai/outputs/<setup>/strategy.npz` (the canonical format),
    then converts to the legacy `Dict[str, np.ndarray]` form this LBR
    reference implementation consumes. The numba LBR
    (`lbr_numba.lbr_exploitability_numba`) skips this conversion and uses
    the composite-int64 keyed FlatStrategy directly.
    """
    from cfr_ai.strategy_io import load_strategy
    from cfr_ai.trainer_numba import (
        LAST_BET_SHIFT, H_M1_SHIFT, H_M2_SHIFT, ABS_ID_SHIFT, ABSENT_CODE,
        _HISTORY_CODE_STRS,
    )

    setup_dir = os.path.join("cfr_ai", "outputs",
                             "_".join(str(x) for x in hand_sizes))
    fs = load_strategy(setup_dir)
    id_to_abs = {v: k for k, v in fs.abs_str_to_id.items()}

    strategies: CFRStrategy = {}
    for k_int, row in fs.key_to_row.items():
        k = int(k_int)
        row = int(row)
        hand_size = k & 0xF
        last_bet = (k >> LAST_BET_SHIFT) & 0xFF
        h_m1_id = (k >> H_M1_SHIFT) & 0xFF
        h_m2_id = (k >> H_M2_SHIFT) & 0xFF
        abs_id = k >> ABS_ID_SHIFT

        key_str = f"{hand_size}-{last_bet}-"
        if h_m1_id != ABSENT_CODE:
            key_str += _HISTORY_CODE_STRS[h_m1_id] + "-"
            if h_m2_id != ABSENT_CODE:
                key_str += _HISTORY_CODE_STRS[h_m2_id] + "-"
        key_str += id_to_abs[abs_id]

        lo = int(fs.lower_action[row])
        hi = int(fs.upper_action[row])
        arr = fs.strategy[row, lo:hi + 1].astype(np.float64)
        s = arr.sum()
        if s > 0:
            arr = arr / s
        strategies[key_str] = arr

    return strategies, fs.min_bet


def _cfr_action_dist(
    cfr_strategy: CFRStrategy,
    hand: List[int],
    hand_abstractions: List[str],
    history: List[int],
    cfr_min_bet: int,
) -> np.ndarray:
    """CFR's action distribution over `get_possible_actions(history, cfr_min_bet)`."""
    n = len(get_possible_actions(history, cfr_min_bet))
    key = make_key(hand, hand_abstractions, history, cfr_min_bet)
    return _resolve_strategy(cfr_strategy.get(key), n)


def _resolve_strategy(arr, n: int) -> np.ndarray:
    """Coerce a loaded strategy array to length n, with check-100% default."""
    if arr is None:
        default = np.zeros(n)
        default[-1] = 1.0
        return default
    if len(arr) < n:
        padded = np.zeros(n)
        padded[: len(arr)] = arr
        return padded
    if len(arr) > n:
        arr = arr[:n]
    if arr.sum() <= 0:
        default = np.zeros(n)
        default[-1] = 1.0
        return default
    return arr


def _per_opp_strategies(
    cfr_strategy: CFRStrategy,
    opp_hands: List[List[int]],
    opp_abstractions: List[List[str]],
    history: List[int],
    cfr_min_bet: int,
    n_actions: int,
) -> np.ndarray:
    """Batched version: returns (N, n_actions) matrix of opp's per-hand CFR
    distributions. Many opp hands share the same CFR strategy (since CFR keys
    only by hand abstraction), so we group by abstraction and look up once per
    group. Saves ~10-100x vs naive per-hand loop on deeper setups.
    """
    # Effective last_bet that make_key uses, recomputed once.
    last_bet_eff = 88 if not history or history[-1] < cfr_min_bet else history[-1]
    # Group opp indices by the part of make_key that varies across opp hands:
    # just hand_abstractions[last_bet_eff]. The fixed parts (hand_size, last_bet,
    # history codes) are the same for all opp hands at this node.
    groups: Dict[str, List[int]] = {}
    for i, abs_list in enumerate(opp_abstractions):
        abs_key = abs_list[last_bet_eff]
        if abs_key not in groups:
            groups[abs_key] = []
        groups[abs_key].append(i)
    out = np.empty((len(opp_hands), n_actions), dtype=np.float64)
    for abs_key, idxs in groups.items():
        # One lookup per abstraction; broadcast to all opp hands sharing it.
        i0 = idxs[0]
        key = make_key(opp_hands[i0], opp_abstractions[i0], history, cfr_min_bet)
        strategy = _resolve_strategy(cfr_strategy.get(key), n_actions)
        out[idxs] = strategy
    return out


def _cfr_vs_cfr_value(
    history: List[int],
    lbr_hand: List[int],
    lbr_abstractions: List[str],
    opp_hands: List[List[int]],
    opp_abstractions: List[List[str]],
    reach_probs: np.ndarray,
    cfr_strategy: CFRStrategy,
    hand_sizes: List[int],
    cfr_min_bet: int,
    lbr_is_active: bool,
    existence_table: np.ndarray,
) -> float:
    """E[LBR's payoff | both sides play CFR from this state]."""
    if Game.check_finish(history):
        bet = history[-2]
        wins = np.where(existence_table[:, bet], 1.0, -1.0)
        sign = 1.0 if lbr_is_active else -1.0
        return sign * np.dot(wins, reach_probs) / reach_probs.sum()

    active_mask = reach_probs > 0
    if not active_mask.all():
        reach_probs = reach_probs[active_mask]
        opp_hands = [h for i, h in enumerate(opp_hands) if active_mask[i]]
        opp_abstractions = [a for i, a in enumerate(opp_abstractions) if active_mask[i]]
        existence_table = existence_table[active_mask]

    if lbr_is_active:
        possible_actions = get_possible_actions(history, cfr_min_bet)
        lbr_dist = _cfr_action_dist(cfr_strategy, lbr_hand, lbr_abstractions, history, cfr_min_bet)
        total = 0.0
        for i, a in enumerate(possible_actions):
            if lbr_dist[i] > 0:
                # Stack-style mutation: avoid allocating a fresh list per child.
                history.append(a)
                v = _cfr_vs_cfr_value(
                    history, lbr_hand, lbr_abstractions, opp_hands, opp_abstractions,
                    reach_probs, cfr_strategy, hand_sizes, cfr_min_bet, False, existence_table,
                )
                history.pop()
                total += lbr_dist[i] * v
        return total
    else:
        possible_actions = get_possible_actions(history, cfr_min_bet)
        n = len(possible_actions)
        per_hand_dists = _per_opp_strategies(
            cfr_strategy, opp_hands, opp_abstractions, history, cfr_min_bet, n,
        )
        joint = reach_probs[:, None] * per_hand_dists
        marginal = joint.sum(axis=0)
        total_reach = reach_probs.sum()
        total = 0.0
        for i, a in enumerate(possible_actions):
            if marginal[i] > 0:
                p_action = marginal[i] / total_reach
                new_reach = joint[:, i] / marginal[i] * total_reach
                history.append(a)
                v = _cfr_vs_cfr_value(
                    history, lbr_hand, lbr_abstractions, opp_hands, opp_abstractions,
                    new_reach, cfr_strategy, hand_sizes, cfr_min_bet, True, existence_table,
                )
                history.pop()
                total += p_action * v
        return total


def _lbr_value(
    history: List[int],
    lbr_hand: List[int],
    lbr_abstractions: List[str],
    opp_hands: List[List[int]],
    opp_abstractions: List[List[str]],
    reach_probs: np.ndarray,
    cfr_strategy: CFRStrategy,
    hand_sizes: List[int],
    cfr_min_bet: int,
    lbr_is_active: bool,
    existence_table: np.ndarray,
    depth: int,
) -> float:
    """E[LBR's payoff | LBR plays depth-d lookahead, CFR plays its strategy]."""
    if Game.check_finish(history):
        bet = history[-2]
        wins = np.where(existence_table[:, bet], 1.0, -1.0)
        sign = 1.0 if lbr_is_active else -1.0
        return sign * np.dot(wins, reach_probs) / reach_probs.sum()

    active_mask = reach_probs > 0
    if not active_mask.all():
        reach_probs = reach_probs[active_mask]
        opp_hands = [h for i, h in enumerate(opp_hands) if active_mask[i]]
        opp_abstractions = [a for i, a in enumerate(opp_abstractions) if active_mask[i]]
        existence_table = existence_table[active_mask]

    if lbr_is_active:
        # depth K means "LBR makes K optimised decisions"; K=0 means LBR plays CFR.
        if depth <= 0:
            return _cfr_vs_cfr_value(
                history, lbr_hand, lbr_abstractions, opp_hands, opp_abstractions,
                reach_probs, cfr_strategy, hand_sizes, cfr_min_bet, True, existence_table,
            )
        # LBR is unrestricted (min_bet=0).
        possible_actions = get_possible_actions(history, min_bet=0)
        best = -np.inf
        for a in possible_actions:
            history.append(a)
            v = _lbr_value(
                history, lbr_hand, lbr_abstractions, opp_hands, opp_abstractions,
                reach_probs, cfr_strategy, hand_sizes, cfr_min_bet, False,
                existence_table, depth - 1,
            )
            history.pop()
            if v > best:
                best = v
        return best
    else:
        possible_actions = get_possible_actions(history, cfr_min_bet)
        n = len(possible_actions)
        per_hand_dists = _per_opp_strategies(
            cfr_strategy, opp_hands, opp_abstractions, history, cfr_min_bet, n,
        )
        joint = reach_probs[:, None] * per_hand_dists
        marginal = joint.sum(axis=0)
        total_reach = reach_probs.sum()
        total = 0.0
        for i, a in enumerate(possible_actions):
            if marginal[i] > 0:
                p_action = marginal[i] / total_reach
                new_reach = joint[:, i] / marginal[i] * total_reach
                history.append(a)
                v = _lbr_value(
                    history, lbr_hand, lbr_abstractions, opp_hands, opp_abstractions,
                    new_reach, cfr_strategy, hand_sizes, cfr_min_bet, True,
                    existence_table, depth,
                )
                history.pop()
                total += p_action * v
        return total


def _lbr_value_ds(
    history, lbr_hand, lbr_abstractions,
    opp_hands_S1, opp_abs_S1, reach_S1, exist_S1,
    opp_hands_S2, opp_abs_S2, reach_S2, exist_S2,
    cfr_strategy, hand_sizes, cfr_min_bet, lbr_is_active, depth,
):
    """Double-sampled LBR value: S1 is used for LBR's argmax (action selection),
    S2 (independent) is used for valuation. Removes the winner's-curse bias of
    single-sample LBR. Returns LBR's value from LBR's perspective, under S2."""
    if Game.check_finish(history):
        bet = history[-2]
        wins = np.where(exist_S2[:, bet], 1.0, -1.0)
        sign = 1.0 if lbr_is_active else -1.0
        return sign * np.dot(wins, reach_S2) / reach_S2.sum()

    mask_S1 = reach_S1 > 0
    if not mask_S1.all():
        reach_S1 = reach_S1[mask_S1]
        opp_hands_S1 = [h for i, h in enumerate(opp_hands_S1) if mask_S1[i]]
        opp_abs_S1 = [a for i, a in enumerate(opp_abs_S1) if mask_S1[i]]
        exist_S1 = exist_S1[mask_S1]
    mask_S2 = reach_S2 > 0
    if not mask_S2.all():
        reach_S2 = reach_S2[mask_S2]
        opp_hands_S2 = [h for i, h in enumerate(opp_hands_S2) if mask_S2[i]]
        opp_abs_S2 = [a for i, a in enumerate(opp_abs_S2) if mask_S2[i]]
        exist_S2 = exist_S2[mask_S2]

    if lbr_is_active:
        if depth <= 0:
            # No more LBR decisions; CFR rollout valued under S2 only.
            return _cfr_vs_cfr_value(
                history, lbr_hand, lbr_abstractions, opp_hands_S2, opp_abs_S2, reach_S2,
                cfr_strategy, hand_sizes, cfr_min_bet, True, exist_S2,
            )
        # Choose action using S1's belief via a standard single-sample LBR call.
        possible_actions = get_possible_actions(history, min_bet=0)
        best_a = None
        best_v_S1 = -np.inf
        for a in possible_actions:
            history.append(a)
            v_S1 = _lbr_value(
                history, lbr_hand, lbr_abstractions,
                opp_hands_S1, opp_abs_S1, reach_S1,
                cfr_strategy, hand_sizes, cfr_min_bet, False, exist_S1, depth - 1,
            )
            history.pop()
            if v_S1 > best_v_S1:
                best_v_S1 = v_S1
                best_a = a
        # Evaluate chosen action under S2, propagating both for any further LBR turns.
        history.append(best_a)
        v = _lbr_value_ds(
            history, lbr_hand, lbr_abstractions,
            opp_hands_S1, opp_abs_S1, reach_S1, exist_S1,
            opp_hands_S2, opp_abs_S2, reach_S2, exist_S2,
            cfr_strategy, hand_sizes, cfr_min_bet, False, depth - 1,
        )
        history.pop()
        return v
    else:
        possible_actions = get_possible_actions(history, cfr_min_bet)
        n = len(possible_actions)
        per_hand_dists_S1 = _per_opp_strategies(cfr_strategy, opp_hands_S1, opp_abs_S1, history, cfr_min_bet, n)
        per_hand_dists_S2 = _per_opp_strategies(cfr_strategy, opp_hands_S2, opp_abs_S2, history, cfr_min_bet, n)
        joint_S1 = reach_S1[:, None] * per_hand_dists_S1
        marginal_S1 = joint_S1.sum(axis=0)
        total_reach_S1 = reach_S1.sum()
        joint_S2 = reach_S2[:, None] * per_hand_dists_S2
        marginal_S2 = joint_S2.sum(axis=0)
        total_reach_S2 = reach_S2.sum()
        total = 0.0
        for i, a in enumerate(possible_actions):
            if marginal_S2[i] <= 0:
                continue
            p_action = marginal_S2[i] / total_reach_S2  # unbiased action weight via S2
            new_reach_S2 = joint_S2[:, i] / marginal_S2[i] * total_reach_S2
            if marginal_S1[i] > 0:
                new_reach_S1 = joint_S1[:, i] / marginal_S1[i] * total_reach_S1
            else:
                # S1 didn't anticipate this opp action; keep belief unchanged as fallback.
                new_reach_S1 = reach_S1
            history.append(a)
            v = _lbr_value_ds(
                history, lbr_hand, lbr_abstractions,
                opp_hands_S1, opp_abs_S1, new_reach_S1, exist_S1,
                opp_hands_S2, opp_abs_S2, new_reach_S2, exist_S2,
                cfr_strategy, hand_sizes, cfr_min_bet, True, depth,
            )
            history.pop()
            total += p_action * v
        return total


def expected_value_lbr_at_position_ds(
    lbr_player, starting_player, hand_sizes, cfr_strategy, cfr_min_bet, depth,
    n_belief_samples, n_lbr_hand_samples=None, seed=42, show_progress=True,
):
    """Double-sampled LBR over LBR hands. Two independent belief vectors per
    lbr_hand: S1 for action selection, S2 for valuation. Conservative lower bound
    on LBR_true.

    Returns (mean, std_err, n_lbr_hands_used). std_err uses FPC."""
    if n_belief_samples is None:
        raise ValueError("Double-sampled LBR requires n_belief_samples to be set")
    rng = np.random.default_rng(seed)
    lbr_hand_size = hand_sizes[lbr_player]
    opp_hand_size = hand_sizes[1 - lbr_player]
    all_lbr_hands = list(itertools.combinations(range(24), lbr_hand_size))
    lbr_hand_population = len(all_lbr_hands)
    if n_lbr_hand_samples is not None and n_lbr_hand_samples < len(all_lbr_hands):
        idx = rng.choice(len(all_lbr_hands), size=n_lbr_hand_samples, replace=False)
        all_lbr_hands = [all_lbr_hands[i] for i in idx]
    per_hand_values: List[float] = []
    iterator = tqdm(all_lbr_hands, desc=f"LBR-DS seat={lbr_player}, start={starting_player}") if show_progress else all_lbr_hands
    for lbr_hand_t in iterator:
        lbr_hand = list(lbr_hand_t)
        lbr_abstractions = get_hand_abstraction(lbr_hand, hand_sizes)
        remaining = [c for c in range(24) if c not in lbr_hand_t]

        all_opp_hands = list(itertools.combinations(remaining, opp_hand_size))
        pop = len(all_opp_hands)

        def _sample_belief():
            if n_belief_samples >= pop:
                hands = [sorted(h) for h in all_opp_hands]
            else:
                idx = rng.choice(pop, size=n_belief_samples, replace=False)
                hands = [sorted(all_opp_hands[i]) for i in idx]
            abstractions = [get_hand_abstraction(h, hand_sizes) for h in hands]
            existence = np.zeros((len(hands), 88), dtype=np.bool_)
            for i, oh in enumerate(hands):
                if lbr_player == 0:
                    existence[i] = Game.precompute_set_existence([lbr_hand, oh])
                else:
                    existence[i] = Game.precompute_set_existence([oh, lbr_hand])
            return hands, abstractions, existence

        opp_hands_S1, opp_abs_S1, exist_S1 = _sample_belief()
        opp_hands_S2, opp_abs_S2, exist_S2 = _sample_belief()
        reach_S1 = np.ones(len(opp_hands_S1), dtype=np.float64)
        reach_S2 = np.ones(len(opp_hands_S2), dtype=np.float64)
        lbr_is_active = (lbr_player == starting_player)
        v = _lbr_value_ds(
            [], lbr_hand, lbr_abstractions,
            opp_hands_S1, opp_abs_S1, reach_S1, exist_S1,
            opp_hands_S2, opp_abs_S2, reach_S2, exist_S2,
            cfr_strategy, hand_sizes, cfr_min_bet, lbr_is_active, depth,
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
    return mean, std_err, K


def expected_value_lbr_at_position(
    lbr_player: int,
    starting_player: int,
    hand_sizes: List[int],
    cfr_strategy: CFRStrategy,
    cfr_min_bet: int,
    depth: int,
    n_belief_samples: int = None,
    n_lbr_hand_samples: int = None,
    seed: int = 42,
    show_progress: bool = True,
) -> float:
    """E_{deal} [ LBR's payoff | LBR plays seat lbr_player, starting_player fixed ].

    When `n_belief_samples` is None, the opp-hand belief is enumerated exhaustively.
    When set, the belief is approximated by sampling that many opp hands uniformly
    *without* replacement. Variance shrinks as 1/sqrt(N), reduced further by the
    finite-population correction (P-N)/(P-1) when N is a meaningful fraction of P.

    When `n_lbr_hand_samples` is None, the LBR's hand is enumerated. When set, only
    that many LBR hands are sampled uniformly without replacement.

    Returns (mean, std_err, n_lbr_hands_used). std_err is the sample std of
    per-lbr_hand values divided by sqrt(K), with FPC; 0 if K == population.
    Note this only captures lbr_hand-sampling noise; opp-sampling noise within
    each per-lbr_hand value is not propagated.
    """
    rng = np.random.default_rng(seed)
    lbr_hand_size = hand_sizes[lbr_player]
    opp_hand_size = hand_sizes[1 - lbr_player]
    all_lbr_hands = list(itertools.combinations(range(24), lbr_hand_size))
    lbr_hand_population = len(all_lbr_hands)
    if n_lbr_hand_samples is not None and n_lbr_hand_samples < len(all_lbr_hands):
        idx = rng.choice(len(all_lbr_hands), size=n_lbr_hand_samples, replace=False)
        all_lbr_hands = [all_lbr_hands[i] for i in idx]
    per_hand_values: List[float] = []
    iterator = tqdm(all_lbr_hands, desc=f"LBR seat={lbr_player}, start={starting_player}") if show_progress else all_lbr_hands
    for lbr_hand_t in iterator:
        lbr_hand = list(lbr_hand_t)
        lbr_abstractions = get_hand_abstraction(lbr_hand, hand_sizes)
        remaining = [c for c in range(24) if c not in lbr_hand_t]
        all_opp_hands = list(itertools.combinations(remaining, opp_hand_size))
        if n_belief_samples is None or n_belief_samples >= len(all_opp_hands):
            opp_hands = [sorted(h) for h in all_opp_hands]
        else:
            idx = rng.choice(len(all_opp_hands), size=n_belief_samples, replace=False)
            opp_hands = [sorted(all_opp_hands[i]) for i in idx]
        opp_abstractions = [get_hand_abstraction(h, hand_sizes) for h in opp_hands]
        existence_table = np.zeros((len(opp_hands), 88), dtype=np.bool_)
        for i, oh in enumerate(opp_hands):
            if lbr_player == 0:
                existence_table[i] = Game.precompute_set_existence([lbr_hand, oh])
            else:
                existence_table[i] = Game.precompute_set_existence([oh, lbr_hand])
        reach_probs = np.ones(len(opp_hands), dtype=np.float64)
        lbr_is_active = (lbr_player == starting_player)
        v_lbr = _lbr_value(
            [], lbr_hand, lbr_abstractions, opp_hands, opp_abstractions, reach_probs,
            cfr_strategy, hand_sizes, cfr_min_bet, lbr_is_active, existence_table, depth,
        )
        per_hand_values.append(v_lbr)
    arr = np.array(per_hand_values)
    K = len(arr)
    mean = float(arr.mean())
    if K > 1:
        # Sample std err of the mean, with finite-population correction.
        sample_std = float(arr.std(ddof=1))
        fpc = max(0.0, (lbr_hand_population - K) / (lbr_hand_population - 1))
        std_err = sample_std / (K ** 0.5) * (fpc ** 0.5)
    else:
        std_err = 0.0
    return mean, std_err, K


def lbr_exploitability(
    hand_sizes: List[int],
    starting_player: int,
    cfr_strategy: CFRStrategy,
    cfr_min_bet: int,
    depth: int,
    n_belief_samples: int = None,
    n_lbr_hand_samples: int = None,
    seed: int = 42,
) -> Dict[str, float]:
    """Computes a conservative LOWER bound on LBR-K exploitability.

    Returns {expl, se_worst, K_lbr_hand}:
      - expl: double-sampled LBR estimate (≤ LBR_true in expectation)
      - se_worst: upper end of the 95% CI for the std error (worst-case),
                  combining lbr_hand sampling noise and opp belief noise
      - K_lbr_hand: effective sample size used (min over the two seat calls)

    When belief is enumerated (n_belief_samples is None or >= population for
    every lbr_hand), DS reduces to the exact LBR-K value and se_worst = 0.
    """
    fn = expected_value_lbr_at_position_ds if n_belief_samples is not None else expected_value_lbr_at_position
    v_sp, se_sp, K_sp = fn(
        lbr_player=starting_player, starting_player=starting_player,
        hand_sizes=hand_sizes, cfr_strategy=cfr_strategy, cfr_min_bet=cfr_min_bet,
        depth=depth, n_belief_samples=n_belief_samples,
        n_lbr_hand_samples=n_lbr_hand_samples, seed=seed,
    )
    v_other, se_other, K_other = fn(
        lbr_player=1 - starting_player, starting_player=starting_player,
        hand_sizes=hand_sizes, cfr_strategy=cfr_strategy, cfr_min_bet=cfr_min_bet,
        depth=depth, n_belief_samples=n_belief_samples,
        n_lbr_hand_samples=n_lbr_hand_samples, seed=seed + 1,
    )
    expl = (v_sp - (-v_other)) / 2.0
    se = (se_sp ** 2 + se_other ** 2) ** 0.5 / 2.0
    # Effective K for the se's chi-squared CI: the larger of the two seat samples,
    # since the side that's actually sampled dominates the variance contribution
    # (the enumerated side contributes 0 to se).
    K = max(K_sp, K_other)
    rel = se_relative_ci_upper(K)
    se_worst = se * (1.0 + rel)
    return {"expl": expl, "se_worst": se_worst, "K_lbr_hand": K}


def se_relative_ci_upper(K: int, alpha: float = 0.05) -> float:
    """Approx 95% upper-CI relative width on the sample std dev given K samples.
    se_true could be up to se_observed * (1 + r). Normal approx to chi-squared,
    good for K >= 20."""
    if K <= 1:
        return 0.0
    z = 1.96 if alpha == 0.05 else 2.576
    return z / (2 * (K - 1)) ** 0.5


SUMMARY_PATH = os.path.join("cfr_ai", "outputs", "lbr_summary.csv")
# Fixed columns on the left; LBR-K columns grow to the right as new depths are run.
SUMMARY_BASE_FIELDS = ["Setup", "Finished", "Sampling"]
SUMMARY_TRAILING_FIELDS: List[str] = []


def _depth_label(depth: int) -> str:
    return "inf" if depth >= INF_DEPTH else str(depth)


def _depth_cols(label: str) -> Tuple[str, str]:
    return f"LBR-{label} expl", f"LBR-{label} duration"


def _sort_depth_labels(labels):
    """Sort numeric depths ascending, with 'inf' at the end."""
    def key(d):
        return (1, 0) if d == "inf" else (0, int(d))
    return sorted(labels, key=key)


def _sampling_label(n_belief: int, n_lbr_hand: int) -> str:
    """Returns (lbr_cap, opp_cap). Populations smaller than the cap are enumerated."""
    lbr_str = str(n_lbr_hand) if n_lbr_hand is not None else "all"
    opp_str = str(n_belief) if n_belief is not None else "all"
    return f"({lbr_str}, {opp_str})"


def _update_summary(hand_sizes, depth, per_sp_results, sampling_labels):
    """Read existing rows, set this setup's LBR-<depth> columns (preserving any
    other depth columns already present), sort by setup size, rewrite. Will
    raise PermissionError if the file is locked (e.g. open in Excel)."""
    from datetime import datetime
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    rows: Dict[str, Dict[str, str]] = {}
    existing_labels = set()
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows[r["Setup"]] = dict(r)
        # Discover what depths are already in the file.
        for row in rows.values():
            for k in row.keys():
                if k.startswith("LBR-") and k.endswith(" expl"):
                    existing_labels.add(k[len("LBR-"):-len(" expl")])

    this_label = _depth_label(depth)
    all_labels = _sort_depth_labels(existing_labels | {this_label})
    depth_cols: List[str] = []
    for label in all_labels:
        e, d = _depth_cols(label)
        depth_cols.extend([e, d])
    fields = SUMMARY_BASE_FIELDS + depth_cols + SUMMARY_TRAILING_FIELDS

    setup_key = ",".join(str(x) for x in hand_sizes)
    sps_sorted = sorted(per_sp_results.keys())

    def _expl_str(r):
        if r["se_worst"] == 0:
            return f"{r['expl']*100:+.3f}%"
        return f"{r['expl']*100:+.3f}% ± {r['se_worst']*100:.3f}pp"

    def _join_str(values, fn):
        return " | ".join(fn(values[sp]) for sp in sps_sorted)

    row = rows.get(setup_key, {})
    row["Setup"] = setup_key
    row["Sampling"] = sampling_labels[sps_sorted[0]]
    e_col, d_col = _depth_cols(this_label)
    row[e_col] = _join_str(
        {sp: per_sp_results[sp][0] for sp in sps_sorted}, _expl_str,
    )
    row[d_col] = _join_str(
        {sp: per_sp_results[sp][1] for sp in sps_sorted},
        lambda v: f"{v:.0f}",
    )
    row["Finished"] = datetime.now().strftime("%Y-%m-%d")
    rows[setup_key] = row

    def _sort_key(s):
        try:
            xs = [int(x) for x in s.split(",")]
            return (sum(xs), xs)
        except Exception:
            return (10**9, [])

    with open(SUMMARY_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for key in sorted(rows.keys(), key=_sort_key):
            w.writerow({k: rows[key].get(k, "") for k in fields})
    print(f"\nWrote {setup_key} (LBR-{this_label}) to {SUMMARY_PATH}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="LBR-based exploitability for the Blef CFR AI.")
    p.add_argument("--hand-sizes", nargs=2, type=int, required=True)
    p.add_argument(
        "--depth", type=int, default=INF_DEPTH,
        help="Number of LBR-optimised decisions per game. 0 = LBR plays CFR (sanity baseline). "
             "1 = LBR-1 (classic Lisý-Bowling local best response). "
             "Default is effectively infinity, which equals exact best response.",
    )
    p.add_argument(
        "--starting-player", type=int, default=None, choices=[0, 1],
        help="Fix starting player (0 or 1). If omitted, runs both (when hand sizes differ) or just 0 (when equal).",
    )
    p.add_argument(
        "--n-belief-samples", type=int, default=None,
        help="Sample opponent hands without replacement instead of enumerating. "
             "Default: enumerate all (which is unbiased and recommended; only sample if "
             "you genuinely need to cap compute).",
    )
    p.add_argument(
        "--n-lbr-hand-samples", type=int, default=None,
        help="Sample LBR hands without replacement instead of enumerating all C(24, lbr_size). "
             "Unbiased; adds variance.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--update-summary", action="store_true",
        help=f"Append/update this setup's row in {SUMMARY_PATH}.",
    )
    args = p.parse_args()

    hand_sizes = sorted(args.hand_sizes)
    print(f"Loading CFR strategies for {hand_sizes}...")
    t0 = time.time()
    cfr_strategy, cfr_min_bet = load_cfr_strategy(hand_sizes)
    print(f"  loaded {len(cfr_strategy)} infosets in {time.time() - t0:.1f}s; min_bet={cfr_min_bet}")

    if args.starting_player is not None:
        sps = [args.starting_player]
    elif hand_sizes[0] == hand_sizes[1]:
        sps = [0]
    else:
        sps = [0, 1]

    depth_str = "inf (= BR)" if args.depth >= INF_DEPTH else str(args.depth)
    print(f"Depth: {depth_str}")

    per_sp_results: Dict[int, Tuple[Dict[str, float], float]] = {}
    sampling_labels: Dict[int, str] = {}
    for sp in sps:
        t0 = time.time()
        result = lbr_exploitability(
            hand_sizes, sp, cfr_strategy, cfr_min_bet, args.depth,
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
