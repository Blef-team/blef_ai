"""Outcome-sampling MCCFR for Blef.

Status: implementation is canonical (matches OpenSpiel's outcome_sampling_mccfr
and Lanctot 2009) but does NOT converge competitively on Blef setups in
reasonable iteration budgets. Kept here as a starting point for any future
variance-reduced variant. See cfr_ai/analysis/sampling_comparison.py and the
findings in SESSION_NOTES.md for the measured gap vs external sampling.

Each iteration samples a single trajectory through the game tree (instead of
enumerating all of the traverser's actions at each decision node as external
sampling does), then updates regrets along that path using importance-weighted
single-sample estimators.

The standard OS regret update at traverser infoset I (sampled action a*) is:

    util  = u_traverser(terminal) / sample_reach_at_terminal
    Δr(a*)   = util · (1 − σ(a*)) · opp_reach · tail
    Δr(a≠a*) = util · (−σ(a*))    · opp_reach · tail

where opp_reach = π_-i^σ(to I), tail = π_i^σ(below I in trajectory).
Strategy averages are σ-reach-weighted for both players.

Why this struggles on Blef:
  • 88-action infosets + ε-on-policy mixing → σ̃(a) ≈ ε/n ≈ 0.007 for any
    non-σ-concentrated action. When such an action is sampled, sample_reach
    along that subtree includes that 0.007 factor and util = u/sample_reach
    blows up to ≈ ±150.
  • Most bets in Blef are wrong (the sampled hand-claim doesn't exist), so
    these huge-magnitude updates land negative ~70-95% of the time. The
    sampled action's regret drops by ~−150 each such visit; non-sampled
    actions only get the small ±σ(a*)·util·opp_reach·tail correction.
  • Across a few hundred visits to an infoset, this drives almost every
    action's cumulative regret strongly negative, σ collapses to the
    last-action default, and strategy_sum is poisoned toward action 87.

To make OS competitive on Blef would need variance reduction (baseline
subtraction / VR-MCCFR / MIX-MCCFR) or 10-100× more iterations. Either is
a larger effort than producing a tuned ES improvement.

Shares the InformationSet store with cfr_ai.trainer so the result is
consumed by cfr_ai/lbr.py and saving code in cfr_ai/training.py without
changes.
"""
from typing import Any, Dict, List, Tuple
import random
import time

import numpy as np
from tqdm import trange, tqdm

from cfr_ai.information_set import (
    InformationSet, make_key, get_hand_abstraction,
)
from cfr_ai.game import Game


class OutcomeSamplingTrainer:
    def __init__(
        self,
        hand_sizes: List[int],
        min_bet: int,
        log_points: int,
        exploration: float = 0.6,
        regret_clip: bool = False,
        dcfr_alpha: float = None,
        strategy_weighting: str = "reach",
    ):
        """
        exploration: epsilon for epsilon-on-policy sampling at traverser nodes.
        regret_clip: if True, apply max(0, .) to cumulative regrets after each
            update (CFR+-style on top of OS). Addresses the random-walk-with-
            negative-drift-reflected-at-0 bias in plain OS.
        dcfr_alpha: positive-regret per-iter discount exponent (DCFR-style).
            None disables. Typical values: 0.5, 1.0, 1.5.
        strategy_weighting: "reach" (default) adds (my_reach * sigma) to
            strategy_sum at traverser visits; "linear" adds ((iter+1) *
            my_reach * sigma), emphasising late iterations.
        """
        self.infoset_map: Dict[str, InformationSet] = {}
        self.hand_sizes = hand_sizes
        self.min_bet = min_bet
        self.log_points = log_points
        self.exploration = exploration
        self.regret_clip = regret_clip
        self.dcfr_alpha = dcfr_alpha
        if strategy_weighting not in ("reach", "linear"):
            raise ValueError(f"strategy_weighting must be 'reach' or 'linear', got {strategy_weighting!r}")
        self.strategy_weighting = strategy_weighting
        self.nodes_touched = 0
        self._log_alpha_cum: np.ndarray = None  # set in train() when dcfr_alpha is given

    def _walk(
        self,
        hands: List[np.ndarray],
        hand_abstractions: List[List[str]],
        history: List[int],
        active: int,
        traverser: int,
        my_reach: float,
        opp_reach: float,
        sample_reach: float,
        existence_array: np.ndarray,
        iter_num: int,
    ) -> Tuple[float, float, float]:
        """Single-trajectory walk; returns (util, tail, raw_u_traverser).

        util         = u_traverser(terminal) / sample_reach_at_terminal.
                       Constant along a trajectory; propagated up unchanged.
        tail         = π_i^σ(below current node, in trajectory). 1.0 at the
                       terminal; multiplied by σ_i(sampled action) when
                       returning through a traverser's node.
        raw_u        = unscaled terminal payoff; used only for the utility log.
        """
        if Game.check_finish(history):
            u_active = 1.0 if existence_array[history[-2]] else -1.0
            u_traverser = u_active if active == traverser else -u_active
            return u_traverser / sample_reach, 1.0, u_traverser

        key = make_key(hands[active], hand_abstractions[active], history, self.min_bet)
        if key not in self.infoset_map:
            self.infoset_map[key] = InformationSet(history, iter_num, self.min_bet)
        info_set = self.infoset_map[key]

        actions = info_set.possible_actions
        n = len(actions)
        sigma = info_set.get_strategy(0.0)  # don't touch strategy_sum; we do it below

        if active == traverser:
            sigma_tilde = (1.0 - self.exploration) * sigma + self.exploration / n
        else:
            sigma_tilde = sigma

        a_idx = random.choices(range(n), weights=sigma_tilde, k=1)[0]
        action = actions[a_idx]

        opp = (active + 1) % 2
        if active == traverser:
            new_my_reach = my_reach * sigma[a_idx]
            new_opp_reach = opp_reach
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * sigma[a_idx]
        new_sample_reach = sample_reach * sigma_tilde[a_idx]

        util, tail, raw_u = self._walk(
            hands, hand_abstractions, history + [action], opp, traverser,
            new_my_reach, new_opp_reach, new_sample_reach,
            existence_array, iter_num,
        )

        if active == traverser:
            # DCFR-style α-discount on existing positive regrets, applied
            # lazily based on time since last visit.
            if self.dcfr_alpha is not None:
                last_t = info_set.last_touched
                if last_t < iter_num:
                    delta_log = self._log_alpha_cum[iter_num] - (
                        self._log_alpha_cum[last_t] if last_t > 0 else 0.0
                    )
                    pos_factor = float(np.exp(delta_log))
                    pos_mask = info_set.regrets > 0
                    if pos_mask.any():
                        info_set.regrets[pos_mask] *= pos_factor

            scale = opp_reach * tail
            coef_sampled = (1.0 - sigma[a_idx]) * util * scale
            coef_unsampled = -sigma[a_idx] * util * scale
            for i in range(n):
                info_set.regrets[i] += coef_sampled if i == a_idx else coef_unsampled

            # CFR+ regret clipping (clip cumulative regrets at 0).
            if self.regret_clip:
                np.maximum(info_set.regrets, 0.0, out=info_set.regrets)

            # Strategy averaging weight.
            avg_weight = my_reach
            if self.strategy_weighting == "linear":
                avg_weight = my_reach * (iter_num + 1)
            info_set.strategy_sum += (avg_weight * sigma).astype(np.float32)
            new_tail = tail * sigma[a_idx]
        else:
            avg_weight = opp_reach
            if self.strategy_weighting == "linear":
                avg_weight = opp_reach * (iter_num + 1)
            info_set.strategy_sum += (avg_weight * sigma).astype(np.float32)
            new_tail = tail

        info_set.times_touched += 1
        info_set.last_touched = iter_num
        self.nodes_touched += 1

        return util, new_tail, raw_u

    def train(
        self,
        num_iterations: int,
        snapshot_callback=None,
        snapshot_every_log_points: int = 1,
    ) -> Tuple[float, float, Dict[str, Any]]:
        utils = [0.0, 0.0]
        last_utils = [0.0, 0.0]
        utility_log: Dict[str, Any] = {}
        log_point_idx = 0

        train_start = time.time()
        last_log_time = train_start
        last_log_iter = 0
        print(f"[train_os] hand_sizes={self.hand_sizes} target_iters={num_iterations:,} "
              f"epsilon={self.exploration} regret_clip={self.regret_clip} "
              f"dcfr_alpha={self.dcfr_alpha} strategy_weighting={self.strategy_weighting}",
              flush=True)

        if self.dcfr_alpha is not None:
            iters_arr = np.arange(1, num_iterations + 1, dtype=np.float64)
            alpha_factor = iters_arr ** self.dcfr_alpha / (iters_arr ** self.dcfr_alpha + 1.0)
            self._log_alpha_cum = np.cumsum(np.log(alpha_factor))

        for i in trange(num_iterations, desc="Training (OS)"):
            if i == int(num_iterations * 0.3):
                for v in self.infoset_map.values():
                    v.strategy_sum *= 0.02
            for t in range(4, 10):
                if i == int(t * num_iterations / 10):
                    for v in self.infoset_map.values():
                        v.strategy_sum *= (t / (t + 1))

            traverser = (i // 2) % 2
            starting_player = i % 2
            hands = Game.deal_cards(self.hand_sizes)
            existence_array = Game.precompute_set_existence(hands)
            hand_abstractions = [
                get_hand_abstraction(h, self.hand_sizes) for h in hands
            ]

            _, _, raw_u_traverser = self._walk(
                hands, hand_abstractions, [], starting_player, traverser,
                1.0, 1.0, 1.0, existence_array, i,
            )
            u_starting = raw_u_traverser if starting_player == traverser else -raw_u_traverser
            utils[starting_player] += u_starting

            if (self.log_points > 0
                    and (i + 1) % (num_iterations // self.log_points) == 0):
                now = time.time()
                chunk_iters = (i + 1) - last_log_iter
                chunk_secs = now - last_log_time
                overall_secs = now - train_start
                overall_rate = (i + 1) / overall_secs if overall_secs > 0 else 0.0
                chunk_rate = chunk_iters / chunk_secs if chunk_secs > 0 else 0.0
                remaining_iters = num_iterations - (i + 1)
                eta_secs = remaining_iters / overall_rate if overall_rate > 0 else 0.0
                eta_hms = time.strftime('%H:%M:%S', time.gmtime(eta_secs))
                util0_chunk = (utils[0] - last_utils[0]) / num_iterations * self.log_points * 2
                util1_chunk = (utils[1] - last_utils[1]) / num_iterations * self.log_points * 2
                print(
                    f"[train_os] iter {i + 1:>10,}/{num_iterations:,} "
                    f"({100*(i+1)/num_iterations:>3.0f}%) | "
                    f"chunk {chunk_rate:>5.0f} it/s | "
                    f"overall {overall_rate:>5.0f} it/s | "
                    f"ETA {eta_hms} | "
                    f"P0={util0_chunk:+.4f} P1={util1_chunk:+.4f}",
                    flush=True,
                )
                utility_log[f"P0 Utility at Iter {i + 1}"] = f"{util0_chunk:.4f}"
                utility_log[f"P1 Utility at Iter {i + 1}"] = f"{util1_chunk:.4f}"
                last_utils = list(utils)
                last_log_time = now
                last_log_iter = i + 1
                log_point_idx += 1
                if (snapshot_callback is not None
                        and log_point_idx % snapshot_every_log_points == 0):
                    snapshot_callback(i + 1, self.infoset_map)

        return (
            utils[0] * 2 / num_iterations,
            utils[1] * 2 / num_iterations,
            utility_log,
        )
