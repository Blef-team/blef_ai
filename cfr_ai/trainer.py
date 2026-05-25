from typing import List, Dict, Any, Tuple
from cfr_ai.information_set import *
from cfr_ai.game import *
import numpy as np
import random
import time
from tqdm import trange, tqdm

class Trainer():
    def __init__(self, hand_sizes: List[int], min_bet: int, pruning_range: List[int], penalty: float, log_points: int, algorithm: str = 'es', dcfr_alpha: float = 1.5, dcfr_gamma: float = 1.0):
        """
        algorithm:
          'es'       — external sampling (current default) with the existing
                       linear-ish discount staircase + pruning combo.
          'cfr_plus' — CFR+ regret matching with linear strategy averaging:
                       clip cumulative regrets at 0, weight each iter's σ
                       contribution by (iter+1), no pruning, no staircase.
          'dcfr'     — CFR+ plus a per-iter α-discount on positive regrets,
                       addressing the MCCFR-specific failure where a lucky
                       sample spikes regret on a bad action and contaminates
                       strategy_sum. β=0 (clip negatives at 0) is the same
                       as cfr_plus. γ controls strategy-averaging weight power
                       (1 = linear, 2 = canonical DCFR; defaulting to 1 to
                       stay safely inside float32 precision for the strategy
                       sum at large iteration counts).

        dcfr_alpha: positive-regret discount factor per iter is t^α/(t^α+1).
        dcfr_gamma: strategy_sum weight per iter is (iter+1)^γ.
        """
        self.infoset_map: Dict[str, InformationSet] = {}
        self.hand_sizes = hand_sizes
        self.nodes_touched = 0
        self.pruning_threshold = pruning_range[0]
        self.min_regret = pruning_range[1]
        self.penalty = penalty
        self.log_points = log_points
        self.min_bet = min_bet
        if algorithm not in ('es', 'cfr_plus', 'dcfr'):
            raise ValueError(f"algorithm must be 'es', 'cfr_plus' or 'dcfr', got {algorithm!r}")
        self.algorithm = algorithm
        self.dcfr_alpha = dcfr_alpha
        self.dcfr_gamma = dcfr_gamma
        # log-cumulative α-discount, set when train() is called and num_iter known.
        self._log_alpha_cum: np.ndarray = None

    def get_node_value(self, hands: List[np.ndarray], hand_abstractions: List[str], history: List[int], reach_probability: float, active_player: int, traverser: int, prune_feast: bool, existence_array: np.ndarray, iter: int) -> float:
        if Game.check_finish(history):
            return 1 if existence_array[history[-2]] else -1
        
        key = make_key(hands[active_player], hand_abstractions[active_player], history, self.min_bet)
        if key not in self.infoset_map:
            self.infoset_map[key] = InformationSet(history, iter, self.min_bet)
        info_set = self.infoset_map[key]

        if info_set.last_touched == iter:
            return info_set.temporary_value
        else:
            possible_actions = info_set.possible_actions
            counterfactual_values = np.zeros(len(possible_actions))
            opponent = (active_player + 1) % 2

            if active_player == traverser:
                if self.algorithm == 'cfr_plus':
                    # CFR+: explore every action (no pruning); after the regret
                    # update, clip the cumulative regrets at 0. Strategy
                    # averaging uses the linear (γ=1) weight (iter+1) so that
                    # the averaged policy emphasises later iterations.
                    strategy = info_set.get_strategy(reach_probability * (iter + 1))
                    for i, action in enumerate(possible_actions):
                        counterfactual_values[i] = -self.get_node_value(hands, hand_abstractions, history + [action], reach_probability * strategy[i], opponent, traverser, prune_feast, existence_array, iter)
                    node_value = np.dot(counterfactual_values, strategy)
                    info_set.regrets = np.maximum(info_set.regrets + counterfactual_values - node_value, 0.0)
                elif self.algorithm == 'dcfr':
                    # DCFR: CFR+ plus α-discount on positive regrets.
                    # Apply lazy cumulative discount from last_touched up to
                    # the current iter (covers iters where this infoset wasn't
                    # visited). β=0 ⇒ negatives were already clipped to 0 at
                    # the previous visit (and any new negative drift is also
                    # clipped below); positives get t^α/(t^α+1) per iter.
                    # Canonical DCFR order is: discount at iter t, then add
                    # instantaneous regret of iter t. With 0-indexed `iter`
                    # (= 1-indexed t-1) and 0-indexed `last_t`, the cumulative
                    # discount we owe covers 1-indexed iters (last_t+2)..(iter+1),
                    # i.e. log_alpha_cum[iter] - log_alpha_cum[last_t].
                    # (First visit to an infoset starts with regrets all zero,
                    # so discount is a no-op even if we apply it.)
                    last_t = info_set.last_touched
                    if last_t < iter:
                        delta_log = self._log_alpha_cum[iter] - (
                            self._log_alpha_cum[last_t] if last_t > 0 else 0.0
                        )
                        pos_factor = float(np.exp(delta_log))
                        pos_mask = info_set.regrets > 0
                        if pos_mask.any():
                            info_set.regrets[pos_mask] *= pos_factor
                    weight = reach_probability * (iter + 1) ** self.dcfr_gamma
                    strategy = info_set.get_strategy(weight)
                    for i, action in enumerate(possible_actions):
                        counterfactual_values[i] = -self.get_node_value(hands, hand_abstractions, history + [action], reach_probability * strategy[i], opponent, traverser, prune_feast, existence_array, iter)
                    node_value = np.dot(counterfactual_values, strategy)
                    info_set.regrets = np.maximum(info_set.regrets + counterfactual_values - node_value, 0.0)
                else:
                    strategy = info_set.get_strategy(reach_probability)
                    for i, action in enumerate(possible_actions):
                        if info_set.regrets[i] >= self.pruning_threshold or prune_feast:
                            counterfactual_values[i] = -self.get_node_value(hands, hand_abstractions, history + [action], reach_probability * strategy[i], opponent, traverser, prune_feast, existence_array, iter)
                    node_value = np.dot(counterfactual_values, strategy)
                    if prune_feast:
                        info_set.regrets = np.maximum(info_set.regrets + counterfactual_values - node_value, self.min_regret)
                    else:
                        to_update = info_set.regrets >= self.pruning_threshold
                        info_set.regrets[to_update] += counterfactual_values[to_update] - node_value

            else:
                strategy = info_set.get_strategy(0.0)
                action = random.choices(possible_actions, weights=strategy, k=1)[0]
                node_value = -self.get_node_value(hands, hand_abstractions, history + [action], reach_probability, opponent, traverser, prune_feast, existence_array, iter) + self.penalty
            info_set.times_touched += 1
            info_set.last_touched = iter
            info_set.temporary_value = node_value
            self.nodes_touched += 1
            return node_value

    def train(self, num_iterations: int, snapshot_callback=None, snapshot_every_log_points: int = 1) -> Tuple[float, float, Dict[str, Any]]:
        """
        snapshot_callback: optional fn(iter_num, infoset_map) called at every
            `snapshot_every_log_points`-th utility log point. Lets the caller
            compute exploitability mid-training.
        """
        utils = [0.0, 0.0]
        last_utils = [0.0, 0.0]
        utility_log: Dict[str, Any] = {}
        log_point_idx = 0

        # Live progress: at every log point print elapsed/rate/ETA with
        # flush=True so the .output file of a backgrounded run shows real
        # progress instead of staying empty until the process exits.
        train_start = time.time()
        last_log_time = train_start
        last_log_iter = 0
        print(f"[train] algorithm={self.algorithm} hand_sizes={self.hand_sizes} "
              f"target_iters={num_iterations:,}", flush=True)

        if self.algorithm == 'dcfr':
            # Precompute log-cumulative α-discount for lazy per-visit application.
            # alpha_factor at iter t (1-indexed) = t^α / (t^α + 1).
            # _log_alpha_cum[t-1] = sum_{k=1..t} log(alpha_factor_k).
            iters_arr = np.arange(1, num_iterations + 1, dtype=np.float64)
            alpha_factor = iters_arr ** self.dcfr_alpha / (iters_arr ** self.dcfr_alpha + 1.0)
            self._log_alpha_cum = np.cumsum(np.log(alpha_factor))

        for i in trange(num_iterations, desc = "Training"):
            if self.algorithm == 'es':
                # CFR+ uses smooth linear weighting via the (iter+1) factor on
                # strategy_sum updates; the staircase discount below is the
                # current ES algorithm's coarser substitute for that and must
                # not be combined.
                if i == int(num_iterations * 0.3):
                    for _,v in self.infoset_map.items():
                        v.strategy_sum *= 0.02
                for t in range(4, 10):
                    if i == int(t * num_iterations / 10):
                        for _,v in self.infoset_map.items():
                            v.strategy_sum *= (t / (t + 1)) # LINEAR MCCFR
            prune_feast = int(i/4) % 20 == 0
            traverser = int(i/2) % 2
            starting_player = i % 2
            hands = Game.deal_cards(self.hand_sizes)
            existence_array = Game.precompute_set_existence(hands)
            hand_abstractions = [get_hand_abstraction(hand, self.hand_sizes) for hand in hands]
            utils[starting_player] += self.get_node_value(hands, hand_abstractions, [], 1.0, starting_player, traverser, prune_feast, existence_array, i)
            if (self.log_points > 0 and (i + 1) % (num_iterations // self.log_points) == 0):
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
                    f"[train] iter {i + 1:>10,}/{num_iterations:,} "
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
                if snapshot_callback is not None and log_point_idx % snapshot_every_log_points == 0:
                    snapshot_callback(i + 1, self.infoset_map)
        return utils[0] * 2 / num_iterations, utils[1] * 2 / num_iterations, utility_log
