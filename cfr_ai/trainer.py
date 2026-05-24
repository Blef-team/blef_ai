from typing import List, Dict, Any, Tuple
from cfr_ai.information_set import *
from cfr_ai.game import *
import numpy as np
import random
from tqdm import trange, tqdm

class Trainer():
    def __init__(self, hand_sizes: List[int], min_bet: int, pruning_range: List[int], penalty: float, log_points: int):
        self.infoset_map: Dict[str, InformationSet] = {}
        self.hand_sizes = hand_sizes
        self.nodes_touched = 0
        self.pruning_threshold = pruning_range[0]
        self.min_regret = pruning_range[1]
        self.penalty = penalty
        self.log_points = log_points
        self.min_bet = min_bet

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
        for i in trange(num_iterations, desc = "Training"):
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
                util0_chunk = (utils[0] - last_utils[0]) / num_iterations * self.log_points * 2
                util1_chunk = (utils[1] - last_utils[1]) / num_iterations * self.log_points * 2
                tqdm.write(f"Iter {i + 1}: P0 Util: {util0_chunk:.4f}, P1 Util: {util1_chunk:.4f}")
                utility_log[f"P0 Utility at Iter {i + 1}"] = f"{util0_chunk:.4f}"
                utility_log[f"P1 Utility at Iter {i + 1}"] = f"{util1_chunk:.4f}"
                last_utils = list(utils)
                log_point_idx += 1
                if snapshot_callback is not None and log_point_idx % snapshot_every_log_points == 0:
                    snapshot_callback(i + 1, self.infoset_map)
        return utils[0] * 2 / num_iterations, utils[1] * 2 / num_iterations, utility_log
