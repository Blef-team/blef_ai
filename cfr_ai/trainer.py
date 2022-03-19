from typing import List, Dict
from cfr_ai.information_set import *
from cfr_ai.game import *
import numpy as np
from tqdm import trange

class Trainer():
    def __init__(self, hand_sizes: List[int], pruning_range: List[int], penalty: float):
        self.infoset_map: Dict[str, InformationSet] = {}
        self.hand_sizes = hand_sizes
        self.nodes_touched = 0
        self.pruning_threshold = pruning_range[0]
        self.min_regret = pruning_range[1]
        self.penalty = penalty

    def get_node_value(self, hands: List[List[str]], hand_abstractions: List[str], history: List[int], reach_probability: float, active_player: int, mc_player: int, prune_feast: bool, iter: int):
        if Game.check_finish(history):
            return Game.get_payoff(history, hands)

        key = make_key(hands[active_player], hand_abstractions[active_player], history)
        if key not in self.infoset_map:
            self.infoset_map[key] = InformationSet(self.hand_sizes, history, iter)
        info_set = self.infoset_map[key]

        possible_actions = get_possible_actions(history, self.hand_sizes)
        counterfactual_values = np.zeros(len(possible_actions))
        opponent = (active_player + 1) % 2

        if active_player == mc_player:
            strategy = info_set.get_strategy(reach_probability)
            for i, action in enumerate(possible_actions):
                if info_set.regrets[i] >= self.pruning_threshold or prune_feast:
                    counterfactual_values[i] = -self.get_node_value(hands, hand_abstractions, history + [action], reach_probability * strategy[i], opponent, mc_player, prune_feast, iter)
            node_value = np.dot(counterfactual_values, strategy)
            if prune_feast:
                info_set.regrets = np.maximum(info_set.regrets + counterfactual_values - node_value, self.min_regret)
            else:
                to_update = info_set.regrets >= self.pruning_threshold
                info_set.regrets[to_update] += counterfactual_values[to_update] - node_value

        else:
            strategy = info_set.get_strategy(1.0)
            action = random.choices(possible_actions, weights=strategy, k=1)[0]
            node_value = -self.get_node_value(hands, hand_abstractions, history + [action], reach_probability, opponent, mc_player, prune_feast, iter) + self.penalty
        info_set.times_touched += 1
        info_set.last_touched = iter
        self.nodes_touched += 1
        return node_value

    def train(self, num_iterations: int):
        utils = [0, 0]
        for i in trange(num_iterations, desc = "MC iterations of Blef CFR"):
            if i == int(num_iterations * 0.3):
                for _,v in self.infoset_map.items():
                    v.strategy_sum *= 0
            for t in range(4, 10):
                if i == int(t * num_iterations / 10):
                    for _,v in self.infoset_map.items():
                        v.strategy_sum *= (t / (t + 1)) # LINEAR MCCFR
            prune_feast = i % 20 == 0
            for mc_player in range(2):
                for starting_player in range(2):
                    hands = Game.deal_cards(self.hand_sizes)
                    hand_abstractions = [get_hand_abstraction(hand, self.hand_sizes) for hand in hands]
                    utils[starting_player] += self.get_node_value(hands, hand_abstractions, [], 1.0, starting_player, mc_player, prune_feast, i)
        return utils[0] / num_iterations / 2, utils[1] / num_iterations / 2
