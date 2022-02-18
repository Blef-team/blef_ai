from typing import List, Dict
from cfr_ai.information_set import *
from cfr_ai.game import *
import numpy as np
from tqdm import trange

class Trainer():
    def __init__(self, hand_sizes: List[int]):
        self.infoset_map: Dict[str, InformationSet] = {}
        self.hand_sizes = hand_sizes

    def get_node_value(self, hands: List[List[str]], history: List[int], reach_probability: float, active_player: int, mc_player: int, warm_up: bool, prune_feast: bool):
        if Game.check_finish(history):
            return Game.get_payoff(history, hands)
        hand = hands[active_player]

        key = make_key(hand, history, self.hand_sizes)
        if key not in self.infoset_map:
            self.infoset_map[key] = InformationSet(self.hand_sizes, history)
        info_set = self.infoset_map[key]

        possible_actions = get_possible_actions(history, self.hand_sizes)
        counterfactual_values = np.zeros(len(possible_actions))
        opponent = (active_player + 1) % 2

        if active_player == mc_player:
            strategy = info_set.get_strategy(reach_probability, warm_up)
            for i, action in enumerate(possible_actions):
                if info_set.regrets[i] >= -300 or prune_feast:
                    counterfactual_values[i] = -self.get_node_value(hands, history + [action], reach_probability * strategy[i], opponent, mc_player, warm_up, prune_feast)
            node_value = np.dot(counterfactual_values, strategy)
            if prune_feast:
                info_set.regrets = np.maximum(info_set.regrets + counterfactual_values - node_value, -310)
            else:
                to_update = info_set.regrets >= -300
                info_set.regrets[to_update] += counterfactual_values[to_update] - node_value

        else:
            strategy = info_set.get_strategy(1.0, warm_up)
            action = random.choices(possible_actions, weights=strategy, k=1)[0]
            node_value = -self.get_node_value(hands, history + [action], reach_probability, opponent, mc_player, warm_up, prune_feast)
        return node_value

    def train(self, num_iterations: int):
        util0, util1 = 0, 0
        for i in trange(num_iterations, desc = "MC iterations of Blef CFR"):
            warm_up = i < num_iterations * 0.3
            prune_feast = i % 20 == 0
            for mc_player in range(2):
                hands = Game.deal_cards(self.hand_sizes)
                util0 += self.get_node_value(hands, [], 1.0, 0, mc_player, warm_up, prune_feast)
                hands = Game.deal_cards(self.hand_sizes)
                util1 += self.get_node_value(hands, [], 1.0, 1, mc_player, warm_up, prune_feast)
            for t in range(1, 10):
                if i == int(t * num_iterations / 10):
                    for _,v in self.infoset_map.items():
                        v.strategy_sum *= (t / (t + 1)) # LINEAR MCCFR
        return util0 / 2 / num_iterations, util1 / 2 / num_iterations
