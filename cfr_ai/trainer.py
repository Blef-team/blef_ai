from typing import List, Dict
from cfr_ai.information_set import *
from cfr_ai.game import *
from cfr_ai.utils import *
import numpy as np

class Trainer():
    def __init__(self, BlefCards: List[str], Params: object):
        self.infoset_map: Dict[str, InformationSet] = {}
        self.BlefCards = BlefCards
        self.NumCards = Params.NumCards

    def get_node_value(self, hands: List[List[str]], history: List[int], reach_probability: float, active_player: int, mc_player: int, warm_up: bool, prune_feast: bool):
        if Game.check_finish(history):
            return Game.get_payoff(history, hands)
        hand = hands[active_player]

        key = make_key(hand, history, self.NumCards)
        if key not in self.infoset_map:
            self.infoset_map[key] = InformationSet(history, self.NumCards)
        info_set = self.infoset_map[key]

        counterfactual_values = np.zeros(info_set.num_actions)
        opponent = (active_player + 1) % 2

        if active_player == mc_player:
            strategy = info_set.get_strategy(reach_probability, warm_up)
            for i, action in enumerate(info_set.possible_actions):
                if info_set.regrets[i] >= -300 or prune_feast:
                    counterfactual_values[i] = -self.get_node_value(hands, history + [action], reach_probability * strategy[i], opponent, mc_player, warm_up, prune_feast)
            node_value = counterfactual_values.dot(strategy)
            if prune_feast:
                info_set.regrets = np.maximum(info_set.regrets + counterfactual_values - node_value, -310)
            else:
                to_update = info_set.regrets >= -300
                info_set.regrets[to_update] += counterfactual_values[to_update] - node_value

        else:
            strategy = info_set.get_strategy(1.0, warm_up)
            action = random.choices(info_set.possible_actions, weights=strategy, k=1)[0]
            node_value = -self.get_node_value(hands, history + [action], reach_probability, opponent, mc_player, warm_up, prune_feast)
        return node_value

    def train(self, num_iterations: int) -> int:
        util = 0
        for i in range(num_iterations):
            report_progress(i, num_iterations)
            warm_up = i < num_iterations * 0.3
            prune_feast = i % 20 == 0
            for mc_player in range(2):
                hands = Game.deal_cards(self.BlefCards, self.NumCards)
                util += self.get_node_value(hands, [], 1.0, 0, mc_player, warm_up, prune_feast)
                hands = Game.deal_cards(self.BlefCards, self.NumCards)
                util -= self.get_node_value(hands, [], 1.0, 1, mc_player, warm_up, prune_feast)
            for t in range(1, 10):
                if i == int(t * num_iterations / 10):
                    for _,(k,v) in enumerate(self.infoset_map.items()):
                        v.strategy_sum *= (t / (t + 1)) # LINEAR MCCFR
        return util / 4
