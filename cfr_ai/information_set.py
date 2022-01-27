from typing import List
import numpy as np

relevant_actions_list = []
for i in range(25):
    if i <= 1:
        relevant_actions = []
    elif i <= 2:
        relevant_actions = [a for a in range(12)] + [88]
    elif i <= 3:
        relevant_actions = [a for a in range(12)] + [a for a in range(30, 36)] + [88]
    elif i <= 4:
        relevant_actions = [a for a in range(27)] + [a for a in range(30, 36)] + [88] # Skipped four of a kind
    elif i <= 5:
        relevant_actions = [a for a in range(29)] + [a for a in range(30, 66)] + [a for a in range(70, 76)] + [88] # Skipped flush and straight flush
    elif i <= 7:
        relevant_actions = [a for a in range(66)] + [a for a in range(70, 76)] + [88] # Skipped flush and straight flush
    elif i <= 13:
        relevant_actions = [a for a in range(0, 89)] 
    else:
        relevant_actions = [a for a in range(27, 89)] # Skipped high card, pair, two pair
    relevant_actions_list.append(relevant_actions)


def get_possible_actions(history: List[int], NumCards: List[int]):
    relevant_actions = relevant_actions_list[sum(NumCards)]
    if (len(history) == 0):
        return [a for a in relevant_actions if a != 88]
    else: 
        last_action = history[-1]
        return [a for a in relevant_actions if a > last_action]


def make_key(my_cards: List[str], history: List[int], NumCards: List[int]) -> str:
    # Abstraction trick: cluster hands
    my_cards.sort()
    key = ''
    if (sum(NumCards) <= 8):
        for x in my_cards:
            key += x[0]
    else:
        for x in my_cards:
            key += x[0]

    Actions = relevant_actions_list[sum(NumCards)]

    # Abstraction trick: only consider three bets. Also, 'irrelevant' bets are clustered together
    abstracted_history = [min([a for a in Actions if a > x]) - 1 for x in history[-3:]]
    for h in abstracted_history:
        key += ' ' + str(h) 
    return key


def make_full_key(my_cards: List[str], history: List[int], NumCards: List[int]) -> str:
    my_cards.sort()
    return str(my_cards) + str(history)


class InformationSet():
    def __init__(self, NumCards: List[int], history: List[int]):
        possible_actions = get_possible_actions(history, NumCards)
        self.regrets = np.zeros(len(possible_actions))
        self.strategy_sum = np.zeros(len(possible_actions), dtype=np.float32)

    def get_strategy(self, reach_probability: float, warm_up: bool = False) -> np.array:
        if any(self.regrets > 0):
            strategy = np.maximum(0, self.regrets)
            strategy /= sum(strategy)
        else:
            strategy = np.array([0.0] * (len(self.regrets) - 1) + [1.0])

        if not warm_up: # LITERATURE TRICK: DISCOUNTING STRATEGIES FROM FIRST X% of iterations
            self.strategy_sum += reach_probability * strategy
        return strategy

    def get_final_strategy(self) -> np.array:
        if any(self.strategy_sum):
            return self.strategy_sum / sum(self.strategy_sum)
        else:
            return np.array([0.0] * (len(self.strategy_sum) - 1) + [1.0])
