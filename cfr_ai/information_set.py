from typing import List
import numpy as np

history_codes = np.genfromtxt('cfr_ai/history.csv', delimiter=',', dtype='|U5', skip_header=0)


def get_possible_actions(history: List[int], hand_sizes: List[int]):
    if (len(history) == 0):
        return [a for a in range(88)]
    else: 
        last_action = history[-1]
        return [a for a in range(89) if a > last_action]


def get_hand_abstraction(hand: List[str], hand_sizes: List[int]) -> str:
    out = ''
    # Rounds 1-5: get values
    if (sum(hand_sizes) <= 8):
        hand.sort()
        for x in hand:
            out += x[0]
    # Rounds 6-7: Get the strongest four of a kind or flush, if any. 
    # Otherwise, get all values
    else:
        strengths = np.arange(10)
        for x in hand:
            strengths[int(x[0]) + 4] += 10
            strengths[int(x[1])] += 10
        if np.max(strengths) >= 44:
            out += str(np.max(strengths))
        else:
            out += ' '.join([str(x) for x in strengths[4:]])
    return out


def make_key(hand: List[str], hand_abstraction: str, history: List[int]) -> str:
    key = str(len(hand))
    
    # History abstraction
    if len(history) == 0:
        key += '-88-'
    else:
        key += '-' + str(history[-1]) + '-'
        if len(history) > 1:
            key += history_codes[history[-1], history[-2]] + '-'
            if len(history) > 2:
                key += history_codes[history[-1], history[-3]] + '-'
    
    # Hand abstraction
    key += hand_abstraction

    return key


def make_full_key(my_cards: List[str], history: List[int], hand_sizes: List[int]) -> str:
    my_cards.sort()
    return str(my_cards) + str(history)


class InformationSet():
    def __init__(self, hand_sizes: List[int], history: List[int], iter: int):
        possible_actions = get_possible_actions(history, hand_sizes)
        self.regrets = np.zeros(len(possible_actions))
        self.strategy_sum = np.zeros(len(possible_actions))
        self.times_touched = 0
        self.first_touched = iter
        self.last_touched = 0

    def get_strategy(self, reach_probability: float) -> np.array:
        if any(self.regrets > 0):
            strategy = np.maximum(0, self.regrets)
            strategy /= sum(strategy)
        else:
            strategy = np.array([0.0] * (len(self.regrets) - 1) + [1.0])

        self.strategy_sum += reach_probability * strategy
        return strategy

    def get_final_strategy(self) -> np.array:
        if any(self.strategy_sum):
            return self.strategy_sum / sum(self.strategy_sum)
        else:
            return np.array([0.0] * (len(self.strategy_sum) - 1) + [1.0])
