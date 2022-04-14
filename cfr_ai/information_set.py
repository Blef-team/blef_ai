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
    # Rounds 1-5: get values
    if (sum(hand_sizes) <= 6):
        values = ''
        hand.sort()
        for x in hand:
            values += x[0]
        out = [values] * 89
    # Rounds 6-21: use main abstraction
    else:
        # Make counts of suits and values
        counts = np.zeros(10, dtype=int)
        for x in hand:
            counts[int(x[0]) + 4] += 1
            counts[int(x[1])] += 1
        # Sort the counts by opinionated strength (N of a value is better than N+1 of a suit)
        strengths = np.arange(10, dtype=int)
        strengths[0:4] -= 10
        for i in range(10):
            strengths[i] += 10 * counts[i]
        # Keep the top 1 strength handy
        top_strength = np.max(strengths)
        # Make an augmented version of suit strengths, with nines and aces, for use in straight flushes
        sf_strengths = strengths[0:4].copy().astype(float)
        for x in hand:
            if x[0] == '0':
                sf_strengths[int(x[1])] += 0.1
            if x[0] == '5':
                sf_strengths[int(x[1])] += 0.2
        augmented_strengths = np.append(sf_strengths, strengths[4:10])
        # Pre-straight:
        ## If there's four of a kind or flush on hand, get the top strength
        ## If there's a three of a kind on hand, get the top 2 value strengths
        ## Else, get all card values
        if top_strength >= 40:
            pre_straight_abstraction = str(top_strength)
        elif np.max(strengths[4:10]) >= 34:
            pre_straight_abstraction = str(sorted(strengths[4:10], reverse=True)[0]) + ' ' + str(sorted(strengths[4:10], reverse=True)[1])
        else:
            values = ''
            hand.sort()
            for x in hand:
                values += x[0]
            pre_straight_abstraction = values
        out = [pre_straight_abstraction] * 27
        # Straight to full: if there's four of a kind or flush, report just it
        if top_strength >= 40:
            out += [str(top_strength)] * 39
        else:
            ## Else for straights: check which values we have and get top strength
            straight_part = str(min(counts[4], 1)) + str(sum([min(x, 1) for x in counts[5:9]])) + str(min(counts[9], 1)) + ' ' + str(top_strength)
            out += [straight_part] * 3
            ## Else for three of a kind: check how many we have of that value and get top strength
            for i in range(4, 10):
                out += [str(counts[i]) + ' ' + str(top_strength)]
            ## Else for full house: check how many we have of the two values each and get top strength
            first_value = 0
            second_value = 0
            for i in range(30):
                second_value += 1
                if second_value == 6:
                    first_value += 1
                    second_value = 0
                if second_value == first_value:
                    second_value += 1
                out += [str(counts[4 + first_value]) + str(counts[4 + second_value]) + ' ' + str(np.max(np.delete(strengths, [4 + first_value, 4 + second_value], 0)))]
        # Flush: check how many we have of that suit and get top (augmented) strength
        for i in range(4):
            out += [str(counts[i]) + ' ' + str(np.max(augmented_strengths))]
        # Four of a kind: check how many we have of that value and get top (augmented) strength, skipping already irrelevant ones
        temp_strengths = augmented_strengths.copy()
        for i in range(4, 10):
            temp_strengths = np.delete(temp_strengths, 4, 0)
            out += [str(counts[i]) + ' ' + str(np.max(temp_strengths))]
        # Small / big straight flush: get augmented information about the suit being bet on and our strongest suit
        for i in range(2):
            for j in range(4):
                out += [str(sf_strengths[j]) + ' ' + str(np.max(np.delete(sf_strengths, j, 0)))]
        # Great straight flush: get augmented information about the suit being bet on and our strongest suit, skipping already irrelevant ones
        temp_strengths = sf_strengths
        for j in range(3):
            temp_strengths = temp_strengths[1:]
            out += [str(sf_strengths[j]) + ' ' + str(np.max(temp_strengths))]
        # Great straight flush spades: ignore the hand
        out += ['X']
        # Beginning of round (at index 88): same as pre-straight
        out += [pre_straight_abstraction]
    return out


def make_key(hand: List[str], hand_abstractions: str, history: List[int]) -> str:
    key = str(len(hand))
    
    # History abstraction
    last_bet = 88 if len(history) == 0 else history[-1]
    key += '-' + str(last_bet) + '-'
    if len(history) > 1:
        key += history_codes[last_bet, history[-2]] + '-'
        if len(history) > 2:
            key += history_codes[last_bet, history[-3]] + '-'
    
    # Hand abstraction
    key += hand_abstractions[last_bet]

    return key


def make_full_key(my_cards: List[str], history: List[int]) -> str:
    my_cards.sort()
    return str(my_cards) + str(history)


class InformationSet():
    def __init__(self, hand_sizes: List[int], history: List[int], iter: int):
        possible_actions = get_possible_actions(history, hand_sizes)
        self.regrets = np.zeros(len(possible_actions))
        self.strategy_sum = np.zeros(len(possible_actions), dtype=np.float32)
        self.times_touched = 0
        self.first_touched = iter
        self.last_touched = 0
        self.temporary_value = 0.0

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
