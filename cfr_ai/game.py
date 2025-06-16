from typing import List
import random
import itertools
import numpy as np
from numba import njit

BlefCards = np.arange(24, dtype=np.int64) 
rng = np.random.default_rng()


@njit
def is_all_in(a, b):
    for i in range(a.shape[0]):
        found = False
        for j in range(b.shape[0]):
            if a[i] == b[j]:
                found = True
                break
        if not found:
            return False
    return True


class Game():
    @staticmethod
    def check_finish(history: List[int]) -> bool:
        return len(history) > 0 and history[-1] == 88

    @njit
    def get_payoff(history: List[int], all_cards) -> int:
        """get payoff for player who made last bet"""
        last_bet = history[-2]
        
        card_values = all_cards // 4
        card_suits = all_cards % 4

        if last_bet < 6: # High card (e.g., last_bet=0 for card '9')
            correct = np.sum(card_values == last_bet) >= 1
        elif last_bet < 12: # Pair (e.g., last_bet=6 for pair of '9's)
            correct = np.sum(card_values == (last_bet - 6)) >= 2
        elif last_bet < 27: # Two pair
            if last_bet == 12: v1, v2 = 1, 0
            elif last_bet < 15: v1, v2 = 2, last_bet - 13
            elif last_bet < 18: v1, v2 = 3, last_bet - 15
            elif last_bet < 22: v1, v2 = 4, last_bet - 18
            else: v1, v2 = 5, last_bet - 22
            correct = (np.sum(card_values == v1) >= 2) and (np.sum(card_values == v2) >= 2)
        elif last_bet < 30: # Straights
            if last_bet == 27:
                correct = is_all_in(np.array([0, 1, 2, 3, 4]), card_values)
            elif last_bet == 28:
                correct = is_all_in(np.array([1, 2, 3, 4, 5]), card_values)
            elif last_bet == 29:
                correct = is_all_in(np.array([0, 1, 2, 3, 4, 5]), card_values)
        elif last_bet < 36: # Three of a kind
            correct = np.sum(card_values == (last_bet - 30)) >= 3
        elif last_bet < 66: # Full house
            if last_bet < 41:   v_three, v_two = 0, last_bet - 35
            elif last_bet == 41: v_three, v_two = 1, 0
            elif last_bet < 46: v_three, v_two = 1, last_bet - 40
            elif last_bet < 48: v_three, v_two = 2, last_bet - 46
            elif last_bet < 51: v_three, v_two = 2, last_bet - 45
            elif last_bet < 54: v_three, v_two = 3, last_bet - 51
            elif last_bet < 56: v_three, v_two = 3, last_bet - 50
            elif last_bet < 60: v_three, v_two = 4, last_bet - 56
            elif last_bet == 60: v_three, v_two = 4, 5
            else: v_three, v_two = 5, last_bet - 61
            correct = (np.sum(card_values == v_three) >= 3) and (np.sum(card_values == v_two) >= 2)
        elif last_bet < 70: # Flush
            correct = np.sum(card_suits == (last_bet - 66)) >= 5
        elif last_bet < 76: # Four of a kind
            correct = np.sum(card_values == (last_bet - 70)) >= 4
        elif last_bet < 88: # Straight Flushes
            suit = last_bet % 4
            if last_bet in range(76, 80): required_values = np.array([0, 1, 2, 3, 4])
            elif last_bet in range(80, 84): required_values = np.array([1, 2, 3, 4, 5])
            else: required_values = np.array([0, 1, 2, 3, 4, 5])
            required_cards = required_values * 4 + suit
            correct = is_all_in(required_cards, all_cards)
        if correct:
            return 1
        else:
            return -1
    
    @staticmethod
    def deal_cards(hand_sizes: List[int]):
        all_cards = rng.choice(BlefCards, size=sum(hand_sizes), replace=False)
        hands = []
        for i in range(0, len(hand_sizes)):
            cumsum = [0] + list(itertools.accumulate(hand_sizes))
            i_cards = [all_cards[i] for i in range(cumsum[i], cumsum[i+1])]
            i_cards.sort()
            hands.append(i_cards)
        return hands

    @staticmethod
    def hand_combinations(hand_sizes: List[int]):
        def generate(remaining_num_cards, possible_cards):
            if not remaining_num_cards:
                yield []
            else:
                player_hand_size = remaining_num_cards[0]
                for player_hand in itertools.combinations(possible_cards, player_hand_size):
                    for tail in generate(remaining_num_cards[1:], possible_cards - set(player_hand)):
                        yield [sorted(player_hand)] + tail
        return generate(hand_sizes, set(BlefCards))
