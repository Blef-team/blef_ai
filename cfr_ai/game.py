from typing import List, Generator
import itertools
import numpy as np

BlefCards = np.arange(24, dtype=np.int64) 
rng = np.random.default_rng()

class Game():
    @staticmethod
    def check_finish(history: List[int]) -> bool:
        return len(history) > 0 and history[-1] == 88

    @staticmethod
    def precompute_set_existence(hands: List[List[int]]) -> np.ndarray:
        """
        Calculates the outcome for all 88 possible bets at once for a given deal.
        Returns a boolean numpy array of size 88.
        """
        all_cards = np.concatenate(hands)
        card_values = all_cards // 4
        card_suits = all_cards % 4

        existence_array = np.zeros(88, dtype=np.bool_)

        for bet in range(88):
            if bet < 6: # High card (e.g., last_bet=0 for card '9')
                correct = np.sum(card_values == bet) >= 1
            elif bet < 12: # Pair (e.g., last_bet=6 for pair of '9's)
                correct = np.sum(card_values == (bet - 6)) >= 2
            elif bet < 27: # Two pair
                if bet == 12: v1, v2 = 1, 0
                elif bet < 15: v1, v2 = 2, bet - 13
                elif bet < 18: v1, v2 = 3, bet - 15
                elif bet < 22: v1, v2 = 4, bet - 18
                else: v1, v2 = 5, bet - 22
                correct = (np.sum(card_values == v1) >= 2) and (np.sum(card_values == v2) >= 2)
            elif bet < 30: # Straights
                if bet == 27:
                    correct = np.all(np.isin(np.array([0, 1, 2, 3, 4]), card_values))
                elif bet == 28:
                    correct = np.all(np.isin(np.array([1, 2, 3, 4, 5]), card_values))
                elif bet == 29:
                    correct = np.all(np.isin(np.array([0, 1, 2, 3, 4, 5]), card_values))
            elif bet < 36: # Three of a kind
                correct = np.sum(card_values == (bet - 30)) >= 3
            elif bet < 66: # Full house
                if bet < 41:   v_three, v_two = 0, bet - 35
                elif bet == 41: v_three, v_two = 1, 0
                elif bet < 46: v_three, v_two = 1, bet - 40
                elif bet < 48: v_three, v_two = 2, bet - 46
                elif bet < 51: v_three, v_two = 2, bet - 45
                elif bet < 54: v_three, v_two = 3, bet - 51
                elif bet < 56: v_three, v_two = 3, bet - 50
                elif bet < 60: v_three, v_two = 4, bet - 56
                elif bet == 60: v_three, v_two = 4, 5
                else: v_three, v_two = 5, bet - 61
                correct = (np.sum(card_values == v_three) >= 3) and (np.sum(card_values == v_two) >= 2)
            elif bet < 70: # Flush
                correct = np.sum(card_suits == (bet - 66)) >= 5
            elif bet < 76: # Four of a kind
                correct = np.sum(card_values == (bet - 70)) >= 4
            elif bet < 88: # Straight Flushes
                suit = bet % 4
                if bet in range(76, 80): required_values = np.array([0, 1, 2, 3, 4])
                elif bet in range(80, 84): required_values = np.array([1, 2, 3, 4, 5])
                else: required_values = np.array([0, 1, 2, 3, 4, 5])
                required_cards = required_values * 4 + suit
                correct = np.all(np.isin(required_cards, all_cards))
            existence_array[bet] = correct
        return existence_array
    
    @staticmethod
    def deal_cards(hand_sizes: List[int]) -> List[List[int]]:
        all_cards = rng.choice(BlefCards, size=sum(hand_sizes), replace=False)
        hands = []
        for i in range(0, len(hand_sizes)):
            cumsum = [0] + list(itertools.accumulate(hand_sizes))
            i_cards = [all_cards[i] for i in range(cumsum[i], cumsum[i+1])]
            i_cards.sort()
            hands.append(i_cards)
        return hands

    @staticmethod
    def hand_combinations(hand_sizes: List[int]) -> Generator[List[List[int]], None, None]:
        def generate(remaining_num_cards, possible_cards):
            if not remaining_num_cards:
                yield []
            else:
                player_hand_size = remaining_num_cards[0]
                for player_hand in itertools.combinations(possible_cards, player_hand_size):
                    for tail in generate(remaining_num_cards[1:], possible_cards - set(player_hand)):
                        yield [sorted(player_hand)] + tail
        return generate(hand_sizes, set(BlefCards))
