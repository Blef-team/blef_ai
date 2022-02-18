from typing import List
import random
import itertools

BlefCards = [str(value) + str(suit) for value in range(6) for suit in range(4)] 

class Game():
    @staticmethod
    def check_finish(history: List[int]) -> bool:
        if (len(history) == 0): 
            return False 
        return history[-1] == 88

    @staticmethod
    def get_payoff(history: List[int], hands: List[List[str]]) -> int:
        """get payoff for player who made last bet"""
        last_bet = history[-2]
        all_cards = hands[0] + hands[1]
        if last_bet in range(0, 6): # High card
            correct = sum([x[0] == str(last_bet) for x in all_cards]) >= 1
        elif last_bet in range(6, 12): # Pair
            correct = sum([x[0] == str(last_bet - 6) for x in all_cards]) >= 2
        elif last_bet in range(12, 27): # Two pair
            if last_bet == 12:
                correct = sum([x[0] == '1' for x in all_cards]) >= 2 & sum([x[0] == '0' for x in all_cards]) >= 2
            elif last_bet in range(13, 15):
                correct = sum([x[0] == '2' for x in all_cards]) >= 2 & sum([x[0] == str(last_bet - 13) for x in all_cards]) >= 2
            elif last_bet in range(15, 18):
                correct = sum([x[0] == '3' for x in all_cards]) >= 2 & sum([x[0] == str(last_bet - 15) for x in all_cards]) >= 2
            elif last_bet in range(18, 22):
                correct = sum([x[0] == '4' for x in all_cards]) >= 2 & sum([x[0] == str(last_bet - 18) for x in all_cards]) >= 2
            else:
                correct = sum([x[0] == '5' for x in all_cards]) >= 2 & sum([x[0] == str(last_bet - 22) for x in all_cards]) >= 2
        elif last_bet == 27: # Small straight
            correct = sum([x[0] == '0' for x in all_cards]) >= 1 & sum([x[0] == '1' for x in all_cards]) >= 1 & sum([x[0] == '2' for x in all_cards]) >= 1 & sum([x[0] == '3' for x in all_cards]) >= 1 & sum([x[0] == '4' for x in all_cards]) >= 1
        elif last_bet == 28: # Big straight
            correct = sum([x[0] == '1' for x in all_cards]) >= 1 & sum([x[0] == '2' for x in all_cards]) >= 1 & sum([x[0] == '3' for x in all_cards]) >= 1 & sum([x[0] == '4' for x in all_cards]) >= 1 & sum([x[0] == '5' for x in all_cards]) >= 1
        elif last_bet == 29: # Great straight
            correct = sum([x[0] == '0' for x in all_cards]) >= 1 & sum([x[0] == '1' for x in all_cards]) >= 1 & sum([x[0] == '2' for x in all_cards]) >= 1 & sum([x[0] == '3' for x in all_cards]) >= 1 & sum([x[0] == '4' for x in all_cards]) >= 1 & sum([x[0] == '5' for x in all_cards]) >= 1
        elif last_bet in range(30, 36): # Three of a kind
            correct = sum([x[0] == str(last_bet - 30) for x in all_cards]) >= 3
        elif last_bet in range(36, 66): # Full house
            if last_bet in range(36, 41):
                correct = sum([x[0] == '0' for x in all_cards]) >= 3 & sum([x[0] == str(last_bet - 36 + 1) for x in all_cards]) >= 2
            elif last_bet in range(41, 42):
                correct = sum([x[0] == '1' for x in all_cards]) >= 3 & sum([x[0] == str(last_bet - 41) for x in all_cards]) >= 2
            elif last_bet in range(42, 46):
                correct = sum([x[0] == '1' for x in all_cards]) >= 3 & sum([x[0] == str(last_bet - 41 + 1) for x in all_cards]) >= 2
            elif last_bet in range(46, 48):
                correct = sum([x[0] == '2' for x in all_cards]) >= 3 & sum([x[0] == str(last_bet - 46) for x in all_cards]) >= 2
            elif last_bet in range(48, 51):
                correct = sum([x[0] == '2' for x in all_cards]) >= 3 & sum([x[0] == str(last_bet - 46 + 1) for x in all_cards]) >= 2
            elif last_bet in range(51, 54):
                correct = sum([x[0] == '3' for x in all_cards]) >= 3 & sum([x[0] == str(last_bet - 51) for x in all_cards]) >= 2
            elif last_bet in range(54, 56):
                correct = sum([x[0] == '3' for x in all_cards]) >= 3 & sum([x[0] == str(last_bet - 51 + 1) for x in all_cards]) >= 2
            elif last_bet in range(56, 60):
                correct = sum([x[0] == '4' for x in all_cards]) >= 3 & sum([x[0] == str(last_bet - 56) for x in all_cards]) >= 2
            elif last_bet in range(60, 61):
                correct = sum([x[0] == '4' for x in all_cards]) >= 3 & sum([x[0] == str(last_bet - 56 + 1) for x in all_cards]) >= 2
            else:
                correct = sum([x[0] == '5' for x in all_cards]) >= 3 & sum([x[0] == str(last_bet - 61) for x in all_cards]) >= 2
        elif last_bet in range(66, 70): # Flush
            correct = sum([x[1] == str(last_bet - 66) for x in all_cards]) >= 5
        elif last_bet in range(70, 76): # Four of a kind
            correct = sum([x[0] == str(last_bet - 70) for x in all_cards]) >= 4
        elif last_bet in range(76, 80): # Small straight flush
            suit = str(last_bet - 76)
            relevant_cards = ['0' + suit, '1' + suit, '2' + suit, '3' + suit, '4' + suit]
            correct = len([x for x in all_cards if x in relevant_cards]) >= 5
        elif last_bet in range(80, 84): # Big straight flush
            suit = str(last_bet - 80)
            relevant_cards = ['1' + suit, '2' + suit, '3' + suit, '4' + suit, '5' + suit]
            correct = len([x for x in all_cards if x in relevant_cards]) >= 5
        elif last_bet in range(84, 88): # Great straight flush
            suit = str(last_bet - 84)
            relevant_cards = ['0' + suit, '1' + suit, '2' + suit, '3' + suit, '4' + suit, '5' + suit]
            correct = len([x for x in all_cards if x in relevant_cards]) >= 6
        if correct:
            return 1
        else:
            return -1
    
    @staticmethod
    def deal_cards(hand_sizes: List[int]):
        all_cards = random.sample(BlefCards, sum(hand_sizes))
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
