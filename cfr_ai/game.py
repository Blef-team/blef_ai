from typing import List
import random
import itertools


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
        if (last_bet in range(0, 6)): # High card
            correct = sum([x[0] == str(last_bet) for x in all_cards]) >= 1
        elif (last_bet in range(6, 12)): # Pair
            correct = sum([x[0] == str(last_bet - 6) for x in all_cards]) >= 2
        elif (last_bet in range(12, 27)): # Two pair
            if (last_bet == 12):
                correct = sum([x[0] == '1' for x in all_cards]) >= 2 & sum([x[0] == '0' for x in all_cards]) >= 2
            elif (last_bet in range(13, 15)):
                correct = sum([x[0] == '2' for x in all_cards]) >= 2 & sum([x[0] == str(last_bet - 13) for x in all_cards]) >= 2
            elif (last_bet in range(15, 18)):
                correct = sum([x[0] == '3' for x in all_cards]) >= 2 & sum([x[0] == str(last_bet - 15) for x in all_cards]) >= 2
            elif (last_bet in range(18, 22)):
                correct = sum([x[0] == '4' for x in all_cards]) >= 2 & sum([x[0] == str(last_bet - 18) for x in all_cards]) >= 2
            else:
                correct = sum([x[0] == str(5) for x in all_cards]) >= 2 & sum([x[0] == str(last_bet - 22) for x in all_cards]) >= 2
        elif (last_bet == 27): # Small straight
            correct = sum([x[0] == '0' for x in all_cards]) >= 1 & sum([x[0] == '1' for x in all_cards]) >= 1 & sum([x[0] == '2' for x in all_cards]) >= 1 & sum([x[0] == '3' for x in all_cards]) >= 1 & sum([x[0] == '4' for x in all_cards]) >= 1
        elif (last_bet == 28): # Big straight
            correct = sum([x[0] == '1' for x in all_cards]) >= 1 & sum([x[0] == '2' for x in all_cards]) >= 1 & sum([x[0] == '3' for x in all_cards]) >= 1 & sum([x[0] == '4' for x in all_cards]) >= 1 & sum([x[0] == '5' for x in all_cards]) >= 1
        elif (last_bet == 29): # Great straight
            correct = sum([x[0] == '0' for x in all_cards]) >= 1 & sum([x[0] == '1' for x in all_cards]) >= 1 & sum([x[0] == '2' for x in all_cards]) >= 1 & sum([x[0] == '3' for x in all_cards]) >= 1 & sum([x[0] == '4' for x in all_cards]) >= 1 & sum([x[0] == '5' for x in all_cards]) >= 1
        elif (last_bet in range(30, 36)): # Three of a kind
            correct = sum([x[0] == str(last_bet - 12) for x in all_cards]) >= 3
        else:
            correct = False
        if correct:
            return 1
        else:
            return -1
    
    @staticmethod
    def deal_cards(BlefCards, NumCards):
        all_cards = random.sample(BlefCards, sum(NumCards))
        hands = []
        for i in range(0, len(NumCards)):
            cumsum = [0] + list(itertools.accumulate(NumCards))
            i_cards = [all_cards[i] for i in range(cumsum[i], cumsum[i+1])]
            i_cards.sort()
            hands.append(i_cards)
        return hands

    @staticmethod
    def hand_combinations(BlefCards, NumCards):
        def generate(remaining_num_cards, possible_cards):
            if not remaining_num_cards:
                yield []
            else:
                player_hand_size = remaining_num_cards[0]
                for player_hand in itertools.combinations(possible_cards, player_hand_size):
                    for tail in generate(remaining_num_cards[1:], possible_cards - set(player_hand)):
                        yield [sorted(player_hand)] + tail
        return generate(NumCards, set(BlefCards))
