import random
import csv
from cfr_ai.encoding import decode_probabilities
from cfr_ai.information_set import make_key, get_possible_actions, get_hand_abstraction

def determine_action(game_state):
    agent_nickname = game_state["cp_nickname"]
    players = game_state.get("players", [])
    hand_sizes = [player.get("n_cards") for player in players]
    hand_sizes.sort()
    history = []
    if game_state.get("history"):
        history = [action["action_id"] for action in game_state.get("history")]
    matching_hands = [hand for hand in game_state.get("hands", []) if hand.get("nickname") == agent_nickname]
    my_cards = [card["value"] * 4 + card["colour"] for card in matching_hands[0]["hand"]]
    
    with open('metadata.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                key, value = row[0].strip(), row[1].strip()
                if key == 'Minimum bet':
                    min_bet = int(value)
    hand_abstraction = get_hand_abstraction(my_cards, hand_sizes)
    key = make_key(my_cards, hand_abstraction, history, min_bet)
    split_key = key.split('-')
    filename = 'cfr_ai/outputs/' + "_".join(str(x) for x in hand_sizes) + '/' + split_key[0] + '/' + split_key[1] + '.csv'
    relevant_actions = get_possible_actions(history, min_bet)
    with open(filename, 'r', encoding='utf-8') as f:
        strategy_list = csv.reader(f)
        matching_strategies = [x[1] for x in strategy_list if x[0] == '-'.join(split_key[2:])]
    if len(matching_strategies) == 1:
        strategy = decode_probabilities(matching_strategies[0])
        return random.choices(relevant_actions, weights=strategy, k=1)[0]
    elif len(history) == 0:
        print("No policy found though the round has just begun. Betting great straight flush spades (hopefully that was intended)")
        return 87
    else:
        return 88
