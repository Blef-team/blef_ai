import random
import csv
from os import path
from conservative_crawling_ai.agent import determine_action as ask_porevit
from cfr_ai.encoding import decode_probabilities
from cfr_ai.information_set import make_key, get_possible_actions

def determine_action(game_state):
    agent_nickname = game_state["cp_nickname"]
    players = game_state.get("players", [])
    hand_sizes = [player.get("n_cards") for player in players]
    hand_sizes.sort()
    directory = 'cfr_ai/outputs/' + "_".join(str(x) for x in hand_sizes)
    if not path.exists(directory):
        print("Asking Porevit")
        sampled_action = ask_porevit(game_state)
    else:
        history = []
        if game_state.get("history"):
            history = [action["action_id"] for action in game_state.get("history")]
        matching_hands = [hand for hand in game_state.get("hands", []) if hand.get("nickname") == agent_nickname]
        my_cards = [str(card["value"]) + str(card["colour"]) for card in matching_hands[0]["hand"]]
        key = make_key(my_cards, history, hand_sizes)
        split_key = key.split('-')
        filename = 'cfr_ai/outputs/' + "_".join(str(x) for x in hand_sizes) + '/' + split_key[0] + '/' + split_key[1] + '.csv'
        relevant_actions = get_possible_actions(history, hand_sizes)
        with open(filename, 'r', encoding='utf-8') as f:
            strategy_list = csv.reader(f)
            matching_strategies = [x[1] for x in strategy_list if x[0] == '-'.join(split_key[2:])]
        if len(matching_strategies) == 1:
            strategy = decode_probabilities(matching_strategies[0])
            sampled_action = random.choices(relevant_actions, weights=strategy, k=1)[0]
        elif len(history) == 0:
            print("Asking Porevit")
            sampled_action = ask_porevit(game_state)
        else:
            sampled_action = 88
    return sampled_action
