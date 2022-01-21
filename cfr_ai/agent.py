import random
import csv
from os import path
from conservative_crawling_ai.agent import determine_action as ask_porevit
from cfr_ai.encoding import decode_probabilities
from cfr_ai.information_set import make_key, get_possible_actions

def determine_action(game_state):
    agent_nickname = game_state["cp_nickname"]
    players = game_state.get("players", [])
    if game_state.get("history"):
        starting_player = game_state.get("history")[0]["player"]
        starting_player_index = [i for i in range(len(players)) if players[i].get("nickname") == starting_player][0]
    else:
        starting_player_index = [i for i in range(len(players)) if players[i].get("nickname") == agent_nickname][0]
    reorganised_players = []
    for i in range(len(players)):
        reorganised_players.append(players[(starting_player_index + i) % len(players)])

    NumCards = [player.get("n_cards") for player in reorganised_players]
    filename = 'cfr_ai/outputs/' + "_".join(str(x) for x in NumCards) + '.csv'
    if not path.exists(filename):
        print("Asking Porevit")
        sampled_action = ask_porevit(game_state)
    else:
        history = []
        if game_state.get("history"):
            history = [action["action_id"] for action in game_state.get("history")]
        matching_hands = [hand for hand in game_state.get("hands", []) if hand.get("nickname") == agent_nickname]
        my_cards = [str(card["value"]) + str(card["colour"]) for card in matching_hands[0]["hand"]]
        key = make_key(my_cards, history, NumCards)
        relevant_actions = get_possible_actions(history, NumCards)
        with open(filename, 'r', encoding='utf-8') as f:
            strategy_list = csv.reader(f)
            matching_strategies = [x[1] for x in strategy_list if x[0] == key]
        if len(matching_strategies) == 1:
            strategy = decode_probabilities(matching_strategies[0])
            sampled_action = random.choices(relevant_actions, weights=strategy, k=1)[0]
        elif len(history) == 0:
            print("Asking Porevit")
            sampled_action = ask_porevit(game_state)
        else:
            sampled_action = 88
    return sampled_action
