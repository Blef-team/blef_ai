import uuid
from functools import lru_cache
from math import floor
from random import shuffle
from itertools import islice, product
import json
import os

from shared.game_utils import (
    GameRules,
    get_set_details_from_action_id as _shared_get_set_details,
    determine_set_existence as _shared_determine_set_existence,
)


DEFAULT_DECK_SIZE = 24


def _resolve_deck_size(rules: dict) -> int:
    return int(rules.get("deck_size", DEFAULT_DECK_SIZE))


@lru_cache(maxsize=4)
def _get_rules_cache(deck_size: int) -> GameRules:
    return GameRules(deck_size)


def _get_game_rules(rules: dict) -> GameRules:
    return _get_rules_cache(_resolve_deck_size(rules))


def _get_check_action_id(rules: dict) -> int:
    return _get_game_rules(rules).check_action_id


def _get_num_actions(rules: dict) -> int:
    return _get_game_rules(rules).num_actions


def _get_set_details(action_id: int, rules: dict) -> dict:
    deck_size = _resolve_deck_size(rules)
    return _shared_get_set_details(action_id, deck_size)


def save(game, dir="games"):
    if not dir:
        return
    state_id = game["game_uuid"] + "_" + str(int(game['round_number']))
    filename = os.path.join(dir, state_id)
    with open(filename, "w") as filehandle:
        json.dump(game, filehandle)


def determine_set_existence(all_cards, action_id, rules=None, num_jokers=None):
    rules = rules or {"deck_size": DEFAULT_DECK_SIZE}
    check_action_id = _get_check_action_id(rules)
    if action_id == check_action_id:
        raise ValueError("determine_set_existence called with CHECK action_id")

    if num_jokers is None:
        num_jokers = sum(1 for card in all_cards if int(card.get("value", -3)) == -1)
    num_jokers = max(0, int(num_jokers))

    return _shared_determine_set_existence(all_cards, action_id, rules, num_jokers)


def _legacy_determine_set_existence(all_cards, action_id, rules=None, num_jokers=None, num_blanks=None):
    """Check if a claimed set exists in the combined cards, allowing for jokers."""
    rules = rules or {"deck_size": DEFAULT_DECK_SIZE}
    check_action_id = _get_check_action_id(rules)
    if action_id == check_action_id:
        raise ValueError("determine_set_existence called with CHECK action_id")

    deck_size = _resolve_deck_size(rules)
    num_values = deck_size // 4
    details = _get_set_details(action_id, rules)
    set_type = details.get("set_type", "")
    detail_1 = details.get("detail_1")
    detail_2 = details.get("detail_2")
    required_values = details.get("details", [])

    value_counts = [0] * num_values
    suit_counts = [0] * 4
    value_suit_counts = [[0] * 4 for _ in range(num_values)]
    total_jokers = 0
    total_blanks = 0

    for card in all_cards:
        value = int(card.get("value", -3))
        colour = int(card.get("colour", -1))
        if value >= 0:
            if not (0 <= value < num_values):
                raise ValueError(f"Card value {value} incompatible with deck size {deck_size}")
            value_counts[value] += 1
            if 0 <= colour < 4:
                suit_counts[colour] += 1
                value_suit_counts[value][colour] += 1
        elif value == -1:
            total_jokers += 1
        elif value == -2:
            total_blanks += 1

    if num_jokers is None:
        num_jokers = total_jokers
    num_jokers = max(0, int(num_jokers))

    if num_blanks is None:
        num_blanks = total_blanks
    num_blanks = max(0, int(num_blanks))  # currently unused but retained for future rules

    def need(target: int, have: int) -> int:
        return max(0, target - have)

    if set_type in {"High card", "Pair", "Three of a kind", "Four of a kind"}:
        required = {"High card": 1, "Pair": 2, "Three of a kind": 3, "Four of a kind": 4}[set_type]
        if detail_1 is None:
            return False
        missing = need(required, value_counts[detail_1])
        return missing <= num_jokers

    if set_type == "Two pairs":
        if detail_1 is None or detail_2 is None:
            return False
        missing = need(2, value_counts[detail_1]) + need(2, value_counts[detail_2])
        return missing <= num_jokers

    if set_type == "Three of a kind":  # already handled above, but keep for safety
        if detail_1 is None:
            return False
        missing = need(3, value_counts[detail_1])
        return missing <= num_jokers

    if set_type == "Full house":
        if detail_1 is None or detail_2 is None:
            return False
        missing = need(3, value_counts[detail_1]) + need(2, value_counts[detail_2])
        return missing <= num_jokers

    if set_type == "Flush":
        if detail_1 is None:
            return False
        missing = need(5, suit_counts[detail_1])
        return missing <= num_jokers

    if set_type == "Four of a kind":  # redundancy guard
        if detail_1 is None:
            return False
        missing = need(4, value_counts[detail_1])
        return missing <= num_jokers

    if set_type == "Straight flush":
        suit = detail_1
        if suit is None or not required_values:
            return False
        missing = sum(1 for v in required_values if value_suit_counts[v][suit] == 0)
        return missing <= num_jokers

    if "Straight" in set_type:
        if not required_values:
            return False
        missing = sum(1 for v in required_values if value_counts[v] == 0)
        return missing <= num_jokers

    return False


def draw_cards(players, rules):
    deck_size = _resolve_deck_size(rules)
    num_values = deck_size // 4
    deck = [{"value": v, "colour": c} for v, c in product(range(num_values), range(4))]
    for _ in range(int(rules.get("jokers", 0))):
        deck.append({"value": -1, "colour": -1})
    for _ in range(int(rules.get("blanks", 0))):
        deck.append({"value": -2, "colour": -1})
    shuffle(deck)

    common_count = max(0, int(rules.get("common_cards", 0)))
    if common_count > len(deck):
        raise ValueError("Not enough cards in deck to deal common cards.")
    common_hand = list(islice(deck, common_count))
    del deck[:common_count]

    hands = []
    for p in players:
        if p["n_cards"] == 0:
            continue
        hand = list(islice(deck, p["n_cards"]))
        del deck[:p["n_cards"]]
        hands.append({"nickname": p["nickname"], "hand": hand})
    return hands, common_hand


def get_player_by_nickname(players, nickname):
    filtered_players = [p for p in players if p["nickname"] == nickname]
    if filtered_players:
        return filtered_players[0]


def find_next_active_player(players, nickname):
    # Find the index of the current player
    current_idx = next((i for i, p in enumerate(players) if p["nickname"] == nickname), None)
    if current_idx is None:
        return None
    # Iterate through players to find the next with n_cards > 0
    n = len(players)
    for i in range(1, n + 1):
        candidate = players[(current_idx + i) % n]
        if candidate["n_cards"] > 0:
            return candidate
    return None


def arrange_players(players):
    # First, include only active players
    active_players = [p for p in players if p.get("n_cards", 0) > 0]
    if not active_players:
        return []

    # Split human and AI players
    human_players = [p for p in active_players if p.get("nickname", "").startswith("HUMAN")]
    ai_players    = [p for p in active_players if not p.get("nickname", "").startswith("HUMAN")]

    # Ensure alternating sequence starting with the most numerous type if needed
    players = []
    while human_players or ai_players:
        if len(human_players) >= len(ai_players) and human_players:
            players.append(human_players.pop(0))
        elif ai_players:
            players.append(ai_players.pop(0))
        elif human_players:
            players.append(human_players.pop(0))

    return players

def create_game(
    n_agents,
    deck_size=DEFAULT_DECK_SIZE,
    max_cards=None,
    jokers=0,
    blanks=0,
    common_cards=0,
    verbose=False,
    init_card_dist=None,
):
    if n_agents < 2:
        raise ValueError("n_agents < 2")
    if n_agents > 8:
        raise ValueError("n_agents > 8")
    game_uuid = str(uuid.uuid4())
    if init_card_dist is None:
        players = [{"nickname": str(i), "n_cards": 1} for i in range(n_agents)]
    else:
        if len(init_card_dist) != n_agents:
            raise ValueError(
                f"init_card_dist length {len(init_card_dist)} must equal n_agents {n_agents}"
            )
        players = [
            {"nickname": str(i), "n_cards": int(max(1, init_card_dist[i]))}
            for i in range(n_agents)
        ]
    if len(players) > 2:
        default_max_cards = floor(deck_size / len(players))
    else:
        default_max_cards = max(1, deck_size // 2 - 1)

    if common_cards < 0:
        raise ValueError("common_cards must be non-negative")
    rules = {
        "deck_size": int(deck_size),
        "jokers": int(jokers),
        "blanks": int(blanks),
        "common_cards": int(common_cards),
    }
    hands, common_hand = draw_cards(players, rules)

    max_cards_value = default_max_cards
    if max_cards is not None and max_cards > 0:
        max_cards_value = min(max_cards, default_max_cards)
    return {
        "game_uuid": game_uuid,
        "status": "Running",
        "rules": rules,
        "common_hand": common_hand,
        "round_number": 1,
        "max_cards": max_cards_value,
        "hands": hands,
        "players": players,
        "cp_nickname": players[0]["nickname"],
        "history": []
    }


def handle_check(game, save_dir="games"):
    cp_nickname = game["cp_nickname"]

    # Identify bettor (the one being checked) = previous action
    bettor_nickname = game["history"][-2]["player"]

    # All cards in round = all players' hands + common hand
    all_cards = [card for hand in game["hands"] for card in hand["hand"]]
    all_cards.extend(game.get("common_hand", []))

    # Count usable jokers: only from bettor's hand + common hand
    bettor_hand = next((h["hand"] for h in game["hands"] if h["nickname"] == bettor_nickname), [])
    num_jokers = sum(1 for c in bettor_hand if int(c.get("value")) == -1)
    num_jokers += sum(1 for c in game.get("common_hand", []) if int(c.get("value")) == -1)

    set_exists = determine_set_existence(
        all_cards,
        game["history"][-2]["action_id"],
        game.get("rules", {}),
        num_jokers,
    )

    if set_exists:
        # Checker loses
        losing_player = get_player_by_nickname(game["players"], game["history"][-1]["player"])
    else:
        # Bettor loses
        losing_player = get_player_by_nickname(game["players"], game["history"][-2]["player"])

    elimination_action_id = _get_num_actions(game["rules"])
    game["history"].append({"player": losing_player["nickname"], "action_id": elimination_action_id})
    game["cp_nickname"] = None

    # Store the last round separately
    save(game, dir=save_dir)

    # Penalize loser
    losing_player["n_cards"] += 1
    # Elimination if exceeding max_cards
    if losing_player["n_cards"] > game["max_cards"]:
        losing_player["n_cards"] = 0
        # Check if game is finished
        if sum(p["n_cards"] > 0 for p in game["players"]) == 1:
            game["status"] = "Finished"
        else:
            # If the checking player was eliminated, figure out the next player; otherwise keep CP
            if losing_player["nickname"] == cp_nickname:
                game["cp_nickname"] = find_next_active_player(game["players"], cp_nickname)["nickname"]
            else:
                game["cp_nickname"] = cp_nickname
    else:
        # If no one is kicked out, next player is the loser (as they take first in next round)
        game["cp_nickname"] = losing_player["nickname"]

    # New round: reset history, re-deal
    game["round_number"] += 1
    game["history"] = []
    game["hands"], game["common_hand"] = draw_cards(game["players"], game["rules"])

    save(game, dir=save_dir)


def play(game, action_id, save_dir="games", verbose=False):
    # Validate action_id
    try:
        action_id = int(action_id)
    except (TypeError, ValueError):
        raise ValueError(f"action_id must be int, got {action_id}")
    rules = game.get("rules", {})
    check_action_id = _get_check_action_id(rules)
    # Legality checks
    if action_id == check_action_id:
        if not game["history"]:
            raise ValueError("Cannot CHECK as first action; no bet to check.")
        last = game["history"][-1]["action_id"]
        if last == check_action_id:
            raise ValueError("Consecutive CHECK not allowed.")
    else:
        if not (0 <= action_id < check_action_id):
            raise ValueError(f"Bet action_id out of range: {action_id}")
        if game["history"]:
            last = game["history"][-1]["action_id"]
            if last == check_action_id:
                raise ValueError("Cannot bet immediately after a CHECK within the same unresolved round.")
            if action_id <= last:
                raise ValueError(f"Illegal bet {action_id}: must be strictly greater than previous bet {last}.")
    # Append to history
    game["history"].append({"player": game["cp_nickname"], "action_id": action_id})

    if action_id != check_action_id:
        move_name = _get_set_details(action_id, rules)['set_type']
        if verbose:
            print(f"player {game['cp_nickname']} plays {move_name}")
        game["cp_nickname"] = find_next_active_player(game["players"], game["cp_nickname"])["nickname"]

    if action_id == check_action_id:
        if verbose:
            print(f"player {game['cp_nickname']} checks")
        handle_check(game, save_dir=save_dir)
