import uuid
from math import floor
from random import shuffle, sample, choice
from itertools import islice, product
import json
import os


CHECK = 88

INDEXATION_CSV = """action_id,set_type,detail_1,detail_2
0,High card,0,
1,High card,1,
2,High card,2,
3,High card,3,
4,High card,4,
5,High card,5,
6,Pair,0,
7,Pair,1,
8,Pair,2,
9,Pair,3,
10,Pair,4,
11,Pair,5,
12,Two pairs,1,0
13,Two pairs,2,0
14,Two pairs,2,1
15,Two pairs,3,0
16,Two pairs,3,1
17,Two pairs,3,2
18,Two pairs,4,0
19,Two pairs,4,1
20,Two pairs,4,2
21,Two pairs,4,3
22,Two pairs,5,0
23,Two pairs,5,1
24,Two pairs,5,2
25,Two pairs,5,3
26,Two pairs,5,4
27,Small straight,,
28,Big straight,,
29,Great straight,,
30,Three of a kind,0,
31,Three of a kind,1,
32,Three of a kind,2,
33,Three of a kind,3,
34,Three of a kind,4,
35,Three of a kind,5,
36,Full house,0,1
37,Full house,0,2
38,Full house,0,3
39,Full house,0,4
40,Full house,0,5
41,Full house,1,0
42,Full house,1,2
43,Full house,1,3
44,Full house,1,4
45,Full house,1,5
46,Full house,2,0
47,Full house,2,1
48,Full house,2,3
49,Full house,2,4
50,Full house,2,5
51,Full house,3,0
52,Full house,3,1
53,Full house,3,2
54,Full house,3,4
55,Full house,3,5
56,Full house,4,0
57,Full house,4,1
58,Full house,4,2
59,Full house,4,3
60,Full house,4,5
61,Full house,5,0
62,Full house,5,1
63,Full house,5,2
64,Full house,5,3
65,Full house,5,4
66,Colour,0,
67,Colour,1,
68,Colour,2,
69,Colour,3,
70,Four of a kind,0,
71,Four of a kind,1,
72,Four of a kind,2,
73,Four of a kind,3,
74,Four of a kind,4,
75,Four of a kind,5,
76,Small flush,0,
77,Small flush,1,
78,Small flush,2,
79,Small flush,3,
80,Big flush,0,
81,Big flush,1,
82,Big flush,2,
83,Big flush,3,
84,Great flush,0,
85,Great flush,1,
86,Great flush,2,
87,Great flush,3,"""


def load_indexation():
    header = None
    indexation = []
    for line in INDEXATION_CSV.split("\n"):
        if not header:
            header = line
            continue
        if not line.strip():
            continue
        action_id, set_type, detail_1, detail_2 = line.split(",")
        indexation.append({"action_id": action_id,
                           "set_type": set_type,
                           "detail_1": detail_1,
                           "detail_2": detail_2
                           })
    return indexation

INDEXATION = load_indexation()

def get_set_details_from_action_id(action_id, deck_size=24):
    """Return set details dict based on local INDEXATION for deck_size 24.
    For future 32-card support, extend INDEXATION_CSV accordingly.
    { 'set_type': str, 'detail_1': int|None, 'detail_2': int|None }
    """
    action_id = int(action_id)
    if deck_size != 24:
        raise ValueError("Only deck_size=24 supported in local manager mapping.")
    if not (0 <= action_id < len(INDEXATION)):
        if action_id == CHECK:
            return {'set_type':'Check'}
        raise ValueError(f"Invalid action_id {action_id}")
    row = INDEXATION[action_id]
    d1 = int(row['detail_1']) if str(row['detail_1']).strip() != '' else None
    d2 = int(row['detail_2']) if str(row['detail_2']).strip() != '' else None
    return {'set_type': row['set_type'], 'detail_1': d1, 'detail_2': d2}


def save(game, dir="games"):
    if not dir:
        return
    state_id = game["game_uuid"] + "_" + str(int(game['round_number']))
    filename = os.path.join(dir, state_id)
    with open(filename, "w") as filehandle:
        json.dump(game, filehandle)


def determine_set_existence(all_cards, action_id, rules=None, num_jokers=0):
    """Check if a claimed set exists in the combined cards, allowing for jokers.
    - all_cards: list of card dicts with keys 'value' and 'colour'
    - action_id: int from local indexation (0..87) or CHECK
    - rules: optional dict; only 'deck_size' used (must be 24 here)
    - num_jokers: usable jokers (value == -1) available to the *bettor* (their hand + common)
    """
    rules = rules or {'deck_size': 24}
    if action_id == CHECK:
        raise ValueError("determine_set_existence called with CHECK action_id")
    details = get_set_details_from_action_id(action_id, rules.get('deck_size', 24))
    set_type = details['set_type']
    d1 = details.get('detail_1')
    d2 = details.get('detail_2')

    # Extract values and suits ignoring jokers (-1)
    values = [int(c['value']) for c in all_cards if int(c.get('value')) >= 0]
    suits = [int(c['colour']) for c in all_cards if int(c.get('value')) >= 0]

    def count_value(v): return sum(1 for x in values if x == v)
    def count_suit(s): return sum(1 for sc in suits if sc == s)
    def need_n(target, have): return max(0, target - have)

    # Straight rank sets for 24-deck (values 0..5)
    small = [0,1,2,3,4]
    big = [1,2,3,4,5]
    great = [0,1,2,3,4,5]

    if set_type == 'High card':
        return need_n(1, count_value(d1)) <= num_jokers

    if set_type == 'Pair':
        return need_n(2, count_value(d1)) <= num_jokers

    if set_type == 'Two pairs':
        miss1 = need_n(2, count_value(d1))
        miss2 = need_n(2, count_value(d2))
        return miss1 + miss2 <= num_jokers

    if set_type == 'Three of a kind':
        return need_n(3, count_value(d1)) <= num_jokers

    if set_type == 'Full house':
        miss_trips = need_n(3, count_value(d1))
        miss_pair = need_n(2, count_value(d2))
        return miss_trips + miss_pair <= num_jokers

    if set_type in ('Colour','Flush'):
        return need_n(5, count_suit(d1)) <= num_jokers

    if set_type == 'Four of a kind':
        return need_n(4, count_value(d1)) <= num_jokers

    # Straights (any suits)
    if set_type == 'Small straight':
        present = set(v for v in values if v in small)
        return len(set(small) - present) <= num_jokers

    if set_type == 'Big straight':
        present = set(v for v in values if v in big)
        return len(set(big) - present) <= num_jokers

    if set_type == 'Great straight':
        present = set(v for v in values if v in great)
        return len(set(great) - present) <= num_jokers

    # Straight flushes (CSV names: Small/Big/Great flush) per suit d1
    if set_type in ('Small flush','Straight flush (small)'):
        suit_vals = [int(c['value']) for c in all_cards if int(c.get('value')) >= 0 and int(c['colour']) == d1]
        present = set(v for v in suit_vals if v in small)
        return len(set(small) - present) <= num_jokers

    if set_type in ('Big flush','Straight flush (big)'):
        suit_vals = [int(c['value']) for c in all_cards if int(c.get('value')) >= 0 and int(c['colour']) == d1]
        present = set(v for v in suit_vals if v in big)
        return len(set(big) - present) <= num_jokers

    if set_type in ('Great flush','Straight flush'):
        suit_vals = [int(c['value']) for c in all_cards if int(c.get('value')) >= 0 and int(c['colour']) == d1]
        present = set(v for v in suit_vals if v in great)
        return len(set(great) - present) <= num_jokers

    return False


def draw_cards(players):
    # Create 24-card deck: values 0..5 (6 ranks) x 4 suits (colours 0..3)
    # If you want jokers locally, you can append {'value': -1, 'colour': -1} to the deck before shuffling.
    deck = [{"value": v, "colour": c} for v, c in product(range(6), range(4))]
    shuffle(deck)
    hands = []
    for p in players:
        if p["n_cards"] == 0:
            continue
        hand = list(islice(deck, p["n_cards"]))
        del deck[:p["n_cards"]]
        hands.append({"nickname": p["nickname"], "hand": hand})
    return hands


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

def create_game(n_agents, max_cards=None, verbose=False):
    if n_agents < 2:
        raise ValueError("n_agents < 2")
    if n_agents > 8:
        raise ValueError("n_agents > 8")
    game_uuid = str(uuid.uuid4())
    players = [{"nickname": str(i), "n_cards": 1} for i in range(n_agents)]
    default_max_cards = floor(24 / len(players)) if len(players) > 2 else 11
    
    rules = {"deck_size": 24}
    
    max_cards_value = default_max_cards
    if max_cards is not None and max_cards > 0:
        max_cards_value = min(max_cards, default_max_cards)
    return {
        "game_uuid": game_uuid,
        "status": "Running",
        "rules": rules,
        "common_hand": [],
        "round_number": 1,
        "max_cards": max_cards_value,
        "hands": draw_cards(players),
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

    set_exists = determine_set_existence(all_cards, game["history"][-2]["action_id"], game.get("rules", {}), num_jokers)

    if set_exists:
        # Checker loses
        losing_player = get_player_by_nickname(game["players"], game["history"][-1]["player"])
    else:
        # Bettor loses
        losing_player = get_player_by_nickname(game["players"], game["history"][-2]["player"])

    game["history"].append({"player": losing_player["nickname"], "action_id": 89})
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
    game["hands"] = draw_cards(game["players"])

    save(game, dir=save_dir)


def play(game, action_id, save_dir="games", verbose=False):
    # Validate action_id
    try:
        action_id = int(action_id)
    except (TypeError, ValueError):
        raise ValueError(f"action_id must be int, got {action_id}")
    # Legality checks
    if action_id == CHECK:
        if not game["history"]:
            raise ValueError("Cannot CHECK as first action; no bet to check.")
        last = game["history"][-1]["action_id"]
        if last == CHECK:
            raise ValueError("Consecutive CHECK not allowed.")
    else:
        if not (0 <= action_id < CHECK):
            raise ValueError(f"Bet action_id out of range: {action_id}")
        if game["history"]:
            last = game["history"][-1]["action_id"]
            if last == CHECK:
                raise ValueError("Cannot bet immediately after a CHECK within the same unresolved round.")
            if action_id <= last:
                raise ValueError(f"Illegal bet {action_id}: must be strictly greater than previous bet {last}.")
    # Append to history
    game["history"].append({"player": game["cp_nickname"], "action_id": action_id})

    if action_id != CHECK:
        move_name = INDEXATION[action_id]['set_type']
        if verbose:
            print(f"player {game['cp_nickname']} plays {move_name}")
        game["cp_nickname"] = find_next_active_player(game["players"], game["cp_nickname"])["nickname"]

    if action_id == CHECK:
        if verbose:
            print(f"player {game['cp_nickname']} checks")
        handle_check(game, save_dir=save_dir)
