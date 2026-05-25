import os
import random
import csv
from typing import Dict, Tuple
from cfr_ai.encoding import decode_probabilities
from cfr_ai.information_set import make_key, get_possible_actions, get_hand_abstraction


# Module-level caches so a warm Lambda container doesn't re-read disk per move.
_min_bet_cache: Dict[Tuple[int, ...], int] = {}
_strategy_file_cache: Dict[Tuple[Tuple[int, ...], int, int], Dict[str, str]] = {}


def _get_min_bet(hand_sizes: Tuple[int, ...]) -> int:
    if hand_sizes in _min_bet_cache:
        return _min_bet_cache[hand_sizes]
    # The deployed worker keeps the per-setup metadata.csv in its working dir;
    # the dev/test path is under cfr_ai/outputs/<setup>/. Try both, and fail
    # loudly if neither is present (matches the pre-refactor behaviour where
    # the missing-file case raised FileNotFoundError).
    candidates = [
        'metadata.csv',
        os.path.join('cfr_ai', 'outputs', '_'.join(str(x) for x in hand_sizes), 'metadata.csv'),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(
            f"No metadata.csv found for setup {hand_sizes!r}. Tried: {candidates}"
        )
    min_bet = 0  # default if the "Minimum bet" row is absent from the file
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip() == 'Minimum bet':
                try:
                    min_bet = int(row[1].strip())
                except ValueError:
                    pass
                break
    _min_bet_cache[hand_sizes] = min_bet
    return min_bet


def _get_strategy_row(hand_sizes: Tuple[int, ...], hand_size: int, last_bet: int, lookup_key: str):
    cache_key = (hand_sizes, hand_size, last_bet)
    if cache_key not in _strategy_file_cache:
        filename = os.path.join(
            'cfr_ai', 'outputs', '_'.join(str(x) for x in hand_sizes),
            str(hand_size), f'{last_bet}.csv',
        )
        rows: Dict[str, str] = {}
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if row and row[0]:
                        rows[row[0]] = row[1]
        _strategy_file_cache[cache_key] = rows
    return _strategy_file_cache[cache_key].get(lookup_key)


def determine_action(game_state):
    agent_nickname = game_state["cp_nickname"]
    players = game_state.get("players", [])
    hand_sizes = tuple(sorted(player.get("n_cards") for player in players))
    history = []
    if game_state.get("history"):
        history = [action["action_id"] for action in game_state.get("history")]
    matching_hands = [hand for hand in game_state.get("hands", []) if hand.get("nickname") == agent_nickname]
    my_cards = [card["value"] * 4 + card["colour"] for card in matching_hands[0]["hand"]]

    min_bet = _get_min_bet(hand_sizes)
    hand_abstraction = get_hand_abstraction(my_cards, list(hand_sizes))
    key = make_key(my_cards, hand_abstraction, history, min_bet)
    split_key = key.split('-')
    hand_size = int(split_key[0])
    last_bet = int(split_key[1])
    lookup_key = '-'.join(split_key[2:])
    relevant_actions = get_possible_actions(history, min_bet)

    encoded_strategy = _get_strategy_row(hand_sizes, hand_size, last_bet, lookup_key)
    if encoded_strategy is not None:
        strategy = decode_probabilities(encoded_strategy)
        return random.choices(relevant_actions, weights=strategy, k=1)[0]
    if len(history) == 0:
        # Round-start fallback: bet the most senior set (great straight flush
        # spades) as a loud signal that the relevant policy is missing 
        print("No policy found though the round has just begun. Betting great straight flush spades (hopefully that was intended)")
        return 87
    return 88  # mid-round fallback: check
