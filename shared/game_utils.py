from functools import lru_cache
from itertools import combinations
from typing import Dict, List, Tuple

try:
    from shared.constants import RuleValues  # type: ignore
except Exception:  # pragma: no cover - fallback for local runs without shared.constants
    class RuleValues:  # type: ignore
        DECK_SIZE_DEFAULT = 24

import logging

logger = logging.getLogger(__name__)


def _build_rule_components(deck_size: int) -> Tuple[int, Dict[str, List[int]], Dict[str, List[int]], Dict[str, int]]:
    if deck_size == RuleValues.DECK_SIZE_DEFAULT:
        vals = 6
        straight_types = {
            "Small straight": list(range(5)),
            "Big straight": list(range(1, 6)),
            "Great straight": list(range(6)),
        }
        flush_straight_types = straight_types
    else:  # deck_size == 32
        vals = 8
        straight_types = {f"Straight {i+1}": list(range(i, i + 5)) for i in range(4)}
        flush_straight_types = {f"Straight flush {i+1}": list(range(i, i + 5)) for i in range(4)}

    current_boundary = 0
    boundaries: Dict[str, int] = {}
    boundaries["High card"] = current_boundary + vals
    current_boundary = boundaries["High card"]
    boundaries["Pair"] = current_boundary + vals
    current_boundary = boundaries["Pair"]
    boundaries["Two pairs"] = current_boundary + (vals * (vals - 1)) // 2
    current_boundary = boundaries["Two pairs"]
    boundaries["Straight"] = current_boundary + len(straight_types)
    current_boundary = boundaries["Straight"]
    boundaries["Three of a kind"] = current_boundary + vals
    current_boundary = boundaries["Three of a kind"]
    boundaries["Full house"] = current_boundary + (vals * (vals - 1))
    current_boundary = boundaries["Full house"]
    boundaries["Flush"] = current_boundary + 4
    current_boundary = boundaries["Flush"]
    boundaries["Four of a kind"] = current_boundary + vals
    current_boundary = boundaries["Four of a kind"]
    boundaries["Straight flush"] = current_boundary + (len(flush_straight_types) * 4)

    return vals, straight_types, flush_straight_types, boundaries


@lru_cache(maxsize=2)
class GameRules:
    """Cached computation of rule-dependent quantities."""

    def __init__(self, deck_size: int = RuleValues.DECK_SIZE_DEFAULT):
        self.deck_size = int(deck_size)
        self.vals, self.straight_types, self.flush_straight_types, self.boundaries = _build_rule_components(self.deck_size)
        self.num_actions = self.boundaries["Straight flush"] + 1
        self.check_action_id = self.num_actions - 1


def get_set_details_from_action_id(action_id: int, deck_size: int = RuleValues.DECK_SIZE_DEFAULT) -> Dict[str, object]:
    action_id = int(action_id)
    vals, straight_types, flush_straight_types, boundaries = _build_rule_components(int(deck_size))

    if action_id < boundaries["High card"]:
        return {"set_type": "High card", "detail_1": action_id}
    if action_id < boundaries["Pair"]:
        return {"set_type": "Pair", "detail_1": action_id - boundaries["High card"]}
    if action_id < boundaries["Two pairs"]:
        offset = action_id - boundaries["Pair"]
        pairs = list(combinations(reversed(range(vals)), 2))
        pair_index = len(pairs) - 1 - offset
        d1, d2 = pairs[pair_index]
        return {"set_type": "Two pairs", "detail_1": d1, "detail_2": d2}
    if action_id < boundaries["Straight"]:
        offset = action_id - boundaries["Two pairs"]
        set_name, details = list(straight_types.items())[offset]
        return {"set_type": set_name, "details": details}
    if action_id < boundaries["Three of a kind"]:
        return {"set_type": "Three of a kind", "detail_1": action_id - boundaries["Straight"]}
    if action_id < boundaries["Full house"]:
        offset = action_id - boundaries["Three of a kind"]
        d1 = offset // (vals - 1)
        d2 = offset % (vals - 1)
        if d2 >= d1:
            d2 += 1
        return {"set_type": "Full house", "detail_1": d1, "detail_2": d2}
    if action_id < boundaries["Flush"]:
        return {"set_type": "Flush", "detail_1": action_id - boundaries["Full house"]}
    if action_id < boundaries["Four of a kind"]:
        return {"set_type": "Four of a kind", "detail_1": action_id - boundaries["Flush"]}
    if action_id < boundaries["Straight flush"]:
        offset = action_id - boundaries["Four of a kind"]
        suit = offset % 4
        straight_type_index = offset // 4
        set_name, details = list(flush_straight_types.items())[straight_type_index]
        return {"set_type": "Straight flush", "detail_1": suit, "details": details}
    return None


def check_high_card(cards: List[int], value: int, num_jokers: int) -> bool:
    return cards.count(value) + num_jokers >= 1


def check_pair(cards: List[int], value: int, num_jokers: int) -> bool:
    return cards.count(value) + num_jokers >= 2


def check_two_pairs(cards: List[int], value1: int, value2: int, num_jokers: int) -> bool:
    jokers_needed_for_value1 = max(0, 2 - cards.count(value1))
    jokers_needed_for_value2 = max(0, 2 - cards.count(value2))
    return (jokers_needed_for_value1 + jokers_needed_for_value2) <= num_jokers


def check_straight(cards: List[int], required_values: List[int], num_jokers: int) -> bool:
    unique_cards = set(cards)
    missing_cards = sum(1 for value in required_values if value not in unique_cards)
    return missing_cards <= num_jokers


def check_three_of_a_kind(cards: List[int], value: int, num_jokers: int) -> bool:
    return cards.count(value) + num_jokers >= 3


def check_full_house(cards: List[int], value1: int, value2: int, num_jokers: int) -> bool:
    jokers_needed_for_value1 = max(0, 3 - cards.count(value1))
    jokers_needed_for_value2 = max(0, 2 - cards.count(value2))
    return (jokers_needed_for_value1 + jokers_needed_for_value2) <= num_jokers


def check_flush(cards_by_colour: Dict[int, int], colour: int, num_jokers: int) -> bool:
    return cards_by_colour.get(colour, 0) + num_jokers >= 5


def check_four_of_a_kind(cards: List[int], value: int, num_jokers: int) -> bool:
    return cards.count(value) + num_jokers >= 4


def check_straight_flush(cards_with_colour: List[Tuple[int, int]], colour: int, required_values: List[int], num_jokers: int) -> bool:
    card_set = set(cards_with_colour)
    missing_cards = 0
    for value in required_values:
        if (value, colour) not in card_set:
            missing_cards += 1
    return missing_cards <= num_jokers


def determine_set_existence(all_cards: List[dict], action_id: int, rules: Dict[str, int], num_jokers: int) -> bool:
    try:
        set_details = get_set_details_from_action_id(action_id, rules.get("deck_size", RuleValues.DECK_SIZE_DEFAULT))
        if not set_details:
            logger.error("Could not get set details from action_id.")
            return False
        logger.info(f"Set details from action_id: {set_details}")

        set_type = set_details["set_type"]
        detail_1 = set_details.get("detail_1")
        detail_2 = set_details.get("detail_2")
        details = set_details.get("details")

        card_values = [int(card["value"]) for card in all_cards if int(card["value"]) not in [-1, -2]]
        card_colours = [int(card["colour"]) for card in all_cards if int(card["colour"]) not in [-1, -2]]
        cards_with_colour = [(int(c["value"]), int(c["colour"])) for c in all_cards if int(c["value"]) not in [-1, -2]]
        cards_by_colour = {colour: card_colours.count(colour) for colour in set(card_colours)}

        if set_type == "High card":
            return check_high_card(card_values, detail_1, num_jokers)
        if set_type == "Pair":
            return check_pair(card_values, detail_1, num_jokers)
        if set_type == "Two pairs":
            return check_two_pairs(card_values, detail_1, detail_2, num_jokers)
        if "straight" in set_type.lower() and "flush" not in set_type.lower():
            return check_straight(card_values, details, num_jokers)
        if set_type == "Three of a kind":
            return check_three_of_a_kind(card_values, detail_1, num_jokers)
        if set_type == "Full house":
            return check_full_house(card_values, detail_1, detail_2, num_jokers)
        if set_type == "Flush":
            return check_flush(cards_by_colour, detail_1, num_jokers)
        if set_type == "Four of a kind":
            return check_four_of_a_kind(card_values, detail_1, num_jokers)
        if "straight flush" in set_type.lower():
            return check_straight_flush(cards_with_colour, detail_1, details, num_jokers)
        return False
    except Exception as err:
        print(f"Error in determine_set_existence: {err}")
        return False
