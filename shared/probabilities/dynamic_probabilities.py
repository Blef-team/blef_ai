import math
from itertools import product
from functools import lru_cache
from typing import Iterable, Tuple, NamedTuple

from shared.game_utils import GameRules, get_set_details_from_action_id


def binom(n, k):
    if n < k or k < 0:
        return 0
    return math.comb(n, k)

def _calculate_prob_simple_set(needed, num_cards_of_value_in_deck, jokers_in_deck, other_cards_in_deck, unknown_cards_num):
    """Helper for High Card, Pair, Three of a Kind, Four of a Kind."""
    total_unknown_in_deck = num_cards_of_value_in_deck + jokers_in_deck + other_cards_in_deck
    if total_unknown_in_deck < unknown_cards_num: return 0.0

    # Calculate the number of ways to FAIL to get the needed cards, then subtract from 1
    failing_outcomes = 0
    for i in range(needed): # i is the number of "good" cards (value or joker) drawn
        # Sum the ways to draw i good cards
        failing_outcomes += binom(num_cards_of_value_in_deck + jokers_in_deck, i) * binom(other_cards_in_deck, unknown_cards_num - i)

    total_outcomes = binom(total_unknown_in_deck, unknown_cards_num)
    return 1 - (failing_outcomes / total_outcomes if total_outcomes > 0 else 0.0)

def _calculate_prob_two_pairs(needed1, needed2, cards1_in_deck, cards2_in_deck, jokers_in_deck, other_cards_in_deck, unknown_cards_num):
    """
    Calculates the exact probability of forming two pairs using a multivariate hypergeometric distribution.

    This is done by iterating through all possible combinations of relevant cards (cards for pair 1,
    cards for pair 2, and jokers) that could be drawn from the deck, and summing the probabilities
    of the combinations that satisfy the conditions for making both pairs.
    """
    total_unknown_in_deck = cards1_in_deck + cards2_in_deck + jokers_in_deck + other_cards_in_deck
    if total_unknown_in_deck < unknown_cards_num:
        return 0.0

    successful_outcomes = 0

    # Iterate through all possible numbers of cards of value 1 that can be drawn
    for i in range(min(unknown_cards_num, cards1_in_deck) + 1):
        # Iterate through all possible numbers of cards of value 2 that can be drawn
        for j in range(min(unknown_cards_num - i, cards2_in_deck) + 1):
            # Iterate through all possible numbers of jokers that can be drawn
            for k in range(min(unknown_cards_num - i - j, jokers_in_deck) + 1):
                
                # Check if the number of jokers drawn is sufficient to complete both pairs
                jokers_needed_for_1 = max(0, needed1 - i)
                jokers_needed_for_2 = max(0, needed2 - j)

                if jokers_needed_for_1 + jokers_needed_for_2 <= k:
                    # This combination of drawn cards results in a success.
                    # Calculate how many "other" cards must be in this hand.
                    num_other_cards = unknown_cards_num - i - j - k
                    if num_other_cards >= 0 and num_other_cards <= other_cards_in_deck:
                        
                        # Calculate the number of ways this specific successful hand can be formed
                        term = (binom(cards1_in_deck, i) *
                                binom(cards2_in_deck, j) *
                                binom(jokers_in_deck, k) *
                                binom(other_cards_in_deck, num_other_cards))
                        
                        successful_outcomes += term

    total_outcomes = binom(total_unknown_in_deck, unknown_cards_num)

    return successful_outcomes / total_outcomes if total_outcomes > 0 else 0.0

def _calculate_prob_straight(needed_card_values, cards_in_deck_counts, jokers_in_deck, other_cards_in_deck, unknown_cards_num):
    """
    Calculates the exact probability of completing a straight by summing all successful outcomes.
    This function iterates through all combinations of drawing the required cards and jokers.
    """
    total_unknown_in_deck = sum(cards_in_deck_counts) + jokers_in_deck + other_cards_in_deck
    if total_unknown_in_deck < unknown_cards_num:
        return 0.0

    successful_outcomes = 0
    total_outcomes = binom(total_unknown_in_deck, unknown_cards_num)
    if total_outcomes == 0:
        return 0.0

    # Create iterators for the number of cards to draw for each needed rank
    iter_ranges = [range(min(unknown_cards_num, count) + 1) for count in cards_in_deck_counts]

    # Iterate through all combinations of counts for each needed rank
    for rank_counts in product(*iter_ranges):
        num_rank_cards_drawn = sum(rank_counts)
        if num_rank_cards_drawn > unknown_cards_num:
            continue

        # Count how many ranks we missed (i.e., drew 0 cards for)
        num_ranks_missed = sum(1 for count in rank_counts if count == 0)

        # Now, iterate through the number of jokers we could draw
        max_jokers = min(jokers_in_deck, unknown_cards_num - num_rank_cards_drawn)
        for num_jokers_drawn in range(max_jokers + 1):
            
            # Success condition: The number of jokers drawn is enough to cover the missing ranks
            if num_jokers_drawn >= num_ranks_missed:
                
                num_other_cards_drawn = unknown_cards_num - num_rank_cards_drawn - num_jokers_drawn
                if 0 <= num_other_cards_drawn <= other_cards_in_deck:
                    
                    # Calculate the number of ways this combination can happen
                    term = binom(jokers_in_deck, num_jokers_drawn) * binom(other_cards_in_deck, num_other_cards_drawn)
                    for i, rank_count in enumerate(rank_counts):
                        term *= binom(cards_in_deck_counts[i], rank_count)
                    
                    successful_outcomes += term

    return successful_outcomes / total_outcomes if total_outcomes > 0 else 0.0

def _calculate_prob_full_house(needed3, needed2, cards3_in_deck, cards2_in_deck, jokers_in_deck, other_cards_in_deck, unknown_cards_num):
    """
    Calculates the exact probability of forming a full house by counting all successful outcomes.

    This function iterates through every possible combination of drawn cards that could form the
    three-of-a-kind and the pair, including the use of jokers. It sums the number of ways
    each successful combination can occur to find the total number of successful hands.
    """
    total_unknown_in_deck = cards3_in_deck + cards2_in_deck + jokers_in_deck + other_cards_in_deck
    if total_unknown_in_deck < unknown_cards_num:
        return 0.0

    successful_outcomes = 0

    # Iterate through all possible numbers of cards for the three-of-a-kind
    for i in range(min(unknown_cards_num, cards3_in_deck) + 1):
        # Iterate through all possible numbers of cards for the pair
        for j in range(min(unknown_cards_num - i, cards2_in_deck) + 1):
            # Iterate through all possible numbers of jokers
            for k in range(min(unknown_cards_num - i - j, jokers_in_deck) + 1):

                # How many jokers do we still need after drawing i and j cards?
                jokers_needed_for_3 = max(0, needed3 - i)
                jokers_needed_for_2 = max(0, needed2 - j)

                # Is the number of jokers drawn sufficient?
                if jokers_needed_for_3 + jokers_needed_for_2 <= k:
                    # This is a successful draw. Calculate how many ways it can happen.
                    num_other_cards = unknown_cards_num - i - j - k
                    
                    if num_other_cards >= 0 and num_other_cards <= other_cards_in_deck:
                        term = (binom(cards3_in_deck, i) *
                                binom(cards2_in_deck, j) *
                                binom(jokers_in_deck, k) *
                                binom(other_cards_in_deck, num_other_cards))
                        successful_outcomes += term

    total_outcomes = binom(total_unknown_in_deck, unknown_cards_num)

    return successful_outcomes / total_outcomes if total_outcomes > 0 else 0.0

def _normalize_cards(cards: Iterable[Tuple[int, int]]):
    # Order-agnostic representation for caching
    return tuple(sorted((int(v), int(c)) for (v, c) in cards))


def _normalize_rules(rules: dict):
    deck_size = int(rules.get("deck_size", 24))
    jokers = int(rules.get("jokers", 0))
    blanks = int(rules.get("blanks", 0))
    return (deck_size, jokers, blanks)


def _rebuild_rules(rules_key: Tuple[int, int, int]) -> dict:
    deck_size, jokers, blanks = rules_key
    return {"deck_size": deck_size, "jokers": jokers, "blanks": blanks}


class ProbContext(NamedTuple):
    hand_cards: Tuple[Tuple[int, int], ...]
    allied_cards: Tuple[Tuple[int, int], ...]
    common_cards: Tuple[Tuple[int, int], ...]
    known_cards: Tuple[Tuple[int, int], ...]
    value_counts: Tuple[int, ...]
    colour_counts: Tuple[int, ...]
    my_jokers: int
    allied_jokers: int
    common_jokers: int
    deck_size: int
    total_jokers: int
    total_blanks: int


@lru_cache(maxsize=50000)
def _prepare_prob_context(
    hand_key: Tuple[Tuple[int, int], ...], 
    allied_key: Tuple[Tuple[int, int], ...], 
    common_key: Tuple[Tuple[int, int], ...], 
    rules_key: Tuple[int, int, int]
) -> ProbContext:
    deck_size, total_jokers, total_blanks = rules_key
    hand_cards = tuple((int(v), int(c)) for v, c in hand_key)
    allied_cards = tuple((int(v), int(c)) for v, c in allied_key)
    common_cards = tuple((int(v), int(c)) for v, c in common_key)
    known_cards = hand_cards + allied_cards + common_cards

    if deck_size % 4 != 0:
        raise ValueError(f"Deck size {deck_size} unsupported (must be divisible by 4).")
    num_values = deck_size // 4
    value_counts = [0] * num_values
    colour_counts = [0] * 4

    my_jokers = sum(1 for v, c in hand_cards if v == -1)
    allied_jokers = sum(1 for v, c in allied_cards if v == -1)
    common_jokers = sum(1 for v, c in common_cards if v == -1)

    for value, colour in known_cards:
        if value >= 0:
            if value >= num_values:
                raise ValueError(f"Card value {value} exceeds deck bounds for deck size {deck_size}.")
            value_counts[value] += 1
            if colour >= 0:
                colour_counts[colour] += 1

    return ProbContext(
        hand_cards=hand_cards,
        allied_cards=allied_cards,
        common_cards=common_cards,
        known_cards=known_cards,
        value_counts=tuple(value_counts),
        colour_counts=tuple(colour_counts),
        my_jokers=my_jokers,
        allied_jokers=allied_jokers,
        common_jokers=common_jokers,
        deck_size=int(deck_size),
        total_jokers=int(total_jokers),
        total_blanks=int(total_blanks),
    )

def _calculate_prob_with_context(action_id: int, ctx: ProbContext, unknown_cards_num: int, for_betting: bool) -> float:
    rules = {"deck_size": ctx.deck_size, "jokers": ctx.total_jokers, "blanks": ctx.total_blanks}
    set_details = get_set_details_from_action_id(action_id, rules.get("deck_size", 24))
    if not set_details: return 0.0

    known_cards = ctx.known_cards
    value_counts = list(ctx.value_counts)
    colour_counts = list(ctx.colour_counts)
    my_jokers = ctx.my_jokers
    allied_jokers = ctx.allied_jokers
    common_jokers = ctx.common_jokers
    known_jokers = my_jokers + allied_jokers + common_jokers

    jokers_in_play = my_jokers + common_jokers if for_betting else common_jokers

    deck_size = ctx.deck_size
    total_jokers = ctx.total_jokers
    total_blanks = ctx.total_blanks

    set_type = set_details.get("set_type")

    if set_type in ["High card", "Pair", "Three of a kind", "Four of a kind"]:
        value_to_check = set_details["detail_1"]
        if value_to_check >= len(value_counts):
            return 0.0
        needed_map = {"High card": 1, "Pair": 2, "Three of a kind": 3, "Four of a kind": 4}
        needed = needed_map[set_type] - value_counts[value_to_check] - jokers_in_play
        if needed <= 0: return 1.0

        cards_of_value_in_deck = max(0, 4 - value_counts[value_to_check])
        jokers_in_deck = total_jokers - known_jokers
        other_cards_in_deck = deck_size + rules.get("blanks", 0) - len(known_cards) + known_jokers - cards_of_value_in_deck
        
        return _calculate_prob_simple_set(needed, cards_of_value_in_deck, jokers_in_deck, other_cards_in_deck, unknown_cards_num)
        
    if set_type == "Two pairs":
        val1, val2 = set_details["detail_1"], set_details["detail_2"]
        if val1 >= len(value_counts) or val2 >= len(value_counts):
            return 0.0
        needed1 = 2 - value_counts[val1]
        needed2 = 2 - value_counts[val2]
        
        if needed1 + needed2 <= jokers_in_play: return 1.0
        
        cards1_in_deck = max(0, 4 - value_counts[val1])
        cards2_in_deck = max(0, 4 - value_counts[val2])
        jokers_in_deck = total_jokers - known_jokers
        other_cards_in_deck = deck_size + rules.get("blanks", 0) - len(known_cards) + known_jokers - cards1_in_deck - cards2_in_deck
        
        return _calculate_prob_two_pairs(needed1, needed2, cards1_in_deck, cards2_in_deck, jokers_in_deck, other_cards_in_deck, unknown_cards_num)

    if "straight" in set_type.lower() and "flush" not in set_type:
        required_values = set(set_details["details"])
        
        present_values = {c[0] for c in known_cards if c[0] in required_values}
        needed_values = list(required_values - present_values)
        
        jokers_needed = len(needed_values) - jokers_in_play
        if jokers_needed <= 0: return 1.0
        
        cards_in_deck_counts = [4 - value_counts[v] for v in needed_values]
        cards_in_deck_counts = [max(0, c) for c in cards_in_deck_counts]
        jokers_in_deck = total_jokers - known_jokers
        other_cards_in_deck = deck_size + rules.get("blanks", 0) - len(known_cards) + known_jokers - sum(cards_in_deck_counts)
        
        return _calculate_prob_straight(needed_values, cards_in_deck_counts, jokers_in_deck, other_cards_in_deck, unknown_cards_num)

    if set_type == "Flush":
        color_to_check = set_details["detail_1"]
        needed = 5 - colour_counts[color_to_check] - jokers_in_play
        if needed <= 0: return 1.0
        
        cards_in_deck = (deck_size // 4) - colour_counts[color_to_check]
        jokers_in_deck = total_jokers - known_jokers
        other_cards_in_deck = deck_size + rules.get("blanks", 0) - len(known_cards) + known_jokers - cards_in_deck
        
        return _calculate_prob_simple_set(needed, cards_in_deck, jokers_in_deck, other_cards_in_deck, unknown_cards_num)

    if set_type == "Full house":
        val3, val2 = set_details["detail_1"], set_details["detail_2"]
        if val3 >= len(value_counts) or val2 >= len(value_counts):
            return 0.0
        needed3 = 3 - value_counts[val3]
        needed2 = 2 - value_counts[val2]
        
        if needed3 + needed2 <= jokers_in_play: return 1.0
        
        cards3_in_deck = max(0, 4 - value_counts[val3])
        cards2_in_deck = max(0, 4 - value_counts[val2])
        jokers_in_deck = total_jokers - known_jokers
        other_cards_in_deck = deck_size + rules.get("blanks", 0) - len(known_cards) + known_jokers - cards3_in_deck - cards2_in_deck
        
        return _calculate_prob_full_house(needed3, needed2, cards3_in_deck, cards2_in_deck, jokers_in_deck, other_cards_in_deck, unknown_cards_num)


    if "Straight flush" in set_type:
        required_values = set(set_details["details"])
        color_to_check = set_details["detail_1"]
        
        required_cards = {(v, color_to_check) for v in required_values}
        
        present_cards = {c for c in known_cards if c in required_cards}
        needed = len(required_cards) - len(present_cards) - jokers_in_play
        if needed <= 0: return 1.0
        
        # Each needed card for a straight flush is unique, so there is only 1 of each in the deck.
        cards_of_value_in_deck = len(required_cards) - len(present_cards)
        jokers_in_deck = total_jokers - known_jokers
        other_cards_in_deck = deck_size + total_blanks - len(known_cards) + known_jokers - cards_of_value_in_deck
        
        return _calculate_prob_simple_set(needed, cards_of_value_in_deck, jokers_in_deck, other_cards_in_deck, unknown_cards_num)

def _calculate_prob_no_cache(action_id, hand, allied_hand, common_hand, unknown_cards_num, rules, for_betting):
    hand_key = _normalize_cards(hand)
    allied_key = _normalize_cards(allied_hand)
    common_key = _normalize_cards(common_hand)
    rules_key = _normalize_rules(rules)
    ctx = _prepare_prob_context(hand_key, allied_key, common_key, rules_key)
    
    return _calculate_prob_with_context(
        int(action_id),
        ctx,
        int(unknown_cards_num),
        bool(for_betting),
    )


def calculate_prob(action_id, hand, allied_hand, common_hand, unknown_cards_num, rules, for_betting):
    return _calculate_prob_no_cache(
        action_id,
        hand,
        allied_hand,
        common_hand,
        unknown_cards_num,
        rules,
        for_betting,
    )

@lru_cache(maxsize=20000)
def _probability_vector_cached(
    hand_key: Tuple[Tuple[int, int], ...],
    allied_key: Tuple[Tuple[int, int], ...],
    common_key: Tuple[Tuple[int, int], ...],
    rules_key: Tuple[int, int, int],
    unknown_cards_num: int,
    for_betting_flag: bool,
) -> Tuple[float, ...]:
    ctx = _prepare_prob_context(hand_key, allied_key, common_key, rules_key)
    game_rules = GameRules(rules_key[0])
    vector = [
        _calculate_prob_with_context(action_id, ctx, unknown_cards_num, for_betting_flag)
        for action_id in range(game_rules.check_action_id)
    ]
    return tuple(vector)


def get_bet_probabilities(game_state, for_betting=False, last_bet=None, specific_action_id=None):
    rules = game_state.get("rules", {})
    game_rules = GameRules(rules.get("deck_size", 24))
    
    players = game_state.get("players", [])
    agent_nickname = game_state["cp_nickname"]

    agent_team = next((p.get("team") for p in players if p.get("nickname") == agent_nickname), None)
    allies = {p.get("nickname") for p in players if p.get("team") == agent_team and p.get("nickname") != agent_nickname} if agent_team is not None else set()

    hands = game_state.get("hands", [])
    hand, allied_hand = [], []
    for h in hands:
        if h.get("nickname") == agent_nickname:
            hand = [(card["value"], card["colour"]) for card in h.get("hand", [])]
        elif h.get("nickname") in allies:
            allied_hand.extend([(card["value"], card["colour"]) for card in h.get("hand", [])])

    # Unseen cards belong only to opponents
    others_card_num = sum(p.get("n_cards", 0) for p in players if p.get("nickname") not in allies and p.get("nickname") != agent_nickname)
    common_hand = [(card["value"], card["colour"]) for card in game_state.get("common_hand", [])]

    hand_key = _normalize_cards(hand)
    allied_key = _normalize_cards(allied_hand)
    common_key = _normalize_cards(common_hand)
    rules_key = _normalize_rules(rules)

    vector = list(
        _probability_vector_cached(
            hand_key,
            allied_key,
            common_key,
            rules_key,
            int(others_card_num),
            bool(for_betting),
        )
    )

    if specific_action_id is not None:
        action_id_int = int(specific_action_id)
        return vector[action_id_int] if 0 <= action_id_int < game_rules.check_action_id else 0.0

    start_action_id = last_bet + 1 if last_bet is not None else 0
    for i in range(min(start_action_id, len(vector))):
        vector[i] = 0.0
    return vector[:game_rules.check_action_id]

def get_generic_bet_probabilities(game_state, last_bet=None):
    rules = game_state.get("rules", {})
    game_rules = GameRules(rules.get("deck_size", 24))

    players = game_state.get("players", [])
    agent_nickname = game_state["cp_nickname"]
    my_card_num = next((p.get("n_cards", 0) for p in players if p.get("nickname") == agent_nickname), 0)
    others_card_num = sum(p.get("n_cards", 0) for p in players if p.get("nickname") != agent_nickname)
    
    common_hand = [(card["value"], card["colour"]) for card in game_state.get("common_hand", [])]
    unknown_cards_num = sum(p.get("n_cards", 0) for p in players)

    hand_key: Tuple[Tuple[int, int], ...] = ()
    allied_key: Tuple[Tuple[int, int], ...] = ()
    common_key = _normalize_cards(common_hand)
    rules_key = _normalize_rules(rules)

    vector = list(
        _probability_vector_cached(
            hand_key,
            allied_key,
            common_key,
            rules_key,
            int(unknown_cards_num),
            True,
        )
    )

    start_action_id = last_bet + 1 if last_bet is not None else 0
    for i in range(min(start_action_id, len(vector))):
        vector[i] = 0.0
    return vector[:game_rules.check_action_id]
