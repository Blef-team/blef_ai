import math
from itertools import combinations, product
from collections import Counter
from functools import lru_cache

@lru_cache(maxsize=2)
class GameRules:
    """A cached class to hold and calculate game rule details once."""
    def __init__(self, deck_size=24):
        self.deck_size = deck_size
        if deck_size == 24:
            self.vals = 6
            self.straight_types = {
                "Small straight": list(range(5)),
                "Big straight": list(range(1, 6)),
                "Great straight": list(range(6)),
            }
        else:  # deck_size == 32
            self.vals = 8
            self.straight_types = {f"Straight {i+1}": list(range(i, i + 5)) for i in range(4)}
        
        self.flush_straight_types = self.straight_types
        self._calculate_boundaries()
        self.num_actions = self.boundaries["Straight flush"] + 1
        self.check_action_id = self.num_actions - 1

    def _calculate_boundaries(self):
        vals = self.vals
        current_boundary = 0
        self.boundaries = {}
        self.boundaries["High card"] = current_boundary + vals
        current_boundary = self.boundaries["High card"]
        self.boundaries["Pair"] = current_boundary + vals
        current_boundary = self.boundaries["Pair"]
        self.boundaries["Two pairs"] = current_boundary + (vals * (vals - 1)) // 2
        current_boundary = self.boundaries["Two pairs"]
        self.boundaries["Straight"] = current_boundary + len(self.straight_types)
        current_boundary = self.boundaries["Straight"]
        self.boundaries["Three of a kind"] = current_boundary + vals
        current_boundary = self.boundaries["Three of a kind"]
        self.boundaries["Full house"] = current_boundary + (vals * (vals - 1))
        current_boundary = self.boundaries["Full house"]
        self.boundaries["Flush"] = current_boundary + 4
        current_boundary = self.boundaries["Flush"]
        self.boundaries["Four of a kind"] = current_boundary + vals
        current_boundary = self.boundaries["Four of a kind"]
        self.boundaries["Straight flush"] = current_boundary + (len(self.flush_straight_types) * 4)

def get_set_details_from_action_id(action_id, deck_size=24):
    """
    Determines the set type and details from the action_id and deck size.
    This is based on the documentation in the game engine's api/README.md.
    """
    rules = GameRules(deck_size)
    vals = rules.vals
    boundaries = rules.boundaries
    straight_types = rules.straight_types
    flush_straight_types = rules.flush_straight_types
    action_id = int(action_id)
    
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
        if d2 >= d1: d2 += 1
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

def _calculate_prob_straight(needed, cards_in_deck_counts, jokers_in_deck, other_cards_in_deck, unknown_cards_num):
    """
    Calculates the exact probability of completing a straight using the Principle of Inclusion-Exclusion.

    The function computes the probability of the complementary event: failing to acquire at least one
    of the needed card values. This is done by summing the probabilities of failing to get one specific value,
    subtracting the probabilities of failing to get two specific values, adding for three, and so on.
    """
    
    # Total cards in the deck we don't know the location of.
    total_unknown_in_deck = sum(cards_in_deck_counts) + jokers_in_deck + other_cards_in_deck
    
    # The total number of ways to draw the unknown cards. This is our denominator.
    total_outcomes = binom(total_unknown_in_deck, unknown_cards_num)
    if total_outcomes == 0:
        return 0.0

    # This will store the total number of hands that are MISSING AT LEAST ONE of the required cards.
    total_failing_outcomes = 0

    # Loop through the number of card values we might be missing (from 1 up to 'needed').
    # This corresponds to the terms in the Inclusion-Exclusion formula.
    for k in range(1, needed + 1):
        
        # Get all combinations of 'k' card types to exclude.
        # e.g., if we need a 9, 10, J, and k=2, this would be [(9,10), (9,J), (10,J)]
        # We use indices to represent the card types for simplicity.
        for excluded_indices in combinations(range(needed), k):
            
            # Sum the counts of the specific card values we are excluding in this iteration.
            num_cards_to_exclude = sum(cards_in_deck_counts[i] for i in excluded_indices)
            
            # The "bad" pool of cards for this iteration consists of everything EXCEPT the excluded values and jokers.
            pool_of_bad_cards = total_unknown_in_deck - num_cards_to_exclude - jokers_in_deck
            
            # Calculate the number of hands that can be formed using ONLY cards from this "bad" pool.
            # These are the hands that are guaranteed to be missing the 'k' excluded card values.
            num_hands_missing_k_values = binom(pool_of_bad_cards, unknown_cards_num)
            
            # Add or subtract from the total based on the Inclusion-Exclusion principle.
            if (k - 1) % 2 == 0:  # For k=1, 3, 5... we add.
                total_failing_outcomes += num_hands_missing_k_values
            else:  # For k=2, 4, 6... we subtract.
                total_failing_outcomes -= num_hands_missing_k_values

    # The probability of failure is the total failing outcomes divided by all possible outcomes.
    prob_of_failure = total_failing_outcomes / total_outcomes
    
    # The probability of success is 1 minus the probability of failure.
    return 1 - prob_of_failure

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

def _calculate_prob_straight_flush(needed, cards_in_deck, jokers_in_deck, other_cards_in_deck, unknown_cards_num):
    return _calculate_prob_simple_set(needed, len(cards_in_deck), jokers_in_deck, other_cards_in_deck, unknown_cards_num)


def calculate_prob(action_id, hand, common_hand, unknown_cards_num, rules, for_betting):
    set_details = get_set_details_from_action_id(action_id, rules.get("deck_size", 24))
    if not set_details: return 0.0

    known_cards = hand + common_hand
    known_cards_counter = Counter(c[0] for c in known_cards)
    known_cards_by_color = Counter(c[1] for c in known_cards)
    known_jokers = known_cards_counter.get(-1, 0)
    
    my_jokers = Counter(c[0] for c in hand).get(-1, 0)
    common_jokers = Counter(c[0] for c in common_hand).get(-1, 0)

    jokers_in_play = my_jokers + common_jokers if for_betting else common_jokers
    
    deck_size = rules.get("deck_size", 24)
    total_jokers = rules.get("jokers", 0)
    
    set_type = set_details.get("set_type")

    if set_type in ["High card", "Pair", "Three of a kind", "Four of a kind"]:
        value_to_check = set_details["detail_1"]
        needed_map = {"High card": 1, "Pair": 2, "Three of a kind": 3, "Four of a kind": 4}
        needed = needed_map[set_type] - known_cards_counter.get(value_to_check, 0) - jokers_in_play
        if needed <= 0: return 1.0

        cards_of_value_in_deck = 4 - known_cards_counter.get(value_to_check, 0)
        jokers_in_deck = total_jokers - known_jokers
        other_cards_in_deck = deck_size + rules.get("blanks", 0) - len(known_cards) + known_jokers - cards_of_value_in_deck
        
        return _calculate_prob_simple_set(needed, cards_of_value_in_deck, jokers_in_deck, other_cards_in_deck, unknown_cards_num)
        
    if set_type == "Two pairs":
        val1, val2 = set_details["detail_1"], set_details["detail_2"]
        needed1 = 2 - known_cards_counter.get(val1, 0)
        needed2 = 2 - known_cards_counter.get(val2, 0)
        
        if needed1 + needed2 <= jokers_in_play: return 1.0
        
        cards1_in_deck = 4 - known_cards_counter.get(val1, 0)
        cards2_in_deck = 4 - known_cards_counter.get(val2, 0)
        jokers_in_deck = total_jokers - known_jokers
        other_cards_in_deck = deck_size + rules.get("blanks", 0) - len(known_cards) + known_jokers - cards1_in_deck - cards2_in_deck
        
        return _calculate_prob_two_pairs(needed1, needed2, cards1_in_deck, cards2_in_deck, jokers_in_deck, other_cards_in_deck, unknown_cards_num)

    if "Straight" in set_type and "flush" not in set_type:
        required_values = set(set_details["details"])
        
        present_values = {c[0] for c in known_cards if c[0] in required_values}
        needed = len(required_values) - len(present_values) - jokers_in_play
        if needed <= 0: return 1.0
        
        cards_in_deck_counts = [4 - known_cards_counter.get(v, 0) for v in required_values if v not in present_values]
        jokers_in_deck = total_jokers - known_jokers
        other_cards_in_deck = deck_size + rules.get("blanks", 0) - len(known_cards) + known_jokers - sum(cards_in_deck_counts)
        
        return _calculate_prob_straight(needed, cards_in_deck_counts, jokers_in_deck, other_cards_in_deck, unknown_cards_num)

    if set_type == "Flush":
        color_to_check = set_details["detail_1"]
        needed = 5 - known_cards_by_color.get(color_to_check, 0) - jokers_in_play
        if needed <= 0: return 1.0
        
        cards_in_deck = (deck_size // 4) - known_cards_by_color.get(color_to_check, 0)
        jokers_in_deck = total_jokers - known_jokers
        other_cards_in_deck = deck_size + rules.get("blanks", 0) - len(known_cards) + known_jokers - cards_in_deck
        
        return _calculate_prob_simple_set(needed, cards_in_deck, jokers_in_deck, other_cards_in_deck, unknown_cards_num)

    if set_type == "Full house":
        val3, val2 = set_details["detail_1"], set_details["detail_2"]
        needed3 = 3 - known_cards_counter.get(val3, 0)
        needed2 = 2 - known_cards_counter.get(val2, 0)
        
        if needed3 + needed2 <= jokers_in_play: return 1.0
        
        cards3_in_deck = 4 - known_cards_counter.get(val3, 0)
        cards2_in_deck = 4 - known_cards_counter.get(val2, 0)
        jokers_in_deck = total_jokers - known_jokers
        other_cards_in_deck = deck_size + rules.get("blanks", 0) - len(known_cards) + known_jokers - cards3_in_deck - cards2_in_deck
        
        return _calculate_prob_full_house(needed3, needed2, cards3_in_deck, cards2_in_deck, jokers_in_deck, other_cards_in_deck, unknown_cards_num)


    if "Straight flush" in set_type:
        required_values = set(set_details["details"])
        color_to_check = set_details["detail_1"]
        
        present_cards = {c for c in known_cards if c[0] in required_values and c[1] == color_to_check}
        needed = len(required_values) - len(present_cards) - jokers_in_play
        if needed <= 0: return 1.0
        
        cards_in_deck = [c for v in required_values for c in product([v], [color_to_check]) if c not in present_cards]
        jokers_in_deck = total_jokers - known_jokers
        other_cards_in_deck = deck_size + rules.get("blanks", 0) - len(known_cards) + known_jokers - len(cards_in_deck)
        
        return _calculate_prob_straight_flush(needed, cards_in_deck, jokers_in_deck, other_cards_in_deck, unknown_cards_num)

    return 0.0

def get_bet_probabilities(game_state, for_betting=False, last_bet=None, specific_action_id=None):
    rules = game_state.get("rules", {})
    game_rules = GameRules(rules.get("deck_size", 24))
    
    players = game_state.get("players", [])
    agent_nickname = game_state["cp_nickname"]

    matching_hands = [h for h in game_state.get("hands", []) if h.get("nickname") == agent_nickname]
    if not matching_hands:
        return 0.0 if specific_action_id is not None else [0.0] * (game_rules.num_actions - 1)
    hand = [(card["value"], card["colour"]) for card in matching_hands[0]["hand"]]

    others_card_num = sum(p.get("n_cards", 0) for p in players if p.get("nickname") != agent_nickname)
    common_hand = [(card["value"], card["colour"]) for card in game_state.get("common_hand", [])]
    
    # If a specific action is requested, calculate and return only that probability
    if specific_action_id is not None:
        return calculate_prob(specific_action_id, hand, common_hand, others_card_num, rules, for_betting)

    # Otherwise, calculate the vector for legal betting moves
    start_action_id = last_bet + 1 if last_bet is not None else 0
    
    bet_probs = [0.0] * start_action_id
    for action_id in range(start_action_id, game_rules.check_action_id):
        prob = calculate_prob(action_id, hand, common_hand, others_card_num, rules, for_betting)
        bet_probs.append(prob)
        
    return bet_probs

def get_generic_bet_probabilities(game_state, last_bet=None):
    rules = game_state.get("rules", {})
    game_rules = GameRules(rules.get("deck_size", 24))

    players = game_state.get("players", [])
    agent_nickname = game_state["cp_nickname"]
    my_card_num = next((p.get("n_cards", 0) for p in players if p.get("nickname") == agent_nickname), 0)
    others_card_num = sum(p.get("n_cards", 0) for p in players if p.get("nickname") != agent_nickname)
    
    common_hand = [(card["value"], card["colour"]) for card in game_state.get("common_hand", [])]
    unknown_cards_num = my_card_num + others_card_num
    
    start_action_id = last_bet + 1 if last_bet is not None else 0

    bet_probs = [0.0] * start_action_id
    for action_id in range(start_action_id, game_rules.check_action_id):
        prob = calculate_prob(action_id, [], common_hand, unknown_cards_num, rules, for_betting=True)
        bet_probs.append(prob)
        
    return bet_probs
