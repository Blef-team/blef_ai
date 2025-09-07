from functools import lru_cache
from itertools import combinations


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
