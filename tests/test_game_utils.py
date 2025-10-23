import unittest
import random

from shared.game_utils import GameRules, get_set_details_from_action_id, determine_set_existence


class GameUtilsMappingTest(unittest.TestCase):
    def test_action_details_exist_for_all_actions(self) -> None:
        for deck_size in (24, 32):
            rules = {"deck_size": deck_size}
            rules_obj = GameRules(deck_size)
            for action_id in range(rules_obj.check_action_id):
                details = get_set_details_from_action_id(action_id, deck_size)
                self.assertIsNotNone(details, f"Missing details for action {action_id} deck {deck_size}")
            self.assertEqual(
                rules_obj.num_actions,
                rules_obj.check_action_id + 1,
                f"num_actions mismatch for deck {deck_size}",
            )

    def test_great_straight_union(self) -> None:
        deck_size = 24
        rules = {"deck_size": deck_size}
        rules_obj = GameRules(deck_size)
        target = None
        for action_id in range(rules_obj.check_action_id):
            details = get_set_details_from_action_id(action_id, deck_size)
            if details and details.get("set_type") == "Great straight" and details.get("details") == list(range(6)):
                target = action_id
                break
        self.assertIsNotNone(target, "Could not locate Great straight action id")
        action_id = target

        cards = [
            {"value": 0, "colour": 0},
            {"value": 1, "colour": 1},
            {"value": 2, "colour": 2},
            {"value": 3, "colour": 0},
            {"value": 4, "colour": 1},
            {"value": 5, "colour": 2},
        ]
        self.assertTrue(determine_set_existence(cards, action_id, rules, num_jokers=0))

        missing = cards[:-1]
        self.assertFalse(determine_set_existence(missing, action_id, rules, num_jokers=0))

    def test_straight_flush_examples(self) -> None:
        deck_size = 32
        rules = {"deck_size": deck_size}
        rules_obj = GameRules(deck_size)
        candidates = []
        for action_id in range(rules_obj.check_action_id):
            details = get_set_details_from_action_id(action_id, deck_size)
            if details and details.get("set_type") == "Straight flush":
                candidates.append((action_id, details))
        self.assertGreater(len(candidates), 0)

        action_id, details = random.choice(candidates)
        suit = details["detail_1"]
        values = details["details"]
        cards = [{"value": v, "colour": suit} for v in values]
        self.assertTrue(determine_set_existence(cards, action_id, rules, num_jokers=0))


if __name__ == "__main__":
    unittest.main()
