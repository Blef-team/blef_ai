import unittest

from shared.api.simpleschema_local_manager import create_game, determine_set_existence, play
from shared.game_utils import GameRules


class SimpleschemaManagerDeckSizeTest(unittest.TestCase):
    def test_create_game_32_deck(self):
        game = create_game(2, deck_size=32)
        self.assertEqual(game["rules"]["deck_size"], 32)
        self.assertEqual(len(game["hands"]), 2)
        self.assertEqual(len(game["hands"][0]["hand"]), 1)
        self.assertEqual(game["common_hand"], [])

    def test_create_game_with_common_cards(self):
        game = create_game(3, deck_size=24, common_cards=3)
        self.assertEqual(game["rules"]["common_cards"], 3)
        self.assertEqual(len(game["common_hand"]), 3)
        board_ids = set((card["value"], card["colour"]) for card in game["common_hand"])
        for hand in game["hands"]:
            for card in hand["hand"]:
                self.assertNotIn((card["value"], card["colour"]), board_ids)

    def test_determine_set_existence_straight_flush_32(self):
        rules = {"deck_size": 32}
        gr = GameRules(32)
        straight_flush_action = gr.boundaries["Four of a kind"]  # first straight flush entry
        cards = [{"value": v, "colour": 0} for v in range(5)]
        self.assertTrue(determine_set_existence(cards, straight_flush_action, rules, num_jokers=0))
        missing_card = cards[:-1]
        self.assertFalse(determine_set_existence(missing_card, straight_flush_action, rules, num_jokers=0))

    def test_play_allows_dynamic_check_action(self):
        game = create_game(2, deck_size=32)
        # initial bet (lowest high card)
        play(game, 0)
        # check using dynamic check id
        check_action_id = GameRules(32).check_action_id
        try:
            play(game, check_action_id)
        except Exception as exc:
            self.fail(f"play raised an unexpected exception for 32-card CHECK action: {exc}")


if __name__ == "__main__":
    unittest.main()
