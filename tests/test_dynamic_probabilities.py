import unittest

from itertools import combinations
from typing import List, Tuple

from shared.game_utils import GameRules
from shared.probabilities.dynamic_probabilities import get_bet_probabilities

Card = Tuple[int, int]


class DynamicProbabilitiesTest(unittest.TestCase):
    def _make_state(self, deck_size: int, hand: List[Card], other_counts: List[int]):
        rules = {"deck_size": deck_size, "jokers": 0, "blanks": 0}
        players = [{"nickname": "0", "n_cards": len(hand)}]
        for i, cnt in enumerate(other_counts, start=1):
            players.append({"nickname": str(i), "n_cards": cnt})
        hands = [{"nickname": "0", "hand": [{"value": v, "colour": c} for v, c in hand]}]
        return {
            "rules": rules,
            "players": players,
            "hands": hands,
            "common_hand": [],
            "cp_nickname": "0",
        }

    def _get_probs(self, deck_size: int, hand: List[Card], others: List[int], for_betting: bool = True):
        state = self._make_state(deck_size, hand, others)
        return get_bet_probabilities(state, for_betting=for_betting)

    def test_probabilities_24_deck_vector_length(self):
        hand = [(0, 0)]
        others = [2]
        state = self._make_state(24, hand, others)
        probs = get_bet_probabilities(state, for_betting=True)
        expected_length = GameRules(24).num_actions - 1
        self.assertEqual(len(probs), expected_length)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_probabilities_32_deck_vector_length(self):
        hand = [(0, 0)]
        others = [2]
        state = self._make_state(32, hand, others)
        probs = get_bet_probabilities(state, for_betting=True)
        expected_length = GameRules(32).num_actions - 1
        self.assertEqual(len(probs), expected_length)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_pair_probability_matches_bruteforce_24(self):
        deck_size = 24
        hand = [(0, 0)]
        others = [1]
        probs = self._get_probs(deck_size, hand, others)
        rules = GameRules(deck_size)
        pair_action = rules.boundaries["High card"] + 0  # lowest rank pair
        computed = probs[pair_action]

        brute = self._brute_pair_probability(deck_size, hand, others, value=0)
        self.assertAlmostEqual(computed, brute, places=10)

    def test_pair_probability_matches_bruteforce_32(self):
        deck_size = 32
        hand = [(7, 0)]  # highest rank
        others = [1]
        probs = self._get_probs(deck_size, hand, others)
        rules = GameRules(deck_size)
        pair_action = rules.boundaries["High card"] + 7
        computed = probs[pair_action]

        brute = self._brute_pair_probability(deck_size, hand, others, value=7)
        self.assertAlmostEqual(computed, brute, places=10)

    def test_two_pairs_with_excess_rank_does_not_short_circuit(self):
        # Regression: with three 0s and one 1 already visible, the prob of
        # two pairs (1, 0) needs one more rank-1 card. Pre-fix, needed1=1 and
        # needed2=-1 cancelled to 0 and triggered the early-return at 1.0.
        deck_size = 24
        hand = [(0, 0), (0, 1), (0, 2), (1, 0)]
        others = [1]
        probs = self._get_probs(deck_size, hand, others)
        rules = GameRules(deck_size)
        two_pairs_action = rules.boundaries["Pair"] + 0  # Two pairs of (1, 0)

        computed = probs[two_pairs_action]
        brute = self._brute_two_pairs_probability(deck_size, hand, others, val1=1, val2=0)
        self.assertAlmostEqual(computed, brute, places=10)
        self.assertLess(computed, 1.0)

    def test_full_house_with_excess_triple_rank_does_not_short_circuit(self):
        # Regression: with four 0s and one 1 already visible, the prob of a
        # full house (three 0s, pair of 1s) needs one more rank-1 card. Pre-fix,
        # needed3=-1 and needed2=1 cancelled to 0 and triggered the early-return at 1.0.
        deck_size = 24
        hand = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
        others = [1]
        probs = self._get_probs(deck_size, hand, others)
        rules = GameRules(deck_size)
        full_house_action = rules.boundaries["Three of a kind"] + 0  # Full house (0, 1)

        computed = probs[full_house_action]
        brute = self._brute_full_house_probability(deck_size, hand, others, val3=0, val2=1)
        self.assertAlmostEqual(computed, brute, places=10)
        self.assertLess(computed, 1.0)

    def test_flush_probability_matches_bruteforce(self):
        for deck_size in (24, 32):
            hand = [(0, 0)]
            others = [4]
            probs = self._get_probs(deck_size, hand, others)
            rules = GameRules(deck_size)
            flush_action = rules.boundaries["Full house"] + 0  # Flush, clubs
            computed = probs[flush_action]
            brute = self._brute_flush_probability(deck_size, hand, others, suit=0)
            self.assertAlmostEqual(computed, brute, places=10)

    # ---- brute-force helpers -------------------------------------------------
    def _remaining_deck(self, deck_size: int, known_cards: List[Card]) -> List[Card]:
        num_values = deck_size // 4
        deck = [(v, c) for v in range(num_values) for c in range(4)]
        for card in known_cards:
            deck.remove(card)
        return deck

    def _brute_pair_probability(self, deck_size: int, hand: List[Card], others: List[int], value: int) -> float:
        known = list(hand)
        remaining = self._remaining_deck(deck_size, known)
        draws = sum(others)
        successes = 0
        total = 0
        for combo in combinations(remaining, draws):
            total += 1
            all_cards = known + list(combo)
            count = sum(1 for v, _ in all_cards if v == value)
            if count >= 2:
                successes += 1
        return successes / total if total else 0.0

    def _brute_two_pairs_probability(self, deck_size: int, hand: List[Card], others: List[int], val1: int, val2: int) -> float:
        known = list(hand)
        remaining = self._remaining_deck(deck_size, known)
        draws = sum(others)
        successes = 0
        total = 0
        for combo in combinations(remaining, draws):
            total += 1
            all_cards = known + list(combo)
            count1 = sum(1 for v, _ in all_cards if v == val1)
            count2 = sum(1 for v, _ in all_cards if v == val2)
            if count1 >= 2 and count2 >= 2:
                successes += 1
        return successes / total if total else 0.0

    def _brute_full_house_probability(self, deck_size: int, hand: List[Card], others: List[int], val3: int, val2: int) -> float:
        known = list(hand)
        remaining = self._remaining_deck(deck_size, known)
        draws = sum(others)
        successes = 0
        total = 0
        for combo in combinations(remaining, draws):
            total += 1
            all_cards = known + list(combo)
            count3 = sum(1 for v, _ in all_cards if v == val3)
            count2 = sum(1 for v, _ in all_cards if v == val2)
            if count3 >= 3 and count2 >= 2:
                successes += 1
        return successes / total if total else 0.0

    def _brute_flush_probability(self, deck_size: int, hand: List[Card], others: List[int], suit: int) -> float:
        known = list(hand)
        draws = sum(others)
        remaining = self._remaining_deck(deck_size, known)
        successes = 0
        total = 0
        for combo in combinations(remaining, draws):
            total += 1
            all_cards = known + list(combo)
            suit_count = sum(1 for _, s in all_cards if s == suit)
            if suit_count >= 5:
                successes += 1
        return successes / total if total else 0.0


if __name__ == "__main__":
    unittest.main()
