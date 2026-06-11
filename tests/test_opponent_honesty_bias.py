"""Unit tests for mechanism E: opponent honesty bias.

Two layers:
  - ``resample_to_honest``: the pure resampler given an action, a legal mask,
    a hand, and rules. No env coupling. Most tests live here.
  - ``maybe_apply_opponent_honesty_bias``: trainer-side hook that pulls
    env / spec state. One end-to-end test to confirm the wiring.
"""
import random
import unittest

import numpy as np

from nfsp_ai.personality_train import (
    PersonalityTrainSpec, _bet_is_truthful, resample_to_honest,
    resample_to_aggressive, resample_to_passive,
    maybe_apply_opponent_honesty_bias, spec_from_dict,
)


def _legal_all(check_id):
    """All bets + CHECK legal."""
    m = np.ones(check_id + 1, dtype=np.float32)
    return m


class TruthfulnessOracleTest(unittest.TestCase):
    """``_bet_is_truthful`` mirrors determine_set_existence; just a thin
    sanity check. The real coverage lives in shared.game_utils tests.

    24-deck action layout (verified):
        High card: aid 0..5 -> detail_1 in {0..5} (the 6 ranks).
        Pair:      aid 6..11 -> detail_1 in {0..5}.
        ...
        check_id = 88.
    """

    def test_high_card_present(self):
        # actor holds a 4 -> "High card 4" (id 4) is truthful.
        hand = [{"value": 4, "colour": 0}]
        self.assertTrue(_bet_is_truthful(4, hand, [], deck_size=24, num_jokers=0))

    def test_high_card_absent(self):
        # actor holds a 4 -> "High card 0" (id 0) is a bluff.
        hand = [{"value": 4, "colour": 0}]
        self.assertFalse(_bet_is_truthful(0, hand, [], deck_size=24, num_jokers=0))

    def test_pair_absent(self):
        # actor holds one 4 -> "Pair of 4s" (id 10) requires 2 fours.
        hand = [{"value": 4, "colour": 0}]
        self.assertFalse(_bet_is_truthful(10, hand, [], deck_size=24, num_jokers=0))

    def test_joker_lifts_pair(self):
        # Same hand + 1 joker -> Pair of 4s becomes reachable.
        hand = [{"value": 4, "colour": 0}]
        self.assertTrue(_bet_is_truthful(10, hand, [], deck_size=24, num_jokers=1))


class ResamplerTest(unittest.TestCase):

    def test_bias_zero_is_passthrough(self):
        hand = [{"value": 4, "colour": 0}]  # actor cannot truthfully bet a pair
        mask = _legal_all(check_id=88)
        out = resample_to_honest(
            intended_action=10,        # Pair of 4s — bluff
            legal_mask=mask, actor_hand=hand, common_hand=[],
            deck_size=24, num_jokers=0, check_action_id=88,
            honesty_bias=0.0, rng=random.Random(0),
        )
        self.assertEqual(out, 10)

    def test_already_truthful_passthrough(self):
        hand = [{"value": 4, "colour": 0}]
        mask = _legal_all(check_id=88)
        out = resample_to_honest(
            intended_action=4,           # High card 4 — truthful
            legal_mask=mask, actor_hand=hand, common_hand=[],
            deck_size=24, num_jokers=0, check_action_id=88,
            honesty_bias=1.0, rng=random.Random(0),
        )
        self.assertEqual(out, 4)

    def test_check_never_resampled(self):
        hand = []   # totally empty hand, every bet is a bluff
        mask = _legal_all(check_id=88)
        out = resample_to_honest(
            intended_action=88,          # CHECK
            legal_mask=mask, actor_hand=hand, common_hand=[],
            deck_size=24, num_jokers=0, check_action_id=88,
            honesty_bias=1.0, rng=random.Random(0),
        )
        self.assertEqual(out, 88)

    def test_bluff_replaced_with_truthful_at_full_bias(self):
        # Hand holds values 4 + 5 (different colours). High card 4 (id 4) and
        # High card 5 (id 5) are the only truthful bets — no pair (one of
        # each rank), no two pairs, etc. Intended action is a bluff pair;
        # at full bias the resampler must pick one of {4, 5}.
        hand = [{"value": 4, "colour": 0}, {"value": 5, "colour": 1}]
        mask = _legal_all(check_id=88)
        seen = set()
        rng = random.Random(42)
        for _ in range(50):
            out = resample_to_honest(
                intended_action=10,      # Pair of 4s (bluff)
                legal_mask=mask, actor_hand=hand, common_hand=[],
                deck_size=24, num_jokers=0, check_action_id=88,
                honesty_bias=1.0, rng=rng,
            )
            seen.add(out)
        self.assertTrue(seen.issubset({4, 5}), msg=f"unexpected ids: {seen}")
        self.assertGreater(len(seen), 0)

    def test_no_truthful_alternative_keeps_intended(self):
        # Actor's hand is empty -> nothing is truthful. Bluffs stay bluffs.
        hand = []
        mask = _legal_all(check_id=88)
        out = resample_to_honest(
            intended_action=12,
            legal_mask=mask, actor_hand=hand, common_hand=[],
            deck_size=24, num_jokers=0, check_action_id=88,
            honesty_bias=1.0, rng=random.Random(0),
        )
        self.assertEqual(out, 12)

    def test_partial_bias_split(self):
        # With 0.5 bias on a bluff that has a truthful alternative, roughly
        # half should stay, half should resample. Wide CI on n=400.
        hand = [{"value": 4, "colour": 0}]   # only "High card 4" (id 4) truthful
        mask = _legal_all(check_id=88)
        rng = random.Random(7)
        kept = 0
        n = 400
        for _ in range(n):
            out = resample_to_honest(
                intended_action=10,          # Pair of 4s — bluff
                legal_mask=mask, actor_hand=hand, common_hand=[],
                deck_size=24, num_jokers=0, check_action_id=88,
                honesty_bias=0.5, rng=rng,
            )
            if out == 10:
                kept += 1
        frac = kept / n
        self.assertGreater(frac, 0.25, msg=f"kept frac {frac:.2f}")
        self.assertLess(frac, 0.75, msg=f"kept frac {frac:.2f}")

    def test_respects_legal_mask(self):
        # Truthful set is {4, 5}, but if we forbid id 5 only id 4 should
        # appear in resampled output.
        hand = [{"value": 4, "colour": 0}, {"value": 5, "colour": 1}]
        mask = _legal_all(check_id=88)
        mask[5] = 0.0
        rng = random.Random(0)
        seen = set()
        for _ in range(50):
            out = resample_to_honest(
                intended_action=10,
                legal_mask=mask, actor_hand=hand, common_hand=[],
                deck_size=24, num_jokers=0, check_action_id=88,
                honesty_bias=1.0, rng=rng,
            )
            seen.add(out)
        self.assertEqual(seen, {4})


class HookTest(unittest.TestCase):
    """End-to-end through the trainer hook with a stub env."""

    class _StubEnv:
        def __init__(self, ref_nick, cp_nick, hand, deck_spec):
            self._ref_nick = ref_nick
            self.game = {
                "cp_nickname": cp_nick,
                "hands": [{"nickname": cp_nick, "hand": hand}],
                "common_hand": [],
                "rules": {"deck_size": 24, "jokers": 0, "blanks": 0, "common_cards": 0},
            }
            self.deck_spec = deck_spec

    class _StubDeckSpec:
        def __init__(self, check_id=88, personality_spec=None):
            self.check_action_id = check_id
            self.personality_spec = personality_spec

    def test_learner_action_not_resampled(self):
        # Even with honesty bias = 1.0, the learner's own action stays.
        spec = PersonalityTrainSpec(name="x", opponent_honesty_bias=1.0)
        deck = self._StubDeckSpec(personality_spec=spec)
        env = self._StubEnv(ref_nick="L", cp_nick="L", hand=[], deck_spec=deck)
        mask = _legal_all(check_id=88)
        out = maybe_apply_opponent_honesty_bias(env, mask, 17, random)
        self.assertEqual(out, 17)

    def test_opponent_bluff_resampled(self):
        spec = PersonalityTrainSpec(name="x", opponent_honesty_bias=1.0)
        deck = self._StubDeckSpec(personality_spec=spec)
        env = self._StubEnv(
            ref_nick="L", cp_nick="O",
            hand=[{"value": 4, "colour": 0}], deck_spec=deck,
        )
        mask = _legal_all(check_id=88)
        # Intended is "Pair of 4s" (id 10) — bluff. With bias=1.0 the resampler
        # must replace it with the only truthful action {4 = High card 4}.
        out = maybe_apply_opponent_honesty_bias(env, mask, 10, random.Random(0))
        self.assertEqual(out, 4)

    def test_no_spec_passthrough(self):
        deck = self._StubDeckSpec(personality_spec=None)
        env = self._StubEnv(ref_nick="L", cp_nick="O", hand=[], deck_spec=deck)
        mask = _legal_all(check_id=88)
        out = maybe_apply_opponent_honesty_bias(env, mask, 5, random)
        self.assertEqual(out, 5)


class AggressionResamplerTest(unittest.TestCase):
    """Quartile-sample redesign (2026-06-09). At full bias, output lands
    uniformly in the top quartile of legal bets — not a single point —
    preserving the diversity the trained policy needs to generalise."""

    def test_zero_bias_passthrough(self):
        mask = _legal_all(check_id=88)
        out = resample_to_aggressive(
            intended_action=10, legal_mask=mask, check_action_id=88,
            aggression_bias=0.0, rng=random.Random(0),
        )
        self.assertEqual(out, 10)

    def test_full_bias_stays_in_top_quartile(self):
        # 88 legal bets [0..87]; top quartile cutoff = floor(3*88/4) = 66.
        # All resampled actions must land in [66, 87].
        mask = _legal_all(check_id=88)
        seen = set()
        rng = random.Random(0)
        for _ in range(400):
            out = resample_to_aggressive(
                intended_action=10, legal_mask=mask, check_action_id=88,
                aggression_bias=1.0, rng=rng,
            )
            seen.add(out)
        # All outputs in the upper quartile.
        self.assertTrue(seen.issubset(set(range(66, 88))), msg=f"out: {seen}")
        # And there's genuine spread (not a single point).
        self.assertGreaterEqual(len(seen), 5,
            msg=f"quartile sampling collapsed to {seen}")

    def test_full_bias_partial_legal(self):
        # Legal bets [10..29] (20 bets). Top quartile cutoff = 15 -> [25..29]
        # (positions 15..19 in `legal` list, which are bets 25..29).
        mask = np.zeros(89, dtype=np.float32)
        for aid in range(10, 30):
            mask[aid] = 1.0
        seen = set()
        rng = random.Random(7)
        for _ in range(200):
            out = resample_to_aggressive(
                intended_action=10, legal_mask=mask, check_action_id=88,
                aggression_bias=1.0, rng=rng,
            )
            seen.add(out)
        self.assertTrue(seen.issubset({25, 26, 27, 28, 29}), msg=f"out: {seen}")
        self.assertGreaterEqual(len(seen), 3)

    def test_check_never_overridden(self):
        mask = _legal_all(check_id=88)
        out = resample_to_aggressive(
            intended_action=88, legal_mask=mask, check_action_id=88,
            aggression_bias=1.0, rng=random.Random(0),
        )
        self.assertEqual(out, 88)

    def test_no_legal_bets_passthrough(self):
        # Only CHECK is legal.
        mask = np.zeros(89, dtype=np.float32); mask[88] = 1.0
        out = resample_to_aggressive(
            intended_action=10, legal_mask=mask, check_action_id=88,
            aggression_bias=1.0, rng=random.Random(0),
        )
        self.assertEqual(out, 10)

    def test_single_legal_bet(self):
        # Only one bet (id 5) and CHECK legal. Quartile collapses to the
        # one bet.
        mask = np.zeros(89, dtype=np.float32); mask[5] = 1.0; mask[88] = 1.0
        out = resample_to_aggressive(
            intended_action=5, legal_mask=mask, check_action_id=88,
            aggression_bias=1.0, rng=random.Random(0),
        )
        # The only top-quartile pick is the single legal bet itself.
        self.assertEqual(out, 5)


class PassivityResamplerTest(unittest.TestCase):
    """Quartile-sample redesign (2026-06-09). At full bias, 50/50 split
    between CHECK (when legal) and a uniform pick from the bottom quartile
    of legal bets."""

    def test_zero_bias_passthrough(self):
        mask = _legal_all(check_id=88)
        out = resample_to_passive(
            intended_action=20, legal_mask=mask, check_action_id=88,
            passivity_bias=0.0, rng=random.Random(0),
        )
        self.assertEqual(out, 20)

    def test_full_bias_50_50_check_and_bottom_quartile(self):
        # 88 legal bets [0..87] + CHECK. Bottom quartile bets are [0..21]
        # (ceil(88/4) = 22 entries). At full bias we should see ~50% CHECK
        # and ~50% bottom-quartile bets.
        mask = _legal_all(check_id=88)
        rng = random.Random(0)
        n = 800
        n_check = 0
        n_other = 0
        bet_seen = set()
        for _ in range(n):
            out = resample_to_passive(
                intended_action=50, legal_mask=mask, check_action_id=88,
                passivity_bias=1.0, rng=rng,
            )
            if out == 88:
                n_check += 1
            else:
                n_other += 1
                bet_seen.add(out)
        # CHECK fraction roughly 50%.
        frac = n_check / n
        self.assertGreater(frac, 0.40, msg=f"CHECK frac {frac:.2f}")
        self.assertLess(frac, 0.60, msg=f"CHECK frac {frac:.2f}")
        # All non-CHECK outputs in bottom quartile.
        self.assertTrue(bet_seen.issubset(set(range(0, 22))),
            msg=f"non-quartile: {bet_seen}")
        # And spread across that quartile.
        self.assertGreaterEqual(len(bet_seen), 5)

    def test_check_illegal_picks_bottom_quartile_only(self):
        # CHECK illegal; only bets [5..29] legal (25 bets). Bottom quartile
        # = first ceil(25/4) = 7 -> bets {5,6,7,8,9,10,11}.
        mask = np.zeros(89, dtype=np.float32)
        for aid in range(5, 30):
            mask[aid] = 1.0
        rng = random.Random(11)
        seen = set()
        n_check = 0
        for _ in range(200):
            out = resample_to_passive(
                intended_action=25, legal_mask=mask, check_action_id=88,
                passivity_bias=1.0, rng=rng,
            )
            if out == 88:
                n_check += 1
            else:
                seen.add(out)
        self.assertEqual(n_check, 0, msg="CHECK was illegal; never pick it")
        self.assertTrue(seen.issubset(set(range(5, 12))), msg=f"out: {seen}")

    def test_already_check_unchanged(self):
        mask = _legal_all(check_id=88)
        out = resample_to_passive(
            intended_action=88, legal_mask=mask, check_action_id=88,
            passivity_bias=1.0, rng=random.Random(0),
        )
        self.assertEqual(out, 88)

    def test_partial_bias_split(self):
        # At 0.5 bias: ~50% stay at 20, ~50% resample. Wide CI.
        mask = _legal_all(check_id=88)
        kept = 0; n = 400
        rng = random.Random(13)
        for _ in range(n):
            out = resample_to_passive(
                intended_action=20, legal_mask=mask, check_action_id=88,
                passivity_bias=0.5, rng=rng,
            )
            if out == 20:
                kept += 1
        frac = kept / n
        self.assertGreater(frac, 0.30)
        self.assertLess(frac, 0.70)


class SpecValidationTest(unittest.TestCase):

    def test_at_most_one_e_knob(self):
        with self.assertRaises(ValueError):
            spec_from_dict({
                "name": "x",
                "opponent_honesty_bias": 0.5,
                "opponent_aggression_bias": 0.5,
            })
        # OK if only one is set.
        spec_from_dict({"name": "x", "opponent_honesty_bias": 0.5})
        spec_from_dict({"name": "x", "opponent_aggression_bias": 0.5})
        spec_from_dict({"name": "x", "opponent_passivity_bias": 0.5})
        # All zero is fine.
        spec_from_dict({"name": "x"})

    def test_bias_range_validated(self):
        for k in ("opponent_honesty_bias", "opponent_aggression_bias",
                  "opponent_passivity_bias"):
            with self.assertRaises(ValueError, msg=k):
                spec_from_dict({"name": "x", k: 1.5})
            with self.assertRaises(ValueError, msg=k):
                spec_from_dict({"name": "x", k: -0.1})


class HookDispatchTest(unittest.TestCase):
    """Verify the trainer-side hook routes to the right resampler."""

    class _StubEnv:
        def __init__(self, ref_nick, cp_nick, deck_spec):
            self._ref_nick = ref_nick
            self.game = {
                "cp_nickname": cp_nick,
                "hands": [{"nickname": cp_nick, "hand": []}],
                "common_hand": [],
                "rules": {"deck_size": 24, "jokers": 0, "blanks": 0, "common_cards": 0},
            }
            self.deck_spec = deck_spec

    class _StubDeckSpec:
        def __init__(self, spec):
            self.check_action_id = 88
            self.personality_spec = spec

    def test_aggression_routes_to_top_quartile(self):
        spec = PersonalityTrainSpec(name="x", opponent_aggression_bias=1.0)
        env = self._StubEnv("L", "O", self._StubDeckSpec(spec))
        mask = _legal_all(check_id=88)
        # 88 legal bets; top quartile = [66..87]. Output must land there.
        out = maybe_apply_opponent_honesty_bias(env, mask, 10, random.Random(0))
        self.assertGreaterEqual(out, 66)
        self.assertLessEqual(out, 87)

    def test_passivity_routes_to_check_or_bottom_quartile(self):
        # Post-quartile-redesign: at full bias, passivity 50/50s between
        # CHECK and a uniform pick from the bottom quartile of legal bets.
        # Run many trials, confirm both CHECK and at least one bottom-
        # quartile bet appear in the output set, and nothing outside.
        spec = PersonalityTrainSpec(name="x", opponent_passivity_bias=1.0)
        env = self._StubEnv("L", "O", self._StubDeckSpec(spec))
        mask = _legal_all(check_id=88)
        seen = set()
        rng = random.Random(0)
        for _ in range(200):
            out = maybe_apply_opponent_honesty_bias(env, mask, 50, rng)
            seen.add(out)
        # Allowed: CHECK (88) or bottom-quartile bets [0..21].
        allowed = set(range(0, 22)) | {88}
        self.assertTrue(seen.issubset(allowed), msg=f"out: {seen}")
        # Both modes should appear.
        self.assertIn(88, seen, msg="CHECK never picked")
        self.assertTrue(
            any(0 <= a < 22 for a in seen),
            msg="bottom-quartile bet never picked")

    def test_learner_action_never_modified_by_any_knob(self):
        for field_name in ("opponent_honesty_bias",
                           "opponent_aggression_bias",
                           "opponent_passivity_bias"):
            spec = PersonalityTrainSpec(name="x", **{field_name: 1.0})
            env = self._StubEnv("L", "L", self._StubDeckSpec(spec))
            mask = _legal_all(check_id=88)
            out = maybe_apply_opponent_honesty_bias(env, mask, 10, random.Random(0))
            self.assertEqual(out, 10, msg=f"{field_name} modified learner")


if __name__ == "__main__":
    unittest.main()
