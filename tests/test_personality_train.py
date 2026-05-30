"""Unit tests for nfsp_ai.personality_train.

Covers:
  - PersonalityTrainSpec JSON round-trip and validation
  - layout-block resolver: disjoint + contiguous + matches DeckSpec.obs_dim
  - apply_personality_to_obs: zeroes exactly the named slice
  - apply_personality_to_mask: safety rail always leaves ≥1 legal action
  - compute_personality_terminal_scale: win/loss multipliers + bluff oracle
    + hand-type bias
"""

import json
import os
import tempfile
import unittest

import numpy as np
import torch

from nfsp_ai.nfsp_run_local import _build_deck_spec
from nfsp_ai.personality_train import (
    PersonalityTrainSpec,
    apply_personality_to_mask,
    apply_personality_to_obs,
    compute_personality_terminal_scale,
    load_spec,
    resolve_obs_block_slices,
    spec_from_dict,
    spec_to_dict,
)


class TestSpecRoundTrip(unittest.TestCase):
    def test_dict_round_trip_preserves_fields(self):
        s = PersonalityTrainSpec(
            name="testbot",
            win_multiplier=1.5,
            loss_multiplier=2.0,
            bluff_caught_penalty=0.3,
            hand_type_bias={"Pair": -0.4, "Flush": 0.2},
            forbid_hand_types=("Pair",),
            forbid_check_unless_only=True,
            obs_blind_spots=("history", "common_jokers"),
        )
        d = spec_to_dict(s)
        # tuples become lists in dict form (JSON-safe)
        self.assertEqual(d["forbid_hand_types"], ["Pair"])
        self.assertEqual(d["obs_blind_spots"], ["history", "common_jokers"])
        # round-trip
        s2 = spec_from_dict(d)
        self.assertEqual(s2, s)

    def test_load_spec_from_disk(self):
        s = PersonalityTrainSpec(
            name="testbot",
            obs_blind_spots=("history",),
        )
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(spec_to_dict(s), f)
            path = f.name
        try:
            loaded = load_spec(path)
            self.assertEqual(loaded, s)
        finally:
            os.unlink(path)

    def test_invalid_hand_type_rejected(self):
        with self.assertRaises(ValueError):
            spec_from_dict({"name": "bad", "forbid_hand_types": ["Bogus"]})
        with self.assertRaises(ValueError):
            spec_from_dict({"name": "bad", "hand_type_bias": {"Bogus": 0.5}})

    def test_invalid_obs_block_rejected(self):
        with self.assertRaises(ValueError):
            spec_from_dict({"name": "bad", "obs_blind_spots": ["wibble"]})


class TestLayoutResolver(unittest.TestCase):
    """The block-slice resolver must match _compute_obs_dim exactly."""

    def _deck_spec(self, team_aware: bool, deck_size: int = 24):
        return _build_deck_spec(
            {"deck_size": deck_size},
            card_embedding=None,
            history_embedding=None,
            team_aware=team_aware,
        )

    def test_layout_disjoint_and_contiguous_team_aware(self):
        spec = self._deck_spec(team_aware=True)
        slices = resolve_obs_block_slices(spec)
        self.assertIn("team_flags", slices)
        cov = 0
        for blk, (s, e) in sorted(slices.items(), key=lambda kv: kv[1][0]):
            self.assertEqual(s, cov, f"gap before block {blk!r}")
            self.assertLess(s, e, f"empty block {blk!r}")
            cov = e
        self.assertEqual(cov, spec.obs_dim)

    def test_layout_disjoint_and_contiguous_no_team(self):
        spec = self._deck_spec(team_aware=False)
        slices = resolve_obs_block_slices(spec)
        self.assertNotIn("team_flags", slices)
        cov = 0
        for blk, (s, e) in sorted(slices.items(), key=lambda kv: kv[1][0]):
            self.assertEqual(s, cov, f"gap before block {blk!r}")
            cov = e
        self.assertEqual(cov, spec.obs_dim)


class TestObsZeroMask(unittest.TestCase):
    def setUp(self):
        self.spec = _build_deck_spec(
            {"deck_size": 24}, None, None, team_aware=True
        )
        self.layout = resolve_obs_block_slices(self.spec)

    def test_blanks_only_named_slice(self):
        obs = np.ones(self.spec.obs_dim, dtype=np.float32)
        ps = spec_from_dict({"name": "t", "obs_blind_spots": ["history"]})
        apply_personality_to_obs(obs, ps, self.spec)
        hs, he = self.layout["history"]
        # Named slice zeroed:
        self.assertEqual(float(obs[hs:he].sum()), 0.0)
        # Every other dim untouched:
        nonzero = int((obs != 0).sum())
        self.assertEqual(nonzero, self.spec.obs_dim - (he - hs))

    def test_multiple_blocks(self):
        obs = np.ones(self.spec.obs_dim, dtype=np.float32)
        ps = spec_from_dict(
            {"name": "t", "obs_blind_spots": ["history", "common_jokers", "blanks"]}
        )
        apply_personality_to_obs(obs, ps, self.spec)
        for blk in ("history", "common_jokers", "blanks"):
            s, e = self.layout[blk]
            self.assertEqual(float(obs[s:e].sum()), 0.0, f"{blk} not zeroed")

    def test_none_spec_no_op(self):
        obs = np.ones(self.spec.obs_dim, dtype=np.float32)
        apply_personality_to_obs(obs, None, self.spec)
        self.assertEqual(int(obs.sum()), self.spec.obs_dim)

    def test_missing_block_silently_skipped(self):
        # team_flags only exists when DeckSpec.team_aware; for a no-team spec
        # asking to zero "team_flags" should silently no-op.
        spec_noteam = _build_deck_spec(
            {"deck_size": 24}, None, None, team_aware=False
        )
        obs = np.ones(spec_noteam.obs_dim, dtype=np.float32)
        ps = spec_from_dict({"name": "t", "obs_blind_spots": ["team_flags"]})
        # Should NOT raise; all dims stay intact.
        apply_personality_to_obs(obs, ps, spec_noteam)
        self.assertEqual(int(obs.sum()), spec_noteam.obs_dim)


class TestMaskRestriction(unittest.TestCase):
    def setUp(self):
        self.spec = _build_deck_spec(
            {"deck_size": 24}, None, None, team_aware=True
        )
        self.check_id = self.spec.check_action_id

    def _full_mask(self):
        m = np.ones(self.spec.num_actions, dtype=np.float32)
        return m

    def test_forbid_pair_zeros_pair_action_ids(self):
        ps = spec_from_dict({"name": "t", "forbid_hand_types": ["Pair"]})
        m = self._full_mask()
        apply_personality_to_mask(m, ps, self.spec)
        # 24-deck Pair = ids 6..11
        self.assertEqual(float(m[6:12].sum()), 0.0)
        # Other bets still legal
        self.assertGreater(float(m[12:self.check_id].sum()), 0.0)
        # CHECK preserved
        self.assertEqual(float(m[self.check_id]), 1.0)

    def test_forbid_check_unless_only_zeros_check_when_bets_legal(self):
        ps = spec_from_dict({"name": "t", "forbid_check_unless_only": True})
        m = self._full_mask()
        apply_personality_to_mask(m, ps, self.spec)
        # Bets are legal so CHECK should be masked
        self.assertEqual(float(m[self.check_id]), 0.0)
        self.assertGreater(float(m[:self.check_id].sum()), 0.0)

    def test_safety_rail_when_only_check_was_legal(self):
        """If forbid_check_unless_only would empty the mask, the rail lifts."""
        ps = spec_from_dict({"name": "t", "forbid_check_unless_only": True})
        m = np.zeros(self.spec.num_actions, dtype=np.float32)
        m[self.check_id] = 1  # only CHECK is legal
        apply_personality_to_mask(m, ps, self.spec)
        # Safety rail must restore CHECK; never empty mask.
        self.assertEqual(float(m[self.check_id]), 1.0)

    def test_safety_rail_when_all_bets_were_forbidden_types(self):
        """If the entire legal-bet set falls in forbidden types, restore CHECK."""
        ps = spec_from_dict(
            {"name": "t", "forbid_hand_types": ["Pair"]}
        )
        m = np.zeros(self.spec.num_actions, dtype=np.float32)
        # Only Pair bets legal + CHECK
        m[6:12] = 1
        m[self.check_id] = 1
        apply_personality_to_mask(m, ps, self.spec)
        # CHECK still legal (safety rail)
        self.assertEqual(float(m[self.check_id]), 1.0)

    def test_none_spec_no_op(self):
        m = self._full_mask()
        apply_personality_to_mask(m, None, self.spec)
        self.assertEqual(int(m.sum()), self.spec.num_actions)


class TestTerminalScale(unittest.TestCase):
    """compute_personality_terminal_scale: win/loss + bluff + hand-type."""

    def _gs(self, history):
        return {"history": history}

    def test_no_op_when_spec_none(self):
        gs = self._gs([{"player": "A", "action_id": 5}])
        self.assertEqual(
            compute_personality_terminal_scale(None, gs, "A", "A", 24),
            1.0,
        )

    def test_no_op_when_loser_none(self):
        ps = spec_from_dict({"name": "t", "loss_multiplier": 99.0})
        gs = self._gs([])
        self.assertEqual(
            compute_personality_terminal_scale(ps, gs, "A", None, 24),
            1.0,
        )

    def test_win_multiplier_applied(self):
        ps = spec_from_dict({"name": "t", "win_multiplier": 1.5})
        gs = self._gs([{"player": "B", "action_id": 5}])  # B bet, A checked, B lost.
        # actor=A, loser=B → A won → win_mult applies
        self.assertAlmostEqual(
            compute_personality_terminal_scale(ps, gs, "A", "B", 24), 1.5
        )

    def test_loss_multiplier_applied(self):
        ps = spec_from_dict({"name": "t", "loss_multiplier": 2.0})
        gs = self._gs([{"player": "A", "action_id": 5}])
        # actor=A, loser=A → A lost → loss_mult
        self.assertAlmostEqual(
            compute_personality_terminal_scale(ps, gs, "A", "A", 24), 2.0
        )

    def test_bluff_caught_penalty_on_self_caught(self):
        ps = spec_from_dict(
            {"name": "t", "loss_multiplier": 2.0, "bluff_caught_penalty": 0.5}
        )
        # A bet (last bet), B checked. A lost (the bet-maker lost → bluff).
        gs = self._gs([{"player": "A", "action_id": 5}])
        self.assertAlmostEqual(
            compute_personality_terminal_scale(ps, gs, "A", "A", 24), 2.5
        )

    def test_bluff_call_bonus_on_catching_bluff(self):
        ps = spec_from_dict(
            {"name": "t", "win_multiplier": 1.0, "bluff_call_bonus": 0.3}
        )
        # B bet (last bet), A checked. B lost → A caught a bluff.
        gs = self._gs([{"player": "B", "action_id": 5}])
        self.assertAlmostEqual(
            compute_personality_terminal_scale(ps, gs, "A", "B", 24), 1.3
        )

    def test_bluff_caught_penalty_does_not_apply_to_honest_loss(self):
        """When the bet was honest (bet-maker WON), no caught-bluff penalty."""
        ps = spec_from_dict(
            {"name": "t", "loss_multiplier": 2.0, "bluff_caught_penalty": 0.5}
        )
        # B bet (last bet), A checked. A lost → bet was honest → no penalty.
        gs = self._gs([{"player": "B", "action_id": 5}])
        self.assertAlmostEqual(
            compute_personality_terminal_scale(ps, gs, "A", "A", 24), 2.0
        )

    def test_hand_type_bias_on_actor_bets(self):
        """Positive bias = likes: wins bigger, losses smaller."""
        # Pair action ids 6..11; let's bet 7 (Pair). +0.2 = likes Pair.
        ps = spec_from_dict({"name": "t", "hand_type_bias": {"Pair": 0.2}})
        gs = self._gs([{"player": "A", "action_id": 7}])
        # A won → +0.2 (likes Pair, bigger reward)
        self.assertAlmostEqual(
            compute_personality_terminal_scale(ps, gs, "A", "B", 24), 1.2
        )
        # A lost → -0.2 (likes Pair, but loss is smaller — penalty reduced by 0.2)
        gs_lose = self._gs([{"player": "A", "action_id": 7}])
        self.assertAlmostEqual(
            compute_personality_terminal_scale(ps, gs_lose, "A", "A", 24), 0.8
        )

    def test_hand_type_bias_only_on_actor_not_opponent(self):
        ps = spec_from_dict({"name": "t", "hand_type_bias": {"Pair": 0.5}})
        gs = self._gs([{"player": "B", "action_id": 7}])
        # Opponent bet pair; actor never bet pair → no bias applied.
        self.assertAlmostEqual(
            compute_personality_terminal_scale(ps, gs, "A", "B", 24), 1.0
        )


if __name__ == "__main__":
    unittest.main()
