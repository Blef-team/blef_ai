"""Unit tests for the trust_history obs override.

Trust override: for each non-CHECK action_id in game.history whose actor
matches the selector, set BOTH private_priors[id] and public_priors[id]
to 1.0. Inference-only — never used during training.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from nfsp_ai.personality_train import (
    apply_trust_history_to_obs,
    resolve_obs_block_slices,
)
from nfsp_ai.nfsp_run_local import _deck_spec_from_game


def _mk_game(deck_size=24, n_players=2):
    """Minimal game-like dict sufficient for _deck_spec_from_game()."""
    return {
        "rules": {"deck_size": deck_size, "jokers": 0, "blanks": 0, "common_cards": 0},
        "max_cards": 11,
        "players": [
            {"nickname": "0", "n_cards": 1, "team": None},
            {"nickname": "1", "n_cards": 1, "team": None},
        ],
        "common_hand": [],
        "history": [],
        "hands": [
            {"nickname": "0", "hand": [{"value": 4, "colour": 3}]},
            {"nickname": "1", "hand": [{"value": 6, "colour": 1}]},
        ],
        "cp_nickname": "0",
    }


def _build_spec(game):
    """Build a DeckSpec with predictable embedding-free dims."""
    return _deck_spec_from_game(game, card_embedding=None,
                                history_embedding=None, team_aware=False)


class TrustHistoryObsTest(unittest.TestCase):

    def test_empty_selector_no_op(self):
        game = _mk_game()
        spec = _build_spec(game)
        obs = np.full(spec.obs_dim, 0.4, dtype=np.float32)
        before = obs.copy()
        apply_trust_history_to_obs(obs, spec, game, trust_history=())
        np.testing.assert_array_equal(obs, before)

    def test_empty_history_no_op(self):
        game = _mk_game()
        spec = _build_spec(game)
        obs = np.full(spec.obs_dim, 0.5, dtype=np.float32)
        before = obs.copy()
        apply_trust_history_to_obs(obs, spec, game, trust_history=("self", "opponent"))
        np.testing.assert_array_equal(obs, before)

    def test_opponent_bet_sets_both_priors(self):
        game = _mk_game()
        spec = _build_spec(game)
        layout = resolve_obs_block_slices(spec)
        pvt_s, _ = layout["private_priors"]
        pub_s, _ = layout["public_priors"]
        # Actor is "0"; opponent is "1" placing a bet at action 17.
        game["history"] = [{"player": "1", "action_id": 17}]
        obs = np.zeros(spec.obs_dim, dtype=np.float32)
        apply_trust_history_to_obs(obs, spec, game, trust_history=("opponent",))
        self.assertEqual(obs[pvt_s + 17], 1.0)
        self.assertEqual(obs[pub_s + 17], 1.0)
        # No other index touched.
        obs[pvt_s + 17] = 0.0; obs[pub_s + 17] = 0.0
        self.assertEqual(obs.sum(), 0.0)

    def test_self_bet_not_overridden_when_selector_is_opponent_only(self):
        game = _mk_game()
        spec = _build_spec(game)
        layout = resolve_obs_block_slices(spec)
        pvt_s, _ = layout["private_priors"]
        # Actor is "0", placing bet 9.
        game["history"] = [{"player": "0", "action_id": 9}]
        obs = np.zeros(spec.obs_dim, dtype=np.float32)
        apply_trust_history_to_obs(obs, spec, game, trust_history=("opponent",))
        # selector is opponent only — actor's own bet must NOT be overridden
        self.assertEqual(obs[pvt_s + 9], 0.0)

    def test_self_only_selector(self):
        game = _mk_game()
        spec = _build_spec(game)
        layout = resolve_obs_block_slices(spec)
        pvt_s, _ = layout["private_priors"]
        pub_s, _ = layout["public_priors"]
        game["history"] = [
            {"player": "0", "action_id": 5},
            {"player": "1", "action_id": 23},
        ]
        obs = np.zeros(spec.obs_dim, dtype=np.float32)
        apply_trust_history_to_obs(obs, spec, game, trust_history=("self",))
        # Only self bet (id 5) overridden; opponent bet (id 23) untouched.
        self.assertEqual(obs[pvt_s + 5], 1.0)
        self.assertEqual(obs[pub_s + 5], 1.0)
        self.assertEqual(obs[pvt_s + 23], 0.0)
        self.assertEqual(obs[pub_s + 23], 0.0)

    def test_both_selector_overrides_all_non_check(self):
        game = _mk_game()
        spec = _build_spec(game)
        layout = resolve_obs_block_slices(spec)
        pvt_s, _ = layout["private_priors"]
        pub_s, _ = layout["public_priors"]
        check_id = int(spec.check_action_id)
        game["history"] = [
            {"player": "0", "action_id": 3},
            {"player": "1", "action_id": 19},
            {"player": "0", "action_id": check_id},  # CHECK — must NOT override
        ]
        obs = np.zeros(spec.obs_dim, dtype=np.float32)
        apply_trust_history_to_obs(obs, spec, game, trust_history=("self", "opponent"))
        for aid in (3, 19):
            self.assertEqual(obs[pvt_s + aid], 1.0, msg=f"aid={aid}")
            self.assertEqual(obs[pub_s + aid], 1.0, msg=f"aid={aid}")
        # CHECK slot is at the boundary — must not write anything for it.
        # The priors slice length is check_id (only bets indexed 0..check_id-1).
        # The above loop already only writes bet ids; this test confirms via
        # the empty rest-of-obs invariant.
        # Zero out the touched slots and confirm rest is untouched.
        for aid in (3, 19):
            obs[pvt_s + aid] = 0.0
            obs[pub_s + aid] = 0.0
        self.assertEqual(obs.sum(), 0.0)

    def test_other_obs_blocks_untouched(self):
        """The override must NOT bleed into adjacent obs blocks."""
        game = _mk_game()
        spec = _build_spec(game)
        layout = resolve_obs_block_slices(spec)
        pvt_s, pvt_e = layout["private_priors"]
        pub_s, pub_e = layout["public_priors"]
        game["history"] = [{"player": "1", "action_id": 0}]  # first bet id
        obs = np.full(spec.obs_dim, 0.42, dtype=np.float32)
        apply_trust_history_to_obs(obs, spec, game, trust_history=("opponent",))
        # The overridden slots are 1.0
        self.assertEqual(obs[pvt_s + 0], 1.0)
        self.assertEqual(obs[pub_s + 0], 1.0)
        # Adjacent slots in same slice keep their original value
        self.assertAlmostEqual(float(obs[pvt_s + 1]), 0.42)
        self.assertAlmostEqual(float(obs[pub_s + 1]), 0.42)
        # Everything outside the priors slices keeps original value
        for i in range(spec.obs_dim):
            if pvt_s <= i < pvt_e or pub_s <= i < pub_e:
                continue
            self.assertAlmostEqual(float(obs[i]), 0.42, msg=f"idx={i}")

    def test_malformed_history_entries_safe(self):
        game = _mk_game()
        spec = _build_spec(game)
        game["history"] = [
            None,
            {},
            {"player": None, "action_id": -5},
            {"player": "1", "action_id": "not_an_int"},
            {"player": "1"},
            {"action_id": 12},  # missing player — treated as not-self
        ]
        obs = np.zeros(spec.obs_dim, dtype=np.float32)
        # Should not raise.
        apply_trust_history_to_obs(obs, spec, game, trust_history=("self", "opponent"))
        # The only well-formed bet (action_id=12 from a "neutral" entry) is
        # classified as opponent (not actor) -> overridden.
        layout = resolve_obs_block_slices(spec)
        pvt_s, _ = layout["private_priors"]
        pub_s, _ = layout["public_priors"]
        self.assertEqual(obs[pvt_s + 12], 1.0)
        self.assertEqual(obs[pub_s + 12], 1.0)

    def test_oob_action_id_ignored(self):
        game = _mk_game()
        spec = _build_spec(game)
        check_id = int(spec.check_action_id)
        # Action id beyond check_id (impossible, but defensive).
        game["history"] = [{"player": "1", "action_id": check_id + 10}]
        obs = np.zeros(spec.obs_dim, dtype=np.float32)
        apply_trust_history_to_obs(obs, spec, game, trust_history=("opponent",))
        self.assertEqual(obs.sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
