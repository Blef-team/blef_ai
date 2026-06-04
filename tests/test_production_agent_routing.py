"""Unit tests for variant-aware routing in production_agent."""

import os
import unittest

from nfsp_ai.production_agent import (
    _canonicalize_to_active,
    _model_key_from_filename,
    _routing_keys,
    _resolve_model_paths,
)


class FilenameParserTest(unittest.TestCase):
    def test_legacy_24(self):
        self.assertEqual(_model_key_from_filename("artifacts/nfsp_inference_24.pt"), "24")

    def test_legacy_32(self):
        self.assertEqual(_model_key_from_filename("nfsp_inference_32.pt"), "32")

    def test_variant_1v1(self):
        self.assertEqual(_model_key_from_filename("nfsp_inference_24_1v1.pt"), "24_1v1")

    def test_variant_multi(self):
        self.assertEqual(
            _model_key_from_filename("/abs/path/artifacts/nfsp_inference_32_multi.pt"),
            "32_multi",
        )

    def test_variant_team(self):
        self.assertEqual(_model_key_from_filename("nfsp_inference_24_team.pt"), "24_team")

    def test_unknown_filename(self):
        self.assertIsNone(_model_key_from_filename("something_else.pt"))
        self.assertIsNone(_model_key_from_filename("nfsp_inference_99.pt"))


class RoutingKeysTest(unittest.TestCase):
    def test_1v1_vanilla_24(self):
        rules = {"deck_size": 24, "jokers": 0, "blanks": 0, "common_cards": 0}
        keys = _routing_keys(rules, n_active=2)
        # Most-specific first; 1v1 then multi then legacy
        self.assertEqual(keys, ["24_1v1", "24_multi", "24"])

    def test_1v1_vanilla_32(self):
        rules = {"deck_size": 32, "jokers": 0, "blanks": 0, "common_cards": 0}
        keys = _routing_keys(rules, n_active=2)
        self.assertEqual(keys, ["32_1v1", "32_multi", "32"])

    def test_multiplayer_routes_to_multi(self):
        rules = {"deck_size": 24, "jokers": 0, "blanks": 0, "common_cards": 0}
        keys = _routing_keys(rules, n_active=4)
        # n>2 disqualifies 1v1 even with j=b=cc=0
        self.assertNotIn("24_1v1", keys)
        self.assertEqual(keys, ["24_multi", "24"])

    def test_jokers_disqualify_1v1(self):
        rules = {"deck_size": 24, "jokers": 1, "blanks": 0, "common_cards": 0}
        keys = _routing_keys(rules, n_active=2)
        self.assertNotIn("24_1v1", keys)
        self.assertEqual(keys, ["24_multi", "24"])

    def test_blanks_disqualify_1v1(self):
        rules = {"deck_size": 24, "jokers": 0, "blanks": 2, "common_cards": 0}
        keys = _routing_keys(rules, n_active=2)
        self.assertNotIn("24_1v1", keys)

    def test_common_cards_disqualify_1v1(self):
        rules = {"deck_size": 24, "jokers": 0, "blanks": 0, "common_cards": 1}
        keys = _routing_keys(rules, n_active=2)
        self.assertNotIn("24_1v1", keys)

    def test_team_takes_priority(self):
        rules = {"deck_size": 24, "jokers": 0, "blanks": 0, "common_cards": 0, "teams": True}
        keys = _routing_keys(rules, n_active=4)
        self.assertEqual(keys[0], "24_team")
        # teams disqualify 1v1 even at n=2
        rules2 = {"deck_size": 24, "jokers": 0, "blanks": 0, "common_cards": 0, "teams": True}
        keys2 = _routing_keys(rules2, n_active=2)
        self.assertEqual(keys2[0], "24_team")
        self.assertNotIn("24_1v1", keys2)

    def test_team_detected_from_players_list(self):
        # Engine schema: team field on each player, no rules.teams flag.
        rules = {"deck_size": 24, "jokers": 0, "blanks": 0, "common_cards": 0}
        players = [
            {"nickname": "0", "team": 1},
            {"nickname": "1", "team": 2},
            {"nickname": "2", "team": 1},
            {"nickname": "3", "team": 2},
        ]
        keys = _routing_keys(rules, n_active=4, players=players)
        self.assertEqual(keys[0], "24_team")

    def test_no_team_when_all_player_teams_none(self):
        rules = {"deck_size": 24, "jokers": 0, "blanks": 0, "common_cards": 0}
        players = [{"nickname": str(i), "team": None} for i in range(2)]
        keys = _routing_keys(rules, n_active=2, players=players)
        self.assertNotIn("24_team", keys)
        self.assertEqual(keys[0], "24_1v1")

    def test_team_then_multi_then_legacy(self):
        rules = {"deck_size": 32, "jokers": 1, "teams": True}
        keys = _routing_keys(rules, n_active=4)
        self.assertEqual(keys, ["32_team", "32_multi", "32"])

    def test_default_deck_size_when_missing(self):
        # Deck not specified — defaults to 24
        keys = _routing_keys({}, n_active=2)
        self.assertIn("24_multi", keys)
        self.assertIn("24", keys)

    def test_4p_collapsed_to_2_active_routes_to_1v1(self):
        # 4-player vanilla game collapsed to 2 active should route to 1v1.
        rules = {"deck_size": 24, "jokers": 0, "blanks": 0, "common_cards": 0}
        players = [
            {"nickname": "0", "n_cards": 3, "team": None},
            {"nickname": "1", "n_cards": 0, "team": None},  # eliminated
            {"nickname": "2", "n_cards": 5, "team": None},
            {"nickname": "3", "n_cards": 0, "team": None},  # eliminated
        ]
        keys = _routing_keys(rules, n_active=2, players=players)
        self.assertEqual(keys[0], "24_1v1")

    def test_4p_with_3_active_does_not_route_to_1v1(self):
        # 3 active players — neither 1v1 nor multi-eliminated; goes to multi.
        rules = {"deck_size": 24, "jokers": 0, "blanks": 0, "common_cards": 0}
        keys = _routing_keys(rules, n_active=3)
        self.assertNotIn("24_1v1", keys)
        self.assertEqual(keys[0], "24_multi")

    def test_collapsed_with_jokers_does_not_route_to_1v1(self):
        # Even at n_active=2, jokers disqualify 1v1.
        rules = {"deck_size": 32, "jokers": 1, "blanks": 0, "common_cards": 0}
        keys = _routing_keys(rules, n_active=2)
        self.assertNotIn("32_1v1", keys)


class CanonicalizeToActiveTest(unittest.TestCase):
    def test_drops_eliminated_players(self):
        gs = {
            "cp_nickname": "alice",
            "rules": {"deck_size": 24},
            "players": [
                {"nickname": "alice", "n_cards": 3},
                {"nickname": "bob", "n_cards": 0},
                {"nickname": "carol", "n_cards": 5},
                {"nickname": "dave", "n_cards": 0},
            ],
            "history": [],
        }
        out = _canonicalize_to_active(gs)
        self.assertEqual([p["nickname"] for p in out["players"]], ["alice", "carol"])

    def test_preserves_input_when_no_eliminations(self):
        gs = {
            "cp_nickname": "alice",
            "rules": {"deck_size": 24},
            "players": [
                {"nickname": "alice", "n_cards": 3},
                {"nickname": "bob", "n_cards": 5},
            ],
        }
        out = _canonicalize_to_active(gs)
        self.assertIs(out, gs)  # short-circuit returns the same dict

    def test_does_not_mutate_input(self):
        gs = {
            "cp_nickname": "alice",
            "players": [
                {"nickname": "alice", "n_cards": 3},
                {"nickname": "bob", "n_cards": 0},
            ],
        }
        original_players = list(gs["players"])
        _ = _canonicalize_to_active(gs)
        self.assertEqual(gs["players"], original_players)


class ResolveModelPathsTest(unittest.TestCase):
    def setUp(self):
        # Build a temporary artifacts/ in /tmp
        import tempfile, shutil
        self.tmp = tempfile.mkdtemp()
        self.art = os.path.join(self.tmp, "artifacts")
        os.makedirs(self.art)
        self.cwd = os.getcwd()
        os.chdir(self.tmp)

    def tearDown(self):
        import shutil
        os.chdir(self.cwd)
        shutil.rmtree(self.tmp)

    def _touch(self, name):
        p = os.path.join(self.art, name)
        with open(p, "wb") as f:
            f.write(b"")
        return p

    def test_discovers_legacy_only(self):
        self._touch("nfsp_inference_24.pt")
        self._touch("nfsp_inference_32.pt")
        out = _resolve_model_paths(None)
        self.assertEqual(set(out.keys()), {"24", "32"})

    def test_discovers_variant_files(self):
        self._touch("nfsp_inference_24_1v1.pt")
        self._touch("nfsp_inference_24_multi.pt")
        self._touch("nfsp_inference_32_multi.pt")
        out = _resolve_model_paths(None)
        self.assertEqual(set(out.keys()), {"24_1v1", "24_multi", "32_multi"})

    def test_mixed_legacy_and_variant(self):
        self._touch("nfsp_inference_24.pt")
        self._touch("nfsp_inference_24_1v1.pt")
        self._touch("nfsp_inference_32_multi.pt")
        out = _resolve_model_paths(None)
        self.assertEqual(set(out.keys()), {"24", "24_1v1", "32_multi"})

    def test_explicit_path(self):
        p = self._touch("nfsp_inference_24_multi.pt")
        out = _resolve_model_paths(p)
        self.assertEqual(out, {"24_multi": p})

    def test_explicit_path_nonstandard_filename(self):
        p = os.path.join(self.tmp, "my_custom_model.pt")
        with open(p, "wb") as f:
            f.write(b"")
        out = _resolve_model_paths(p)
        # Non-standard filename: stored under "explicit", to be rewritten
        # at load time once act_dim is probed.
        self.assertEqual(out, {"explicit": p})

    def test_explicit_path_missing(self):
        out = _resolve_model_paths("/nonexistent/path.pt")
        self.assertEqual(out, {})

    def test_ignores_random_files(self):
        self._touch("nfsp_inference_24.pt")
        self._touch("readme.txt")
        self._touch("nfsp_blef_5M.pt")
        out = _resolve_model_paths(None)
        # Only the proper inference file is picked up
        self.assertEqual(set(out.keys()), {"24"})


# ---------------------------------------------------------------------------
# Personality-keyed routing additions (new in feat/trained-personalities)
# ---------------------------------------------------------------------------

class PersonalityFilenameRegexTest(unittest.TestCase):
    """The personality suffix is captured separately from the variant suffix."""

    def test_personality_with_variant(self):
        self.assertEqual(
            _model_key_from_filename("nfsp_inference_24_1v1_kupala.pt"),
            "24_1v1_kupala",
        )
        self.assertEqual(
            _model_key_from_filename("nfsp_inference_24_multi_zorya.pt"),
            "24_multi_zorya",
        )
        self.assertEqual(
            _model_key_from_filename("nfsp_inference_32_team_leshy.pt"),
            "32_team_leshy",
        )

    def test_personality_without_variant(self):
        # Deck-only + personality (no variant slot).
        self.assertEqual(
            _model_key_from_filename("nfsp_inference_24_kupala.pt"),
            "24_kupala",
        )

    def test_uppercase_personality_rejected(self):
        """Personalities are lowercase by convention; reject UPPERCASE entries."""
        self.assertIsNone(_model_key_from_filename("nfsp_inference_24_KUPALA.pt"))
        self.assertIsNone(_model_key_from_filename("nfsp_inference_24_1v1_LESHY.pt"))


class PersonalityRoutingChainTest(unittest.TestCase):
    """Routing must prepend personality-keyed candidates when one is resolved.

    The variant-only chain remains the fallback, so sculpted-only
    personalities (no trained checkpoint) keep routing to the baseline
    where the existing `personality_action` sculpting layer runs.
    """

    def test_personality_prepended_for_1v1(self):
        chain = _routing_keys(
            {"deck_size": 24, "jokers": 0, "blanks": 0, "common_cards": 0},
            n_active=2,
            players=[],
            personality="kupala",
        )
        self.assertEqual(
            chain,
            [
                "24_1v1_kupala",
                "24_multi_kupala",
                "24_kupala",
                "24_1v1",
                "24_multi",
                "24",
            ],
        )

    def test_personality_prepended_for_multi(self):
        # For non-team multi-player games, the chain now includes the trained
        # 1v1 personality ckpt as a fallback before the generic variant
        # baseline — the 1v1-trained obs schema is compatible with non-team
        # play, and the design choice is "trained trait is more important
        # than variant-specific strategic refinement."
        chain = _routing_keys(
            {"deck_size": 24, "jokers": 1},
            n_active=3,
            players=[],
            personality="zorya",
        )
        self.assertEqual(
            chain,
            ["24_multi_zorya", "24_zorya", "24_1v1_zorya", "24_multi", "24"],
        )

    def test_personality_chain_for_32_deck_multi(self):
        # Same cross-variant trained-1v1 fallback applies to 32-deck.
        chain = _routing_keys(
            {"deck_size": 32, "jokers": 1},
            n_active=3,
            players=[],
            personality="zorya",
        )
        self.assertEqual(
            chain,
            ["32_multi_zorya", "32_zorya", "32_1v1_zorya", "32_multi", "32"],
        )

    def test_personality_team_does_not_fall_back_to_1v1(self):
        # Team mode: the trained 1v1 ckpt has obs_dim 390 (team_aware=False),
        # incompatible with team obs (398-dim). Routing chain must NOT include
        # the cross-variant 1v1 fallback so we don't try to load a shape-
        # mismatched ckpt. Variant-only chain handles team mode.
        chain = _routing_keys(
            {"deck_size": 24, "teams": True},
            n_active=4,
            players=[],
            personality="mavka",
        )
        self.assertNotIn("24_1v1_mavka", chain[chain.index("24_team_mavka") + 1:])

    def test_personality_prepended_for_team(self):
        chain = _routing_keys(
            {"deck_size": 24, "teams": True},
            n_active=4,
            players=[],
            personality="poludnica",
        )
        # team gets the most-specific personality key first.
        self.assertEqual(chain[0], "24_team_poludnica")
        # Variant-only fallback chain preserved.
        self.assertIn("24_team", chain)
        self.assertIn("24_multi", chain)
        self.assertIn("24", chain)

    def test_personality_chain_for_32_deck(self):
        chain = _routing_keys(
            {"deck_size": 32, "jokers": 0, "blanks": 0, "common_cards": 0},
            n_active=2,
            players=[],
            personality="leshy",
        )
        self.assertEqual(
            chain,
            [
                "32_1v1_leshy",
                "32_multi_leshy",
                "32_leshy",
                "32_1v1",
                "32_multi",
                "32",
            ],
        )

    def test_personality_none_falls_back_to_variant_chain(self):
        """When no personality is resolved, the chain is the original."""
        chain_a = _routing_keys(
            {"deck_size": 24}, n_active=2, players=[], personality=None
        )
        chain_b = _routing_keys(
            {"deck_size": 24}, n_active=2, players=[]
        )
        self.assertEqual(chain_a, chain_b)
        self.assertEqual(chain_a, ["24_1v1", "24_multi", "24"])

    def test_chain_is_deduped(self):
        chain = _routing_keys(
            {"deck_size": 24}, n_active=2, players=[], personality="mavka"
        )
        self.assertEqual(len(chain), len(set(chain)))


if __name__ == "__main__":
    unittest.main()
