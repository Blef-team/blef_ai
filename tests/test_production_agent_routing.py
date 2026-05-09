"""Unit tests for variant-aware routing in production_agent."""

import os
import unittest

from nfsp_ai.production_agent import (
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
        keys = _routing_keys(rules, n_players=2)
        # Most-specific first; 1v1 then multi then legacy
        self.assertEqual(keys, ["24_1v1", "24_multi", "24"])

    def test_1v1_vanilla_32(self):
        rules = {"deck_size": 32, "jokers": 0, "blanks": 0, "common_cards": 0}
        keys = _routing_keys(rules, n_players=2)
        self.assertEqual(keys, ["32_1v1", "32_multi", "32"])

    def test_multiplayer_routes_to_multi(self):
        rules = {"deck_size": 24, "jokers": 0, "blanks": 0, "common_cards": 0}
        keys = _routing_keys(rules, n_players=4)
        # n>2 disqualifies 1v1 even with j=b=cc=0
        self.assertNotIn("24_1v1", keys)
        self.assertEqual(keys, ["24_multi", "24"])

    def test_jokers_disqualify_1v1(self):
        rules = {"deck_size": 24, "jokers": 1, "blanks": 0, "common_cards": 0}
        keys = _routing_keys(rules, n_players=2)
        self.assertNotIn("24_1v1", keys)
        self.assertEqual(keys, ["24_multi", "24"])

    def test_blanks_disqualify_1v1(self):
        rules = {"deck_size": 24, "jokers": 0, "blanks": 2, "common_cards": 0}
        keys = _routing_keys(rules, n_players=2)
        self.assertNotIn("24_1v1", keys)

    def test_common_cards_disqualify_1v1(self):
        rules = {"deck_size": 24, "jokers": 0, "blanks": 0, "common_cards": 1}
        keys = _routing_keys(rules, n_players=2)
        self.assertNotIn("24_1v1", keys)

    def test_team_takes_priority(self):
        rules = {"deck_size": 24, "jokers": 0, "blanks": 0, "common_cards": 0, "teams": True}
        keys = _routing_keys(rules, n_players=4)
        self.assertEqual(keys[0], "24_team")
        # teams disqualify 1v1 even at n=2
        rules2 = {"deck_size": 24, "jokers": 0, "blanks": 0, "common_cards": 0, "teams": True}
        keys2 = _routing_keys(rules2, n_players=2)
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
        keys = _routing_keys(rules, n_players=4, players=players)
        self.assertEqual(keys[0], "24_team")

    def test_no_team_when_all_player_teams_none(self):
        rules = {"deck_size": 24, "jokers": 0, "blanks": 0, "common_cards": 0}
        players = [{"nickname": str(i), "team": None} for i in range(2)]
        keys = _routing_keys(rules, n_players=2, players=players)
        self.assertNotIn("24_team", keys)
        self.assertEqual(keys[0], "24_1v1")

    def test_team_then_multi_then_legacy(self):
        rules = {"deck_size": 32, "jokers": 1, "teams": True}
        keys = _routing_keys(rules, n_players=4)
        self.assertEqual(keys, ["32_team", "32_multi", "32"])

    def test_default_deck_size_when_missing(self):
        # Deck not specified — defaults to 24
        keys = _routing_keys({}, n_players=2)
        self.assertIn("24_multi", keys)
        self.assertIn("24", keys)


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


if __name__ == "__main__":
    unittest.main()
