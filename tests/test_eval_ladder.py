"""Smoke tests for tools.eval_ladder.

These tests run very small games (n_games=4) against random and conservative
opponents to verify the harness wires up correctly.
"""

import os
import unittest
import torch

from tools.eval_ladder import (
    RandomOpponent,
    ConservativeOpponent,
    CFROpponent,
    head_to_head,
    build_opponents,
    MatchResult,
)
from nfsp_ai.agent import NFSPAgent, NFSPConfig
from nfsp_ai.nfsp_run_local import MyEnv


def _fresh_agent(deck_size: int = 24, n_players: int = 2):
    env = MyEnv(n_agents=n_players, deck_size=deck_size, max_cards=2)
    obs, mask, _pid = env.reset()
    obs_dim = int(obs.shape[-1])
    act_dim = int(mask.shape[-1])
    cfg = NFSPConfig(rl_capacity=1, sl_capacity=1)
    agent = NFSPAgent(obs_dim, act_dim, device=torch.device("cpu"), cfg=cfg)
    return agent


class EvalLadderSmokeTest(unittest.TestCase):
    def test_random_opponent_against_random_learner(self):
        agent = _fresh_agent()
        opp = RandomOpponent()
        r = head_to_head(
            agent, opp,
            n_games=4, deck_size=24, n_players=2, max_cards=2, seed=123,
        )
        self.assertIsInstance(r, MatchResult)
        self.assertEqual(r.opponent, "random")
        self.assertEqual(r.n_games, 4)
        self.assertEqual(r.wins + r.losses + r.draws, 4)
        self.assertGreaterEqual(r.mean_game_length_actions, 1.0)
        self.assertGreater(r.elapsed_seconds, 0.0)

    def test_conservative_opponent_runs(self):
        agent = _fresh_agent()
        opp = ConservativeOpponent()
        r = head_to_head(
            agent, opp,
            n_games=2, deck_size=24, n_players=2, max_cards=1, seed=7,
        )
        self.assertEqual(r.opponent, "conservative")
        self.assertEqual(r.wins + r.losses + r.draws, 2)

    def test_cfr_unavailable_when_outputs_missing(self):
        # cfr_ai/outputs/ should not exist locally for this PR's test env.
        opp = CFROpponent()
        # Either truly unavailable, or skip if a developer has populated the dir.
        if not opp.available():
            self.assertFalse(opp.available())
            built = build_opponents("cfr", 1, 1)
            self.assertEqual(built, [])
        else:
            self.skipTest("CFR strategies present locally; availability test skipped")

    def test_winrate_ci_bounds(self):
        agent = _fresh_agent()
        r = head_to_head(
            agent, RandomOpponent(),
            n_games=8, deck_size=24, n_players=2, max_cards=1, seed=1,
        )
        self.assertGreaterEqual(r.winrate, 0.0)
        self.assertLessEqual(r.winrate, 1.0)
        self.assertGreaterEqual(r.winrate_ci_95, 0.0)


if __name__ == "__main__":
    unittest.main()
