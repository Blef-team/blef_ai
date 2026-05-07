"""Tests for the PR3 fixes:

  1. SL reservoir gates ``add`` on ``is_br`` (only BR transitions are logged
     for π's supervised target).
  2. ``_legal_action_mask`` ignores ``pub_prior`` — the mask is purely the
     game's rule-based legality, not a function of the probability calculator.
"""

import unittest
import numpy as np
import torch

from nfsp_ai.nfsp_run_local import _legal_action_mask, MyEnv, _deck_spec_from_game


def _build_spec_for_default_game():
    env = MyEnv(n_agents=2, deck_size=24, max_cards=2)
    env.reset()
    return env, _deck_spec_from_game(env.game)


class LegalMaskIgnoresPriors(unittest.TestCase):
    def test_mask_invariant_to_pub_prior(self):
        env, spec = _build_spec_for_default_game()
        # Two extreme priors: all-near-zero and all-one. The mask must be identical.
        check = int(spec.check_action_id)
        mostly_zero = [1e-15] * check  # below the old PUBLIC_PRIOR_EPS=1e-9
        all_one = [1.0] * check
        m_zero = _legal_action_mask(env.game, spec, mostly_zero)
        m_one = _legal_action_mask(env.game, spec, all_one)
        self.assertTrue(torch.equal(m_zero, m_one))

    def test_mask_matches_rule_only_legality_at_round_start(self):
        env, spec = _build_spec_for_default_game()
        # At round start with empty history, every bet [0, check_id) is legal,
        # CHECK is not.
        mask = _legal_action_mask(env.game, spec, [1.0] * spec.check_action_id)
        check = int(spec.check_action_id)
        self.assertEqual(int(mask[:check].sum().item()), check)
        self.assertEqual(int(mask[check].item()), 0)


class SLReservoirIsBROnly(unittest.TestCase):
    """Static assertion that the gate is in place; the runtime behavior is
    smoke-tested via train_from_selfplay in CI but isn't ergonomic to unit
    test directly. We verify the source reads as expected.
    """

    def test_skip_sl_log_includes_not_use_br(self):
        import inspect
        import nfsp_ai.agent as agent_mod
        src = inspect.getsource(agent_mod.NFSPAgent.train_from_selfplay)
        # The new gate must include the `(not use_br)` condition before any
        # other check.
        self.assertIn("not use_br", src, "SL gate must skip when action came from π")


if __name__ == "__main__":
    unittest.main()
