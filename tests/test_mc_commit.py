"""Verify the MC-commit / TD-machinery removal in PR2.

Each test below corresponds to a specific cleanup item from the audit plan:

  1. ReplayBuffer.add accepts the new 7-arg signature (no gamma_power).
  2. ReplayBuffer.sample returns a 7-tuple (no gpow).
  3. NFSPAgent has no `q_tgt` attribute.
  4. NFSPConfig has no `target_tau`, `hard_target_interval`, or `use_double_dqn`.
  5. Saved checkpoints round-trip through save/load on a fresh agent.
  6. Older checkpoints containing `q_tgt` load with the field silently dropped.
  7. Control-plane silently ignores `tau` / `hard_target_interval` overrides.
"""

import os
import tempfile
import unittest

import torch

from nfsp_ai.agent import NFSPAgent, NFSPConfig, ReplayBuffer
from nfsp_ai.control_plane import JsonControlPlane, write_control_file
from nfsp_ai.nfsp_run_local import MyEnv


def _fresh_agent():
    env = MyEnv(n_agents=2, deck_size=24, max_cards=2)
    obs, mask, _ = env.reset()
    obs_dim = int(obs.shape[-1])
    act_dim = int(mask.shape[-1])
    cfg = NFSPConfig(rl_capacity=8, sl_capacity=8)
    return NFSPAgent(obs_dim, act_dim, device=torch.device("cpu"), cfg=cfg)


class ReplayBufferShape(unittest.TestCase):
    def test_add_signature_is_7_args(self):
        buf = ReplayBuffer(4, 3, 2, torch.device("cpu"))
        obs = torch.zeros(3)
        mask = torch.ones(2)
        # 7 args: obs, mask, act, rew, nobs, nmask, done — no gamma_power
        buf.add(obs, mask, torch.tensor(0), torch.tensor(0.5), obs, mask, False)
        self.assertEqual(buf.size, 1)

    def test_sample_returns_7_tuple(self):
        buf = ReplayBuffer(4, 3, 2, torch.device("cpu"))
        for _ in range(2):
            buf.add(
                torch.zeros(3), torch.ones(2),
                torch.tensor(0), torch.tensor(1.0),
                torch.zeros(3), torch.ones(2), False,
            )
        out = buf.sample(2)
        self.assertEqual(len(out), 7)


class AgentShape(unittest.TestCase):
    def test_no_q_tgt(self):
        agent = _fresh_agent()
        self.assertFalse(hasattr(agent, "q_tgt"), "q_tgt should be removed for MC-only loss")

    def test_config_drops_td_fields(self):
        cfg = NFSPConfig()
        for dead in ("target_tau", "hard_target_interval", "use_double_dqn"):
            self.assertFalse(
                hasattr(cfg, dead),
                f"NFSPConfig should not carry {dead!r} after MC commit",
            )


class Persistence(unittest.TestCase):
    def test_roundtrip_save_load(self):
        agent = _fresh_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            agent.save(path)
            agent2 = _fresh_agent()
            agent2.load(path)
            # state_dict round-trip preserves Q parameters
            for k in agent.q.state_dict():
                self.assertTrue(torch.allclose(agent.q.state_dict()[k], agent2.q.state_dict()[k]))

    def test_load_tolerates_legacy_q_tgt_field(self):
        agent = _fresh_agent()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "legacy.pt")
            payload = {
                "q": agent.q.state_dict(),
                "q_tgt": agent.q.state_dict(),  # legacy
                "pi": agent.pi.state_dict(),
                "cfg": {**agent.cfg.__dict__, "target_tau": 0.005, "hard_target_interval": 0, "use_double_dqn": True},
                "steps": 0,
            }
            torch.save(payload, path)
            fresh = _fresh_agent()
            fresh.load(path)  # must not raise


class ControlPlaneTolerance(unittest.TestCase):
    def test_deprecated_keys_silently_skipped(self):
        agent = _fresh_agent()
        env = MyEnv(n_agents=2, deck_size=24, max_cards=2)
        env.reset()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "control.json")
            write_control_file(path, {
                "meta": {"cooldown_steps": 0},
                "overrides": {"tau": 0.01, "hard_target_interval": 1000, "epsilon": 0.05},
            })
            cp = JsonControlPlane(path, cooldown_steps=0, verbose=False)
            cp.tick(agent, env, {"step": 1})
        # Reaching here without an exception means deprecated keys were tolerated.


if __name__ == "__main__":
    unittest.main()
