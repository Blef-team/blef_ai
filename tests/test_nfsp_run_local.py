import unittest

from nfsp_ai.nfsp_run_local import MyEnv


class NFSPRunLocalDeckSizeTest(unittest.TestCase):
    def test_observation_and_action_dims_24(self):
        env = MyEnv(n_agents=2, deck_size=24)
        obs, mask, _ = env.reset()
        self.assertEqual(obs.numel(), env.deck_spec.obs_dim)
        self.assertEqual(mask.numel(), env.deck_spec.num_actions)

    def test_observation_and_action_dims_32(self):
        env = MyEnv(n_agents=2, deck_size=32)
        obs, mask, _ = env.reset()
        self.assertEqual(obs.numel(), env.deck_spec.obs_dim)
        self.assertEqual(mask.numel(), env.deck_spec.num_actions)


if __name__ == "__main__":
    unittest.main()
