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

    def test_pick_jokers_blanks_common_in_range(self):
        env = MyEnv(
            n_agents=2,
            deck_size=24,
            jokers=2,
            blanks=3,
            common_cards=4,
            pick_jokers_in_range=True,
            pick_blanks_in_range=True,
            pick_common_cards_in_range=True,
        )
        for _ in range(5):
            env.reset()
            rules = env.rules
            self.assertTrue(0 <= rules.get("jokers", 0) <= 2)
            self.assertTrue(0 <= rules.get("blanks", 0) <= 3)
            self.assertTrue(0 <= rules.get("common_cards", 0) <= 4)


if __name__ == "__main__":
    unittest.main()
