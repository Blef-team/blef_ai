import unittest

import torch

from tools.history_embedding_dataset import (
    ClosureIterableDataset,
    enumerate_minimal_action_hands,
)
from shared.game_utils import GameRules


class HistoryEmbeddingDatasetTest(unittest.TestCase):
    def test_enumerate_pools_excludes_check(self) -> None:
        deck_size = 24
        pools = enumerate_minimal_action_hands(deck_size)
        gr = GameRules(deck_size)
        self.assertEqual(len(pools), gr.check_action_id)
        self.assertTrue(all(aid < gr.check_action_id for aid in pools))
        # spot check a couple of action ids have non-empty pools
        self.assertGreater(len(pools[0]), 0)   # lowest high card
        self.assertGreater(len(pools[10]), 0)  # sample pair action

    def test_dataset_shapes(self) -> None:
        dataset = ClosureIterableDataset(deck_size=24, num_samples=4, history_len=8, min_k=1, max_k=3, seed=7)
        iterator = iter(dataset)
        history, mask, labels = next(iterator)
        self.assertEqual(history.shape, torch.Size([8]))
        self.assertEqual(mask.shape, torch.Size([8]))
        self.assertEqual(labels.shape, torch.Size([GameRules(24).check_action_id]))
        self.assertTrue((history[~mask] == -1).all())
        self.assertTrue((history[mask] >= 0).all())
        self.assertTrue(torch.all((labels == 0.0) | (labels == 1.0)))


if __name__ == "__main__":
    unittest.main()
