import unittest
import torch

from nfsp_ai.embedding import HistoryEmbeddingConfig, HistoryEmbeddingEncoder
from tools.history_embedding_dataset import ClosureIterableDataset
from tools.pretrain_history_embeddings import HistoryClassifier


class HistoryEmbeddingPretrainTest(unittest.TestCase):
    def test_model_forward(self) -> None:
        dataset = ClosureIterableDataset(deck_size=24, num_samples=4, history_len=8, seed=1)
        actions, mask, labels = next(iter(dataset))
        cfg = HistoryEmbeddingConfig(num_actions=dataset.num_actions, history_len=8, embedding_dim=16)
        encoder = HistoryEmbeddingEncoder(cfg)
        model = HistoryClassifier(encoder, hidden_sizes=[32], dropout=0.0, num_actions=dataset.num_actions)
        logits = model(actions.unsqueeze(0), mask.unsqueeze(0))
        self.assertEqual(logits.shape, (1, dataset.num_actions))


if __name__ == "__main__":
    unittest.main()
