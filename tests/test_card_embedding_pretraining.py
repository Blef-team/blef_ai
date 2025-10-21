import unittest

import torch

from nfsp_ai.embedding import CardEmbeddingConfig, CardEmbeddingEncoder
from tools.pretrain_card_embeddings import (
    CardExistenceIterableDataset,
    CardExistenceModel,
)
from shared.api.simpleschema_local_manager import determine_set_existence


class CardEmbeddingEncoderTest(unittest.TestCase):
    def test_encoder_shapes_and_counts(self) -> None:
        cfg = CardEmbeddingConfig(
            rank_dim=4,
            suit_dim=2,
            aggregator="mean",
            use_layer_norm=False,
            num_joker_ids=2,
            num_blank_ids=1,
            num_joker_embeddings=1,
            num_blank_embeddings=1,
        )
        encoder = CardEmbeddingEncoder(cfg)
        card_ids = torch.tensor([[0, 5, -1], [23, -1, -1], [24, -1, -1], [26, -1, -1]], dtype=torch.long)
        embeddings, counts = encoder(card_ids)

        self.assertEqual(embeddings.shape, (4, cfg.rank_dim + cfg.suit_dim))
        self.assertEqual(counts.shape, (4, 1))
        self.assertTrue(torch.allclose(counts[:, 0], torch.tensor([2.0, 1.0, 1.0, 1.0])))

        empty_input = torch.full((1, 3), -1, dtype=torch.long)
        empty_emb, empty_count = encoder(empty_input)
        self.assertTrue(torch.all(empty_count == 0))
        self.assertTrue(torch.allclose(empty_emb, torch.zeros_like(empty_emb)))


class CardExistencePipelineTest(unittest.TestCase):
    def test_dataset_produces_valid_sample(self) -> None:
        dataset = CardExistenceIterableDataset(
            num_samples=4,
            private_card_options=[1, 2],
            common_card_options=[0, 1],
            seed=42,
            num_jokers=2,
            num_blanks=1,
        )
        dataset.set_epoch(0)
        sample = next(iter(dataset))
        private_ids, common_ids, labels = sample
        self.assertEqual(private_ids.dtype, torch.long)
        self.assertEqual(common_ids.dtype, torch.long)
        self.assertEqual(labels.dtype, torch.float32)
        self.assertGreaterEqual(private_ids.max().item(), -1)
        self.assertGreaterEqual(common_ids.max().item(), -1)
        self.assertEqual(labels.shape[-1], 88)
        self.assertTrue(torch.all((labels == 0.0) | (labels == 1.0)))

    def test_model_forward(self) -> None:
        cfg = CardEmbeddingConfig(
            rank_dim=4,
            suit_dim=2,
            aggregator="mean",
            use_layer_norm=False,
            num_joker_ids=2,
            num_blank_ids=1,
            num_joker_embeddings=1,
            num_blank_embeddings=1,
        )
        encoder = CardEmbeddingEncoder(cfg)
        model = CardExistenceModel(encoder, hidden_sizes=[16], dropout=0.0)

        private_ids = torch.tensor(
            [
                [0, 24, -1],
                [2, 26, -1],
            ],
            dtype=torch.long,
        )
        common_ids = torch.tensor(
            [
                [3, -1],
                [-1, -1],
            ],
            dtype=torch.long,
        )
        logits = model(private_ids, common_ids)
        self.assertEqual(logits.shape, (2, 88))


class DetermineSetExistenceTest(unittest.TestCase):
    def test_jokers_substitute_cards(self) -> None:
        cards = [
            {"value": 0, "colour": 0},
            {"value": -1, "colour": -1},  # joker
        ]
        self.assertTrue(determine_set_existence(cards, action_id=6))  # Pair of lowest rank

    def test_blanks_are_inert(self) -> None:
        cards = [
            {"value": 0, "colour": 0},
            {"value": -2, "colour": -2},  # blank
        ]
        self.assertFalse(determine_set_existence(cards, action_id=6))

        cards_with_pair = [
            {"value": 0, "colour": 0},
            {"value": 0, "colour": 1},
            {"value": -2, "colour": -2},
        ]
        self.assertTrue(determine_set_existence(cards_with_pair, action_id=6))


if __name__ == "__main__":
    unittest.main()
