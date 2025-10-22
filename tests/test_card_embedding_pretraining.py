import unittest

import torch

from nfsp_ai.embedding import CardEmbeddingConfig, CardEmbeddingEncoder
from tools.pretrain_card_embeddings import CardExistenceIterableDataset, CardExistenceModel, build_deck_domain
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
        domain = build_deck_domain(24)
        rank_feature_dim = domain.num_values
        suit_feature_dim = 4
        max_hand_cards = domain.deck_size + 3
        dataset = CardExistenceIterableDataset(
            num_samples=4,
            private_card_options=[1, 2],
            common_card_options=[0, 1],
            seed=42,
            domain=domain,
            rank_feature_dim=rank_feature_dim,
            suit_feature_dim=suit_feature_dim,
            max_hand_cards=max_hand_cards,
            num_jokers=2,
            num_blanks=1,
        )
        dataset.set_epoch(0)
        sample = next(iter(dataset))
        hand_ids, aux_features, labels = sample
        self.assertEqual(hand_ids.dtype, torch.long)
        self.assertEqual(aux_features.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.float32)
        self.assertEqual(hand_ids.shape[-1], dataset.max_hand_cards)
        self.assertEqual(aux_features.shape[-1], dataset.extra_feature_dim)
        self.assertGreaterEqual(hand_ids.max().item(), -1)
        self.assertEqual(labels.shape[-1], domain.num_actions)
        self.assertTrue(torch.all((labels == 0.0) | (labels == 1.0)))

    def test_dataset_supports_32_deck(self) -> None:
        domain = build_deck_domain(32)
        rank_feature_dim = domain.num_values
        suit_feature_dim = 4
        max_hand_cards = domain.deck_size + 2
        dataset = CardExistenceIterableDataset(
            num_samples=2,
            private_card_options=[0, 1],
            common_card_options=[0, 1],
            seed=1,
            domain=domain,
            rank_feature_dim=rank_feature_dim,
            suit_feature_dim=suit_feature_dim,
            max_hand_cards=max_hand_cards,
            num_jokers=0,
            num_blanks=0,
        )
        sample = next(iter(dataset))
        _, aux_features, labels = sample
        self.assertEqual(aux_features.shape[-1], dataset.extra_feature_dim)
        self.assertEqual(labels.shape[-1], domain.num_actions)

    def test_model_forward(self) -> None:
        domain = build_deck_domain(24)
        rank_feature_dim = domain.num_values
        suit_feature_dim = 4
        extra_features = rank_feature_dim + suit_feature_dim + 2
        max_hand_cards = domain.deck_size + 3
        cfg = CardEmbeddingConfig(
            rank_dim=4,
            suit_dim=2,
            aggregator="mean",
            use_layer_norm=False,
            num_ranks=domain.num_values,
            base_deck_size=domain.deck_size,
            num_joker_ids=2,
            num_blank_ids=1,
            num_joker_embeddings=1,
            num_blank_embeddings=1,
        )
        encoder = CardEmbeddingEncoder(cfg)
        model = CardExistenceModel(
            encoder,
            hidden_sizes=[16],
            dropout=0.0,
            num_actions=domain.num_actions,
            extra_features=extra_features,
        )

        hand_ids = torch.full((2, max_hand_cards), -1, dtype=torch.long)
        hand_ids[0, :3] = torch.tensor([0, 1, 24])  # include a joker id
        hand_ids[1, :2] = torch.tensor([2, 3])
        aux_features = torch.zeros((2, extra_features), dtype=torch.float32)
        aux_features[0, 0] = 2.0  # rank count indicator
        aux_features[0, -2] = 1.0  # joker count
        aux_features[1, 1] = 1.0
        logits = model(hand_ids, aux_features)
        self.assertEqual(logits.shape, (2, domain.num_actions))


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
