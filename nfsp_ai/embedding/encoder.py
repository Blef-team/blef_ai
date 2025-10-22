from dataclasses import dataclass, asdict
from typing import Tuple

import torch
from torch import nn


@dataclass
class CardEmbeddingConfig:
    """Configuration for the card embedding encoder."""

    rank_dim: int = 16
    suit_dim: int = 8
    num_ranks: int = 6
    num_suits: int = 4
    num_joker_ids: int = 0
    num_blank_ids: int = 0
    num_joker_embeddings: int = 1
    num_blank_embeddings: int = 1
    base_deck_size: int = 24
    aggregator: str = "mean"
    use_layer_norm: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class CardEmbeddingEncoder(nn.Module):
    """
    Maps deck card indices to dense embeddings using separate rank/suit tables.

    Expects padded inputs of shape [batch, max_cards] with -1 denoting padding slots.
    Returns an aggregated embedding per batch element along with the valid card counts.
    """

    def __init__(self, config: CardEmbeddingConfig):
        super().__init__()
        self.config = config
        self.rank_embedding = nn.Embedding(config.num_ranks, config.rank_dim)
        self.suit_embedding = nn.Embedding(config.num_suits, config.suit_dim)
        out_dim = config.rank_dim + config.suit_dim
        self.output_dim = out_dim
        self._normalizer = nn.LayerNorm(out_dim) if config.use_layer_norm else None
        self.joker_embedding = None
        self.blank_embedding = None
        if config.num_joker_ids > 0 and config.num_joker_embeddings > 0:
            self.joker_embedding = nn.Embedding(config.num_joker_embeddings, out_dim)
        if config.num_blank_ids > 0 and config.num_blank_embeddings > 0:
            self.blank_embedding = nn.Embedding(config.num_blank_embeddings, out_dim)

    def forward(self, card_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            card_ids: tensor [batch, max_cards] with values in [-1, num_ranks*num_suits-1].
                      -1 marks padding and is excluded from the aggregation.

        Returns:
            embeddings: [batch, output_dim]
            counts: [batch, 1] float32 with the number of valid cards per sample
        """
        if card_ids.dim() != 2:
            raise ValueError(f"card_ids must have shape [batch, max_cards], got {tuple(card_ids.shape)}")

        device = card_ids.device
        dtype = self.rank_embedding.weight.dtype
        valid_mask = card_ids >= 0
        counts = valid_mask.sum(dim=1, keepdim=True).to(torch.float32)

        if not valid_mask.any():
            embeddings = torch.zeros((card_ids.size(0), self.output_dim), device=device, dtype=dtype)
            return embeddings, counts

        base = self.config.base_deck_size
        total_regular_cards = self.config.num_ranks * self.config.num_suits
        if base != total_regular_cards:
            base = total_regular_cards

        joker_start = base
        joker_end = joker_start + max(0, self.config.num_joker_ids)
        blank_start = joker_end
        blank_end = blank_start + max(0, self.config.num_blank_ids)

        regular_mask = valid_mask & (card_ids < base)
        joker_mask = valid_mask & (card_ids >= joker_start) & (card_ids < joker_end)
        blank_mask = valid_mask & (card_ids >= blank_start) & (card_ids < blank_end)
        unknown_mask = valid_mask & ~(regular_mask | joker_mask | blank_mask)
        if unknown_mask.any():
            raise ValueError("Encountered card ids outside configured ranges.")

        card_emb = torch.zeros(
            (card_ids.size(0), card_ids.size(1), self.output_dim),
            device=device,
            dtype=dtype,
        )

        if regular_mask.any():
            sanitized_regular = torch.where(regular_mask, card_ids, torch.zeros_like(card_ids))
            rank_ids = sanitized_regular // self.config.num_suits
            suit_ids = sanitized_regular % self.config.num_suits

            rank_emb = self.rank_embedding(rank_ids)
            suit_emb = self.suit_embedding(suit_ids)
            regular_emb = torch.cat((rank_emb, suit_emb), dim=-1)
            card_emb = card_emb + regular_emb * regular_mask.unsqueeze(-1)

        if joker_mask.any():
            if self.joker_embedding is not None:
                joker_indices = torch.where(
                    joker_mask,
                    card_ids - joker_start,
                    torch.zeros_like(card_ids),
                )
                if self.config.num_joker_embeddings > 0:
                    joker_indices = torch.remainder(joker_indices, self.config.num_joker_embeddings)
                joker_emb = self.joker_embedding(joker_indices)
                card_emb = card_emb + joker_emb * joker_mask.unsqueeze(-1)

        if blank_mask.any():
            if self.blank_embedding is not None:
                blank_indices = torch.where(
                    blank_mask,
                    card_ids - blank_start,
                    torch.zeros_like(card_ids),
                )
                if self.config.num_blank_embeddings > 0:
                    blank_indices = torch.remainder(blank_indices, self.config.num_blank_embeddings)
                blank_emb = self.blank_embedding(blank_indices)
                card_emb = card_emb + blank_emb * blank_mask.unsqueeze(-1)

        card_emb = card_emb * valid_mask.unsqueeze(-1)

        summed = card_emb.sum(dim=1)
        if self.config.aggregator == "sum":
            aggregated = summed
        elif self.config.aggregator == "mean":
            denom = torch.clamp(counts, min=1.0)
            aggregated = torch.where(
                counts > 0,
                summed / denom,
                torch.zeros_like(summed),
            )
        else:
            raise ValueError(f"Unsupported aggregator '{self.config.aggregator}'")

        if self._normalizer is not None:
            aggregated = self._normalizer(aggregated)

        return aggregated, counts
