from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class HistoryEmbeddingConfig:
    """Configuration for action-history encoder."""

    num_actions: int
    history_len: int = 8
    embedding_dim: int = 64
    use_positional: bool = True
    use_layer_norm: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class HistoryEmbeddingEncoder(nn.Module):
    """
    Encodes sequences of action IDs (with padding) into dense embeddings.
    """

    def __init__(self, config: HistoryEmbeddingConfig):
        super().__init__()
        self.config = config
        num_embeddings = config.num_actions + 1  # extra slot for PAD
        self.pad_idx = config.num_actions
        self.action_embed = nn.Embedding(num_embeddings, config.embedding_dim, padding_idx=self.pad_idx)
        if config.use_positional:
            self.position_embed = nn.Embedding(config.history_len, config.embedding_dim)
        else:
            self.position_embed = None
        self.layer_norm = nn.LayerNorm(config.embedding_dim) if config.use_layer_norm else None

    def forward(self, actions: torch.LongTensor, mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """
        Args:
            actions: [batch, history_len] tensor with -1 indicating padding.
            mask: optional [batch, history_len] bool tensor where True marks valid entries.

        Returns:
            embeddings: [batch, embedding_dim]
        """
        if actions.dim() != 2:
            raise ValueError("actions must have shape [batch, history_len]")
        batch, history_len = actions.shape
        if history_len != self.config.history_len:
            raise ValueError(
                f"actions length {history_len} does not match configured history_len {self.config.history_len}"
            )

        device = actions.device
        pad_idx = self.pad_idx
        if mask is None:
            mask = actions >= 0
        else:
            mask = mask.to(device=device)
            mask = mask & (actions >= 0)
        clamped = actions.clamp(min=0)
        clamped = torch.where(mask, clamped, torch.full_like(clamped, pad_idx))

        token_emb = self.action_embed(clamped)
        if self.position_embed is not None:
            positions = torch.arange(history_len, device=device).unsqueeze(0).expand(batch, -1)
            token_emb = token_emb + self.position_embed(positions)

        mask_f = mask.unsqueeze(-1).to(token_emb.dtype)
        summed = (token_emb * mask_f).sum(dim=1)
        counts = mask_f.sum(dim=1).clamp(min=1.0)
        mean_emb = summed / counts
        if self.layer_norm is not None:
            mean_emb = self.layer_norm(mean_emb)
        return mean_emb
