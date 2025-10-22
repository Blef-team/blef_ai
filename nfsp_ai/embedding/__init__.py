"""
Embedding utilities for NFSP Blef.

This package currently exposes the card embedding encoder used for both
pretraining and downstream NFSP integration.
"""

from .encoder import CardEmbeddingEncoder, CardEmbeddingConfig

__all__ = ["CardEmbeddingEncoder", "CardEmbeddingConfig"]
