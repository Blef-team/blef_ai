"""
Lightweight inference wrapper for the NFSP Blef agent.

This module loads an exported inference checkpoint (see `nfsp_run_local.py --export-inference`)
alongside optional card/history embedding artifacts.  It exposes a single helper,
`determine_action(game_state)`, which mirrors the ConservativeAgent interface used in production.
"""

from __future__ import annotations

import os
from dataclasses import replace
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import torch

from nfsp_ai.agent import NFSPAgent, NFSPConfig
from nfsp_ai.nfsp_run_local import (
    _deck_spec_from_game,
    vectorize_obs,
    _legal_action_mask,
    _load_card_embedding,
    _load_history_embedding,
    CardEmbeddingRuntime,
    HistoryEmbeddingRuntime,
)

# ---------------------------------------------------------------------------
# Defaults (override via environment variables in Lambda / CLI wrappers)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = os.environ.get("NFSP_MODEL_PATH", "artifacts/nfsp_inference.pt")
DEFAULT_CARD_EMBED_PATH = os.environ.get("NFSP_CARD_EMBEDDING")
DEFAULT_HISTORY_EMBED_PATH = os.environ.get("NFSP_HISTORY_EMBEDDING")
DEFAULT_DEVICE = os.environ.get("NFSP_DEVICE", "cpu")
DEFAULT_GREEDY = os.environ.get("NFSP_GREEDY", "1") not in {"0", "false", "False"}


def _load_card_embedding_map(path: Optional[str], device: torch.device) -> dict[int, CardEmbeddingRuntime]:
    candidates: list[str] = []
    seen: set[str] = set()
    if path:
        candidates.append(path)
    else:
        if DEFAULT_CARD_EMBED_PATH:
            candidates.append(DEFAULT_CARD_EMBED_PATH)
        candidates.extend(
            [
                "artifacts/card_embedding_pretrain_32.pt",
                "artifacts/card_embedding_pretrain_24.pt",
                "artifacts/card_embedding_pretrain.pt",
            ]
        )
    embeddings: dict[int, CardEmbeddingRuntime] = {}
    for cand in candidates:
        if not cand or cand in seen or not os.path.exists(cand):
            continue
        seen.add(cand)
        try:
            runtime = _load_card_embedding(cand, device=str(device))
        except FileNotFoundError:
            continue
        embeddings[int(runtime.config.base_deck_size)] = runtime
        if path:
            break
    return embeddings


def _load_history_embedding_map(path: Optional[str], device: torch.device) -> dict[int, HistoryEmbeddingRuntime]:
    candidates: list[str] = []
    seen: set[str] = set()
    if path:
        candidates.append(path)
    else:
        if DEFAULT_HISTORY_EMBED_PATH:
            candidates.append(DEFAULT_HISTORY_EMBED_PATH)
        candidates.extend(
            [
                "artifacts/history_embedding_pretrain_32.pt",
                "artifacts/history_embedding_pretrain_24.pt",
                "artifacts/history_embedding_pretrain.pt",
            ]
        )
    embeddings: dict[int, HistoryEmbeddingRuntime] = {}
    for cand in candidates:
        if not cand or cand in seen or not os.path.exists(cand):
            continue
        seen.add(cand)
        try:
            runtime = _load_history_embedding(cand, device=str(device))
        except FileNotFoundError:
            continue
        embeddings[int(runtime.config.base_deck_size)] = runtime
        if path:
            break
    return embeddings


class NFSPProductionAgent:
    """
    Thin inference wrapper around `NFSPAgent`.  Loads policy weights and embeds a single
    call to `select_action` with deterministic (greedy) behaviour by default.
    """

    def __init__(
        self,
        model_path: str,
        card_embedding_path: Optional[str] = None,
        history_embedding_path: Optional[str] = None,
        device: str = DEFAULT_DEVICE,
        greedy: bool = DEFAULT_GREEDY,
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Inference checkpoint not found at '{model_path}'")

        self.device = torch.device(device)
        self.greedy = bool(greedy)

        checkpoint = torch.load(model_path, map_location=self.device)
        if "obs_dim" not in checkpoint or "act_dim" not in checkpoint:
            raise ValueError(
                "Checkpoint missing obs_dim/act_dim. Please export with `nfsp_run_local.py --export-inference`."
            )

        ckpt_cfg = checkpoint.get("cfg", {})
        base_cfg = NFSPConfig()
        cfg = replace(
            base_cfg,
            hidden=ckpt_cfg.get("hidden", base_cfg.hidden),
            anticipatory_eta=ckpt_cfg.get("anticipatory_eta", base_cfg.anticipatory_eta),
        )
        # Minimise buffer sizes and training metadata for inference-only usage
        cfg.rl_capacity = 1
        cfg.sl_capacity = 1
        cfg.batch_rl = 1
        cfg.batch_sl = 1
        cfg.train_rl_every = 1
        cfg.train_sl_every = 1
        cfg.warmup_steps = 0

        self.agent = NFSPAgent(
            obs_dim=int(checkpoint["obs_dim"]),
            act_dim=int(checkpoint["act_dim"]),
            device=self.device,
            cfg=cfg,
        )
        self.agent.q.load_state_dict(checkpoint["q"])
        self.agent.pi.load_state_dict(checkpoint["pi"])
        self.agent.q.eval()
        self.agent.pi.eval()
        # Release replay/supervised buffers immediately to keep memory low
        self.agent.rl_buf = None
        self.agent.sl_buf = None

        self.card_embeddings = _load_card_embedding_map(card_embedding_path, self.device)
        self.history_embeddings = _load_history_embedding_map(history_embedding_path, self.device)
        if not self.card_embeddings:
            print("[embeddings] no card embedding artifacts loaded; using multi-hot card features.")
        if not self.history_embeddings:
            print("[embeddings] no history embedding artifacts loaded; using legacy history multi-hot features.")

    @torch.no_grad()
    def determine_action(self, game_state: dict) -> int:
        """
        Compute the greedy action for the current player described by `game_state`.
        """
        cp = game_state.get("cp_nickname")
        if not cp:
            raise ValueError("Game state missing 'cp_nickname'")

        deck_size = int(game_state.get("rules", {}).get("deck_size", 24))
        card_embedding = self.card_embeddings.get(deck_size)
        history_embedding = self.history_embeddings.get(deck_size)
        spec = _deck_spec_from_game(
            game_state,
            card_embedding=card_embedding,
            history_embedding=history_embedding,
        )
        obs_vec, pub_prior = vectorize_obs(
            game_state,
            cp,
            spec,
            card_embedding=card_embedding,
            history_embedding=history_embedding,
        )
        mask = _legal_action_mask(game_state, spec, pub_prior)
        if mask.sum() <= 0:
            raise RuntimeError("No legal actions available for current game state")

        obs_tensor = torch.from_numpy(np.asarray(obs_vec, dtype=np.float32)).to(self.device)
        mask_tensor = mask.to(device=self.device, dtype=torch.float32)
        action = self.agent.select_action(
            obs_tensor,
            mask_tensor,
            use_average_policy=True,
            greedy=self.greedy,
        )
        return int(action)


# ---------------------------------------------------------------------------
# Module-level singleton used by production code (e.g., AWS Lambda handler)
# ---------------------------------------------------------------------------

_PRODUCTION_AGENT: Optional[NFSPProductionAgent] = None


def load_agent(
    model_path: Optional[str] = None,
    *,
    card_embedding_path: Optional[str] = None,
    history_embedding_path: Optional[str] = None,
    device: str = DEFAULT_DEVICE,
    greedy: bool = DEFAULT_GREEDY,
) -> NFSPProductionAgent:
    """
    Construct (or return) the cached production agent. Paths default to environment variables.
    """
    global _PRODUCTION_AGENT
    if _PRODUCTION_AGENT is None:
        model = model_path or DEFAULT_MODEL_PATH
        card_path = card_embedding_path or DEFAULT_CARD_EMBED_PATH
        history_path = history_embedding_path or DEFAULT_HISTORY_EMBED_PATH
        _PRODUCTION_AGENT = NFSPProductionAgent(
            model,
            card_embedding_path=card_path,
            history_embedding_path=history_path,
            device=device,
            greedy=greedy,
        )
    return _PRODUCTION_AGENT


def determine_action(game_state: dict) -> int:
    """
    Public entry point mirroring the legacy `agent.determine_action` API.
    """
    return load_agent().determine_action(game_state)
