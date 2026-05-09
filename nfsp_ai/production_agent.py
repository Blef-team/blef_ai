"""
Lightweight inference wrapper for the NFSP Blef agent.

This module loads an exported inference checkpoint (see `nfsp_run_local.py --export-inference`)
alongside optional card/history embedding artifacts.  It exposes a single helper,
`determine_action(game_state)`, which mirrors the ConservativeAgent interface used in production.
"""

from __future__ import annotations

import glob
import os
import re
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
from shared.game_utils import GameRules

# ---------------------------------------------------------------------------
# Defaults (override via environment variables in Lambda / CLI wrappers)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = os.environ.get("NFSP_MODEL_PATH", "artifacts/nfsp_inference.pt")
DEFAULT_CARD_EMBED_PATH = os.environ.get("NFSP_CARD_EMBEDDING")
DEFAULT_HISTORY_EMBED_PATH = os.environ.get("NFSP_HISTORY_EMBEDDING")
DEFAULT_DEVICE = os.environ.get("NFSP_DEVICE", "cpu")
DEFAULT_GREEDY = os.environ.get("NFSP_GREEDY", "1") not in {"0", "false", "False"}


# ---------------------------------------------------------------------------
# Variant-aware model routing
# ---------------------------------------------------------------------------
#
# Naming convention for inference artifacts:
#
#     artifacts/nfsp_inference_<deck>_<variant>.pt   (preferred)
#     artifacts/nfsp_inference_<deck>.pt              (legacy fallback)
#
# `<deck>` is "24" or "32"; `<variant>` is one of:
#
#     "1v1"   — exactly two players, jokers=blanks=common_cards=0, no teams
#     "multi" — anything that isn't 1v1 (multi-player, jokers, blanks,
#               common cards, or any combination)
#     "team"  — reserved for future team / partnership game variants
#
# Multiple checkpoints can coexist; at inference, `_routing_keys` produces
# a priority chain (most specific first). The first key with a loaded
# agent wins. The bare `<deck>` key acts as a final fallback so legacy
# deployments without variant-specific files keep working unchanged.

_INFERENCE_FILENAME_RE = re.compile(
    r"^nfsp_inference_(?P<deck>24|32)(?:_(?P<variant>[A-Za-z0-9]+))?\.pt$"
)
_KNOWN_VARIANTS = ("1v1", "multi", "team")


def _model_key_from_filename(path: str) -> Optional[str]:
    """Map an inference file path to its routing key.

    Returns None if the filename doesn't match the convention.
    """
    name = os.path.basename(path)
    m = _INFERENCE_FILENAME_RE.match(name)
    if not m:
        return None
    deck = m.group("deck")
    variant = m.group("variant")
    if variant is None:
        return deck  # legacy
    return f"{deck}_{variant}"


def _routing_keys(rules: dict, n_active: int, players: Optional[list] = None) -> list[str]:
    """Return preferred routing keys, most-specific first.

    Routes by *active* player count (players with cards remaining), not
    seated count. A 4-player game collapsed to 2 active players is
    structurally a 1v1 subtree and routes to the 1v1 specialist —
    provided the obs is canonicalized to drop eliminated seats before
    inference (see `_canonicalize_to_active`).

    The caller walks this list and picks the first key with a loaded
    agent. Always ends with the bare `<deck>` legacy key so a
    single-model deployment continues to work.

    Team mode is detected by inspecting `players[*].team` (the engine
    schema). A players list with any non-None team value means team
    mode. The legacy `rules.teams` flag is also honoured for callers
    that pre-compute it.
    """
    deck = int(rules.get("deck_size", 24))
    is_team = bool(rules.get("teams", False))
    if not is_team and players:
        is_team = any(p.get("team") is not None for p in players)
    j = int(rules.get("jokers", 0))
    b = int(rules.get("blanks", 0))
    cc = int(rules.get("common_cards", 0))
    is_1v1 = (n_active == 2 and j == 0 and b == 0 and cc == 0 and not is_team)

    keys: list[str] = []
    if is_team:
        keys.append(f"{deck}_team")
    if is_1v1:
        keys.append(f"{deck}_1v1")
    keys.append(f"{deck}_multi")
    keys.append(str(deck))  # legacy fallback
    # de-dupe while preserving order
    out: list[str] = []
    seen: set[str] = set()
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def _canonicalize_to_active(game_state: dict) -> dict:
    """Return a shallow copy of `game_state` with eliminated players removed.

    Players with `n_cards == 0` are dropped from the players list so the
    seat block in `vectorize_obs` matches a fresh game at the same
    active-player count. The 1v1 specialist trained without
    eliminations sees in-distribution input on collapsed games.

    History is round-scoped and only contains active-player events
    within the current round (eliminations happen at round boundaries),
    so it does not need filtering.
    """
    players = game_state.get("players") or []
    active = [p for p in players if int(p.get("n_cards", 0)) > 0]
    if len(active) == len(players):
        return game_state
    out = dict(game_state)
    out["players"] = active
    return out


def _resolve_model_paths(path: Optional[str]) -> dict[str, str]:
    """Discover inference checkpoints. Returns {routing_key: path}.

    If `path` is given, treat it as a single explicit checkpoint and
    derive its key from the filename (or use a synthesised key when the
    filename is non-standard).
    """
    out: dict[str, str] = {}
    if path:
        if not os.path.exists(path):
            return out
        key = _model_key_from_filename(path)
        if key is None:
            # Non-standard filename: store under a synthetic deck-only key
            # using the act_dim probe at load time. We cannot infer it from
            # the filename, so store under "explicit" and let the loader
            # rewrite the key.
            key = "explicit"
        out[key] = path
        return out

    # Discovery: scan artifacts/ for files matching the naming convention.
    # Search both relative ("artifacts/...") and an explicit absolute env
    # override (DEFAULT_MODEL_PATH) so deployments can point at a custom
    # directory.
    search_globs = ["artifacts/nfsp_inference_*.pt"]
    extra_dir = os.path.dirname(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH else ""
    if extra_dir and extra_dir not in {"", "artifacts"}:
        search_globs.append(os.path.join(extra_dir, "nfsp_inference_*.pt"))
    for pattern in search_globs:
        for cand in glob.glob(pattern):
            key = _model_key_from_filename(cand)
            if key is None:
                continue
            # First file wins per key; prefer the order produced by glob (alphabetical).
            out.setdefault(key, cand)
    return out


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
        num_actions = runtime.config.num_actions
        if abs(num_actions - GameRules(24).num_actions) < 2:
            deck_size = 24
        elif abs(num_actions - GameRules(32).num_actions) < 2:
            deck_size = 32
        embeddings[deck_size] = runtime
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
        model_path: Optional[str],
        card_embedding_path: Optional[str] = None,
        history_embedding_path: Optional[str] = None,
        device: str = DEFAULT_DEVICE,
        greedy: bool = DEFAULT_GREEDY,
    ) -> None:
        self.device = torch.device(device)
        self.greedy = bool(greedy)

        # agents: routing-key -> NFSPAgent
        # routing-key examples: "24" (legacy), "24_1v1", "24_multi", "32_team"
        self.agents: dict[str, NFSPAgent] = {}
        self.model_sources: dict[str, str] = {}
        for key, cand in _resolve_model_paths(model_path).items():
            ckpt = torch.load(cand, map_location=self.device)
            act_dim = int(ckpt.get("act_dim", 0))
            if act_dim == GameRules(24).num_actions:
                deck_size = 24
            elif act_dim == GameRules(32).num_actions:
                deck_size = 32
            else:
                raise ValueError(
                    f"Inference checkpoint '{cand}' has unsupported act_dim={act_dim}; "
                    "expected 89 (24-card) or 141 (32-card)."
                )
            # If the filename was non-standard ("explicit"), rewrite the
            # key from the probed deck size so the legacy fallback still
            # kicks in.
            if key == "explicit":
                key = str(deck_size)
            if key in self.agents:
                continue
            agent = self._build_agent_from_checkpoint(ckpt)
            self.agents[key] = agent
            self.model_sources[key] = cand

        if not self.agents:
            raise FileNotFoundError(
                "No inference checkpoints found. Provide NFSP_MODEL_PATH or place "
                "artifacts/nfsp_inference_<deck>[_<variant>].pt alongside the Lambda package."
            )

        self.card_embeddings = _load_card_embedding_map(card_embedding_path, self.device)
        self.history_embeddings = _load_history_embedding_map(history_embedding_path, self.device)
        if not self.card_embeddings:
            print("[embeddings] no card embedding artifacts loaded; using multi-hot card features.")
        if not self.history_embeddings:
            print("[embeddings] no history embedding artifacts loaded; using legacy history multi-hot features.")

    def _build_agent_from_checkpoint(self, checkpoint: dict) -> NFSPAgent:
        if "obs_dim" not in checkpoint or "act_dim" not in checkpoint:
            raise ValueError(
                "Checkpoint missing obs_dim/act_dim. Please export with `nfsp_run_local.py --export-inference`."
            )
        ckpt_cfg = checkpoint.get("cfg", {})
        base_cfg = NFSPConfig()
        # Hidden width: prefer the value saved in cfg, but fall back to
        # detecting it from the Q-net state dict (older exports omitted
        # `hidden` from cfg, so checkpoints with hidden != default would
        # mismatch on load_state_dict).
        hidden = ckpt_cfg.get("hidden")
        if hidden is None:
            q_sd = checkpoint.get("q", {})
            for key in ("net.0.weight", "trunk.0.weight"):
                if key in q_sd:
                    hidden = int(q_sd[key].shape[0])
                    break
            if hidden is None:
                hidden = base_cfg.hidden
        cfg = replace(
            base_cfg,
            hidden=int(hidden),
            anticipatory_eta=ckpt_cfg.get("anticipatory_eta", base_cfg.anticipatory_eta),
        )
        cfg.rl_capacity = 1
        cfg.sl_capacity = 1
        cfg.batch_rl = 1
        cfg.batch_sl = 1
        cfg.train_rl_every = 1
        cfg.train_sl_every = 1
        cfg.warmup_steps = 0

        agent = NFSPAgent(
            obs_dim=int(checkpoint["obs_dim"]),
            act_dim=int(checkpoint["act_dim"]),
            device=self.device,
            cfg=cfg,
        )
        agent.q.load_state_dict(checkpoint["q"])
        agent.pi.load_state_dict(checkpoint["pi"])
        agent.q.eval()
        agent.pi.eval()
        agent.rl_buf = None
        agent.sl_buf = None
        return agent

    @torch.no_grad()
    def determine_action(self, game_state: dict) -> int:
        """
        Compute the greedy action for the current player described by `game_state`.
        """
        cp = game_state.get("cp_nickname")
        if not cp:
            raise ValueError("Game state missing 'cp_nickname'")

        rules = game_state.get("rules", {}) or {}
        deck_size = int(rules.get("deck_size", 24))
        players = game_state.get("players", []) or []
        n_seated = len(players)
        n_active = sum(1 for p in players if int(p.get("n_cards", 0)) > 0)
        keys = _routing_keys(rules, n_active, players=players)
        agent = None
        chosen_key = None
        for key in keys:
            agent = self.agents.get(key)
            if agent is not None:
                chosen_key = key
                break
        if agent is None:
            available = ", ".join(sorted(self.agents.keys()))
            raise RuntimeError(
                f"No NFSP checkpoint loaded for deck={deck_size} "
                f"n_active={n_active} (seated={n_seated}). "
                f"Tried keys: {keys}. Available: {available or 'none'}."
            )
        # The 1v1 specialist trained without eliminations; canonicalize
        # the state (drop eliminated seats) so the obs matches its
        # training distribution. multi/team specialists trained on
        # games that include eliminations and expect the seat block
        # to retain eliminated slots, so they pass through unchanged.
        if chosen_key and chosen_key.endswith("_1v1") and n_active < n_seated:
            game_state = _canonicalize_to_active(game_state)
        # Embeddings remain keyed by deck size; variant doesn't change
        # the card / history vocabulary.
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
        action = agent.select_action(
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
        model = model_path
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
