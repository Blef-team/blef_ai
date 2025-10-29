# nfsp_run_local.py

# RUN FROM REPOSITORY ROOT /

# Adapter for NFSP <-> simpleschema_local_manager with legality, jokers, and common cards.
import argparse
import os
import random
from datetime import datetime
import math
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import torch

import shared.api.simpleschema_local_manager as gm                 # local manager
from shared.probabilities.dynamic_probabilities import get_bet_probabilities, get_generic_bet_probabilities
from nfsp_ai.agent import NFSPAgent, NFSPConfig, _evaluate_policy          # your NFSP implementation
from nfsp_ai.control_plane import JsonControlPlane, build_control_snapshot, write_control_file
from nfsp_ai.embedding import (
    CardEmbeddingEncoder,
    CardEmbeddingConfig,
    HistoryEmbeddingEncoder,
    HistoryEmbeddingConfig,
)
from shared.game_utils import GameRules


@dataclass
class DeckSpec:
    deck_size: int
    num_values: int
    num_actions: int
    check_action_id: int
    hand_vec_dim: int
    hist_dim: int
    obs_dim: int
    card_feature_dim: int
    history_feature_dim: int
    use_card_embeddings: bool
    use_history_embeddings: bool


def _compute_obs_dim(
    hand_vec_dim: int,
    hist_dim: int,
    card_feature_dim: Optional[int] = None,
    history_feature_dim: Optional[int] = None,
) -> int:
    card_dim = card_feature_dim if card_feature_dim is not None else hand_vec_dim
    history_dim = history_feature_dim if history_feature_dim is not None else hist_dim
    return (
        card_dim
        + 1  # rules jokers
        + 1  # private jokers
        + 1  # common jokers
        + card_dim
        + MAX_PLAYERS
        + 1  # round scaling
        + history_dim
        + 1  # last bet prob
        + hist_dim  # private priors
        + hist_dim  # public priors
        + 1  # total blanks
    )


def _build_deck_spec(
    rules: Dict,
    card_embedding: Optional["CardEmbeddingRuntime"] = None,
    history_embedding: Optional["HistoryEmbeddingRuntime"] = None,
) -> DeckSpec:
    deck_size = int(rules.get("deck_size", 24))
    if deck_size % 4 != 0:
        raise ValueError(f"Unsupported deck_size {deck_size}; must be divisible by four.")
    gr = GameRules(deck_size)
    num_values = deck_size // 4
    num_actions = gr.num_actions
    check_action_id = gr.check_action_id
    hand_vec_dim = num_values * 4
    hist_dim = check_action_id
    use_card_embeddings = card_embedding is not None
    use_history_embeddings = history_embedding is not None
    if use_card_embeddings:
        base_deck = int(card_embedding.config.base_deck_size)
        if base_deck != deck_size:
            raise ValueError(
                f"Card embedding artifact deck size {base_deck} does not match environment deck size {deck_size}"
            )
        if card_embedding.rank_feature_dim != num_values:
            raise ValueError(
                f"Card embedding rank feature dim {card_embedding.rank_feature_dim} "
                f"does not match expected num values {num_values}"
            )
        card_feature_dim = card_embedding.feature_dim
    else:
        card_feature_dim = hand_vec_dim
    if use_history_embeddings:
        history_feature_dim = history_embedding.feature_dim
    else:
        history_feature_dim = hist_dim
    obs_dim = _compute_obs_dim(
        hand_vec_dim,
        hist_dim,
        card_feature_dim=card_feature_dim,
        history_feature_dim=history_feature_dim,
    )
    return DeckSpec(
        deck_size=deck_size,
        num_values=num_values,
        num_actions=num_actions,
        check_action_id=check_action_id,
        hand_vec_dim=hand_vec_dim,
        hist_dim=hist_dim,
        obs_dim=obs_dim,
        card_feature_dim=card_feature_dim,
        history_feature_dim=history_feature_dim,
        use_card_embeddings=use_card_embeddings,
        use_history_embeddings=use_history_embeddings,
    )

MAX_PLAYERS = 8
HISTORY_SLOTS = 8
HISTORY_LEN = 8
DEFAULT_CARD_EMBEDDING_PATH = "artifacts/card_embedding_pretrain.pt"
DEFAULT_HISTORY_EMBEDDING_PATH = "artifacts/history_embedding_pretrain.pt"


@dataclass
class CardEmbeddingRuntime:
    encoder: CardEmbeddingEncoder
    config: CardEmbeddingConfig
    device: torch.device
    feature_dim: int
    rank_feature_dim: int
    suit_feature_dim: int
    extra_feature_dim: int
    max_hand_cards: int
    num_joker_ids: int
    num_blank_ids: int
    source_path: Optional[str] = None


@dataclass
class HistoryEmbeddingRuntime:
    encoder: HistoryEmbeddingEncoder
    config: HistoryEmbeddingConfig
    device: torch.device
    feature_dim: int  # flattened across slots
    history_len: int
    slots: int
    num_actions: int
    source_path: Optional[str] = None


# ---------- Utilities ----------

def _nickname_to_idx(players: List[dict], nick: str) -> int:
    return next((i for i, p in enumerate(players) if p["nickname"] == nick), -1)


def _deck_spec_from_game(
    game: dict,
    card_embedding: Optional["CardEmbeddingRuntime"] = None,
    history_embedding: Optional["HistoryEmbeddingRuntime"] = None,
) -> DeckSpec:
    rules = game.get("rules", {}) or {}
    return _build_deck_spec(rules, card_embedding=card_embedding, history_embedding=history_embedding)


def _current_hand(game: dict, nick: str) -> List[dict]:
    for h in game.get("hands", []):
        if h["nickname"] == nick:
            return h["hand"]
    return []


def _count_jokers(cards: List[dict]) -> int:
    return sum(1 for c in cards if int(c.get("value", 0)) < 0)


def _last_bet_action_id(game: dict, spec: DeckSpec) -> int:
    """Return the last *bet* action id (< check_action_id) if any; else -1."""
    hist = game.get("history", []) or []
    if not hist:
        return -1
    check = spec.check_action_id
    for ev in reversed(hist):
        try:
            act = int(ev.get("action_id", -1))
        except Exception:
            continue
        if 0 <= act < check:
            return act
    return -1


def _cards_to_embedding_inputs(
    cards: List[dict],
    rules: dict,
    spec: DeckSpec,
    embedding: "CardEmbeddingRuntime",
) -> Tuple[np.ndarray, np.ndarray]:
    hand_ids = np.full((embedding.max_hand_cards,), -1, dtype=np.int64)
    rank_counts = np.zeros((embedding.rank_feature_dim,), dtype=np.float32)
    suit_counts = np.zeros((embedding.suit_feature_dim,), dtype=np.float32)
    joker_count = 0.0
    blank_count = 0.0

    joker_cap = max(0, embedding.num_joker_ids)
    blank_cap = max(0, embedding.num_blank_ids)
    joker_total = min(joker_cap, max(0, _rules_total_jokers(rules)))
    blank_total = min(blank_cap, max(0, _rules_total_blanks(rules)))

    base_deck = int(embedding.config.base_deck_size)
    joker_base = base_deck
    blank_base = base_deck + joker_cap

    regular_ids: List[int] = []
    joker_ids: List[int] = []
    blank_ids: List[int] = []

    for card in cards or []:
        value = int(card.get("value", -99))
        colour = int(card.get("colour", -99))
        if value >= 0 and colour >= 0:
            card_id = value * embedding.suit_feature_dim + colour
            regular_ids.append(card_id)
            if 0 <= value < rank_counts.shape[0]:
                rank_counts[value] += 1.0
            if 0 <= colour < suit_counts.shape[0]:
                suit_counts[colour] += 1.0
        elif value == -1:
            if joker_cap <= 0:
                raise ValueError("Encountered joker but embedding artifact has no joker slots")
            offset = min(len(joker_ids), max(joker_total - 1, 0))
            card_id = joker_base + offset
            joker_ids.append(card_id)
            joker_count += 1.0
        elif value == -2:
            if blank_cap <= 0:
                raise ValueError("Encountered blank but embedding artifact has no blank slots")
            offset = min(len(blank_ids), max(blank_total - 1, 0))
            card_id = blank_base + offset
            blank_ids.append(card_id)
            blank_count += 1.0

    ordered_ids = sorted(regular_ids) + sorted(joker_ids) + sorted(blank_ids)
    for idx, card_id in enumerate(ordered_ids[:embedding.max_hand_cards]):
        hand_ids[idx] = int(card_id)

    aux_features = np.concatenate(
        (
            rank_counts,
            suit_counts,
            np.array([joker_count, blank_count], dtype=np.float32),
        )
    ).astype(np.float32, copy=False)

    return hand_ids, aux_features


def _encode_hand_with_embedding(
    cards: List[dict],
    rules: dict,
    spec: DeckSpec,
    embedding: "CardEmbeddingRuntime",
) -> np.ndarray:
    hand_ids, aux = _cards_to_embedding_inputs(cards, rules, spec, embedding)
    device = embedding.device
    hand_tensor = torch.from_numpy(hand_ids).unsqueeze(0).to(device=device, dtype=torch.long)
    aux_tensor = torch.from_numpy(aux).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        emb, _ = embedding.encoder(hand_tensor)
    features = torch.cat((emb, aux_tensor), dim=1)
    return features.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)


def _load_card_embedding(
    artifact_path: str,
    device: Optional[str] = None,
) -> "CardEmbeddingRuntime":
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Card embedding artifact not found at '{artifact_path}'")

    payload = torch.load(artifact_path, map_location="cpu")
    cfg_dict = payload.get("encoder_config")
    if not isinstance(cfg_dict, dict):
        raise ValueError("encoder_config missing from embedding artifact")
    encoder_cfg = CardEmbeddingConfig(**cfg_dict)
    encoder = CardEmbeddingEncoder(encoder_cfg)
    state = payload.get("encoder_state_dict")
    if not isinstance(state, dict):
        raise ValueError("encoder_state_dict missing from embedding artifact")
    encoder.load_state_dict(state)
    encoder.eval()

    target_device = torch.device(device) if device else torch.device("cpu")
    encoder.to(target_device)
    hyperparams = payload.get("hyperparams", {}) or {}
    rank_feature_dim = int(hyperparams.get("rank_feature_dim", encoder_cfg.num_ranks))
    suit_feature_dim = int(hyperparams.get("suit_feature_dim", encoder_cfg.num_suits))
    extra_feature_dim = int(hyperparams.get("aux_feature_dim", rank_feature_dim + suit_feature_dim + 2))
    max_hand_cards = int(
        hyperparams.get(
            "max_hand_cards",
            encoder_cfg.base_deck_size + encoder_cfg.num_joker_ids + encoder_cfg.num_blank_ids,
        )
    )
    feature_dim = encoder.output_dim + extra_feature_dim
    if max_hand_cards <= 0:
        max_hand_cards = encoder_cfg.base_deck_size + encoder_cfg.num_joker_ids + encoder_cfg.num_blank_ids

    return CardEmbeddingRuntime(
        encoder=encoder,
        config=encoder_cfg,
        device=target_device,
        feature_dim=feature_dim,
        rank_feature_dim=rank_feature_dim,
        suit_feature_dim=suit_feature_dim,
        extra_feature_dim=extra_feature_dim,
        max_hand_cards=max_hand_cards,
        num_joker_ids=int(encoder_cfg.num_joker_ids),
        num_blank_ids=int(encoder_cfg.num_blank_ids),
        source_path=os.path.abspath(artifact_path),
    )


def _load_history_embedding(
    artifact_path: str,
    device: Optional[str] = None,
) -> "HistoryEmbeddingRuntime":
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"History embedding artifact not found at '{artifact_path}'")

    payload = torch.load(artifact_path, map_location="cpu")
    cfg_dict = payload.get("encoder_config")
    if not isinstance(cfg_dict, dict):
        raise ValueError("encoder_config missing from history embedding artifact")
    encoder_cfg = HistoryEmbeddingConfig(**cfg_dict)
    encoder = HistoryEmbeddingEncoder(encoder_cfg)
    state = payload.get("encoder_state_dict")
    if not isinstance(state, dict):
        raise ValueError("encoder_state_dict missing from history embedding artifact")
    encoder.load_state_dict(state)
    encoder.eval()

    target_device = torch.device(device) if device else torch.device("cpu")
    encoder.to(target_device)
    feature_dim = encoder.config.embedding_dim * HISTORY_SLOTS

    return HistoryEmbeddingRuntime(
        encoder=encoder,
        config=encoder_cfg,
        device=target_device,
        feature_dim=feature_dim,
        history_len=encoder_cfg.history_len,
        slots=HISTORY_SLOTS,
        num_actions=encoder_cfg.num_actions,
        source_path=os.path.abspath(artifact_path),
    )


def _legal_action_mask(game: dict, spec: DeckSpec, pub_prior: list) -> torch.Tensor:
    """Compute legality exactly as enforced by manager.play(),
    then *augment* with public priors by hard-zeroing actions whose public probability is 0.
    - Bets are action ids [0, check_id)
    - CHECK is action id == check_id
    """
    mask = np.zeros((spec.num_actions,), dtype=np.float32)
    check = int(spec.check_action_id)
    hist = game.get("history", []) or []
    # Base legality from history:
    if not hist:
        # Start of round: only bets are legal (CHECK not allowed)
        mask[:check] = 1.0
    else:
        try:
            last = int((hist[-1] or {}).get("action_id", -1))
        except Exception:
            last = -1
        # If last action was CHECK, the round is resolved; no actions legal
        if last == check:
            return torch.from_numpy(mask)
        next_min = min(check, last + 1) if last >= 0 else 0
        if next_min < check:
            mask[next_min:check] = 1.0
        # CHECK legal once at least one bet has happened
        mask[check] = 1.0
    # ---- overlay public priors (hard mask zeros with tolerance) ----
    # pub_prior covers only bet actions; append a slot for CHECK
    pri = np.asarray(list(pub_prior) + [1.0], dtype=np.float32)
    if pri.shape[0] != spec.num_actions:
        raise ValueError(f"pub_prior length {pri.shape[0]} != num_actions {spec.num_actions}")
    # Anything <= eps is treated as 0-prob (publicly impossible)
    PUBLIC_PRIOR_EPS = 1e-9
    mask *= (pri > PUBLIC_PRIOR_EPS).astype(np.float32)
    return torch.from_numpy(mask)


# ---------- Observation encoding ----------

def _seat_index(players: List[dict], nick: str) -> int:
    for i, p in enumerate(players or []):
        if p.get("nickname") == nick:
            return i
    return 0

def _rotate_list(xs: List[int], k: int) -> List[int]:
    if not xs:
        return xs
    k %= len(xs)
    return xs[k:] + xs[:k]

def _rules_total_jokers(rules: dict) -> int:
    """
    Total possible jokers per rules (not just seen on table).
    Accepts either:
      - rules["jokers"] as int, or
      - rules["jokers"] as dict with "num"
    Fallback: 0
    """
    j = rules.get("jokers", 0)
    if isinstance(j, dict):
        return int(j.get("num", 0))
    return int(j)

def _rules_total_blanks(rules: dict) -> int:
    """Total possible blanks per rules (not just seen on table)."""
    return int(rules.get("blanks", 0))

def _card_index(value: int, colour: int, spec: DeckSpec) -> int:
    if value < 0 or colour < 0:
        return -1
    if value >= spec.num_values or colour >= 4:
        return -1
    return value * 4 + colour


def _multi_hot_cards(cards: List[dict], spec: DeckSpec) -> np.ndarray:
    vec = np.zeros((spec.hand_vec_dim,), dtype=np.float32)
    for card in cards or []:
        v = int(card.get("value", -2))
        c = int(card.get("colour", -2))
        idx = _card_index(v, c, spec)
        if idx >= 0:
            vec[idx] = 1.0
    return vec

def _count_jokers_from_ints(cards: List[dict]) -> int:
    return sum(1 for c in (cards or []) if int(c.get("value", -2)) == -1)

def _count_blanks_from_ints(cards: List[dict]) -> int:
    return sum(1 for c in (cards or []) if int(c.get("value", -2)) == -2)

def _current_hand_int(game: dict, nick: str) -> List[dict]:
    for h in game.get("hands", []) or []:
        if h.get("nickname") == nick:
            return h.get("hand", []) or []
    return []

def _common_hand_int(game: dict) -> List[dict]:
    return game.get("common_hand", []) or []

def _round_history_action_ids(game: dict, hist_dim: int) -> List[int]:
    ids = []
    for ev in game.get("history", []) or []:
        try:
            a = int(ev.get("action_id"))
            if 0 <= a < hist_dim:
                ids.append(a)
        except Exception:
            continue
    return ids


_HISTORY_SLOT_MAP = {
    2: [None, None, None, 0, 1, None, None, None],
    3: [None, None, 2, 0, 1, None, None, None],
    4: [None, None, 3, 0, 1, 2, None, None],
    5: [None, 3, 4, 0, 1, 2, None, None],
    6: [None, 4, 5, 0, 1, 2, 3, None],
    7: [4, 5, 6, 0, 1, 2, 3, None],
    8: [5, 6, 7, 0, 1, 2, 3, 4],
}


def _ordered_history_slots(players: List[dict], current_idx: int) -> List[Optional[str]]:
    n = len(players)
    mapping = _HISTORY_SLOT_MAP.get(n)
    if mapping is None:
        raise ValueError(f"Unsupported player count {n} for history embeddings")
    ordered = [players[(current_idx + offset) % n] for offset in range(n)]
    slots: List[Optional[str]] = [None] * HISTORY_SLOTS
    for slot_idx, rel in enumerate(mapping):
        if rel is None or rel >= n:
            continue
        slots[slot_idx] = ordered[rel]["nickname"]
    return slots


def _collect_player_histories(
    game: dict,
    spec: DeckSpec,
    history_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    players = game.get("players", []) or []
    if not players:
        return (
            np.full((HISTORY_SLOTS, history_len), -1, dtype=np.int64),
            np.zeros((HISTORY_SLOTS, history_len), dtype=bool),
        )
    cp_nick = game.get("cp_nickname")
    cp_idx = _nickname_to_idx(players, cp_nick)
    if cp_idx < 0:
        cp_idx = 0
    slot_order = _ordered_history_slots(players, cp_idx)

    history_map: Dict[str, List[int]] = {p.get("nickname"): [] for p in players}
    for event in reversed(game.get("history", []) or []):
        try:
            action_id = int(event.get("action_id", -1))
        except Exception:
            continue
        if action_id < 0 or action_id >= spec.hist_dim:
            continue
        nick = event.get("player")
        if nick not in history_map:
            continue
        seq = history_map[nick]
        if len(seq) < history_len:
            seq.append(action_id)

    actions = np.full((HISTORY_SLOTS, history_len), -1, dtype=np.int64)
    mask = np.zeros((HISTORY_SLOTS, history_len), dtype=bool)
    for slot_idx, nick in enumerate(slot_order):
        if nick is None:
            continue
        seq = history_map.get(nick, [])
        limit = min(len(seq), history_len)
        if limit == 0:
            continue
        actions[slot_idx, :limit] = seq[:limit]
        mask[slot_idx, :limit] = True
    return actions, mask


def _encode_histories_with_embedding(
    actions: np.ndarray,
    mask: np.ndarray,
    runtime: "HistoryEmbeddingRuntime",
) -> np.ndarray:
    if actions.shape != (runtime.slots, runtime.history_len):
        raise ValueError("actions shape mismatch for history embedding")
    device = runtime.device
    tensor_actions = torch.from_numpy(actions).to(device=device, dtype=torch.long)
    tensor_mask = torch.from_numpy(mask).to(device=device, dtype=torch.bool)
    with torch.no_grad():
        emb = runtime.encoder(tensor_actions, tensor_mask)
    return emb.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)


def vectorize_obs(
    game: dict,
    nick: str,
    spec: DeckSpec,
    card_embedding: Optional["CardEmbeddingRuntime"] = None,
    history_embedding: Optional["HistoryEmbeddingRuntime"] = None,
) -> Tuple[float, List[float]]:
    """
    Updated observation for current player (nick) with minimal allocations.
    """
    rules = game.get("rules", {}) or {}
    players = game.get("players", []) or []

    obs = np.zeros((spec.obs_dim,), dtype=np.float32)
    idx = 0

    # Private hand multi-hot
    my_hand = _current_hand_int(game, nick)
    if spec.use_card_embeddings and card_embedding is not None:
        priv_feat = _encode_hand_with_embedding(my_hand, rules, spec, card_embedding)
        if priv_feat.shape[0] != spec.card_feature_dim:
            raise ValueError(
                f"Card embedding produced dim {priv_feat.shape[0]}, expected {spec.card_feature_dim}"
            )
        obs[idx:idx + spec.card_feature_dim] = priv_feat
        idx += spec.card_feature_dim
    else:
        obs[idx:idx + spec.hand_vec_dim] = _multi_hot_cards(my_hand, spec)
        idx += spec.hand_vec_dim

    # Totals & joker counts
    obs[idx] = float(_rules_total_jokers(rules)); idx += 1
    obs[idx] = float(_count_jokers_from_ints(my_hand)); idx += 1
    common_hand = _common_hand_int(game)
    obs[idx] = float(_count_jokers_from_ints(common_hand)); idx += 1

    # Common cards features
    if spec.use_card_embeddings and card_embedding is not None:
        common_feat = _encode_hand_with_embedding(common_hand, rules, spec, card_embedding)
        if common_feat.shape[0] != spec.card_feature_dim:
            raise ValueError(
                f"Card embedding produced dim {common_feat.shape[0]}, expected {spec.card_feature_dim}"
            )
        obs[idx:idx + spec.card_feature_dim] = common_feat
        idx += spec.card_feature_dim
    else:
        obs[idx:idx + spec.hand_vec_dim] = _multi_hot_cards(common_hand, spec)
        idx += spec.hand_vec_dim

    # Seat counts rotated so current player is seat 0
    my_seat = _seat_index(players, game.get("cp_nickname", nick))
    counts_map = {p.get("nickname"): int(p.get("n_cards", 0)) for p in players}
    try:
        slot_order = _ordered_history_slots(players, my_seat)
    except ValueError:
        slot_order = [players[(my_seat + offset) % len(players)].get("nickname") for offset in range(len(players))]
        slot_order += [None] * (MAX_PLAYERS - len(slot_order))
    ordered_counts = np.zeros((MAX_PLAYERS,), dtype=np.float32)
    for slot_idx in range(min(MAX_PLAYERS, len(slot_order))):
        nick = slot_order[slot_idx] if slot_idx < len(slot_order) else None
        if nick is None:
            ordered_counts[slot_idx] = 0.0
        else:
            ordered_counts[slot_idx] = float(counts_map.get(nick, 0))
    obs[idx:idx + MAX_PLAYERS] = ordered_counts
    idx += MAX_PLAYERS

    # Round scaling
    cur_round = int(game.get("round_number", 1))
    max_rounds = int(rules.get("max_rounds", 4))
    obs[idx] = float(min(max(cur_round / max(1, max_rounds), 0.0), 1.0))
    idx += 1

    # Action history features
    if spec.use_history_embeddings and history_embedding is not None:
        actions_mat, mask_mat = _collect_player_histories(game, spec, history_embedding.history_len)
        hist_feat = _encode_histories_with_embedding(actions_mat, mask_mat, history_embedding)
        if hist_feat.shape[0] != spec.history_feature_dim:
            raise ValueError(
                f"History embedding produced dim {hist_feat.shape[0]}, expected {spec.history_feature_dim}"
            )
        obs[idx:idx + spec.history_feature_dim] = hist_feat
        idx += spec.history_feature_dim
    else:
        hist_ids = _round_history_action_ids(game, spec.hist_dim)
        if hist_ids:
            obs[idx + np.unique(hist_ids)] = 1.0
        idx += spec.hist_dim

    # Last bet probability
    last_bet_id = _last_bet_action_id(game, spec)
    prob_last_bet_exists = get_bet_probabilities(
        game_state=game,
        for_betting=False,
        specific_action_id=last_bet_id,
    )
    obs[idx] = float(prob_last_bet_exists)
    idx += 1

    # Private priors (depends on my hand)
    pvt_slice = slice(idx, idx + spec.hist_dim)
    pvt_prior = get_bet_probabilities(
        game_state=game,
        for_betting=True,
        last_bet=last_bet_id,
    )
    obs[pvt_slice] = pvt_prior
    idx += spec.hist_dim

    # Public priors
    pub_slice = slice(idx, idx + spec.hist_dim)
    pub_prior = get_generic_bet_probabilities(
        game_state=game,
        last_bet=last_bet_id,
    )
    obs[pub_slice] = pub_prior
    idx += spec.hist_dim

    # Total blanks
    obs[idx] = float(_rules_total_blanks(rules))
    idx += 1

    if idx != spec.obs_dim:
        raise ValueError(f"vectorize_obs produced dim {idx}, expected {spec.obs_dim}")

    # Validations on probability slices
    if len(pvt_prior) != spec.hist_dim:
        raise ValueError(
            f"get_bet_probabilities() returned length {len(pvt_prior)}, expected {spec.hist_dim}"
        )
    if len(pub_prior) != spec.hist_dim:
        raise ValueError(
            f"get_generic_bet_probabilities() returned length {len(pub_prior)}, expected {spec.hist_dim}"
        )

    if not np.isfinite(obs[pvt_slice]).all():
        raise ValueError("get_bet_probabilities() returned non-finite values")
    if not np.isfinite(obs[pub_slice]).all():
        raise ValueError("get_generic_bet_probabilities() returned non-finite values")

    if not ((0.0 <= obs[pvt_slice]).all() and (obs[pvt_slice] <= 1.0).all()):
        raise ValueError("get_bet_probabilities() returned values outside [0, 1]")
    if not ((0.0 <= obs[pub_slice]).all() and (obs[pub_slice] <= 1.0).all()):
        raise ValueError("get_generic_bet_probabilities() returned values outside [0, 1]")

    return torch.from_numpy(obs).float(), pub_prior

# ---------- Environment Adapter ----------

class MyEnv:
    """
    Minimal TurnEnvAdapter for nfsp_complex.NFSPAgent:
      reset() -> (obs, mask, pid)
      step(action) -> (obs, mask, reward, done, pid)

    reward: ±1 on CHECK resolution, measured from a fixed reference player's perspective.
            0 otherwise.
    done: True only when the game finishes or after the reference player is eliminated.
    """

    def __init__(
        self,
        n_agents: int = 2,
        max_cards: int = 11,
        deck_size: int = 24,
        jokers: int = 0,
        blanks: int = 0,
        common_cards: int = 0,
        verbose: bool = False,
        illegal_penalty: float = -0.01,
        game_save_dir: Optional[str] = None,
        save_sample_rate: int = 5000,
        card_embedding: Optional["CardEmbeddingRuntime"] = None,
        history_embedding: Optional["HistoryEmbeddingRuntime"] = None,
        reset_schedules_to: int = 100_000,
        reset_schedules_at: int = 0,
        pick_n_agents_in_range: bool = False,
        pick_jokers_in_range: bool = False,
        pick_blanks_in_range: bool = False,
        pick_common_cards_in_range: bool = False,
    ):
        if n_agents < 2 or n_agents > 8:
            raise ValueError("n_agents must be in [2, 8]")

        self.n_agents = n_agents
        self.max_cards = max_cards
        self.joker_cap = max(0, int(jokers))
        self.blank_cap = max(0, int(blanks))
        self.common_card_cap = max(0, int(common_cards))
        self.rules = {
            "deck_size": int(deck_size),
            "jokers": self.joker_cap,
            "blanks": self.blank_cap,
            "common_cards": self.common_card_cap,
        }

        self.card_embedding = card_embedding
        self.history_embedding = history_embedding
        self.deck_spec = _build_deck_spec(
            self.rules,
            card_embedding=self.card_embedding,
            history_embedding=self.history_embedding,
        )

        self.verbose = verbose
        self.illegal_penalty = float(illegal_penalty)
        self.game_save_dir = game_save_dir
        self.save_sample_rate = max(1, int(save_sample_rate))

        self.reset_schedules_to = reset_schedules_to
        self.reset_schedules_at = reset_schedules_at

        self.pick_n_agents_in_range = pick_n_agents_in_range
        self.pick_jokers_in_range = pick_jokers_in_range
        self.pick_blanks_in_range = pick_blanks_in_range
        self.pick_common_cards_in_range = pick_common_cards_in_range

        self.game: Dict = {}
        self._last_obs = None
        self._last_mask = None
        self._ref_nick: Optional[str] = None

        # --- Added state for multi-round handling ---
        self.rounds_since_reset = 0           # counts rounds in current game
        self._round_boundary_pending = False  # True if last step ended a round (not full game)
        # --------------------------------------------

        self._save_counter = 0
        self._saved_games = 0


    def _pid(self) -> int:
        return _nickname_to_idx(self.game["players"], self.game.get("cp_nickname", ""))

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Reset for the learner:
        - If the last step ended a ROUND (self._round_boundary_pending == True) and the game is still running,
          continue the SAME GAME at the start of the next round, but as a NEW EPISODE for learning.
        - Otherwise (game finished or no game yet), create a brand new game.
        """
        # Fast path: start of next round within the same game, as a new learning episode
        if getattr(self, "_round_boundary_pending", False) and self.game and self.game.get("status") != "Finished":
            self._round_boundary_pending = False  # we've consumed the boundary
            # Keep rules and ref nick as-is; do NOT reset rounds_since_reset here.
            # Rebuild deck_spec in case embeddings/rules changed.
            self.deck_spec = _deck_spec_from_game(
                self.game,
                card_embedding=self.card_embedding,
                history_embedding=self.history_embedding,
            )
            cp = self.game.get("cp_nickname")
            obs, pub_prior = vectorize_obs(
                self.game,
                cp,
                self.deck_spec,
                card_embedding=self.card_embedding,
                history_embedding=self.history_embedding,
            )
            mask = _legal_action_mask(self.game, self.deck_spec, pub_prior).float()
            pid = self._pid()
            self._last_obs, self._last_mask = obs, mask
            return obs, mask, pid

        n_agents = self.n_agents
        if self.pick_n_agents_in_range:
            n_agents = random.choice(range(2, self.n_agents + 1))  # Run a game with up to n_agents
        jokers = self.joker_cap
        if self.pick_jokers_in_range:
            jokers = random.randint(0, self.joker_cap)
        blanks = self.blank_cap
        if self.pick_blanks_in_range:
            blanks = random.randint(0, self.blank_cap)
        common_cards = self.common_card_cap
        if self.pick_common_cards_in_range:
            common_cards = random.randint(0, self.common_card_cap)
        # Normal path: start a completely new game (either there was no pending round boundary or the game finished)
        self.game = gm.create_game(
            n_agents,
            deck_size=self.rules["deck_size"],
            max_cards=self.max_cards,
            jokers=jokers,
            blanks=blanks,
            common_cards=common_cards,
            verbose=self.verbose,
        )
        # Sync rules from the created game (single assignment; remove duplicate)
        self.rules = dict(self.game.get("rules", self.rules))

        # Fresh deck spec for the new game
        self.deck_spec = _deck_spec_from_game(
            self.game,
            card_embedding=self.card_embedding,
            history_embedding=self.history_embedding,
        )

        # New match bookkeeping
        self.rounds_since_reset = 0
        self._round_boundary_pending = False  # starting a truly new match

        # Choose/refresh the reference player ONLY at the start of a new match
        players = self.game.get("players", []) or []
        if not players:
            raise RuntimeError("Game manager returned no players")
        self._ref_nick = random.choice([p["nickname"] for p in players])

        # Build initial observation/mask for the new match
        cp = self.game["cp_nickname"]
        obs, pub_prior = vectorize_obs(
            self.game,
            cp,
            self.deck_spec,
            card_embedding=self.card_embedding,
            history_embedding=self.history_embedding,
        )
        mask = _legal_action_mask(self.game, self.deck_spec, pub_prior).float()
        pid = self._pid()
        self._last_obs, self._last_mask = obs, mask
        return obs, mask, pid

    def step(self, action: int):
        """
        Apply action for the current player.
        - If action is illegal (shouldn't happen if mask is used), return same state + small penalty.
        - If the chosen action is the check action: the manager resolves the round internally; we compute reward by
          comparing players' n_cards before vs after the move.
        """
        actor_nick = self.game.get("cp_nickname")
        before_counts = {p["nickname"]: int(p["n_cards"]) for p in self.game.get("players", [])}
        status_before = self.game.get("status", "Running")
        history_before = [
            {"player": ev.get("player"), "action_id": int(ev.get("action_id", -1))}
            for ev in (self.game.get("history") or [])
        ]

        # Try to act; catch and handle illegal attempts gracefully.
        save_dir = None
        should_save = False
        try:
            action_int = int(action)
        except Exception:
            obs = self._last_obs.clone()
            mask = self._last_mask.clone()
            pid = self._pid()
            info = {"next_pid": pid, "illegal": 1}
            return obs, mask, float(self.illegal_penalty), False, info

        is_check = action_int == self.deck_spec.check_action_id
        if self.game_save_dir and is_check:
            if self._save_counter >= self.save_sample_rate - 1:
                save_dir = self.game_save_dir
                should_save = True

        try:
            gm.play(self.game, action_int, save_dir=save_dir, verbose=self.verbose)
            if should_save:
                self._save_counter = 0
                self._saved_games += 1
            elif self.game_save_dir and is_check:
                self._save_counter += 1
        except Exception:
            # Return same obs/mask/pid with a penalty; do NOT advance player.
            obs = self._last_obs.clone()
            mask = self._last_mask.clone()
            pid = self._pid()
            info = {"next_pid": pid, "illegal": 1}
            return obs, mask, float(self.illegal_penalty), False, info

        # Refresh deck spec and compute new obs/mask/pid (state may already be at the next round)
        self.deck_spec = _deck_spec_from_game(
            self.game,
            card_embedding=self.card_embedding,
            history_embedding=self.history_embedding,
        )
        cp = self.game.get("cp_nickname")
        obs, pub_prior = vectorize_obs(
            self.game,
            cp,
            self.deck_spec,
            card_embedding=self.card_embedding,
            history_embedding=self.history_embedding,
        )
        mask = _legal_action_mask(self.game, self.deck_spec, pub_prior).float()
        pid = self._pid()

        # Determine terminal and reward
        done = False
        reward = 0.0
        round_result = None
        history_for_log = None
        done_reason = None

        if is_check:
            # Round has been resolved by the manager, and a new round likely started.
            # Identify loser by delta in n_cards (one player +1, possibly -> 0 on elimination).
            self.rounds_since_reset += 1
            after_counts = {p["nickname"]: int(p["n_cards"]) for p in self.game.get("players", [])}
            loser_candidates = []
            for nick, before in before_counts.items():
                after = after_counts.get(nick, before)
                if after == 0 and before > 0:
                    # Could be elimination at max; treat as increased then zeroed
                    loser_candidates.append(nick)
                elif after > before:
                    loser_candidates.append(nick)

            loser = loser_candidates[0] if loser_candidates else None
            ref = self._ref_nick
            if ref is None:
                reward = 0.0
            elif loser is None:
                reward = 0.0
            elif loser == actor_nick:
                reward = -1.0
            else:
                reward = 1.0

            round_result = {
                "loser": loser,
                "ref": ref,
                "before_counts": before_counts,
                "after_counts": after_counts,
            }
            history_for_log = history_before

            # --- Key change: make the round boundary terminal ---
            done = True
            done_reason = "round_terminal"
            self._round_boundary_pending = True  # tell reset() to continue same game
            # ----------------------------------------------------

        # Game finished? Mark terminal regardless of action (overrides reason).
        if self.game.get("status") == "Finished" and status_before != "Finished":
            done = True
            done_reason = "game_finished"
            self._round_boundary_pending = False  # force new game on reset

        else:
            # If the reference player has been eliminated from the table, we treat the episode as done
            # and also force a new game on reset (no ref left to act in current game).
            players_now = {p["nickname"] for p in self.game.get("players", []) if p["n_cards"] > 0}
            if self._ref_nick is not None and self._ref_nick not in players_now:
                done = True
                done_reason = "ref_eliminated"
                self._round_boundary_pending = False  # start a new game on reset

        # safety cap to avoid non-terminating matches
        MAX_ROUNDS = 50
        assert done or self.rounds_since_reset < MAX_ROUNDS

        self._last_obs, self._last_mask = obs, mask
        info = {
            "next_pid": pid,
            "illegal": 0,
            "action": action_int,
            "actor": actor_nick,
            "history": history_for_log,
            "round_result": round_result,
            "reward": float(reward),
            # Helpful breadcrumbs for debugging:
            "round_terminal": bool(is_check),
            "game_status": self.game.get("status", ""),
            "done_reason": done_reason,
        }
        return obs, mask, float(reward), bool(done), info

def main():
    parser = argparse.ArgumentParser(description="Run NFSP Blef self-play locally.")
    parser.add_argument(
        "--resume",
        dest="resume_path",
        type=str,
        default=None,
        help="Optional path to an NFSP checkpoint (.pt) to resume from.",
    )
    parser.add_argument(
        "--initialise-with",
        dest="init_path",
        type=str,
        default=None,
        help="Load model weights from checkpoint but restart schedules (does not carry over step counters).",
    )
    parser.add_argument(
        "--total-steps",
        dest="total_steps",
        type=int,
        default=25_000_000,
        help="Number of environment steps to train for (default: 25,000,000).",
    )
    parser.add_argument(
        "--n-agents",
        dest="n_agents",
        type=int,
        default=2,
        help="Number of seated agents in self-play (default: 2).",
    )
    parser.add_argument(
        "--max-cards",
        dest="max_cards",
        type=int,
        default=11,
        help="Maximum cards per player for Blef variant (default: 3).",
    )
    parser.add_argument(
        "--deck-size",
        dest="deck_size",
        type=int,
        default=24,
        help="Deck size to use (24 or 32).",
    )
    parser.add_argument(
        "--jokers",
        dest="jokers",
        type=int,
        default=0,
        help="Number of jokers to include in the deck (default: 0).",
    )
    parser.add_argument(
        "--blanks",
        dest="blanks",
        type=int,
        default=0,
        help="Number of blanks to include in the deck (default: 0).",
    )
    parser.add_argument(
        "--common-cards",
        dest="common_cards",
        type=int,
        default=0,
        help="Number of community cards dealt each round (default: 0).",
    )
    parser.add_argument(
        "--pick-n-agents-in-range",
        dest="pick_n_agents_in_range",
        action="store_true",
        help="Sample the number of seated agents uniformly from [2, --n-agents] for each new game.",
    )
    parser.add_argument(
        "--pick_n_agents_in_range",
        dest="pick_n_agents_in_range",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pick-jokers-in-range",
        dest="pick_jokers_in_range",
        action="store_true",
        help="Sample the number of jokers uniformly from [0, --jokers] for each new game.",
    )
    parser.add_argument(
        "--pick-blanks-in-range",
        dest="pick_blanks_in_range",
        action="store_true",
        help="Sample the number of blanks uniformly from [0, --blanks] for each new game.",
    )
    parser.add_argument(
        "--pick-common-cards-in-range",
        dest="pick_common_cards_in_range",
        action="store_true",
        help="Sample the number of community cards uniformly from [0, --common-cards] for each new game.",
    )
    parser.add_argument(
        "--eval-only",
        dest="eval_only",
        action="store_true",
        help="Run evaluation only (no training, buffers, or self-play logging).",
    )
    parser.add_argument(
        "--eval-episodes",
        dest="eval_episodes",
        type=int,
        default=200,
        help="Number of evaluation episodes to run when eval-only is enabled (default: 200).",
    )
    parser.add_argument(
        "--export-inference",
        dest="export_inference",
        type=str,
        default=None,
        help="Optional path to save an inference-only checkpoint (no training buffers).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging from the game manager.",
    )
    parser.add_argument(
        "--reset-schedules-to",
        dest="reset_schedules_to",
        type=int,
        default=100_000,
        help="Number of environment steps to reset schedules to (default: 100,000).",
    )
    parser.add_argument(
        "--reset-schedules-at",
        dest="reset_schedules_at",
        type=int,
        default=0,
        help="Number of environment steps to reset schedules at (default: 0).",
    )
    parser.add_argument(
        "--control-plane",
        dest="control_plane",
        type=str,
        default=None,
        help="Path to control-plane JSON file for runtime overrides.",
    )
    parser.add_argument(
        "--control-plane-cooldown",
        dest="control_plane_cooldown",
        type=int,
        default=50_000,
        help="Minimum env steps between control-plane applications (default: 50k).",
    )
    parser.add_argument(
        "--history-sample-every",
        dest="history_sample_every",
        type=int,
        default=100_000,
        help="Env steps between stored action-history samples (default: 100k).",
    )
    parser.add_argument(
        "--history-sample-limit",
        dest="history_sample_limit",
        type=int,
        default=1_000,
        help="Maximum number of history samples to record (default: 1000, <=0 disables limit).",
    )
    parser.add_argument(
        "--history-sample-path",
        dest="history_sample_path",
        type=str,
        default=None,
        help="Optional override for the action-history sample JSONL output.",
    )
    parser.add_argument(
        "--save-game-every",
        dest="save_game_every",
        type=int,
        default=5000,
        help="Persist one full game out of this many (default: 5000).",
    )
    parser.add_argument(
        "--use-card-embeddings",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Enable pretrained card embedding encoder; optionally provide the .pt artifact path. "
            "If omitted, legacy multi-hot features are used."
        ),
    )
    parser.add_argument(
        "--card-embedding-device",
        dest="card_embedding_device",
        type=str,
        default="cpu",
        help="Torch device for card embedding encoder (default: cpu).",
    )
    parser.add_argument(
        "--use-history-embeddings",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Enable pretrained history embedding encoder; optionally provide the .pt artifact path. "
            "If omitted, legacy history multi-hot features are used."
        ),
    )
    parser.add_argument(
        "--history-embedding-device",
        dest="history_embedding_device",
        type=str,
        default="cpu",
        help="Torch device for history embedding encoder (default: cpu).",
    )
    args = parser.parse_args()

    postfix = datetime.now().strftime("%Y%m%d%H%M%S")
    model_save_path = f"nfsp_blef_{postfix}.pt"
    game_save_dir = f"games_{postfix}"
    os.makedirs(game_save_dir, exist_ok=True)
    eval_game_save_dir = f"{game_save_dir}_eval"
    os.makedirs(eval_game_save_dir, exist_ok=True)
    history_sample_limit = None if args.history_sample_limit <= 0 else args.history_sample_limit
    history_sample_path = args.history_sample_path or os.path.join(
        "logs", f"action_samples_{postfix}.jsonl"
    )
    if history_sample_path:
        print(
            f"[history] samples -> {history_sample_path} (every {args.history_sample_every} steps)"
        )

    EVAL_SAVED_GAMES = 2
    print(
        f"[games] eval samples -> {eval_game_save_dir} ({EVAL_SAVED_GAMES} per eval run)"
    )

    card_embedding_bundle: Optional[CardEmbeddingRuntime] = None
    embedding_flag = args.use_card_embeddings
    if embedding_flag:
        embedding_path = (
            DEFAULT_CARD_EMBEDDING_PATH
            if embedding_flag == "auto" or embedding_flag is True
            else embedding_flag
        )
        try:
            card_embedding_bundle = _load_card_embedding(
                embedding_path,
                device=args.card_embedding_device,
            )
            print(f"[embeddings] card encoder loaded from {card_embedding_bundle.source_path}")
        except FileNotFoundError:
            if embedding_flag == "auto":
                print(
                    f"[embeddings] no artifact at {os.path.abspath(embedding_path)}; "
                    "falling back to legacy multi-hot features."
                )
                card_embedding_bundle = None
            else:
                raise
        except Exception as exc:
            raise RuntimeError(f"Failed to load card embeddings: {exc}") from exc
    else:
        print("[embeddings] using legacy card multi-hot features.")

    history_embedding_bundle: Optional[HistoryEmbeddingRuntime] = None
    history_flag = args.use_history_embeddings
    if history_flag:
        history_path = (
            DEFAULT_HISTORY_EMBEDDING_PATH
            if history_flag == "auto" or history_flag is True
            else history_flag
        )
        try:
            history_embedding_bundle = _load_history_embedding(
                history_path,
                device=args.history_embedding_device,
            )
            print(f"[embeddings] history encoder loaded from {history_embedding_bundle.source_path}")
        except FileNotFoundError:
            if history_flag == "auto":
                print(
                    f"[embeddings] no artifact at {os.path.abspath(history_path)}; "
                    "falling back to legacy history multi-hot features."
                )
                history_embedding_bundle = None
            else:
                raise
        except Exception as exc:
            raise RuntimeError(f"Failed to load history embeddings: {exc}") from exc
    else:
        print("[embeddings] using legacy history multi-hot features.")

    env = MyEnv(
        n_agents=args.n_agents,
        max_cards=args.max_cards,
        deck_size=args.deck_size,
        jokers=args.jokers,
        blanks=args.blanks,
        common_cards=args.common_cards,
        verbose=args.verbose,
        game_save_dir=game_save_dir,
        save_sample_rate=args.save_game_every,
        card_embedding=card_embedding_bundle,
        history_embedding=history_embedding_bundle,
        reset_schedules_to=args.reset_schedules_to,
        reset_schedules_at=args.reset_schedules_at,
        pick_n_agents_in_range=args.pick_n_agents_in_range,
        pick_jokers_in_range=args.pick_jokers_in_range,
        pick_blanks_in_range=args.pick_blanks_in_range,
        pick_common_cards_in_range=args.pick_common_cards_in_range,
    )
    obs0, mask0, _ = env.reset()

    agent = NFSPAgent(
        obs_dim=obs0.numel(),
        act_dim=mask0.numel(),
        cfg=NFSPConfig(
            anticipatory_eta=0.25,
            batch_rl=64,
            train_rl_every=32,
            batch_sl=256,
            train_sl_every=8,
            lr_q=1e-4,
            lr_pi=3e-4,
            target_tau=0.01,
            hard_target_interval=0,
            warmup_steps=5_000,
            max_grad_norm=10.0,
            gamma=0.666,
            use_double_dqn=True,
            rl_capacity=200_000,
            sl_capacity=200_000,
            n_step=6,
            burst_rl_updates_on_reward=4,
            burst_reward_threshold=0.5,
            hidden=128
        ),
    )

    if args.resume_path and args.init_path:
        raise ValueError("--resume and --initialise-with are mutually exclusive")

    if args.init_path:
        print(f"[load] initialising weights from {args.init_path} (resetting schedules)")
        agent.load(args.init_path, reset_schedules=True)
    elif args.resume_path:
        print(f"[load] resuming from {args.resume_path}")
        agent.load(args.resume_path)

    eval_env = MyEnv(
        n_agents=env.n_agents,
        max_cards=env.max_cards,
        deck_size=env.rules["deck_size"],
        jokers=env.rules["jokers"],
        blanks=env.rules["blanks"],
        common_cards=env.rules.get("common_cards", 0),
        verbose=env.verbose,
        illegal_penalty=env.illegal_penalty,
        game_save_dir=eval_game_save_dir,
        save_sample_rate=1,
        card_embedding=card_embedding_bundle,
        history_embedding=history_embedding_bundle,
        pick_n_agents_in_range=args.pick_n_agents_in_range,
        pick_jokers_in_range=args.pick_jokers_in_range,
        pick_blanks_in_range=args.pick_blanks_in_range,
        pick_common_cards_in_range=args.pick_common_cards_in_range,
    )

    if args.export_inference:
        export_path = os.path.abspath(args.export_inference)
        os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
        agent.export_inference(export_path)
        print(f"[export] inference-only checkpoint saved to {export_path}")
        return

    if args.eval_only:
        eval_stats = _evaluate_policy(
            agent,
            eval_env,
            max_cards=env.max_cards,
            n_agents=env.n_agents,
            episodes=args.eval_episodes,
            save_dir=eval_game_save_dir,
            save_games=EVAL_SAVED_GAMES,
            pick_n_agents_in_range=args.pick_n_agents_in_range,
            pick_jokers_in_range=args.pick_jokers_in_range,
            pick_blanks_in_range=args.pick_blanks_in_range,
            pick_common_cards_in_range=args.pick_common_cards_in_range,
        )
        print(
            "EVALUATION (eval-only mode):\n"
            f"[steps=0] len={eval_stats['avg_len']:.3f} "
            f"avgR={eval_stats['avg_reward']:.4f} win={eval_stats['win_rate']:.3f}"
        )
        return

    control_plane = None
    if args.control_plane:
        cp_path = os.path.abspath(args.control_plane)
        if not os.path.exists(cp_path):
            snapshot = build_control_snapshot(agent, env, cooldown_steps=args.control_plane_cooldown)
            write_control_file(cp_path, snapshot)
            print(f"[control] bootstrap control-plane file written to {cp_path}")
        control_plane = JsonControlPlane(
            cp_path,
            cooldown_steps=args.control_plane_cooldown,
            verbose=True,
        )
    agent.train_from_selfplay(
        env,
        total_steps=args.total_steps,
        log_every=50_000,
        save_path=model_save_path,
        game_save_dir=game_save_dir,
        eval_env_factory=lambda: eval_env,
        eval_save_dir=eval_game_save_dir,
        eval_save_games=EVAL_SAVED_GAMES,
        control_plane=control_plane,
        history_sample_path=history_sample_path,
        history_sample_every=args.history_sample_every,
        history_sample_limit=history_sample_limit,
    )


if __name__ == "__main__":
    main()
