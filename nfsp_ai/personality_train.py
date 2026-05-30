"""Training-time personality bias toolkit.

Three mechanisms keyed on a single ``PersonalityTrainSpec``:

  A. Terminal-magnitude shaping — multiply the MC terminal credit by a
     per-trajectory scalar derived from the spec, the win/loss outcome, a
     bluff oracle, and a hand-type bias keyed on the actor's bets.

  B. Action-mask restriction — hard-zero forbidden bet ranges and (optionally)
     the CHECK action, with a mandatory safety rail that always leaves at
     least one legal action.

  C. Observation feature zero-mask — zero named obs blocks at vectorize time
     to produce trained-in perceptual deficits. The block layout mirrors
     ``nfsp_ai.nfsp_run_local._compute_obs_dim`` and is resolved from the
     same ``DeckSpec`` fields, so layout drift is impossible.

The spec lives in a JSON file (one personality per file) and is also stamped
into the exported inference checkpoint's ``cfg["personality"]`` so the
production loader can reapply C and B at serve time symmetrically to training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np

from shared.game_utils import (
    GameRules,
    determine_set_existence,
    get_set_details_from_action_id,
)


# Hand-type strings as emitted by shared.game_utils.get_set_details_from_action_id.
KNOWN_HAND_TYPES = (
    "High card",
    "Pair",
    "Two pairs",
    "Straight",
    "Three of a kind",
    "Full house",
    "Flush",
    "Four of a kind",
    "Straight flush",
)

# Obs block names recognised by the blind-spot mask. Order matches the
# layout produced by nfsp_ai.nfsp_run_local.vectorize_obs / _compute_obs_dim.
KNOWN_OBS_BLOCKS = (
    "private_hand",
    "rules_jokers",
    "private_jokers",
    "common_jokers",
    "common_hand",
    "seat_counts",
    "team_flags",       # only present when DeckSpec.team_aware is True
    "round",
    "history",
    "last_bet_prob",
    "private_priors",
    "public_priors",
    "blanks",
)

MAX_PLAYERS = 8  # mirrors nfsp_ai.nfsp_run_local.MAX_PLAYERS


@dataclass(frozen=True)
class PersonalityTrainSpec:
    name: str
    # --- A: terminal-magnitude shaping ---
    win_multiplier: float = 1.0
    loss_multiplier: float = 1.0
    bluff_caught_penalty: float = 0.0       # added to loss scale when actor was caught bluffing
    bluff_call_bonus: float = 0.0           # added to win scale when actor caught a bluff
    hand_type_bias: dict = field(default_factory=dict)  # {set_type: float}; positive = likes
    # --- B: action-mask restriction ---
    forbid_hand_types: tuple = ()
    forbid_check_unless_only: bool = False
    # --- C: observation feature zero-mask ---
    obs_blind_spots: tuple = ()


def load_spec(path: str) -> PersonalityTrainSpec:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return spec_from_dict(data)


def spec_to_dict(spec: PersonalityTrainSpec) -> dict:
    d = asdict(spec)
    d["forbid_hand_types"] = list(d["forbid_hand_types"])
    d["obs_blind_spots"] = list(d["obs_blind_spots"])
    return d


def spec_from_dict(data: dict) -> PersonalityTrainSpec:
    d = dict(data)
    if "forbid_hand_types" in d:
        d["forbid_hand_types"] = tuple(d["forbid_hand_types"])
    if "obs_blind_spots" in d:
        d["obs_blind_spots"] = tuple(d["obs_blind_spots"])
    for ht in d.get("forbid_hand_types", ()):
        if ht not in KNOWN_HAND_TYPES:
            raise ValueError(
                f"Unknown hand type {ht!r} in forbid_hand_types; known: {KNOWN_HAND_TYPES}"
            )
    for ht in (d.get("hand_type_bias") or {}):
        if ht not in KNOWN_HAND_TYPES:
            raise ValueError(
                f"Unknown hand type {ht!r} in hand_type_bias; known: {KNOWN_HAND_TYPES}"
            )
    for blk in d.get("obs_blind_spots", ()):
        if blk not in KNOWN_OBS_BLOCKS:
            raise ValueError(
                f"Unknown obs block {blk!r} in obs_blind_spots; known: {KNOWN_OBS_BLOCKS}"
            )
    return PersonalityTrainSpec(**d)


# ---------------------------------------------------------------------------
# Layout-block resolver (C)
# ---------------------------------------------------------------------------

def resolve_obs_block_slices(deck_spec) -> dict:
    """Return ``{block_name: (start, end)}`` for every block in the obs vector.

    Mirrors the order in ``nfsp_ai.nfsp_run_local.vectorize_obs`` /
    ``_compute_obs_dim`` exactly. The resolver verifies the computed total
    matches ``deck_spec.obs_dim`` so any drift in the obs layout fails fast.
    """
    card_dim = int(deck_spec.card_feature_dim)
    history_dim = int(deck_spec.history_feature_dim)
    hist_dim = int(deck_spec.hist_dim)
    team_aware = bool(getattr(deck_spec, "team_aware", True))
    slices: dict = {}
    idx = 0

    slices["private_hand"] = (idx, idx + card_dim); idx += card_dim
    slices["rules_jokers"] = (idx, idx + 1); idx += 1
    slices["private_jokers"] = (idx, idx + 1); idx += 1
    slices["common_jokers"] = (idx, idx + 1); idx += 1
    slices["common_hand"] = (idx, idx + card_dim); idx += card_dim
    slices["seat_counts"] = (idx, idx + MAX_PLAYERS); idx += MAX_PLAYERS
    if team_aware:
        slices["team_flags"] = (idx, idx + MAX_PLAYERS); idx += MAX_PLAYERS
    slices["round"] = (idx, idx + 1); idx += 1
    slices["history"] = (idx, idx + history_dim); idx += history_dim
    slices["last_bet_prob"] = (idx, idx + 1); idx += 1
    slices["private_priors"] = (idx, idx + hist_dim); idx += hist_dim
    slices["public_priors"] = (idx, idx + hist_dim); idx += hist_dim
    slices["blanks"] = (idx, idx + 1); idx += 1

    if idx != int(deck_spec.obs_dim):
        raise ValueError(
            f"Layout resolver computed obs_dim={idx} but DeckSpec.obs_dim="
            f"{int(deck_spec.obs_dim)}. Layout out of sync with _compute_obs_dim."
        )
    return slices


def apply_personality_to_obs(
    obs: np.ndarray,
    spec: Optional[PersonalityTrainSpec],
    deck_spec,
) -> np.ndarray:
    """Zero the obs blocks named in ``spec.obs_blind_spots`` (in place)."""
    if spec is None or not spec.obs_blind_spots:
        return obs
    layout = resolve_obs_block_slices(deck_spec)
    for blk in spec.obs_blind_spots:
        rng = layout.get(blk)
        if rng is None:
            # team_flags only exists when DeckSpec.team_aware; ignore silently.
            continue
        start, end = rng
        obs[start:end] = 0.0
    return obs


# ---------------------------------------------------------------------------
# Action-mask restriction (B)
# ---------------------------------------------------------------------------

_HAND_TYPE_ACTION_IDS_CACHE: dict = {}


def _action_ids_for_hand_type(set_type: str, deck_size: int, check_id: int) -> list:
    """Action ids in ``[0, check_id)`` whose decoded set_type matches.

    Cached by (set_type, deck_size, check_id) — the partition is static for a
    given deck.
    """
    key = (set_type, int(deck_size), int(check_id))
    cached = _HAND_TYPE_ACTION_IDS_CACHE.get(key)
    if cached is not None:
        return cached
    ids: list = []
    for aid in range(check_id):
        info = get_set_details_from_action_id(aid, deck_size)
        if info and info.get("set_type") == set_type:
            ids.append(aid)
    _HAND_TYPE_ACTION_IDS_CACHE[key] = ids
    return ids


def apply_personality_to_mask(
    mask: np.ndarray,
    spec: Optional[PersonalityTrainSpec],
    deck_spec,
) -> np.ndarray:
    """Apply hard mask restrictions with a mandatory safety rail.

    Order:
      1. Snapshot the original mask and whether CHECK was originally legal.
      2. Zero all bet ids matching ``spec.forbid_hand_types``.
      3. If ``spec.forbid_check_unless_only`` and any bet is still legal, zero CHECK.
      4. If the mask is now all zero, restore CHECK (if originally legal) or
         fall back to the original mask. Never produce an empty mask.
    """
    if spec is None or (not spec.forbid_hand_types and not spec.forbid_check_unless_only):
        return mask
    check_id = int(deck_spec.check_action_id)
    deck_size = int(deck_spec.deck_size)

    original = mask.copy()
    check_legal = bool(mask[check_id]) if check_id < len(mask) else False

    for ht in spec.forbid_hand_types:
        for aid in _action_ids_for_hand_type(ht, deck_size, check_id):
            if aid < len(mask):
                mask[aid] = 0

    if spec.forbid_check_unless_only and check_id < len(mask):
        bets_legal = bool(mask[:check_id].any()) if check_id > 0 else False
        if bets_legal:
            mask[check_id] = 0

    if not mask.any():
        if check_legal:
            mask[check_id] = 1
        else:
            mask[:] = original
    return mask


# ---------------------------------------------------------------------------
# Terminal-magnitude shaping (A)
# ---------------------------------------------------------------------------

def compute_personality_terminal_scale(
    spec: Optional[PersonalityTrainSpec],
    game_state: dict,
    actor_nick: str,
    loser_nick: Optional[str],
    deck_size: int,
) -> float:
    """Compute the scalar multiplied into the MC terminal credit.

    Returns 1.0 when no shaping applies. Mechanics:

      * ``win_multiplier`` / ``loss_multiplier`` set the BASE scale.
      * ``bluff_caught_penalty`` adds to the loss scale if the actor was the
        bet-maker AND the bet was a bluff (the bet-maker lost the round).
      * ``bluff_call_bonus`` adds to the win scale if the actor caught a bluff
        (the actor was NOT the bet-maker and the bet-maker lost).
      * ``hand_type_bias`` adds ``+bias`` to the scale per actor-bet of that
        set_type when the actor won, and ``-bias`` when the actor lost — so
        positive bias = the personality "likes" the hand type (wins bigger,
        loses smaller) and negative = "dislikes" (wins smaller, loses bigger).
    """
    if spec is None or loser_nick is None:
        return 1.0

    actor_won = loser_nick != actor_nick
    scale = float(spec.win_multiplier if actor_won else spec.loss_multiplier)

    check_id = GameRules(int(deck_size)).check_action_id
    history = game_state.get("history") or []

    # Find the last bet and its maker.
    bet_maker_nick: Optional[str] = None
    last_bet_id = -1
    for ev in reversed(history):
        try:
            aid = int(ev.get("action_id", -1))
        except Exception:
            continue
        if 0 <= aid < check_id:
            last_bet_id = aid
            bet_maker_nick = ev.get("player")
            break

    bet_was_bluff = bet_maker_nick is not None and bet_maker_nick == loser_nick

    if bet_was_bluff:
        actor_was_bet_maker = bet_maker_nick == actor_nick
        if actor_won and not actor_was_bet_maker:
            scale += float(spec.bluff_call_bonus)
        if (not actor_won) and actor_was_bet_maker:
            scale += float(spec.bluff_caught_penalty)

    if spec.hand_type_bias:
        for ev in history:
            if ev.get("player") != actor_nick:
                continue
            try:
                aid = int(ev.get("action_id", -1))
            except Exception:
                continue
            if not (0 <= aid < check_id):
                continue
            info = get_set_details_from_action_id(aid, int(deck_size))
            if not info:
                continue
            bias = float(spec.hand_type_bias.get(info.get("set_type"), 0.0))
            scale += bias if actor_won else -bias

    return float(scale)
