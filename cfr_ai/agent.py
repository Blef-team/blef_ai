"""Deployed-agent entry point: sample an action from the trained CFR
strategy for the active game state.

The agent loads the entire per-setup `strategy.npz` once per setup
(cached on the warm Lambda container) and serves lookups via the
composite int64 key. The legacy per-(hand_size, last_bet) CSV layout
has been retired in favour of NPZ; see `cfr_ai/strategy_io.py`.

Public entry point: `determine_action(game_state)`.
"""

import os
import random
from typing import Dict, Tuple, Optional

import numpy as np

from cfr_ai.information_set import (
    make_key, get_possible_actions, get_hand_abstraction,
)
from cfr_ai.strategy_io import load_strategy
from cfr_ai.lbr import _split_suffix
from cfr_ai.trainer import (
    LAST_BET_SHIFT, H_M1_SHIFT, H_M2_SHIFT, ABS_ID_SHIFT,
)


# Module-level caches: warm Lambda containers reuse these across calls.
_strategy_cache: Dict[Tuple[int, ...], object] = {}


def _resolve_setup_dir(hand_sizes: Tuple[int, ...]) -> str:
    """Find the strategy directory for `hand_sizes`. Looks first in the
    Lambda working dir (deployed layout) and then in cfr_ai/outputs/
    (dev/test layout)."""
    setup = "_".join(str(x) for x in hand_sizes)
    candidates = [
        setup,
        os.path.join("cfr_ai", "outputs", setup),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "strategy.npz")):
            return c
    raise FileNotFoundError(
        f"No strategy.npz found for setup {hand_sizes!r}. Tried: {candidates}"
    )


def _get_strategy(hand_sizes: Tuple[int, ...]):
    """Load (or fetch from cache) the FlatStrategy for `hand_sizes`."""
    if hand_sizes not in _strategy_cache:
        setup_dir = _resolve_setup_dir(hand_sizes)
        _strategy_cache[hand_sizes] = load_strategy(setup_dir)
    return _strategy_cache[hand_sizes]


def _compose_key(hand_size: int, last_bet: int,
                 h_m1_id: int, h_m2_id: int, abs_id: int) -> int:
    return (hand_size
            | (last_bet << LAST_BET_SHIFT)
            | (h_m1_id << H_M1_SHIFT)
            | (h_m2_id << H_M2_SHIFT)
            | (abs_id << ABS_ID_SHIFT))


def determine_action(game_state):
    agent_nickname = game_state["cp_nickname"]
    players = game_state.get("players", [])
    hand_sizes = tuple(sorted(player.get("n_cards") for player in players))
    history = []
    if game_state.get("history"):
        history = [action["action_id"] for action in game_state.get("history")]
    matching_hands = [
        hand for hand in game_state.get("hands", [])
        if hand.get("nickname") == agent_nickname
    ]
    my_cards = [card["value"] * 4 + card["colour"]
                for card in matching_hands[0]["hand"]]

    fs = _get_strategy(hand_sizes)
    min_bet = fs.min_bet
    hand_abstraction = get_hand_abstraction(my_cards, list(hand_sizes))
    key = make_key(my_cards, hand_abstraction, history, min_bet)
    split_key = key.split('-')
    hand_size = int(split_key[0])
    last_bet = int(split_key[1])
    suffix = '-'.join(split_key[2:])
    relevant_actions = get_possible_actions(history, min_bet)

    # Reconstruct the composite int64 key the on-disk index uses.
    h_m1_id, h_m2_id, abs_str = _split_suffix(suffix)
    abs_id = fs.abs_str_to_id.get(abs_str)
    if abs_id is None:
        # Unknown abstraction in the trained policy. Fall back.
        if len(history) == 0:
            print(
                "No policy found though the round has just begun. "
                "Betting great straight flush spades (hopefully that was intended)"
            )
            return 87
        return 88
    comp_key = _compose_key(hand_size, last_bet, h_m1_id, h_m2_id, abs_id)
    k64 = np.int64(comp_key)
    if k64 not in fs.key_to_row:
        if len(history) == 0:
            print(
                "No policy found though the round has just begun. "
                "Betting great straight flush spades (hopefully that was intended)"
            )
            return 87
        return 88
    row = int(fs.key_to_row[k64])
    lo = int(fs.lower_action[row])
    hi = int(fs.upper_action[row])
    probs = fs.strategy[row, lo:hi + 1]
    # Probabilities are stored normalised at save time; we still defend
    # against the all-zero edge case (shouldn't happen post-load-time-norm).
    total = float(probs.sum())
    if total <= 0:
        return 88
    weights = (probs / total).astype(float).tolist()
    # `relevant_actions` may be wider than `probs` if min_bet differs
    # from the agent's training-time min_bet — pad / trim defensively.
    if len(weights) != len(relevant_actions):
        out = [0.0] * len(relevant_actions)
        n = min(len(weights), len(relevant_actions))
        out[:n] = weights[:n]
        if sum(out) <= 0:
            out[-1] = 1.0
        s = sum(out)
        weights = [x / s for x in out]
    return random.choices(relevant_actions, weights=weights, k=1)[0]
