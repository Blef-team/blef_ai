"""Deployed-agent entry point: sample an action from the trained CFR
strategy for the active game state.

Loader: `strategy_io.load_strategy_for_agent` — uses `np.searchsorted` on
a sorted-keys array instead of building a `numba.typed.Dict`. ~70× faster
cold load on the biggest setups (50ms vs 3-5s) and avoids importing numba
entirely, which drops ~130 MB from the deployed image.

Cache policy: size = 1. Game state progresses linearly through
(hand_size_a, hand_size_b) configurations as cards are won/lost; the
previous setup is very unlikely to come back before the next one
displaces it. Evict-before-load keeps peak memory at exactly one loaded
strategy (~260-340 MB) + base runtime, so the Lambda fits comfortably in
512 MB.

Public entry point: `determine_action(game_state)`.
"""

import os
import random
from typing import Tuple, Optional

import numpy as np

from cfr_ai.information_set import (
    make_key, get_possible_actions, get_hand_abstraction,
)
from cfr_ai.strategy_io import load_strategy_for_agent
from cfr_ai.keys import (
    LAST_BET_SHIFT, H_M1_SHIFT, H_M2_SHIFT, ABS_ID_SHIFT,
    _split_suffix,
)


# Module-level strategy cache, shared across calls on a warm Lambda
# container. Capacity is hard-coded at 1 (override only if you know what
# you're doing — bigger cache means bigger Lambda memory tier).
_STRATEGY_CACHE_SIZE = int(os.environ.get("CFR_STRATEGY_CACHE_SIZE", "1"))
_current_setup: Optional[Tuple[int, ...]] = None
_current_strategy = None


def _resolve_setup_dir(hand_sizes: Tuple[int, ...]) -> str:
    """Find the strategy directory for `hand_sizes`. Looks first in the
    Lambda working dir (deployed layout) and then in cfr_ai/outputs/
    (dev/test layout). Accepts EITHER the deployed sparse-mmap layout
    (`strategy_meta.npz` + `probs_sparse_*.npy`) OR the legacy
    compressed single-file layout (`strategy.npz`)."""
    setup = "_".join(str(x) for x in hand_sizes)
    candidates = [
        setup,
        os.path.join("cfr_ai", "outputs", setup),
    ]
    for c in candidates:
        if (os.path.exists(os.path.join(c, "strategy_meta.npz"))
                or os.path.exists(os.path.join(c, "strategy.npz"))):
            return c
    raise FileNotFoundError(
        f"No strategy_meta.npz or strategy.npz found for setup {hand_sizes!r}. "
        f"Tried: {candidates}"
    )


def _get_strategy(hand_sizes: Tuple[int, ...]):
    """Load (or fetch from cache) the strategy for `hand_sizes`.

    Cache holds at most ONE strategy. On a miss: drop the current strategy
    BEFORE loading the new one, so peak RAM is one strategy + base runtime
    (not two strategies in flight as you transition setups).
    """
    global _current_setup, _current_strategy
    if hand_sizes == _current_setup and _current_strategy is not None:
        return _current_strategy
    # Drop the old one first so the loader peak ~= 1 strategy in flight.
    _current_strategy = None
    _current_setup = None
    setup_dir = _resolve_setup_dir(hand_sizes)
    fs = load_strategy_for_agent(setup_dir)
    _current_strategy = fs
    _current_setup = hand_sizes
    return fs


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
    row = fs.lookup(comp_key)
    if row is None:
        if len(history) == 0:
            print(
                "No policy found though the round has just begun. "
                "Betting great straight flush spades (hopefully that was intended)"
            )
            return 87
        return 88
    probs = fs.get_strategy(row)
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
