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

# Persistent per-setup strategy cache (each value is a FlatStrategyAgent, which
# may carry macro masses/kinds). Size = CFR_STRATEGY_CACHE_SIZE (default 1 =
# single-slot, low RAM for the Lambda). Evaluation harnesses that revisit many
# setups (whole-game sims traverse all 66) set it high so each setup is loaded
# once instead of every time play returns to it.
_SERVER_CACHE: dict = {}

# Optional override of the strategy source directory. None => production
# behaviour (Lambda working dir, then cfr_ai/outputs/). Evaluation tools set
# this to a specific model version's outputs dir (e.g. cfr_ai/archive/<tag>/
# outputs or cfr_ai/experiments/<tag>/outputs) so the same agent code can
# serve any trained CFR version. See `set_outputs_base`.
_OUTPUTS_BASE: Optional[str] = None


def set_outputs_base(path: Optional[str]) -> None:
    """Point the agent at a specific model version's outputs dir (the dir
    that contains <setup>/strategy.npz). Pass None to restore the default
    production resolution. Invalidates the strategy cache."""
    global _OUTPUTS_BASE, _current_setup, _current_strategy
    _OUTPUTS_BASE = path
    _current_setup = None
    _current_strategy = None
    _SERVER_CACHE.clear()


def _resolve_setup_dir(hand_sizes: Tuple[int, ...]) -> str:
    """Find the strategy directory for `hand_sizes`. If an outputs base has
    been set (evaluation), look only there; otherwise use the production
    resolution: the Lambda working dir (deployed layout) then cfr_ai/outputs/
    (dev/test). Accepts EITHER the deployed sparse-mmap layout
    (`strategy_meta.npz` + `probs_sparse_*.npy`) OR the legacy compressed
    single-file layout (`strategy.npz`)."""
    setup = "_".join(str(x) for x in hand_sizes)
    if _OUTPUTS_BASE is not None:
        candidates = [os.path.join(_OUTPUTS_BASE, setup)]
    else:
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


def _ensure_loaded(hand_sizes: Tuple[int, ...]) -> None:
    """Point `_current_strategy` at the FlatStrategyAgent for `hand_sizes`.
    Strategies are held in `_SERVER_CACHE` (size CFR_STRATEGY_CACHE_SIZE,
    default 1 = single-slot / low RAM). A cached setup is rebound for free; only
    a miss loads from disk, so whole-game sims that revisit setups don't reload
    every round. A macro strategy (a FlatStrategyAgent carrying `.kinds`/
    `.masses`) and a plain concrete one go through the same `determine_action`."""
    global _current_setup, _current_strategy
    key = tuple(hand_sizes)
    fs = _SERVER_CACHE.get(key)
    if fs is None:
        fs = load_strategy_for_agent(_resolve_setup_dir(hand_sizes))
        if _STRATEGY_CACHE_SIZE <= 1:
            _SERVER_CACHE.clear()
        elif len(_SERVER_CACHE) >= _STRATEGY_CACHE_SIZE:
            _SERVER_CACHE.pop(next(iter(_SERVER_CACHE)))
        _SERVER_CACHE[key] = fs
    _current_strategy = fs
    _current_setup = key


def _compose_key(hand_size: int, last_bet: int,
                 h_m1_id: int, h_m2_id: int, abs_id: int) -> int:
    return (hand_size
            | (last_bet << LAST_BET_SHIFT)
            | (h_m1_id << H_M1_SHIFT)
            | (h_m2_id << H_M2_SHIFT)
            | (abs_id << ABS_ID_SHIFT))


def _no_policy_fallback(history_len: int) -> int:
    """Action to take when the trained policy has nothing for this infoset.
    At the start of a round we punt to action 87 (great straight flush spades);
    mid-round we check (88)."""
    if history_len == 0:
        print(
            "No policy found though the round has just begun. "
            "Betting great straight flush spades (hopefully that was intended)"
        )
        return 87
    return 88


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

    _ensure_loaded(hand_sizes)
    fs = _current_strategy
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
        return _no_policy_fallback(len(history))
    comp_key = _compose_key(hand_size, last_bet, h_m1_id, h_m2_id, abs_id)
    row = fs.lookup(comp_key)
    if row is None:
        return _no_policy_fallback(len(history))

    # Slice/pad the stored strategy into a right-sized array up front:
    # `relevant_actions` may be wider than `probs` if the agent's min_bet
    # differs from the training-time min_bet. Single normalisation handles
    # both the equal-width and mismatched-width cases; if everything ends
    # up zero (shouldn't happen post-load-time-norm), fall back to a check.
    probs = fs.get_strategy(row).astype(float)
    n = min(len(probs), len(relevant_actions))
    weights = np.zeros(len(relevant_actions))
    weights[:n] = probs[:n]

    # Macro strategy (V3+): fold each macro's mass onto its per-hand b* — the
    # argmax (value/difftruthy) or argmin (bluff) of the macro's score over the
    # legal bets, random tie-break — then clear_lows, exactly as during training.
    # The existence kernel here is the pure-Python `p_vector_factored`, which is
    # bit-exact to the numba `p_vector_fast` used in training (probs_jit self-test:
    # max abs diff 0.0) — so the resolved b* matches, and the Lambda image stays
    # numba-free (one p/g build per decision; JIT speed is irrelevant at serve).
    if fs.kinds:
        from cfr_ai.abstraction.probs import g_vector, p_vector_factored
        from cfr_ai.encoding import clear_lows
        masses = fs.masses[row]
        total_cards = sum(hand_sizes)
        pv = g = None
        for k, kind in enumerate(fs.kinds):
            if float(masses[k]) <= 0.0:
                continue
            if pv is None:
                pv = p_vector_factored(list(my_cards), total_cards - len(my_cards))
                g = g_vector(total_cards)
            score = pv if kind == "value" else (pv - g if kind == "difftruthy" else g - pv)
            best, bs, ties = -1.0e18, -1, 0
            for a in relevant_actions:
                if a <= 87:
                    s = score[a]
                    if s > best + 1e-9:
                        best, bs, ties = s, a, 1
                    elif s > best - 1e-9:
                        ties += 1
                        if random.random() * ties < 1.0:
                            bs = a
            if bs >= 0:
                weights[relevant_actions.index(bs)] += float(masses[k])
        tot = float(weights.sum())
        if tot <= 0.0:
            return 88
        w = [float(x) for x in clear_lows(weights / tot)]
        if sum(w) <= 0.0:
            return 88
        return random.choices(relevant_actions, weights=w, k=1)[0]

    total = float(weights.sum())
    if total <= 0:
        return 88
    weights = (weights / total).tolist()
    return random.choices(relevant_actions, weights=weights, k=1)[0]
