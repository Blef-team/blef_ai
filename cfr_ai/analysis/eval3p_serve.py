"""3-player (1v1v1) CFR serving for evaluation. Mirrors cfr_ai/agent.py's
lookup + macro-fold, but selects the strategy file by the DIRECTED cyclic
play order seen from the acting seat. The deployed 2-player agent.py uses
`tuple(sorted(n_cards))`, which discards direction — wrong for 3-player.
Self-contained so it doesn't perturb the deployed agent.

    serve(game_state, outputs_base, cache) -> action_id

`cache` is a plain dict {canonical_setup_tuple: FlatStrategyAgent} the caller
owns (so whole-game sims load each directed file once).
"""
import os
import random

import numpy as np

from cfr_ai.information_set import make_key, get_possible_actions, get_hand_abstraction
from cfr_ai.strategy_io import load_strategy_for_agent
from cfr_ai.keys import LAST_BET_SHIFT, H_M1_SHIFT, H_M2_SHIFT, ABS_ID_SHIFT, _split_suffix

# Diagnostic toggle: set False to serve the concrete distribution only (ignore
# macro masses), to isolate whether the macro-fold helps or hurts.
USE_MACROS = True

# Diagnostic counters: lookup hit vs fallback (no policy for the infoset).
STATS = {"hit": 0, "miss_abs": 0, "miss_row": 0}


def directed_setup(game_state):
    """Canonical (min-rotation) tuple of hand sizes in PLAY ORDER from the
    acting player. Same for every seat in a game (rotations of one cycle share
    a min); equals the trained directory name a_b_c."""
    cp = int(game_state["cp_nickname"])
    counts = {int(p["nickname"]): int(p["n_cards"]) for p in game_state["players"]}
    n = len(counts)
    order = tuple(counts[(cp + i) % n] for i in range(n))
    return min(order[i:] + order[:i] for i in range(n))


def _compose_key(hand_size, last_bet, h_m1_id, h_m2_id, abs_id):
    return (hand_size | (last_bet << LAST_BET_SHIFT) | (h_m1_id << H_M1_SHIFT)
            | (h_m2_id << H_M2_SHIFT) | (abs_id << ABS_ID_SHIFT))


def serve(game_state, outputs_base, cache):
    nick = game_state["cp_nickname"]
    setup = directed_setup(game_state)                 # directed, NOT sorted
    total_cards = sum(setup)
    history = ([a["action_id"] for a in game_state["history"]]
               if game_state.get("history") else [])
    my = [h for h in game_state.get("hands", []) if h.get("nickname") == nick][0]
    my_cards = [c["value"] * 4 + c["colour"] for c in my["hand"]]

    fs = cache.get(setup)
    if fs is None:
        fs = load_strategy_for_agent(os.path.join(outputs_base, "_".join(map(str, setup))))
        cache[setup] = fs
    min_bet = fs.min_bet
    hand_abs = get_hand_abstraction(my_cards, list(setup))
    sk = make_key(my_cards, hand_abs, history, min_bet).split('-')
    hand_size, last_bet, suffix = int(sk[0]), int(sk[1]), '-'.join(sk[2:])
    rel = get_possible_actions(history, min_bet)
    h_m1, h_m2, abs_str = _split_suffix(suffix)
    abs_id = fs.abs_str_to_id.get(abs_str)
    if abs_id is None:
        STATS["miss_abs"] += 1
        return 87 if len(history) == 0 else 88
    row = fs.lookup(_compose_key(hand_size, last_bet, h_m1, h_m2, abs_id))
    if row is None:
        STATS["miss_row"] += 1
        return 87 if len(history) == 0 else 88
    STATS["hit"] += 1

    probs = fs.get_strategy(row).astype(float)
    nlen = min(len(probs), len(rel))
    w = np.zeros(len(rel))
    w[:nlen] = probs[:nlen]
    if fs.kinds and USE_MACROS:
        from cfr_ai.abstraction.probs import g_vector, p_vector_factored
        from cfr_ai.encoding import clear_lows
        masses = fs.masses[row]
        pv = g = None
        for k, kind in enumerate(fs.kinds):
            if float(masses[k]) <= 0.0:
                continue
            if pv is None:
                pv = p_vector_factored(list(my_cards), total_cards - len(my_cards))
                g = g_vector(total_cards)
            score = pv if kind == "value" else (pv - g if kind == "difftruthy" else g - pv)
            best, bs, ties = -1.0e18, -1, 0
            for a in rel:
                if a <= 87:
                    s = score[a]
                    if s > best + 1e-9:
                        best, bs, ties = s, a, 1
                    elif s > best - 1e-9:
                        ties += 1
                        if random.random() * ties < 1.0:
                            bs = a
            if bs >= 0:
                w[rel.index(bs)] += float(masses[k])
        tot = float(w.sum())
        if tot <= 0.0:
            return 88
        wl = [float(x) for x in clear_lows(w / tot)]
        if sum(wl) <= 0.0:
            return 88
        return random.choices(rel, weights=wl, k=1)[0]

    tot = float(w.sum())
    if tot <= 0.0:
        return 88
    return random.choices(rel, weights=(w / tot).tolist(), k=1)[0]
