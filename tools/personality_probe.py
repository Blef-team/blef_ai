"""Behavioral probe for the Blef personality layer.

Runs every personality (and the unmodified baseline policy) over many realistic
game states and measures the *signature* of the moves they choose, so we can
verify each bot plays the way it was designed to — and tune the knobs against
real numbers rather than vibes.

Metrics per personality (deltas are vs the baseline policy on the SAME states):
  check%   how often it challenges (only over states where check is legal)
  pPriv    mean P(chosen bet is true | my hand)  — LOW => bluffy, HIGH => honest
  div      mean (pPriv - pub) of chosen bet      — HIGH => leaky/readable
  rank     mean position of the bet in the legal range — HIGH => big senior leaps
  rankStd  spread of rank                        — HIGH => chaotic/unpredictable

Usage:
  venv_nfsp/bin/python tools/personality_probe.py            # full report
  venv_nfsp/bin/python tools/personality_probe.py --n 600    # more states
"""

from __future__ import annotations

import argparse
import random
import statistics as st
from collections import defaultdict

import numpy as np
import torch

import warnings
warnings.filterwarnings("ignore")

from nfsp_ai import production_agent as pa
from nfsp_ai import personalities as P
from shared.game_utils import GameRules
from shared.probabilities.dynamic_probabilities import (
    get_bet_probabilities,
    get_generic_bet_probabilities,
)

AGENT = pa.load_agent()


# --------------------------------------------------------------------------- #
# State generation
# --------------------------------------------------------------------------- #
def _deal(deck, sizes, rng):
    vals = 6 if deck == 24 else 8
    cards = [(v, c) for v in range(vals) for c in range(4)]
    rng.shuffle(cards)
    out, i = [], 0
    for k in sizes:
        out.append(cards[i:i + k])
        i += k
    return out


def build_state(rng, *, deck=24, kcards=None, n=2, round_number=None,
                last_bet="auto", team=False):
    """Build a valid game-state dict (+ its check_action_id)."""
    gr = GameRules(deck)
    check = gr.check_action_id
    if kcards is None:
        kcards = [rng.randint(1, 3) for _ in range(n)]
    n = len(kcards)
    hands_cards = _deal(deck, kcards, rng)
    names = [f"p{i}" for i in range(n)]
    teams = [None] * n
    if team and n >= 2:
        teams = [1 if i % 2 == 0 else 2 for i in range(n)]
    players = [{"nickname": names[i], "n_cards": kcards[i], "team": teams[i]} for i in range(n)]
    hands = [{"nickname": names[i],
              "hand": [{"value": v, "colour": c} for (v, c) in hands_cards[i]]}
             for i in range(n)]
    if last_bet == "auto":
        last_bet = rng.randint(0, check - 2) if rng.random() < 0.75 else None
    hist = [] if last_bet is None else [{"action_id": int(last_bet), "player": names[-1]}]
    if round_number is None:
        round_number = rng.randint(1, 6)
    state = {
        "cp_nickname": names[0],
        "rules": {"deck_size": deck, "jokers": 0, "blanks": 0,
                  "common_cards": 0, "max_rounds": 8},
        "players": players,
        "hands": hands,
        "common_hand": [],
        "history": hist,
        "round_number": round_number,
        "game_uuid": "probe",
    }
    return state, check


def state_probs(state):
    last = state["history"][-1]["action_id"] if state["history"] else -1
    pvt = get_bet_probabilities(game_state=state, for_betting=True, last_bet=last)
    pub = get_generic_bet_probabilities(game_state=state, last_bet=last)
    return pvt, pub, last


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def eval_personality(states, name, repeats=1):
    """name=None => baseline. repeats>1 averages stochastic gods."""
    checks = 0
    checkable = 0
    pprivs, divs, ranks = [], [], []
    for (state, check, pvt, pub, last) in states:
        is_checkable = bool(state["history"])
        checkable += int(is_checkable)
        lo = (last + 1) if last >= 0 else 0
        hi = check - 1
        rng_span = max(1, hi - lo)
        for _ in range(repeats):
            probe = dict(state)
            probe["personality"] = name
            a = AGENT.determine_action(probe)
            if a == check:
                checks += 1
                continue
            if a < check:
                ranks.append((a - lo) / rng_span)
                pprivs.append(float(pvt[a]))
                divs.append(float(pvt[a] - pub[a]))
    n_check_obs = max(1, checkable * repeats)
    return {
        "check": checks / n_check_obs,
        "pPriv": st.fmean(pprivs) if pprivs else float("nan"),
        "div": st.fmean(divs) if divs else float("nan"),
        "rank": st.fmean(ranks) if ranks else float("nan"),
        "rankStd": st.pstdev(ranks) if len(ranks) > 1 else 0.0,
        "n_bet": len(pprivs),
    }


def interpret(name, base, m):
    """Auto-generate a one-line read of how this god deviates from baseline."""
    bits = []
    dchk = (m["check"] - base["check"]) * 100
    if abs(dchk) >= 4:
        bits.append(f"{'challenges more' if dchk > 0 else 'rarely checks'} ({dchk:+.0f}pp)")
    dpp = m["pPriv"] - base["pPriv"]
    if abs(dpp) >= 0.02:
        bits.append(f"{'bluffier' if dpp < 0 else 'more honest'} ({dpp:+.3f} pPriv)")
    ddiv = m["div"] - base["div"]
    if abs(ddiv) >= 0.02:
        bits.append(f"{'leakier/readable' if ddiv > 0 else 'more concealed'} ({ddiv:+.3f} div)")
    drank = m["rank"] - base["rank"]
    if abs(drank) >= 0.03:
        bits.append(f"{'bigger leaps' if drank > 0 else 'min increments'} ({drank:+.3f} rank)")
    drs = m["rankStd"] - base["rankStd"]
    if drs >= 0.04:
        bits.append(f"more chaotic ({drs:+.3f} rankStd)")
    return "; ".join(bits) if bits else "~ baseline"


def table(title, states, names, repeats_for):
    base = eval_personality(states, None)
    print(f"\n=== {title}  (states={len(states)}, baseline "
          f"check={base['check']*100:.0f}% pPriv={base['pPriv']:.3f} "
          f"div={base['div']:.3f} rank={base['rank']:.3f} rankStd={base['rankStd']:.3f}) ===")
    print(f"{'god':12s} {'src':6s} {'dChk':>6s} {'dpPriv':>8s} {'ddiv':>8s} "
          f"{'drank':>7s} {'rankStd':>8s}   interpretation")
    rows = {}
    for name in names:
        cfg = P.PERSONALITIES[name]
        if P.is_delegated(name):
            m = eval_personality(states, name)
            rows[name] = m
            print(f"{name:12s} {cfg.source[:6]:6s} {'(delegated heuristic engine)':>40s}")
            continue
        m = eval_personality(states, name, repeats=repeats_for.get(name, 1))
        rows[name] = m
        print(f"{name:12s} {cfg.source[:6]:6s} "
              f"{(m['check']-base['check'])*100:+6.0f} {m['pPriv']-base['pPriv']:+8.3f} "
              f"{m['div']-base['div']:+8.3f} {m['rank']-base['rank']:+7.3f} "
              f"{m['rankStd']:8.3f}   {interpret(name, base, m)}")
    return base, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    # Stochastic gods (chaos>0) get repeats so their averages are stable.
    repeats_for = {n: 4 for n, c in P.PERSONALITIES.items()
                   if c.chaos > 0.05 or c.mood in ("random_chaos", "spike_random", "win_ramp")}

    def make(nstates, **kw):
        out = []
        for _ in range(nstates):
            s, chk = build_state(rng, **kw)
            pvt, pub, last = state_probs(s)
            out.append((s, chk, pvt, pub, last))
        return out

    names = P.PERSONALITY_NAMES

    # Main 1v1 / 24 mixed-depth table
    main_states = make(args.n, deck=24, n=2)
    table("1v1 deck-24 (mixed depth)", main_states, names, repeats_for)

    # Multi-player / 24
    multi_states = make(args.n // 2, deck=24, n=4)
    table("4-player deck-24 (mixed depth)", multi_states, names, repeats_for)

    # Deck 32
    d32 = make(args.n // 2, deck=32, n=2)
    table("1v1 deck-32 (mixed depth)", d32, names, repeats_for)

    # --- Mood cohorts: early vs late (depth) for phase gods --------------- #
    phase_gods = ["triglav", "veles", "poludnica", "mokosh", "rusalka", "porevit"]
    early = make(args.n // 2, deck=24, n=2, kcards=[1, 1], round_number=1)
    late = make(args.n // 2, deck=24, n=2, kcards=[5, 5], round_number=6)
    print("\n\n########## MOOD: EARLY (round1, 1 card) vs LATE (round6, 5 cards) ##########")
    be, re_ = table("PHASE early", early, phase_gods, repeats_for)
    bl, rl = table("PHASE late", late, phase_gods, repeats_for)
    print("\n-- phase deltas (late - early) for phase gods --")
    for g in phase_gods:
        print(f"  {g:10s} d_pPriv={rl[g]['pPriv']-re_[g]['pPriv']:+.3f}  "
              f"d_rank={rl[g]['rank']-re_[g]['rank']:+.3f}  "
              f"d_chk={(rl[g]['check']-re_[g]['check'])*100:+.0f}pp  "
              f"d_rankStd={rl[g]['rankStd']-re_[g]['rankStd']:+.3f}")

    # --- Mood cohorts: ahead vs behind (standing) for standing gods ------- #
    stand_gods = ["perun", "kupala", "dazhbog"]
    ahead = make(args.n // 2, deck=24, kcards=[1, 4], round_number=4)   # cp has fewer => winning
    behind = make(args.n // 2, deck=24, kcards=[4, 1], round_number=4)  # cp has more => losing
    print("\n\n########## MOOD: AHEAD (cp 1 vs 4) vs BEHIND (cp 4 vs 1) ##########")
    ba, ra = table("STANDING ahead", ahead, stand_gods, repeats_for)
    bb, rb = table("STANDING behind", behind, stand_gods, repeats_for)
    print("\n-- standing deltas (ahead - behind) --")
    for g in stand_gods:
        print(f"  {g:10s} d_pPriv={ra[g]['pPriv']-rb[g]['pPriv']:+.3f}  "
              f"d_rank={ra[g]['rank']-rb[g]['rank']:+.3f}  "
              f"d_chk={(ra[g]['check']-rb[g]['check'])*100:+.0f}pp")

    # --- Zorya parity: even vs odd round ---------------------------------- #
    print("\n\n########## MOOD: ZORYA parity (even vs odd round) ##########")
    even = make(args.n // 2, deck=24, n=2, round_number=2)
    odd = make(args.n // 2, deck=24, n=2, round_number=3)
    be2, re2 = table("ZORYA even-round (dawn)", even, ["zorya"], repeats_for)
    bo2, ro2 = table("ZORYA odd-round (dusk)", odd, ["zorya"], repeats_for)
    print(f"  zorya dawn-dusk d_chk={(re2['zorya']['check']-ro2['zorya']['check'])*100:+.0f}pp "
          f"d_pPriv={re2['zorya']['pPriv']-ro2['zorya']['pPriv']:+.3f}")


if __name__ == "__main__":
    main()
