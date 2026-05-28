"""Head-to-head win-rate of each personality vs the standard NFSP baseline.

Plays full Blef games using the SAME orchestration as NFSP self-play training
(`shared.api.simpleschema_local_manager`: create_game + play), driving each seat
with `production_agent.determine_action`. The opponent is the unmodified policy
(no personality => the standard variant-routed NFSP model). Seats are alternated
to remove first-move bias.

A "playable" personality should be WEAKER than the baseline (it deviates from
near-equilibrium, so it is exploitable) but must stay competitive — not lose
almost every game. Use this to bound tempo/chaos/risk so bots remain fun, not
free wins.

Usage:
  venv_nfsp/bin/python tools/personality_winrate.py --games 200
  venv_nfsp/bin/python tools/personality_winrate.py --games 400 --deck 24 --only perun,czernobog
"""

from __future__ import annotations

import argparse
import random
import warnings

warnings.filterwarnings("ignore")

import torch

from nfsp_ai import production_agent as pa
from nfsp_ai import personalities as P
import shared.api.simpleschema_local_manager as gm

AGENT = pa.load_agent()


def play_game(pers0, pers1, *, deck=24, max_cards=11, max_moves=4000):
    """One full 1v1 game. Returns winning seat ('0'/'1') or None on timeout."""
    game = gm.create_game(2, deck_size=deck, max_cards=max_cards,
                          jokers=0, blanks=0, common_cards=0)
    pmap = {"0": pers0, "1": pers1}
    for pl in game["players"]:
        pl["personality"] = pmap[pl["nickname"]]
    moves = 0
    while game.get("status") != "Finished" and moves < max_moves:
        if game.get("cp_nickname") is None:
            break
        action = AGENT.determine_action(game)
        gm.play(game, int(action))
        # gm.play strips/re-deals players; re-stamp personality each round.
        for pl in game["players"]:
            pl["personality"] = pmap.get(pl["nickname"])
        moves += 1
    active = [p for p in game["players"] if p["n_cards"] > 0]
    return active[0]["nickname"] if len(active) == 1 else None


def winrate(name, games, *, deck=24):
    wins = decided = 0
    for i in range(games):
        if i % 2 == 0:
            w = play_game(name, None, deck=deck)
            me = "0"
        else:
            w = play_game(None, name, deck=deck)
            me = "1"
        if w is None:
            continue
        decided += 1
        wins += int(w == me)
    return (wins / decided if decided else float("nan")), decided


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--deck", type=int, default=24)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    names = P.PERSONALITY_NAMES
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]

    # Calibration: baseline vs baseline should sit near 50%.
    bb, bd = winrate(None, args.games, deck=args.deck)
    print(f"\n=== win-rate vs standard NFSP baseline (deck {args.deck}, "
          f"{args.games} games/seat-balanced) ===")
    print(f"{'(baseline vs baseline)':16s}  win%={bb*100:5.1f}  decided={bd}/{args.games}")
    print(f"{'god':16s}  {'win%':>6s}  {'decided':>8s}   note")
    rows = []
    for n in names:
        wr, dec = winrate(n, args.games, deck=args.deck)
        rows.append((n, wr, dec))
        flag = ""
        if wr < 0.12:
            flag = "  <-- TOO WEAK (loses almost always)"
        elif wr > 0.55:
            flag = "  <-- stronger than baseline?"
        print(f"{n:16s}  {wr*100:5.1f}  {dec:6d}/{args.games}{flag}")
    print("\nTarget: roughly 20-45% (weaker but competitive). "
          "Wall gods (perun) may approach ~50%.")


if __name__ == "__main__":
    main()
