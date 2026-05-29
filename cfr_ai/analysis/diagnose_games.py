"""Phase-1 diagnostic analysis of CFR-vs-NFSP whole-game logs.

Reads the compact per-round table produced by cfr_vs_nfsp_games.py and
reconstructs every decision offline (the deal + history vectors are
sufficient), then reports — all as AGGREGATE RATES vs a reference, never
per-loss blame (a single lost check isn't a mistake under hidden info):

  A. Game-level: CFR game win-rate (overall + by who opens), game length.
  B. Round-loss attribution: of CFR's lost rounds, the as-checker (challenged
     an existing bet -> "checks too much") vs as-bettor (bluff caught ->
     "bets too much") split, vs NFSP, bucketed by hand size.
  C. Check calibration by faced-bet category: CFR's check-rate vs the actual
     not-exists rate (the naive-optimal check signal) and vs NFSP.
  D. Bet/bluff behaviour by category: CFR's bet mix and bluff rate (bet made
     that doesn't exist on the table) vs NFSP.

Decision reconstruction per round: bets are strictly increasing, so the set
bits of `history` in ascending order are the play sequence; the opener makes
bet 1, players alternate, and the player after the last bet checked. `cfr_seat`
maps each decision to Morana or Perun.

    python -m cfr_ai.analysis.diagnose_games --in cfr_ai/analysis/cfr_vs_nfsp_games_current.npy
"""

import argparse
import sys
from collections import defaultdict

import numpy as np

from shared.api.simpleschema_local_manager import determine_set_existence
from shared.game_utils import get_set_details_from_action_id

DECK_SIZE = 24
N_BETS = 88
RULES = {"deck_size": DECK_SIZE, "jokers": 0, "blanks": 0, "common_cards": 0}

# action_id -> broad category (collapse straight/straight-flush variants).
def _category(aid: int) -> str:
    st = get_set_details_from_action_id(aid, DECK_SIZE)["set_type"]
    low = st.lower()
    if "straight" in low and "flush" in low:
        return "Straight flush"
    if "straight" in low:
        return "Straight"
    return st

CATEGORIES = [_category(a) for a in range(N_BETS)]
CAT_ORDER = ["High card", "Pair", "Two pairs", "Straight", "Three of a kind",
             "Full house", "Flush", "Four of a kind", "Straight flush"]


def _cards_from_deal(deal):
    """List of {value,colour} dicts for all dealt cards (owner != 255)."""
    return [{"value": int(c // 4), "colour": int(c % 4)}
            for c in range(24) if deal[c] != 255]


def _exists(cards, aid):
    return bool(determine_set_existence(cards, int(aid), RULES, 0))


def analyze(path):
    a = np.load(path)
    n = len(a)
    print(f"loaded {n} rounds from {path}\n")

    # ---- A. game-level ----
    last = {}
    for r in a:
        g = int(r["game_id"])
        if g not in last or r["round_idx"] > last[g]["round_idx"]:
            last[g] = r
    ngames = len(last)
    cfr_game_wins = sum(1 for r in last.values() if (1 - r["loser_seat"]) == r["cfr_seat"])
    # split by who opened the game (cfr_seat 0 == CFR opened round 1)
    opened = [r for r in last.values() if r["cfr_seat"] == 0]
    notopened = [r for r in last.values() if r["cfr_seat"] == 1]
    wlen = defaultdict(int)
    for g, r in last.items():
        wlen[int(r["round_idx"]) + 1] += 1
    print("=== A. Game level ===")
    print(f"  games: {ngames} | CFR game win-rate: {cfr_game_wins/ngames:.3f} "
          f"(SE {0.5/ngames**0.5:.3f})")
    print(f"    when CFR opens game: {sum(1 for r in opened if (1-r['loser_seat'])==r['cfr_seat'])/max(1,len(opened)):.3f}"
          f" | when Perun opens: {sum(1 for r in notopened if (1-r['loser_seat'])==r['cfr_seat'])/max(1,len(notopened)):.3f}")
    lens = sorted(wlen)
    print(f"  game length (rounds): min {lens[0]}, median {np.median([int(r['round_idx'])+1 for r in last.values()]):.0f}, max {lens[-1]}")

    # ---- single pass over rounds: reconstruct decisions ----
    # B. loss attribution
    loss = {ag: {"as_checker": 0, "as_bettor": 0} for ag in ("CFR", "NFSP")}
    loss_by_size = defaultdict(lambda: {"checker": 0, "bettor": 0})  # CFR only
    # C. facing-a-bet decisions: per (agent, category) -> [checks, total, exists_faced]
    faced = defaultdict(lambda: [0, 0, 0])
    # D. bets made: per (agent, category) -> [count, bluffs]
    made = defaultdict(lambda: [0, 0])

    for r in a:
        s = int(r["starter_seat"]); c = int(r["cfr_seat"])
        bets = list(np.flatnonzero(r["history"]))
        if not bets:
            continue
        B = len(bets)
        cards = _cards_from_deal(r["deal"])
        # bet j (1-indexed) actor: starter if j odd else other
        actor = lambda j: s if (j % 2 == 1) else (1 - s)
        ag = lambda seat: "CFR" if seat == c else "NFSP"
        # D: every bet made
        for j, b in enumerate(bets, start=1):
            cat = CATEGORIES[b]
            m = made[(ag(actor(j)), cat)]
            m[0] += 1
            if not _exists(cards, b):
                m[1] += 1
        # C: decisions FACING a bet = raises (j=2..B) chose raise; checker chose check
        for j in range(2, B + 1):                      # raiser at step j faced bets[j-2]
            fb = bets[j - 2]
            key = (ag(actor(j)), CATEGORIES[fb])
            faced[key][1] += 1                          # total faced
            faced[key][2] += 1 if _exists(cards, fb) else 0
            # action = raise (did NOT check)
        checker = s if (B % 2 == 0) else (1 - s)
        fb = bets[-1]
        key = (ag(checker), CATEGORIES[fb])
        faced[key][0] += 1                              # checked
        faced[key][1] += 1
        faced[key][2] += 1 if _exists(cards, fb) else 0
        # B: round loss attribution
        loser = int(r["loser_seat"])
        ag_loser = ag(loser)
        if loser == checker:
            loss[ag_loser]["as_checker"] += 1
            if ag_loser == "CFR":
                loss_by_size[int(r["start_size"] if checker == s else r["nonstart_size"])]["checker"] += 1
        else:
            loss[ag_loser]["as_bettor"] += 1
            if ag_loser == "CFR":
                # bettor of final bet had this many cards
                bettor_size = int(r["start_size"] if (1 - checker) == s else r["nonstart_size"])
                loss_by_size[bettor_size]["bettor"] += 1

    # ---- B report ----
    print("\n=== B. Round-loss attribution (where each agent's losses come from) ===")
    for agn in ("CFR", "NFSP"):
        cw, bw = loss[agn]["as_checker"], loss[agn]["as_bettor"]
        tot = cw + bw
        print(f"  {agn}: {tot} round losses | as-checker (challenged existing bet) "
              f"{cw/tot:.1%} | as-bettor (bluff caught) {bw/tot:.1%}" if tot else f"  {agn}: no losses")
    print("  CFR losses by hand size (checker vs bettor):")
    for sz in sorted(loss_by_size):
        d = loss_by_size[sz]; t = d["checker"] + d["bettor"]
        print(f"    size {sz:2d}: {t:5d} losses | checker {d['checker']/t:.0%} bettor {d['bettor']/t:.0%}" if t else "")

    # ---- C report ----
    print("\n=== C. Check-rate facing a bet, by category (CFR vs Perun vs not-exists) ===")
    print(f"  {'category':16} {'CFR chk':>8} {'Perun chk':>10} {'not-exists':>11} {'(CFR n)':>9}")
    for cat in CAT_ORDER:
        cf = faced.get(("CFR", cat), [0, 0, 0]); nf = faced.get(("NFSP", cat), [0, 0, 0])
        if cf[1] == 0 and nf[1] == 0:
            continue
        cf_chk = cf[0] / cf[1] if cf[1] else float("nan")
        nf_chk = nf[0] / nf[1] if nf[1] else float("nan")
        ne = 1 - (cf[2] / cf[1]) if cf[1] else float("nan")   # not-exists rate of bets CFR faced
        print(f"  {cat:16} {cf_chk:8.3f} {nf_chk:10.3f} {ne:11.3f} {cf[1]:9d}")

    # ---- D report ----
    print("\n=== D. Bet mix + bluff rate by category (bet made that doesn't exist) ===")
    cfr_total = sum(made[("CFR", cat)][0] for cat in CAT_ORDER)
    nf_total = sum(made[("NFSP", cat)][0] for cat in CAT_ORDER)
    print(f"  {'category':16} {'CFR mix':>8} {'CFR bluff':>10} {'Perun mix':>10} {'Perun bluff':>12}")
    for cat in CAT_ORDER:
        cf = made.get(("CFR", cat), [0, 0]); nf = made.get(("NFSP", cat), [0, 0])
        if cf[0] == 0 and nf[0] == 0:
            continue
        print(f"  {cat:16} {cf[0]/cfr_total:8.3f} {(cf[1]/cf[0] if cf[0] else float('nan')):10.3f} "
              f"{nf[0]/nf_total:10.3f} {(nf[1]/nf[0] if nf[0] else float('nan')):12.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inp",
                    default="cfr_ai/analysis/cfr_vs_nfsp_games_current.npy")
    args = ap.parse_args()
    analyze(args.inp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
