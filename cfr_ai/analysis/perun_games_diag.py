#!/usr/bin/env python3
"""Morana-vs-Perun diagnostic, mirroring morana_diag but reading the
cfr_vs_nfsp per-round structured array. Reconstructs the betting sequence from
the monotonic-bet bitmask + alternation from starter_seat, then computes the
same check% / bluff% / mean-(p-g) by last-bet band (rounds 6+), per side."""
import argparse
import os
import sys
from collections import defaultdict
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cfr_ai.game import Game
from cfr_ai.abstraction.probs import g_vector, _band
from cfr_ai.abstraction.probs_jit import p_vector_fast


@lru_cache(maxsize=300000)
def pg_of(hand, total, a):
    n_opp = total - len(hand)
    return float(p_vector_fast(list(hand), n_opp)[a] - g_vector(total)[a])


def _pct(num, den):
    return f"{num / den * 100:5.1f}" if den else "    -"


def print_behavior_table(bands, opp, chk, bet, bluff, truthful,
                         dec_e, chk_e, dec_a, chk_a):
    """Merged response-behaviour table by responded-to (ctx) band, rounds 6+.
    M = Morana, O = opponent. Every column is conditioned on the band being
    RESPONDED to (ctx = last bet before the decision):
      %T       share of the faced bets that are true (the OTHER side's truthfulness at
               band X): M:%T = opp's band-X bets that are true; O:%T = Morana's.
      chkT     check-rate when faced set is TRUE/exists  (challenging a TRUE bet -> loses)
      chkF     check-rate when faced set is FALSE/absent (challenging a BLUFF -> wins)
      blf%     of this side's RAISES, % bluffed (p-g<0, intent)
      tru%     of this side's RAISES, % whose own claimed set exists (ex-post)
               == response CREDIBILITY: tru% < 50 => a rational opponent always calls.
    nM/nO = total decisions per side at that band (watch thin cells)."""
    print(f"\n=== Behavioural signature by responded-to band (rounds 6+) -- "
          f"Morana(M) vs {opp}(O) ===")
    print(f"{'band':9} {'M:%T':>7} {'M:chkT':>7} {'M:chkF':>7} "
          f"{'O:%T':>7} {'O:chkT':>7} {'O:chkF':>7} {'M:blf%':>7} {'O:blf%':>7} "
          f"{'M:tru%':>7} {'O:tru%':>7} {'nM/nO':>11}")
    for b in bands:
        M, O = ("Morana", b), (opp, b)
        nM, nO = chk[M] + bet[M], chk[O] + bet[O]
        if nM + nO == 0:
            continue
        print(f"{b:9} "
              f"{_pct(dec_e[M], dec_e[M] + dec_a[M]):>7} "
              f"{_pct(chk_e[M], dec_e[M]):>7} {_pct(chk_a[M], dec_a[M]):>7} "
              f"{_pct(dec_e[O], dec_e[O] + dec_a[O]):>7} "
              f"{_pct(chk_e[O], dec_e[O]):>7} {_pct(chk_a[O], dec_a[O]):>7} "
              f"{_pct(bluff[M], bet[M]):>7} {_pct(bluff[O], bet[O]):>7} "
              f"{_pct(truthful[M], bet[M]):>7} {_pct(truthful[O], bet[O]):>7} "
              f"{f'{nM}/{nO}':>11}")


OC5 = ["chk_lost", "chk_won", "rai_lost", "rai_won", "rai_unchk"]
OC5_H = ["cl", "cw", "rl", "rw", "ru"]


def tally_outcome5_perun(bets, starter, m, checker_seat, resp_seat, resp_lost, oc):
    """5-way response outcomes for `resp_seat` (rounds 6+), keyed by faced band,
    reconstructed from the monotonic bet sequence:
      non-final bet by resp                  -> rai_unchk
      final bet by resp (challenged by opp)  -> rai_lost / rai_won
      resp's terminal check                  -> chk_lost / chk_won
    resp_lost = did resp lose the round."""
    for k in range(m):
        bettor = starter if k % 2 == 0 else 1 - starter
        if bettor != resp_seat or k == 0:
            continue
        if k + 1 < m:
            cat = "rai_unchk"
        else:
            cat = "rai_lost" if resp_lost else "rai_won"
        oc[(_band(bets[k - 1]), cat)] += 1
    if checker_seat == resp_seat and m >= 1:
        oc[(_band(bets[m - 1]), "chk_lost" if resp_lost else "chk_won")] += 1


def print_outcome_compare(bands, oc_m, oc_o, opp="Opp"):
    """Side-by-side 5-way response outcomes, Morana(M) vs opponent(O), by faced band."""
    print(f"\n=== response outcomes by faced band (rounds 6+) -- M=Morana, O={opp} ===")
    print("  cl=chk&lost cw=chk&won rl=rai&lost(called) rw=rai&won(called) "
          "ru=rai&unchecked   (% of n)")
    print(f"{'band':9} " + "".join(f"{'M:' + h:>6}" for h in OC5_H)
          + "  " + "".join(f"{'O:' + h:>6}" for h in OC5_H) + f"{'nM/nO':>13}")
    for b in bands:
        nM = sum(oc_m[(b, c)] for c in OC5)
        nO = sum(oc_o[(b, c)] for c in OC5)
        if nM + nO == 0:
            continue
        vm = "".join(f"{(oc_m[(b, c)] / nM * 100 if nM else 0):>6.1f}" for c in OC5)
        vo = "".join(f"{(oc_o[(b, c)] / nO * 100 if nO else 0):>6.1f}" for c in OC5)
        print(f"{b:9} {vm}  {vo}{f'{nM}/{nO}':>13}")


def run(path):
    arr = np.load(path, allow_pickle=True)
    print(f"{path}: {len(arr)} rounds")
    chk = defaultdict(int)
    bet = defaultdict(int)
    bluff = defaultdict(int)
    truthful = defaultdict(int)
    # decisions split by whether the FACED (responded-to) set exists:
    dec_e = defaultdict(int); chk_e = defaultdict(int)   # faced set EXISTS
    dec_a = defaultdict(int); chk_a = defaultdict(int)   # faced set ABSENT
    sym_loss = defaultdict(lambda: [0, 0])
    oc_m = defaultdict(int)   # (faced-band, cat) Morana as responder, rounds 6+
    oc_o = defaultdict(int)   # (faced-band, cat) Perun as responder, rounds 6+
    game_final = {}   # game_id -> (max_round_idx, loser_seat, cfr_seat)

    for row in arr:
        cfr_seat = int(row["cfr_seat"])
        starter = int(row["starter_seat"])
        total = int(row["start_size"]) + int(row["nonstart_size"])
        deal = row["deal"]
        gid = int(row["game_id"])
        ridx = int(row["round_idx"])
        # game outcome bookkeeping (final round = max round_idx)
        if gid not in game_final or ridx > game_final[gid][0]:
            game_final[gid] = (ridx, int(row["loser_seat"]), cfr_seat)
        # symmetric per-(N,N) round loss -- ALL N (incl rounds 1-5)
        sym = int(row["start_size"]) == int(row["nonstart_size"])
        if sym:
            n = int(row["start_size"])
            sym_loss[n][1] += 1
            if int(row["loser_seat"]) == cfr_seat:
                sym_loss[n][0] += 1
        if total < 7:           # band tables: rounds 6+ only
            continue
        h0 = [c for c in range(24) if deal[c] == 0]
        h1 = [c for c in range(24) if deal[c] == 1]
        hands = {0: tuple(h0), 1: tuple(h1)}
        try:
            exist = Game.precompute_set_existence([h0, h1])
        except Exception:
            exist = None
        bets = [a for a in range(88) if row["history"][a]]
        last_bet = None
        for k, a in enumerate(bets):
            bettor = starter if k % 2 == 0 else 1 - starter
            side = "Morana" if bettor == cfr_seat else "Perun"
            ctx = _band(last_bet) if last_bet is not None else "ROOT"
            key = (side, ctx)
            bet[key] += 1
            if pg_of(hands[bettor], total, a) < 0:
                bluff[key] += 1
            if exist is not None:
                truthful[key] += int(exist[a])
            fc = (int(exist[last_bet])
                  if (exist is not None and last_bet is not None) else None)
            if fc == 1:
                dec_e[key] += 1
            elif fc == 0:
                dec_a[key] += 1
            last_bet = a
        m = len(bets)
        checker = starter if m % 2 == 0 else 1 - starter
        side = "Morana" if checker == cfr_seat else "Perun"
        ctx = _band(last_bet) if last_bet is not None else "ROOT"
        ckey = (side, ctx)
        chk[ckey] += 1
        fc = (int(exist[last_bet])
              if (exist is not None and last_bet is not None) else None)
        if fc == 1:
            dec_e[ckey] += 1; chk_e[ckey] += 1
        elif fc == 0:
            dec_a[ckey] += 1; chk_a[ckey] += 1
        # 5-category response outcomes for both seats, keyed by faced band
        mor_lost = (int(row["loser_seat"]) == cfr_seat)
        tally_outcome5_perun(bets, starter, m, checker, cfr_seat, mor_lost, oc_m)
        tally_outcome5_perun(bets, starter, m, checker, 1 - cfr_seat, not mor_lost, oc_o)

    w = l = 0
    for gid, (_, loser, cseat) in game_final.items():
        if loser == cseat:
            l += 1
        else:
            w += 1
    print(f"games {len(game_final)} | Morana wins {w} losses {l} "
          f"(winrate {w/(w+l)*100:.1f}%)")
    bands = ["ROOT", "high", "pair", "2pair", "straight", "trips",
             "fullhouse", "flush", "quads", "sflush", "great_sf"]
    print_behavior_table(bands, "Perun", chk, bet, bluff, truthful,
                         dec_e, chk_e, dec_a, chk_a)
    print("\n=== Morana's round win-rate by symmetric setup NvN (ALL N: 1-3 = rounds 1-5) ===")
    print(f"{'N':>4} {'Morana win%':>13} {'rounds':>8}")
    for nn in sorted(sym_loss):
        ll, tt = sym_loss[nn]
        if tt:
            print(f"{nn:>4} {(tt - ll) / tt * 100:>12.1f}% {tt:>8}")
    print_outcome_compare(bands, oc_m, oc_o, "Perun")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", default="cfr_ai/analysis/data/cfr_vs_nfsp_games_v1_greedy.npy")
    args = ap.parse_args()
    run(args.npy)
