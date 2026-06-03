#!/usr/bin/env python3
"""Human-vs-Morana (and Morana-vs-Perun) strategic diagnostic.

Reads the gzipped JSONL dump of the `games` table. Splits live states (bare uuid)
from rounds (uuid_N). Qualifies games: 1v1, one player ai_agent=='cfr' (Morana),
max_cards==11, status=='Finished', last_modified >= cutoff. For each qualifying
game's rounds, parses the betting history and computes, for rounds where
sum(n_cards) >= 7 (the LOSSY abstraction regime):
  - check%  (action 88 rate) by side, bucketed by the previous bet's set-type band
  - bluff%  (bets with p-g < 0 = hand under-supports the claim) by side and band
  - ground-truth truthfulness of the *challenged* bet via action 89 (lost round):
      89.player == 88.player  -> checker lost -> set existed -> bet was truthful
      89.player != 88.player  -> bettor lost  -> set absent  -> bet was a bluff
  - per-(N,N) symmetric-setup round-loss rate, and overall game winrate.

Action ids: 0-87 bets (claims), 88 check (challenge), 89 lost-round (loser).
"""
import gzip
import json
import os
import re
import sys
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cfr_ai.game import Game
from cfr_ai.abstraction.probs import g_vector, _band
from cfr_ai.abstraction.probs_jit import p_vector_fast

CHECK, LOST = 88, 89


def card_int(c):
    return int(c["value"]) * 4 + int(c["colour"])


def analyze_round(rd, morana_nick):
    """Return per-decision records + round outcome for one round item."""
    ncards = {p["nickname"]: int(p["n_cards"]) for p in rd.get("players", [])}
    total = sum(ncards.values())
    hands = {h["nickname"]: [card_int(c) for c in h.get("hand", [])]
             for h in rd.get("hands", [])}
    g = g_vector(total) if total >= 2 else None
    exist = None
    if len(hands) == 2:
        try:
            exist = Game.precompute_set_existence(list(hands.values()))
        except Exception:
            exist = None
    recs = []
    last_bet = None
    last_side = None
    checker = loser = None
    for ev in rd.get("history", []):
        pl = ev.get("player")
        a = int(ev["action_id"])
        side = "Morana" if pl == morana_nick else "Human"
        last_band = _band(last_bet) if last_bet is not None else "ROOT"
        # ground-truth existence of the bet being RESPONDED to (None at ROOT)
        faced = (int(exist[last_bet])
                 if (exist is not None and last_bet is not None) else None)
        if a < CHECK:                      # a bet (claim 0-87)
            hand = hands.get(pl, [])
            pg = None
            if g is not None and 0 <= a < 88:
                n_opp = total - len(hand)
                if 0 <= n_opp <= 24 - len(hand):
                    pg = float(p_vector_fast(hand, n_opp)[a] - g[a])
            recs.append({"t": "bet", "side": side, "band": _band(a),
                         "ctx": last_band, "action": a, "pg": pg, "faced": faced,
                         "exists": (int(exist[a]) if exist is not None else None)})
            last_bet = a
            last_side = side
        elif a == CHECK:
            recs.append({"t": "check", "side": side, "band": None,
                         "ctx": last_band, "faced": faced})
            checker = pl
        elif a == LOST:
            loser = pl
    truthful = None
    if checker is not None and loser is not None:
        truthful = (loser == checker)   # checker lost => challenged set existed
    return {"total": total, "setup": tuple(sorted(ncards.values())),
            "recs": recs, "checker": checker, "loser": loser,
            "challenged_truthful": truthful,
            "last_band": (_band(last_bet) if last_bet is not None else None),
            "last_side": last_side}


def date_ok(last_modified, cutoff_epoch):
    if cutoff_epoch is None or last_modified is None:
        return True
    try:                                  # epoch seconds or ms
        v = float(last_modified)
        if v > 1e12:
            v /= 1000.0
        return v >= cutoff_epoch
    except (ValueError, TypeError):
        return str(last_modified) >= "2025-07-10"   # ISO fallback


def is_morana(p):
    return p.get("ai_agent") == "cfr"


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


def classify_outcome5(recs, responder, resp_lost, oc):
    """Tally `responder`'s decisions vs the band they FACE into 5 outcomes
    (keyed by faced ctx band; resp_lost = did `responder` lose the round):
      chk_lost   challenged a TRUE bet (lost)
      chk_won    challenged a BLUFF (won)
      rai_lost   raised, opp immediately challenged it, lost (caught bluff-raise)
      rai_won    raised, opp immediately challenged it, won  (credible raise, opp wrongly called)
      rai_unchk  raised, opp did NOT immediately challenge (re-raised / round continued)"""
    for i, rec in enumerate(recs):
        if rec["side"] != responder or rec["ctx"] == "ROOT":
            continue
        if rec["t"] == "check":
            cat = "chk_lost" if resp_lost else "chk_won"
        else:
            nxt = recs[i + 1] if i + 1 < len(recs) else None
            if nxt is not None and nxt["t"] == "check":
                cat = "rai_lost" if resp_lost else "rai_won"
            else:
                cat = "rai_unchk"
        oc[(rec["ctx"], cat)] += 1


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


# --------------------------------------------------------------------------- #
def run(dump_path, cutoff_epoch):
    op = gzip.open if dump_path.endswith(".gz") else open
    live, rounds = {}, defaultdict(dict)
    n = 0
    with op(dump_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            gid = d["game_uuid"]
            base, _, rn = gid.rpartition("_")
            if base and rn.isdigit():
                rounds[base][int(rn)] = d
            else:
                live[gid] = d
    print(f"loaded {n} items: {len(live)} live, {len(rounds)} games-with-rounds")

    # all keyed by (side, ctx-band) where ctx = the band being RESPONDED to
    chk = defaultdict(int)       # checks (challenges)
    bet = defaultdict(int)       # bets (raises)
    bluff = defaultdict(int)     # raises with p-g<0 (bluff INTENT)
    truthful = defaultdict(int)  # raises whose own claimed set exists (ex-post)
    # decisions split by whether the FACED (responded-to) set exists:
    dec_e = defaultdict(int); chk_e = defaultdict(int)   # faced set EXISTS
    dec_a = defaultdict(int); chk_a = defaultdict(int)   # faced set ABSENT
    sym_loss = defaultdict(lambda: [0, 0])   # (N,N) -> [morana_losses, rounds]
    oc_m = defaultdict(int)                  # (faced-band, cat) Morana as responder
    oc_o = defaultdict(int)                  # (faced-band, cat) opponent as responder
    wins = losses = 0
    qual = 0
    for gid, ls in live.items():
        pls = ls.get("players", [])
        if len(pls) != 2 or ls.get("status") != "Finished":
            continue
        if int(ls.get("max_cards", 0)) != 11:
            continue
        mor = [p for p in pls if is_morana(p)]
        hum = [p for p in pls if not is_morana(p)]
        if len(mor) != 1 or len(hum) != 1:
            continue
        if not date_ok(ls.get("last_modified"), cutoff_epoch):
            continue
        qual += 1
        mnick = mor[0]["nickname"]
        # game outcome: loser of the final (max round_number) round
        grs = rounds.get(gid, {})
        if grs:
            last_rn = max(grs)
            fin = analyze_round(grs[last_rn], mnick)
            if fin["loser"] is not None:
                if fin["loser"] == mnick:
                    losses += 1
                else:
                    wins += 1
        for rn, rd in grs.items():
            r = analyze_round(rd, mnick)
            if r["setup"][0] == r["setup"][1]:    # symmetric NvN (ALL N, incl rounds 1-5)
                key = r["setup"][0]
                sym_loss[key][1] += 1
                if r["loser"] == mnick:
                    sym_loss[key][0] += 1
            if r["total"] < 7:           # band tables: lossy regime (rounds 6+) only
                continue
            for rec in r["recs"]:
                k = (rec["side"], rec["ctx"])
                is_chk = rec["t"] == "check"
                if is_chk:
                    chk[k] += 1
                else:
                    bet[k] += 1
                    if rec["pg"] is not None and rec["pg"] < 0:
                        bluff[k] += 1
                    if rec.get("exists") is not None:
                        truthful[k] += rec["exists"]
                fc = rec.get("faced")            # responded-to set existence
                if fc == 1:
                    dec_e[k] += 1; chk_e[k] += int(is_chk)
                elif fc == 0:
                    dec_a[k] += 1; chk_a[k] += int(is_chk)
            if r["loser"] is not None:
                classify_outcome5(r["recs"], "Morana", r["loser"] == mnick, oc_m)
                classify_outcome5(r["recs"], "Human", r["loser"] != mnick, oc_o)

    print(f"\nqualifying 1v1/cfr/mc11/Finished games: {qual}")
    print(f"overall: Morana wins {wins}, losses {losses}  "
          f"(winrate {wins/(wins+losses)*100:.1f}%)" if wins + losses else "no outcomes")
    bands = ["ROOT", "high", "pair", "2pair", "straight", "trips",
             "fullhouse", "flush", "quads", "sflush", "great_sf"]
    print_behavior_table(bands, "Human", chk, bet, bluff, truthful,
                         dec_e, chk_e, dec_a, chk_a)
    print("\n=== Morana's round win-rate by symmetric setup NvN (ALL N: 1-3 = rounds 1-5) ===")
    print(f"{'N':>4} {'Morana win%':>13} {'rounds':>8}")
    for nn in sorted(sym_loss):
        l, t = sym_loss[nn]
        if t:
            print(f"{nn:>4} {(t - l) / t * 100:>12.1f}% {t:>8}")
    print_outcome_compare(bands, oc_m, oc_o, "Human")


def test_file(path):
    with open(path, encoding="utf-8") as f:
        rd = json.load(f)
    pls = rd.get("players", [])
    mnick = pls[0]["nickname"] if pls else "0"
    r = analyze_round(rd, mnick)
    print(f"file={os.path.basename(path)} total={r['total']} setup={r['setup']} "
          f"checker={r['checker']} loser={r['loser']} challenged_truthful={r['challenged_truthful']}")
    for rec in r["recs"]:
        print("  ", rec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="cfr_ai/analysis/data/games_current.jsonl.gz")
    ap.add_argument("--cutoff-epoch", type=float, default=None,
                    help="unix seconds; games with last_modified before are dropped")
    ap.add_argument("--test-file", default=None, help="parse one round JSON and dump records")
    args = ap.parse_args()
    if args.test_file:
        test_file(args.test_file)
    else:
        run(args.dump, args.cutoff_epoch)
