#!/usr/bin/env python3
"""Memory/storage-allocation audit by responded-to (last-bet) band, rounds 6+.

Three distributions per band:
  faced%   % of Morana's decisions (real V1 games) made in response to that band
           -- how often the band actually comes up in play.
  RAM%     % of ALL infosets whose last_bet is in that band (incl. pure-check nodes)
           -- the training/RAM footprint.
  store%   % of NON-pure-check infosets in that band (what the deployed table keeps)
           -- the storage footprint.

Infoset counts come from the V1 *CSV* archive (archive/v1_csv/outputs/<setup>/),
which partitions infosets into files <last_bet>.csv under per-hand-size dirs:
  <hs>/             = final strategy -> non-pure-check infosets only  (store)
  <hs>_diagnostic/  = full table     -> ALL infosets incl pure-check  (RAM)
(The deployment .npz format drops pure-check infosets, so RAM can't be read there.)
last_bet is the filename; rows are counted directly. Setups with total >= 7 only.
"""
import argparse
import gzip
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cfr_ai.abstraction.probs import _band
from cfr_ai.analysis.morana_games_diag import (analyze_round, is_morana, date_ok)

BANDS = ["ROOT", "high", "pair", "2pair", "straight", "trips",
         "fullhouse", "flush", "quads", "sflush", "great_sf"]


def band_of_lastbet(lb):
    return "ROOT" if int(lb) == 88 else _band(int(lb))


def count_rows(path):
    """Fast line count minus the `k,v` header."""
    n = 0
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1 << 20), b""):
            n += buf.count(b"\n")
    return max(0, n - 1)


def model_allocation(csv_root):
    ram = defaultdict(int)        # all infosets (diagnostic dirs)
    store = defaultdict(int)      # non-pure-check infosets (final dirs)
    setups = []
    out = os.path.join(csv_root, "outputs")
    for setup in sorted(os.listdir(out)):
        sp = os.path.join(out, setup)
        parts = setup.split("_")
        if (not os.path.isdir(sp) or len(parts) != 2
                or not all(p.isdigit() for p in parts)
                or int(parts[0]) + int(parts[1]) < 7):
            continue
        used = False
        for sub in os.listdir(sp):
            subp = os.path.join(sp, sub)
            if not os.path.isdir(subp):
                continue
            if sub.endswith("_diagnostic"):
                tgt = ram
            elif sub.isdigit():
                tgt = store
            else:
                continue
            for fn in os.listdir(subp):
                if fn.endswith(".csv") and fn[:-4].isdigit():
                    tgt[band_of_lastbet(int(fn[:-4]))] += count_rows(os.path.join(subp, fn))
                    used = True
        if used:
            setups.append(setup)
    return ram, store, setups


def faced_freq(dump, cutoff):
    """Morana's faced-band distribution (rounds 6+) from the human games."""
    op = gzip.open if dump.endswith(".gz") else open
    live, rounds = {}, defaultdict(dict)
    with op(dump, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            gid = d["game_uuid"]
            base, _, rn = gid.rpartition("_")
            if base and rn.isdigit():
                rounds[base][int(rn)] = d
            else:
                live[gid] = d
    faced = defaultdict(int)
    for gid, ls in live.items():
        pls = ls.get("players", [])
        if len(pls) != 2 or ls.get("status") != "Finished":
            continue
        if int(ls.get("max_cards", 0)) != 11:
            continue
        mor = [p for p in pls if is_morana(p)]
        hum = [p for p in pls if not is_morana(p)]
        if len(mor) != 1 or len(hum) != 1 or not date_ok(ls.get("last_modified"), cutoff):
            continue
        mnick = mor[0]["nickname"]
        for rn, rd in rounds.get(gid, {}).items():
            r = analyze_round(rd, mnick)
            if r["total"] < 7:
                continue
            for rec in r["recs"]:
                if rec["side"] == "Morana":
                    faced[rec["ctx"]] += 1
    return faced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-root", default="cfr_ai/archive/v1_csv")
    ap.add_argument("--dump", default="cfr_ai/analysis/data/games_current.jsonl.gz")
    ap.add_argument("--cutoff-epoch", type=float, default=1752105600.0)
    args = ap.parse_args()

    ram, store, setups = model_allocation(args.csv_root)
    faced = faced_freq(args.dump, args.cutoff_epoch)
    tot_r = sum(ram.values()) or 1
    tot_s = sum(store.values()) or 1
    tot_f = sum(faced.values()) or 1

    print(f"csv={args.csv_root}  setups(total>=7)={len(setups)}  "
          f"RAM(all)={tot_r:,}  store(non-check)={tot_s:,}  "
          f"pure-check={tot_r - tot_s:,}  faced-decisions={tot_f:,}")
    print(f"\n{'band':10} {'faced%':>8} {'RAM%':>7} {'store%':>7} "
          f"{'n_RAM':>13} {'n_store':>13}")
    for b in BANDS:
        if ram[b] == 0 and store[b] == 0 and faced[b] == 0:
            continue
        print(f"{b:10} {faced[b]/tot_f*100:>7.1f}% {ram[b]/tot_r*100:>6.1f}% "
              f"{store[b]/tot_s*100:>6.1f}% {ram[b]:>13,} {store[b]:>13,}")
    print(f"\nsetups used ({len(setups)}): {', '.join(setups)}")


if __name__ == "__main__":
    main()
