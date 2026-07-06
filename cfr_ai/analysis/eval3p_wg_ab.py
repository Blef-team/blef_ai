"""3-player WHOLE-GAME first-out A/B between two CFR model folders (depth-2 vs depth-3).

Plays 3p games from 1,1,1 to the first elimination with mixed seats, reports each
version's first-out rate (per appearance; LOWER = stronger; fair = 33.3%). Reuses the
depth-aware `eval3p_serve.serve` but with a RAM-BOUNDED LRU cache (whole games traverse
many setups; an unbounded cache OOMs the box). No NFSP/conservative — just the two CFR
models, so no torch in forked workers.

    python -m cfr_ai.analysis.eval3p_wg_ab --base-a cfr_ai/p3_2x_hist2/outputs \
        --base-b cfr_ai/p3_2x/outputs --n 3000 --procs 6 --out wg_ab.csv
    (A = depth-2, B = depth-3)
"""
import argparse
import csv
import itertools
import math
import os
import sys
from collections import OrderedDict
from multiprocessing import Pool

import shared.api.simpleschema_local_manager as gm
from cfr_ai.analysis.eval3p_serve import serve


class LRU(OrderedDict):
    """Bounded LRU: evict oldest past `cap`. A single game's setup path is short
    (~escalation depth), so cap=24 avoids in-game reloads while bounding RAM."""
    def __init__(self, cap=24):
        super().__init__()
        self.cap = cap

    def __getitem__(self, k):
        self.move_to_end(k)
        return OrderedDict.__getitem__(self, k)

    def __setitem__(self, k, v):
        OrderedDict.__setitem__(self, k, v)
        self.move_to_end(k)
        while len(self) > self.cap:
            self.popitem(last=False)


WORKER = {}


def _init(base_a, base_b):
    # Unbounded cache: loading (npz decompress / mmap open) is far slower than the
    # policy lookup, and whole games traverse most of the 176 setups, so every setup
    # should load ONCE and stay. Kept low-RAM by staging the mmap layout first (each
    # strategy ~50 MB lazy-paged), matching cfr_vs_cfr_games.
    ca, cb = {}, {}
    WORKER["fns"] = {"A": lambda g: serve(g, base_a, ca),
                     "B": lambda g: serve(g, base_b, cb)}


def play_to_first_elim(seat_fns, seed):
    import random
    random.seed(seed)
    g = gm.create_game(3, deck_size=24, max_cards=8, init_card_dist=[1, 1, 1])
    for _ in range(4000):
        if g.get("status") == "Finished":
            break
        cp = int(g["cp_nickname"])
        a = int(seat_fns[cp](g))
        gm.play(g, a, save_dir=None)
        out = [int(p["nickname"]) for p in g["players"] if int(p["n_cards"]) == 0]
        if out:
            return out[0]
    return None


def run_perm(task):
    perm, n, seedb = task
    fns = WORKER["fns"]
    seat_fns = [fns[perm[s]] for s in range(3)]
    fo, app = {}, {}
    games = unres = 0
    for i in range(n):
        e = play_to_first_elim(seat_fns, seed=(seedb + i) & 0x7fffffff)
        if e is None:
            unres += 1
            continue
        games += 1
        for s in range(3):
            app[perm[s]] = app.get(perm[s], 0) + 1
        fo[perm[e]] = fo.get(perm[e], 0) + 1
    return {"fo": fo, "app": app, "games": games, "unres": unres}


# Two compositions so A and B get equal total appearances.
COMPS = [("A", "B", "B"), ("A", "A", "B")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-a", required=True, help="depth-2 outputs dir (A)")
    ap.add_argument("--base-b", required=True, help="depth-3 outputs dir (B)")
    ap.add_argument("--n", type=int, default=3000, help="games per distinct seat-perm")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = []
    for ci, comp in enumerate(COMPS):
        for pi, perm in enumerate(sorted(set(itertools.permutations(comp)))):
            tasks.append((perm, args.n, (ci * 100 + pi) * 1_000_003 + 7))

    fo, app = {}, {}
    pool = Pool(args.procs, initializer=_init, initargs=(args.base_a, args.base_b))
    done = 0
    for r in pool.imap_unordered(run_perm, tasks):
        done += 1
        for l, c in r["fo"].items():
            fo[l] = fo.get(l, 0) + c
        for l, c in r["app"].items():
            app[l] = app.get(l, 0) + c
        print(f"[{done}/{len(tasks)}] games={r['games']} unres={r['unres']}", flush=True)

    lab = {"A": "depth-2", "B": "depth-3"}
    print("\n==== 3p WHOLE-GAME FIRST-OUT (per appearance; lower=stronger; fair=33.3%) ====")
    for l in ("A", "B"):
        rate = fo.get(l, 0) / app[l] if app.get(l) else 0
        print("  %-8s %.3f%%  (%d first-outs / %d appearances)"
              % (lab[l], rate * 100, fo.get(l, 0), app.get(l, 0)))
    ra = fo.get("A", 0) / app["A"]; rb = fo.get("B", 0) / app["B"]
    # z on the difference of two proportions (independent-ish appearances)
    pa_, na = ra, app["A"]; pb_, nb = rb, app["B"]
    se = math.sqrt(pa_ * (1 - pa_) / na + pb_ * (1 - pb_) / nb)
    z = (ra - rb) / se if se else 0
    who = "depth-2 stronger" if ra < rb else "depth-3 stronger"
    print("  -> depth-2 first-out %+.2fpp vs depth-3  (z=%+.2f)  => %s"
          % ((ra - rb) * 100, z, who))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["version", "first_out", "appearances", "rate"])
        for l in ("A", "B"):
            w.writerow([lab[l], fo.get(l, 0), app.get(l, 0),
                        fo.get(l, 0) / app[l] if app.get(l) else 0])

    # Results are printed + saved. The Pool's clean shutdown (join) deadlocks under
    # numba+fork, so terminate the workers and hard-exit past it.
    sys.stdout.flush()
    pool.terminate()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
