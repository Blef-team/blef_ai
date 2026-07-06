"""Whole-game 3-player eval, measured to the FIRST elimination (no 1v1 endgame).
Each game starts 1_1_1; round losers gain a card; the first player to hit
max_cards (8) is eliminated. 1 CFR + 1 conservative + 1 NFSP, seat assignment
permuted over all 6 orders to vary direction. Reports each agent's FIRST-OUT
rate (lower = stronger; the first player eliminated is the weakest that game).

Only the 3-player strategies (cfr_ai/p3/outputs) are needed.
"""
import itertools
import os
import random
from collections import OrderedDict

import psutil

import shared.api.simpleschema_local_manager as gm
from cfr_ai.analysis.eval3p_serve import serve as cfr3_serve

NAMES = ("CFR", "conservative", "NFSP")
P3_OUT = "cfr_ai/p3/outputs"


class _RSSCache(OrderedDict):
    """RAM-budgeted LRU strategy cache. The full 3p strategy set (~26 GB shipped,
    more for 2x) exceeds this 30 GB box, so a plain dict OOMs. Cap by process RSS
    instead of count: whole-games hit the small low/mid-total setups constantly
    and the big high-total ones only briefly near elimination, so eviction falls
    on the rarely-used big strategies and reloads stay minimal."""

    def __init__(self, budget_gb=20.0):
        super().__init__()
        self._budget = budget_gb * 1e9
        self._proc = psutil.Process(os.getpid())

    def get(self, key, default=None):
        if key in self:
            self.move_to_end(key)
            return OrderedDict.__getitem__(self, key)
        return default

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self.move_to_end(key)
        while len(self) > 1 and self._proc.memory_info().rss > self._budget:
            self.popitem(last=False)


def play_to_first_elim(seat_fns, cfr_cache, seed):
    """Play from 1_1_1 until one player is eliminated (n_cards -> 0). Returns the
    first-eliminated seat, or None on guard/forfeit."""
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


def run(cons_fn, nfsp_fn, n_per_perm=150):
    cfr_cache = _RSSCache(20.0)
    cfr = lambda g: cfr3_serve(g, P3_OUT, cfr_cache)
    base = {"CFR": cfr, "conservative": cons_fn, "NFSP": nfsp_fn}
    first_out = {n: 0 for n in NAMES}
    games = unresolved = 0
    for perm in itertools.permutations(NAMES):
        seat_fns = [base[perm[s]] for s in range(3)]
        for i in range(n_per_perm):
            e = play_to_first_elim(seat_fns, cfr_cache, seed=hash((perm, i)) & 0x7fffffff)
            if e is None:
                unresolved += 1
                continue
            first_out[perm[e]] += 1
            games += 1
    return {"games": games, "unresolved": unresolved,
            "first_out_rate": {n: first_out[n] / games if games else float("nan") for n in NAMES},
            "first_out": first_out}


def _fmt(res):
    r = res["first_out_rate"]
    parts = " | ".join(f"{n} {r[n]*100:.1f}%" for n in NAMES)
    return f"FIRST-OUT rate (lower=stronger, fair=33.3%): {parts}  [{res['games']} games, {res['unresolved']} unresolved]"
