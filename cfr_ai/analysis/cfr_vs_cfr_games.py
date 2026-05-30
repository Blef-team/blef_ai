"""Whole-game CFR-vs-CFR simulator with the real inter-round mechanics.

The intra-CFR analogue of cfr_vs_nfsp_games.py: plays full real-mechanic Blef
games (start 1v1 card, round loser gains a card and opens the next round, losing
at max hand size 11 ends the game) between TWO CFR model versions (e.g. V2.1 vs
the V2 baseline), recording one compact row per round. This yields BOTH:
  * the whole-game win-rate of model B over model A (occurrence-weighted,
    compounded over the escalation), and
  * a per-(a,b)-setup round win-rate breakdown (the setup IS a concept here,
    unlike CFR-vs-NFSP).

Why a new tool: the deployed `cfr_ai.agent` uses module-global state (one
outputs-base + a 1-slot cache), so it cannot host two model versions in one
process. `CFRAgent` below replicates `cfr_ai.agent.determine_action` EXACTLY
(same key build, same lookup, same sampling) but as an instance with its own
outputs-base and 1-slot cache, so two versions can play head-to-head. Decision
parity vs the deployed agent is checked by scratch/verify_cfr_agent.py.

Model A = the BASELINE (model1), Model B = the NEW version (model2); B's seat is
game_id % 2 (B opens exactly half the games). Win-rate reported is for B.

Run from the repo root:
    python -m cfr_ai.analysis.cfr_vs_cfr_games \
        --model-a-folder cfr_ai --model-b-folder cfr_ai/v2.1 \
        --games 10000 --workers 6 --out cfr_ai/analysis/cfr_vs_cfr_v21_v2.npy
"""

import argparse
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

DECK_SIZE = 24
MAX_CARDS = 11
N_BETS = 88

ROW_DTYPE = np.dtype([
    ("game_id", "u4"), ("round_idx", "u1"), ("b_seat", "u1"),
    ("starter_seat", "u1"), ("start_size", "u1"), ("nonstart_size", "u1"),
    ("deal", "u1", (24,)), ("history", "u1", (N_BETS,)),
    ("loser_seat", "u1"), ("final_bet_existed", "u1"),
])


# --------------------------------------------------------------------------
# A standalone CFR agent. To GUARANTEE decision-parity with production (rather
# than re-deriving the key/lookup/sampling and risking drift), we load an
# INDEPENDENT instance of the deployed `cfr_ai.agent` module via importlib —
# same code, but its own module globals (_OUTPUTS_BASE + 1-slot cache). Two
# such instances can therefore serve two model versions in one process.
# --------------------------------------------------------------------------

import importlib.util


def _load_independent_agent_module(instance_tag):
    """Return a fresh, independent instance of cfr_ai.agent (its own globals)."""
    import cfr_ai.agent as _ref
    spec = importlib.util.spec_from_file_location(
        f"cfr_ai_agent_{instance_tag}", _ref.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CFRAgent:
    def __init__(self, outputs_base, instance_tag):
        # outputs_base is the dir CONTAINING <setup>/strategy.npz.
        self._mod = _load_independent_agent_module(instance_tag)
        self._mod.set_outputs_base(outputs_base)

    def determine_action(self, game_state: dict) -> int:
        return self._mod.determine_action(game_state)


# --------------------------------------------------------------------------
# Whole-game playout (mirrors cfr_vs_nfsp_games._play_game; B seat = id % 2)
# --------------------------------------------------------------------------

def _play_game(game_id, agent_a, agent_b, round_guard=300, game_guard=120):
    import shared.api.simpleschema_local_manager as gm
    b_seat = str(game_id % 2)              # seat playing model B (the new one)
    rows = []
    game = gm.create_game(2, deck_size=DECK_SIZE, max_cards=MAX_CARDS,
                          init_card_dist=[1, 1])
    round_idx = 0
    for _ in range(game_guard):
        if game.get("status") == "Finished":
            break
        starter = game["cp_nickname"]
        counts = {p["nickname"]: int(p["n_cards"]) for p in game["players"]}
        other = "1" if starter == "0" else "0"
        deal = np.full(24, 255, dtype=np.uint8)
        for h in game["hands"]:
            seat = int(h["nickname"])
            for c in h["hand"]:
                deal[int(c["value"]) * 4 + int(c["colour"])] = seat
        history = np.zeros(N_BETS, dtype=np.uint8)
        before = dict(counts)
        resolved = False
        for _ in range(round_guard):
            cp = game["cp_nickname"]
            try:
                act = int((agent_b if cp == b_seat else agent_a).determine_action(game))
                gm.play(game, act, save_dir=None)
            except Exception:
                return None
            after = {p["nickname"]: int(p["n_cards"]) for p in game["players"]}
            if after != before:
                checker_seat = int(cp)
                loser = None
                for nick, b in before.items():
                    af = after.get(nick, b)
                    if af > b or (b > 0 and af == 0):
                        loser = nick
                        break
                if loser is None:
                    return None
                loser_seat = int(loser)
                rows.append((
                    game_id, round_idx, game_id % 2, int(starter),
                    counts[starter], counts[other],
                    deal, history,
                    loser_seat, 1 if loser_seat == checker_seat else 0,
                ))
                round_idx += 1
                resolved = True
                break
            elif 0 <= act < N_BETS:
                history[act] = 1
        if not resolved:
            return None
    return rows


# --------------------------------------------------------------------------
# Parallel workers
# --------------------------------------------------------------------------
_A = None
_B = None


def _init_worker(a_base, b_base, base_seed):
    global _A, _B
    _A = CFRAgent(a_base, "a")
    _B = CFRAgent(b_base, "b")
    random.seed(base_seed + os.getpid())


def _game_task(game_id):
    return _play_game(game_id, _A, _B)


def _ensure_mmap(outputs_base):
    """Stage sparse-mmap sidecars for any setup that lacks them. Additive —
    never modifies strategy.npz. Returns count staged."""
    from cfr_ai.strategy_io import write_mmap_layout
    staged = 0
    for name in sorted(os.listdir(outputs_base)):
        sd = os.path.join(outputs_base, name)
        if not os.path.isdir(sd):
            continue
        if not os.path.exists(os.path.join(sd, "strategy.npz")):
            continue
        if os.path.exists(os.path.join(sd, "probs_sparse_indices.npy")):
            continue
        write_mmap_layout(sd, sd)
        staged += 1
    return staged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-a-folder", required=True,
                    help="Baseline model folder (contains outputs/). Model A.")
    ap.add_argument("--model-b-folder", required=True,
                    help="New model folder (contains outputs/). Model B; win-rate is B's.")
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-stage", action="store_true",
                    help="Skip sparse-mmap staging (use if dirs are read-only / "
                         "already staged).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    a_base = os.path.join(args.model_a_folder, "outputs")
    b_base = os.path.join(args.model_b_folder, "outputs")
    for base in (a_base, b_base):
        if not os.path.isdir(base):
            print(f"[error] not found: {base}", file=sys.stderr)
            return 1
    if not args.no_stage:
        for base in (a_base, b_base):
            n = _ensure_mmap(base)
            if n:
                print(f"[stage] {n} setups -> sparse-mmap in {base}", flush=True)

    out = args.out or "cfr_ai/analysis/cfr_vs_cfr_games.npy"
    print(f"[cfr_vs_cfr_games] A(baseline)={args.model_a_folder} "
          f"B(new)={args.model_b_folder}; {args.games} games, "
          f"{args.workers} workers -> {out}", flush=True)

    rows = []
    t0 = time.time()
    done = bad = 0

    def _save():
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        arr = np.array(rows, dtype=ROW_DTYPE) if rows else np.empty(0, ROW_DTYPE)
        tmp = out + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, out)

    def _record(game_rows):
        nonlocal done, bad
        done += 1
        if game_rows is None:
            bad += 1
        else:
            rows.extend(game_rows)
        if done % 500 == 0 or done == args.games:
            _save()
            el = time.time() - t0
            print(f"[{done}/{args.games}] {len(rows)} rounds | bad={bad} | "
                  f"{done/el:.1f} games/s | {el:.0f}s", flush=True)

    if args.workers <= 1:
        _init_worker(a_base, b_base, args.seed)
        for gid in range(args.games):
            _record(_play_game(gid, _A, _B))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=(a_base, b_base, args.seed)) as pool:
            for game_rows in pool.imap_unordered(_game_task, range(args.games),
                                                 chunksize=8):
                _record(game_rows)

    _save()
    print(f"\nWrote {len(rows)} rounds from {done-bad} games ({bad} malformed) "
          f"to {out} in {time.time()-t0:.0f}s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
