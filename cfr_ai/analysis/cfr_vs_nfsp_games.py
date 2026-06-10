"""Whole-game CFR-vs-NFSP simulator with compact per-round logging.

Phase 1 of the game-level diagnostic (see EXPERIMENTS_PLAN / the spec). Plays
full real-mechanic Blef games — start 1v1 card, the round loser gains a card
and opens the next round, losing at the max hand size (11) ends the game —
CFR (Morana) vs the deployed NFSP agent (Perun), and records ONE compact row
per round. CFR opens exactly half the games (cfr_seat = game_id % 2).

The deal + history vectors are sufficient to reconstruct every decision
offline, so this generator stays dumb; all analysis lives in a separate
analyzer that reads the table. JSON would be GBs; this is ~123 bytes/round.

Per-round row (numpy structured array):
    game_id u32 | round_idx u8 | cfr_seat u8 | starter_seat u8
    start_size u8 | nonstart_size u8
    deal u8[24]      deal[card] = owner seat (0/1), 255 = undealt; card=value*4+colour
    history u8[88]   history[a] = 1 if bet action_id a was made this round
    loser_seat u8 | final_bet_existed u8   (mutually derivable; both kept for clarity)

CFR strategies are read via the sparse-mmap layout (staged in place if absent)
so per-round reloads are cheap and resident RAM stays low and shared across
workers — keep --workers 6 / RAM under ~3 GB.

Run from the repo root:
    python -m cfr_ai.analysis.cfr_vs_nfsp_games --cfr current --games 10000 --workers 6
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
N_BETS = 88   # action ids 0..87 are bets; the check is the round terminal

ROW_DTYPE = np.dtype([
    ("game_id", "u4"), ("round_idx", "u1"), ("cfr_seat", "u1"),
    ("starter_seat", "u1"), ("start_size", "u1"), ("nonstart_size", "u1"),
    ("deal", "u1", (24,)), ("history", "u1", (N_BETS,)),
    ("loser_seat", "u1"), ("final_bet_existed", "u1"),
])

DEF_NFSP_MODEL = "nfsp_ai/artifacts/nfsp_inference_24_1v1.pt"
DEF_NFSP_CARD = "nfsp_ai/artifacts/card_embedding_pretrain_24.pt"
DEF_NFSP_HIST = "nfsp_ai/artifacts/history_embedding_pretrain_24.pt"


# --------------------------------------------------------------------------
# CFR version resolution + in-place sparse-mmap staging
# --------------------------------------------------------------------------

def _cfr_outputs_base(tag, folder):
    if folder:
        model_dir = folder
    elif tag in (None, "current", "."):
        model_dir = "cfr_ai"
    else:
        model_dir = os.path.join("cfr_ai", "archive", tag)
    return os.path.join(model_dir, "outputs")


def _ensure_mmap(outputs_base):
    """Stage any setup under outputs_base that has strategy.npz but no
    sparse-mmap files. Idempotent. Keeps strategy.npz as the source."""
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


# --------------------------------------------------------------------------
# Game simulation
# --------------------------------------------------------------------------

def _play_game(game_id, cfr_act, nfsp_act, round_guard=300, game_guard=120):
    """Play one full game; return a list of per-round row tuples (or None on a
    malformed game). cfr_seat = game_id % 2 (the seat that opens round 1)."""
    import shared.api.simpleschema_local_manager as gm
    cfr_seat = str(game_id % 2)
    rows = []
    game = gm.create_game(2, deck_size=DECK_SIZE, max_cards=MAX_CARDS,
                          init_card_dist=[1, 1])
    round_idx = 0
    for _ in range(game_guard):
        if game.get("status") == "Finished":
            break
        # --- round start: snapshot deal, sizes, opener ---
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
        # --- play the round to resolution ---
        resolved = False
        for _ in range(round_guard):
            cp = game["cp_nickname"]
            try:
                act = int(cfr_act(game) if cp == cfr_seat else nfsp_act(game))
                gm.play(game, act, save_dir=None)
            except Exception:
                return None
            after = {p["nickname"]: int(p["n_cards"]) for p in game["players"]}
            if after != before:
                # `act` was the check by `cp`; round resolved.
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
                history[act] = 1   # it was a bet
        if not resolved:
            return None
    return rows


# --------------------------------------------------------------------------
# Parallel workers
# --------------------------------------------------------------------------
_CFR_ACT = None
_NFSP_ACT = None


def _init_worker(cfr_base, nfsp_model, nfsp_card, nfsp_hist, nfsp_greedy, base_seed):
    import torch
    torch.set_num_threads(1)
    import cfr_ai.agent as cfr_agent
    cfr_agent.set_outputs_base(cfr_base)
    from nfsp_ai.production_agent import load_agent, determine_action as nfsp
    load_agent(model_path=nfsp_model, card_embedding_path=nfsp_card,
               history_embedding_path=nfsp_hist, greedy=nfsp_greedy)
    seed = base_seed + os.getpid()
    random.seed(seed)
    torch.manual_seed(seed)
    global _CFR_ACT, _NFSP_ACT
    _CFR_ACT, _NFSP_ACT = cfr_agent.determine_action, nfsp


def _game_task(game_id):
    return _play_game(game_id, _CFR_ACT, _NFSP_ACT)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--cfr", default="current",
                   help="CFR version: 'current' or an archive tag. Must have ALL "
                        "66 setups trained (games wander through every hand-size).")
    g.add_argument("--cfr-folder", default=None,
                   help="Explicit CFR model folder (its outputs/ is used).")
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=6,
                    help="Worker processes (default 6). With the sparse-mmap "
                         "layout each holds ~tens of MB of CFR, shared across "
                         "workers, so 6 stays well under ~3 GB.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nfsp-greedy", action="store_true",
                    help="NFSP argmax of average policy (default OFF = sample).")
    ap.add_argument("--nfsp-model", default=DEF_NFSP_MODEL)
    ap.add_argument("--nfsp-card-embed", default=DEF_NFSP_CARD)
    ap.add_argument("--nfsp-history-embed", default=DEF_NFSP_HIST)
    ap.add_argument("--out", default=None,
                    help="Output .npy. Default cfr_ai/analysis/cfr_vs_nfsp_games_<label>.npy")
    args = ap.parse_args()

    cfr_base = _cfr_outputs_base(args.cfr, args.cfr_folder)
    label = (os.path.basename(args.cfr_folder.rstrip("/\\")) if args.cfr_folder
             else args.cfr)
    if not os.path.isdir(cfr_base):
        print(f"[error] CFR outputs dir not found: {cfr_base}", file=sys.stderr)
        return 1
    for p in (args.nfsp_model, args.nfsp_card_embed, args.nfsp_history_embed):
        if not os.path.exists(p):
            print(f"[error] missing NFSP artifact: {p}", file=sys.stderr)
            return 1

    staged = _ensure_mmap(cfr_base)
    if staged:
        print(f"[cfr_vs_nfsp_games] staged {staged} setups to sparse-mmap in place",
              flush=True)

    out = args.out or os.path.join("cfr_ai", "analysis",
                                   f"cfr_vs_nfsp_games_{label}.npy")
    print(f"[cfr_vs_nfsp_games] CFR='{label}' ({cfr_base}) vs NFSP "
          f"({os.path.basename(args.nfsp_model)}, "
          f"{'greedy' if args.nfsp_greedy else 'sampled'}); {args.games} games, "
          f"{args.workers} workers -> {out}", flush=True)

    rows = []
    t0 = time.time()
    done = 0
    bad = 0

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
                  f"{done/el:.0f} games/s | {el:.0f}s", flush=True)

    if args.workers <= 1:
        import cfr_ai.agent as cfr_agent
        cfr_agent.set_outputs_base(cfr_base)
        from nfsp_ai.production_agent import load_agent, determine_action as nfsp_act
        load_agent(model_path=args.nfsp_model, card_embedding_path=args.nfsp_card_embed,
                   history_embedding_path=args.nfsp_history_embed,
                   greedy=bool(args.nfsp_greedy))
        random.seed(args.seed)
        for gid in range(args.games):
            _record(_play_game(gid, cfr_agent.determine_action, nfsp_act))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=(cfr_base, args.nfsp_model, args.nfsp_card_embed,
                                args.nfsp_history_embed, bool(args.nfsp_greedy),
                                args.seed)) as pool:
            for game_rows in pool.imap_unordered(_game_task, range(args.games), chunksize=8):
                _record(game_rows)

    _save()
    print(f"\nWrote {len(rows)} rounds from {done-bad} games "
          f"({bad} malformed) to {out} in {time.time()-t0:.0f}s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
