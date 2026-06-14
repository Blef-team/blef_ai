#!/usr/bin/env python3
"""Poll a training run's latest checkpoint and run a team-aware eval each
time it changes. Emit one line per eval — feeds the Monitor.

Usage:
  team_curve_logger.py LABEL RUN_DIR DECK_SIZE [GAMES] [SEEDS]
"""
from __future__ import annotations

import os
import sys
import time

# Make repo importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.team_eval import run_team_eval


def _ckpt_step_from_metrics(run_dir: str) -> int:
    p = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(p):
        return 0
    try:
        with open(p) as f:
            last = ""
            for line in f:
                if line.strip():
                    last = line
        return int(float(last.split(",")[0]))
    except Exception:
        return 0


def main():
    if len(sys.argv) < 4:
        print("usage: team_curve_logger.py LABEL RUN_DIR DECK [GAMES] [SEEDS] [PID]", file=sys.stderr)
        sys.exit(2)
    label = sys.argv[1]
    run = sys.argv[2]
    deck = int(sys.argv[3])
    games = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    seeds = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    pid = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    poll = 120  # seconds; checkpoint saves usually less frequent than this

    ckpt = os.path.join(run, "checkpoints", "nfsp_blef.pt")
    card = os.path.join(_ROOT, "artifacts", f"card_embedding_pretrain_{deck}.pt")
    hist = os.path.join(_ROOT, "artifacts", f"history_embedding_pretrain_{deck}.pt")

    print(f"[{label}] team-curve armed; ckpt={ckpt} games={games} seeds={seeds} poll={poll}s")
    sys.stdout.flush()

    last_mtime = 0.0
    last_eval_step = -1

    def _alive() -> bool:
        if pid <= 0:
            return True  # no pid given => skip liveness check
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    while True:
        if not _alive():
            print(f"[{label}] CRASH: pid {pid} exited.")
            sys.stdout.flush()
            return
        if not os.path.exists(ckpt):
            time.sleep(poll)
            continue
        try:
            mtime = os.path.getmtime(ckpt)
        except OSError:
            time.sleep(poll)
            continue
        if mtime <= last_mtime:
            time.sleep(poll)
            continue
        last_mtime = mtime

        step = _ckpt_step_from_metrics(run)
        if step <= last_eval_step:
            time.sleep(poll)
            continue
        last_eval_step = step

        # Average over `seeds` runs
        twrs = []
        trws = []
        iwrs = []
        try:
            for seed in range(seeds):
                r = run_team_eval(
                    ckpt,
                    deck_size=deck,
                    n_players=4,
                    n_teams=2,
                    max_cards=11,
                    games=games,
                    seed=seed,
                    card_embedding_path=card,
                    history_embedding_path=hist,
                )
                twrs.append(r["team_winrate"])
                trws.append(r["team_avg_reward"])
                iwrs.append(r["indiv_winrate"])
        except Exception as e:
            print(f"[{label}] eval failed at step={step}: {e!r}")
            sys.stdout.flush()
            time.sleep(poll)
            continue
        twr = sum(twrs) / len(twrs)
        trw = sum(trws) / len(trws)
        iwr = sum(iwrs) / len(iwrs)
        std = (sum((x - twr) ** 2 for x in twrs) / len(twrs)) ** 0.5
        stderr = std / (len(twrs) ** 0.5)
        print(
            f"[{label}] team-eval step={step:>9d} "
            f"team_wr={twr:.4f}±{stderr:.4f} "
            f"team_rew={trw:+.4f} "
            f"indiv_wr={iwr:.4f}"
        )
        sys.stdout.flush()
        time.sleep(poll)


if __name__ == "__main__":
    main()
