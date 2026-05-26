"""Run a single 5M-iteration NumbaTrainer benchmark with peak-RAM and LBR
measurement, then append a row to bench_5M_results.csv.

Designed to be invoked in parallel from multiple processes; each invocation
writes its own row at the end (locking-free append).

Usage:
    python -m cfr_ai.analysis.bench_5M --hand-sizes 1 1 --iter 5000000 \
        --dtype fp64 --capacity 10000

CSV columns:
    setup, dtype, iter, wall_s, it_per_s, peak_RAM_MB, n_rows,
    nodes_touched, lbr1_sp0, lbr1_sp1
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import psutil

import cfr_ai.game as game_mod
from cfr_ai.trainer_numba import NumbaTrainer, _seed_numba
from cfr_ai.lbr import lbr_exploitability


RESULTS_CSV = os.path.join("cfr_ai", "outputs", "bench_5M_results.csv")
HEADER = [
    "setup", "dtype", "iter", "wall_s", "it_per_s",
    "peak_RAM_MB", "n_rows", "nodes_touched",
    "lbr1_sp0_expl", "lbr1_sp0_se", "lbr1_sp1_expl", "lbr1_sp1_se",
    "lbr1_wall_s", "started_at",
]


def measure_peak_rss(proc, start_rss):
    """Sample once now and return max-so-far."""
    rss = proc.memory_info().rss / 1024 / 1024
    return max(start_rss, rss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand-sizes", nargs=2, type=int, required=True)
    ap.add_argument("--iter", type=int, default=5_000_000)
    ap.add_argument("--dtype", choices=["fp64", "fp32"], default="fp64")
    ap.add_argument("--capacity", type=int, default=200_000)
    ap.add_argument("--min-bet", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-points", type=int, default=10)
    ap.add_argument("--lbr-depth", type=int, default=1)
    ap.add_argument("--lbr-n-belief", type=int, default=300)
    ap.add_argument("--lbr-n-hand", type=int, default=500)
    ap.add_argument("--skip-lbr", action="store_true")
    args = ap.parse_args()

    setup_label = f"{args.hand_sizes[0]},{args.hand_sizes[1]}"
    regret_dtype = np.float64 if args.dtype == "fp64" else np.float32
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"[bench_5M] setup={setup_label} dtype={args.dtype} iter={args.iter:,} "
          f"capacity={args.capacity:,}", flush=True)
    proc = psutil.Process(os.getpid())
    start_rss = proc.memory_info().rss / 1024 / 1024
    print(f"  start RSS: {start_rss:.1f} MB", flush=True)

    # Warm JIT outside timing (separate small instance, same dtype).
    print("  warming JIT...", flush=True)
    _seed_numba(0)
    game_mod.rng = np.random.default_rng(0)
    warm = NumbaTrainer([1, 1], 0, [-20, -22], 0.0, 0,
                        algorithm='es', initial_capacity=2000,
                        numba_seed=0, regret_dtype=regret_dtype)
    warm.train(50)
    del warm

    # Real run
    print(f"  starting training run at {started_at}", flush=True)
    random.seed(args.seed)
    game_mod.rng = np.random.default_rng(args.seed)
    _seed_numba(args.seed)
    trainer = NumbaTrainer(
        args.hand_sizes, args.min_bet, [-20, -22], 0.0, args.log_points,
        algorithm='es', initial_capacity=args.capacity,
        numba_seed=args.seed, regret_dtype=regret_dtype,
    )
    rss_at_alloc = proc.memory_info().rss / 1024 / 1024
    print(f"  RSS after array alloc: {rss_at_alloc:.1f} MB", flush=True)

    t0 = time.time()
    u0, u1, utility_log = trainer.train(args.iter)
    wall = time.time() - t0
    peak_rss = proc.memory_info().rss / 1024 / 1024
    it_per_s = args.iter / wall

    print(f"  training done: wall={wall:.1f}s it/s={it_per_s:.0f} "
          f"rows={trainer.n_rows:,} nodes={trainer.nodes_touched:,} "
          f"peak_RSS={peak_rss:.1f} MB", flush=True)

    # LBR-1
    lbr_results = {0: (None, None), 1: (None, None)}
    lbr_wall = 0.0
    if not args.skip_lbr:
        strat = trainer.get_final_strategy_dict()
        sps = [0] if args.hand_sizes[0] == args.hand_sizes[1] else [0, 1]
        lbr0 = time.time()
        for sp in sps:
            print(f"  LBR-{args.lbr_depth} sp={sp}...", flush=True)
            r = lbr_exploitability(
                args.hand_sizes, sp, strat, args.min_bet, args.lbr_depth,
                n_belief_samples=args.lbr_n_belief,
                n_lbr_hand_samples=args.lbr_n_hand,
                seed=42,
            )
            lbr_results[sp] = (r['expl'], r['se_worst'])
            print(f"    sp={sp}: {r['expl']*100:+.3f}% \xb1 {r['se_worst']*100:.3f}pp", flush=True)
        lbr_wall = time.time() - lbr0
        del strat

    # Append CSV row (create with header if first row).
    new_file = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(HEADER)
        sp0_expl, sp0_se = lbr_results[0]
        sp1_expl, sp1_se = lbr_results[1]
        w.writerow([
            setup_label, args.dtype, args.iter,
            f"{wall:.1f}", f"{it_per_s:.0f}",
            f"{peak_rss:.1f}", trainer.n_rows, trainer.nodes_touched,
            f"{sp0_expl*100:+.4f}%" if sp0_expl is not None else "",
            f"{sp0_se*100:.4f}pp" if sp0_se is not None else "",
            f"{sp1_expl*100:+.4f}%" if sp1_expl is not None else "",
            f"{sp1_se*100:.4f}pp" if sp1_se is not None else "",
            f"{lbr_wall:.1f}",
            started_at,
        ])
    print(f"  wrote results to {RESULTS_CSV}", flush=True)


if __name__ == "__main__":
    main()
