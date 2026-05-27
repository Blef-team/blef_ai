"""Production training entry point using the JIT-compiled NumbaTrainer.

This is the recommended path for training a setup. The pure-Python
`training.py` remains as a slower, debuggable reference implementation
(it also supports a few extras the JIT trainer does not, like the
mid-training exploitability snapshots and the DCFR/CFR+ algorithm
options). See `cfr_ai/README.md` for the relative trade-offs.

By default this saves the strategy to the production layout:
    cfr_ai/outputs/<setup>/<hand_size>/<last_bet>.csv
    cfr_ai/outputs/<setup>/metadata.csv

Pass `--archive-tag X` to instead save into an archive snapshot at
    cfr_ai/archive/X/outputs/<setup>/...
along with copies of `history.csv` and `information_set.py` (the format
expected by `analysis/head_to_head.py`).

CLI:
    python -m cfr_ai.training_numba --hand-sizes 3 7 --iter 5000000 \\
        --penalty 0.1 --dtype fp32
"""

import argparse
import csv
import os
import random
import shutil
import time
from datetime import datetime

import numpy as np
import psutil

import cfr_ai.game as game_mod
from cfr_ai.trainer_numba import NumbaTrainer, _seed_numba, GROW_CHUNK


def _bump_priority():
    try:
        proc = psutil.Process(os.getpid())
        if hasattr(psutil, "HIGH_PRIORITY_CLASS"):
            proc.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            proc.nice(-5)
        print(f"  bumped priority to high (PID {os.getpid()})", flush=True)
    except Exception as e:
        print(f"  priority bump failed: {e}", flush=True)


def save_strategies(trainer: NumbaTrainer, out_root: str, hand_sizes,
                    iter_count, penalty, training_seconds, peak_ram_mb,
                    min_bet):
    """Save the trained strategy as `<out_root>/outputs/<setup>/strategy.npz`
    (+ sidecar JSON) plus a human-readable `metadata.csv` with training
    parameters. `out_root` is either `cfr_ai/` (production saves) or
    `cfr_ai/archive/<tag>/` (snapshot saves)."""
    from cfr_ai.strategy_io import save_strategy

    setup = "_".join(str(x) for x in hand_sizes)
    setup_dir = os.path.join(out_root, "outputs", setup)
    os.makedirs(setup_dir, exist_ok=True)

    fs, meaningful_policies = trainer.get_final_flat_strategy(
        drop_check_only=True, clear_lows_threshold=0.01)
    save_strategy(fs, setup_dir, compressed=True)

    duration_hhmm = time.strftime('%H:%M', time.gmtime(training_seconds))
    md_path = os.path.join(setup_dir, "metadata.csv")
    with open(md_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "v"])
        w.writeheader()
        w.writerow({"k": "Time finished", "v": datetime.now().strftime("%Y-%m-%d, %H:%M:%S")})
        w.writerow({"k": "Training duration", "v": duration_hhmm})
        w.writerow({"k": "Iterations", "v": iter_count})
        w.writerow({"k": "Minimum bet", "v": min_bet})
        w.writerow({"k": "Penalty", "v": penalty})
        w.writerow({"k": "Nodes touched", "v": trainer.nodes_touched})
        w.writerow({"k": "Explored infosets", "v": trainer.n_rows})
        w.writerow({"k": "Non-checking infosets", "v": meaningful_policies})
        w.writerow({"k": "RAM taken (MB)", "v": peak_ram_mb})
        w.writerow({"k": "Version code", "v": "NUMBA-FP32-ES"})

    return meaningful_policies, md_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand-sizes", nargs=2, type=int, required=True)
    ap.add_argument("--iter", type=int, default=5_000_000)
    ap.add_argument("--penalty", type=float, default=0.0)
    ap.add_argument("--dtype", choices=["fp64", "fp32"], default="fp32")
    ap.add_argument("--capacity", type=int, default=GROW_CHUNK,
                    help="Initial row capacity. Auto-grows by GROW_CHUNK rows when full.")
    ap.add_argument("--min-bet", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-points", type=int, default=20)
    ap.add_argument("--archive-tag", type=str, default=None,
                    help="If set, save to cfr_ai/archive/<tag>/ (and copy "
                         "history.csv + information_set.py there) instead of "
                         "the default cfr_ai/outputs/<setup>/.")
    ap.add_argument("--high-priority", action="store_true")
    args = ap.parse_args()

    setup = "_".join(str(x) for x in args.hand_sizes)
    if args.archive_tag is not None:
        out_root = os.path.join("cfr_ai", "archive", args.archive_tag)
        save_label = f"archive tag {args.archive_tag!r}"
    else:
        out_root = "cfr_ai"
        save_label = "production cfr_ai/outputs/"

    if args.high_priority:
        _bump_priority()

    print(f"[training_numba] setup={setup} iter={args.iter:,} penalty={args.penalty} "
          f"dtype={args.dtype} capacity={args.capacity:,} -> {save_label}",
          flush=True)

    proc = psutil.Process(os.getpid())
    print(f"  start RSS: {proc.memory_info().rss/1024/1024:.1f} MB", flush=True)

    regret_dtype = np.float32 if args.dtype == "fp32" else np.float64

    print("  warming JIT...", flush=True)
    _seed_numba(0)
    game_mod.rng = np.random.default_rng(0)
    warm = NumbaTrainer([1, 1], 0, [-20, -22], 0.0, 0,
                        algorithm='es', initial_capacity=2000,
                        numba_seed=0, regret_dtype=regret_dtype)
    warm.train(50)
    del warm

    print(f"  starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
          flush=True)
    random.seed(args.seed)
    game_mod.rng = np.random.default_rng(args.seed)
    _seed_numba(args.seed)
    trainer = NumbaTrainer(
        args.hand_sizes, args.min_bet, [-20, -22], args.penalty,
        args.log_points, algorithm='es', initial_capacity=args.capacity,
        numba_seed=args.seed, regret_dtype=regret_dtype,
    )
    after_alloc = proc.memory_info().rss / 1024 / 1024
    print(f"  RSS after alloc: {after_alloc:.1f} MB", flush=True)

    t0 = time.time()
    u0, u1, _ = trainer.train(args.iter)
    training_seconds = time.time() - t0
    peak_ram = proc.memory_info().rss / 1024 / 1024
    print(f"  training done: wall={training_seconds:.1f}s "
          f"it/s={args.iter/training_seconds:.0f} "
          f"rows={trainer.n_rows:,} nodes={trainer.nodes_touched:,} "
          f"peak_RSS={peak_ram:.1f} MB",
          flush=True)
    print(f"  utils: P0={u0:+.4f}, P1={u1:+.4f}", flush=True)

    # If archiving, copy supporting files so head_to_head can load the snapshot.
    if args.archive_tag is not None:
        os.makedirs(out_root, exist_ok=True)
        for src, name in [(os.path.join("cfr_ai", "history.csv"), "history.csv"),
                          (os.path.join("cfr_ai", "information_set.py"), "information_set.py")]:
            dst = os.path.join(out_root, name)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"  copied {src} -> {dst}", flush=True)

    print(f"  saving strategies under {out_root}/outputs/{setup}/ ...", flush=True)
    save_start = time.time()
    n_meaningful, md_path = save_strategies(
        trainer, out_root, args.hand_sizes,
        args.iter, args.penalty, training_seconds, peak_ram, args.min_bet,
    )
    save_seconds = time.time() - save_start
    print(f"  save done in {save_seconds:.1f}s; "
          f"non-checking policies: {n_meaningful:,} of {trainer.n_rows:,}",
          flush=True)
    print(f"  metadata at: {md_path}", flush=True)


if __name__ == "__main__":
    main()
