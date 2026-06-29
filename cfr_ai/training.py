"""Training entry point for the JIT-compiled Trainer.

By default this saves the strategy to the production layout:
    cfr_ai/outputs/<setup>/strategy.npz       (deployment data)
    cfr_ai/outputs/<setup>/strategy.abs.json  (abstraction-string -> id)
    cfr_ai/outputs/<setup>/diagnostic.npz     (regrets, ssum, touch counters)
    cfr_ai/outputs/<setup>/metadata.csv       (human-readable training params)

Pass `--archive-tag X` to instead save into an archive snapshot at
    cfr_ai/archive/X/outputs/<setup>/...
along with copies of `history.csv` and `information_set.py` (the format
expected by `analysis/head_to_head.py`).

CLI:
    python -m cfr_ai.training --hand-sizes 3 7 --iter 5000000 \\
        --penalty 0.1 --dtype fp32
"""

import argparse
import csv
import os
import random
import shutil
import time
from datetime import datetime
from typing import Dict

import numpy as np
import psutil

import cfr_ai.game as game_mod
from cfr_ai.trainer import Trainer, _seed_numba, GROW_CHUNK


# Default pruning-range. `--pruning-range a b` overrides on the CLI
DEFAULT_PRUNING_THRESHOLD = -20
DEFAULT_MIN_REGRET = -22


def _build_exploitability_callback(args):
    """Build the snapshot callback + the dict it appends to. If
    `--get-exploitability` is off, returns (None, {}) so train() runs at full
    speed."""
    if not getattr(args, "get_exploitability", False):
        return None, {}
    if len(args.hand_sizes) != 2:
        raise SystemExit(
            "--get-exploitability is only supported for 2-player setups "
            "(the LBR oracle is 2-player). Drop the flag for 3+ players.")
    from cfr_ai.lbr import lbr_exploitability

    log: Dict[str, str] = {}
    depth = args.exploitability_depth
    sps = [0] if args.hand_sizes[0] == args.hand_sizes[1] else [0, 1]

    def callback(iter_num: int, flat_strategy) -> None:
        for sp in sps:
            r = lbr_exploitability(
                args.hand_sizes, sp, flat_strategy, depth=depth,
                n_belief_samples=args.exploitability_n_belief,
                n_lbr_hand_samples=args.exploitability_n_lbr_hand,
                seed=42, show_progress=False,
            )
            if r["se_worst"] == 0:
                cell = f"{r['expl']*100:+.4f}%"
            else:
                # ASCII "+/-", not "±", to keep metadata.csv mojibake-free.
                cell = f"{r['expl']*100:+.4f}% +/- {r['se_worst']*100:.4f}pp"
            log[f"LBR-{depth} expl sp={sp} at Iter {iter_num}"] = cell
            print(f"  iter {iter_num:,} LBR-{depth} sp={sp}: {cell}", flush=True)

    return callback, log


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


VERSION = "V3.2"  # Shows up in metadata


def save_strategies(trainer: Trainer, out_root: str, hand_sizes,
                    iter_count, penalty, training_seconds, peak_ram_mb,
                    min_bet, pruning_threshold, min_regret,
                    values, utility_log, exploitability_log):
    """Save the trained strategy as `<out_root>/outputs/<setup>/strategy.npz`
    (+ abs.json), the diagnostic data as `diagnostic.npz`, plus a
    human-readable `metadata.csv` with training parameters. `out_root` is
    either `cfr_ai/` (production saves) or `cfr_ai/archive/<tag>/`. Production
    saves additionally update the unified summary CSV; archive saves write
    only to their snapshot directory."""
    from cfr_ai.strategy_io import save_strategy, save_diagnostic
    from cfr_ai import summary as summary_mod

    setup = "_".join(str(x) for x in hand_sizes)
    setup_dir = os.path.join(out_root, "outputs", setup)
    os.makedirs(setup_dir, exist_ok=True)

    fs, meaningful_policies = trainer.get_final_flat_strategy(
        drop_check_only=True, clear_lows_threshold=0.01)
    save_strategy(fs, setup_dir, compressed=True)
    del fs # Drop to avoid RAM usage spike during save

    diag = trainer.get_diagnostic_arrays()
    save_diagnostic(
        setup_dir,
        keys=diag["keys"], lower=diag["lower"], upper=diag["upper"],
        first_touched=diag["first_touched"],
        last_touched=diag["last_touched"],
        times_touched=diag["times_touched"],
        regrets=diag["regrets"],
        strategy_sum=diag["strategy_sum"],
        compressed=True,
    )
    del diag # Drop to avoid RAM usage spike during save

    duration_hhmm = time.strftime('%H:%M', time.gmtime(training_seconds))
    md_path = os.path.join(setup_dir, "metadata.csv")
    with open(md_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "v"])
        w.writeheader()
        w.writerow({"k": "Time finished", "v": datetime.now().strftime("%Y-%m-%d, %H:%M:%S")})
        w.writerow({"k": "Training duration", "v": duration_hhmm})
        w.writerow({"k": "Iterations", "v": iter_count})
        w.writerow({"k": "Minimum bet", "v": min_bet})
        w.writerow({"k": "Pruning threshold", "v": pruning_threshold})
        w.writerow({"k": "Minimum regret", "v": min_regret})
        w.writerow({"k": "Penalty", "v": penalty})
        if getattr(trainer, "macro_kinds", None):
            w.writerow({"k": "Macro kinds", "v": ",".join(trainer.macro_kinds)})
        w.writerow({"k": "Nodes touched", "v": trainer.nodes_touched})
        w.writerow({"k": "Explored infosets", "v": trainer.n_rows})
        w.writerow({"k": "Non-checking infosets", "v": meaningful_policies})
        w.writerow({"k": "RAM taken (MB)", "v": peak_ram_mb})
        for p, val in enumerate(values):
            w.writerow({"k": f"Player {p + 1} game value", "v": val})
        w.writerow({"k": "Version code", "v": VERSION})
        if utility_log:
            w.writerow({"k": "--- Utility Log ---", "v": ""})
            for k, v in utility_log.items():
                w.writerow({"k": k, "v": v})
        if exploitability_log:
            w.writerow({"k": "--- Exploitability Log ---", "v": ""})
            for k, v in exploitability_log.items():
                w.writerow({"k": k, "v": v})

    # Update the unified summary for production saves only. Archive snapshots
    # live in their own directory tree and shouldn't touch the shared CSV.
    if os.path.normpath(out_root) == "cfr_ai":
        from datetime import datetime as _dt
        summary_mod.update_training_row(hand_sizes, {
            "Finished": _dt.now().strftime("%Y-%m-%d"),
            "Iterations": iter_count,
            "Penalty": penalty,
            "Min bet": min_bet,
            "Macro kinds": ",".join(getattr(trainer, "macro_kinds", []) or []),
            "Pruning threshold": pruning_threshold,
            "Minimum regret": min_regret,
            "Duration": duration_hhmm,
            "Nodes touched": trainer.nodes_touched,
            "Explored infosets": trainer.n_rows,
            "Non-checking infosets": meaningful_policies,
            "RAM (MB)": round(float(peak_ram_mb)),
            **{f"P{p} value": f"{val:.4f}" for p, val in enumerate(values)},
            "Version": VERSION,
        })

    return meaningful_policies, md_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand-sizes", nargs="+", type=int, required=True,
                    help="Per-player hand sizes, e.g. `3 7` (1v1) or `1 1 2` "
                         "(3-player). 2 or more values.")
    ap.add_argument("--iter", type=int, default=5_000_000)
    ap.add_argument("--penalty", type=float, default=0.0)
    ap.add_argument("--dtype", choices=["fp64", "fp32"], default="fp32")
    ap.add_argument("--capacity", type=int, default=GROW_CHUNK,
                    help="Initial row capacity. Auto-grows by GROW_CHUNK rows when full.")
    ap.add_argument("--min-bet", type=int, default=0)
    ap.add_argument("--pruning-range", nargs=2, type=int,
                    default=[DEFAULT_PRUNING_THRESHOLD, DEFAULT_MIN_REGRET],
                    help="Two ints: pruning threshold and minimum regret "
                         "(default -20 -22). The trainer skips regret updates "
                         "for branches whose regret is below `min_regret` and "
                         "occasionally re-explores them (feast). See README.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-points", type=int, default=20)
    ap.add_argument("--get-exploitability", action="store_true",
                    help="Compute LBR-K exploitability at several evenly-spaced "
                         "points during training and record them under a "
                         "`--- Exploitability Log ---` block in metadata.csv. "
                         "Off by default to keep production training fast.")
    ap.add_argument("--exploitability-points", type=int, default=5,
                    help="Number of evenly-spaced points at which to compute "
                         "exploitability (default 5 = quintiles).")
    ap.add_argument("--exploitability-depth", type=int, default=1,
                    help="LBR depth for the periodic measurements. Default 1 "
                         "(LBR-1) — cheap and a good convergence proxy.")
    ap.add_argument("--exploitability-n-belief", type=int, default=300,
                    help="Belief-sample cap for periodic LBR (default 300).")
    ap.add_argument("--exploitability-n-lbr-hand", type=int, default=500,
                    help="LBR-hand-sample cap for periodic LBR (default 500). "
                         "Matches the production LBR default to keep noise low.")
    ap.add_argument("--archive-tag", type=str, default=None,
                    help="If set, save to cfr_ai/archive/<tag>/ (and copy "
                         "history.csv + information_set.py there) instead of "
                         "the default cfr_ai/outputs/<setup>/.")
    ap.add_argument("--out-root", type=str, default=None,
                    help="Explicit save root, e.g. cfr_ai/experiments/<tag>. "
                         "Like --archive-tag (isolated outputs/, supporting "
                         "files copied, summary CSV untouched) but at an "
                         "arbitrary path. Takes precedence over --archive-tag.")
    ap.add_argument("--high-priority", action="store_true")
    ap.add_argument("--no-temporary-value", action="store_true",
                    help="Disable the same-iteration temporary_value cache so "
                         "every repeated infoset reach recomputes and re-updates "
                         "regret. Theoretically cleaner (no stale cross-reach "
                         "value reuse) but much slower and spikier on "
                         "transposition-heavy setups; experimental.")
    args = ap.parse_args()
    if len(args.hand_sizes) < 2:
        ap.error("--hand-sizes needs at least 2 values (one per player)")

    setup = "_".join(str(x) for x in args.hand_sizes)
    if args.out_root is not None:
        out_root = args.out_root
        save_label = f"out-root {out_root!r}"
    elif args.archive_tag is not None:
        out_root = os.path.join("cfr_ai", "archive", args.archive_tag)
        save_label = f"archive tag {args.archive_tag!r}"
    else:
        out_root = "cfr_ai"
        save_label = "production cfr_ai/outputs/"
    # Any non-production root is an isolated snapshot (archive or experiment):
    # it needs history.csv + information_set.py copied so head_to_head.py can
    # load it, and it must NOT touch the shared summary CSV.
    is_isolated_root = os.path.normpath(out_root) != "cfr_ai"

    if args.high_priority:
        _bump_priority()

    print(f"[training] setup={setup} iter={args.iter:,} penalty={args.penalty} "
          f"dtype={args.dtype} capacity={args.capacity:,} "
          f"temp_value={'OFF' if args.no_temporary_value else 'on'} "
          f"-> {save_label}",
          flush=True)

    proc = psutil.Process(os.getpid())
    print(f"  start RSS: {proc.memory_info().rss/1024/1024:.1f} MB", flush=True)

    regret_dtype = np.float32 if args.dtype == "fp32" else np.float64

    print("  warming JIT...", flush=True)
    _seed_numba(0)
    game_mod.rng = np.random.default_rng(0)
    warm = Trainer([1, 1], 0,
                   [DEFAULT_PRUNING_THRESHOLD, DEFAULT_MIN_REGRET], 0.0, 0,
                        initial_capacity=2000,
                        numba_seed=0, regret_dtype=regret_dtype)
    warm.train(50)
    del warm

    print(f"  starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
          flush=True)
    random.seed(args.seed)
    game_mod.rng = np.random.default_rng(args.seed)
    _seed_numba(args.seed)
    trainer = Trainer(
        args.hand_sizes, args.min_bet,
        args.pruning_range, args.penalty,
        args.log_points, initial_capacity=args.capacity,
        numba_seed=args.seed, regret_dtype=regret_dtype,
        use_temp_value=not args.no_temporary_value,
    )
    after_alloc = proc.memory_info().rss / 1024 / 1024
    print(f"  RSS after alloc: {after_alloc:.1f} MB", flush=True)

    snapshot_cb, exploitability_log = _build_exploitability_callback(args)
    snapshot_every = max(1, args.log_points // args.exploitability_points)

    t0 = time.time()
    values, utility_log = trainer.train(
        args.iter,
        snapshot_callback=snapshot_cb,
        snapshot_every_log_points=snapshot_every,
    )
    training_seconds = time.time() - t0
    peak_ram = proc.memory_info().rss / 1024 / 1024
    print(f"  training done: wall={training_seconds:.1f}s "
          f"it/s={args.iter/training_seconds:.0f} "
          f"rows={trainer.n_rows:,} nodes={trainer.nodes_touched:,} "
          f"peak_RSS={peak_ram:.1f} MB",
          flush=True)
    print("  utils: " + ", ".join(f"P{p}={v:+.4f}" for p, v in enumerate(values)),
          flush=True)

    # Isolated roots (archive snapshot or experiment variant) need supporting
    # files copied so head_to_head can load them.
    if is_isolated_root:
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
        args.pruning_range[0], args.pruning_range[1],
        values, utility_log, exploitability_log,
    )
    save_seconds = time.time() - save_start
    print(f"  save done in {save_seconds:.1f}s; "
          f"non-checking policies: {n_meaningful:,} of {trainer.n_rows:,}",
          flush=True)
    print(f"  metadata at: {md_path}", flush=True)


if __name__ == "__main__":
    main()
