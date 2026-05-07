"""
Backfill eval-ladder results across historical NFSP checkpoints.

Given a glob of .pt files (e.g. `nfsp_blef_*_*M.pt` saved at million-step
boundaries), runs `tools.eval_ladder` for each checkpoint and writes one
`eval_results/<checkpoint-stem>/ladder.csv` per checkpoint. Resumable: skips
checkpoints whose ladder.csv already exists.

Run from the repository root:

  python -m tools.backfill_eval \\
    --pattern 'nfsp_blef_*_*M.pt' \\
    --opponents random,conservative \\
    --n-games 200 \\
    --deck-size 24 --n-players 2 --max-cards 11

Inference-only exports (`*-inference.pt`) are skipped by default.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import List, Optional

from tools.eval_ladder import (
    build_opponents,
    load_learner,
    run_ladder,
    write_results_csv,
)

DEFAULT_OUTPUT_ROOT = "eval_results"


def _checkpoint_stem(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".pt"):
        base = base[:-3]
    return base


def _iter_checkpoints(pattern: str, exclude_inference: bool = True) -> List[str]:
    paths = sorted(glob.glob(pattern))
    if exclude_inference:
        paths = [p for p in paths if "-inference" not in os.path.basename(p)]
    return paths


def _ladder_path_for(ckpt: str, output_root: str = DEFAULT_OUTPUT_ROOT) -> str:
    stem = _checkpoint_stem(ckpt)
    return os.path.join(output_root, stem, "ladder.csv")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pattern", required=True, help="Glob for checkpoint .pt files (e.g. 'nfsp_blef_*_*M.pt').")
    p.add_argument(
        "--opponents",
        default="random,conservative",
        help="Comma-separated subset of {random, conservative, cfr, snapshot}.",
    )
    p.add_argument("--snapshot-paths", default="", help="Comma-separated checkpoint paths if opponents includes 'snapshot'.")
    p.add_argument("--n-games", type=int, default=200)
    p.add_argument("--deck-size", type=int, default=24, choices=[24, 32])
    p.add_argument("--n-players", type=int, default=2)
    p.add_argument("--max-cards", type=int, default=11)
    p.add_argument("--jokers", type=int, default=0)
    p.add_argument("--blanks", type=int, default=0)
    p.add_argument("--common-cards", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--force", action="store_true", help="Re-run even if ladder.csv already exists.")
    p.add_argument("--include-inference", action="store_true", help="Include *-inference.pt exports.")
    args = p.parse_args(argv)

    checkpoints = _iter_checkpoints(args.pattern, exclude_inference=not args.include_inference)
    if not checkpoints:
        print(f"No checkpoints matched pattern {args.pattern!r}", file=sys.stderr)
        return 2

    snap_paths = [p.strip() for p in args.snapshot_paths.split(",") if p.strip()]

    n_done = n_skipped = n_failed = 0
    t0 = time.time()
    for ckpt in checkpoints:
        ladder_path = _ladder_path_for(ckpt, args.output_root)
        if os.path.exists(ladder_path) and not args.force:
            n_skipped += 1
            print(f"[skip] {ckpt}: ladder.csv already exists", file=sys.stderr)
            continue
        try:
            learner = load_learner(ckpt)
            opponents = build_opponents(
                args.opponents, learner.obs_dim, learner.act_dim,
                snapshot_paths=snap_paths,
            )
            if not opponents:
                print(f"[fail] {ckpt}: no opponents could be constructed", file=sys.stderr)
                n_failed += 1
                continue
            results = run_ladder(
                learner, opponents,
                n_games=args.n_games,
                deck_size=args.deck_size,
                n_players=args.n_players,
                max_cards=args.max_cards,
                jokers=args.jokers,
                blanks=args.blanks,
                common_cards=args.common_cards,
                seed=args.seed,
            )
            write_results_csv(ladder_path, results)
            n_done += 1
            print(f"[done] {ckpt} -> {ladder_path}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            print(f"[fail] {ckpt}: {exc!r}", file=sys.stderr)

    elapsed = time.time() - t0
    print(
        f"Backfill: {n_done} processed, {n_skipped} skipped, {n_failed} failed in {elapsed:.1f}s",
        file=sys.stderr,
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
