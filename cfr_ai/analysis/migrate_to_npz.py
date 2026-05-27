"""One-shot CSV -> NPZ converter for the strategy storage format change.

For each setup directory containing the legacy per-(hand_size, last_bet)
CSV tree, this tool:

  1. Loads the strategy from the CSVs (via `_load_flat_strategy_from_csv`).
  2. Saves it as `strategy.npz` + `strategy.abs.json` (via `strategy_io`).
  3. Optionally re-loads and verifies bit-equivalence.
  4. Optionally deletes the original CSVs (only after step 3 passes).

Targets traversed by default:
  - `cfr_ai/outputs/<setup>/` (working tree)
  - `cfr_ai/archive/<tag>/outputs/<setup>/` (every archive tag)

Diagnostic CSVs (`<setup>/<hand_size>_diagnostic/<last_bet>.csv`) are
NOT migrated by this script — they're a separate, optional layer. See
`--migrate-diagnostic` below.

Run examples:
  # Dry run, just one setup, see what would happen
  python -m cfr_ai.analysis.migrate_to_npz --setup-dir cfr_ai/outputs/1_3

  # Convert the full working tree, verify, do NOT delete CSVs yet
  python -m cfr_ai.analysis.migrate_to_npz --root cfr_ai/outputs --verify

  # Convert + verify + delete CSVs (point of no return — do this only
  # after you've committed the NPZ versions to git)
  python -m cfr_ai.analysis.migrate_to_npz \\
      --root cfr_ai/outputs --root cfr_ai/archive --verify --delete-csv
"""

import argparse
import os
import re
import shutil
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cfr_ai.lbr_numba import _load_flat_strategy_from_csv
from cfr_ai.strategy_io import save_strategy, load_strategy


SETUP_RE = re.compile(r"^(\d+)_(\d+)$")


def _find_setup_dirs(root: str) -> List[Tuple[str, List[int]]]:
    """Walk `root` and return every directory matching `<a>_<b>` that
    contains at least one strategy CSV. Returns (path, [a, b]) pairs.

    Handles both:
      cfr_ai/outputs/1_3/                 (working tree)
      cfr_ai/archive/v0_baseline/outputs/1_3/   (archive tag)
    """
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        base = os.path.basename(dirpath)
        m = SETUP_RE.match(base)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        # Must contain at least one hand_size subfolder with CSVs.
        hs_dirs = [d for d in dirnames if d.isdigit()]
        if not hs_dirs:
            continue
        has_csv = False
        for hs in hs_dirs:
            sub = os.path.join(dirpath, hs)
            if any(f.endswith(".csv") for f in os.listdir(sub)):
                has_csv = True
                break
        if has_csv:
            found.append((dirpath, sorted([a, b])))
    return found


def _bit_equivalent(a, b) -> bool:
    """Compare two FlatStrategy objects after a round-trip. Returns True
    if every key's (lower, upper, padded probs) matches exactly."""
    if a.min_bet != b.min_bet:
        return False
    if len(a.key_to_row) != len(b.key_to_row):
        return False
    for k_int, ref_row in a.key_to_row.items():
        k = np.int64(int(k_int))
        if k not in b.key_to_row:
            return False
        new_row = int(b.key_to_row[k])
        ref_row = int(ref_row)
        if int(a.lower_action[ref_row]) != int(b.lower_action[new_row]):
            return False
        if int(a.upper_action[ref_row]) != int(b.upper_action[new_row]):
            return False
        lo = int(a.lower_action[ref_row])
        hi = int(a.upper_action[ref_row])
        if not np.array_equal(
            a.strategy[ref_row, lo:hi + 1],
            b.strategy[new_row, lo:hi + 1],
        ):
            return False
    return True


def migrate_one(
    setup_dir: str,
    hand_sizes: List[int],
    *,
    verify: bool = False,
    delete_csv: bool = False,
    dry_run: bool = False,
) -> dict:
    """Convert one setup directory. Returns a result dict for reporting."""
    out: dict = {
        "setup_dir": setup_dir,
        "hand_sizes": hand_sizes,
        "skipped": False,
        "skip_reason": None,
        "csv_mb": 0.0,
        "npz_mb": 0.0,
        "load_csv_s": 0.0,
        "save_npz_s": 0.0,
        "verify_s": 0.0,
        "verified": None,
        "deleted_csv": False,
    }
    npz_path = os.path.join(setup_dir, "strategy.npz")
    if os.path.exists(npz_path):
        out["skipped"] = True
        out["skip_reason"] = "strategy.npz already present"
        return out

    # Measure CSV footprint
    csv_bytes = 0
    for hs in set(hand_sizes):
        sub = os.path.join(setup_dir, str(hs))
        if not os.path.isdir(sub):
            out["skipped"] = True
            out["skip_reason"] = f"missing {sub}"
            return out
        for f in os.listdir(sub):
            if f.endswith(".csv"):
                csv_bytes += os.path.getsize(os.path.join(sub, f))
    out["csv_mb"] = csv_bytes / 1024 / 1024

    if dry_run:
        return out

    # Load CSVs
    t0 = time.time()
    fs = _load_flat_strategy_from_csv(hand_sizes, setup_dir=setup_dir)
    out["load_csv_s"] = time.time() - t0

    # Save NPZ
    t0 = time.time()
    save_strategy(fs, setup_dir, compressed=True)
    out["save_npz_s"] = time.time() - t0
    out["npz_mb"] = (
        os.path.getsize(npz_path)
        + os.path.getsize(os.path.join(setup_dir, "strategy.abs.json"))
    ) / 1024 / 1024

    # Verify
    if verify:
        t0 = time.time()
        fs2 = load_strategy(setup_dir)
        ok = _bit_equivalent(fs, fs2)
        out["verify_s"] = time.time() - t0
        out["verified"] = ok
        if not ok:
            # Don't delete CSVs if verification fails — leave the NPZ for
            # inspection but keep the source of truth.
            out["delete_csv"] = False
            return out

    if delete_csv:
        for hs in set(hand_sizes):
            sub = os.path.join(setup_dir, str(hs))
            if os.path.isdir(sub):
                shutil.rmtree(sub)
        out["deleted_csv"] = True

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup-dir", type=str, default=None,
                    help="Single setup directory to migrate. Mutually "
                         "exclusive with --root.")
    ap.add_argument("--root", action="append", default=[],
                    help="Root directory to walk for setup folders. May be "
                         "given multiple times.")
    ap.add_argument("--verify", action="store_true",
                    help="After save, re-load and check bit equivalence.")
    ap.add_argument("--delete-csv", action="store_true",
                    help="Delete the per-hand_size CSV folders after a "
                         "successful migration. Requires --verify.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be migrated without doing it.")
    args = ap.parse_args()

    if args.delete_csv and not args.verify:
        ap.error("--delete-csv requires --verify (don't trash CSVs without "
                 "round-trip checking the NPZ first)")

    targets: List[Tuple[str, List[int]]] = []
    if args.setup_dir:
        base = os.path.basename(args.setup_dir.rstrip(os.sep))
        m = SETUP_RE.match(base)
        if not m:
            ap.error(f"--setup-dir basename {base!r} doesn't look like NxM")
        a, b = int(m.group(1)), int(m.group(2))
        targets.append((args.setup_dir, sorted([a, b])))
    elif args.root:
        for root in args.root:
            if not os.path.isdir(root):
                print(f"  WARNING: --root {root} doesn't exist; skipping",
                      flush=True)
                continue
            targets.extend(_find_setup_dirs(root))
    else:
        ap.error("either --setup-dir or --root is required")

    print(f"Found {len(targets)} setup directories to consider", flush=True)
    print(f"  verify={args.verify}, delete_csv={args.delete_csv}, "
          f"dry_run={args.dry_run}", flush=True)

    totals = {"csv_mb": 0.0, "npz_mb": 0.0, "n": 0, "skipped": 0,
              "verify_fail": 0, "deleted": 0}
    print(f"\n  {'setup':>20}  {'csv_MB':>7}  {'npz_MB':>7}  {'ratio':>5}  "
          f"{'load':>5}  {'save':>5}  {'verify':>6}  status",
          flush=True)
    for setup_dir, hand_sizes in targets:
        res = migrate_one(
            setup_dir, hand_sizes,
            verify=args.verify, delete_csv=args.delete_csv,
            dry_run=args.dry_run,
        )
        if res["skipped"]:
            totals["skipped"] += 1
            print(f"  {os.path.basename(setup_dir):>20}  -        -        -      "
                  f"-      -      -       skipped: {res['skip_reason']}",
                  flush=True)
            continue
        ratio = (res["npz_mb"] / res["csv_mb"]) if res["csv_mb"] > 0 else 0.0
        status = "OK"
        if res["verified"] is False:
            status = "VERIFY-FAIL"
            totals["verify_fail"] += 1
        elif res["deleted_csv"]:
            status = "OK+deleted"
            totals["deleted"] += 1
        print(f"  {os.path.basename(setup_dir):>20}  "
              f"{res['csv_mb']:>7.1f}  {res['npz_mb']:>7.1f}  "
              f"{ratio:>5.2f}  "
              f"{res['load_csv_s']:>5.1f}  {res['save_npz_s']:>5.1f}  "
              f"{res['verify_s']:>6.1f}  {status}",
              flush=True)
        totals["csv_mb"] += res["csv_mb"]
        totals["npz_mb"] += res["npz_mb"]
        totals["n"] += 1

    print("", flush=True)
    print(f"  totals: n={totals['n']} csv={totals['csv_mb']/1024:.2f} GB "
          f"npz={totals['npz_mb']/1024:.2f} GB "
          f"shrink={totals['csv_mb']/max(1,totals['npz_mb']):.1f}x  "
          f"skipped={totals['skipped']}  verify_fail={totals['verify_fail']}  "
          f"deleted={totals['deleted']}",
          flush=True)
    if totals["verify_fail"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
