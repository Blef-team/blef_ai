"""Forecast the memory cost of the flat-array trainer (trainer_numba.py)
versus the current dict-of-objects trainer (trainer.py), from existing
diagnostic file row counts.

The flat trainer pads regrets / strategy_sum to width 89 per row regardless
of how many actions are actually legal at that infoset. For infosets late
in the game tree (high `last_bet`), legal width is small (`88 - last_bet`)
so padding wastes most of the row. For infosets near the root (`last_bet`
in the sentinel slot 88), legal width is ~88 and padding wastes little.

For each setup with diagnostic files on disk, this script counts rows in
`outputs/{setup}/{hand_size}_diagnostic/{last_bet}.csv`. The row count for
file `LB.csv` is the number of infosets at that `last_bet` for that
hand_size (summed across hand_sizes within a setup).

Per-row sizes assumed
---------------------
trainer.py (current, per InformationSet object + dict entry):
  - regrets fp64 W-array      : 112 + 8 * W      bytes
  - strategy_sum fp32 W-array : 112 + 4 * W      bytes
  - possible_actions list      : 56  + 28 * W     bytes
  - 4 int + 1 float scalar     : ~136             bytes
  - InformationSet obj+__dict__: ~336             bytes
  - dict entry (string key)    : ~120             bytes
  Total = 872 + 40 * W

trainer_numba.py (flat, per row):
  - regrets fp64 (89-pad)      : 89 * 8 = 712     bytes
  - strategy_sum fp32 (89-pad) : 89 * 4 = 356     bytes
  - per-row metadata           : 16              bytes (int16x2 + int32x3 + fp64)
  - typed.Dict entry (int key) : ~28             bytes
  Total = 1112 bytes regardless of W

(These are rough — numpy array headers vary by version, list overhead
likewise, and dict load factor inflates the per-entry footprint. Treat
the numbers as order-of-magnitude, not exact.)
"""

import argparse
import csv
import os
from typing import Dict, List, Tuple


OUTPUTS_DIR = os.path.join("cfr_ai", "outputs")


# Per-row bytes models
def per_row_current(W: int) -> int:
    return 872 + 40 * W


def per_row_numba_fp64(_W: int) -> int:
    return 1112  # 89*8 + 89*4 + 16 + 28


def per_row_numba_fp32(_W: int) -> int:
    return 760  # 89*4 + 89*4 + 16 + 28  (regrets switched to fp32)


def width_for_last_bet(last_bet: int, min_bet: int) -> int:
    """Same logic as get_possible_actions in information_set.py."""
    if last_bet == 88 or last_bet < min_bet:
        return 88 - min_bet      # actions [min_bet..87]; no check
    return 88 - last_bet         # actions [last_bet+1..88]; includes check


def count_rows_in_csv(path: str) -> int:
    """Count rows minus header. Each row is one infoset."""
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        rdr = csv.reader(f)
        next(rdr, None)  # header
        for _ in rdr:
            n += 1
    return n


def setup_min_bet(setup_dir: str) -> int:
    metadata = os.path.join(setup_dir, "metadata.csv")
    if not os.path.exists(metadata):
        return 0
    with open(metadata, "r", encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip() == "Minimum bet":
                try:
                    return int(row[1].strip())
                except Exception:
                    return 0
    return 0


def analyze_setup(setup_name: str) -> Dict:
    """Returns dict with per-last_bet counts and memory totals."""
    setup_dir = os.path.join(OUTPUTS_DIR, setup_name)
    if not os.path.isdir(setup_dir):
        return None

    min_bet = setup_min_bet(setup_dir)

    # Walk subdirs ending in "_diagnostic"
    counts_per_last_bet: Dict[int, int] = {}
    found = False
    for entry in os.listdir(setup_dir):
        full = os.path.join(setup_dir, entry)
        if not entry.endswith("_diagnostic"):
            continue
        if not os.path.isdir(full):
            continue
        found = True
        for fname in os.listdir(full):
            if not fname.endswith(".csv"):
                continue
            try:
                lb = int(fname[:-4])
            except ValueError:
                continue
            n = count_rows_in_csv(os.path.join(full, fname))
            counts_per_last_bet[lb] = counts_per_last_bet.get(lb, 0) + n
    if not found:
        return None

    total_rows = sum(counts_per_last_bet.values())
    if total_rows == 0:
        return None

    # Memory totals
    bytes_current = 0
    bytes_numba_fp64 = 0
    bytes_numba_fp32 = 0
    bytes_flat_tight_fp64 = 0   # hypothetical: flat trainer with tight per-row widths
    for lb, n in counts_per_last_bet.items():
        W = width_for_last_bet(lb, min_bet)
        bytes_current += n * per_row_current(W)
        bytes_numba_fp64 += n * per_row_numba_fp64(W)
        bytes_numba_fp32 += n * per_row_numba_fp32(W)
        # Tight = current per-row math but flat layout: 8W + 4W + 16 + 28 = 12W + 44
        bytes_flat_tight_fp64 += n * (12 * W + 44)

    # Width-weighted average
    avg_W = sum(width_for_last_bet(lb, min_bet) * n for lb, n in counts_per_last_bet.items()) / total_rows

    return {
        "setup": setup_name,
        "min_bet": min_bet,
        "total_rows": total_rows,
        "avg_width": avg_W,
        "counts_per_last_bet": counts_per_last_bet,
        "current_MB": bytes_current / 1024 / 1024,
        "numba_fp64_MB": bytes_numba_fp64 / 1024 / 1024,
        "numba_fp32_MB": bytes_numba_fp32 / 1024 / 1024,
        "flat_tight_fp64_MB": bytes_flat_tight_fp64 / 1024 / 1024,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setups", nargs="*",
                    help="Specific setups to analyze (default: all)")
    ap.add_argument("--summary", action="store_true",
                    help="Print per-last_bet distribution for each setup")
    args = ap.parse_args()

    if args.setups:
        setup_names = args.setups
    else:
        setup_names = sorted(
            d for d in os.listdir(OUTPUTS_DIR)
            if os.path.isdir(os.path.join(OUTPUTS_DIR, d))
            and "_" in d
            and all(c.isdigit() or c == "_" for c in d)
        )

    results = []
    for s in setup_names:
        r = analyze_setup(s)
        if r is not None:
            results.append(r)

    if not results:
        print("No setups with diagnostic files found.")
        return

    # Header
    print(f"{'Setup':<10} {'min_bet':>3} {'rows':>10} {'avg_W':>7} "
          f"{'current_MB':>10} {'numba64_MB':>10} {'numba32_MB':>10} "
          f"{'tight64_MB':>10} {'n64/cur':>7} {'n32/cur':>7}")
    print("-" * 100)
    for r in results:
        n64 = r["numba_fp64_MB"] / r["current_MB"]
        n32 = r["numba_fp32_MB"] / r["current_MB"]
        print(f"{r['setup']:<10} {r['min_bet']:>3} {r['total_rows']:>10,} "
              f"{r['avg_width']:>7.1f} "
              f"{r['current_MB']:>10.1f} {r['numba_fp64_MB']:>10.1f} "
              f"{r['numba_fp32_MB']:>10.1f} {r['flat_tight_fp64_MB']:>10.1f} "
              f"{n64:>7.2f} {n32:>7.2f}")

    if args.summary:
        for r in results:
            print(f"\n--- {r['setup']} (min_bet={r['min_bet']}) ---")
            counts = r["counts_per_last_bet"]
            # Group by last_bet bucket
            for lb in sorted(counts):
                W = width_for_last_bet(lb, r["min_bet"])
                pct = 100 * counts[lb] / r["total_rows"]
                if pct >= 1:
                    print(f"  LB={lb:>2} W={W:>2} : {counts[lb]:>8,}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
