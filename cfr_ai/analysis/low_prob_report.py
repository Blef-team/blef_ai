"""Report how often each probability magnitude appears in trained strategies,
weighted by node importance (times_touched). Helps decide where the clear_lows
threshold should sit: if a meaningful fraction of weighted policy mass lives
below 0.01 in important infosets, the current 1% floor is dropping signal."""
import argparse
import csv
import os
from typing import List, Tuple

import numpy as np

from cfr_ai.encoding import decode_probabilities


# Bin edges in probability units (exclusive of right edge).
BINS: List[Tuple[float, float, str]] = [
    (0.0004, 0.001, "[0.04%, 0.1%)"),
    (0.001, 0.002, "[0.1%, 0.2%)"),
    (0.002, 0.005, "[0.2%, 0.5%)"),
    (0.005, 0.010, "[0.5%, 1.0%)"),
    (0.010, 0.020, "[1.0%, 2.0%)"),
    (0.020, 0.050, "[2.0%, 5.0%)"),
    (0.050, 0.100, "[5.0%, 10%)"),
    (0.100, 0.250, "[10%, 25%)"),
    (0.250, 0.500, "[25%, 50%)"),
    (0.500, 1.000 + 1e-9, "[50%, 100%]"),
]


def main():
    p = argparse.ArgumentParser(description="Low-probability distribution report.")
    p.add_argument("--hand-sizes", nargs=2, type=int, required=True)
    args = p.parse_args()
    hand_sizes = sorted(args.hand_sizes)
    setup_dir = os.path.join("cfr_ai", "outputs", "_".join(str(x) for x in hand_sizes))

    # Per-bin: [count_entries, sum_times_touched, sum_prob_mass_times_touched]
    bin_count = np.zeros(len(BINS), dtype=np.int64)
    bin_weight = np.zeros(len(BINS), dtype=np.float64)
    bin_mass = np.zeros(len(BINS), dtype=np.float64)
    total_weight = 0.0
    total_mass = 0.0
    total_infosets = 0
    skipped_files = 0

    for hand_size in sorted(set(hand_sizes)):
        diag_dir = os.path.join(setup_dir, f"{hand_size}_diagnostic")
        if not os.path.isdir(diag_dir):
            print(f"Skipping {diag_dir}: not found")
            skipped_files += 1
            continue
        for fname in sorted(os.listdir(diag_dir)):
            if not fname.endswith(".csv"):
                continue
            with open(os.path.join(diag_dir, fname), "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row.get("v"):
                        continue
                    try:
                        tt = int(row.get("times_touched", 0))
                    except ValueError:
                        continue
                    if tt <= 0:
                        continue
                    probs = decode_probabilities(row["v"])
                    nonzero = probs[probs > 0]
                    if nonzero.size == 0:
                        continue
                    total_infosets += 1
                    total_weight += tt
                    # tt counts how often the infoset was used. Each visit
                    # contributes "tt * prob" expected uses of that action; we
                    # also weight per-entry by tt to value the infoset's per-
                    # action distribution by its importance.
                    for prob in nonzero:
                        # Bin the probability.
                        for i, (lo, hi, _) in enumerate(BINS):
                            if lo <= prob < hi:
                                bin_count[i] += 1
                                bin_weight[i] += tt
                                bin_mass[i] += tt * prob
                                break
                    total_mass += tt * nonzero.sum()

    print(f"\nSetup {hand_sizes}: {total_infosets} infosets scanned, "
          f"total times_touched = {total_weight:.0f}\n")
    print(f"{'bin':>14s}  {'#entries':>10s}  {'wt #entries':>14s}  "
          f"{'wt prob mass':>14s}  {'% of mass':>9s}")
    print("-" * 70)
    for i, (_, _, label) in enumerate(BINS):
        share = (bin_mass[i] / total_mass * 100) if total_mass > 0 else 0.0
        print(f"{label:>14s}  {bin_count[i]:>10d}  {bin_weight[i]:>14.0f}  "
              f"{bin_mass[i]:>14.2f}  {share:>8.3f}%")
    print()
    # Cumulative loss if we apply a clear_lows-style threshold at various levels.
    print("Cumulative weighted policy mass clipped if clear_lows threshold is set at:")
    cumulative = 0.0
    threshold_idx = -1
    for i, (lo, hi, label) in enumerate(BINS):
        cumulative_share = sum(bin_mass[:i+1]) / total_mass * 100 if total_mass > 0 else 0.0
        share_thresh = hi
        print(f"   threshold = {share_thresh*100:>5.2f}%: clipped {cumulative_share:>6.3f}% of weighted mass")


if __name__ == "__main__":
    main()
