"""Paired-seed comparison between two NFSP checkpoint variants.

Reads a multi-seed CSV (columns: variant, seed, mc, winrate, [...]) where
EACH variant has results at the SAME set of seeds. Pairs observations by
seed and reports:
  - per-cell paired difference (variant_a - variant_b) winrate
  - paired t-statistic and 95% CI on the mean difference
  - sign of the effect

Variance reduction vs unpaired comparison is typically 2-3× when the
random env state dominates (same hands → outcome difference reflects
policy difference, not deal luck).
"""
from __future__ import annotations
import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Multi-seed CSV with columns variant,seed,mc,winrate,...")
    p.add_argument("--variant-a", required=True)
    p.add_argument("--variant-b", required=True)
    p.add_argument("--cell-key", default="mc",
                   help="Column name to use as cell key (default 'mc'). "
                        "Per-cell paired stats are computed.")
    args = p.parse_args(argv)

    rows_a = defaultdict(dict)  # cell -> {seed: winrate}
    rows_b = defaultdict(dict)
    with open(args.input) as f:
        for row in csv.DictReader(f):
            v = row["variant"]
            s = row["seed"]
            cell = row.get(args.cell_key, "")
            wr = float(row["winrate"])
            if v == args.variant_a:
                rows_a[cell][s] = wr
            elif v == args.variant_b:
                rows_b[cell][s] = wr

    if not rows_a or not rows_b:
        print(f"ERROR: missing variants in CSV", file=sys.stderr)
        return 2

    print(f"Paired-seed comparison: {args.variant_a} vs {args.variant_b}")
    print(f"  cell column: {args.cell_key}")
    print()
    header = f"{'cell':>8} {'n_pairs':>7} {'mean_a':>7} {'mean_b':>7} {'Δ':>7} {'Δ_sd':>7} {'paired-t':>9} {'95% CI':>20}"
    print(header)
    print("-" * len(header))

    overall_diffs = []
    for cell in sorted(set(rows_a) | set(rows_b)):
        sa = rows_a.get(cell, {})
        sb = rows_b.get(cell, {})
        common_seeds = sorted(set(sa) & set(sb))
        if len(common_seeds) < 2:
            continue
        diffs = [sa[s] - sb[s] for s in common_seeds]
        wa = [sa[s] for s in common_seeds]
        wb = [sb[s] for s in common_seeds]
        mu_a = statistics.mean(wa)
        mu_b = statistics.mean(wb)
        d = statistics.mean(diffs)
        sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        sem = sd / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
        # 95% CI assumes normality
        t_crit = 2.776 if len(diffs) <= 5 else 2.228 if len(diffs) <= 10 else 2.0
        ci_lo = d - t_crit * sem
        ci_hi = d + t_crit * sem
        t_stat = d / sem if sem > 1e-12 else float("nan")
        overall_diffs.extend(diffs)
        ci_str = f"[{ci_lo:+.4f},{ci_hi:+.4f}]"
        print(f"{str(cell):>8} {len(diffs):>7} {mu_a:>7.4f} {mu_b:>7.4f} {d:+7.4f} {sd:>7.4f} {t_stat:>9.3f} {ci_str:>20}")

    if overall_diffs:
        d_all = statistics.mean(overall_diffs)
        sd_all = statistics.stdev(overall_diffs) if len(overall_diffs) > 1 else 0.0
        sem_all = sd_all / math.sqrt(len(overall_diffs)) if len(overall_diffs) > 1 else 0.0
        t = d_all / sem_all if sem_all > 1e-12 else float("nan")
        print()
        print(f"OVERALL across all cells: Δ={d_all:+.4f} ± {sem_all:.4f} (sem)  paired-t={t:.3f}  n={len(overall_diffs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
