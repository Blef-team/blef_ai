"""Forecast per-setup numba-fp32 RAM and GB-hours, calibrated from the
(3,7) production-scale measurement.

Calibration (from `cfr_ai/outputs/3_7/metadata.csv` and the matching numba
run saved as `cfr_ai/archive/v_numba_3_7_test/`):
  - Production: 4 375 MB at 3 023 119 infosets, avg legal width W = 40.5
    -> 1 448 B/infoset, matching the linear fit  prod_per_row ~ 507 + 23-W
  - Numba fp32: 2 208 MB at 3 018 581 infosets
    -> 731 B/infoset, W-independent (flat 89-wide arrays)
    Subtracting ~70 MB Python baseline + ~30 MB typed-dict overhead ~
    constant ~720 B/row + 100 MB fixed.

Speedup assumed: 5x on a clean single-tenant cloud machine (matches the
small-setup 5M benchmarks (1,1)/(1,2)/(2,2)/(1,3)). The 3.0x measured on
(3,7) was held back by five competing Python processes on the user's
laptop; a clean instance recovers the full 5x.

Inputs:
  - `cfr_ai/outputs/summary_of_all_runs.csv` (production wall + RAM)
  - `cfr_ai/outputs/<setup>/metadata.csv` (n_infosets per setup)
  - Per-setup avg_W from the diagnostic-file walk in `memory_forecast.py`
"""

import argparse
import csv
import os
import sys
from typing import Dict

# Re-use the diagnostic-file walk for avg_W per setup.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory_forecast import analyze_setup  # noqa: E402


SUMMARY_CSV = os.path.join("cfr_ai", "outputs", "summary_of_all_runs.csv")
OUTPUTS_DIR = os.path.join("cfr_ai", "outputs")

# Calibration constants (B/row)
PROD_PER_ROW_INTERCEPT = 507.0
PROD_PER_ROW_SLOPE = 23.0
NUMBA_FP32_PER_ROW = 720.0
NUMBA_FIXED_MB = 100.0    # Python + typed.Dict overhead, calibrated from (3,7)
SPEEDUP_CLEAN = 5.0       # numba/prod, single-tenant cloud
SPEEDUP_CONTENDED = 3.0   # numba/prod, observed on (3,7) with 5 competing Pythons


def parse_hhmm(s: str) -> float:
    """'10:03' -> 10.05 hours."""
    try:
        h, m = s.strip().split(":")
        return int(h) + int(m) / 60.0
    except Exception:
        return 0.0


def n_rows_for_setup(setup_underscored: str) -> int:
    md = os.path.join(OUTPUTS_DIR, setup_underscored, "metadata.csv")
    if not os.path.exists(md):
        return 0
    with open(md, "r", encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip() == "Explored infosets":
                try:
                    return int(row[1].strip())
                except Exception:
                    return 0
    return 0


def avg_W_for_setup(setup_underscored: str) -> float:
    """Average legal action width across all infosets, from the diagnostic
    files. Cached in memory_forecast.analyze_setup output."""
    r = analyze_setup(setup_underscored)
    return r["avg_width"] if r else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speedup", type=float, default=SPEEDUP_CLEAN,
                    help=f"Numba/prod wall-clock speedup. Default {SPEEDUP_CLEAN}x "
                         f"(clean single-tenant). Use {SPEEDUP_CONTENDED} for the "
                         f"contended-laptop case observed on (3,7).")
    ap.add_argument("--full", action="store_true",
                    help="Show all setups, not just summary stats")
    args = ap.parse_args()

    # Read the master summary
    rows = []
    with open(SUMMARY_CSV, "r", encoding="utf-8", errors="ignore") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)

    print(f"Calibration: prod ~ {PROD_PER_ROW_INTERCEPT:.0f} + {PROD_PER_ROW_SLOPE:.0f}-W B/row;  "
          f"numba fp32 ~ {NUMBA_FP32_PER_ROW:.0f} B/row + {NUMBA_FIXED_MB:.0f} MB baseline;  "
          f"speedup = {args.speedup}x")
    print()

    out = []
    skipped = []
    for r in rows:
        setup = r["Setup"].strip().strip('"').strip()
        setup_us = setup.replace(",", "_")
        try:
            prod_ram = float(r["RAM (MB)"].strip())
        except Exception:
            skipped.append(setup); continue
        prod_h = parse_hhmm(r["Duration"])
        if prod_h <= 0:
            skipped.append(setup); continue

        n_rows = n_rows_for_setup(setup_us)
        avg_W = avg_W_for_setup(setup_us)
        if n_rows == 0 or avg_W == 0:
            skipped.append(setup); continue

        # Predicted numba fp32 RAM (peak training)
        numba_ram = (n_rows * NUMBA_FP32_PER_ROW) / (1024 * 1024) + NUMBA_FIXED_MB
        ram_ratio = numba_ram / prod_ram

        # Predicted numba time
        numba_h = prod_h / args.speedup

        prod_gb_h = prod_ram / 1024.0 * prod_h
        numba_gb_h = numba_ram / 1024.0 * numba_h

        out.append({
            "setup": setup, "n_rows": n_rows, "avg_W": avg_W,
            "prod_ram": prod_ram, "prod_h": prod_h,
            "numba_ram": numba_ram, "numba_h": numba_h,
            "ram_ratio": ram_ratio,
            "prod_gb_h": prod_gb_h, "numba_gb_h": numba_gb_h,
            "gb_h_ratio": numba_gb_h / prod_gb_h,
        })

    out.sort(key=lambda x: x["prod_ram"])

    if args.full:
        print(f"{'Setup':<8} {'rows':>10} {'W':>5}  "
              f"{'prod_MB':>8} {'numba_MB':>8} {'ratio':>5}   "
              f"{'prod_h':>6} {'numba_h':>7}  "
              f"{'prod_GBh':>9} {'numba_GBh':>10} {'gbh_red':>7}")
        for r in out:
            print(f"{r['setup']:<8} {r['n_rows']:>10,} {r['avg_W']:>5.1f}  "
                  f"{r['prod_ram']:>8.0f} {r['numba_ram']:>8.0f} {r['ram_ratio']:>5.2f}   "
                  f"{r['prod_h']:>6.2f} {r['numba_h']:>7.2f}  "
                  f"{r['prod_gb_h']:>9.2f} {r['numba_gb_h']:>10.2f} "
                  f"{r['gb_h_ratio']:>7.2f}")

    # Aggregates
    total_prod_gb_h = sum(r["prod_gb_h"] for r in out)
    total_numba_gb_h = sum(r["numba_gb_h"] for r in out)
    total_prod_h = sum(r["prod_h"] for r in out)
    total_numba_h = sum(r["numba_h"] for r in out)

    ratios_ram = sorted(r["ram_ratio"] for r in out)
    ratios_gbh = sorted(r["gb_h_ratio"] for r in out)

    print(f"\n--- Aggregate over {len(out)} setups (skipped: {len(skipped)}) ---")
    print(f"  Total prod  CPU-hours : {total_prod_h:>8.1f} h")
    print(f"  Total numba CPU-hours : {total_numba_h:>8.1f} h  ({total_numba_h/total_prod_h*100:.1f}% of prod)")
    print(f"  Total prod  GB-hours  : {total_prod_gb_h:>8.1f} GB-h")
    print(f"  Total numba GB-hours  : {total_numba_gb_h:>8.1f} GB-h  ({total_numba_gb_h/total_prod_gb_h*100:.1f}% of prod)")
    print(f"  Reduction factor      : {total_prod_gb_h/total_numba_gb_h:>8.2f}x")
    print()
    print(f"  Per-setup RAM ratio    : min {ratios_ram[0]:.2f}, p50 {ratios_ram[len(ratios_ram)//2]:.2f}, "
          f"mean {sum(ratios_ram)/len(ratios_ram):.2f}, max {ratios_ram[-1]:.2f}")
    print(f"  Per-setup GB-h ratio   : min {ratios_gbh[0]:.3f}, p50 {ratios_gbh[len(ratios_gbh)//2]:.3f}, "
          f"mean {sum(ratios_gbh)/len(ratios_gbh):.3f}, max {ratios_gbh[-1]:.3f}")

    # Spotlight: extremes
    print("\n  Smallest RAM saving (worst numba/prod ratio):")
    for r in sorted(out, key=lambda x: -x["ram_ratio"])[:5]:
        print(f"    ({r['setup']:<5}) W={r['avg_W']:>4.1f}  prod={r['prod_ram']:>6.0f} MB  "
              f"numba={r['numba_ram']:>6.0f} MB  ratio={r['ram_ratio']:.2f}")
    print("\n  Biggest RAM saving (best numba/prod ratio):")
    for r in sorted(out, key=lambda x: x["ram_ratio"])[:5]:
        print(f"    ({r['setup']:<5}) W={r['avg_W']:>4.1f}  prod={r['prod_ram']:>6.0f} MB  "
              f"numba={r['numba_ram']:>6.0f} MB  ratio={r['ram_ratio']:.2f}")

    if skipped:
        print(f"\n  Skipped setups (missing diag or metadata): {skipped}")


if __name__ == "__main__":
    main()
