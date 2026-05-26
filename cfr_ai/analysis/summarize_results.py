"""Print a markdown summary of the 5M-iter NumbaTrainer benchmark results
combined with reference production stats and memory forecast.

Reads:
- cfr_ai/outputs/bench_5M_results.csv   (numba results, fp32 + fp64)
- cfr_ai/outputs/{setup}/metadata.csv   (production ES/DCFR stats)
- cfr_ai/outputs/lbr_summary.csv         (production LBR-1)
"""

import csv
import os
import sys


BENCH_CSV = os.path.join("cfr_ai", "outputs", "bench_5M_results.csv")
LBR_SUMMARY = os.path.join("cfr_ai", "outputs", "lbr_summary.csv")


def parse_duration_hhmm(s: str) -> float:
    """Returns seconds, parsing '01:12' as HH:MM."""
    try:
        parts = s.strip().split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60
    except Exception:
        return 0.0


def production_stats(setup_underscored: str):
    """Return dict from metadata.csv for a production setup.

    Note: (2,2) and (1,3) metadata.csv was overwritten by the DCFR validation
    runs documented in SESSION_NOTES.md. Their ES training durations are
    hardcoded here from the SESSION_NOTES record:
        (2,2) ES: 1h 14m, (1,3) ES: 1h 18m. RAM left as the DCFR-overwritten
    value (same algorithm, same array sizes — RAM unaffected by DCFR's α-discount).
    """
    md = os.path.join("cfr_ai", "outputs", setup_underscored, "metadata.csv")
    if not os.path.exists(md):
        return None
    stats = {}
    with open(md, "r", encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                stats[row[0].strip()] = row[1].strip()
    # Override durations where DCFR overwrote ES baseline.
    if setup_underscored == "2_2":
        stats["Training duration"] = "01:14"
    elif setup_underscored == "1_3":
        stats["Training duration"] = "01:18"
    return stats


def production_lbr(setup_with_comma: str):
    if not os.path.exists(LBR_SUMMARY):
        return None
    with open(LBR_SUMMARY, "r", encoding="cp1252", errors="ignore") as f:
        rdr = csv.reader(f)
        next(rdr, None)
        for row in rdr:
            if len(row) >= 4 and row[0].strip() == setup_with_comma:
                return row[3]
    return None


def main() -> int:
    if not os.path.exists(BENCH_CSV):
        print(f"No bench results at {BENCH_CSV}")
        return 1

    rows = []
    with open(BENCH_CSV, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            rows.append(row)

    # Group by setup
    by_setup = {}
    for r in rows:
        by_setup.setdefault(r["setup"], {})[r["dtype"]] = r

    print("# NumbaTrainer 5M-iter benchmark vs production\n")
    print("Each setup trained for 5,000,000 iterations under both dtypes.")
    print("`it/s` here is the steady-state full-training rate (not the early-iter rate).\n")

    # Speed + memory table
    print("## Wall-clock and memory\n")
    print("| Setup | Prod ES wall | Prod ES it/s | Numba fp64 wall | Numba fp64 it/s | Numba fp32 wall | Numba fp32 it/s | Speedup (fp64) | Prod RAM | Numba fp64 RAM | Numba fp32 RAM |")
    print("|-------|-------------:|-------------:|----------------:|----------------:|----------------:|----------------:|---------------:|---------:|---------------:|---------------:|")
    for setup in sorted(by_setup):
        setup_us = setup.replace(",", "_")
        ps = production_stats(setup_us)
        if ps:
            prod_wall = parse_duration_hhmm(ps.get("Training duration", ""))
            prod_its = 5_000_000 / prod_wall if prod_wall else 0.0
            prod_ram = ps.get("RAM taken (MB)", "")
        else:
            prod_wall, prod_its, prod_ram = 0, 0, "?"
        d = by_setup[setup]
        fp64 = d.get("fp64", {})
        fp32 = d.get("fp32", {})
        fp64_wall = float(fp64.get("wall_s", 0)) if fp64 else 0
        fp32_wall = float(fp32.get("wall_s", 0)) if fp32 else 0
        speedup = (prod_wall / fp64_wall) if fp64_wall else 0
        print(f"| ({setup}) | {prod_wall/60:>6.1f}min | {prod_its:>7.0f} | "
              f"{fp64_wall/60:>6.1f}min | {fp64.get('it_per_s','-'):>7} | "
              f"{fp32_wall/60:>6.1f}min | {fp32.get('it_per_s','-'):>7} | "
              f"{speedup:>5.2f}x | {prod_ram} MB | {fp64.get('peak_RAM_MB','-')} MB | "
              f"{fp32.get('peak_RAM_MB','-')} MB |")

    # LBR comparison
    print("\n## LBR-1 (300, 500) — quality at fixed 5M iters\n")
    print("| Setup | Production LBR-1 | Numba fp64 LBR-1 | Numba fp32 LBR-1 |")
    print("|-------|-----------------:|-----------------:|-----------------:|")
    for setup in sorted(by_setup):
        prod_lbr = production_lbr(setup) or "?"
        d = by_setup[setup]
        fp64 = d.get("fp64", {})
        fp32 = d.get("fp32", {})
        def fmt(r, key):
            sp0 = r.get("lbr1_sp0_expl", "")
            sp1 = r.get("lbr1_sp1_expl", "")
            if sp1:
                return f"{sp0} | {sp1}"
            return sp0
        print(f"| ({setup}) | {prod_lbr} | {fmt(fp64, 'fp64')} | {fmt(fp32, 'fp32')} |")

    # FP32 vs FP64 summary
    print("\n## fp32 vs fp64 head-to-head\n")
    print("| Setup | wall (fp32/fp64) | RAM (fp32/fp64) | LBR diff (fp32 - fp64) |")
    print("|-------|-----------------:|----------------:|-----------------------:|")
    for setup in sorted(by_setup):
        d = by_setup[setup]
        if "fp32" not in d or "fp64" not in d:
            continue
        w64 = float(d["fp64"]["wall_s"])
        w32 = float(d["fp32"]["wall_s"])
        r64 = float(d["fp64"]["peak_RAM_MB"])
        r32 = float(d["fp32"]["peak_RAM_MB"])
        # parse LBR percentages
        def to_pct(s):
            s = s.strip()
            if s.endswith("%"):
                s = s[:-1]
            try:
                return float(s)
            except Exception:
                return None
        diff_str = []
        for sp in ("sp0", "sp1"):
            v64 = to_pct(d["fp64"].get(f"lbr1_{sp}_expl", ""))
            v32 = to_pct(d["fp32"].get(f"lbr1_{sp}_expl", ""))
            if v64 is not None and v32 is not None:
                diff_str.append(f"{v32-v64:+.4f}pp")
        diff_str = " | ".join(diff_str) if diff_str else "?"
        print(f"| ({setup}) | {w32:.0f}s / {w64:.0f}s ({(w32/w64-1)*100:+.1f}%) | "
              f"{r32:.0f} / {r64:.0f} MB ({(r32/r64-1)*100:+.1f}%) | "
              f"{diff_str} |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
