"""Rebuild the unified summary CSV from per-setup metadata.csv files and
regenerate the utility charts.

The unified summary (`cfr_ai/outputs/summary_of_all_runs.csv`) is maintained
incrementally by `cfr_ai/training.py` (training cols) and `cfr_ai/lbr.py`
(LBR cols), so this script is normally not needed. Use it after a migration,
or to fix a corrupted summary file.

Rebuild logic (delegated to `cfr_ai/summary.py:rebuild_from_metadata`):
  - Read every `outputs/<setup>/metadata.csv`, project to training cols.
  - Preserve LBR cols from the EXISTING summary for setups whose training
    timestamp hasn't changed; otherwise blank them (training was re-run, the
    LBR result is stale).

Chart regeneration walks the same metadata.csv files and emits
`outputs/<setup>/utility_chart_<setup>.png` from the `--- Utility Log ---`
block. Setups without a utility log are skipped.

Usage:
    python -m cfr_ai.analysis.training_analytics
    python -m cfr_ai.analysis.training_analytics --no-charts  # only rebuild CSV
    python -m cfr_ai.analysis.training_analytics --only-charts # only charts
"""

import argparse
import csv
import os
import re
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # safe headless backend
import matplotlib.pyplot as plt

from cfr_ai import summary as summary_mod


UTIL_RE = re.compile(r"P(\d) Utility at Iter (\d+)")


def _parse_utility_log(path: str) -> Dict[str, List]:
    """Extract the utility log from a metadata.csv. Returns {iter, p0, p1}.
    Falls back to latin-1 for older files that have ± in exploitability lines."""
    log = {"iter": [], "p0": [], "p1": []}
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(path, "r", newline="", encoding=encoding) as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    m = UTIL_RE.match(row[0])
                    if not m:
                        continue
                    player = int(m.group(1))
                    it = int(m.group(2))
                    try:
                        val = float(row[1])
                    except ValueError:
                        continue
                    if player == 0:
                        if it not in log["iter"]:
                            log["iter"].append(it)
                            log["p0"].append(val)
                            log["p1"].append(None)
                    else:
                        if it in log["iter"]:
                            idx = log["iter"].index(it)
                            log["p1"][idx] = val
            return log
        except UnicodeDecodeError:
            continue
    return log


def _generate_chart(setup_dir: str, setup_name: str) -> Optional[str]:
    md_path = os.path.join(setup_dir, "metadata.csv")
    if not os.path.exists(md_path):
        return None
    log = _parse_utility_log(md_path)
    if not log["iter"]:
        return None

    fig = plt.figure(figsize=(10, 6))
    plt.plot(log["iter"], log["p0"], marker="o", linestyle="-", label="Player 0 Utility")
    plt.plot(log["iter"], log["p1"], marker="o", linestyle="-", label="Player 1 Utility")
    plt.title(f"Utility During Training for Setup {setup_name}")
    plt.xlabel("Training Iteration")
    plt.ylabel("Average Utility in Chunk")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    chart_path = os.path.join(setup_dir, f"utility_chart_{setup_name}.png")
    fig.savefig(chart_path)
    plt.close(fig)
    return chart_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outputs-dir", default=os.path.join("cfr_ai", "outputs"))
    ap.add_argument("--no-charts", action="store_true",
                    help="Skip utility-chart regeneration; only rebuild CSV.")
    ap.add_argument("--only-charts", action="store_true",
                    help="Skip CSV rebuild; only regenerate utility charts.")
    args = ap.parse_args()

    if not args.only_charts:
        n = summary_mod.rebuild_from_metadata(outputs_dir=args.outputs_dir)
        print(f"Rebuilt {summary_mod.SUMMARY_PATH} with {n} rows.")

    if not args.no_charts:
        regenerated = 0
        for entry in sorted(os.scandir(args.outputs_dir), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            chart_path = _generate_chart(entry.path, entry.name)
            if chart_path:
                regenerated += 1
        print(f"Regenerated {regenerated} utility charts.")


if __name__ == "__main__":
    main()
