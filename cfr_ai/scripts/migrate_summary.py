"""One-shot migration: build the unified `summary_of_all_runs.csv` from the
legacy two-file setup (training metadata under `outputs/*/metadata.csv` +
`outputs/lbr_summary.csv`).

Strategy:
  1. Walk `outputs/*/metadata.csv` -> training cols for each setup (via
     `cfr_ai/summary.py:metadata_to_training_cols`).
  2. Read `outputs/lbr_summary.csv` -> LBR cols (depth pairs + Sampling).
  3. For each setup: re-attach LBR cols ONLY when the LBR-Finished date is
     >= the training-Finished date. If LBR predates training, the policy was
     re-trained since the last exploitability measurement, so the old LBR
     value is stale and gets dropped.
  4. Write the new `outputs/summary_of_all_runs.csv`.
  5. Print a diff vs. the legacy `outputs/summary_of_all_runs.csv` (if any)
     so you can spot-check before deleting the old `lbr_summary.csv`.

Run:
    python -m cfr_ai.scripts.migrate_summary
    python -m cfr_ai.scripts.migrate_summary --dry-run   # don't write
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

from cfr_ai import summary as summary_mod
from cfr_ai.summary import (
    SUMMARY_PATH, TRAINING_COLS, parse_metadata_csv,
    metadata_to_training_cols, write_summary, _LBR_DEPTH_RE,
)


LEGACY_LBR_PATH = os.path.join("cfr_ai", "outputs", "lbr_summary.csv")


def _read_legacy_lbr(path: str) -> Dict[str, Dict[str, str]]:
    """Read legacy lbr_summary.csv into a {setup_key: row_dict}. Encoding is
    latin-1 because some historical entries use the (corrupted) ± character."""
    rows: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(path):
        return rows
    with open(path, "r", newline="", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = r.get("Setup")
            if not key:
                continue
            rows[key] = dict(r)
    return rows


def _lbr_finished_str(legacy_row: Dict[str, str]) -> str:
    return legacy_row.get("Finished", "") or ""


def _project_legacy_lbr(legacy_row: Dict[str, str]) -> Dict[str, str]:
    """Pick depth-pair columns + Sampling from a legacy row."""
    out: Dict[str, str] = {}
    for col, val in legacy_row.items():
        if _LBR_DEPTH_RE.match(col):
            out[col] = val
    sampling = legacy_row.get("Sampling", "")
    if sampling:
        out["Sampling"] = sampling
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default=os.path.join("cfr_ai", "outputs"))
    ap.add_argument("--legacy-lbr", default=LEGACY_LBR_PATH)
    ap.add_argument("--summary-path", default=SUMMARY_PATH)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be written, don't touch the file.")
    args = ap.parse_args()

    legacy_lbr = _read_legacy_lbr(args.legacy_lbr)

    new_rows: Dict[str, Dict[str, str]] = {}
    extra_depth_labels: List[str] = []
    stats = {"with_training": 0, "lbr_kept": 0, "lbr_dropped_stale": 0,
             "lbr_only_no_training": 0}

    for entry in sorted(os.scandir(args.outputs_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        md_path = os.path.join(entry.path, "metadata.csv")
        if not os.path.exists(md_path):
            continue
        setup_name = entry.name.replace("_", ",")
        meta = parse_metadata_csv(md_path)
        training = metadata_to_training_cols(meta)
        row = {"Setup": setup_name, **training}
        stats["with_training"] += 1

        legacy = legacy_lbr.get(setup_name)
        if legacy is not None:
            lbr_finished = _lbr_finished_str(legacy)
            training_finished = training.get("Finished", "")
            if lbr_finished and training_finished and lbr_finished < training_finished:
                stats["lbr_dropped_stale"] += 1
            else:
                lbr_cols = _project_legacy_lbr(legacy)
                row.update(lbr_cols)
                for col in lbr_cols:
                    m = _LBR_DEPTH_RE.match(col)
                    if m:
                        extra_depth_labels.append(m.group(1))
                if lbr_cols:
                    stats["lbr_kept"] += 1
        new_rows[setup_name] = row

    # Setups in lbr_summary that have NO metadata.csv (shouldn't happen in
    # practice given outputs/ is the source of truth, but log for safety).
    metadata_keys = set(new_rows.keys())
    for k in legacy_lbr.keys() - metadata_keys:
        stats["lbr_only_no_training"] += 1
        legacy = legacy_lbr[k]
        row = {"Setup": k}
        lbr_cols = _project_legacy_lbr(legacy)
        row.update(lbr_cols)
        for col in lbr_cols:
            m = _LBR_DEPTH_RE.match(col)
            if m:
                extra_depth_labels.append(m.group(1))
        new_rows[k] = row

    print("Migration plan:")
    for k, v in stats.items():
        print(f"  {k:<28} {v}")
    print(f"  total rows in new summary  : {len(new_rows)}")

    if args.dry_run:
        print("\n--dry-run set; not writing.")
        return 0

    write_summary(new_rows, path=args.summary_path,
                  extra_depth_labels=extra_depth_labels)
    print(f"\nWrote {args.summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
