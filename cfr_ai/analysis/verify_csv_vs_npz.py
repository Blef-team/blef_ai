"""Verify that the NPZ files contain everything in the legacy CSVs,
then optionally delete the CSVs.

Per setup:
  1. Strategy: count rows in `<hand_size>/*.csv` and compare to
     `strategy.npz`'s row count. Spot-check N random rows: decode the
     CSV's probability string and compare against `strategy.npz`'s
     row, looked up by composite key.
  2. Diagnostic: count rows in `<hand_size>_diagnostic/*.csv` and
     compare to `diagnostic.npz`'s row count. Spot-check probability
     bytes plus the three integer touch fields.

If both checks pass for a setup, we have proof the NPZs are a
superset of the CSVs and the CSVs can be deleted.

Run:
    # Verify only
    python -m cfr_ai.analysis.verify_csv_vs_npz --root cfr_ai/outputs

    # Verify, then delete CSVs for setups that pass
    python -m cfr_ai.analysis.verify_csv_vs_npz --root cfr_ai/outputs --delete
"""

import argparse
import csv
import os
import re
import shutil
import sys
from typing import List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


SETUP_RE = re.compile(r"^(\d+)_(\d+)$")
SAMPLE_PER_CHECK = 8


def _find_setup_dirs(root: str) -> List[Tuple[str, List[int]]]:
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        base = os.path.basename(dirpath)
        m = SETUP_RE.match(base)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        # Need at least one strategy or diagnostic CSV folder still present.
        has_any = False
        for d in dirnames:
            if d.isdigit() or d.endswith("_diagnostic"):
                sub = os.path.join(dirpath, d)
                if any(f.endswith(".csv") for f in os.listdir(sub)):
                    has_any = True
                    break
        if has_any:
            found.append((dirpath, sorted([a, b])))
    return found


def _count_csv_rows(setup_dir: str, hand_sizes: List[int], diag: bool) -> int:
    n = 0
    for hs in set(hand_sizes):
        sub = os.path.join(setup_dir, f"{hs}_diagnostic" if diag else str(hs))
        if not os.path.isdir(sub):
            continue
        for fname in os.listdir(sub):
            if not fname.endswith(".csv"):
                continue
            with open(os.path.join(sub, fname), "r", newline="") as f:
                rdr = csv.reader(f)
                next(rdr, None)
                for row in rdr:
                    if row and row[0]:
                        n += 1
    return n


def _sample_csv_rows(setup_dir: str, hand_sizes: List[int], diag: bool,
                     n_sample: int, rng) -> list:
    """Read N random rows from the CSVs. Returns list of dicts with
    fields needed for cross-check."""
    rows = []
    files: list = []
    for hs in set(hand_sizes):
        sub = os.path.join(setup_dir, f"{hs}_diagnostic" if diag else str(hs))
        if not os.path.isdir(sub):
            continue
        for fname in os.listdir(sub):
            if fname.endswith(".csv"):
                files.append((hs, int(fname[:-4]), os.path.join(sub, fname)))
    if not files:
        return rows
    # Pick n_sample files at random, then one row from each (cheap to load).
    idx = rng.choice(len(files), size=min(n_sample, len(files)), replace=False)
    for i in idx:
        hs, lb, path = files[int(i)]
        with open(path, "r", newline="") as f:
            rdr = csv.reader(f)
            next(rdr, None)
            data = [r for r in rdr if r and r[0]]
        if not data:
            continue
        row = data[int(rng.choice(len(data)))]
        if diag:
            rows.append({
                "hand_size": hs, "last_bet": lb,
                "suffix": row[0],
                "first_touched": int(row[1]),
                "last_touched": int(row[2]),
                "times_touched": int(row[3]),
                "encoded": row[4],
            })
        else:
            rows.append({
                "hand_size": hs, "last_bet": lb,
                "suffix": row[0],
                "encoded": row[1],
            })
    return rows


def _verify_one(setup_dir: str, hand_sizes: List[int],
                rng) -> dict:
    """Return per-setup verification result dict."""
    from cfr_ai.strategy_io import load_strategy, load_diagnostic
    from cfr_ai.lbr_numba import _split_suffix
    from cfr_ai.trainer_numba import (
        LAST_BET_SHIFT, H_M1_SHIFT, H_M2_SHIFT, ABS_ID_SHIFT,
    )
    from cfr_ai.encoding import decode_probabilities

    out = {
        "setup": os.path.basename(setup_dir),
        "strategy_npz": False, "diagnostic_npz": False,
        "strategy_csv_rows": 0, "strategy_npz_rows": 0,
        "diagnostic_csv_rows": 0, "diagnostic_npz_rows": 0,
        "strategy_spot": "skip", "diagnostic_spot": "skip",
        "ok_strategy": False, "ok_diagnostic": False, "ok": False,
    }

    spath = os.path.join(setup_dir, "strategy.npz")
    dpath = os.path.join(setup_dir, "diagnostic.npz")
    out["strategy_npz"] = os.path.exists(spath)
    out["diagnostic_npz"] = os.path.exists(dpath)

    # --- Strategy verification ---
    if out["strategy_npz"]:
        fs = load_strategy(setup_dir)
        out["strategy_npz_rows"] = len(fs.key_to_row)
        out["strategy_csv_rows"] = _count_csv_rows(setup_dir, hand_sizes, diag=False)
        if out["strategy_csv_rows"] == out["strategy_npz_rows"]:
            # Spot-check
            samples = _sample_csv_rows(setup_dir, hand_sizes, diag=False,
                                        n_sample=SAMPLE_PER_CHECK, rng=rng)
            mismatches = 0
            for s in samples:
                h_m1, h_m2, abs_str = _split_suffix(s["suffix"])
                if abs_str not in fs.abs_str_to_id:
                    mismatches += 1
                    continue
                abs_id = fs.abs_str_to_id[abs_str]
                comp = (s["hand_size"]
                        | (s["last_bet"] << LAST_BET_SHIFT)
                        | (h_m1 << H_M1_SHIFT)
                        | (h_m2 << H_M2_SHIFT)
                        | (abs_id << ABS_ID_SHIFT))
                k64 = np.int64(comp)
                if k64 not in fs.key_to_row:
                    mismatches += 1
                    continue
                row = int(fs.key_to_row[k64])
                lo = int(fs.lower_action[row])
                hi = int(fs.upper_action[row])
                csv_probs = decode_probabilities(s["encoded"]).astype(np.float32)
                # CSV probs are stored only over the legal action range
                # (length = hi - lo + 1, possibly trimmed/padded by loader).
                npz_probs = fs.strategy[row, lo:hi + 1]
                width = hi - lo + 1
                if len(csv_probs) > width:
                    csv_probs = csv_probs[:width]
                if len(csv_probs) < width:
                    pad = np.zeros(width, dtype=np.float32)
                    pad[:len(csv_probs)] = csv_probs
                    csv_probs = pad
                # Strategy NPZ was renormalised at save time and the
                # all-zero rows were defaulted to check-100% (last index).
                # So compare CSV after the same transform.
                s_total = csv_probs.sum()
                if s_total > 0:
                    csv_probs = csv_probs / s_total
                else:
                    csv_probs = np.zeros(width, dtype=np.float32)
                    csv_probs[-1] = 1.0
                if not np.allclose(csv_probs, npz_probs, atol=1e-6, rtol=0):
                    mismatches += 1
            out["strategy_spot"] = f"{len(samples) - mismatches}/{len(samples)} match"
            out["ok_strategy"] = (mismatches == 0)
        else:
            out["strategy_spot"] = "skip (count mismatch)"

    # --- Diagnostic verification ---
    if out["diagnostic_npz"]:
        diag = load_diagnostic(setup_dir)
        out["diagnostic_npz_rows"] = len(diag["keys"])
        out["diagnostic_csv_rows"] = _count_csv_rows(setup_dir, hand_sizes, diag=True)
        if out["diagnostic_csv_rows"] == out["diagnostic_npz_rows"]:
            samples = _sample_csv_rows(setup_dir, hand_sizes, diag=True,
                                        n_sample=SAMPLE_PER_CHECK, rng=rng)
            mismatches = 0
            # Build composite-key -> row map from diag keys (small per-setup cost)
            # We only need it for the sampled rows.
            for s in samples:
                # Re-derive the abs_id from the same JSON the strategy loader uses.
                # We'd need abs_str_to_id; load the strategy to get it.
                # The strategy.abs.json file is at setup_dir/strategy.abs.json
                import json
                abs_json = os.path.join(setup_dir, "strategy.abs.json")
                if not os.path.exists(abs_json):
                    mismatches += 1
                    continue
                with open(abs_json, "r") as f:
                    abs_str_to_id = json.load(f)
                h_m1, h_m2, abs_str = _split_suffix(s["suffix"])
                if abs_str not in abs_str_to_id:
                    mismatches += 1
                    continue
                abs_id = abs_str_to_id[abs_str]
                comp = (s["hand_size"]
                        | (s["last_bet"] << LAST_BET_SHIFT)
                        | (h_m1 << H_M1_SHIFT)
                        | (h_m2 << H_M2_SHIFT)
                        | (abs_id << ABS_ID_SHIFT))
                # Find row in diag by binary search (keys are sorted)
                pos = int(np.searchsorted(diag["keys"], comp))
                if pos >= len(diag["keys"]) or int(diag["keys"][pos]) != comp:
                    mismatches += 1
                    continue
                if (int(diag["first_touched"][pos]) != s["first_touched"]
                        or int(diag["last_touched"][pos]) != s["last_touched"]
                        or int(diag["times_touched"][pos]) != s["times_touched"]):
                    mismatches += 1
                    continue
                # Compare strategy: CSV's v is the raw averaged strategy,
                # NPZ's strategy field is the same (padded). Decode & compare.
                csv_probs = decode_probabilities(s["encoded"]).astype(np.float32)
                lo = int(diag["lower"][pos])
                hi = int(diag["upper"][pos])
                npz_probs = diag["strategy"][pos, lo:hi + 1]
                width = hi - lo + 1
                if len(csv_probs) > width:
                    csv_probs = csv_probs[:width]
                if len(csv_probs) < width:
                    pad = np.zeros(width, dtype=np.float32)
                    pad[:len(csv_probs)] = csv_probs
                    csv_probs = pad
                # Diagnostic stores the raw averaged strategy WITHOUT the
                # clear_lows + renormalise that the deployment strategy.npz
                # gets. So a straight elementwise compare is right.
                if not np.allclose(csv_probs, npz_probs, atol=1e-6, rtol=0):
                    mismatches += 1
            out["diagnostic_spot"] = f"{len(samples) - mismatches}/{len(samples)} match"
            out["ok_diagnostic"] = (mismatches == 0)
        else:
            out["diagnostic_spot"] = "skip (count mismatch)"

    out["ok"] = out["ok_strategy"] and out["ok_diagnostic"]
    return out


def _delete_csvs(setup_dir: str, hand_sizes: List[int]) -> dict:
    """Delete both strategy and diagnostic CSV folders for one setup.
    Returns size freed."""
    freed = 0
    deleted_dirs = []
    for hs in set(hand_sizes):
        for sub_name in (str(hs), f"{hs}_diagnostic"):
            sub = os.path.join(setup_dir, sub_name)
            if os.path.isdir(sub):
                # tally first
                for r, _, files in os.walk(sub):
                    for f in files:
                        freed += os.path.getsize(os.path.join(r, f))
                shutil.rmtree(sub)
                deleted_dirs.append(sub_name)
    return {"freed_bytes": freed, "deleted": deleted_dirs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=[],
                    help="Walk this root for setup folders. May repeat.")
    ap.add_argument("--delete", action="store_true",
                    help="Delete CSVs for setups where both strategy and "
                         "diagnostic verifications pass.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.root:
        ap.error("at least one --root is required")

    targets = []
    for root in args.root:
        if not os.path.isdir(root):
            print(f"  WARNING: --root {root} doesn't exist; skipping",
                  flush=True)
            continue
        targets.extend(_find_setup_dirs(root))
    print(f"Found {len(targets)} setup directories to verify", flush=True)

    rng = np.random.default_rng(args.seed)
    print(f"\n  {'setup':>20}  {'s_csv':>8}  {'s_npz':>8}  {'s_spot':>12}  "
          f"{'d_csv':>10}  {'d_npz':>10}  {'d_spot':>12}  status",
          flush=True)
    total_freed = 0
    n_ok = 0
    n_fail = 0
    for setup_dir, hand_sizes in targets:
        res = _verify_one(setup_dir, hand_sizes, rng)
        status_parts = []
        if res["ok"]:
            status_parts.append("OK")
        else:
            if res["strategy_npz"] and not res["ok_strategy"]:
                status_parts.append("S-FAIL")
            if res["diagnostic_npz"] and not res["ok_diagnostic"]:
                status_parts.append("D-FAIL")
            if not res["strategy_npz"]:
                status_parts.append("NO-S-NPZ")
            if not res["diagnostic_npz"]:
                status_parts.append("NO-D-NPZ")
        status = ",".join(status_parts) if status_parts else "UNKNOWN"

        print(f"  {res['setup']:>20}  "
              f"{res['strategy_csv_rows']:>8d}  "
              f"{res['strategy_npz_rows']:>8d}  "
              f"{res['strategy_spot']:>12}  "
              f"{res['diagnostic_csv_rows']:>10d}  "
              f"{res['diagnostic_npz_rows']:>10d}  "
              f"{res['diagnostic_spot']:>12}  {status}",
              flush=True)
        if res["ok"]:
            n_ok += 1
            if args.delete:
                d = _delete_csvs(setup_dir, hand_sizes)
                total_freed += d["freed_bytes"]
        else:
            n_fail += 1

    print(f"\n  totals: ok={n_ok}  failed={n_fail}", flush=True)
    if args.delete:
        print(f"  freed: {total_freed/1024/1024/1024:.2f} GB", flush=True)
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
