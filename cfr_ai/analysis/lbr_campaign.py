"""LBR campaign closing the EXPERIMENTS.md exploitability gaps (run on the box).

Three measurements, written to one CSV (no production-summary writes):
  [gate] depth-3 fp LBR-1 on 1_1 + 2_2 must reproduce the recorded +0.017% / +0.615%.
  [tv]   H2H penalty-0 (temp-value ON) vs penalty-0-notv (OFF) on 4_5 + 5_11 - the
         un-numbered claim in EXPERIMENTS 2.3.
  [q8]   LBR on the uint8-quantised serving staging of V3.2x4 (what prod actually
         serves) over the recorded LBR band - closes 5.9's open gate.
  [d2]   LBR on the depth-2 4x models (v32x4_hist2) over the same band - the
         history-depth exploitability cost at matched budget (5.11's 2p gate).

Sampling matches the recorded production runs: n_belief=500, n_lbr_hand=1000,
seed=42; symmetric setups run sp0 only, asymmetric sp0+sp1 (summary convention).

    .venv/bin/python -m cfr_ai.analysis.lbr_campaign --workers 6 --out /root/lbr_campaign.csv
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

# The 12 setups with recorded LBR-2 (all concrete, total <= 7; the macro band
# would need the lbr_macro path + a macro-aware quantised loader - out of scope).
LBR2_BAND = [(1, 1), (1, 2), (1, 3), (2, 2), (1, 4), (2, 3),
             (1, 5), (2, 4), (3, 3), (1, 6), (2, 5), (3, 4)]
N_BELIEF, N_LBR_HAND, SEED = 500, 1000, 42

# NB the box's cfr_ai/outputs is the OLD V2 2p model - the true V3.2x4 band
# setups are uploaded from the laptop into v32x4_band (gate verifies them).
Q8_SRC = "cfr_ai/experiments/v32x4_band"                 # V3.2x4 depth-3 fp (uploaded)
Q8_STAGE = "cfr_ai/experiments/q8_stage"                 # uint8 staging (built here)
D2_SRC = "cfr_ai/experiments/v32x4_hist2/outputs"        # depth-2 4x fp (box-trained)


def load_quantised(setup_dir):
    """Staged uint8/uint16 mmap layout -> lbr.FlatStrategy (dequantised).
    All-zero rows keep their key OUT of the dict so lookups fall back to the
    check-100% default - exactly what the deployed agent serves for them."""
    from numba import types
    from numba.typed import Dict as NbDict
    from cfr_ai.lbr import FlatStrategy
    meta = np.load(os.path.join(setup_dir, "strategy_meta.npz"), allow_pickle=True)
    assert "kinds" not in meta.files, "macro setups not supported by this loader"
    keys = np.load(os.path.join(setup_dir, "meta_keys.npy"))
    off = np.load(os.path.join(setup_dir, "meta_offset.npy"))
    lower = np.load(os.path.join(setup_dir, "meta_lower.npy"))
    upper = np.load(os.path.join(setup_dir, "meta_upper.npy"))
    idx = np.load(os.path.join(setup_dir, "probs_sparse_indices.npy"))
    val = np.load(os.path.join(setup_dir, "probs_sparse_values.npy"))
    with open(os.path.join(setup_dir, "strategy.abs.json"), encoding="utf-8") as f:
        abs_map = json.load(f)
    scale = 255.0 if val.dtype == np.uint8 else 65535.0
    n = keys.shape[0]
    probs = np.zeros((n, 89), dtype=np.float32)
    key_to_row = NbDict.empty(key_type=types.int64, value_type=types.int64)
    for i in range(n):
        a, b = int(off[i]), int(off[i + 1])
        if a == b:
            continue
        v = val[a:b].astype(np.float64) / scale
        cols = int(lower[i]) + idx[a:b].astype(np.int64)
        probs[i, cols] = (v / v.sum()).astype(np.float32)
        key_to_row[np.int64(keys[i])] = np.int64(i)
    return FlatStrategy(
        key_to_row=key_to_row, strategy=probs,
        lower_action=np.ascontiguousarray(lower, dtype=np.int16),
        upper_action=np.ascontiguousarray(upper, dtype=np.int16),
        abs_str_to_id=abs_map, min_bet=int(meta["min_bet"]),
        history_depth=int(meta["history_depth"]) if "history_depth" in meta.files else 3,
    )


def _load(variant, setup):
    from cfr_ai.strategy_io import load_strategy
    d = "_".join(map(str, setup))
    if variant == "q8":
        return load_quantised(os.path.join(Q8_STAGE, d))
    if variant == "d2":
        return load_strategy(os.path.join(D2_SRC, d))
    if variant == "fp":
        return load_strategy(os.path.join(Q8_SRC, d))
    raise ValueError(variant)


def run_task(task):
    variant, setup, depth = task
    from cfr_ai.lbr import lbr_exploitability
    t0 = time.time()
    fs = _load(variant, list(setup))
    sps = (0,) if setup[0] == setup[1] else (0, 1)
    out = []
    for sp in sps:
        r = lbr_exploitability(list(setup), sp, fs, depth=depth,
                               n_belief_samples=N_BELIEF,
                               n_lbr_hand_samples=N_LBR_HAND,
                               seed=SEED, show_progress=False)
        out.append({"variant": variant, "setup": "_".join(map(str, setup)),
                    "depth": depth, "sp": sp, "hist_depth": fs.history_depth,
                    "expl_pct": round(r["expl"] * 100, 4),
                    "se_worst_pp": round(r["se_worst"] * 100, 4),
                    "K": r["K_lbr_hand"], "secs": round(time.time() - t0)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="/root/lbr_campaign.csv")
    args = ap.parse_args()
    os.environ.setdefault("TQDM_DISABLE", "1")

    # --- [gate] depth-3 regression: must reproduce the recorded numbers.
    print("[gate] depth-3 fp regression...", flush=True)
    for setup, want in (((1, 1), 0.017), ((2, 2), 0.615)):
        r = run_task(("fp", setup, 1))[0]
        ok = abs(r["expl_pct"] - want) < 5e-3
        print(f"  {setup} LBR-1 sp0 = {r['expl_pct']:+.3f}% (recorded {want:+.3f}%) "
              f"{'OK' if ok else 'MISMATCH - ABORT'}", flush=True)
        if not ok:
            return 1

    # --- [tv] H2H penalty-0 vs penalty-0-notv (temp-value isolation, EXPERIMENTS 2.3).
    for s in ("4 5", "5 11"):
        cmd = [sys.executable, "-m", "cfr_ai.analysis.h2h_persetup_mc",
               "--hand-sizes", *s.split(),
               "--model-a-folder", "cfr_ai/experiments/penalty-0",
               "--model-b-folder", "cfr_ai/experiments/penalty-0-notv",
               "--num-deals", "40000", "--seed", "0", "--out", "/root/tv_h2h.csv"]
        print(f"[tv] h2h {s} (B = no-temp-value)...", flush=True)
        rc = subprocess.run(cmd, capture_output=True, text=True)
        print(rc.stdout.strip()[-300:] or rc.stderr.strip()[-300:], flush=True)

    # --- stage the uint8 layouts (fast; skip already-staged).
    from cfr_ai.strategy_io import write_mmap_layout
    band = LBR2_BAND
    for setup in band:
        d = "_".join(map(str, setup))
        dst = os.path.join(Q8_STAGE, d)
        if not os.path.exists(os.path.join(dst, "strategy_meta.npz")):
            write_mmap_layout(os.path.join(Q8_SRC, d), dst, value_bits=8)
            print(f"[stage] q8 {d}", flush=True)

    # --- LBR task pool, most expensive first so workers stay busy.
    tasks = []
    for variant in ("q8", "d2"):
        for setup in band:
            tasks.append((variant, setup, 1))
        for setup in LBR2_BAND:
            tasks.append((variant, setup, 2))
    cost = {(1, 1): 1, (1, 2): 1, (1, 3): 2, (2, 2): 2, (1, 4): 3, (2, 3): 20,
            (1, 5): 8, (2, 4): 60, (3, 3): 50, (1, 6): 16, (2, 5): 110, (3, 4): 290,
            (1, 7): 30, (1, 8): 160}
    tasks.sort(key=lambda t: -cost.get(tuple(t[1]), 1) * (30 if t[2] == 2 else 1))

    fields = ["variant", "setup", "depth", "sp", "hist_depth",
              "expl_pct", "se_worst_pp", "K", "secs"]
    with open(args.out, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    done = 0
    with Pool(args.workers) as pool:
        for rows in pool.imap_unordered(run_task, tasks):
            with open(args.out, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerows(rows)
            done += 1
            r = rows[0]
            cells = " | ".join(f"{x['expl_pct']:+.3f}%" for x in rows)
            print(f"[{done}/{len(tasks)}] {r['variant']} {r['setup']} LBR-{r['depth']}"
                  f" -> {cells} ({r['secs']}s)", flush=True)
    print("LBR-CAMPAIGN-DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
