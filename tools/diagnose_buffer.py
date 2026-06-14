"""
Phase 0.5a/b free diagnostics.

Loads an NFSP checkpoint and reports:

  0.5a — Buffer-distribution histograms by total card count (own hand size,
         visible-card-count proxy from observation slice). Confirms or rules
         out the buffer-distribution-starvation hypothesis from
         docs/phase_plan.md §111.

  0.5b — Q-loss in-distribution vs head-to-head winrate held-out from the
         run's metrics.csv, plus policy entropy. Tests the overfitting
         hypothesis (train↓ eval↔ ⇒ overfit).

Run from repo root:

  python -m tools.diagnose_buffer \\
    --checkpoint runs/<id>/checkpoints/nfsp_blef_NM.pt \\
    --metrics runs/<id>/metrics.csv \\
    --output runs/diagnose_<id>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

import torch


def _hand_size_from_obs(obs: torch.Tensor, deck_size: int) -> int:
    """The observation packs the agent's own hand near the start of the vector
    as a multi-hot of length deck_size (24 or 32). Sum of those bits is the
    own-hand card count. Schema is stable across recent training (`_compute_obs_dim`).
    """
    return int(obs[:deck_size].sum().item())


def histogram_buf(buf, deck_size: int, name: str, max_size: int = 100_000) -> Dict[str, Any]:
    """Histogram total card count across stored transitions.

    Accepts either a buffer-object (with `.obs` / `.size` attrs) or the dict
    form serialized into checkpoints (`{'obs': tensor, 'size': int, ...}`).
    Sub-samples to `max_size` to bound runtime on huge buffers.
    """
    if isinstance(buf, dict):
        obs = buf.get("obs")
        size = int(buf.get("size") or buf.get("filled") or 0)
    else:
        obs = getattr(buf, "obs", None)
        size = int(getattr(buf, "size", 0) or 0)
    if obs is None or size == 0:
        return {"name": name, "size": 0, "histogram": {}, "note": "buffer empty or absent"}

    if size > max_size:
        idx = torch.randperm(size)[:max_size]
    else:
        idx = torch.arange(size)

    counts: Counter = Counter()
    for i in idx.tolist():
        c = _hand_size_from_obs(obs[i], deck_size)
        counts[c] += 1

    n = sum(counts.values())
    hist = {str(k): {"count": v, "frac": round(v / n, 4)} for k, v in sorted(counts.items())}

    # Compute distribution skew metrics
    weighted_mean = sum(int(k) * v for k, v in counts.items()) / n
    return {
        "name": name,
        "size_total": size,
        "size_sampled": int(idx.numel()),
        "histogram_card_count": hist,
        "mean_card_count_per_obs": round(weighted_mean, 3),
        "p50_card_count": _percentile(counts, 0.5),
        "p90_card_count": _percentile(counts, 0.9),
        "p99_card_count": _percentile(counts, 0.99),
    }


def _percentile(counts: Counter, p: float) -> int:
    """Weighted percentile of an integer-keyed Counter."""
    n = sum(counts.values())
    target = p * n
    cum = 0
    for k in sorted(counts.keys()):
        cum += counts[k]
        if cum >= target:
            return int(k)
    return int(max(counts.keys()))


def metrics_summary(metrics_path: str) -> Dict[str, Any]:
    """Read metrics.csv and report Q-loss / winrate / entropy windows."""
    if not os.path.exists(metrics_path):
        return {"note": f"metrics.csv not found at {metrics_path}"}

    import csv as _csv

    rows: List[Dict[str, str]] = []
    with open(metrics_path) as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    def f(row, key):
        v = row.get(key, "")
        try:
            return float(v) if v not in ("", "nan") else None
        except ValueError:
            return None

    def window(rows, start_step, end_step, key):
        vals = []
        for r in rows:
            try:
                s = int(float(r["step"]))
            except (KeyError, ValueError):
                continue
            if start_step <= s <= end_step:
                v = f(r, key)
                if v is not None:
                    vals.append(v)
        return vals

    if not rows:
        return {"note": "metrics.csv empty"}

    last_step = int(float(rows[-1]["step"]))
    first_step = int(float(rows[0]["step"]))

    def window_stats(start, end):
        return {
            "q_loss_mean": _mean(window(rows, start, end, "q_loss")),
            "sl_loss_mean": _mean(window(rows, start, end, "sl_loss")),
            "entropy_mean": _mean(window(rows, start, end, "policy_entropy")),
            "winrate_mean": _mean(window(rows, start, end, "win_rate")),
            "avg_reward_mean": _mean(window(rows, start, end, "avg_reward")),
            "illegal_rate_mean": _mean(window(rows, start, end, "illegal_rate")),
            "n_rows": len(window(rows, start, end, "q_loss")),
        }

    early_end = first_step + (last_step - first_step) // 5
    late_start = last_step - (last_step - first_step) // 5

    return {
        "first_step": first_step,
        "last_step": last_step,
        "n_rows": len(rows),
        "early_window": {"start": first_step, "end": early_end, **window_stats(first_step, early_end)},
        "late_window": {"start": late_start, "end": last_step, **window_stats(late_start, last_step)},
    }


def _mean(xs):
    if not xs:
        return None
    return round(sum(xs) / len(xs), 6)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--deck-size", type=int, default=24)
    args = p.parse_args(argv)

    out: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "metrics": args.metrics,
        "deck_size": args.deck_size,
    }

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    out["checkpoint_keys"] = sorted(list(ckpt.keys()))
    out["steps"] = int(ckpt.get("steps", -1))

    rl_buf = ckpt.get("rl_buf")
    sl_buf = ckpt.get("sl_buf")
    if rl_buf is not None:
        out["rl_buffer"] = histogram_buf(rl_buf, args.deck_size, "rl_buf")
    if sl_buf is not None:
        out["sl_buffer"] = histogram_buf(sl_buf, args.deck_size, "sl_buf")

    out["metrics_summary"] = metrics_summary(args.metrics)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
