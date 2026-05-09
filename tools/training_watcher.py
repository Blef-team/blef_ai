#!/usr/bin/env python3
"""Watch NFSP training metrics for plateau / regression / divergence.

Reads metrics.csv every poll, computes rolling slopes, and prints one
line per scheduled snapshot or anomaly. Exit when training PID dies.

Schema (current): step, avg_reward, win_rate, avg_len, q_loss, sl_loss,
policy_entropy, illegal_rate, epsilon, anticipatory_eta, lr_q, n_step,
train_rl_every, check_explore_prob, max_cards, rl_buf, sl_buf, ...

Each step appears twice in the file (BR row + behavior row); we use the
BR rows (those with non-empty q_loss / sl_loss).
"""
from __future__ import annotations

import csv
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


def _read_br_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            # BR rows have q_loss / sl_loss populated (not empty / nan)
            qv = r.get("q_loss", "")
            if qv == "" or qv == "nan":
                continue
            try:
                rows.append({
                    "step": int(r["step"]),
                    "avg_reward": float(r["avg_reward"]),
                    "win_rate": float(r["win_rate"]),
                    "q_loss": float(r["q_loss"]),
                    "sl_loss": float(r["sl_loss"]),
                    "policy_entropy": float(r.get("policy_entropy", "nan") or "nan"),
                    "epsilon": float(r.get("epsilon", "nan") or "nan"),
                    "eta": float(r.get("anticipatory_eta", "nan") or "nan"),
                    "rl_buf": int(float(r.get("rl_buf", 0) or 0)),
                    "sl_buf": int(float(r.get("sl_buf", 0) or 0)),
                })
            except (ValueError, KeyError):
                continue
    return rows


@dataclass
class Slope:
    delta: float
    span_steps: int

    @property
    def per_M(self) -> float:
        return self.delta / max(1, self.span_steps) * 1_000_000


def _slope_over_steps(rows: list[dict], field: str, window_steps: int) -> Optional[Slope]:
    if len(rows) < 2:
        return None
    last = rows[-1]
    target_step = last["step"] - window_steps
    # Find the row with step closest to (and >=) target_step
    pivot = None
    for r in rows:
        if r["step"] >= target_step:
            pivot = r
            break
    if pivot is None or pivot is last:
        return None
    return Slope(delta=last[field] - pivot[field], span_steps=last["step"] - pivot["step"])


def main():
    if len(sys.argv) < 4:
        print("usage: training_watcher.py <label> <run_dir> <pid> [poll_sec]", file=sys.stderr)
        sys.exit(2)
    label = sys.argv[1]
    run_dir = sys.argv[2]
    pid = int(sys.argv[3])
    poll = int(sys.argv[4]) if len(sys.argv) >= 5 else 60
    csv_path = os.path.join(run_dir, "metrics.csv")
    last_step_seen = -1
    last_snapshot_step = -1
    snapshot_every = 200_000  # one snapshot line per 200K steps
    anomaly_seen = set()  # avoid spamming the same anomaly

    def alive() -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    print(f"[{label}] watcher armed; pid={pid} poll={poll}s")
    sys.stdout.flush()

    while True:
        if not alive():
            rows = _read_br_rows(csv_path)
            last = rows[-1] if rows else None
            print(f"[{label}] CRASH: pid {pid} exited. last step={last['step'] if last else 'n/a'}")
            sys.stdout.flush()
            return

        rows = _read_br_rows(csv_path)
        if not rows:
            time.sleep(poll)
            continue
        last = rows[-1]
        if last["step"] == last_step_seen:
            time.sleep(poll)
            continue
        last_step_seen = last["step"]

        # Snapshot line every snapshot_every steps
        if last["step"] - last_snapshot_step >= snapshot_every:
            wr_slope_1M = _slope_over_steps(rows, "win_rate", 1_000_000)
            rew_slope_1M = _slope_over_steps(rows, "avg_reward", 1_000_000)
            slope_str = ""
            if wr_slope_1M:
                slope_str = (
                    f" wrΔ/1M={wr_slope_1M.per_M:+.3f}"
                    f" rewΔ/1M={rew_slope_1M.per_M:+.3f}"
                    if rew_slope_1M else ""
                )
            print(
                f"[{label}] step={last['step']:>9d} "
                f"rew={last['avg_reward']:.3f} wr={last['win_rate']:.3f} "
                f"q={last['q_loss']:.3f} sl={last['sl_loss']:.3f} "
                f"H={last['policy_entropy']:.2f} "
                f"η={last['eta']:.2f} ε={last['epsilon']:.2f}"
                f"{slope_str}"
            )
            sys.stdout.flush()
            last_snapshot_step = last["step"]

        # === Anomaly checks (each fires once per training) ===
        # Only run after we have at least 1M steps of data
        if last["step"] < 1_000_000:
            time.sleep(poll)
            continue

        # 1) win-rate plateau over last 2M steps — slope < 0.005 per 1M
        if "wr_plateau_2M" not in anomaly_seen:
            s = _slope_over_steps(rows, "win_rate", 2_000_000)
            if s and s.span_steps >= 1_500_000 and abs(s.per_M) < 0.005:
                print(
                    f"[{label}] ANOMALY win_rate plateau: Δ/1M={s.per_M:+.4f} "
                    f"over {s.span_steps:,} steps (last wr={last['win_rate']:.3f}). "
                    "consider eta↑ or epsilon adjust."
                )
                sys.stdout.flush()
                anomaly_seen.add("wr_plateau_2M")

        # 2) avg_reward regression — sustained drop > 0.05 over 1.5M steps.
        #    eval stderr is ~0.035 per sample (50K-step eval = ~50-100 games),
        #    so 500K windows trip on noise. 1.5M window is ~30 samples,
        #    which dampens single-sample swings.
        if "rew_regression" not in anomaly_seen:
            s = _slope_over_steps(rows, "avg_reward", 1_500_000)
            if s and s.span_steps >= 1_200_000 and s.delta < -0.05:
                print(
                    f"[{label}] ANOMALY avg_reward regression: Δ={s.delta:+.3f} "
                    f"over {s.span_steps:,} steps (now {last['avg_reward']:.3f}). "
                    "investigate."
                )
                sys.stdout.flush()
                anomaly_seen.add("rew_regression")

        # 3) sl_loss explosion — abs > 3.0 sustained or rise > 0.5 over 500K
        if "sl_explosion" not in anomaly_seen:
            if last["sl_loss"] > 3.0:
                print(
                    f"[{label}] ANOMALY sl_loss > 3.0 (={last['sl_loss']:.3f}). "
                    "SL divergence; consider sl_learning_off pulse or lr_pi cut."
                )
                sys.stdout.flush()
                anomaly_seen.add("sl_explosion")
            else:
                s = _slope_over_steps(rows, "sl_loss", 500_000)
                if s and s.span_steps >= 400_000 and s.delta > 0.5:
                    print(
                        f"[{label}] ANOMALY sl_loss climb: Δ={s.delta:+.3f} "
                        f"over {s.span_steps:,} steps (now {last['sl_loss']:.3f})."
                    )
                    sys.stdout.flush()
                    anomaly_seen.add("sl_explosion")

        # 4) q_loss reversal — was decreasing, now climbing > 0.05 over 500K
        if "q_reversal" not in anomaly_seen:
            s = _slope_over_steps(rows, "q_loss", 500_000)
            if s and s.span_steps >= 400_000 and s.delta > 0.05:
                # Check that q_loss had been decreasing previously
                s_prev = _slope_over_steps(rows[:-1], "q_loss", 1_000_000)
                if s_prev and s_prev.delta < -0.05:
                    print(
                        f"[{label}] ANOMALY q_loss reversal: "
                        f"recent Δ={s.delta:+.3f}, prior trend was {s_prev.delta:+.3f}."
                    )
                    sys.stdout.flush()
                    anomaly_seen.add("q_reversal")

        time.sleep(poll)


if __name__ == "__main__":
    main()
