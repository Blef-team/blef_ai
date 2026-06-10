"""Sweep per-decision latency, RAM, and posterior structure across many
setups. The goal is to:

  1. Identify which setups fit in the AWS Lambda budget (2 s/decision,
     1 GB RAM) at default settings.
  2. Identify which setups need belief sampling and at what `N`.
  3. Confirm correctness sanity (V values plausible, posterior non-zero).

For each setup we time the median and worst-case decision over a small
number of realistic states (blueprint-walked histories).

CSV output: `cfr_ai/outputs/subgame_sweep.csv` with columns:
  setup, n_belief_samples, n_opp, ram_mb, load_s, decisions, p50_ms,
  p95_ms, max_ms, mean_V

Run:
  python -m cfr_ai.analysis.subgame_setup_sweep \\
    --setups 1_1 1_2 2_2 1_3 2_3 3_3 1_4 2_4 3_4 4_4 \\
    --n-decisions 8 \\
    --n-belief-samples 300
"""

import argparse
import csv
import itertools
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import psutil

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cfr_ai.game import Game
from cfr_ai.lbr import load_flat_strategy
from cfr_ai.subgame import subgame_pick_action, make_context, warmup


def _walk(fs, hand_sizes, deal, rng, max_steps):
    from cfr_ai.information_set import make_key, get_hand_abstraction
    from cfr_ai.lbr import _parse_key_to_composite
    history = []
    for step in range(max_steps):
        seat = step % 2
        hand = deal[seat]
        hand_size = hand_sizes[seat]
        hand_abs = get_hand_abstraction(hand, hand_sizes)
        key = make_key(hand, hand_abs, history, fs.min_bet)
        parts = key.split("-")
        last_bet = int(parts[1])
        suffix = "-".join(parts[2:])
        if suffix.split("-")[-1] not in fs.abs_str_to_id:
            return history
        comp_key = _parse_key_to_composite(
            suffix, hand_size, last_bet, dict(fs.abs_str_to_id))
        row = fs.key_to_row.get(np.int64(comp_key))
        if row is None:
            return history
        lo = int(fs.lower_action[row])
        hi = int(fs.upper_action[row])
        probs = np.array(fs.strategy[row, lo:hi + 1], dtype=np.float64)
        if probs.sum() <= 0:
            return history
        probs /= probs.sum()
        chosen_offset = int(rng.choice(len(probs), p=probs))
        a = lo + chosen_offset
        history.append(a)
        if a == 88:
            history.pop()
            return history
    return history


def _sweep_setup(setup_str: str, n_decisions: int, n_belief_samples,
                 action_topk, action_prob_threshold,
                 rng_seed: int) -> dict:
    hand_sizes = sorted(int(x) for x in setup_str.split("_"))
    rng = np.random.default_rng(rng_seed)
    proc = psutil.Process(os.getpid())
    ram_before = proc.memory_info().rss / 1024 / 1024

    t0 = time.time()
    fs = load_flat_strategy(hand_sizes)
    load_s = time.time() - t0
    ram_after_load = proc.memory_info().rss / 1024 / 1024

    # n_opp upper bound
    full_n_opp_per_seat = [
        len(list(itertools.combinations(range(24 - hand_sizes[s]), hand_sizes[1 - s])))
        for s in (0, 1)
    ]
    full_max = max(full_n_opp_per_seat)
    if n_belief_samples is not None:
        max_n_opp = min(n_belief_samples, full_max)
        n_opp_actual = max_n_opp
    else:
        max_n_opp = full_max
        n_opp_actual = full_max

    # Sanity: refuse if RAM forecast would exceed 2 GB
    forecast_mb = (max_n_opp * 89 * 92 * 8 + max_n_opp * 92 * 8 + max_n_opp * 8) / (1024*1024)
    if forecast_mb > 1500:
        return {
            "setup": setup_str,
            "n_belief_samples": n_belief_samples,
            "n_opp": max_n_opp,
            "ram_mb_load": ram_after_load - ram_before,
            "ram_mb_total": None,
            "load_s": load_s,
            "decisions": 0,
            "p50_ms": None, "p95_ms": None, "max_ms": None,
            "mean_V": None,
            "note": f"refused: forecast pd_buf {forecast_mb:.0f} MB > 1500",
        }

    ctx = make_context(hand_sizes, fs, max_n_opp)
    warmup(hand_sizes, fs, ctx, n_belief_samples=n_belief_samples)
    ram_after_warmup = proc.memory_info().rss / 1024 / 1024

    # Per-decision sweep
    times = []
    Vs = []
    for k in range(n_decisions):
        deal = Game.deal_cards(hand_sizes)
        walk_len = int(rng.integers(0, 6))
        history = _walk(fs, hand_sizes, deal, rng, max_steps=walk_len)
        seat = len(history) % 2
        my_hand = deal[seat]
        t0 = time.time()
        a, diag = subgame_pick_action(
            my_hand, hand_sizes, seat, history, fs,
            n_belief_samples=n_belief_samples,
            context=ctx,
            action_topk=action_topk,
            action_prob_threshold=action_prob_threshold,
            return_diagnostics=True,
        )
        dt = time.time() - t0
        times.append(dt)
        if "best_value" in diag:
            Vs.append(diag["best_value"])

    arr = np.array(times) * 1000
    return {
        "setup": setup_str,
        "n_belief_samples": n_belief_samples,
        "n_opp": n_opp_actual,
        "ram_mb_load": ram_after_load - ram_before,
        "ram_mb_total": ram_after_warmup - ram_before,
        "load_s": load_s,
        "decisions": len(arr),
        "p50_ms": float(np.median(arr)),
        "p95_ms": float(np.quantile(arr, 0.95)),
        "max_ms": float(arr.max()),
        "mean_V": float(np.mean(Vs)) if Vs else None,
        "note": "",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setups", nargs="+", required=True,
                   help="Setup folder names like '1_3', '3_7'.")
    p.add_argument("--n-decisions", type=int, default=8)
    p.add_argument("--n-belief-samples", type=int, default=None)
    p.add_argument("--action-topk", type=int, default=None)
    p.add_argument("--action-prob-threshold", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-csv", type=str,
                   default=os.path.join("cfr_ai", "outputs", "subgame_sweep.csv"))
    args = p.parse_args()

    results = []
    print(f"  {'setup':>6} {'n_belief':>9} {'n_opp':>6} {'ram_MB':>7} "
          f"{'load_s':>7} {'p50_ms':>7} {'p95_ms':>7} {'max_ms':>7} "
          f"{'mean_V':>8}  note", flush=True)
    for setup in args.setups:
        try:
            r = _sweep_setup(
                setup, args.n_decisions,
                args.n_belief_samples,
                args.action_topk,
                args.action_prob_threshold,
                args.seed,
            )
        except Exception as e:
            r = {
                "setup": setup, "n_belief_samples": args.n_belief_samples,
                "n_opp": None, "ram_mb_load": None, "ram_mb_total": None,
                "load_s": None, "decisions": 0,
                "p50_ms": None, "p95_ms": None, "max_ms": None,
                "mean_V": None, "note": f"ERROR: {type(e).__name__}: {e}",
            }
        results.append(r)
        s = r
        print(f"  {s['setup']:>6} "
              f"{str(s.get('n_belief_samples') or '-'):>9} "
              f"{str(s.get('n_opp') or '-'):>6} "
              f"{str(round(s['ram_mb_total'])) if s.get('ram_mb_total') else '-':>7} "
              f"{str(round(s['load_s'], 1)) if s.get('load_s') else '-':>7} "
              f"{str(round(s['p50_ms'], 0)) if s.get('p50_ms') else '-':>7} "
              f"{str(round(s['p95_ms'], 0)) if s.get('p95_ms') else '-':>7} "
              f"{str(round(s['max_ms'], 0)) if s.get('max_ms') else '-':>7} "
              f"{str(round(s['mean_V'], 4)) if s.get('mean_V') is not None else '-':>8}  "
              f"{s.get('note', '')}",
              flush=True)

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        write_header = not os.path.exists(args.output_csv)
        fields = ["setup", "n_belief_samples", "n_opp", "ram_mb_load",
                  "ram_mb_total", "load_s", "decisions", "p50_ms",
                  "p95_ms", "max_ms", "mean_V", "note"]
        with open(args.output_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()
            for r in results:
                w.writerow({k: r.get(k) for k in fields})
        print(f"\nResults appended to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
