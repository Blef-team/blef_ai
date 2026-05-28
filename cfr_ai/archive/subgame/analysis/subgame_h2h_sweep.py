"""Run subgame-vs-blueprint H2H across many setups and append the
results to `cfr_ai/outputs/subgame_h2h.csv`.

Mostly just a loop wrapper around `subgame_h2h.run_h2h` so we can fire
off an overnight sweep with one command.

Run:
  python -m cfr_ai.analysis.subgame_h2h_sweep \\
      --setups 1_1 1_2 2_2 1_3 2_3 3_3 1_4 2_4 3_4 4_4 1_5 2_5 \\
      --num-deals 300 --n-belief-samples 300 --action-topk 8
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cfr_ai.lbr import load_flat_strategy
from cfr_ai.analysis.subgame_h2h import run_h2h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setups", nargs="+", required=True)
    p.add_argument("--num-deals", type=int, default=300)
    p.add_argument("--n-belief-samples", type=int, default=300)
    p.add_argument("--action-topk", type=int, default=None)
    p.add_argument("--action-prob-threshold", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-csv", type=str,
                   default=os.path.join("cfr_ai", "outputs", "subgame_h2h.csv"))
    args = p.parse_args()

    fields = ["setup", "num_deals", "n_belief", "action_topk", "threshold",
              "seed", "advantage_start_seat0", "advantage_start_seat1",
              "wall_s", "load_s"]
    write_header = not os.path.exists(args.output_csv)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    f = open(args.output_csv, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        w.writeheader()
        f.flush()

    print(f"  {'setup':>6} {'deals':>5} {'belief':>6} {'topk':>4} "
          f"{'load_s':>6} {'wall_s':>6} {'adv_sp0':>8} {'adv_sp1':>8}",
          flush=True)
    for setup in args.setups:
        hand_sizes = sorted(int(x) for x in setup.split("_"))
        try:
            t0 = time.time()
            fs = load_flat_strategy(hand_sizes)
            load_s = time.time() - t0
            res = run_h2h(
                hand_sizes, fs, args.num_deals,
                args.n_belief_samples, args.action_topk,
                args.action_prob_threshold, args.seed,
                show_progress=False,
            )
            row = {
                "setup": setup, "num_deals": args.num_deals,
                "n_belief": args.n_belief_samples,
                "action_topk": args.action_topk if args.action_topk is not None else "",
                "threshold": args.action_prob_threshold, "seed": args.seed,
                "advantage_start_seat0": round(res.get("subgame_adv_start_seat0", 0.0), 4),
                "advantage_start_seat1": round(res.get("subgame_adv_start_seat1", 0.0), 4)
                    if "subgame_adv_start_seat1" in res else "",
                "wall_s": round(res["wall_s"], 1),
                "load_s": round(load_s, 1),
            }
            w.writerow(row)
            f.flush()
            print(f"  {setup:>6} {args.num_deals:>5d} "
                  f"{args.n_belief_samples:>6d} "
                  f"{str(args.action_topk or '-'):>4} "
                  f"{load_s:>6.1f} {res['wall_s']:>6.1f} "
                  f"{row['advantage_start_seat0']:>+8.4f} "
                  f"{str(row['advantage_start_seat1']):>8}",
                  flush=True)
        except Exception as e:
            print(f"  {setup:>6}  ERROR: {type(e).__name__}: {e}", flush=True)

    f.close()


if __name__ == "__main__":
    main()
