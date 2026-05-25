"""OS hyperparameter sweep on (1,1) at fixed iter / single seed.

Runs a Cartesian product over (epsilon, variant) on (1,1) at 100k iter,
single seed, and prints LBR-1 per config. Designed to be split across
multiple parallel processes via --start / --count slicing of the config
list.

Cartesian:
  epsilon  in {0.05, 0.1, 0.3, 0.6, 0.9}
  variant  in {plain, cfr_plus, dcfr_linear}
            plain        — current OS (variant A); no regret clip; reach weighting
            cfr_plus     — clip regrets at 0 after each update; reach weighting
            dcfr_linear  — clip regrets at 0 + DCFR alpha=1.5 + linear strategy avg

Usage:
  python -m cfr_ai.analysis.os_sweep --start 0 --count 5    # first 5 configs
  python -m cfr_ai.analysis.os_sweep --start 5 --count 5    # next 5
  python -m cfr_ai.analysis.os_sweep --start 10 --count 5   # last 5
"""
from __future__ import annotations

import argparse
import os
import random
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np

from cfr_ai.lbr import lbr_exploitability
from cfr_ai.trainer_os import OutcomeSamplingTrainer


def make_configs(epsilons, variants):
    return [{"eps": e, "variant": v} for e in epsilons for v in variants]


def trainer_for(cfg, hand_sizes):
    variant = cfg["variant"]
    if variant == "plain":
        return OutcomeSamplingTrainer(hand_sizes, 0, 0, exploration=cfg["eps"])
    if variant == "cfr_plus":
        return OutcomeSamplingTrainer(
            hand_sizes, 0, 0, exploration=cfg["eps"], regret_clip=True
        )
    if variant == "dcfr_linear":
        return OutcomeSamplingTrainer(
            hand_sizes, 0, 0, exploration=cfg["eps"],
            regret_clip=True, dcfr_alpha=1.5, strategy_weighting="linear",
        )
    raise ValueError(f"Unknown variant: {variant}")


def run_config(cfg, hand_sizes, num_iter, seed):
    random.seed(seed)
    np.random.seed(seed)
    trainer = trainer_for(cfg, hand_sizes)
    t0 = time.time()
    trainer.train(num_iter)
    train_sec = time.time() - t0
    cfr_strategy = {k: v.get_final_strategy() for k, v in trainer.infoset_map.items()}
    t0 = time.time()
    sps = [0] if hand_sizes[0] == hand_sizes[1] else [0, 1]
    lbrs = {}
    for sp in sps:
        r = lbr_exploitability(
            hand_sizes, sp, cfr_strategy, cfr_min_bet=0, depth=1,
            n_belief_samples=None, n_lbr_hand_samples=None, seed=42,
        )
        lbrs[sp] = r["expl"]
    lbr_sec = time.time() - t0
    return {
        "cfg": cfg, "train_sec": train_sec, "lbr_sec": lbr_sec, "lbrs": lbrs,
        "n_infosets": len(trainer.infoset_map),
    }


def fmt_lbr(lbrs):
    return " | ".join(f"sp{sp}: {v * 100:+.3f}%" for sp, v in sorted(lbrs.items()))


def main():
    import psutil
    psutil.Process().nice(psutil.NORMAL_PRIORITY_CLASS)

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=99)
    p.add_argument("--iter", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hand-sizes", nargs=2, type=int, default=[1, 1])
    p.add_argument("--epsilons", default="0.05,0.1,0.3,0.6,0.9",
                   help="Comma-separated list of epsilon values.")
    p.add_argument("--variants", default="plain,cfr_plus,dcfr_linear",
                   help="Comma-separated list of regret-matching variants.")
    args = p.parse_args()

    eps_list = [float(x) for x in args.epsilons.split(",")]
    variant_list = [x.strip() for x in args.variants.split(",")]
    configs = make_configs(eps_list, variant_list)
    chunk = configs[args.start:args.start + args.count]
    print(f"[os_sweep] {len(chunk)}/{len(configs)} configs "
          f"(slice [{args.start}:{args.start + args.count}]), "
          f"hand_sizes={args.hand_sizes}, iter={args.iter:,}, seed={args.seed}",
          flush=True)

    results = []
    for i, cfg in enumerate(chunk):
        global_idx = args.start + i
        print(f"\n[os_sweep] config {global_idx} ({i + 1}/{len(chunk)}): {cfg}", flush=True)
        r = run_config(cfg, args.hand_sizes, args.iter, args.seed)
        results.append(r)
        print(
            f"[os_sweep]   train {r['train_sec']:.1f}s  lbr {r['lbr_sec']:.1f}s  "
            f"infosets {r['n_infosets']:,}  LBR-1 {fmt_lbr(r['lbrs'])}",
            flush=True,
        )

    # Compact end-of-process summary.
    print("\n[os_sweep] === slice complete ===")
    for r in results:
        cfg = r["cfg"]
        print(f"  eps={cfg['eps']:<5}  variant={cfg['variant']:<12}  "
              f"LBR-1 {fmt_lbr(r['lbrs'])}", flush=True)


if __name__ == "__main__":
    main()
