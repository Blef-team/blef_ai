"""Side-by-side: external- vs outcome-sampling MCCFR convergence on small setups.

Trains both samplers for the same iteration count, computes LBR-1 on the
resulting in-memory strategy (no disk I/O -- production outputs/ untouched),
and reports wall-clock and final LBR per (setup, sampler).

Example:
    python -m cfr_ai.analysis.sampling_comparison \\
        --setups 1_1 1_2 2_2 --iter 100000 \\
        --n-belief 200 --n-lbr-hand 100
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Tuple

os.environ.setdefault("TQDM_DISABLE", "1")

from cfr_ai.trainer import Trainer
from cfr_ai.trainer_os import OutcomeSamplingTrainer
from cfr_ai.lbr import lbr_exploitability


def _eval_lbr(trainer, hand_sizes: List[int], lbr_depth: int,
              n_belief: int, n_lbr_hand: int) -> Dict[int, Dict[str, float]]:
    cfr_strategy = {k: v.get_final_strategy() for k, v in trainer.infoset_map.items()}
    starting_players = [0, 1] if hand_sizes[0] != hand_sizes[1] else [0]
    out: Dict[int, Dict[str, float]] = {}
    for sp in starting_players:
        out[sp] = lbr_exploitability(
            hand_sizes, sp, cfr_strategy, cfr_min_bet=0, depth=lbr_depth,
            n_belief_samples=n_belief, n_lbr_hand_samples=n_lbr_hand, seed=42,
        )
    return out


def _run_one(name: str, trainer, hand_sizes: List[int], num_iter: int,
             lbr_depth: int, n_belief: int, n_lbr_hand: int
             ) -> Tuple[float, float, int, int, Dict[int, Dict[str, float]]]:
    print(f"  Training {name} for {num_iter:,} iterations...", flush=True)
    t0 = time.time()
    trainer.train(num_iter)
    train_secs = time.time() - t0
    print(f"    Trained in {train_secs:.1f}s "
          f"({num_iter / train_secs:.0f} it/s). Running LBR-{lbr_depth}...", flush=True)
    lbr_t0 = time.time()
    lbr = _eval_lbr(trainer, hand_sizes, lbr_depth, n_belief, n_lbr_hand)
    lbr_secs = time.time() - lbr_t0
    return train_secs, lbr_secs, len(trainer.infoset_map), trainer.nodes_touched, lbr


def _fmt_lbr_cell(lbr: Dict[int, Dict[str, float]]) -> str:
    parts = []
    for sp in sorted(lbr):
        r = lbr[sp]
        parts.append(f"sp{sp}: {r['expl']*100:+.3f}% ±{r['se_worst']*100:.2f}pp")
    return " | ".join(parts)


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setups", nargs="+", default=["1_1", "1_2", "2_2"],
                   help="Setup folder names (e.g. 1_1 1_2 2_2).")
    p.add_argument("--iter", type=int, default=100_000,
                   help="Iterations to train per (setup, sampler).")
    p.add_argument("--lbr-depth", type=int, default=1)
    p.add_argument("--n-belief", type=int, default=200,
                   help="Belief samples for the LBR evaluator (lower = faster).")
    p.add_argument("--n-lbr-hand", type=int, default=100,
                   help="LBR-hand samples (lower = faster but noisier).")
    p.add_argument("--samplers", nargs="+", default=["ES", "OS"],
                   help="Which samplers to run; subset of {ES, OS}.")
    p.add_argument("--os-exploration", type=float, default=0.6,
                   help="ε for OS ε-on-policy sampling at traverser nodes.")
    args = p.parse_args(argv)

    print(f"\nConvergence comparison: {args.samplers}, "
          f"{args.iter:,} iterations each")
    print(f"LBR-{args.lbr_depth} measured with "
          f"N_belief={args.n_belief}, K_lbr_hand={args.n_lbr_hand}\n")

    factories = {
        "ES": lambda hs: Trainer(hs, 0, [-20, -22], 0.0, 0),
        "OS": lambda hs: OutcomeSamplingTrainer(hs, 0, 0, exploration=args.os_exploration),
        "CFR+": lambda hs: Trainer(hs, 0, [-20, -22], 0.0, 0, algorithm='cfr_plus'),
        "DCFR": lambda hs: Trainer(hs, 0, [-20, -22], 0.0, 0, algorithm='dcfr'),
    }
    for s in args.samplers:
        if s not in factories:
            raise SystemExit(f"Unknown sampler: {s}")

    results: List[Tuple[List[int], str, float, float, int, int, Dict]] = []
    for setup_str in args.setups:
        hand_sizes = sorted(int(x) for x in setup_str.split("_"))
        print(f"=== Setup {hand_sizes} ===", flush=True)
        for sampler in args.samplers:
            trainer = factories[sampler](hand_sizes)
            train_secs, lbr_secs, n_info, n_nodes, lbr = _run_one(
                sampler, trainer, hand_sizes, args.iter,
                args.lbr_depth, args.n_belief, args.n_lbr_hand,
            )
            print(f"    LBR (took {lbr_secs:.1f}s): {_fmt_lbr_cell(lbr)}\n", flush=True)
            results.append((hand_sizes, sampler, train_secs, lbr_secs,
                            n_info, n_nodes, lbr))

    print("=" * 110)
    print(f"\n=== Summary (iter={args.iter:,}) ===")
    print(f"{'Setup':<8} {'Sampler':<8} {'Train(s)':>9} {'It/s':>7} "
          f"{'Nodes':>14} {'Infosets':>10}  {'LBR-' + str(args.lbr_depth)}")
    print("-" * 110)
    for hand_sizes, sampler, secs, lbr_secs, n_info, n_nodes, lbr in results:
        print(f"{str(hand_sizes):<8} {sampler:<8} {secs:>8.1f} "
              f"{args.iter/secs:>7.0f} "
              f"{n_nodes:>14,} {n_info:>10,}  {_fmt_lbr_cell(lbr)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
