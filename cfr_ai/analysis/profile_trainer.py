"""Profile harness for trainer.py.

Splits per-iter wall time into:
  - deal_cards
  - precompute_set_existence
  - _build_abs_ids_for_iter (Python-side abstraction string interning)
  - _traverse_jit (the JIT recursive traversal)
  - book-keeping (utility accumulation, log-point arithmetic, staircase discount)

Plus a cProfile pass that exposes any unexpected Python-side hotspots.

Usage:
    python -m cfr_ai.analysis.profile_trainer --hand-sizes 1 2 --iter 10000
"""

import argparse
import cProfile
import io
import pstats
import random
import time
from collections import defaultdict

import numpy as np
import psutil

import cfr_ai.game as game_mod
from cfr_ai.trainer import Trainer, _seed_numba, _traverse_jit, _HISTORY_CODE_ID


class TimedTrainer(Trainer):
    """Trainer that records cumulative time in each per-iter phase."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timings = defaultdict(float)

    def train(self, num_iterations):
        utils = np.zeros(2, dtype=np.float64)
        hand_sizes_arr = np.asarray(self.hand_sizes, dtype=np.int64)

        for i in range(num_iterations):
            t0 = time.perf_counter()
            # --- staircase discount
            n = int(self.state[0])
            if n > 0:
                if i == int(num_iterations * 0.3):
                    self.strategy_sum[:n] *= np.float32(0.02)
                for t in range(4, 10):
                    if i == int(t * num_iterations / 10):
                        self.strategy_sum[:n] *= np.float32(t / (t + 1))
            t1 = time.perf_counter(); self.timings["staircase"] += t1 - t0

            prune_feast = bool(int(i / 4) % 20 == 0)
            traverser = int(i / 2) % 2
            starting_player = i % 2

            # --- deal
            hands = game_mod.Game.deal_cards(self.hand_sizes)
            t2 = time.perf_counter(); self.timings["deal"] += t2 - t1

            # --- existence
            existence_array = game_mod.Game.precompute_set_existence(hands)
            t3 = time.perf_counter(); self.timings["existence"] += t3 - t2

            # --- abstraction interning
            abs_ids = self._build_abs_ids_for_iter(hands)
            t4 = time.perf_counter(); self.timings["abs_ids"] += t4 - t3

            self.state[1] = 0
            # --- the JIT traversal
            v = _traverse_jit(
                self._history_buf, 0, 1.0,
                int(starting_player), int(traverser), prune_feast,
                existence_array, int(i),
                self.regrets, self.strategy_sum,
                self.lower_action, self.upper_action,
                self.last_touched, self.temporary_value,
                self.first_touched, self.times_touched,
                self.key_to_row, self.state, int(self.capacity),
                abs_ids, hand_sizes_arr, _HISTORY_CODE_ID,
                int(self.min_bet), float(self.pruning_threshold),
                float(self.min_regret), float(self.penalty),
                self._strategy_buf, self._cf_buf,
                bool(self.use_temp_value),
            )
            t5 = time.perf_counter(); self.timings["jit_traverse"] += t5 - t4

            utils[starting_player] += v
            t6 = time.perf_counter(); self.timings["bookkeeping"] += t6 - t5

        return float(utils[0]), float(utils[1]), {}


def warm_jit(dtype):
    _seed_numba(0)
    game_mod.rng = np.random.default_rng(0)
    w = Trainer([1, 1], 0, [-20, -22], 0.0, 0,
                     initial_capacity=2000,
                     numba_seed=0, regret_dtype=dtype)
    w.train(50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand-sizes", nargs=2, type=int, default=[1, 2])
    ap.add_argument("--iter", type=int, default=10000)
    ap.add_argument("--capacity", type=int, default=40_000)
    ap.add_argument("--dtype", choices=["fp32", "fp64"], default="fp32")
    ap.add_argument("--cprofile", action="store_true",
                    help="Run a cProfile pass on the train loop")
    ap.add_argument("--no-temporary-value", action="store_true",
                    help="Profile with the same-iteration value cache disabled.")
    args = ap.parse_args()

    dt = np.float32 if args.dtype == "fp32" else np.float64

    print(f"Warming JIT (dtype={args.dtype})...")
    warm_jit(dt)

    # Phase-by-phase
    print(f"\n=== Phase breakdown: {tuple(args.hand_sizes)}, {args.iter:,} iters ===")
    random.seed(42)
    game_mod.rng = np.random.default_rng(42)
    _seed_numba(42)
    t = TimedTrainer(args.hand_sizes, 0, [-20, -22], 0.0, 0,
                     initial_capacity=args.capacity,
                     numba_seed=42, regret_dtype=dt,
                     use_temp_value=not args.no_temporary_value)
    t0 = time.perf_counter()
    t.train(args.iter)
    total = time.perf_counter() - t0
    print(f"  total wall: {total:.2f} s ({args.iter/total:.0f} it/s)\n")
    print(f"  {'phase':<14} {'time (s)':>10} {'%':>6}  {'per-iter (us)':>14}")
    for ph, dt_s in sorted(t.timings.items(), key=lambda x: -x[1]):
        pct = 100 * dt_s / total
        per_iter_us = dt_s * 1e6 / args.iter
        print(f"  {ph:<14} {dt_s:>10.3f} {pct:>5.1f}%  {per_iter_us:>14.1f}")
    overhead = total - sum(t.timings.values())
    print(f"  {'(other)':<14} {overhead:>10.3f} {100*overhead/total:>5.1f}%")

    # cProfile pass — shows Python-side function call hotspots
    # (JIT'd code shows up as one frame)
    if args.cprofile:
        print(f"\n=== cProfile on Python orchestration ===")
        random.seed(42)
        game_mod.rng = np.random.default_rng(42)
        _seed_numba(42)
        prof_t = Trainer(args.hand_sizes, 0, [-20, -22], 0.0, 0,
                              initial_capacity=args.capacity,
                              numba_seed=42, regret_dtype=dt)
        pr = cProfile.Profile()
        pr.enable()
        prof_t.train(args.iter)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(25)
        print(s.getvalue())


if __name__ == "__main__":
    main()
