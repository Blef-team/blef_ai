import argparse
from cfr_ai.trainer import *
from cfr_ai.encoding import clear_lows
from cfr_ai.lbr import lbr_exploitability
import csv, os, psutil
from datetime import datetime
import time

import numpy as np

VERSION_CODE = 'TV-NR-SS-SD-MB'


def _snapshot_exploitability(hand_sizes, min_bet, depth, n_belief, n_lbr_hand):
    """Returns a snapshot callback that computes LBR-K on the live infoset_map
    and appends to the provided log dict. Closure over `exploitability_log`."""
    exploitability_log: Dict[str, Any] = {}

    def callback(iter_num, infoset_map):
        cfr_strategy = {k: v.get_final_strategy() for k, v in infoset_map.items()}
        sps = [0] if hand_sizes[0] == hand_sizes[1] else [0, 1]
        for sp in sps:
            r = lbr_exploitability(
                hand_sizes, sp, cfr_strategy, min_bet, depth,
                n_belief_samples=n_belief, n_lbr_hand_samples=n_lbr_hand, seed=42,
            )
            tqdm.write(
                f"  Iter {iter_num} LBR-{depth} sp={sp}: "
                f"{r['expl']*100:+.3f}% ± {r['se_worst']*100:.3f}pp (K={r['K_lbr_hand']})"
            )
            exploitability_log[f"LBR-{depth} expl sp={sp} at Iter {iter_num}"] = (
                f"{r['expl']*100:+.4f}% ± {r['se_worst']*100:.4f}pp"
            )

    return callback, exploitability_log


def _save_trainer_outputs(cfr_trainer, hand_sizes, min_bet, setup_dir,
                          save_diagnostic: bool = True) -> int:
    """Convert the Python `Trainer.infoset_map` into the on-disk NPZ
    format (strategy.npz + optional diagnostic.npz), matching what
    `cfr_ai.training_numba` writes. Returns the count of meaningful
    (non-check-100%) policies.

    The Python trainer keys infosets by `make_key` strings like
    `hs-lb-(h_m1-h_m2-)abs`. We parse those into the same composite int64
    keys the JIT uses, so the resulting NPZ is bit-equivalent to one
    produced by `cfr_ai.training_numba`.
    """
    from numba.typed import Dict as NbDict
    from numba import types as nb_types
    from cfr_ai.lbr_numba import FlatStrategy, _split_suffix
    from cfr_ai.strategy_io import save_strategy, save_diagnostic as save_diag
    from cfr_ai.trainer_numba import (
        LAST_BET_SHIFT, H_M1_SHIFT, H_M2_SHIFT, ABS_ID_SHIFT,
    )

    os.makedirs(setup_dir, exist_ok=True)

    # First pass: collect (composite_key, lower, upper, strategy, regrets,
    # strategy_sum, first_touched, last_touched, times_touched), interning
    # abstraction strings as we go.
    abs_str_to_id: Dict[str, int] = {}

    keys: List[int] = []
    lowers: List[int] = []
    uppers: List[int] = []
    strategies: List[np.ndarray] = []
    # Diagnostic fields (only populated if save_diagnostic).
    regrets: List[np.ndarray] = []
    strategy_sums: List[np.ndarray] = []
    first_touched: List[int] = []
    last_touched: List[int] = []
    times_touched: List[int] = []

    meaningful_policies = 0
    for k, v in cfr_trainer.infoset_map.items():
        policy = v.get_final_strategy()
        cleaned = clear_lows(policy)
        parts = k.split('-')
        hand_size = int(parts[0])
        last_bet = int(parts[1])
        suffix = '-'.join(parts[2:])
        h_m1_id, h_m2_id, abs_str = _split_suffix(suffix)
        abs_id = abs_str_to_id.get(abs_str)
        if abs_id is None:
            abs_id = len(abs_str_to_id)
            abs_str_to_id[abs_str] = abs_id

        comp = (hand_size
                | (last_bet << LAST_BET_SHIFT)
                | (h_m1_id << H_M1_SHIFT)
                | (h_m2_id << H_M2_SHIFT)
                | (abs_id << ABS_ID_SHIFT))

        if last_bet == 88 or last_bet < min_bet:
            lo, hi = min_bet, 87
        else:
            lo, hi = last_bet + 1, 88
        width = hi - lo + 1

        # Drop check-100% rows from the deployed strategy (the JIT defaults
        # to that on missing keys). Diagnostic always keeps them.
        if cleaned[-1] < 1.0:
            row_probs = np.zeros(89, dtype=np.float32)
            row_probs[lo:hi + 1] = cleaned[:width].astype(np.float32)
            keys.append(comp)
            lowers.append(lo)
            uppers.append(hi)
            strategies.append(row_probs)
            meaningful_policies += 1

        if save_diagnostic:
            row_regrets = np.zeros(89, dtype=np.float32)
            row_regrets[lo:hi + 1] = (
                np.asarray(v.regrets, dtype=np.float32)[:width]
            )
            row_ssum = np.zeros(89, dtype=np.float32)
            row_ssum[lo:hi + 1] = (
                np.asarray(v.strategy_sum, dtype=np.float32)[:width]
            )
            regrets.append(row_regrets)
            strategy_sums.append(row_ssum)
            first_touched.append(int(v.first_touched))
            last_touched.append(int(v.last_touched))
            times_touched.append(int(v.times_touched))
            # Diagnostic uses the same composite key; we store a parallel
            # `diag_keys` array. For rows whose strategy was dropped above,
            # we still record the diag info (so the lists are 1:1 with the
            # full infoset_map).

    # Build a FlatStrategy for the deployment file.
    n = len(keys)
    keys_arr = np.array(keys, dtype=np.int64)
    lower_arr = np.array(lowers, dtype=np.int16)
    upper_arr = np.array(uppers, dtype=np.int16)
    probs_arr = (np.stack(strategies)
                 if strategies else np.zeros((0, 89), dtype=np.float32))
    order = np.argsort(keys_arr, kind="stable")
    keys_arr = keys_arr[order]
    lower_arr = lower_arr[order]
    upper_arr = upper_arr[order]
    probs_arr = probs_arr[order]
    key_to_row = NbDict.empty(key_type=nb_types.int64, value_type=nb_types.int64)
    for i in range(n):
        key_to_row[np.int64(keys_arr[i])] = np.int64(i)
    fs = FlatStrategy(
        key_to_row=key_to_row, strategy=probs_arr,
        lower_action=lower_arr, upper_action=upper_arr,
        abs_str_to_id=abs_str_to_id, min_bet=int(min_bet),
    )
    save_strategy(fs, setup_dir, compressed=True)

    if save_diagnostic and times_touched:
        # Diagnostic has ALL infosets in the trainer (including check-only).
        # Build separate arrays in trainer iteration order, then sort.
        diag_keys: List[int] = []
        for k, _ in cfr_trainer.infoset_map.items():
            parts = k.split('-')
            hand_size = int(parts[0])
            last_bet = int(parts[1])
            suffix = '-'.join(parts[2:])
            h_m1_id, h_m2_id, abs_str = _split_suffix(suffix)
            abs_id = abs_str_to_id[abs_str]
            comp = (hand_size
                    | (last_bet << LAST_BET_SHIFT)
                    | (h_m1_id << H_M1_SHIFT)
                    | (h_m2_id << H_M2_SHIFT)
                    | (abs_id << ABS_ID_SHIFT))
            diag_keys.append(comp)
        diag_keys_arr = np.array(diag_keys, dtype=np.int64)
        diag_lower = np.array([
            min_bet if (lb := int(k.split('-')[1])) == 88 or lb < min_bet else lb + 1
            for k in cfr_trainer.infoset_map.keys()
        ], dtype=np.int16)
        diag_upper = np.array([
            87 if (lb := int(k.split('-')[1])) == 88 or lb < min_bet else 88
            for k in cfr_trainer.infoset_map.keys()
        ], dtype=np.int16)
        regrets_arr = np.stack(regrets)
        ssum_arr = np.stack(strategy_sums)
        first_arr = np.array(first_touched, dtype=np.int32)
        last_arr = np.array(last_touched, dtype=np.int32)
        times_arr = np.array(times_touched, dtype=np.int32)
        order = np.argsort(diag_keys_arr, kind="stable")
        save_diag(
            setup_dir,
            keys=diag_keys_arr[order],
            lower=diag_lower[order],
            upper=diag_upper[order],
            regrets=regrets_arr[order],
            strategy_sum=ssum_arr[order],
            first_touched=first_arr[order],
            last_touched=last_arr[order],
            times_touched=times_arr[order],
            compressed=True,
        )

    return meaningful_policies


def main() -> None:
    CLI = argparse.ArgumentParser(description="Train a CFR AI for Blef.")
    CLI.add_argument("--hand-sizes", nargs=2, type=int, required=True, help="The number of cards per player, sorted ascending.")
    CLI.add_argument("--num-iterations", type=int, default=5000000, help="Total number of training iterations. Default: 5,000,000")
    CLI.add_argument("--min-bet", type=int, default=0, help="The lowest bet acknowledged or made by the AI")
    CLI.add_argument("--pruning-range", nargs=2, type=int, default=[-20, -22], help="Regret pruning threshold and minimum regret value. Default: -20 -22")
    CLI.add_argument("--penalty", type=float, default=0.0, help="Penalty for betting instead of checking. Default: 0.0")
    CLI.add_argument("--log-points", type=int, default=25, help="Number of intervals for logging utility values. Default: 25")
    CLI.add_argument(
        "--get-exploitability", action="store_true",
        help="Compute LBR-1 exploitability at several points during training "
             "and record them alongside utility in metadata.csv.",
    )
    CLI.add_argument("--exploitability-points", type=int, default=10,
                     help="Number of evenly-spaced points at which to compute exploitability (default 10).")
    CLI.add_argument("--exploitability-depth", type=int, default=1)
    CLI.add_argument("--exploitability-n-belief", type=int, default=300)
    CLI.add_argument("--exploitability-n-lbr-hand", type=int, default=200)
    CLI.add_argument("--algorithm", choices=['es', 'cfr_plus', 'dcfr'], default='es',
                     help="Regret-matching algorithm: 'es' (current default), 'cfr_plus', or 'dcfr' (α=1.5, β=0, γ=1).")
    CLI.add_argument("--no-save", action="store_false", dest="save", help="Flag to disable recording any outputs.")
    CLI.set_defaults(save=True)
    args = CLI.parse_args()

    cfr_trainer = Trainer(args.hand_sizes, args.min_bet, args.pruning_range, args.penalty, args.log_points, algorithm=args.algorithm)

    snapshot_cb, exploitability_log = (None, {})
    snapshot_every = 1
    if args.get_exploitability:
        snapshot_cb, exploitability_log = _snapshot_exploitability(
            args.hand_sizes, args.min_bet, args.exploitability_depth,
            args.exploitability_n_belief, args.exploitability_n_lbr_hand,
        )
        # If exploitability-points < log-points, snapshot only every Nth log point.
        snapshot_every = max(1, args.log_points // args.exploitability_points)

    start_time = time.time()
    util0, util1, utility_log = cfr_trainer.train(
        args.num_iterations, snapshot_callback=snapshot_cb,
        snapshot_every_log_points=snapshot_every,
    )
    duration_seconds = time.time() - start_time
    training_duration = time.strftime('%H:%M', time.gmtime(duration_seconds))

    if args.save:
        setup_dir = os.path.join(
            'cfr_ai', 'outputs', "_".join(str(x) for x in args.hand_sizes))
        meaningful_policies = _save_trainer_outputs(
            cfr_trainer, args.hand_sizes, args.min_bet, setup_dir,
            save_diagnostic=True,
        )
        with open(os.path.join(setup_dir, 'metadata.csv'), 'w', newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['k', 'v'])
            csv_row = writer.writerow({"k": "Time finished", "v": datetime.now().strftime("%Y-%m-%d, %H:%M:%S")})
            csv_row = writer.writerow({"k": "Training duration", "v": training_duration})
            csv_row = writer.writerow({"k": "Iterations", "v": args.num_iterations})
            csv_row = writer.writerow({"k": "Minimum bet", "v": args.min_bet})
            csv_row = writer.writerow({"k": "Pruning threshold", "v": args.pruning_range[0]})
            csv_row = writer.writerow({"k": "Minimum regret", "v": args.pruning_range[1]})
            csv_row = writer.writerow({"k": "Penalty", "v": args.penalty})
            csv_row = writer.writerow({"k": "Nodes touched", "v": cfr_trainer.nodes_touched})
            csv_row = writer.writerow({"k": "Explored infosets", "v": len(cfr_trainer.infoset_map)})
            csv_row = writer.writerow({"k": "Non-checking infosets", "v": meaningful_policies})
            csv_row = writer.writerow({"k": "RAM taken (MB)", "v": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024})
            csv_row = writer.writerow({"k": "Player 1 game value", "v": util0})
            csv_row = writer.writerow({"k": "Player 2 game value", "v": util1})
            csv_row = writer.writerow({"k": "Version code", "v": VERSION_CODE})
            writer.writerow({"k": "--- Utility Log ---", "v": ""})
            for log_key, log_value in utility_log.items():
                writer.writerow({"k": log_key, "v": log_value})
            if exploitability_log:
                writer.writerow({"k": "--- Exploitability Log ---", "v": ""})
                for log_key, log_value in exploitability_log.items():
                    writer.writerow({"k": log_key, "v": log_value})

if __name__ == "__main__":
    main()
