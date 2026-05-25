import argparse
from cfr_ai.trainer import *
from cfr_ai.encoding import encode_probabilities, clear_lows
from cfr_ai.lbr import lbr_exploitability
import csv, os, psutil
from datetime import datetime
import time

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
        files = {}
        writers = {}
        meaningful_policies = 0
        for hand_size in set(args.hand_sizes):
            os.makedirs('cfr_ai/outputs/' + "_".join(str(x) for x in args.hand_sizes) + '/' + str(hand_size), exist_ok=True)
            os.makedirs('cfr_ai/outputs/' + "_".join(str(x) for x in args.hand_sizes) + '/' + str(hand_size) + '_diagnostic/', exist_ok=True)
            for last_bet in range(89):
                key = str(hand_size) + '-' + str(last_bet)
                files[key] = open('cfr_ai/outputs/' + "_".join(str(x) for x in args.hand_sizes) + '/' + str(hand_size) + '/' + str(last_bet) + '.csv', 'w', newline="")
                writers[key] = csv.DictWriter(files[key], fieldnames=['k', 'v'])
                header = writers[key].writeheader()
                files[key + '-D'] = open('cfr_ai/outputs/' + "_".join(str(x) for x in args.hand_sizes) + '/' + str(hand_size) + '_diagnostic/' + str(last_bet) + '.csv', 'w', newline="")
                writers[key + '-D'] = csv.DictWriter(files[key + '-D'], fieldnames=['k', 'first_touched', 'last_touched', 'times_touched', 'v'])
                header = writers[key + '-D'].writeheader()
        for k,v in cfr_trainer.infoset_map.items():
            policy = v.get_final_strategy()
            cleaned_policy = clear_lows(policy)
            split_key = k.split('-')
            if cleaned_policy[-1] < 1.0:
                csv_row = writers[split_key[0] + '-' + split_key[1]].writerow({"k": '-'.join(split_key[2:]), "v": encode_probabilities(cleaned_policy)})
                meaningful_policies += 1
            csv_row = writers[split_key[0] + '-' + split_key[1] + '-D'].writerow({"k": '-'.join(split_key[2:]), "v": encode_probabilities(policy), "first_touched": v.first_touched, "last_touched": v.last_touched, "times_touched": v.times_touched})
        for k,v in files.items():
            v.close()
        with open('cfr_ai/outputs/' + "_".join(str(x) for x in args.hand_sizes) + '/metadata.csv', 'w', newline="") as csvfile:
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
