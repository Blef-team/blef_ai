import argparse
from cfr_ai.trainer import *
from cfr_ai.exploitability import *
from cfr_ai.encoding import encode_probabilities, clear_lows
import csv, os, psutil
from datetime import datetime
import time

def main():
    CLI = argparse.ArgumentParser(description="Train a CFR AI for Blef.")
    CLI.add_argument("--hand-sizes", nargs=2, type=int, required=True, help="The number of cards per player, sorted ascending.")
    CLI.add_argument("--num-iterations", type=int, default=5000000, help="Total number of training iterations. Default: 5,000,000")
    CLI.add_argument("--pruning-range", nargs=2, type=int, default=[-20, -22], help="Regret pruning threshold and minimum regret value. Default: -20 -22")
    CLI.add_argument("--penalty", type=float, default=0.0, help="Penalty for betting instead of checking. Default: 0.0")
    CLI.add_argument("--log-points", type=int, default=25, help="Number of intervals for logging utility values. Default: 25")
    CLI.add_argument("--get-exploitability", action=argparse.BooleanOptionalAction, help="Flag to run the (potentially slow) exploitability calculation after training.")
    CLI.add_argument("--no-save", action="store_false", dest="save", help="Flag to disable recording any outputs.")
    CLI.set_defaults(save=True)
    args = CLI.parse_args()

    cfr_trainer = Trainer(args.hand_sizes, args.pruning_range, args.penalty, args.log_points)

    start_time = time.time()
    util0, util1, utility_log = cfr_trainer.train(args.num_iterations)
    duration_seconds = time.time() - start_time
    training_duration = time.strftime('%H:%M', time.gmtime(duration_seconds))

    exploitability_log = {}
    if args.get_exploitability:
        cfr_strategy = {k: v.get_final_strategy() for k,v in cfr_trainer.infoset_map.items()}
        print(f"\nComputing exploitability")
        exploitability_p0 = get_exploitability(cfr_strategy, args.hand_sizes, 0)
        exploitability_log['Exploitability when player 0 starts'] = exploitability_p0
        print(f"\nExploitability when player 0 starts: {exploitability_p0}")
        if args.hand_sizes[0] != args.hand_sizes[1]:
            exploitability_p1 = get_exploitability(cfr_strategy, args.hand_sizes, 1)
            exploitability_log['Exploitability when player 1 starts'] = exploitability_p1
            print(f"\nExploitability when player 1 starts: {exploitability_p1}")

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
            csv_row = writer.writerow({"k": "Pruning threshold", "v": args.pruning_range[0]})
            csv_row = writer.writerow({"k": "Minimum regret", "v": args.pruning_range[1]})
            csv_row = writer.writerow({"k": "Penalty", "v": args.penalty})
            csv_row = writer.writerow({"k": "Nodes touched", "v": cfr_trainer.nodes_touched})
            csv_row = writer.writerow({"k": "Explored infosets", "v": len(cfr_trainer.infoset_map)})
            csv_row = writer.writerow({"k": "Non-checking infosets", "v": meaningful_policies})
            csv_row = writer.writerow({"k": "RAM taken (MB)", "v": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024})
            csv_row = writer.writerow({"k": "Player 1 game value", "v": util0})
            csv_row = writer.writerow({"k": "Player 2 game value", "v": util1})
            writer.writerow({"k": "--- Utility Log ---", "v": ""})
            for log_key, log_value in utility_log.items():
                writer.writerow({"k": log_key, "v": log_value})
            writer.writerow({"k": "--- Exploitability ---", "v": ""})
            for log_key, log_value in exploitability_log.items():
                writer.writerow({"k": log_key, "v": f"{log_value:.6f}"})

if __name__ == "__main__":
    main()
