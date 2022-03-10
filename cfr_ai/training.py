import argparse
from cfr_ai.trainer import *
from cfr_ai.exploitability import *
from cfr_ai.encoding import encode_probabilities
import csv, os, psutil
from datetime import datetime

CLI = argparse.ArgumentParser()
CLI.add_argument("--num_iterations", type=int, default=50000)
CLI.add_argument("--hand_sizes", nargs=2, type=int, default=[1, 1])
CLI.add_argument("--no_save", action=argparse.BooleanOptionalAction)
CLI.add_argument("--get_exploitability", action=argparse.BooleanOptionalAction)
args = CLI.parse_args()

cfr_trainer = Trainer(args.hand_sizes)
util0, util1 = cfr_trainer.train(args.num_iterations)

if not args.no_save:
    cfr_strategy = {k: v.get_final_strategy() for k,v in cfr_trainer.infoset_map.items()}
    meaningful_policies = {k: v for k,v in cfr_strategy.items() if v[-1] < 1.0}
    for hand_size in set(args.hand_sizes):
        for last_bet in range(89):
            with open('cfr_ai/outputs/' + "_".join(str(x) for x in args.hand_sizes) + '/' + str(hand_size) + '/' + str(last_bet) + '.csv', 'w', newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['k', 'v'])
                csv_row = writer.writeheader()
                for k,v in meaningful_policies.items():
                    split_key = k.split('-')
                    if split_key[0] == str(hand_size) and split_key[1] == str(last_bet):
                        csv_row = writer.writerow({"k": '-'.join(split_key[2:]), "v": encode_probabilities(v)})
    with open('cfr_ai/outputs/' + "_".join(str(x) for x in args.hand_sizes) + '/metadata.csv', 'w', newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['k', 'v'])
        csv_row = writer.writerow({"k": "Time finished", "v": datetime.now().strftime("%Y-%m-%d, %H:%M:%S")})
        csv_row = writer.writerow({"k": "Iterations", "v": args.num_iterations})
        csv_row = writer.writerow({"k": "Nodes touched", "v": cfr_trainer.nodes_touched})
        csv_row = writer.writerow({"k": "Explored infosets", "v": len(cfr_strategy)})
        csv_row = writer.writerow({"k": "Non-checking infosets", "v": len(meaningful_policies)})
        csv_row = writer.writerow({"k": "RAM taken (MB)", "v": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024})
        csv_row = writer.writerow({"k": "Player 1 game value", "v": util0})
        csv_row = writer.writerow({"k": "Player 2 game value", "v": util1})

if args.get_exploitability:
    cfr_strategy = {k: v.get_final_strategy() for k,v in cfr_trainer.infoset_map.items()}
    print(f"\nComputing exploitability")
    print(f"\nExploitability when player 0 starts: " + str(get_exploitability(cfr_strategy, args.hand_sizes, 0)))
    if args.hand_sizes[0] != args.hand_sizes[1]:
        print(f"\nExploitability when player 1 starts: " + str(get_exploitability(cfr_strategy, args.hand_sizes, 1)))
