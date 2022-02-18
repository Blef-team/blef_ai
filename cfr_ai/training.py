import argparse
from cfr_ai.trainer import *
from cfr_ai.exploitability import *
from cfr_ai.encoding import encode_probabilities
from datetime import datetime
import csv

CLI = argparse.ArgumentParser()
CLI.add_argument("--num_iterations", type=int, default=300000)
CLI.add_argument("--NumCards", nargs=2, type=int, default=[1, 1])
args = CLI.parse_args()

class Params(object):
    NumCards = args.NumCards

cfr_trainer = Trainer(BlefCards, Params)
util0, util1 = cfr_trainer.train(args.num_iterations)
print(f"Expected values, depenging on who starts, are {util0:.3f} and {util1:.3f}")

cfr_strategy = {k: v.get_final_strategy() for k,v in cfr_trainer.infoset_map.items()}

with open('cfr_ai/outputs/' + "_".join(str(x) for x in args.NumCards) + '.csv', 'w', newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['k', 'v'])
    csv_row = writer.writeheader()
    for k,v in cfr_strategy:
        if v[-1] < 1.0:
            csv_row = writer.writerow({"k": k, "v": encode_probabilities(v)})

print(f"\nComputing exploitability")
start_time = datetime.now()
print(f"\nExploitability when player 0 starts: " + str(get_exploitability(cfr_strategy, BlefCards, Params, 0)))
if Params.NumCards[0] != Params.NumCards[1]:
    print(f"\nExploitability when player 1 starts: " + str(get_exploitability(cfr_strategy, BlefCards, Params, 1)))
print("Time spent: ", datetime.now() - start_time)
