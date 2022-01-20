import numpy as np
import sys
import argparse
from cfr_ai.trainer import *
from cfr_ai.exploitability import *
from cfr_ai.encoding import encode_probabilities
from datetime import datetime
import csv

CLI = argparse.ArgumentParser()
CLI.add_argument("--num_iterations", type=int, default=100000)
CLI.add_argument("--NumCards", nargs=2, type=int, default=[1, 1])
args = CLI.parse_args()

class Params(object):
    NumCards = args.NumCards

np.set_printoptions(precision=3, floatmode='fixed', suppress=True)
cfr_trainer = Trainer(BlefCards, Params)

print(f"\nRunning {args.num_iterations} iterations of Blef CFR")
start_time = datetime.now()
util = cfr_trainer.train(args.num_iterations)
print("Time spent: ", datetime.now() - start_time)
print(f"Approximate expected value for starting player: {(util / args.num_iterations):.3f}\n")

print(f"Strategy map memory size: {sum([sys.getsizeof(x) for x in cfr_trainer.infoset_map.items()]) / 1024 / 1024} MB")
cfr_strategy = [{"k": k, "v": encode_probabilities(v.get_final_strategy())} for k,v in cfr_trainer.infoset_map.items()]
with open('cfr_ai/outputs/' + "_".join(str(x) for x in args.NumCards) + '.csv', 'w', newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['k', 'v'])
    csv_row = writer.writeheader()
    for data in cfr_strategy:
        csv_row = writer.writerow(data)

print(f"\nComputing exploitability")
start_time = datetime.now()
print(f"\nExploitability: " + str(get_exploitability(cfr_trainer.infoset_map, BlefCards, Params)))
print("Time spent: ", datetime.now() - start_time)
