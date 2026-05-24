"""Sanity check: directly look for any sub-1% probabilities in diagnostic files."""
import argparse
import csv
import os
from cfr_ai.encoding import decode_probabilities


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hand-sizes", nargs=2, type=int, required=True)
    p.add_argument("--max-show", type=int, default=10)
    args = p.parse_args()
    hand_sizes = sorted(args.hand_sizes)
    setup_dir = os.path.join("cfr_ai", "outputs", "_".join(str(x) for x in hand_sizes))

    found = 0
    rows_scanned = 0
    rows_with_sub1 = 0
    for hand_size in sorted(set(hand_sizes)):
        diag_dir = os.path.join(setup_dir, f"{hand_size}_diagnostic")
        if not os.path.isdir(diag_dir):
            continue
        for fname in sorted(os.listdir(diag_dir)):
            if not fname.endswith(".csv"):
                continue
            with open(os.path.join(diag_dir, fname), "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows_scanned += 1
                    if not row.get("v"):
                        continue
                    probs = decode_probabilities(row["v"])
                    sub1 = [p for p in probs if 0 < p < 0.01]
                    if sub1:
                        rows_with_sub1 += 1
                        if found < args.max_show:
                            tt = row.get("times_touched", "?")
                            print(f"  {fname} k={row['k']} tt={tt}: sub-1% probs = {[f'{p*100:.3f}%' for p in sub1]}")
                            found += 1
                            if found == args.max_show:
                                print(f"  ... (more not shown)")
    print(f"\nScanned {rows_scanned} rows; {rows_with_sub1} had any probability in [0.04%, 1%).")


if __name__ == "__main__":
    main()
