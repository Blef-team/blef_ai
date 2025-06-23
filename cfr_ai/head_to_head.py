import argparse
import itertools
import numpy as np
import os
import sys
import csv
import random
import importlib.util
from tqdm import tqdm
import math

from cfr_ai.game import Game
from cfr_ai.encoding import decode_probabilities

def load_abstractions_and_strategy_logic(model_folder_path: str):
    """
    Dynamically loads the information_set.py module and history.csv
    from a specific model's folder to get its unique abstraction logic.
    """
    base_paths_to_check = [model_folder_path, os.path.join(model_folder_path, 'cfr_ai')]
    history_csv_path, info_set_module_path = None, None
    for base_path in base_paths_to_check:
        if os.path.exists(os.path.join(base_path, 'history.csv')):
            history_csv_path = os.path.join(base_path, 'history.csv')
        if os.path.exists(os.path.join(base_path, 'information_set.py')):
            info_set_module_path = os.path.join(base_path, 'information_set.py')
    if not history_csv_path: raise FileNotFoundError(f"Could not find 'history.csv' in '{model_folder_path}'")
    if not info_set_module_path: raise FileNotFoundError(f"Could not find 'information_set.py' in '{model_folder_path}'")
    module_name = f"info_set_{os.path.basename(model_folder_path).replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, info_set_module_path)
    info_set_module = importlib.util.module_from_spec(spec)
    info_set_module.history_codes = np.genfromtxt(history_csv_path, delimiter=',', dtype='|U5')
    spec.loader.exec_module(info_set_module)
    return info_set_module

def preload_all_strategies(model_folder, hand_sizes_sorted):
    """
    Reads all strategy files for a model into an in-memory dictionary.
    Returns a dict mapping {info_set_key: encoded_strategy_string}.
    """
    strategies = {}
    setup_name = "_".join(str(x) for x in hand_sizes_sorted)
    outputs_path = os.path.join(model_folder, 'outputs', setup_name)
    
    if not os.path.isdir(outputs_path):
        print(f"Warning: Output directory not found at '{outputs_path}'. Cannot preload strategies.", file=sys.stderr)
        return {}

    print(f"Pre-loading strategies from {outputs_path}...")
    
    for hand_size_str in os.listdir(outputs_path):
        player_folder_path = os.path.join(outputs_path, hand_size_str)
        if not os.path.isdir(player_folder_path) or '_diagnostic' in hand_size_str:
            continue
        
        for csv_filename in os.listdir(player_folder_path):
            if not csv_filename.endswith('.csv'): continue
            last_bet_str = csv_filename[:-4]
            with open(os.path.join(player_folder_path, csv_filename), 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if not row: continue
                    k_part = row[0]
                    full_key = f"{hand_size_str}-{last_bet_str}-{k_part}"
                    strategies[full_key] = row[1]
                    
    print(f"Loaded {len(strategies)} strategy entries from {model_folder}.")
    return strategies

def load_strategy_from_files(model_folder: str, key: str, hand_sizes: list, num_possible_actions: int) -> np.ndarray:
    """
    Loads a single strategy from a CSV file.
    """
    split_key = key.split('-')
    setup_name = "_".join(str(x) for x in sorted(hand_sizes))
    strategy_file_path = os.path.join(model_folder, 'outputs', setup_name, split_key[0], f"{split_key[1]}.csv")
    if os.path.exists(strategy_file_path):
        with open(strategy_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            search_key = '-'.join(split_key[2:])
            for row in reader:
                if row and row[0] == search_key:
                    decoded_strat = decode_probabilities(row[1])
                    if len(decoded_strat) == num_possible_actions:
                         return decoded_strat
    strategy = np.zeros(num_possible_actions)
    strategy[-1] = 1.0
    return strategy

def get_h2h_expected_value(models: tuple, hands: tuple, history: list, active_player_idx: int, hand_sizes: list, existence_array: np.ndarray) -> float:
    """
    Recursively traverses the game tree to calculate expected value.
    """
    if Game.check_finish(history):
        payoff_for_checker = 1.0 if not existence_array[history[-2]] else -1.0
        return -payoff_for_checker

    current_model = models[active_player_idx]
    possible_actions = current_model['module'].get_possible_actions(history)
    my_hand = hands[active_player_idx]
    hand_abstractions = current_model['module'].get_hand_abstraction(my_hand, hand_sizes)
    key = current_model['module'].make_key(my_hand, hand_abstractions, history)

    # --- Get Strategy: either from memory or from file ---
    strategy = None
    if current_model.get('strategies') is not None: # Check if pre-loaded
        encoded_strategy = current_model['strategies'].get(key)
        if encoded_strategy:
            strategy = decode_probabilities(encoded_strategy)
    else: # Fallback to file loading
        strategy = load_strategy_from_files(current_model['folder'], key, hand_sizes, len(possible_actions))

    # Ensure strategy is valid, default to 'check' if not found
    if strategy is None:
        strategy = np.zeros(len(possible_actions))
        strategy[-1] = 1.0
    
    # Pad strategy if its length is less than the number of possible actions
    if len(strategy) < len(possible_actions):
        padded_strategy = np.zeros(len(possible_actions))
        padded_strategy[:len(strategy)] = strategy
        strategy = padded_strategy

    if sum(strategy) > 0 and not np.isclose(sum(strategy), 1.0):
        strategy /= sum(strategy)
    elif sum(strategy) == 0:
        strategy[-1] = 1.0

    node_expected_value = 0.0
    next_player_idx = (active_player_idx + 1) % 2
    for i, action in enumerate(possible_actions):
        if strategy[i] > 0:
            action_ev = -get_h2h_expected_value(models, hands, history + [action], next_player_idx, hand_sizes, existence_array)
            node_expected_value += strategy[i] * action_ev
    return node_expected_value

def main():
    parser = argparse.ArgumentParser(description="Run a head-to-head comparison between two Blef CFR AIs.")
    parser.add_argument("--hand-sizes", nargs=2, type=int, required=True, help="The number of cards for each player (e.g., 1 2).")
    parser.add_argument("--model1-folder", type=str, required=True, help="Path to the folder for the first AI model (Model A).")
    parser.add_argument("--model2-folder", type=str, required=True, help="Path to the folder for the second AI model (Model B).")
    parser.add_argument("--num-deals", type=int, default=1000, help="Number of random card deals to simulate.")
    parser.add_argument("--preload-strategies", action="store_true", help="Pre-load all strategies into memory. Faster but uses more RAM.")
    args = parser.parse_args()

    print("Loading AI models and their abstraction logic...")
    models_list = []
    for i, folder in enumerate([args.model1_folder, args.model2_folder]):
        try:
            module = load_abstractions_and_strategy_logic(folder)
            model_name_char = 'A' if i == 0 else 'B'
            models_list.append({'folder': folder, 'module': module, 'name': f"Model {model_name_char} ({os.path.basename(folder)})", 'strategies': None})
        except FileNotFoundError as e:
            print(f"Error loading files for model in '{folder}': {e}", file=sys.stderr)
            sys.exit(1)
    model_A, model_B = models_list[0], models_list[1]
    hand_sizes = sorted(args.hand_sizes)

    if args.preload_strategies:
        model_A['strategies'] = preload_all_strategies(model_A['folder'], hand_sizes)
        model_B['strategies'] = preload_all_strategies(model_B['folder'], hand_sizes)

    total_deals = math.comb(24, hand_sizes[0]) * math.comb(24 - hand_sizes[0], hand_sizes[1])
    num_to_sample = min(args.num_deals, total_deals)
    if num_to_sample == 0: sys.exit("Error: No deals to simulate.")
    
    print(f"\nMemory-efficiently sampling {num_to_sample} deals from {total_deals} possible combinations...")
    deals_iterator = Game.hand_combinations(hand_sizes)
    reservoir = list(itertools.islice(deals_iterator, num_to_sample))
    pbar = tqdm(deals_iterator, total=total_deals, initial=len(reservoir), desc="Sampling deals")
    for i, deal in enumerate(pbar, start=len(reservoir) + 1):
        j = random.randint(0, i - 1)
        if j < num_to_sample: reservoir[j] = deal
    sampled_deals = reservoir

    print(f"\nStarting head-to-head evaluation for {len(sampled_deals)} deals...")
    total_payoff_A_starts_P0, total_payoff_B_starts_P0 = 0.0, 0.0
    total_payoff_A_starts_P1, total_payoff_B_starts_P1 = 0.0, 0.0

    for hands in tqdm(sampled_deals, desc="Simulating Deals"):
        p0_hand, p1_hand = hands[0], hands[1]
        existence_array = Game.precompute_set_existence(hands)

        # Case 1: Player with hand_sizes[0] starts
        total_payoff_A_starts_P0 += get_h2h_expected_value((model_A, model_B), (p0_hand, p1_hand), [], 0, hand_sizes, existence_array)
        total_payoff_B_starts_P0 += get_h2h_expected_value((model_B, model_A), (p0_hand, p1_hand), [], 0, hand_sizes, existence_array)

        # Case 2: Player with hand_sizes[1] starts
        if hand_sizes[0] != hand_sizes[1]:
            total_payoff_A_starts_P1 += get_h2h_expected_value((model_A, model_B), (p1_hand, p0_hand), [], 0, [hand_sizes[1], hand_sizes[0]], existence_array)
            total_payoff_B_starts_P1 += get_h2h_expected_value((model_B, model_A), (p1_hand, p0_hand), [], 0, [hand_sizes[1], hand_sizes[0]], existence_array)

    avg_A_starts_P0 = total_payoff_A_starts_P0 / num_to_sample
    avg_B_starts_P0 = total_payoff_B_starts_P0 / num_to_sample
    diff_when_P0_starts = avg_B_starts_P0 - avg_A_starts_P0

    print("\n" + "="*80)
    print("--- Head-to-Head Performance Difference (Model B vs. Model A) ---")
    print(f"Comparison between '{model_A['name']}' and '{model_B['name']}'")
    print(f"Setup: {hand_sizes[0]} vs {hand_sizes[1]} cards, based on {len(sampled_deals)} deals.")
    print("\nValue represents the extra payoff Model B gets over Model A when starting.")
    print(f"\nAdvantage when player with {hand_sizes[0]} cards starts: {diff_when_P0_starts:+.6f}")
    
    if hand_sizes[0] != hand_sizes[1]:
        avg_A_starts_P1 = total_payoff_A_starts_P1 / num_to_sample
        avg_B_starts_P1 = total_payoff_B_starts_P1 / num_to_sample
        diff_when_P1_starts = avg_B_starts_P1 - avg_A_starts_P1
        print(f"Advantage when player with {hand_sizes[1]} cards starts: {diff_when_P1_starts:+.6f}")
    
    print("\n--- Detailed Average Payoffs (for the starting player) ---")
    print(f"As Starting Player with {hand_sizes[0]} cards: Model A gets {avg_A_starts_P0:.4f}, Model B gets {avg_B_starts_P0:.4f}")
    if hand_sizes[0] != hand_sizes[1]:
        print(f"As Starting Player with {hand_sizes[1]} cards: Model A gets {avg_A_starts_P1:.4f}, Model B gets {avg_B_starts_P1:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
