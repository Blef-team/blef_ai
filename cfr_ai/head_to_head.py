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
                    strategies[f"{hand_size_str}-{last_bet_str}-{row[0]}"] = row[1]
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
                    return decode_probabilities(row[1])
    strategy = np.zeros(num_possible_actions)
    strategy[-1] = 1.0
    return strategy

def get_strategy(current_model, key, hand_sizes, possible_actions):
    """Helper to get a strategy from pre-loaded dict or from file."""
    strategy = None
    num_actions = len(possible_actions)
    if current_model.get('strategies') is not None:  # Check if pre-loaded
        encoded_strategy = current_model['strategies'].get(key)
        if encoded_strategy:
            strategy = decode_probabilities(encoded_strategy)
    else:  # Fallback to file loading
        strategy = load_strategy_from_files(current_model['folder'], key, hand_sizes, num_actions)

    if strategy is None:
        strategy = np.zeros(num_actions)
        strategy[-1] = 1.0
    if len(strategy) < num_actions:
        padded = np.zeros(num_actions)
        padded[:len(strategy)] = strategy
        strategy = padded
    if sum(strategy) == 0:
        strategy[-1] = 1.0
    return strategy / sum(strategy)

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
    
    strategy = get_strategy(current_model, key, hand_sizes, possible_actions)

    node_ev = 0.0
    next_player_idx = (active_player_idx + 1) % 2
    for i, action in enumerate(possible_actions):
        if strategy[i] > 0:
            action_ev = -get_h2h_expected_value(models, hands, history + [action], next_player_idx, hand_sizes, existence_array)
            node_ev += strategy[i] * action_ev
    return node_ev

def run_mc_playout(models: tuple, hands: tuple, starting_player_idx: int, hand_sizes: list, existence_array: np.ndarray) -> int:
    """
    (New Fast Playout) Simulates a single game path based on sampling actions.
    Returns +1 for a win for the starting player, -1 for a loss.
    """
    history = []
    active_player_idx = 0

    while not Game.check_finish(history):
        current_model = models[active_player_idx]
        my_hand = hands[active_player_idx]
        
        possible_actions = current_model['module'].get_possible_actions(history)
        hand_abstractions = current_model['module'].get_hand_abstraction(my_hand, hand_sizes)
        key = current_model['module'].make_key(my_hand, hand_abstractions, history)
        
        chosen_action = random.choices(possible_actions, weights=get_strategy(current_model, key, hand_sizes, possible_actions), k=1)[0]
        
        history.append(chosen_action)
        active_player_idx = (active_player_idx + 1) % 2

    # Right now, checked player = active player
    payoff_for_checked_player = 1 if existence_array[history[-2]] else -1

    if active_player_idx == 0:
        return payoff_for_checked_player
    else:
        return -payoff_for_checked_player

def main():
    parser = argparse.ArgumentParser(description="Run a head-to-head comparison between two Blef CFR AIs.")
    parser.add_argument("--hand-sizes", nargs=2, type=int, required=True, help="The number of cards for each player (e.g., 1 2).")
    parser.add_argument("--model1-folder", type=str, required=True, help="Path to the folder for the first AI model (Model A).")
    parser.add_argument("--model2-folder", type=str, required=True, help="Path to the folder for the second AI model (Model B).")
    parser.add_argument("--num-deals", type=int, default=1000, help="Number of random card deals to simulate.")
    parser.add_argument("--preload-strategies", action="store_true", help="Pre-load strategies for faster, memory-intensive evaluation.")
    parser.add_argument("--monte-carlo", action="store_true", help="Use fast Monte Carlo playouts instead of full tree traversal.")
    args = parser.parse_args()

    print("Loading AI models and their abstraction logic...")
    models_list = []
    for i, folder in enumerate([args.model1_folder, args.model2_folder]):
        try:
            module = load_abstractions_and_strategy_logic(folder)
            models_list.append({'folder': folder, 'module': module, 'name': f"Model {'A' if i == 0 else 'B'}", 'strategies': None})
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    model_A, model_B = models_list[0], models_list[1]
    hand_sizes = sorted(args.hand_sizes)

    if args.preload_strategies:
        model_A['strategies'] = preload_all_strategies(model_A['folder'], hand_sizes)
        model_B['strategies'] = preload_all_strategies(model_B['folder'], hand_sizes)

    if args.monte_carlo:
        print(f"\nGenerating {args.num_deals} random deals (with replacement)...")
        sampled_deals = [Game.deal_cards(hand_sizes) for _ in range(args.num_deals)]
    else:
        total_deals = math.comb(24, hand_sizes[0]) * math.comb(24 - hand_sizes[0], hand_sizes[1])
        num_to_sample = min(args.num_deals, total_deals)
        print(f"\nMemory-efficiently sampling {num_to_sample} deals from {total_deals} combinations...")
        deals_iterator = Game.hand_combinations(hand_sizes)
        reservoir = list(itertools.islice(deals_iterator, num_to_sample))
        pbar = tqdm(deals_iterator, total=total_deals, initial=len(reservoir), desc="Sampling deals")
        for i, deal in enumerate(pbar, start=len(reservoir) + 1):
            j = random.randint(0, i - 1)
            if j < num_to_sample: reservoir[j] = deal
        sampled_deals = reservoir

    print(f"\nStarting head-to-head evaluation for {len(sampled_deals)} deals...")
    payoffs = {'A_starts_P0': 0.0, 'B_starts_P0': 0.0, 'A_starts_P1': 0.0, 'B_starts_P1': 0.0}

    for hands in tqdm(sampled_deals, desc="Simulating Deals"):
        p0_hand, p1_hand = hands[0], hands[1]
        existence_array = Game.precompute_set_existence(hands)
        eval_func = run_mc_playout if args.monte_carlo else get_h2h_expected_value

        # Case 1: Player with hand_sizes[0] starts
        payoffs['A_starts_P0'] += eval_func((model_A, model_B), (p0_hand, p1_hand), 0, hand_sizes, existence_array)
        payoffs['B_starts_P0'] += eval_func((model_B, model_A), (p0_hand, p1_hand), 0, hand_sizes, existence_array)

        # Case 2: Player with hand_sizes[1] starts
        if hand_sizes[0] != hand_sizes[1]:
            payoffs['A_starts_P1'] += eval_func((model_A, model_B), (p1_hand, p0_hand), 0, [hand_sizes[1], hand_sizes[0]], existence_array)
            payoffs['B_starts_P1'] += eval_func((model_B, model_A), (p1_hand, p0_hand), 0, [hand_sizes[1], hand_sizes[0]], existence_array)

    num_to_sample = len(sampled_deals)
    normalised_advantage_P0 = (payoffs['B_starts_P0'] - payoffs['A_starts_P0']) / num_to_sample / 2.0

    print("\n" + "="*80)
    print("--- Head-to-Head Performance Difference (Model B vs. Model A) ---")
    print(f"\nNormalised (-1 to 1) advantage when player with {hand_sizes[0]} cards starts: {normalised_advantage_P0:+.5f}")

    if hand_sizes[0] != hand_sizes[1]:
        normalised_advantage_P1 = (payoffs['B_starts_P1'] - payoffs['A_starts_P1']) / num_to_sample / 2.0
        print(f"Normalized advantage when player with {hand_sizes[1]} cards starts: {normalised_advantage_P1:+.5f}")
    print("="*80)


if __name__ == "__main__":
    main()
