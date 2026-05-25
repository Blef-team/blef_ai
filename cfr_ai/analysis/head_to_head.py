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
import inspect # New import to check function arguments
from typing import List, Dict, Any, Tuple

from cfr_ai.game import Game
from cfr_ai.encoding import decode_probabilities

Model = Dict[str, Any]
Hand = Tuple[int]
Hands = Tuple[Hand, Hand]


def parse_metadata(model_folder_path: str, hand_sizes_sorted: List[int]) -> Dict[str, int]:
    """
    Parses the metadata.csv file for a specific setup.
    Returns a dictionary of found parameters.
    """
    metadata: Dict[str, int] = {'min_bet': 0} # 0 is the default
    
    setup_name = "_".join(str(x) for x in hand_sizes_sorted)
    metadata_path = os.path.join(model_folder_path, 'outputs', setup_name, 'metadata.csv')

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                key, value = row[0].strip(), row[1].strip()
                if key == 'Minimum bet':
                    try:
                        metadata['min_bet'] = int(value)
                    except (ValueError, TypeError):
                        pass
    else:
        print(f"Warning: Metadata file not found at '{metadata_path}'. Using default min_bet=0.", file=sys.stderr)
    return metadata

def load_model_components(model_folder_path: str, hand_sizes_sorted: List[int]) -> Tuple[Any, Dict[str, int]]:
    """
    Loads a model's information_set module and its metadata for a specific setup.
    """
    metadata = parse_metadata(model_folder_path, hand_sizes_sorted)

    # Dynamically load the information_set.py module
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
    return info_set_module, metadata

def make_key_wrapper(module, my_hand: List[int], hand_abstractions: List[str], history: List[int], min_bet: int):
    """
    Calls make_key, passing min_bet only if the function supports it.
    """
    func = module.make_key
    sig = inspect.signature(func)
    if 'min_bet' in sig.parameters:
        return func(my_hand, hand_abstractions, history, min_bet=min_bet)
    else:
        return func(my_hand, hand_abstractions, history)

def get_possible_actions_wrapper(module, history: List[int], min_bet: int):
    """
    Calls get_possible_actions, passing min_bet only if the function supports it.
    """
    func = module.get_possible_actions
    sig = inspect.signature(func)
    if 'min_bet' in sig.parameters:
        return func(history, min_bet=min_bet)
    else:
        return func(history)

def preload_all_strategies(model_folder: str, hand_sizes_sorted: List[int]) -> Dict[str, str]:
    """
    Reads all strategy files for a model into an in-memory dictionary.
    Returns a dict mapping {info_set_key: encoded_strategy_string}.
    """
    strategies: Dict[str, str] = {}
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

def load_strategy_from_files(model_folder: str, key: str, hand_sizes: List[int], num_possible_actions: int) -> np.ndarray:
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

def get_strategy(current_model: Model, key: str, hand_sizes: List[int], possible_actions: List[int]) -> np.ndarray:
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

def get_h2h_expected_value(models: tuple, hands: tuple, history: List[int], active_player_idx: int, hand_sizes: List[int], existence_array: np.ndarray) -> float:
    """
    Recursively traverses the game tree to calculate expected value.
    """
    if Game.check_finish(history):
        payoff_for_checker = 1.0 if not existence_array[history[-2]] else -1.0
        return -payoff_for_checker

    current_model = models[active_player_idx]
    
    possible_actions = get_possible_actions_wrapper(
        module=current_model['module'],
        history=history,
        min_bet=current_model['metadata']['min_bet']
    )

    my_hand = hands[active_player_idx]
    hand_abstractions = current_model['module'].get_hand_abstraction(my_hand, hand_sizes)

    key = make_key_wrapper(
        module=current_model['module'],
        my_hand=my_hand,
        hand_abstractions=hand_abstractions,
        history=history,
        min_bet=current_model['metadata']['min_bet']
    )
    
    strategy = get_strategy(current_model, key, hand_sizes, possible_actions)

    node_ev = 0.0
    next_player_idx = (active_player_idx + 1) % 2
    for i, action in enumerate(possible_actions):
        if strategy[i] > 0:
            action_ev = -get_h2h_expected_value(models, hands, history + [action], next_player_idx, hand_sizes, existence_array)
            node_ev += strategy[i] * action_ev
    return node_ev

def run_mc_playout(models: Tuple[Model, Model], hands: Hands, starting_player_idx: int, hand_sizes: List[int], existence_array: np.ndarray) -> int:
    """
    Simulates a single game path based on sampling actions.
    """
    history = []
    active_player_idx = 0

    while not Game.check_finish(history):
        current_model = models[active_player_idx]
        my_hand = hands[active_player_idx]
        
        possible_actions = get_possible_actions_wrapper(
            module=current_model['module'],
            history=history,
            min_bet=current_model['metadata']['min_bet']
        )
        
        hand_abstractions = current_model['module'].get_hand_abstraction(my_hand, hand_sizes)

        key = make_key_wrapper(
            module=current_model['module'],
            my_hand=my_hand,
            hand_abstractions=hand_abstractions,
            history=history,
            min_bet=current_model['metadata']['min_bet']
        )

        strategy = get_strategy(current_model, key, hand_sizes, possible_actions)
        
        chosen_action = random.choices(possible_actions, weights=strategy, k=1)[0]
        
        history.append(chosen_action)
        active_player_idx = (active_player_idx + 1) % 2

    # Right now, checked player = active player
    payoff_for_checked_player = 1 if existence_array[history[-2]] else -1

    if active_player_idx == 0:
        return payoff_for_checked_player
    else:
        return -payoff_for_checked_player

def _resolve_model_path(tag: str | None, folder: str | None) -> str:
    """Resolve a model location from either an archive tag or an explicit folder.

    ``tag='current'`` (or ``.``) points at the working ``cfr_ai/`` tree;
    any other tag is looked up under ``cfr_ai/archive/<tag>/``.
    """
    if folder:
        return folder
    if tag in ("current", "."):
        return "cfr_ai"
    return os.path.join("cfr_ai", "archive", tag)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a head-to-head comparison between two Blef CFR AIs.")
    parser.add_argument("--hand-sizes", nargs=2, type=int, required=True, help="The number of cards for each player (e.g., 1 2).")
    g1 = parser.add_mutually_exclusive_group(required=True)
    g1.add_argument("--model1", type=str, help="Archive tag (folder under cfr_ai/archive/) or 'current' for the working tree.")
    g1.add_argument("--model1-folder", type=str, help="Explicit path to the first model's folder (Model A).")
    g2 = parser.add_mutually_exclusive_group(required=True)
    g2.add_argument("--model2", type=str, help="Archive tag (folder under cfr_ai/archive/) or 'current' for the working tree.")
    g2.add_argument("--model2-folder", type=str, help="Explicit path to the second model's folder (Model B).")
    parser.add_argument("--num-deals", type=int, default=1000, help="Number of random card deals to simulate.")
    parser.add_argument("--preload-strategies", action="store_true", help="Pre-load strategies for faster, memory-intensive evaluation.")
    parser.add_argument("--monte-carlo", action="store_true", help="Use fast Monte Carlo playouts instead of full tree traversal.")
    args = parser.parse_args()

    hand_sizes = sorted(args.hand_sizes)
    model1_path = _resolve_model_path(args.model1, args.model1_folder)
    model2_path = _resolve_model_path(args.model2, args.model2_folder)

    print("Loading AI models, logic, and metadata...")
    models_list = []
    for i, folder in enumerate([model1_path, model2_path]):
        try:
            module, metadata = load_model_components(folder, hand_sizes)
            model_name = f"Model {'A' if i == 0 else 'B'}"
            print(f"  - {model_name} from '{os.path.basename(folder)}' | Min Bet: {metadata.get('min_bet', 0)}")
            models_list.append({
                'folder': folder,
                'module': module,
                'metadata': metadata,
                'name': model_name,
                'strategies': None
            })
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    model_A, model_B = models_list[0], models_list[1]

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
