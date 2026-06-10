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
        # metadata.csv from older training runs is ISO-8859/Windows-1252
        # (Polish characters etc.). Tolerate both encodings.
        try:
            f = open(metadata_path, 'r', encoding='utf-8')
            f.read(); f.seek(0)
        except UnicodeDecodeError:
            f = open(metadata_path, 'r', encoding='latin-1')
        with f:
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

def preload_all_strategies(model_folder: str, hand_sizes_sorted: List[int]) -> Dict[str, np.ndarray]:
    """
    Load all strategies for a model from `<model_folder>/outputs/<setup>/strategy.npz`
    into an in-memory dictionary keyed by the legacy string key
    (`hs-lb-(h_m1-h_m2-)abs`). Probabilities are pre-decoded `np.ndarray`
    arrays. Returns an empty dict (with a stderr warning) if the strategy
    file is missing — head_to_head treats this as "model checks at every
    decision", consistent with the old CSV-fallback behaviour.
    """
    from cfr_ai.strategy_io import load_strategy
    from cfr_ai.trainer import (
        LAST_BET_SHIFT, H_M1_SHIFT, H_M2_SHIFT, ABS_ID_SHIFT, ABSENT_CODE,
        _HISTORY_CODE_STRS,
    )

    setup_name = "_".join(str(x) for x in hand_sizes_sorted)
    setup_dir = os.path.join(model_folder, 'outputs', setup_name)
    npz_path = os.path.join(setup_dir, 'strategy.npz')
    if not os.path.exists(npz_path):
        print(
            f"Warning: strategy.npz not found at '{npz_path}'. "
            "Cannot preload strategies.",
            file=sys.stderr,
        )
        return {}

    # Macro (V3+) setups carry their augmenting-action masses separately from the
    # concrete probability slice. This tool reads only the concrete slice, which is
    # sub-stochastic for a macro model, so it would silently evaluate a policy the
    # agent never plays. Refuse rather than mislead.
    with np.load(npz_path, allow_pickle=True) as _z:
        if "kinds" in _z.files or "masses" in _z.files:
            raise RuntimeError(
                f"head_to_head.py cannot evaluate macro (V3+) setup '{setup_name}': "
                f"{npz_path} carries macro masses this tool ignores, which would "
                f"compare a policy the agent never plays. Use cfr_vs_cfr_games.py / "
                f"cfr_vs_nfsp_games.py (which drive the real folding agent) or "
                f"selftest_macros.py for macro models."
            )

    print(f"Pre-loading strategies from {npz_path}...")
    fs = load_strategy(setup_dir)
    id_to_abs = {v: k for k, v in fs.abs_str_to_id.items()}
    strategies: Dict[str, np.ndarray] = {}
    for k_int, row in fs.key_to_row.items():
        k = int(k_int)
        row = int(row)
        hand_size = k & 0xF
        last_bet = (k >> LAST_BET_SHIFT) & 0xFF
        h_m1_id = (k >> H_M1_SHIFT) & 0xFF
        h_m2_id = (k >> H_M2_SHIFT) & 0xFF
        abs_id = k >> ABS_ID_SHIFT
        key_str = f"{hand_size}-{last_bet}-"
        if h_m1_id != ABSENT_CODE:
            key_str += _HISTORY_CODE_STRS[h_m1_id] + "-"
            if h_m2_id != ABSENT_CODE:
                key_str += _HISTORY_CODE_STRS[h_m2_id] + "-"
        key_str += id_to_abs[abs_id]
        lo = int(fs.lower_action[row])
        hi = int(fs.upper_action[row])
        strategies[key_str] = fs.strategy[row, lo:hi + 1].astype(np.float64)
    print(f"Loaded {len(strategies)} strategy entries from {model_folder}.")
    return strategies


def get_strategy(current_model: Model, key: str, hand_sizes: List[int], possible_actions: List[int]) -> np.ndarray:
    """Look up a strategy in the pre-loaded NPZ dict. Falls back to
    check-100% when the key isn't present (matches the original
    fallback behaviour from the CSV days; missing infosets weren't
    stored, so checking is the recorded action there)."""
    num_actions = len(possible_actions)
    strategy = None
    if current_model.get('strategies') is not None:
        arr = current_model['strategies'].get(key)
        if arr is not None:
            strategy = np.asarray(arr, dtype=np.float64)

    if strategy is None:
        strategy = np.zeros(num_actions)
        strategy[-1] = 1.0
    if len(strategy) < num_actions:
        padded = np.zeros(num_actions)
        padded[:len(strategy)] = strategy
        strategy = padded
    if strategy.sum() == 0:
        strategy[-1] = 1.0
    return strategy / strategy.sum()

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
    # Strategies are always pre-loaded now (NPZ load is fast and there is
    # no per-row fallback path). Flag retained for backwards compatibility
    # of CLI invocations but is a no-op.
    parser.add_argument("--preload-strategies", action="store_true", help="(deprecated; always on now) Pre-load strategies into memory.")
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

    # Always preload now; the NPZ loader is fast and there's no per-row fallback.
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
