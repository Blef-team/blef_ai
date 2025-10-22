#!/usr/bin/env python3
"""
Synthetic dataset generation for action-history embedding pretraining.

Implements the closure classification task described in `nfsp_ai/GPT_embedding_plan.txt`:
For each action ID (excluding CHECK), we enumerate the minimal card configurations that
realise the associated set. Training samples are built by picking k action IDs, sampling
one minimal hand per action, merging the cards, and computing the legality vector for
the merged hand. The observation consists of an ordered (most recent first) list of
action IDs padded to a fixed history length.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from shared.api.simpleschema_local_manager import determine_set_existence
from shared.game_utils import GameRules


CardTuple = Tuple[int, ...]


def _build_deck_cards(deck_size: int) -> List[dict]:
    if deck_size % 4 != 0:
        raise ValueError("deck_size must be divisible by 4")
    num_values = deck_size // 4
    cards: List[dict] = []
    for value in range(num_values):
        for suit in range(4):
            cards.append({"value": value, "colour": suit})
    return cards


def _card_to_id(card: dict) -> int:
    return int(card["value"]) * 4 + int(card["colour"])


def _cards_to_ids(cards: Sequence[dict]) -> CardTuple:
    return tuple(sorted(_card_to_id(card) for card in cards))


def _id_to_card(card_id: int) -> dict:
    if card_id < 0:
        raise ValueError("Card id must be non-negative")
    return {"value": card_id // 4, "colour": card_id % 4}


@lru_cache(maxsize=8)
def enumerate_minimal_action_hands(deck_size: int) -> Dict[int, List[CardTuple]]:
    """
    Enumerate minimal card configurations that satisfy each action ID (excluding CHECK).

    Args:
        deck_size: 24 or 32.

    Returns:
        Dictionary mapping action_id -> list of tuples of card ids.
    """
    rules = {"deck_size": deck_size}
    game_rules = GameRules(deck_size)
    check_action = game_rules.check_action_id
    base_cards = _build_deck_cards(deck_size)
    deck_ids = list(range(len(base_cards)))

    pools: Dict[int, List[CardTuple]] = {aid: [] for aid in range(check_action)}

    # Enumerate combinations up to size 5 (maximum set size in Blef).
    max_combo_size = 5
    for r in range(1, max_combo_size + 1):
        for combo_indices in itertools.combinations(deck_ids, r):
            cards = [base_cards[idx] for idx in combo_indices]
            ids_tuple = _cards_to_ids(cards)
            for action_id in range(check_action):
                if determine_set_existence(cards, action_id, rules):
                    # ensure minimal: removing any card breaks the set
                    minimal = True
                    if len(cards) > 1:
                        for drop_idx in range(len(cards)):
                            sub_cards = cards[:drop_idx] + cards[drop_idx + 1 :]
                            if determine_set_existence(sub_cards, action_id, rules):
                                minimal = False
                                break
                    if minimal:
                        pools[action_id].append(ids_tuple)
            # Early exit if combo cannot be minimal for any action.
    # Deduplicate pools
    for action_id, combos in pools.items():
        if combos:
            unique = sorted(set(combos))
            pools[action_id] = unique
        else:
            pools[action_id] = []
    return pools


def _ids_to_cards(ids: Sequence[int]) -> List[dict]:
    return [_id_to_card(card_id) for card_id in ids]


@dataclass
class ClosureSample:
    actions: torch.LongTensor        # shape [history_len]
    mask: torch.BoolTensor           # shape [history_len]
    labels: torch.FloatTensor        # shape [num_actions]


class ClosureIterableDataset(IterableDataset):
    """
    Iterable dataset implementing the closure classification task.
    """

    def __init__(
        self,
        deck_size: int,
        num_samples: int,
        history_len: int = 8,
        min_k: int = 1,
        max_k: int = 8,
        seed: int = 1337,
    ):
        super().__init__()
        if deck_size not in (24, 32):
            raise ValueError("deck_size must be 24 or 32")
        if not (1 <= min_k <= max_k <= history_len):
            raise ValueError("Must satisfy 1 <= min_k <= max_k <= history_len")
        self.deck_size = deck_size
        self.num_samples = int(num_samples)
        self.history_len = int(history_len)
        self.min_k = int(min_k)
        self.max_k = int(max_k)
        self.seed = int(seed)

        game_rules = GameRules(deck_size)
        self.num_actions = game_rules.check_action_id  # exclude CHECK
        pools = enumerate_minimal_action_hands(deck_size)
        self._action_pools = {aid: combos for aid, combos in pools.items() if combos}
        self._action_ids = sorted(self._action_pools.keys())
        if not self._action_ids:
            raise RuntimeError("No action pools generated; check deck size or rules.")

    def _spawn_rng(self, worker_id: int, epoch: int) -> np.random.Generator:
        base_seed = self.seed + 1_003 * epoch + 17 * worker_id
        return np.random.default_rng(base_seed)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[Tuple[torch.LongTensor, torch.BoolTensor, torch.FloatTensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        stride = worker.num_workers if worker else 1
        epoch = getattr(self, "_epoch", 0)
        rng = self._spawn_rng(worker_id, epoch)

        rules = {"deck_size": self.deck_size}
        total = (self.num_samples + stride - 1) // stride
        produced = 0
        action_ids = self._action_ids
        num_actions = self.num_actions

        for idx in range(worker_id, self.num_samples, stride):
            k = int(rng.integers(self.min_k, self.max_k + 1))
            chosen = rng.choice(action_ids, size=min(k, len(action_ids)), replace=False)
            history = -np.ones((self.history_len,), dtype=np.int64)
            mask = np.zeros((self.history_len,), dtype=bool)
            history[: len(chosen)] = chosen[::-1]  # most recent first
            mask[: len(chosen)] = True

            # Sample one minimal hand per action and merge cards.
            merged_cards: List[int] = []
            for action_id in chosen:
                combos = self._action_pools[action_id]
                combo = combos[rng.integers(0, len(combos))]
                merged_cards.extend(combo)
            merged_ids = sorted(set(merged_cards))
            card_dicts = _ids_to_cards(merged_ids)

            labels = np.zeros((num_actions,), dtype=np.float32)
            for action_id in range(num_actions):
                if determine_set_existence(card_dicts, action_id, rules):
                    labels[action_id] = 1.0

            yield (
                torch.from_numpy(history),
                torch.from_numpy(mask),
                torch.from_numpy(labels),
            )
            produced += 1
            if produced >= total:
                break
