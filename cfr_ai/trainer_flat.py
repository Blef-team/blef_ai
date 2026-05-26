"""Flat-array MCCFR trainer for Blef.

Same external-sampling MCCFR algorithm as `trainer.Trainer`, but the per-
infoset state lives in flat numpy arrays indexed by row instead of in a
dict of `InformationSet` objects keyed by string. This eliminates per-node
string concatenation, per-row Python object overhead, and per-row attribute
lookups; what remains is well-suited to Numba JIT (Stage 2).

Key encoding
------------
Composite 64-bit int infoset key:

    [abs_id : 36][h_m2 : 8][h_m1 : 8][last_bet : 8][hand_size : 4]

where
- hand_size is 1..12
- last_bet is 0..87 for a real bet or 88 for the "no bet yet" sentinel
- h_m1, h_m2 are uint8 ids into the global `history_codes` unique-string
  table (96 unique strings -> ids 0..95), with 255 as the ABSENT sentinel
- abs_id is a per-setup int id assigned lazily to each distinct hand
  abstraction string returned by `get_hand_abstraction(...)[last_bet]`.

The per-setup `_abs_to_id` dict and the global `_history_code_id` matrix
are populated once; in the outer loop we pre-compute, for each player,
the 89-entry array `abs_ids[player][last_bet]` (178 string lookups per
iteration -- negligible).

Storage
-------
- `regrets[row, 0..88]` (float64) - padded to 89 columns; only the slice
  `[lower : upper+1]` is meaningful.
- `strategy_sum[row, 0..88]` (float32) - same.
- `lower_action[row]`, `upper_action[row]` (int8): legal-action slice
  endpoints. Width is `upper - upper + 1`.
- `last_touched`, `times_touched`, `first_touched` (int32): match the
  per-`InformationSet` counters in `trainer.py`.
- `temporary_value[row]` (float64): same-iter cache.

Capacity doubles when `n_rows == capacity`. Initial capacity 100k is
sufficient for any setup with fewer than ~100k discovered infosets;
larger setups expand on the fly.

Currently supported algorithms: 'es' only (matching the production
trainer). DCFR/CFR+ will be added if Stage 1 validates and Stage 2's
Numba jit lands.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import random
import time
from tqdm import trange

from cfr_ai.game import Game
from cfr_ai.information_set import (
    get_hand_abstraction, history_codes,
)


# ---------------------------------------------------------------------------
# Composite-key bit layout
# ---------------------------------------------------------------------------

HAND_SIZE_BITS = 4
LAST_BET_BITS = 8
H_CODE_BITS = 8
ABS_ID_BITS = 36

HAND_SIZE_SHIFT = 0
LAST_BET_SHIFT = HAND_SIZE_SHIFT + HAND_SIZE_BITS         # 4
H_M1_SHIFT = LAST_BET_SHIFT + LAST_BET_BITS               # 12
H_M2_SHIFT = H_M1_SHIFT + H_CODE_BITS                     # 20
ABS_ID_SHIFT = H_M2_SHIFT + H_CODE_BITS                   # 28

ABSENT_CODE = 255  # sentinel for "h_m1 / h_m2 absent in this infoset"


# ---------------------------------------------------------------------------
# Precomputed global history_code id matrix
# ---------------------------------------------------------------------------

def _build_history_code_id_matrix() -> Tuple[np.ndarray, List[str]]:
    """Returns (code_id, id_to_str) where code_id[a, b] is the uint8 id
    of history_codes[a, b]. id_to_str inverts the id mapping.

    history_codes is shape (88, 88); we keep that shape in the matrix.
    Entries outside [0..87] are never queried in practice."""
    unique = sorted(set(history_codes.flatten().tolist()))
    str_to_id = {s: i for i, s in enumerate(unique)}
    assert len(unique) < 255, "Too many unique history codes for uint8 id"
    code_id = np.zeros((88, 88), dtype=np.uint8)
    for a in range(88):
        for b in range(88):
            code_id[a, b] = str_to_id[history_codes[a, b]]
    return code_id, unique


_HISTORY_CODE_ID, _HISTORY_CODE_STRS = _build_history_code_id_matrix()


# ---------------------------------------------------------------------------
# FlatTrainer
# ---------------------------------------------------------------------------

class FlatTrainer:
    def __init__(
        self,
        hand_sizes: List[int],
        min_bet: int,
        pruning_range: List[int],
        penalty: float,
        log_points: int,
        algorithm: str = 'es',
        initial_capacity: int = 100_000,
        seed: int = None,
    ):
        if algorithm != 'es':
            raise NotImplementedError(
                "trainer_flat currently supports only 'es'. DCFR/CFR+ would "
                "need the per-row α-discount and linear strategy weighting."
            )
        self.hand_sizes = hand_sizes
        self.min_bet = min_bet
        self.pruning_threshold = pruning_range[0]
        self.min_regret = pruning_range[1]
        self.penalty = penalty
        self.log_points = log_points
        self.algorithm = algorithm

        # Per-setup lazy maps.
        self._abs_to_id: Dict[str, int] = {}
        self._key_to_row: Dict[int, int] = {}

        # Flat row storage.
        self.capacity = int(initial_capacity)
        self.n_rows = 0
        self._alloc_arrays(self.capacity)

        self.nodes_touched = 0

        # RNG seeding (optional, for reproducible benchmarks).
        if seed is not None:
            random.seed(seed)
            # The deal RNG (Game.rng) is a module-level np.random.default_rng();
            # caller is responsible for seeding it externally if needed.

    # -- storage helpers ----------------------------------------------------

    def _alloc_arrays(self, cap: int) -> None:
        self.regrets = np.zeros((cap, 89), dtype=np.float64)
        self.strategy_sum = np.zeros((cap, 89), dtype=np.float32)
        self.lower_action = np.zeros(cap, dtype=np.int16)
        self.upper_action = np.zeros(cap, dtype=np.int16)
        self.last_touched = np.zeros(cap, dtype=np.int32)
        self.times_touched = np.zeros(cap, dtype=np.int32)
        self.first_touched = np.zeros(cap, dtype=np.int32)
        self.temporary_value = np.zeros(cap, dtype=np.float64)

    def _grow(self) -> None:
        new_cap = self.capacity * 2
        regrets = np.zeros((new_cap, 89), dtype=np.float64)
        regrets[: self.capacity] = self.regrets
        strategy_sum = np.zeros((new_cap, 89), dtype=np.float32)
        strategy_sum[: self.capacity] = self.strategy_sum
        lower_action = np.zeros(new_cap, dtype=np.int16)
        lower_action[: self.capacity] = self.lower_action
        upper_action = np.zeros(new_cap, dtype=np.int16)
        upper_action[: self.capacity] = self.upper_action
        last_touched = np.zeros(new_cap, dtype=np.int32)
        last_touched[: self.capacity] = self.last_touched
        times_touched = np.zeros(new_cap, dtype=np.int32)
        times_touched[: self.capacity] = self.times_touched
        first_touched = np.zeros(new_cap, dtype=np.int32)
        first_touched[: self.capacity] = self.first_touched
        temporary_value = np.zeros(new_cap, dtype=np.float64)
        temporary_value[: self.capacity] = self.temporary_value

        self.regrets = regrets
        self.strategy_sum = strategy_sum
        self.lower_action = lower_action
        self.upper_action = upper_action
        self.last_touched = last_touched
        self.times_touched = times_touched
        self.first_touched = first_touched
        self.temporary_value = temporary_value
        self.capacity = new_cap

    def _intern_abs(self, s: str) -> int:
        i = self._abs_to_id.get(s)
        if i is None:
            i = len(self._abs_to_id)
            self._abs_to_id[s] = i
        return i

    # -- per-iter pre-computation ------------------------------------------

    def _build_abs_ids_for_iter(self, hands) -> List[np.ndarray]:
        """Returns abs_ids[player][last_bet_idx] = int id for the hand
        abstraction at that last_bet context. Populates self._abs_to_id
        on first sighting of a new abstraction string."""
        out = []
        for hand in hands:
            strings = get_hand_abstraction(hand, self.hand_sizes)
            arr = np.empty(89, dtype=np.int64)
            for lb in range(89):
                s = strings[lb]
                arr[lb] = self._intern_abs(s)
            out.append(arr)
        return out

    # -- key construction ---------------------------------------------------

    def _make_key(self, hand_size: int, history: List[int], abs_id: int) -> int:
        """Build the composite int64 key for a node. Mirrors `make_key`
        from information_set.py."""
        if len(history) == 0 or history[-1] < self.min_bet:
            last_bet = 88
            h_m1_id = ABSENT_CODE
            h_m2_id = ABSENT_CODE
        else:
            last_bet = history[-1]
            if len(history) > 1 and history[-2] >= self.min_bet:
                h_m1_id = int(_HISTORY_CODE_ID[last_bet, history[-2]])
                if len(history) > 2 and history[-3] >= self.min_bet:
                    h_m2_id = int(_HISTORY_CODE_ID[last_bet, history[-3]])
                else:
                    h_m2_id = ABSENT_CODE
            else:
                h_m1_id = ABSENT_CODE
                h_m2_id = ABSENT_CODE

        return (
            hand_size
            | (last_bet << LAST_BET_SHIFT)
            | (h_m1_id << H_M1_SHIFT)
            | (h_m2_id << H_M2_SHIFT)
            | (abs_id << ABS_ID_SHIFT)
        )

    def _get_or_create_row(self, key: int, history: List[int], iter: int) -> int:
        row = self._key_to_row.get(key)
        if row is not None:
            return row
        if self.n_rows >= self.capacity:
            self._grow()
        row = self.n_rows
        self.n_rows += 1
        if len(history) == 0 or history[-1] < self.min_bet:
            self.lower_action[row] = self.min_bet
            self.upper_action[row] = 87
        else:
            self.lower_action[row] = history[-1] + 1
            self.upper_action[row] = 88
        self.first_touched[row] = iter
        # last_touched stays at 0 sentinel (untouched this iter).
        self._key_to_row[key] = row
        return row

    # -- recursion ----------------------------------------------------------

    def get_node_value(
        self,
        hands,
        hand_size_per_player: List[int],
        abs_ids_per_player: List[np.ndarray],
        history: List[int],
        reach_probability: float,
        active_player: int,
        traverser: int,
        prune_feast: bool,
        existence_array: np.ndarray,
        iter: int,
    ) -> float:
        # Terminal: last action was check (88). Game ended on history[-2].
        if len(history) > 0 and history[-1] == 88:
            return 1.0 if existence_array[history[-2]] else -1.0

        hand_size = hand_size_per_player[active_player]
        if len(history) == 0 or history[-1] < self.min_bet:
            last_bet_idx = 88  # "no bet yet" — also the index into abs_ids
        else:
            last_bet_idx = history[-1]
        abs_id = int(abs_ids_per_player[active_player][last_bet_idx])

        key = self._make_key(hand_size, history, abs_id)
        row = self._get_or_create_row(key, history, iter)

        # Same-iter caching: if this node was already visited in this iter,
        # return the cached value rather than re-recursing.
        if self.last_touched[row] == iter:
            return self.temporary_value[row]

        lower = int(self.lower_action[row])
        upper = int(self.upper_action[row])
        width = upper - lower + 1
        opp = 1 - active_player

        # Regret matching on the legal slice.
        legal_regrets = self.regrets[row, lower : upper + 1]
        # Equivalent to: any(legal_regrets > 0)? cached via local op.
        # np.maximum is fine; we materialise `strategy` either way.
        if (legal_regrets > 0).any():
            strategy = np.maximum(legal_regrets, 0.0)
            strategy = strategy / strategy.sum()
        else:
            strategy = np.zeros(width, dtype=np.float64)
            strategy[-1] = 1.0

        if active_player == traverser:
            # Strategy sum update (only on traverser branch, weighted by reach).
            self.strategy_sum[row, lower : upper + 1] += np.float32(reach_probability) * strategy.astype(np.float32)

            counterfactual_values = np.zeros(width, dtype=np.float64)
            for i in range(width):
                action = lower + i
                if legal_regrets[i] >= self.pruning_threshold or prune_feast:
                    counterfactual_values[i] = -self.get_node_value(
                        hands, hand_size_per_player, abs_ids_per_player,
                        history + [action],
                        reach_probability * float(strategy[i]),
                        opp, traverser, prune_feast, existence_array, iter,
                    )
            node_value = float(np.dot(counterfactual_values, strategy))

            if prune_feast:
                np.maximum(
                    legal_regrets + counterfactual_values - node_value,
                    self.min_regret,
                    out=legal_regrets,
                )
            else:
                to_update = legal_regrets >= self.pruning_threshold
                legal_regrets[to_update] = (
                    legal_regrets[to_update] + counterfactual_values[to_update] - node_value
                )
        else:
            # Opponent: sample one action.
            action_i = random.choices(range(width), weights=strategy.tolist(), k=1)[0]
            action = lower + action_i
            node_value = (
                -self.get_node_value(
                    hands, hand_size_per_player, abs_ids_per_player,
                    history + [action],
                    reach_probability,
                    opp, traverser, prune_feast, existence_array, iter,
                )
                + self.penalty
            )

        self.times_touched[row] += 1
        self.last_touched[row] = iter
        self.temporary_value[row] = node_value
        self.nodes_touched += 1
        return node_value

    # -- training loop ------------------------------------------------------

    def train(self, num_iterations: int) -> Tuple[float, float, Dict[str, Any]]:
        utils = [0.0, 0.0]
        last_utils = [0.0, 0.0]
        utility_log: Dict[str, Any] = {}

        train_start = time.time()
        last_log_time = train_start
        last_log_iter = 0
        print(
            f"[train_flat] algorithm={self.algorithm} "
            f"hand_sizes={self.hand_sizes} target_iters={num_iterations:,}",
            flush=True,
        )

        for i in trange(num_iterations, desc="Training (flat)"):
            # ES staircase strategy_sum discount, matching trainer.Trainer.
            if i == int(num_iterations * 0.3):
                self.strategy_sum[: self.n_rows] *= np.float32(0.02)
            for t in range(4, 10):
                if i == int(t * num_iterations / 10):
                    self.strategy_sum[: self.n_rows] *= np.float32(t / (t + 1))

            prune_feast = int(i / 4) % 20 == 0
            traverser = int(i / 2) % 2
            starting_player = i % 2
            hands = Game.deal_cards(self.hand_sizes)
            existence_array = Game.precompute_set_existence(hands)
            abs_ids_per_player = self._build_abs_ids_for_iter(hands)
            hand_size_per_player = [len(h) for h in hands]

            utils[starting_player] += self.get_node_value(
                hands, hand_size_per_player, abs_ids_per_player,
                [], 1.0, starting_player, traverser, prune_feast,
                existence_array, i,
            )

            if self.log_points > 0 and (i + 1) % (num_iterations // self.log_points) == 0:
                now = time.time()
                chunk_iters = (i + 1) - last_log_iter
                chunk_secs = now - last_log_time
                overall_secs = now - train_start
                overall_rate = (i + 1) / overall_secs if overall_secs > 0 else 0.0
                chunk_rate = chunk_iters / chunk_secs if chunk_secs > 0 else 0.0
                remaining = num_iterations - (i + 1)
                eta = remaining / overall_rate if overall_rate > 0 else 0.0
                util0_chunk = (utils[0] - last_utils[0]) / num_iterations * self.log_points * 2
                util1_chunk = (utils[1] - last_utils[1]) / num_iterations * self.log_points * 2
                print(
                    f"[train_flat] iter {i + 1:>10,}/{num_iterations:,} "
                    f"({100*(i+1)/num_iterations:>3.0f}%) | "
                    f"chunk {chunk_rate:>5.0f} it/s | "
                    f"overall {overall_rate:>5.0f} it/s | "
                    f"ETA {time.strftime('%H:%M:%S', time.gmtime(eta))} | "
                    f"rows={self.n_rows:,} | "
                    f"P0={util0_chunk:+.4f} P1={util1_chunk:+.4f}",
                    flush=True,
                )
                utility_log[f"P0 Utility at Iter {i + 1}"] = f"{util0_chunk:.4f}"
                utility_log[f"P1 Utility at Iter {i + 1}"] = f"{util1_chunk:.4f}"
                last_utils = list(utils)
                last_log_time = now
                last_log_iter = i + 1

        return (
            utils[0] * 2 / num_iterations,
            utils[1] * 2 / num_iterations,
            utility_log,
        )

    # -- export -------------------------------------------------------------

    def get_final_strategy_dict(self) -> Dict[str, np.ndarray]:
        """Return strategies in the same dict format as the reference
        trainer's `{k: v.get_final_strategy() for k, v in infoset_map.items()}`.

        The string key matches information_set.make_key() exactly so the
        output can be fed straight into lbr.lbr_exploitability."""
        # Invert abs_id -> string.
        id_to_abs = [None] * len(self._abs_to_id)
        for s, i in self._abs_to_id.items():
            id_to_abs[i] = s

        out: Dict[str, np.ndarray] = {}
        for key, row in self._key_to_row.items():
            hand_size = key & ((1 << HAND_SIZE_BITS) - 1)
            last_bet = (key >> LAST_BET_SHIFT) & ((1 << LAST_BET_BITS) - 1)
            h_m1_id = (key >> H_M1_SHIFT) & ((1 << H_CODE_BITS) - 1)
            h_m2_id = (key >> H_M2_SHIFT) & ((1 << H_CODE_BITS) - 1)
            abs_id = key >> ABS_ID_SHIFT

            # Reconstruct the original string key.
            key_str = f"{hand_size}-{last_bet}-"
            if h_m1_id != ABSENT_CODE:
                key_str += _HISTORY_CODE_STRS[h_m1_id] + "-"
                if h_m2_id != ABSENT_CODE:
                    key_str += _HISTORY_CODE_STRS[h_m2_id] + "-"
            key_str += id_to_abs[abs_id]

            # Final strategy from strategy_sum on the legal slice.
            lower = int(self.lower_action[row])
            upper = int(self.upper_action[row])
            ssum = self.strategy_sum[row, lower : upper + 1].astype(np.float64)
            total = ssum.sum()
            if total > 0:
                strategy = ssum / total
            else:
                strategy = np.zeros(upper - lower + 1, dtype=np.float64)
                strategy[-1] = 1.0
            out[key_str] = strategy
        return out
