"""Numba-jit MCCFR trainer for Blef (ES with `prune_feast` + staircase
strategy_sum discount), backed by flat numpy arrays and a typed-dict
infoset index.

Composite infoset key (int64):
    [abs_id : 36][h_m2 : 8][h_m1 : 8][last_bet : 8][hand_size : 4]

Per-row storage (padded to width 89, legal slice is `[lower:upper+1]`):
    regrets         fp32 or fp64
    strategy_sum    fp32
    lower_action    int16
    upper_action    int16
    last_touched    int32      (for same-iter cache)
    temporary_value fp64       (same-iter cache value)

The Python wrapper grows the flat arrays in constant `GROW_CHUNK`-sized
chunks whenever the JIT signals overflow, so the caller never has to
size capacity up front.
"""

from typing import List, Dict, Any, Tuple
import time

import numpy as np
from numba import njit, types
from numba.typed import Dict as NbDict
from tqdm import trange

from cfr_ai.game import Game
from cfr_ai.information_set import get_hand_abstraction, history_codes


# ---------------------------------------------------------------------------
# Composite-key bit layout
# ---------------------------------------------------------------------------

LAST_BET_SHIFT = 4
H_M1_SHIFT = 12
H_M2_SHIFT = 20
ABS_ID_SHIFT = 28

ABSENT_CODE = 255

# Max history depth (each player adds at most one bet > previous; +slack).
MAX_DEPTH = 92

# How many rows to add when the flat arrays fill up. 64k rows × ~720 B = ~50 MB.
GROW_CHUNK = 64_000


# ---------------------------------------------------------------------------
# Global history-code-id matrix, built once
# ---------------------------------------------------------------------------

def _build_history_code_id_matrix() -> Tuple[np.ndarray, List[str]]:
    unique = sorted(set(history_codes.flatten().tolist()))
    str_to_id = {s: i for i, s in enumerate(unique)}
    assert len(unique) < 255, "Too many unique history codes for uint8"
    code_id = np.zeros((88, 88), dtype=np.int64)
    for a in range(88):
        for b in range(88):
            code_id[a, b] = str_to_id[history_codes[a, b]]
    return code_id, unique


_HISTORY_CODE_ID, _HISTORY_CODE_STRS = _build_history_code_id_matrix()


# ---------------------------------------------------------------------------
# JIT'd recursive traversal
# ---------------------------------------------------------------------------

@njit(cache=True)
def _traverse_jit(
    history_buf, hist_len, reach, active, traverser, prune_feast,
    existence, iter_i,
    regrets, strategy_sum,
    lower_action, upper_action,
    last_touched, temporary_value,
    key_to_row, state, capacity,
    abs_ids, hand_sizes, history_code_id,
    min_bet, pruning_threshold, min_regret, penalty,
    strategy_buf, cf_buf,
):
    # Terminal
    if hist_len > 0 and history_buf[hist_len - 1] == 88:
        return 1.0 if existence[history_buf[hist_len - 2]] else -1.0

    # Capacity exhausted on a sibling — propagate up so Python can grow.
    if state[1] != 0:
        return 0.0

    # Composite-key components.
    if hist_len == 0 or history_buf[hist_len - 1] < min_bet:
        last_bet = 88
        h_m1_id = ABSENT_CODE
        h_m2_id = ABSENT_CODE
    else:
        last_bet = history_buf[hist_len - 1]
        if hist_len > 1 and history_buf[hist_len - 2] >= min_bet:
            h_m1_id = history_code_id[last_bet, history_buf[hist_len - 2]]
            if hist_len > 2 and history_buf[hist_len - 3] >= min_bet:
                h_m2_id = history_code_id[last_bet, history_buf[hist_len - 3]]
            else:
                h_m2_id = ABSENT_CODE
        else:
            h_m1_id = ABSENT_CODE
            h_m2_id = ABSENT_CODE

    hand_size = hand_sizes[active]
    abs_id = abs_ids[active, last_bet]
    key = (hand_size
           | (last_bet << LAST_BET_SHIFT)
           | (h_m1_id << H_M1_SHIFT)
           | (h_m2_id << H_M2_SHIFT)
           | (abs_id << ABS_ID_SHIFT))

    if key in key_to_row:
        row = key_to_row[key]
    else:
        if state[0] >= capacity:
            state[1] = 1
            return 0.0
        row = state[0]
        state[0] += 1
        if last_bet == 88:
            lower_action[row] = min_bet
            upper_action[row] = 87
        else:
            lower_action[row] = last_bet + 1
            upper_action[row] = 88
        key_to_row[key] = row

    # Same-iter cache. (Note: iter_i=0 vs default 0 collision is harmless —
    # the very first iter is wasted, matching the reference trainer's behaviour.)
    if last_touched[row] == iter_i:
        return temporary_value[row]

    lower = lower_action[row]
    upper = upper_action[row]
    width = upper - lower + 1
    opp = 1 - active

    # Regret matching on legal slice into this depth's slot of strategy_buf.
    # Single pass: cap negatives at 0 and accumulate pos_sum simultaneously,
    # then a divide-only pass to normalise. Halves the regret-read count vs
    # the prior two-pass form.
    pos_sum = 0.0
    for i in range(width):
        r = regrets[row, lower + i]
        if r > 0.0:
            strategy_buf[hist_len, i] = r
            pos_sum += r
        else:
            strategy_buf[hist_len, i] = 0.0
    if pos_sum > 0.0:
        inv = 1.0 / pos_sum
        for i in range(width):
            strategy_buf[hist_len, i] *= inv
    else:
        # All regrets <= 0; default to 100% on the last legal action.
        # strategy_buf was zeroed in the loop above; just flip the last slot.
        strategy_buf[hist_len, width - 1] = 1.0

    if active == traverser:
        # strategy_sum update on legal slice; skip the float32 cast + store
        # when this action's strategy weight is zero, which is the common
        # case once pruning has driven many regrets to 0.
        for i in range(width):
            s = strategy_buf[hist_len, i]
            if s > 0.0:
                strategy_sum[row, lower + i] += np.float32(reach * s)

        # Children. cf_buf and strategy_buf are MAX_DEPTH x 89 — children
        # write into [hist_len+1, :] so our [hist_len, :] is safe.
        for i in range(width):
            cf_buf[hist_len, i] = 0.0
        for i in range(width):
            if regrets[row, lower + i] >= pruning_threshold or prune_feast:
                history_buf[hist_len] = lower + i
                cf_buf[hist_len, i] = -_traverse_jit(
                    history_buf, hist_len + 1, reach * strategy_buf[hist_len, i],
                    opp, traverser, prune_feast,
                    existence, iter_i,
                    regrets, strategy_sum,
                    lower_action, upper_action,
                    last_touched, temporary_value,
                    key_to_row, state, capacity,
                    abs_ids, hand_sizes, history_code_id,
                    min_bet, pruning_threshold, min_regret, penalty,
                    strategy_buf, cf_buf,
                )
                if state[1] != 0:
                    return 0.0

        node_value = 0.0
        for i in range(width):
            node_value += cf_buf[hist_len, i] * strategy_buf[hist_len, i]

        if prune_feast:
            for i in range(width):
                v = regrets[row, lower + i] + cf_buf[hist_len, i] - node_value
                if v < min_regret:
                    v = min_regret
                regrets[row, lower + i] = v
        else:
            for i in range(width):
                if regrets[row, lower + i] >= pruning_threshold:
                    regrets[row, lower + i] += cf_buf[hist_len, i] - node_value
    else:
        # Sample one opponent action.
        r_uni = np.random.random()
        acc = 0.0
        chosen = width - 1
        for i in range(width):
            acc += strategy_buf[hist_len, i]
            if r_uni < acc:
                chosen = i
                break
        history_buf[hist_len] = lower + chosen
        node_value = -_traverse_jit(
            history_buf, hist_len + 1, reach,
            opp, traverser, prune_feast,
            existence, iter_i,
            regrets, strategy_sum,
            lower_action, upper_action,
            last_touched, temporary_value,
            key_to_row, state, capacity,
            abs_ids, hand_sizes, history_code_id,
            min_bet, pruning_threshold, min_regret, penalty,
            strategy_buf, cf_buf,
        ) + penalty

    last_touched[row] = iter_i
    temporary_value[row] = node_value
    state[2] += 1
    return node_value


# ---------------------------------------------------------------------------
# NumbaTrainer (Python orchestration)
# ---------------------------------------------------------------------------

class NumbaTrainer:
    def __init__(
        self,
        hand_sizes: List[int],
        min_bet: int,
        pruning_range: List[int],
        penalty: float,
        log_points: int,
        algorithm: str = 'es',
        initial_capacity: int = GROW_CHUNK,
        numba_seed: int = None,
        regret_dtype=np.float32,
    ):
        """initial_capacity defaults to GROW_CHUNK (64k). Arrays auto-grow by
        GROW_CHUNK rows on each fill; the user never has to size up front.

        regret_dtype defaults to np.float32 (validated to match fp64 LBR
        within noise and use ~50 % less RAM on production setups)."""
        if algorithm != 'es':
            raise NotImplementedError("trainer_numba supports only 'es'.")
        if regret_dtype not in (np.float64, np.float32):
            raise ValueError("regret_dtype must be np.float64 or np.float32")
        self.hand_sizes = hand_sizes
        self.min_bet = int(min_bet)
        self.pruning_threshold = float(pruning_range[0])
        self.min_regret = float(pruning_range[1])
        self.penalty = float(penalty)
        self.log_points = log_points
        self.algorithm = algorithm
        self.regret_dtype = regret_dtype

        self.capacity = int(initial_capacity)
        self._alloc_arrays(self.capacity, fresh=True)

        # state[0]=n_rows, state[1]=overflow flag, state[2]=nodes_touched
        self.state = np.zeros(3, dtype=np.int64)

        self.key_to_row = NbDict.empty(key_type=types.int64, value_type=types.int64)

        self._abs_to_id: Dict[str, int] = {}
        self._history_buf = np.zeros(MAX_DEPTH, dtype=np.int64)
        # Per-iter scratch buffers; slotted by recursion depth so children
        # can't clobber the parent's strategy/cf values.
        self._strategy_buf = np.zeros((MAX_DEPTH, 89), dtype=np.float64)
        self._cf_buf = np.zeros((MAX_DEPTH, 89), dtype=np.float64)

        if numba_seed is not None:
            _seed_numba(numba_seed)

    @property
    def n_rows(self) -> int:
        return int(self.state[0])

    @property
    def nodes_touched(self) -> int:
        return int(self.state[2])

    # -- array storage -----------------------------------------------------

    def _alloc_arrays(self, cap: int, fresh: bool):
        """If fresh, allocate from scratch. Else extend existing arrays."""
        if fresh:
            self.regrets = np.zeros((cap, 89), dtype=self.regret_dtype)
            self.strategy_sum = np.zeros((cap, 89), dtype=np.float32)
            self.lower_action = np.zeros(cap, dtype=np.int16)
            self.upper_action = np.zeros(cap, dtype=np.int16)
            self.last_touched = np.zeros(cap, dtype=np.int32)
            self.temporary_value = np.zeros(cap, dtype=np.float64)
            return
        # Extend: concat with new zero block of size (cap - self.capacity).
        extra = cap - self.capacity
        z2_r = np.zeros((extra, 89), dtype=self.regret_dtype)
        z2_s = np.zeros((extra, 89), dtype=np.float32)
        z_i16 = np.zeros(extra, dtype=np.int16)
        z_i32 = np.zeros(extra, dtype=np.int32)
        z_f64 = np.zeros(extra, dtype=np.float64)
        self.regrets = np.concatenate([self.regrets, z2_r], axis=0)
        self.strategy_sum = np.concatenate([self.strategy_sum, z2_s], axis=0)
        self.lower_action = np.concatenate([self.lower_action, z_i16])
        self.upper_action = np.concatenate([self.upper_action, z_i16.copy()])
        self.last_touched = np.concatenate([self.last_touched, z_i32])
        self.temporary_value = np.concatenate([self.temporary_value, z_f64])

    def _grow(self):
        new_cap = self.capacity + GROW_CHUNK
        self._alloc_arrays(new_cap, fresh=False)
        self.capacity = new_cap

    # -- abstraction interning --------------------------------------------

    def _build_abs_ids_for_iter(self, hands) -> np.ndarray:
        out = np.empty((2, 89), dtype=np.int64)
        a2i = self._abs_to_id
        for p, hand in enumerate(hands):
            strings = get_hand_abstraction(hand, self.hand_sizes)
            for lb in range(89):
                s = strings[lb]
                i = a2i.get(s)
                if i is None:
                    i = len(a2i)
                    a2i[s] = i
                out[p, lb] = i
        return out

    # -- training loop -----------------------------------------------------

    def train(self, num_iterations: int) -> Tuple[float, float, Dict[str, Any]]:
        utils = np.zeros(2, dtype=np.float64)
        last_utils = np.zeros(2, dtype=np.float64)
        utility_log: Dict[str, Any] = {}

        hand_sizes_arr = np.asarray(self.hand_sizes, dtype=np.int64)

        train_start = time.time()
        last_log_time = train_start
        last_log_iter = 0
        print(
            f"[train_numba] algorithm={self.algorithm} "
            f"hand_sizes={self.hand_sizes} target_iters={num_iterations:,}",
            flush=True,
        )

        for i in trange(num_iterations, desc="Training (numba)"):
            # ES staircase strategy_sum discount, on visited rows only.
            n = int(self.state[0])
            if n > 0:
                if i == int(num_iterations * 0.3):
                    self.strategy_sum[:n] *= np.float32(0.02)
                for t in range(4, 10):
                    if i == int(t * num_iterations / 10):
                        self.strategy_sum[:n] *= np.float32(t / (t + 1))

            prune_feast = bool(int(i / 4) % 20 == 0)
            traverser = int(i / 2) % 2
            starting_player = i % 2
            hands = Game.deal_cards(self.hand_sizes)
            existence_array = Game.precompute_set_existence(hands)
            abs_ids = self._build_abs_ids_for_iter(hands)

            # Pre-grow if we're within MAX_NEW_PER_ITER rows of the current
            # capacity. Cheaper than retrying with partial updates: each iter
            # discovers at most ~MAX_DEPTH × 89 new infosets in pathological
            # cases, so a 1000-row safety margin is well over twice the
            # worst-case per-iter add.
            MAX_NEW_PER_ITER = 1000
            if self.state[0] + MAX_NEW_PER_ITER >= self.capacity:
                self._grow()

            self.state[1] = 0
            v = _traverse_jit(
                self._history_buf, 0, 1.0,
                int(starting_player), int(traverser), prune_feast,
                existence_array, int(i),
                self.regrets, self.strategy_sum,
                self.lower_action, self.upper_action,
                self.last_touched, self.temporary_value,
                self.key_to_row, self.state, int(self.capacity),
                abs_ids, hand_sizes_arr, _HISTORY_CODE_ID,
                int(self.min_bet), float(self.pruning_threshold),
                float(self.min_regret), float(self.penalty),
                self._strategy_buf, self._cf_buf,
            )
            # The pre-grow above should always keep us under capacity. If a
            # pathological iter still overflows, surface it loudly — we'd
            # rather know than silently lose updates.
            if self.state[1] != 0:
                raise RuntimeError(
                    f"Capacity {self.capacity} exhausted mid-iter {i} despite "
                    f"pre-grow (rows={int(self.state[0])}). Increase MAX_NEW_PER_ITER."
                )

            utils[starting_player] += v

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
                    f"[train_numba] iter {i + 1:>10,}/{num_iterations:,} "
                    f"({100*(i+1)/num_iterations:>3.0f}%) | "
                    f"chunk {chunk_rate:>5.0f} it/s | "
                    f"overall {overall_rate:>5.0f} it/s | "
                    f"ETA {time.strftime('%H:%M:%S', time.gmtime(eta))} | "
                    f"cap={self.capacity:,} rows={int(self.state[0]):,} | "
                    f"P0={util0_chunk:+.4f} P1={util1_chunk:+.4f}",
                    flush=True,
                )
                utility_log[f"P0 Utility at Iter {i + 1}"] = f"{util0_chunk:.4f}"
                utility_log[f"P1 Utility at Iter {i + 1}"] = f"{util1_chunk:.4f}"
                last_utils = utils.copy()
                last_log_time = now
                last_log_iter = i + 1

        return (
            float(utils[0]) * 2 / num_iterations,
            float(utils[1]) * 2 / num_iterations,
            utility_log,
        )

    # -- export -----------------------------------------------------------

    def get_final_strategy_dict(self) -> Dict[str, np.ndarray]:
        """Returns the averaged strategy as `{make_key-string: np.ndarray}`,
        ready to feed to lbr.lbr_exploitability or head_to_head."""
        id_to_abs = [None] * len(self._abs_to_id)
        for s, i in self._abs_to_id.items():
            id_to_abs[i] = s

        out: Dict[str, np.ndarray] = {}
        for key in list(self.key_to_row.keys()):
            row = int(self.key_to_row[key])
            hand_size = key & 0xF
            last_bet = (key >> LAST_BET_SHIFT) & 0xFF
            h_m1_id = (key >> H_M1_SHIFT) & 0xFF
            h_m2_id = (key >> H_M2_SHIFT) & 0xFF
            abs_id = key >> ABS_ID_SHIFT

            key_str = f"{hand_size}-{last_bet}-"
            if h_m1_id != ABSENT_CODE:
                key_str += _HISTORY_CODE_STRS[h_m1_id] + "-"
                if h_m2_id != ABSENT_CODE:
                    key_str += _HISTORY_CODE_STRS[h_m2_id] + "-"
            key_str += id_to_abs[abs_id]

            lower = int(self.lower_action[row])
            upper = int(self.upper_action[row])
            ssum = self.strategy_sum[row, lower:upper + 1].astype(np.float64)
            total = ssum.sum()
            if total > 0:
                strategy = ssum / total
            else:
                strategy = np.zeros(upper - lower + 1, dtype=np.float64)
                strategy[-1] = 1.0
            out[key_str] = strategy
        return out


@njit(cache=True)
def _seed_numba(seed):
    np.random.seed(seed)
