"""Multi-macro MCCFR trainer (Gen-I++): adds N augmenting macro actions, each
resolving per-hand to a legal bet by argmax of a per-macro score:

    value      => argmax(p)       (most-probable legal set)
    difftruthy => argmax(p - g)   (the set the hand most over-supports)
    bluff      => argmax(g - p)   (= argmin(p-g), the bluffiest)

Each macro is its OWN regret/strategy column (so they're distinct actions that
compete in regret-matching), but resolves per concrete hand at the node — so the
N macros together inject per-hand info the abstraction bucket discards, without
enlarging the infoset key. They add no game nodes: b* is always an existing
legal bet, force-traversed once, its value copied (exact under the TV cache).

Set the kinds via the BLUFF_MACRO_KINDS env var (comma-separated), e.g.
    BLUFF_MACRO_KINDS=value,difftruthy,bluff
Generalises `trainer_bluff.py` (which is the N=1 case). Random tie-break.

Per-row arrays are widened to ACT_W = 89 + n_macros (columns 89..89+n_macros-1
hold the macros). Run through production training.py via run_macros_training.py.
"""

from typing import List, Tuple, Dict, Any
import os
import time

import numpy as np
from numba import njit
from tqdm import trange

from cfr_ai.game import Game
from cfr_ai.trainer import Trainer, GROW_CHUNK
from cfr_ai.keys import (
    LAST_BET_SHIFT, H_M1_SHIFT, H_M2_SHIFT, ABS_ID_SHIFT,
    ABSENT_CODE, MAX_DEPTH, _HISTORY_CODE_ID, _HISTORY_CODE_STRS,
)
from cfr_ai.abstraction.probs import g_vector
from cfr_ai.abstraction.probs_jit import p_vector_fast


MAX_MACROS = 4
MACRO_BASE = 89  # macro columns are MACRO_BASE .. MACRO_BASE + n_macros - 1
VALID_KINDS = ("value", "difftruthy", "bluff")


@njit(cache=False)
def _traverse_jit_macros(
    history_buf, hist_len, reach, active, traverser, n_players, prune_feast,
    existence, iter_i,
    regrets, strategy_sum,
    lower_action, upper_action,
    last_touched, temporary_value,
    first_touched, times_touched,
    key_to_row, state, capacity,
    abs_ids, hand_sizes, history_code_id,
    min_bet, pruning_threshold, min_regret, penalty,
    strategy_buf, cf_buf,
    use_temp_value,
    scores, n_macros, bstar_buf, mcf_buf,
):
    # Traverser-centric (see trainer.py): values are always the TRAVERSER's, so
    # they flow up with no sign flip. Terminal: a check ends the round; loser
    # (gains a card) = checker if the set exists else the bettor; -1 loser,
    # 1/(n-1) others. Reduces to +1/-1 for n_players == 2.
    if hist_len > 0 and history_buf[hist_len - 1] == 88:
        checker = (active + n_players - 1) % n_players
        last_bettor = (active + n_players - 2) % n_players
        if existence[history_buf[hist_len - 2]]:
            loser = checker
        else:
            loser = last_bettor
        if traverser == loser:
            return -1.0
        return 1.0 / (n_players - 1)
    if state[1] != 0:
        return 0.0

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
        first_touched[row] = iter_i

    if use_temp_value and last_touched[row] == iter_i:
        return temporary_value[row]
    times_touched[row] += 1

    lower = lower_action[row]
    upper = upper_action[row]
    width = upper - lower + 1
    nxt = (active + 1) % n_players

    # --- macros: b*[k] = argmax(scores[k]) over legal bets, random tie-break -
    bet_hi = upper
    if bet_hi > 87:
        bet_hi = 87
    macro_active = lower <= bet_hi
    if macro_active:
        for k in range(n_macros):
            best = -1.0e18
            bs = -1
            ties = 0
            for b in range(lower, bet_hi + 1):
                s = scores[k, active, b]
                if s > best + 1e-9:
                    best = s
                    bs = b
                    ties = 1
                elif s > best - 1e-9:
                    ties += 1
                    if np.random.random() * ties < 1.0:
                        bs = b
            bstar_buf[hist_len, k] = bs

    # Regret matching over concrete (0..width-1) + macros (width..width+nm-1).
    pos_sum = 0.0
    for i in range(width):
        r = regrets[row, lower + i]
        if r > 0.0:
            strategy_buf[hist_len, i] = r
            pos_sum += r
        else:
            strategy_buf[hist_len, i] = 0.0
    if macro_active:
        for k in range(n_macros):
            mr = regrets[row, MACRO_BASE + k]
            if mr > 0.0:
                strategy_buf[hist_len, width + k] = mr
                pos_sum += mr
            else:
                strategy_buf[hist_len, width + k] = 0.0
    if pos_sum > 0.0:
        inv = 1.0 / pos_sum
        for i in range(width):
            strategy_buf[hist_len, i] *= inv
        if macro_active:
            for k in range(n_macros):
                strategy_buf[hist_len, width + k] *= inv
    else:
        strategy_buf[hist_len, width - 1] = 1.0
        if macro_active:
            for k in range(n_macros):
                strategy_buf[hist_len, width + k] = 0.0

    if active == traverser:
        for i in range(width):
            s = strategy_buf[hist_len, i]
            if s > 0.0:
                strategy_sum[row, lower + i] += np.float32(reach * s)
        if macro_active:
            for k in range(n_macros):
                sm = strategy_buf[hist_len, width + k]
                if sm > 0.0:
                    strategy_sum[row, MACRO_BASE + k] += np.float32(reach * sm)

        for i in range(width):
            cf_buf[hist_len, i] = 0.0
        # Children: traverse if un-pruned, in feast, or required by a
        # non-pruned macro (force b* so the macro can copy its value).
        for i in range(width):
            do = (regrets[row, lower + i] >= pruning_threshold) or prune_feast
            if (not do) and macro_active:
                for k in range(n_macros):
                    if (regrets[row, MACRO_BASE + k] >= pruning_threshold or prune_feast) \
                            and (lower + i == bstar_buf[hist_len, k]):
                        do = True
                        break
            if do:
                history_buf[hist_len] = lower + i
                cf_buf[hist_len, i] = _traverse_jit_macros(
                    history_buf, hist_len + 1, reach * strategy_buf[hist_len, i],
                    nxt, traverser, n_players, prune_feast,
                    existence, iter_i,
                    regrets, strategy_sum,
                    lower_action, upper_action,
                    last_touched, temporary_value,
                    first_touched, times_touched,
                    key_to_row, state, capacity,
                    abs_ids, hand_sizes, history_code_id,
                    min_bet, pruning_threshold, min_regret, penalty,
                    strategy_buf, cf_buf,
                    use_temp_value,
                    scores, n_macros, bstar_buf, mcf_buf,
                )
                if state[1] != 0:
                    return 0.0

        if macro_active:
            for k in range(n_macros):
                if regrets[row, MACRO_BASE + k] >= pruning_threshold or prune_feast:
                    mcf_buf[hist_len, k] = cf_buf[hist_len, bstar_buf[hist_len, k] - lower]
                else:
                    mcf_buf[hist_len, k] = 0.0

        node_value = 0.0
        for i in range(width):
            node_value += cf_buf[hist_len, i] * strategy_buf[hist_len, i]
        if macro_active:
            for k in range(n_macros):
                node_value += mcf_buf[hist_len, k] * strategy_buf[hist_len, width + k]

        if prune_feast:
            for i in range(width):
                v = regrets[row, lower + i] + cf_buf[hist_len, i] - node_value
                if v < min_regret:
                    v = min_regret
                regrets[row, lower + i] = v
            if macro_active:
                for k in range(n_macros):
                    v = regrets[row, MACRO_BASE + k] + mcf_buf[hist_len, k] - node_value
                    if v < min_regret:
                        v = min_regret
                    regrets[row, MACRO_BASE + k] = v
        else:
            for i in range(width):
                if regrets[row, lower + i] >= pruning_threshold:
                    regrets[row, lower + i] += cf_buf[hist_len, i] - node_value
            if macro_active:
                for k in range(n_macros):
                    if regrets[row, MACRO_BASE + k] >= pruning_threshold:
                        regrets[row, MACRO_BASE + k] += mcf_buf[hist_len, k] - node_value
    else:
        r_uni = np.random.random()
        acc = 0.0
        chosen = width - 1
        found = False
        for i in range(width):
            acc += strategy_buf[hist_len, i]
            if r_uni < acc:
                chosen = i
                found = True
                break
        chosen_macro = -1
        if (not found) and macro_active:
            for k in range(n_macros):
                acc += strategy_buf[hist_len, width + k]
                if r_uni < acc:
                    chosen_macro = k
                    found = True
                    break
        if chosen_macro >= 0:
            history_buf[hist_len] = bstar_buf[hist_len, chosen_macro]
        else:
            history_buf[hist_len] = lower + chosen
        node_value = _traverse_jit_macros(
            history_buf, hist_len + 1, reach,
            nxt, traverser, n_players, prune_feast,
            existence, iter_i,
            regrets, strategy_sum,
            lower_action, upper_action,
            last_touched, temporary_value,
            first_touched, times_touched,
            key_to_row, state, capacity,
            abs_ids, hand_sizes, history_code_id,
            min_bet, pruning_threshold, min_regret, penalty,
            strategy_buf, cf_buf,
            use_temp_value,
            scores, n_macros, bstar_buf, mcf_buf,
        ) + penalty

    last_touched[row] = iter_i
    temporary_value[row] = node_value
    state[2] += 1
    return node_value


class TrainerMacros(Trainer):
    """Drop-in for `Trainer` with N augmenting macro columns (kinds from the
    BLUFF_MACRO_KINDS env var or `macro_kinds` kwarg)."""

    def __init__(self, *args, **kwargs):
        kinds = kwargs.pop("macro_kinds", None) or os.environ.get(
            "BLUFF_MACRO_KINDS", "value,difftruthy,bluff")
        if isinstance(kinds, str):
            kinds = [k.strip().lower() for k in kinds.split(",") if k.strip()]
        for k in kinds:
            if k not in VALID_KINDS:
                raise ValueError(f"macro kind {k!r} not in {VALID_KINDS}")
        if not (1 <= len(kinds) <= MAX_MACROS):
            raise ValueError(f"need 1..{MAX_MACROS} macro kinds, got {kinds}")
        self.macro_kinds = kinds
        self.n_macros = len(kinds)
        self._act_w = MACRO_BASE + self.n_macros
        super().__init__(*args, **kwargs)
        if not self.use_temp_value:
            raise ValueError("TrainerMacros requires use_temp_value=True.")
        self._g_vector = g_vector(sum(self.hand_sizes)).astype(np.float64)
        self._n_total = int(sum(self.hand_sizes))
        self._strategy_buf = np.zeros((MAX_DEPTH, self._act_w), dtype=np.float64)
        self._cf_buf = np.zeros((MAX_DEPTH, self._act_w), dtype=np.float64)
        self._bstar_buf = np.zeros((MAX_DEPTH, MAX_MACROS), dtype=np.int64)
        self._mcf_buf = np.zeros((MAX_DEPTH, MAX_MACROS), dtype=np.float64)
        print(f"[TrainerMacros] kinds={self.macro_kinds} (random tie-break)", flush=True)

    def _alloc_arrays(self, cap: int, fresh: bool):
        w = self._act_w
        if fresh:
            self.regrets = np.zeros((cap, w), dtype=self.regret_dtype)
            self.strategy_sum = np.zeros((cap, w), dtype=np.float32)
            self.lower_action = np.zeros(cap, dtype=np.int16)
            self.upper_action = np.zeros(cap, dtype=np.int16)
            self.last_touched = np.zeros(cap, dtype=np.int32)
            self.first_touched = np.zeros(cap, dtype=np.int32)
            self.times_touched = np.zeros(cap, dtype=np.int32)
            self.temporary_value = np.zeros(cap, dtype=np.float64)
            return
        extra = cap - self.capacity
        self.regrets = np.concatenate([self.regrets, np.zeros((extra, w), dtype=self.regret_dtype)], axis=0)
        self.strategy_sum = np.concatenate([self.strategy_sum, np.zeros((extra, w), dtype=np.float32)], axis=0)
        self.lower_action = np.concatenate([self.lower_action, np.zeros(extra, dtype=np.int16)])
        self.upper_action = np.concatenate([self.upper_action, np.zeros(extra, dtype=np.int16)])
        self.last_touched = np.concatenate([self.last_touched, np.zeros(extra, dtype=np.int32)])
        self.first_touched = np.concatenate([self.first_touched, np.zeros(extra, dtype=np.int32)])
        self.times_touched = np.concatenate([self.times_touched, np.zeros(extra, dtype=np.int32)])
        self.temporary_value = np.concatenate([self.temporary_value, np.zeros(extra, dtype=np.float64)])

    def _score_one(self, pv):
        out = np.empty((self.n_macros, 88), dtype=np.float64)
        for k, kind in enumerate(self.macro_kinds):
            if kind == "value":
                out[k] = pv
            elif kind == "difftruthy":
                out[k] = pv - self._g_vector
            else:  # bluff
                out[k] = self._g_vector - pv
        return out

    def _build_scores_for_iter(self, hands) -> np.ndarray:
        scores = np.empty((self.n_macros, self.n_players, 88), dtype=np.float64)
        for p, hand in enumerate(hands):
            pv = p_vector_fast(hand, self._n_total - len(hand))
            s = self._score_one(pv)
            for k in range(self.n_macros):
                scores[k, p] = s[k]
        return scores

    def train(self, num_iterations, snapshot_callback=None, snapshot_every_log_points=1):
        npl = self.n_players
        utils = np.zeros(npl, dtype=np.float64)
        last_utils = np.zeros(npl, dtype=np.float64)
        utility_log: Dict[str, Any] = {}
        log_point_counter = 0
        hand_sizes_arr = np.asarray(self.hand_sizes, dtype=np.int64)
        train_start = time.time()
        last_log_time = train_start
        last_log_iter = 0
        print(f"[train-macros] hand_sizes={self.hand_sizes} kinds={self.macro_kinds} "
              f"target_iters={num_iterations:,}", flush=True)

        for i in trange(num_iterations, desc="Training (numba+macros)"):
            n = int(self.state[0])
            if n > 0:
                if i == int(num_iterations * 0.3):
                    self.strategy_sum[:n] *= np.float32(0.02)
                for t in range(4, 10):
                    if i == int(t * num_iterations / 10):
                        self.strategy_sum[:n] *= np.float32(t / (t + 1))

            prune_feast = bool(int(i / 4) % 20 == 0)
            traverser = (i // npl) % npl
            starting_player = i % npl
            hands = Game.deal_cards(self.hand_sizes)
            existence_array = Game.precompute_set_existence(hands)
            abs_ids = self._build_abs_ids_for_iter(hands)
            scores = self._build_scores_for_iter(hands)

            if self.state[0] + 1000 >= self.capacity:
                self._grow()

            self.state[1] = 0
            v = _traverse_jit_macros(
                self._history_buf, 0, 1.0,
                int(starting_player), int(traverser), int(npl), prune_feast,
                existence_array, int(i),
                self.regrets, self.strategy_sum,
                self.lower_action, self.upper_action,
                self.last_touched, self.temporary_value,
                self.first_touched, self.times_touched,
                self.key_to_row, self.state, int(self.capacity),
                abs_ids, hand_sizes_arr, _HISTORY_CODE_ID,
                int(self.min_bet), float(self.pruning_threshold),
                float(self.min_regret), float(self.penalty),
                self._strategy_buf, self._cf_buf,
                bool(self.use_temp_value),
                scores, int(self.n_macros), self._bstar_buf, self._mcf_buf,
            )
            if self.state[1] != 0:
                raise RuntimeError(f"Capacity {self.capacity} exhausted mid-iter {i}.")
            # OPENER value: accumulate only when the traverser is also the opener
            # (1/npl^2 of iters per player -> scale by npl^2 below). Traversal runs
            # every iter regardless, so the trained strategy is unaffected.
            if traverser == starting_player:
                utils[traverser] += v

            if self.log_points > 0 and (i + 1) % (num_iterations // self.log_points) == 0:
                now = time.time()
                overall_secs = now - train_start
                overall_rate = (i + 1) / overall_secs if overall_secs > 0 else 0.0
                chunk_rate = ((i + 1) - last_log_iter) / (now - last_log_time) if now > last_log_time else 0.0
                eta = (num_iterations - (i + 1)) / overall_rate if overall_rate > 0 else 0.0
                chunk = [(utils[p] - last_utils[p]) / num_iterations * self.log_points * npl * npl
                         for p in range(npl)]
                util_str = " ".join(f"P{p}={chunk[p]:+.4f}" for p in range(npl))
                print(f"[train-macros] iter {i+1:>10,}/{num_iterations:,} "
                      f"({100*(i+1)/num_iterations:>3.0f}%) | chunk {chunk_rate:>5.0f} it/s | "
                      f"overall {overall_rate:>5.0f} it/s | "
                      f"ETA {time.strftime('%H:%M:%S', time.gmtime(eta))} | "
                      f"rows={int(self.state[0]):,} | {util_str}", flush=True)
                for p in range(npl):
                    utility_log[f"P{p} Utility at Iter {i+1}"] = f"{chunk[p]:.4f}"
                last_utils = utils.copy()
                last_log_time = now
                last_log_iter = i + 1
                log_point_counter += 1
                if snapshot_callback is not None and log_point_counter % snapshot_every_log_points == 0:
                    snap, _ = self.get_final_flat_strategy(drop_check_only=True, clear_lows_threshold=0.01)
                    snapshot_callback(i + 1, snap)

        return ([float(utils[p]) * npl * npl / num_iterations for p in range(npl)],
                utility_log)

    # -- export ------------------------------------------------------------

    def get_macros_strategy(self):
        """Parallel arrays keyed by the int64 composite key: probs = c/T
        (concrete, padded [N,89]), masses[:,k] = m_k/T. Serve: resolved[i] =
        probs[i]; for enabled k: resolved[b*_k] += masses[i,k]."""
        keys_int, lowers, uppers, probs_rows, macro_rows = [], [], [], [], []
        for key in list(self.key_to_row.keys()):
            row = int(self.key_to_row[key])
            lo = int(self.lower_action[row])
            hi = int(self.upper_action[row])
            c = self.strategy_sum[row, lo:hi + 1].astype(np.float64)
            m = self.strategy_sum[row, MACRO_BASE:MACRO_BASE + self.n_macros].astype(np.float64)
            total = c.sum() + m.sum()
            if total <= 0.0:
                continue
            cprob = c / total
            mprob = m / total
            if mprob.sum() <= 0.0 and cprob[-1] >= 1.0 - 1e-9:
                continue
            padded = np.zeros(89, dtype=np.float32)
            padded[lo:hi + 1] = cprob.astype(np.float32)
            keys_int.append(int(key))
            lowers.append(lo)
            uppers.append(hi)
            probs_rows.append(padded)
            macro_rows.append(mprob.astype(np.float32))
        n = len(keys_int)
        return {
            "keys": np.array(keys_int, dtype=np.int64),
            "lower": np.array(lowers, dtype=np.int16),
            "upper": np.array(uppers, dtype=np.int16),
            "probs": np.stack(probs_rows) if n else np.zeros((0, 89), np.float32),
            "masses": np.stack(macro_rows) if n else np.zeros((0, self.n_macros), np.float32),
            "kinds": list(self.macro_kinds),
            "min_bet": int(self.min_bet),
        }
