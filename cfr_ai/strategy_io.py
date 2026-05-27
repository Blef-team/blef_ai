"""Canonical strategy I/O for the Blef CFR AI.

One source of truth for saving and loading trained CFR strategies. NPZ
(compressed) for the flat data plus a sidecar JSON for the abstraction
string -> id table. Replaces the legacy per-(hand_size, last_bet) CSV
tree, which was 5-7x bigger on disk and 15-30x slower to load.

Layout written per setup directory `<dir>/`:
    strategy.npz          arrays: keys (int64 sorted), lower (int16),
                                  upper (int16), probs (float32[N,89]).
                          scalars: min_bet.
    strategy.abs.json     {abs_str: id} mapping for runtime interning.
    metadata.csv          unchanged - small, human-readable, training-time
                          text (Iterations / Penalty / Time finished etc).

The training pipeline writes both `strategy.npz` (deployment payload) and
`diagnostic.npz` (regret arrays + touch counters) for post-hoc analysis.
The deployed agent only needs `strategy.npz`; archive snapshots default
to skipping `diagnostic.npz` (`archive_tool --include-diagnostics` to
include it).

The loader returns a `FlatStrategy` (defined in `lbr.py`), which
is the same in-memory type all downstream code (LBR, subgame, agent.py)
consumes. So switching the storage format is a single-point change:
nothing inside the JIT path or the training core changes.
"""

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
from numba import types
from numba.typed import Dict as NbDict


# Schema versions. Bump if a future change needs to deprecate older files.
STRATEGY_NPZ_VERSION = 1
DIAGNOSTIC_NPZ_VERSION = 1


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_strategy(
    flat_strategy,
    setup_dir: str,
    *,
    compressed: bool = True,
) -> str:
    """Write the deployment-ready strategy.npz + strategy.abs.json under
    `setup_dir`. Returns the path of the .npz."""
    from cfr_ai.lbr import FlatStrategy  # local to avoid circular imports
    assert isinstance(flat_strategy, FlatStrategy)
    os.makedirs(setup_dir, exist_ok=True)

    # Collect keys in row order (the numpy arrays are already row-indexed).
    n = len(flat_strategy.key_to_row)
    keys = np.empty(n, dtype=np.int64)
    for k_int, row in flat_strategy.key_to_row.items():
        keys[int(row)] = int(k_int)
    # We want sorted keys at rest so that the on-disk format is canonical
    # and so a future lazy-load path can do binary search without resorting.
    # Sort the keys AND reorder the parallel arrays (probs, lower, upper)
    # to match. The mapping in the runtime typed.Dict gets rebuilt from
    # the sorted order at load time.
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    lower_sorted = flat_strategy.lower_action[order]
    upper_sorted = flat_strategy.upper_action[order]
    probs_sorted = flat_strategy.strategy[order]

    npz_path = os.path.join(setup_dir, "strategy.npz")
    saver = np.savez_compressed if compressed else np.savez
    saver(
        npz_path,
        version=np.int32(STRATEGY_NPZ_VERSION),
        keys=keys_sorted,
        lower=lower_sorted,
        upper=upper_sorted,
        probs=probs_sorted,
        min_bet=np.int32(flat_strategy.min_bet),
    )

    abs_path = os.path.join(setup_dir, "strategy.abs.json")
    with open(abs_path, "w", encoding="utf-8") as f:
        json.dump(flat_strategy.abs_str_to_id, f, ensure_ascii=False)

    return npz_path


def save_diagnostic(
    setup_dir: str,
    *,
    keys: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    first_touched: np.ndarray,
    last_touched: np.ndarray,
    times_touched: np.ndarray,
    strategy: np.ndarray = None,
    regrets: np.ndarray = None,
    strategy_sum: np.ndarray = None,
    compressed: bool = True,
) -> str:
    """Write `diagnostic.npz` (touch counters + at least one of strategy,
    regrets, or strategy_sum). Caller is responsible for sorting the
    arrays by key in canonical order.

    Two write paths exist, with different optional fields:
      - Fresh training runs (via `cfr_ai/training.py`) save the raw
        `regrets` and `strategy_sum`; `strategy` (the averaged final)
        can be derived from `strategy_sum` at load time, so it's not
        re-stored.
      - Pre-NPZ runs migrated from the legacy CSV diagnostic format only
        have the averaged `strategy` (the CSV never stored regrets); no
        `regrets`/`strategy_sum` for those.

    The loader (`load_diagnostic`) returns whatever's present.
    """
    os.makedirs(setup_dir, exist_ok=True)
    path = os.path.join(setup_dir, "diagnostic.npz")
    arrays = {
        "version": np.int32(DIAGNOSTIC_NPZ_VERSION),
        "keys": keys,
        "lower": lower,
        "upper": upper,
        "first_touched": first_touched,
        "last_touched": last_touched,
        "times_touched": times_touched,
    }
    if strategy is not None:
        arrays["strategy"] = strategy
    if regrets is not None:
        arrays["regrets"] = regrets
    if strategy_sum is not None:
        arrays["strategy_sum"] = strategy_sum
    if "strategy" not in arrays and "strategy_sum" not in arrays:
        raise ValueError("save_diagnostic needs at least one of "
                         "strategy, strategy_sum")
    saver = np.savez_compressed if compressed else np.savez
    saver(path, **arrays)
    return path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_strategy(setup_dir: str):
    """Load `setup_dir/strategy.npz` + sidecar JSON into a `FlatStrategy`.

    The dominant cost on large setups is building the numba `typed.Dict`
    (~3s/M entries). The on-disk read is sub-second even on the biggest
    trained setups.

    Raises `FileNotFoundError` if the npz isn't present (caller can
    handle the migration prompt)."""
    from cfr_ai.lbr import FlatStrategy
    npz_path = os.path.join(setup_dir, "strategy.npz")
    abs_path = os.path.join(setup_dir, "strategy.abs.json")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"strategy.npz not found at {npz_path}")

    data = np.load(npz_path)
    if "version" in data:
        v = int(data["version"])
        if v != STRATEGY_NPZ_VERSION:
            raise ValueError(
                f"strategy.npz version {v} != supported {STRATEGY_NPZ_VERSION}"
            )
    keys = data["keys"]
    lower = data["lower"]
    upper = data["upper"]
    probs = data["probs"]
    min_bet = int(data["min_bet"])

    if os.path.exists(abs_path):
        with open(abs_path, "r", encoding="utf-8") as f:
            abs_str_to_id = json.load(f)
    else:
        abs_str_to_id = {}

    # Build the numba typed dict (row index by composite int64 key).
    key_to_row = NbDict.empty(key_type=types.int64, value_type=types.int64)
    for i in range(keys.shape[0]):
        key_to_row[np.int64(keys[i])] = np.int64(i)

    return FlatStrategy(
        key_to_row=key_to_row,
        strategy=probs,
        lower_action=lower,
        upper_action=upper,
        abs_str_to_id=abs_str_to_id,
        min_bet=min_bet,
    )


def load_diagnostic(setup_dir: str) -> Dict[str, np.ndarray]:
    """Load `setup_dir/diagnostic.npz`. Returns a dict with whichever
    arrays are present. `strategy` is derived from `strategy_sum` if not
    stored explicitly. Raises if `diagnostic.npz` is missing."""
    path = os.path.join(setup_dir, "diagnostic.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"diagnostic.npz not found at {path}")
    data = np.load(path)
    out: Dict[str, np.ndarray] = {
        "version": int(data["version"]) if "version" in data else 0,
        "keys": data["keys"],
        "lower": data["lower"],
        "upper": data["upper"],
        "first_touched": data["first_touched"],
        "last_touched": data["last_touched"],
        "times_touched": data["times_touched"],
    }
    for opt in ("strategy", "regrets", "strategy_sum"):
        if opt in data:
            out[opt] = data[opt]
    # If only `strategy_sum` is present, derive `strategy` row-wise so
    # downstream analysis tools don't have to.
    if "strategy" not in out and "strategy_sum" in out:
        ssum = out["strategy_sum"]
        sums = ssum.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        out["strategy"] = (ssum / sums).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Sanity / unit
# ---------------------------------------------------------------------------

def _roundtrip_smoke(setup_dir: str) -> None:
    """Round-trip a strategy directory through save_strategy/load_strategy
    and confirm the lookup-by-key result is bit-equivalent on a handful of
    keys. Quick sanity check for the I/O path."""
    import tempfile
    ref = load_strategy(setup_dir)
    with tempfile.TemporaryDirectory() as td:
        save_strategy(ref, td, compressed=True)
        new = load_strategy(td)
    assert new.min_bet == ref.min_bet
    assert len(new.key_to_row) == len(ref.key_to_row)
    for k_int, ref_row in list(ref.key_to_row.items())[:200]:
        k = np.int64(int(k_int))
        new_row = int(new.key_to_row[k])
        assert int(new.lower_action[new_row]) == int(ref.lower_action[int(ref_row)])
        assert int(new.upper_action[new_row]) == int(ref.upper_action[int(ref_row)])
        lo = int(ref.lower_action[int(ref_row)])
        hi = int(ref.upper_action[int(ref_row)])
        a = ref.strategy[int(ref_row), lo:hi + 1]
        b = new.strategy[new_row, lo:hi + 1]
        if not np.array_equal(a, b):
            raise AssertionError(
                f"strategy mismatch for key {int(k_int)}: ref={a} new={b}"
            )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Smoke-test the strategy I/O.")
    ap.add_argument("--setup-dir", default="cfr_ai/outputs/1_1",
                    help="Directory containing strategy.npz")
    args = ap.parse_args()
    print(f"Round-trip test for {args.setup_dir}...")
    t0 = time.time()
    _roundtrip_smoke(args.setup_dir)
    print(f"  OK in {time.time() - t0:.2f}s")
