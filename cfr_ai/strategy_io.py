"""Canonical strategy I/O for the Blef CFR AI.

Two load entry points:

* `load_strategy(setup_dir)` — full-fidelity loader for training and LBR.
  Builds a numba `typed.Dict[int64, int64]` for O(1) per-lookup cost.
  The returned `FlatStrategy` (defined in `lbr.py`) is the type expected
  by the JIT recursion in `lbr.py`.

* `load_strategy_for_agent(setup_dir)` — minimal-overhead loader for the
  deployed agent's one-lookup-per-request pattern. Keys stay as a sorted
  numpy array; `np.searchsorted` handles the binary-search lookup. Does
  not import numba, so the Lambda runtime drops the numba dependency
  entirely. Returns a `FlatStrategyAgent` defined locally in this module.
  Two on-disk layouts are supported: the compressed `strategy.npz` that
  training writes by default, and the sparse-mmap layout that
  `write_mmap_layout` produces for the Lambda image.

Layout written per setup directory `<dir>/`:
    strategy.npz          arrays: keys (int64 sorted), lower (int16),
                                  upper (int16), probs (float32[N, 89]).
                          scalars: min_bet.
    strategy.abs.json     {abs_str: id} mapping for runtime interning.
    metadata.csv          small, human-readable, training-time text
                          (Iterations / Penalty / Time finished etc).

The training pipeline writes both `strategy.npz` (deployment payload) and
`diagnostic.npz` (regret arrays + touch counters) for post-hoc analysis.
The deployed agent only needs `strategy.npz` (or the sparse-mmap
artifacts derived from it); archive snapshots default to skipping
`diagnostic.npz` (`archive_tool --include-diagnostics` to include it).
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


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
    masses: Optional[np.ndarray] = None,
    kinds: Optional[List[str]] = None,
) -> str:
    """Write the deployment-ready strategy.npz + strategy.abs.json under
    `setup_dir`. Returns the path of the .npz.

    `masses` ([N, n_macros], row-aligned to `flat_strategy`) + `kinds` make this
    a macro strategy: the concrete `probs` are the per-infoset `c/T` and each
    macro's mass `m_k/T` rides in `masses`, on the same scale (concrete + masses
    sum to the row total). The agent resolves each macro to a per-hand bet and
    folds its mass there at serve. Omit both for a plain concrete strategy."""
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
    extra = {}
    if masses is not None and kinds:
        extra["masses"] = np.asarray(masses)[order]
        extra["kinds"] = np.array(list(kinds), dtype=object)
    saver(
        npz_path,
        version=np.int32(STRATEGY_NPZ_VERSION),
        keys=keys_sorted,
        lower=lower_sorted,
        upper=upper_sorted,
        probs=probs_sorted,
        min_bet=np.int32(flat_strategy.min_bet),
        **extra,
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

@dataclass
class FlatStrategyAgent:
    """Lightweight strategy container for the deployed agent.

    Two underlying storage formats, both expose the same `lookup()` and
    `get_strategy(row)` interface:

    * **Padded fixed-width** (`_padded_probs` set): `[N, 89]` fp32 array.
      Used when loading from a legacy `strategy.npz` (compressed,
      single-file). Wastes space on zero-padded illegal actions but
      works as a local-dev fallback.

    * **Variable-width flat** (`_flat_probs` + `_probs_offset` set):
      `_flat_probs` is a 1-D fp32 array containing each row's legal
      action probabilities concatenated end-to-end. `_probs_offset[i]`
      gives the start index for row i; row i's strategy slice is
      `_flat_probs[_probs_offset[i] : _probs_offset[i+1]]`. ~3-5×
      smaller than the padded layout (no zero-padding for illegal
      actions), and the underlying file is `mmap`able so peak resident
      memory is ~10-50 MB regardless of strategy size. Produced by
      `write_mmap_layout` at Docker-build time.
    """
    keys_sorted: np.ndarray   # int64[N], sorted
    lower_action: np.ndarray  # int16[N]
    upper_action: np.ndarray  # int16[N]
    abs_str_to_id: Dict[str, int]
    min_bet: int
    # Exactly one of these representations is set:
    # 1. Padded fallback: full N × 89 fp32 array.
    _padded_probs: Optional[np.ndarray] = None
    # 2. Sparse mmap (new): only non-zero values + their position within
    #    the legal range per row. ~10× smaller than dense int16 because
    #    strategies are typically 5-10% dense post `clear_lows`.
    _sparse_indices: Optional[np.ndarray] = None  # uint8, 1D
    _sparse_values: Optional[np.ndarray] = None   # int16, 1D
    _probs_offset: Optional[np.ndarray] = None    # int64[N+1]
    # Augmenting macros (V3+). When present, `kinds` lists the macro kinds and
    # `masses[row]` holds each macro's probability mass (`m_k/T`), on the SAME
    # scale as the row's concrete `probs` slice (`c/T`): concrete + masses sum to
    # the row total. The agent resolves each macro to a per-hand bet `b*` and
    # folds its mass there at serve. `None` for plain concrete strategies.
    masses: Optional[np.ndarray] = None  # [N, n_macros]
    kinds: Optional[List[str]] = None

    def lookup(self, comp_key: int) -> Optional[int]:
        """O(log N) binary search. Returns row index or None if absent."""
        k = np.int64(comp_key)
        idx = np.searchsorted(self.keys_sorted, k)
        if idx < self.keys_sorted.shape[0] and self.keys_sorted[idx] == k:
            return int(idx)
        return None

    def get_strategy(self, row: int) -> np.ndarray:
        """Return the strategy slice for `row`, length = upper - lower + 1.

        Sparse path reconstructs a dense int16 array on the fly (~60 byte
        allocation per call, no perceptible cost). Padded fallback
        returns the appropriate fp32 slice directly. Callers sum-and-
        divide so the absolute scale (int16 vs fp32) doesn't matter."""
        if self._sparse_indices is not None:
            s = int(self._probs_offset[row])
            e = int(self._probs_offset[row + 1])
            width = int(self.upper_action[row]) - int(self.lower_action[row]) + 1
            dense = np.zeros(width, dtype=np.uint16)
            idx = self._sparse_indices[s:e]
            val = self._sparse_values[s:e]
            dense[idx] = val
            return dense
        # padded fp32 fallback
        lo = int(self.lower_action[row])
        hi = int(self.upper_action[row])
        return self._padded_probs[row, lo:hi + 1]


def _check_version(data, label: str) -> None:
    """Validate the schema version stamped in `data` (a loaded npz)."""
    if "version" not in data:
        return
    v = int(data["version"])
    if v != STRATEGY_NPZ_VERSION:
        raise ValueError(
            f"{label} version {v} != supported {STRATEGY_NPZ_VERSION}"
        )


def _read_npz_arrays(setup_dir: str):
    """Open strategy.npz + its abs.json companion, validate version, return
    the loose ingredients. Shared by both loader entry points."""
    npz_path = os.path.join(setup_dir, "strategy.npz")
    abs_path = os.path.join(setup_dir, "strategy.abs.json")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"strategy.npz not found at {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    _check_version(data, "strategy.npz")
    arrays = dict(
        keys=data["keys"],
        lower=data["lower"],
        upper=data["upper"],
        probs=data["probs"],
        min_bet=int(data["min_bet"]),
        masses=data["masses"] if "masses" in data.files else None,
        kinds=([str(x) for x in data["kinds"]] if "kinds" in data.files else None),
    )
    if os.path.exists(abs_path):
        with open(abs_path, "r", encoding="utf-8") as f:
            arrays["abs_str_to_id"] = json.load(f)
    else:
        arrays["abs_str_to_id"] = {}
    return arrays


def load_strategy_for_agent(setup_dir: str) -> FlatStrategyAgent:
    """Agent-facing loader. Two-format aware:

    1. **mmap path** (preferred at deployment time): if `setup_dir` has a
       `probs_flat.npy` next to `strategy_meta.npz`, mmap the variable-
       width flat probs array and load the small meta arrays eagerly.
       The flat representation stores only the legal action slice per
       row (no padding for illegal actions) → ~3-5× smaller than the
       padded layout. Peak resident memory is ~10-50 MB regardless of
       strategy size.
    2. **Compressed `.npz` fallback** (used for local dev where the
       single-file format is more convenient): if only `strategy.npz` is
       present, read everything eagerly. Costs the full padded probs
       allocation but keeps `cfr_ai/outputs/` compact on disk.

    `cfr_ai/scripts/stage_for_docker.py` is responsible for converting
    each setup's compressed `strategy.npz` into the flat-mmap layout
    when building the Lambda image — so the build context gets the
    mmap-friendly form while the source tree stays compressed."""
    sparse_idx = os.path.join(setup_dir, "probs_sparse_indices.npy")
    sparse_val = os.path.join(setup_dir, "probs_sparse_values.npy")
    meta_npz = os.path.join(setup_dir, "strategy_meta.npz")
    if (os.path.exists(sparse_idx) and os.path.exists(sparse_val)
            and os.path.exists(meta_npz)):
        return _load_mmap(setup_dir)

    # Fallback: read the legacy single compressed file.
    a = _read_npz_arrays(setup_dir)
    return FlatStrategyAgent(
        keys_sorted=a["keys"],
        _padded_probs=a["probs"],
        lower_action=a["lower"],
        upper_action=a["upper"],
        abs_str_to_id=a["abs_str_to_id"],
        min_bet=a["min_bet"],
        masses=a["masses"],
        kinds=a["kinds"],
    )


def _load_mmap(setup_dir: str) -> FlatStrategyAgent:
    """Mmap sparse_indices + sparse_values + eager-load the small meta arrays."""
    meta = np.load(os.path.join(setup_dir, "strategy_meta.npz"), allow_pickle=True)
    _check_version(meta, "strategy_meta.npz")
    sparse_indices = np.load(
        os.path.join(setup_dir, "probs_sparse_indices.npy"), mmap_mode="r")
    sparse_values = np.load(
        os.path.join(setup_dir, "probs_sparse_values.npy"), mmap_mode="r")
    abs_path = os.path.join(setup_dir, "strategy.abs.json")
    if os.path.exists(abs_path):
        with open(abs_path, "r", encoding="utf-8") as f:
            abs_str_to_id = json.load(f)
    else:
        abs_str_to_id = {}
    return FlatStrategyAgent(
        keys_sorted=np.asarray(meta["keys"]),
        lower_action=np.asarray(meta["lower"]),
        upper_action=np.asarray(meta["upper"]),
        abs_str_to_id=abs_str_to_id,
        min_bet=int(meta["min_bet"]),
        _sparse_indices=sparse_indices,
        _sparse_values=sparse_values,
        _probs_offset=np.asarray(meta["probs_offset"]),
        masses=(np.asarray(meta["masses"]) if "masses" in meta.files else None),
        kinds=([str(x) for x in meta["kinds"]] if "kinds" in meta.files else None),
    )


def write_mmap_layout(setup_dir_src: str, setup_dir_dst: str) -> None:
    """Convert a setup's compressed padded `strategy.npz` into the
    variable-width mmap layout in `setup_dir_dst`:

      * `strategy_meta.npz` (compressed): keys, lower, upper, min_bet,
        probs_offset (the prefix-sum of legal-action widths per row).
      * `probs_flat.npy` (uncompressed): each row's legal-action
        probability slice concatenated end-to-end. mmappable.
      * `strategy.abs.json`: copied unchanged.

    The flat layout drops zero-padding for illegal actions → ~3-5×
    smaller than the padded `.npz` would be uncompressed, while still
    supporting `mmap_mode='r'` so the agent's peak memory stays tiny."""
    a = _read_npz_arrays(setup_dir_src)
    keys = a["keys"]; lower = a["lower"]; upper = a["upper"]; probs = a["probs"]
    masses_src = a["masses"]; kinds = a["kinds"]
    n = keys.shape[0]
    has_macros = masses_src is not None and kinds is not None and len(kinds) > 0
    n_macros = int(masses_src.shape[1]) if has_macros else 0
    masses_out = np.zeros((n, n_macros), dtype=np.uint16) if has_macros else None

    # Single pass: re-normalise each row, scale by 65535, round to uint16, and
    # stash only the non-zero (index, value) pairs. Strategies are typically
    # 5-10% dense after `clear_lows`, so this drops ~90% of the storage volume.
    # For macro strategies the per-row total includes the macro masses, so the
    # concrete slice AND the masses are scaled by the SAME factor — they stay on
    # one comparable scale for the agent's serve-time fold (concrete + masses).
    parts_idx: List[np.ndarray] = []
    parts_val: List[np.ndarray] = []
    nz_per_row = np.zeros(n, dtype=np.int64)
    for i in range(n):
        lo, hi = int(lower[i]), int(upper[i])
        row_slice = probs[i, lo:hi + 1]
        row_total = float(row_slice.sum())
        if has_macros:
            row_total += float(masses_src[i].sum())
        if row_total > 0:
            scaled = np.clip(np.round(row_slice / row_total * 65535.0), 0, 65535).astype(np.uint16)
            if has_macros:
                masses_out[i] = np.clip(np.round(masses_src[i] / row_total * 65535.0),
                                        0, 65535).astype(np.uint16)
        else:
            scaled = np.zeros(hi - lo + 1, dtype=np.uint16)
        nz = np.flatnonzero(scaled)
        parts_idx.append(nz.astype(np.uint8))
        parts_val.append(scaled[nz])
        nz_per_row[i] = nz.size

    sparse_indices = (np.concatenate(parts_idx) if parts_idx
                      else np.empty(0, dtype=np.uint8))
    sparse_values = (np.concatenate(parts_val) if parts_val
                     else np.empty(0, dtype=np.uint16))
    probs_offset = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(nz_per_row, out=probs_offset[1:])

    os.makedirs(setup_dir_dst, exist_ok=True)
    meta = dict(
        version=np.int32(STRATEGY_NPZ_VERSION),
        keys=keys,
        lower=lower,
        upper=upper,
        min_bet=np.int32(a["min_bet"]),
        probs_offset=probs_offset,
    )
    if has_macros:
        meta["masses"] = masses_out
        meta["kinds"] = np.array(kinds, dtype=object)
    np.savez_compressed(os.path.join(setup_dir_dst, "strategy_meta.npz"), **meta)
    np.save(os.path.join(setup_dir_dst, "probs_sparse_indices.npy"), sparse_indices)
    np.save(os.path.join(setup_dir_dst, "probs_sparse_values.npy"), sparse_values)

    abs_path_src = os.path.join(setup_dir_src, "strategy.abs.json")
    if os.path.exists(abs_path_src):
        with open(abs_path_src, "r", encoding="utf-8") as f:
            abs_str_to_id = json.load(f)
        with open(os.path.join(setup_dir_dst, "strategy.abs.json"),
                  "w", encoding="utf-8") as f:
            json.dump(abs_str_to_id, f, ensure_ascii=False)


def save_macro_strategy(setup_dir: str, *, keys, lower, upper, probs,
                        masses, kinds, min_bet) -> str:
    """Write the unified deployment `strategy.npz` for a MACRO model directly
    from arrays: int64 composite `keys`, the concrete `c/T` `probs` (padded
    [N,89]), per-row macro `masses` ([N,n_macros], on the same scale as the
    concrete slice), the `kinds`, and `min_bet`. Sorts by key so the on-disk
    form is canonical / binary-searchable, matching `save_strategy`."""
    keys = np.asarray(keys, dtype=np.int64)
    order = np.argsort(keys, kind="stable")
    os.makedirs(setup_dir, exist_ok=True)
    path = os.path.join(setup_dir, "strategy.npz")
    np.savez_compressed(
        path,
        version=np.int32(STRATEGY_NPZ_VERSION),
        keys=keys[order],
        lower=np.asarray(lower, np.int16)[order],
        upper=np.asarray(upper, np.int16)[order],
        probs=np.asarray(probs, np.float32)[order],
        min_bet=np.int32(min_bet),
        masses=np.asarray(masses, np.float32)[order],
        kinds=np.array(list(kinds), dtype=object),
    )
    return path


def load_strategy(setup_dir: str):
    """Full-fidelity loader for training/LBR/subgame solving. Builds a
    numba `typed.Dict[int64,int64]` (`~3s/M entries`) so the JIT
    recursion in `lbr.py` can get O(1) lookups. For the deployed agent
    use `load_strategy_for_agent()` instead — its `np.searchsorted` is
    fast enough for the one-lookup-per-request pattern and skips the
    typed.Dict cost entirely."""
    from numba import types
    from numba.typed import Dict as NbDict
    from cfr_ai.lbr import FlatStrategy

    a = _read_npz_arrays(setup_dir)
    keys = a["keys"]

    # Build the numba typed dict (row index by composite int64 key).
    key_to_row = NbDict.empty(key_type=types.int64, value_type=types.int64)
    for i in range(keys.shape[0]):
        key_to_row[np.int64(keys[i])] = np.int64(i)

    return FlatStrategy(
        key_to_row=key_to_row,
        strategy=a["probs"],
        lower_action=a["lower"],
        upper_action=a["upper"],
        abs_str_to_id=a["abs_str_to_id"],
        min_bet=a["min_bet"],
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
