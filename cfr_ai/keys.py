"""Numba-free composite key utilities for the Blef CFR AI.

Defines the bit-shift layout for the composite int64 infoset key, the
history-code id matrix, and the `_split_suffix` parser used by both the
deployed agent and the LBR loader. Lives in its own module so the
agent's Lambda runtime can import the key-handling pieces without
pulling in `numba` / `llvmlite` (which `cfr_ai/trainer.py` and
`cfr_ai/lbr.py` both pull in at module top via `@njit` decorators).

Composite key layout (int64):
    [abs_id : 36][h_m2 : 8][h_m1 : 8][last_bet : 8][hand_size : 4]
"""

from typing import Dict, List, Tuple

import numpy as np

from cfr_ai.information_set import history_codes


# ---------------------------------------------------------------------------
# Composite-key bit layout
# ---------------------------------------------------------------------------

LAST_BET_SHIFT = 4
H_M1_SHIFT = 12
H_M2_SHIFT = 20
ABS_ID_SHIFT = 28

ABSENT_CODE = 255

# Max history depth (each player adds at most one bet > previous; +slack).
# Re-exported for code that wants it without the numba-heavy trainer import.
MAX_DEPTH = 92


# ---------------------------------------------------------------------------
# History-code-id matrix, built once from cfr_ai/history.csv
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

# Inverse of _HISTORY_CODE_STRS: code-string -> uint8 id. Built once.
_STR_TO_CODE_ID: Dict[str, int] = {s: i for i, s in enumerate(_HISTORY_CODE_STRS)}


# ---------------------------------------------------------------------------
# Key suffix parser (used by both agent.py and lbr.py)
# ---------------------------------------------------------------------------

def _split_suffix(suffix: str) -> Tuple[int, int, str]:
    """Split a strategy-file key suffix into (h_m1_id, h_m2_id, abs_str).

    Suffix structure as emitted by trainer/agent is one of:
      - "abs"              -> 0 history codes
      - "c1-abs"           -> 1 history code
      - "c1-c2-abs"        -> 2 history codes
    Where c1/c2 come from `_STR_TO_CODE_ID` (history-code strings).

    Complication: the hand-abstraction string for rounds 6+ can contain
    hyphens (e.g. negative numbers like "-9.0 -7.0"), so a naive
    `suffix.split('-')` mis-attributes leading abs tokens as history
    codes. The robust rule: peel off up to 2 leading tokens that are
    valid non-empty history codes; the remainder is the abs. A leading
    '-' means the abs starts with a minus, so no codes to peel.

    The empty string is in `_STR_TO_CODE_ID` (history.csv has empty
    placeholder cells) but is never a real code in a key.
    """
    ids = [ABSENT_CODE, ABSENT_CODE]
    remainder = suffix
    for i in range(2):
        if "-" not in remainder or remainder.startswith("-"):
            break
        head, rest = remainder.split("-", 1)
        if not head or head not in _STR_TO_CODE_ID:
            break
        ids[i] = _STR_TO_CODE_ID[head]
        remainder = rest
    return ids[0], ids[1], remainder
