"""Numba-JIT existence kernel — a bit-exact, ~100x faster drop-in for the pure-
Python `probs.p_vector_factored`. Used to build resolve tables; removes the
per-signature fill cost that dominated high-T training.

Exactness: all binomials here are <= C(24,12) and every product/sum stays under
~1e9 (the unseen pool is <= 22 cards), so float64 represents them exactly. The
JIT result is therefore IDENTICAL to the integer-arithmetic kernel (verified in
__main__: max abs diff 0.0). The verified claim taxonomy/order is unchanged.
"""
import os as _os
import sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import numpy as np
from numba import njit

from cfr_ai.abstraction.probs import TWO_PAIR, _fh

# Pascal's triangle up to n=24 (float64-exact in this range).
_MAXN = 25
_BINOM = np.zeros((_MAXN, _MAXN), dtype=np.float64)
for _n in range(_MAXN):
    _BINOM[_n, 0] = 1.0
    for _k in range(1, _n + 1):
        _BINOM[_n, _k] = _BINOM[_n - 1, _k - 1] + _BINOM[_n - 1, _k]

# Static claim->value maps, baked into the JIT as constants.
_TP = np.array([[v1, v2] for (_b, v1, v2) in TWO_PAIR], dtype=np.int64)   # bets 12..26
_FH = np.array([list(_fh(b)) for b in range(36, 66)], dtype=np.int64)     # bets 36..65


@njit(cache=False)
def _cb(n, k):
    if k < 0 or n < 0 or k > n:
        return 0.0
    return _BINOM[n, k]


@njit(cache=False)
def _ge_j(need, good, other, draws):
    if need <= 0:
        return 1.0
    pool = good + other
    tot = _cb(pool, draws)
    if tot == 0.0:
        return 0.0
    fail = 0.0
    for i in range(need):
        fail += _cb(good, i) * _cb(other, draws - i)
    return 1.0 - fail / tot


@njit(cache=False)
def _two_j(n1, n2, g1, g2, pool, draws):
    if n1 <= 0 and n2 <= 0:
        return 1.0
    other = pool - g1 - g2
    tot = _cb(pool, draws)
    if tot == 0.0:
        return 0.0
    s = 0.0
    imax = min(draws, g1)
    for i in range(max(n1, 0), imax + 1):
        jmax = min(draws - i, g2)
        for j in range(max(n2, 0), jmax + 1):
            rem = draws - i - j
            if 0 <= rem <= other:
                s += _cb(g1, i) * _cb(g2, j) * _cb(other, rem)
    return s / tot


@njit(cache=False)
def _full_j(n3, n2, g3, g2, pool, draws):
    if n3 <= 0 and n2 <= 0:
        return 1.0
    other = pool - g3 - g2
    tot = _cb(pool, draws)
    if tot == 0.0:
        return 0.0
    s = 0.0
    imax = min(draws, g3)
    for i in range(max(n3, 0), imax + 1):
        jmax = min(draws - i, g2)
        for j in range(max(n2, 0), jmax + 1):
            rem = draws - i - j
            if 0 <= rem <= other:
                s += _cb(g3, i) * _cb(g2, j) * _cb(other, rem)
    return s / tot


@njit(cache=False)
def _straight_j(m, pool, draws):
    if m == 0:
        return 1.0
    tot = _cb(pool, draws)
    if tot == 0.0:
        return 0.0
    s = 0.0
    for r in range(m + 1):
        term = _cb(m, r) * _cb(pool - 4 * r, draws)
        if r % 2 == 0:
            s += term
        else:
            s -= term
    return s / tot


@njit(cache=False)
def p_vector_jit(val, suit, vxs, draws):
    pool = 24
    for v in range(6):
        pool -= val[v]
    e = np.zeros(88)
    for v in range(6):
        good = 4 - val[v]
        other = pool - good
        e[v]      = _ge_j(1 - val[v], good, other, draws)
        e[6 + v]  = _ge_j(2 - val[v], good, other, draws)
        e[30 + v] = _ge_j(3 - val[v], good, other, draws)
        e[70 + v] = _ge_j(4 - val[v], good, other, draws)
    for t in range(15):
        v1 = _TP[t, 0]; v2 = _TP[t, 1]
        e[12 + t] = _two_j(2 - val[v1], 2 - val[v2], 4 - val[v1], 4 - val[v2], pool, draws)
    m27 = 0
    for v in range(0, 5):
        if val[v] == 0:
            m27 += 1
    e[27] = _straight_j(m27, pool, draws)
    m28 = 0
    for v in range(1, 6):
        if val[v] == 0:
            m28 += 1
    e[28] = _straight_j(m28, pool, draws)
    m29 = 0
    for v in range(0, 6):
        if val[v] == 0:
            m29 += 1
    e[29] = _straight_j(m29, pool, draws)
    for t in range(30):
        v3 = _FH[t, 0]; v2 = _FH[t, 1]
        e[36 + t] = _full_j(3 - val[v3], 2 - val[v2], 4 - val[v3], 4 - val[v2], pool, draws)
    for s in range(4):
        good = 6 - suit[s]
        e[66 + s] = _ge_j(5 - suit[s], good, pool - good, draws)
    for bet in range(76, 88):
        suit_i = bet % 4
        if bet < 80:
            lo_v, hi_v = 0, 4
        elif bet < 84:
            lo_v, hi_v = 1, 5
        else:
            lo_v, hi_v = 0, 5
        have = 0
        for v in range(lo_v, hi_v + 1):
            if vxs[v, suit_i] > 0:
                have += 1
        need = (hi_v - lo_v + 1) - have
        e[bet] = _ge_j(need, need, pool - need, draws)
    return e


def p_vector_fast(my_cards, n_opp):
    """Drop-in for probs.p_vector_factored (bit-exact, JIT-fast)."""
    val = np.zeros(6, np.int64)
    suit = np.zeros(4, np.int64)
    vxs = np.zeros((6, 4), np.int64)
    for c in my_cards:
        v, s = int(c) // 4, int(c) % 4
        val[v] += 1
        suit[s] += 1
        vxs[v, s] += 1
    return p_vector_jit(val, suit, vxs, int(n_opp))


if __name__ == "__main__":
    import sys, os, time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from cfr_ai.abstraction.probs import p_vector, p_vector_factored
    rng = np.random.default_rng(0)
    p_vector_fast([0, 5], 3)  # warm/compile
    worst = 0.0
    for _ in range(5000):
        k = int(rng.integers(1, 12)); n_opp = int(rng.integers(1, 12))
        if k + n_opp > 24:
            continue
        hand = rng.choice(24, size=k, replace=False).tolist()
        worst = max(worst, float(np.max(np.abs(p_vector(hand, n_opp) - p_vector_fast(hand, n_opp)))))
    print(f"JIT vs integer-exact kernel: max abs diff = {worst:.2e}  "
          f"({'BIT-EXACT' if worst == 0.0 else ('OK' if worst < 1e-9 else 'MISMATCH')})")
    # speed: fill 30k distinct-ish 11-card hands
    hands = [rng.choice(24, size=11, replace=False).tolist() for _ in range(30000)]
    t = time.time(); [p_vector_factored(h, 9) for h in hands]; slow = time.time() - t
    t = time.time(); [p_vector_fast(h, 9) for h in hands]; fast = time.time() - t
    print(f"30k 11-card builds: pure-Python {slow:.2f}s | JIT {fast:.2f}s | speedup {slow/fast:.0f}x")
