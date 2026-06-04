"""24-card Blef existence-probability kernel (no jokers / no blanks).

Specialized, fast, closed-form (hypergeometric) reimplementation of the claim-
existence probabilities, verified bit-exact against the general-but-slow oracle
`shared.probabilities.dynamic_probabilities`. This is the foundation the action
abstraction consumes:

  prob_vector(known_cards, draws) -> P(claim i exists), 88-vector
        known_cards = card ints (0..23) already visible to the actor
        draws       = #cards drawn by the rest from the unseen pool

  g_vector(total_cards)     = prob_vector([], total_cards)          # a-priori / generic
  p_vector(my_cards, n_opp) = prob_vector(my_cards, n_opp)          # a-posteriori / mine
  bet_floor(total_cards)    = highest claim certain a-priori (g == 1) -> principled
                              low-truncation; the a-priori basis for `min_bet`.

Claim layout (verified == game.py == game_utils): high 0-5, pair 6-11,
two_pair 12-26, straight 27-29, trips 30-35, full_house 36-65, flush 66-69,
quads 70-75, straight_flush 76-83, great_sf 84-87. (88 = check, not a claim.)

Pure-Python/closed-form for now (correctness + amortized per-hand use). A numba
port is deferred to trainer integration; per-node cost stays zero either way
because menus/resolutions are precomputed per hand, never recomputed per node.
"""
import os
import sys
from math import comb
from functools import lru_cache

import numpy as np

NVAL, NSUIT, NDECK, NCLAIM = 6, 4, 24, 88


def C(n, k):
    """Guarded binomial: 0 for out-of-range k (matches the oracle's binom)."""
    return comb(n, k) if 0 <= k <= n else 0

TWO_PAIR = [(12, 1, 0), (13, 2, 0), (14, 2, 1), (15, 3, 0), (16, 3, 1), (17, 3, 2),
            (18, 4, 0), (19, 4, 1), (20, 4, 2), (21, 4, 3),
            (22, 5, 0), (23, 5, 1), (24, 5, 2), (25, 5, 3), (26, 5, 4)]


def _fh(bet):
    if bet < 41:    return 0, bet - 35
    elif bet == 41: return 1, 0
    elif bet < 46:  return 1, bet - 40
    elif bet < 48:  return 2, bet - 46
    elif bet < 51:  return 2, bet - 45
    elif bet < 54:  return 3, bet - 51
    elif bet < 56:  return 3, bet - 50
    elif bet < 60:  return 4, bet - 56
    elif bet == 60: return 4, 5
    else:           return 5, bet - 61


def _ge(need, good, other, draws):
    """P(draw >= `need` of the `good` cards) when drawing `draws` from good+other."""
    if need <= 0:
        return 1.0
    pool = good + other
    tot = C(pool, draws)
    if tot == 0:
        return 0.0
    fail = 0
    for i in range(need):                        # i = #good drawn (a failure if < need)
        fail += C(good, i) * C(other, draws - i)
    return 1.0 - fail / tot


def _two(n1, n2, g1, g2, pool, draws):
    """P(>=2 of value1 AND >=2 of value2) via bivariate hypergeometric."""
    if n1 <= 0 and n2 <= 0:
        return 1.0
    other = pool - g1 - g2
    tot = C(pool, draws)
    if tot == 0:
        return 0.0
    s = 0
    for i in range(min(draws, g1) + 1):
        if i < n1:
            continue
        for j in range(min(draws - i, g2) + 1):
            if j < n2:
                continue
            rem = draws - i - j
            if 0 <= rem <= other:
                s += C(g1, i) * C(g2, j) * C(other, rem)
    return s / tot


def _full(n3, n2, g3, g2, pool, draws):
    """P(>=3 of value3 AND >=2 of value2)."""
    if n3 <= 0 and n2 <= 0:
        return 1.0
    other = pool - g3 - g2
    tot = C(pool, draws)
    if tot == 0:
        return 0.0
    s = 0
    for i in range(min(draws, g3) + 1):
        if i < n3:
            continue
        for j in range(min(draws - i, g2) + 1):
            if j < n2:
                continue
            rem = draws - i - j
            if 0 <= rem <= other:
                s += C(g3, i) * C(g2, j) * C(other, rem)
    return s / tot


def _straight(vals, val_counts, pool, draws):
    """P(every value in `vals` is present in the pool), given some already held."""
    needed = [v for v in vals if val_counts[v] == 0]       # absent ranks (4 copies each)
    if not needed:
        return 1.0
    tot = C(pool, draws)
    if tot == 0:
        return 0.0
    m = len(needed)
    s = 0.0
    for r in range(m + 1):                                  # inclusion-exclusion over absences
        s += ((-1) ** r) * C(m, r) * C(pool - 4 * r, draws)
    return s / tot


def prob_vector(known_cards, draws):
    """88-vector of P(claim exists) given `known_cards` (ints 0..23) seen by the
    actor and `draws` cards drawn by the rest from the unseen pool."""
    val = [0] * NVAL
    suit = [0] * NSUIT
    vxs = [[0] * NSUIT for _ in range(NVAL)]
    for c in known_cards:
        v, s = c // 4, c % 4
        val[v] += 1
        suit[s] += 1
        vxs[v][s] += 1
    pool = NDECK - len(known_cards)

    e = [0.0] * NCLAIM
    for v in range(NVAL):
        good = 4 - val[v]
        other = pool - good
        e[v]      = _ge(1 - val[v], good, other, draws)
        e[6 + v]  = _ge(2 - val[v], good, other, draws)
        e[30 + v] = _ge(3 - val[v], good, other, draws)
        e[70 + v] = _ge(4 - val[v], good, other, draws)
    for bet, v1, v2 in TWO_PAIR:
        e[bet] = _two(max(0, 2 - val[v1]), max(0, 2 - val[v2]),
                      4 - val[v1], 4 - val[v2], pool, draws)
    e[27] = _straight((0, 1, 2, 3, 4), val, pool, draws)
    e[28] = _straight((1, 2, 3, 4, 5), val, pool, draws)
    e[29] = _straight((0, 1, 2, 3, 4, 5), val, pool, draws)
    for bet in range(36, 66):
        v3, v2 = _fh(bet)
        e[bet] = _full(max(0, 3 - val[v3]), max(0, 2 - val[v2]),
                       4 - val[v3], 4 - val[v2], pool, draws)
    for s in range(NSUIT):
        good = (NDECK // NSUIT) - suit[s]
        e[66 + s] = _ge(5 - suit[s], good, pool - good, draws)
    for bet in range(70, 76):
        pass  # quads already set above (70+v)
    for bet in range(76, 88):
        suit_i = bet % 4
        vals = (0, 1, 2, 3, 4) if bet < 80 else ((1, 2, 3, 4, 5) if bet < 84 else (0, 1, 2, 3, 4, 5))
        have = sum(1 for v in vals if vxs[v][suit_i])
        need = len(vals) - have
        e[bet] = _ge(need, need, pool - need, draws)         # need distinct specific cards
    return np.array(e, dtype=np.float64)


@lru_cache(maxsize=64)
def g_vector(total_cards):
    return prob_vector((), int(total_cards))


def p_vector(my_cards, n_opp):
    return prob_vector(tuple(sorted(int(c) for c in my_cards)), int(n_opp))


@lru_cache(maxsize=64)
def bet_floor(total_cards):
    """Highest claim index that is certain to exist a priori (g == 1.0); claims at
    or below it are pointless to bet. The principled analog of `min_bet`."""
    g = g_vector(total_cards)
    floor = -1
    for i in range(NCLAIM):
        if g[i] >= 1.0 - 1e-9:
            floor = i
    return floor


# --------------------------------------------------------------------------- #
# Factorized p: claims 0..75 depend only on (rank counts, suit counts); cache
# that expensive part by signature. Claims 76..87 (straight flush) need the
# rank-suit incidence, so they're computed per hand (cheap: 12 simple terms).
# Cuts high-T p-compute from O(C(24,k)) hands to O(#distinct count-signatures).
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=200000)
def _p_counts(val, suit, draws):
    """Claims 0..75 from rank-count tuple `val` (len 6) + suit-count tuple
    `suit` (len 4). pool = 24 - sum(val)."""
    pool = NDECK - sum(val)
    e = [0.0] * 76
    for v in range(NVAL):
        good = 4 - val[v]
        other = pool - good
        e[v]      = _ge(1 - val[v], good, other, draws)
        e[6 + v]  = _ge(2 - val[v], good, other, draws)
        e[30 + v] = _ge(3 - val[v], good, other, draws)
        e[70 + v] = _ge(4 - val[v], good, other, draws)
    for bet, v1, v2 in TWO_PAIR:
        e[bet] = _two(max(0, 2 - val[v1]), max(0, 2 - val[v2]),
                      4 - val[v1], 4 - val[v2], pool, draws)
    e[27] = _straight((0, 1, 2, 3, 4), val, pool, draws)
    e[28] = _straight((1, 2, 3, 4, 5), val, pool, draws)
    e[29] = _straight((0, 1, 2, 3, 4, 5), val, pool, draws)
    for bet in range(36, 66):
        v3, v2 = _fh(bet)
        e[bet] = _full(max(0, 3 - val[v3]), max(0, 2 - val[v2]),
                       4 - val[v3], 4 - val[v2], pool, draws)
    for s in range(NSUIT):
        good = (NDECK // NSUIT) - suit[s]
        e[66 + s] = _ge(5 - suit[s], good, pool - good, draws)
    return np.array(e, dtype=np.float64)


def p_vector_factored(my_cards, n_opp):
    """Same as p_vector but with the count-based part cached by signature."""
    cards = [int(c) for c in my_cards]
    val = [0] * NVAL
    suit = [0] * NSUIT
    vxs = [[0] * NSUIT for _ in range(NVAL)]
    for c in cards:
        v, s = c // 4, c % 4
        val[v] += 1
        suit[s] += 1
        vxs[v][s] += 1
    pool = NDECK - len(cards)
    e = np.empty(NCLAIM, dtype=np.float64)
    e[:76] = _p_counts(tuple(val), tuple(suit), int(n_opp))
    for bet in range(76, 88):                       # straight flush: needs incidence
        suit_i = bet % 4
        vals = (0, 1, 2, 3, 4) if bet < 80 else ((1, 2, 3, 4, 5) if bet < 84 else (0, 1, 2, 3, 4, 5))
        have = sum(1 for v in vals if vxs[v][suit_i])
        need = len(vals) - have
        e[bet] = _ge(need, need, pool - need, int(n_opp))
    return e


# --------------------------------------------------------------------------- #
# Verification + A1 summary (run as a script).
# --------------------------------------------------------------------------- #
def _verify():
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from shared.probabilities.dynamic_probabilities import calculate_prob
    rng = np.random.default_rng(7)
    rules = {"deck_size": 24, "jokers": 0, "blanks": 0}
    worst = 0.0
    nbad = 0
    NTRIAL = 3000
    for _ in range(NTRIAL):
        T = int(rng.integers(2, 23))
        k = int(rng.integers(0, T + 1))            # cards I hold
        deal = rng.choice(24, size=T, replace=False)
        mine = deal[:k]
        draws = T - k                              # opponents' cards from the unseen pool
        mine_vc = [(int(c) // 4, int(c) % 4) for c in mine]
        ref = np.array([calculate_prob(aid, mine_vc, [], [], draws, rules, True)
                        for aid in range(88)])
        got = prob_vector(mine.tolist(), draws)
        d = float(np.max(np.abs(ref - got)))
        worst = max(worst, d)
        if d > 1e-9:
            nbad += 1
    return NTRIAL, nbad, worst


BANDS = [(0, 5, "high"), (6, 11, "pair"), (12, 26, "2pair"), (27, 29, "straight"),
         (30, 35, "trips"), (36, 65, "fullhouse"), (66, 69, "flush"), (70, 75, "quads"),
         (76, 83, "sflush"), (84, 87, "great_sf")]
def _band(i):
    for lo, hi, n in BANDS:
        if lo <= i <= hi:
            return n
    return "?"
MINBET = {t: (0 if t <= 13 else (4 if t <= 15 else 27)) for t in range(2, 23)}


def _summary():
    print("\nPer-total a-priori landmarks (exact). knee = highest claim with g>=0.5; "
          "cliff = largest drop in best-achievable g above bet_floor.\n")
    print("| T | min_bet | bet_floor (band) | knee (band) | top cliff (band, drop) |")
    print("|---|---|---|---|---|")
    for T in range(2, 23):
        g = g_vector(T)
        bf = bet_floor(T)
        ge = np.nonzero(g >= 0.5)[0]
        knee = int(ge[-1]) if ge.size else int(np.argmax(g))
        M = np.maximum.accumulate(g[::-1])[::-1]
        lo = max(bf + 1, 0)
        dM = M[lo:-1] - M[lo + 1:]
        cliff = lo + int(np.argmax(dM)) if dM.size else lo
        bf_lbl = _band(bf) if bf >= 0 else "none"
        print(f"| {T} | {MINBET[T]} | {bf} ({bf_lbl}) | {knee} ({_band(knee)}) | "
              f"{cliff} ({_band(cliff)}, {M[cliff]-M[cliff+1]:.2f}) |")


if __name__ == "__main__":
    n, nbad, worst = _verify()
    print(f"verification vs dynamic_probabilities: {n} random (hand, draws) x 88 claims")
    print(f"  mismatches (>1e-9): {nbad}   max abs diff: {worst:.2e}")
    print("  VERDICT:", "EXACT MATCH" if nbad == 0 else "MISMATCH")
    _summary()
