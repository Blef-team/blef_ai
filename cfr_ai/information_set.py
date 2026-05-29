from typing import List
import numpy as np

history_codes = np.genfromtxt('cfr_ai/history.csv', delimiter=',', dtype='|U5', skip_header=0)


def get_possible_actions(history: List[int], min_bet: int) -> List[int]:
    if (len(history) == 0 or history[-1] < min_bet):
        return [a for a in range(min_bet, 88)]
    else: 
        return [a for a in range(history[-1] + 1, 89)]

def get_hand_abstraction(hand: List[int], hand_sizes: List[int]) -> List[str]:
    """A 89-entry hand abstraction (one per last-bet context; index 88 = round start). 
    Rounds 1-5 (sum(hand_sizes) <= 6) use a cheap sorted-values abstraction; 
    rounds 6+ use the main abstraction."""

    vals = [c // 4 for c in hand]
    suits = [c % 4 for c in hand]
    # Rounds 1-5: just the sorted value multiset.
    if sum(hand_sizes) <= 6:
        return [''.join(sorted(str(v) for v in vals))] * 89

    # counts: index 0-3 = suit counts, 4-9 = value counts (values 0-5).
    counts = [0] * 10
    for v in vals:
        counts[v + 4] += 1
    for s in suits:
        counts[s] += 1
    # strengths: N of a value scores above N of a suit (suits offset by -10).
    base = (-10, -9, -8, -7, 4, 5, 6, 7, 8, 9)
    strengths = [base[i] + 10 * counts[i] for i in range(10)]
    top = max(strengths)
    # sf_strengths: suit strengths augmented with 9s (+0.1) and aces (+0.2).
    sf = [float(strengths[i]) for i in range(4)]
    for v, s in zip(vals, suits):
        if v == 0:
            sf[s] += 0.1
        elif v == 5:
            sf[s] += 0.2
    sf = [round(x, 1) for x in sf]
    aug_vals = [float(strengths[i]) for i in range(4, 10)]
    max_aug = max(sf + aug_vals)

    # Pre-straight: 4-of-a-kind/flush -> top; else 3-of-a-kind -> top 2 value
    # strengths; else the sorted value multiset.
    if top >= 40:
        pre = str(top)
    elif max(strengths[4:10]) >= 34:
        s2 = sorted(strengths[4:10], reverse=True)
        pre = str(s2[0]) + ' ' + str(s2[1])
    else:
        pre = ''.join(sorted(str(v) for v in vals))
    out = [pre] * 27

    # Straight -> full house.
    if top >= 40:
        out += [str(top)] * 39
    else:
        straight_part = (str(min(counts[4], 1))
                         + str(sum(min(counts[k], 1) for k in range(5, 9)))
                         + str(min(counts[9], 1)) + ' ' + str(top))
        out += [straight_part] * 3
        for i in range(4, 10):                       # three of a kind
            out.append(str(counts[i]) + ' ' + str(top))
        first = second = 0                            # full house (30 pairs)
        for _ in range(30):
            second += 1
            if second == 6:
                first += 1
                second = 0
            if second == first:
                second += 1
            e1, e2 = 4 + first, 4 + second
            m = max(strengths[k] for k in range(10) if k != e1 and k != e2)
            out.append(str(counts[e1]) + str(counts[e2]) + ' ' + str(m))

    # Flush.
    for i in range(4):
        out.append(str(counts[i]) + ' ' + str(max_aug))
    # Four of a kind (progressively drop the low value strengths).
    for k in range(6):
        out.append(str(counts[4 + k]) + ' ' + str(max(sf + aug_vals[k + 1:])))
    # Small / big straight flush (identical twice in the original).
    for _ in range(2):
        for j in range(4):
            out.append(str(sf[j]) + ' ' + str(max(sf[t] for t in range(4) if t != j)))
    # Great straight flush (progressively drop the low suits).
    for j in range(3):
        out.append(str(sf[j]) + ' ' + str(max(sf[j + 1:])))
    out.append('X')        # great straight flush spades: hand-independent
    out.append(pre)        # index 88: round start, same as pre-straight
    return out


def make_key(hand: List[int], hand_abstractions: List[str], history: List[int], min_bet: int) -> str:
    key = str(len(hand))
    
    # History abstraction
    last_bet = 88 if len(history) == 0 or history[-1] < min_bet else history[-1]
    key += '-' + str(last_bet) + '-'
    if len(history) > 1 and history[-2] >= min_bet:
        key += history_codes[last_bet, history[-2]] + '-'
        if len(history) > 2 and history[-3] >= min_bet:
            key += history_codes[last_bet, history[-3]] + '-'
    
    # Hand abstraction
    key += hand_abstractions[last_bet]

    return key


