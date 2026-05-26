from typing import List, Generator
import itertools
import numpy as np

BlefCards = np.arange(24, dtype=np.int64) 
rng = np.random.default_rng()

class Game():
    @staticmethod
    def check_finish(history: List[int]) -> bool:
        return len(history) > 0 and history[-1] == 88

    @staticmethod
    def precompute_set_existence(hands: List[List[int]]) -> np.ndarray:
        """
        Calculates the outcome for all 88 possible bets at once for a given deal.
        Returns a boolean numpy array of size 88.

        Optimisation: build value and (value, suit) tables once, then do all
        88 bet checks with plain Python integer comparisons. The previous
        implementation called np.sum / np.isin once per bet, which is ~100x
        more expensive than int lookups on these small arrays (2-22 cards)
        and dominated the per-iteration training cost.
        """
        # Count cards per value, and presence per (value, suit). Cards are
        # unique within a deal, so val_x_suit[v][s] is just a boolean.
        val_counts = [0] * 6
        val_x_suit = [[False] * 4 for _ in range(6)]
        suit_counts = [0] * 4
        for hand in hands:
            for card in hand:
                v = int(card) // 4
                s = int(card) % 4
                val_counts[v] += 1
                val_x_suit[v][s] = True
                suit_counts[s] += 1

        existence = [False] * 88

        # 0..5: high card (any card of value v)
        for v in range(6):
            existence[v] = val_counts[v] >= 1

        # 6..11: pair
        for v in range(6):
            existence[6 + v] = val_counts[v] >= 2

        # 12..26: two pair (mapping from the original code)
        two_pair_specs = (
            (12, 1, 0),
            (13, 2, 0), (14, 2, 1),
            (15, 3, 0), (16, 3, 1), (17, 3, 2),
            (18, 4, 0), (19, 4, 1), (20, 4, 2), (21, 4, 3),
            (22, 5, 0), (23, 5, 1), (24, 5, 2), (25, 5, 3), (26, 5, 4),
        )
        for bet, v1, v2 in two_pair_specs:
            existence[bet] = (val_counts[v1] >= 2) and (val_counts[v2] >= 2)

        # 27..29: straights
        s_low = (val_counts[0] >= 1 and val_counts[1] >= 1 and val_counts[2] >= 1
                 and val_counts[3] >= 1 and val_counts[4] >= 1)
        s_high = (val_counts[1] >= 1 and val_counts[2] >= 1 and val_counts[3] >= 1
                  and val_counts[4] >= 1 and val_counts[5] >= 1)
        existence[27] = s_low
        existence[28] = s_high
        existence[29] = s_low and val_counts[5] >= 1  # full 6-card straight

        # 30..35: three of a kind
        for v in range(6):
            existence[30 + v] = val_counts[v] >= 3

        # 36..65: full house (three of v_three + pair of v_two)
        for bet in range(36, 66):
            if bet < 41:    v_three, v_two = 0, bet - 35
            elif bet == 41: v_three, v_two = 1, 0
            elif bet < 46:  v_three, v_two = 1, bet - 40
            elif bet < 48:  v_three, v_two = 2, bet - 46
            elif bet < 51:  v_three, v_two = 2, bet - 45
            elif bet < 54:  v_three, v_two = 3, bet - 51
            elif bet < 56:  v_three, v_two = 3, bet - 50
            elif bet < 60:  v_three, v_two = 4, bet - 56
            elif bet == 60: v_three, v_two = 4, 5
            else:           v_three, v_two = 5, bet - 61
            existence[bet] = (val_counts[v_three] >= 3) and (val_counts[v_two] >= 2)

        # 66..69: flush (5+ cards of suit s)
        for s in range(4):
            existence[66 + s] = suit_counts[s] >= 5

        # 70..75: four of a kind
        for v in range(6):
            existence[70 + v] = val_counts[v] >= 4

        # 76..87: straight flushes (low / high / great × 4 suits)
        for bet in range(76, 88):
            suit = bet % 4
            if bet < 80:    vals = (0, 1, 2, 3, 4)
            elif bet < 84:  vals = (1, 2, 3, 4, 5)
            else:           vals = (0, 1, 2, 3, 4, 5)
            ok = True
            for v in vals:
                if not val_x_suit[v][suit]:
                    ok = False
                    break
            existence[bet] = ok

        return np.array(existence, dtype=np.bool_)
    
    @staticmethod
    def deal_cards(hand_sizes: List[int]) -> List[List[int]]:
        """Draw sum(hand_sizes) distinct cards from BlefCards and partition them
        into one sorted hand per entry of hand_sizes (in order)."""
        all_cards = rng.choice(BlefCards, size=sum(hand_sizes), replace=False)
        hands: List[List[int]] = []
        offset = 0
        for hs in hand_sizes:
            hands.append(sorted(all_cards[offset:offset + hs].tolist()))
            offset += hs
        return hands

    @staticmethod
    def hand_combinations(hand_sizes: List[int]) -> Generator[List[List[int]], None, None]:
        def generate(remaining_num_cards, possible_cards):
            if not remaining_num_cards:
                yield []
            else:
                player_hand_size = remaining_num_cards[0]
                for player_hand in itertools.combinations(possible_cards, player_hand_size):
                    for tail in generate(remaining_num_cards[1:], possible_cards - set(player_hand)):
                        yield [sorted(player_hand)] + tail
        return generate(hand_sizes, set(BlefCards))
