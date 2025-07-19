import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_win_probabilities():
    """
    Recursively calculates the expected win probability for a game of Blef from every
    possible setup, basing on the summary_of_all_runs file. Prints a table and saves a heatmap.
    Note: P(P1 wins from X,Y | P2 starts) = 1 - P(P1 wins from Y,X | P1 starts)
    """
    # Read in round utilities from the summary file
    try:
        df = pd.read_csv("cfr_ai/outputs/summary_of_all_runs.csv")
    except FileNotFoundError:
        print("Error: 'summary_of_all_runs.csv' not found.")
        print("Please ensure the file is in the 'cfr_ai/outputs/' directory.")
        return

    utilities = {}
    for _, row in df.iterrows():
        try:
            c1, c2 = map(int, row['Setup'].split(','))
            vals = row['Exp. Value'].split(', ')
            if c1 == c2:
                util = float(vals[0])
                utilities[(c1, c2)] = (util, -util)
            else:
                util_c1_starts = float(vals[0])
                util_c2_starts = float(vals[1])
                utilities[(c1, c2)] = (util_c1_starts, -util_c2_starts)
                utilities[(c2, c1)] = (util_c2_starts, -util_c1_starts)
        except (ValueError, IndexError):
            continue

    # Set up Recursive Calculation
    memo = {}

    def get_win_prob(p1_cards, p2_cards, starting_player):
        if p1_cards >= 12: return 0.0
        if p2_cards >= 12: return 1.0
        state = (p1_cards, p2_cards, starting_player)
        if state in memo: return memo[state]

        key = tuple(sorted((p1_cards, p2_cards)))
        if key not in utilities: return 0.5
        
        util_p1_perspective = 0.0
        p1_is_c1 = p1_cards < p2_cards or (p1_cards == p2_cards)
        util_c1_starts, util_c1_vs_c2_starts = utilities[key]

        if p1_is_c1:
            util_p1_perspective = util_c1_starts if starting_player == 1 else util_c1_vs_c2_starts
        else:
            util_p1_perspective = -util_c1_vs_c2_starts if starting_player == 1 else -util_c1_starts

        prob_p1_wins_round = (util_p1_perspective + 1) / 2
        
        # (Loser starts the next round)
        prob_win_after_p1_wins = get_win_prob(p1_cards, p2_cards + 1, 2)
        prob_win_after_p2_wins = get_win_prob(p1_cards + 1, p2_cards, 1)

        total_win_prob = (prob_p1_wins_round * prob_win_after_p1_wins) + \
                         ((1 - prob_p1_wins_round) * prob_win_after_p2_wins)

        memo[state] = total_win_prob
        return total_win_prob

    # Populate the table
    win_table_p1_starts = np.zeros((11, 11))
    for p1 in range(1, 12):
        for p2 in range(1, 12):
            win_table_p1_starts[p1 - 1, p2 - 1] = get_win_prob(p1, p2, 1)

    # Print the table
    df_p1_starts = pd.DataFrame(
        win_table_p1_starts,
        index=range(1, 12),
        columns=range(1, 12)
    )
    df_p1_starts.index.name = 'P1 Cards'
    df_p1_starts.columns.name = 'P2 Cards'

    print("\n" + "="*70)
    print("Game Win Probability Matrix (Assuming Player 1 Starts Round)")
    print("="*70)
    print(df_p1_starts.to_string(float_format="%.3f"))

    # Generate and save the heatmap
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    heatmap = sns.heatmap(
        df_p1_starts,
        annot=True,
        fmt=".0%",
        cmap="viridis_r",
        linewidths=.5,
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Player 1 Win Probability'},
        annot_kws={'fontsize': 15}
    )
    ax.set_title("Expected Game Win Probability from Given Setup (P1 Starts)", fontsize=20, pad=20)
    ax.set_xlabel("Player 2 Card Count", fontsize=15)
    ax.set_ylabel("Player 1 Card Count", fontsize=15)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(15)
    cbar.ax.tick_params(labelsize=13)
    plt.tight_layout()
    
    heatmap_filename = "cfr_ai/outputs/win_probability_by_starting_setup.png"
    plt.savefig(heatmap_filename)
    print(f"\n✅ Heatmap saved successfully as '{heatmap_filename}'")
    print("=" * 70)


if __name__ == '__main__':
    calculate_win_probabilities()