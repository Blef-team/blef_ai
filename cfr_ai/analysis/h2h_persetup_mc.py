"""Per-setup isolated MC head-to-head between two CFR model versions.

For one setup (a, b): deal random hands, play exactly ONE round with the real
game engine, record the loser; symmetrise model B across both seats. Reports
model B's advantage = (B round-wins - A round-wins) / N in [-1, +1] (SE ~ 0.01 at
N = 10k). Positive => B is the stronger model on that setup.

Macro- and history-depth-aware: reuses `cfr_vs_cfr_games.CFRAgent` (an independent
`cfr_ai.agent` module instance per model -> its own outputs-base + cache; resolves
macros and reads each model's stored `history_depth` at serve time) and the same
round resolution as `cfr_vs_cfr_games._play_game`, but stops after the first round
so only the trained setup (a, b) is exercised. Unlike `head_to_head.py` this serves
macro models, and unlike `cfr_vs_cfr_games.py` it isolates a single setup (so the B
model only needs that one setup trained).

Run from the repo root, one setup per invocation (append-CSV friendly for a DAG):
    python -m cfr_ai.analysis.h2h_persetup_mc --hand-sizes 5 7 \
        --model-a-folder cfr_ai/experiments/v32 \
        --model-b-folder cfr_ai/experiments/v32_hist2 \
        --num-deals 10000 --out scratch/hist2_persetup.csv
"""
import argparse
import math
import os
import random
import sys

from cfr_ai.analysis.cfr_vs_cfr_games import CFRAgent, DECK_SIZE, MAX_CARDS


def _play_one_round(game, seat0_agent, seat1_agent, round_guard=400):
    """Play a single round to its showdown. Returns the loser seat (0/1) or None
    on a malformed/guard-exceeded round. seat0_agent/seat1_agent act for the
    players nicknamed '0'/'1'."""
    import shared.api.simpleschema_local_manager as gm
    before = {p["nickname"]: int(p["n_cards"]) for p in game["players"]}
    for _ in range(round_guard):
        cp = game["cp_nickname"]
        agent = seat0_agent if cp == "0" else seat1_agent
        try:
            act = int(agent.determine_action(game))
            gm.play(game, act, save_dir=None)
        except Exception:
            return None
        after = {p["nickname"]: int(p["n_cards"]) for p in game["players"]}
        if after != before:
            # The round resolved: the loser is whoever gained a card (or was
            # knocked to zero). Mirrors cfr_vs_cfr_games._play_game.
            for nick, b in before.items():
                af = after.get(nick, b)
                if af > b or (b > 0 and af == 0):
                    return int(nick)
            return None
    return None


def run_setup(a, b, a_base, b_base, num_deals, seed=0):
    """Return (advantage_B, se, b_wins, a_wins, bad). init_card_dist=[a, b] puts
    a cards on seat0, b cards on seat1; model B is symmetrised onto seat0 for half
    the deals and seat1 for the other half, so it plays both seats/roles equally."""
    import shared.api.simpleschema_local_manager as gm
    agent_a = CFRAgent(a_base, "a")   # baseline (Model A)
    agent_b = CFRAgent(b_base, "b")   # variant  (Model B; advantage is B's)
    random.seed(seed)
    b_wins = a_wins = bad = 0
    for i in range(num_deals):
        b_seat = i % 2
        game = gm.create_game(2, deck_size=DECK_SIZE, max_cards=MAX_CARDS,
                              init_card_dist=[a, b])
        seat0 = agent_b if b_seat == 0 else agent_a
        seat1 = agent_b if b_seat == 1 else agent_a
        loser = _play_one_round(game, seat0, seat1)
        if loser is None:
            bad += 1
            continue
        if loser == b_seat:      # B lost the round
            a_wins += 1
        else:
            b_wins += 1
    n = b_wins + a_wins
    adv = (b_wins - a_wins) / n if n else 0.0
    se = 2.0 * math.sqrt(0.25 / n) if n else 0.0
    return adv, se, b_wins, a_wins, bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hand-sizes", nargs=2, type=int, required=True,
                    help="The two card counts for the setup, e.g. 5 7.")
    ap.add_argument("--model-a-folder", required=True,
                    help="Baseline model folder (contains outputs/). Model A.")
    ap.add_argument("--model-b-folder", required=True,
                    help="Variant model folder (contains outputs/). Model B; the "
                         "reported advantage is B's.")
    ap.add_argument("--num-deals", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="CSV to append one result row to (header written once).")
    args = ap.parse_args()

    a, b = args.hand_sizes
    a_base = os.path.join(args.model_a_folder, "outputs")
    b_base = os.path.join(args.model_b_folder, "outputs")
    for base in (a_base, b_base):
        if not os.path.isdir(os.path.join(base, f"{min(a, b)}_{max(a, b)}")):
            print(f"[error] setup {min(a, b)}_{max(a, b)} not found under {base}",
                  file=sys.stderr)
            return 1

    adv, se, b_wins, a_wins, bad = run_setup(a, b, a_base, b_base,
                                             args.num_deals, args.seed)
    setup = f"{a},{b}"
    print(f"[h2h] setup {setup}: B-vs-A advantage = {adv:+.4f} +/- {se:.4f}pp "
          f"(B {b_wins} / A {a_wins}, {bad} malformed of {args.num_deals})",
          flush=True)

    if args.out:
        import csv
        new = not os.path.exists(args.out)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["setup", "num_deals", "B_advantage", "SE",
                            "B_wins", "A_wins", "malformed"])
            w.writerow([setup, args.num_deals, f"{adv:.4f}", f"{se:.4f}",
                        b_wins, a_wins, bad])
    return 0


if __name__ == "__main__":
    sys.exit(main())
