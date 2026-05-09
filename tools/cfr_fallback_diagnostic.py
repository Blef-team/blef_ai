"""Diagnostic: count how often the CFR opponent falls back to a hardcoded
action (87 / 88) because the info-set lookup misses. Wraps cfr_ai.agent
with instrumentation, then runs a learner head-to-head against CFR for
each max_cards bucket and reports the miss rate.

Usage:
    cd /tmp/blef_train2 && PYTHONPATH=/tmp/blef_train2 python -m tools.cfr_fallback_diagnostic \
        --checkpoint <ckpt.pt> --n-games 500 --deck-size 24 --n-players 2
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys

import cfr_ai.agent as cfr_agent
from cfr_ai.information_set import make_key, get_possible_actions, get_hand_abstraction
from cfr_ai.encoding import decode_probabilities

CALLS = 0
MISS_EARLY = 0
MISS_MID = 0
HIT = 0


def _instrumented_determine_action(game_state):
    global CALLS, HIT, MISS_EARLY, MISS_MID
    CALLS += 1
    agent_nickname = game_state["cp_nickname"]
    players = game_state.get("players", [])
    hand_sizes = [p.get("n_cards") for p in players]
    hand_sizes.sort()
    history = []
    if game_state.get("history"):
        history = [a["action_id"] for a in game_state.get("history")]
    matching_hands = [h for h in game_state.get("hands", []) if h.get("nickname") == agent_nickname]
    my_cards = [c["value"] * 4 + c["colour"] for c in matching_hands[0]["hand"]]

    min_bet = 0
    if os.path.exists("metadata.csv"):
        with open("metadata.csv", "r", encoding="utf-8") as f:
            r = csv.reader(f)
            for row in r:
                if len(row) >= 2 and row[0].strip() == "Minimum bet":
                    min_bet = int(row[1].strip())
    hand_abstraction = get_hand_abstraction(my_cards, hand_sizes)
    key = make_key(my_cards, hand_abstraction, history, min_bet)
    split_key = key.split('-')
    filename = "cfr_ai/outputs/" + "_".join(str(x) for x in hand_sizes) + "/" + split_key[0] + "/" + split_key[1] + ".csv"
    relevant_actions = get_possible_actions(history, min_bet)
    matching = []
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            sr = csv.reader(f)
            matching = [x[1] for x in sr if x[0] == "-".join(split_key[2:])]
    if len(matching) == 1:
        HIT += 1
        strategy = decode_probabilities(matching[0])
        return random.choices(relevant_actions, weights=strategy, k=1)[0]
    if len(history) == 0:
        MISS_EARLY += 1
        return 87
    MISS_MID += 1
    return 88


cfr_agent.determine_action = _instrumented_determine_action

# Import after patch so CFROpponent picks up the instrumented function.
from tools.eval_ladder import head_to_head, load_learner, OPPONENT_REGISTRY  # noqa: E402


def _reset_counters():
    global CALLS, HIT, MISS_EARLY, MISS_MID
    CALLS = HIT = MISS_EARLY = MISS_MID = 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Learner checkpoint to play against CFR.")
    p.add_argument("--n-games", type=int, default=500)
    p.add_argument("--deck-size", type=int, default=24)
    p.add_argument("--n-players", type=int, default=2)
    p.add_argument("--max-cards-list", type=str, default="1,2,4,6,8,11",
                   help="Comma-separated max_cards buckets to probe.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--use-card-embeddings", default=None)
    p.add_argument("--use-history-embeddings", default=None)
    args = p.parse_args()

    print("=== CFR fallback diagnostic ===")
    print(f"checkpoint: {args.checkpoint}")
    print(f"deck={args.deck_size} n_players={args.n_players}")
    print(f"games per max_cards bucket: {args.n_games}")
    print()
    print(f"{'mc':>4}  {'cfr_calls':>10}  {'hits':>8}  {'miss_early':>11}  {'miss_mid':>9}  {'miss_rate':>10}")
    print("-" * 64)

    learner = load_learner(args.checkpoint)
    cfr_opp = OPPONENT_REGISTRY["cfr"](learner.obs_dim, learner.act_dim)

    # Load embeddings if requested — reuse the helpers eval_ladder uses
    card_embedding = history_embedding = None
    if args.use_card_embeddings:
        from nfsp_ai.nfsp_run_local import _resolve_card_embedding_path, _load_card_embedding
        p = _resolve_card_embedding_path(args.use_card_embeddings, args.deck_size)
        if p is not None:
            card_embedding = _load_card_embedding(p, device="cpu")
    if args.use_history_embeddings:
        from nfsp_ai.nfsp_run_local import _resolve_history_embedding_path, _load_history_embedding
        p = _resolve_history_embedding_path(args.use_history_embeddings, args.deck_size)
        if p is not None:
            history_embedding = _load_history_embedding(p, device="cpu")

    mc_list = [int(x) for x in args.max_cards_list.split(",") if x.strip()]
    for mc in mc_list:
        _reset_counters()
        head_to_head(
            learner=learner,
            opponent=cfr_opp,
            n_games=args.n_games,
            deck_size=args.deck_size,
            n_players=args.n_players,
            max_cards=mc,
            seed=args.seed,
            card_embedding=card_embedding,
            history_embedding=history_embedding,
        )
        miss = MISS_EARLY + MISS_MID
        rate = 100.0 * miss / max(1, CALLS)
        print(f"{mc:>4}  {CALLS:>10}  {HIT:>8}  {MISS_EARLY:>11}  {MISS_MID:>9}  {rate:>9.2f}%")


if __name__ == "__main__":
    main()
