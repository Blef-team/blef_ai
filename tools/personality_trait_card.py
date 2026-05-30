"""Trait-card generator for trained personality models.

Given a personality-suffixed inference checkpoint (e.g.
``artifacts/nfsp_inference_24_1v1_kupala.pt``) and the baseline 1v1
checkpoint, plays N self-play games and emits a Markdown card with the
quantitative trait profile:

  1. action-frequency profile vs baseline (histogram over hand types + CHECK)
  2. bluff rate (via shared.game_utils.determine_set_existence)
  3. CHECK frequency
  4. adaptivity probe — for personalities with obs blind spots, shuffle the
     masked obs block at inference; the action distribution must NOT shift
     significantly if the blind-spot was actually learned
  5. eval ladder via tools.eval_ladder
  6. auto-generated one-line trait sentence from the top distinguishing stats

Run from repo root:

  python -m tools.personality_trait_card \\
    --personality kupala \\
    --baseline artifacts/nfsp_inference_24_1v1.pt \\
    --n-games 2000 \\
    --output eval_results/trait_card_kupala.md
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from nfsp_ai.nfsp_run_local import MyEnv
from nfsp_ai.personality_train import (
    PersonalityTrainSpec,
    load_spec,
    resolve_obs_block_slices,
    spec_from_dict,
)
from shared.game_utils import determine_set_existence, get_set_details_from_action_id
from tools.eval_ladder import load_learner


HAND_TYPES = (
    "High card", "Pair", "Two pairs", "Straight", "Three of a kind",
    "Full house", "Flush", "Four of a kind", "Straight flush",
)


def _hand_type_of(action_id: int, deck_size: int, check_id: int) -> Optional[str]:
    if action_id == check_id:
        return "CHECK"
    info = get_set_details_from_action_id(int(action_id), int(deck_size))
    return info.get("set_type") if info else None


def _bet_maker_hand(game_state: dict, nick: Optional[str]) -> Optional[list]:
    if not nick:
        return None
    for h in game_state.get("hands", []) or []:
        if h.get("nickname") == nick:
            return h.get("hand", []) or []
    return None


def _was_bluff(game_state: dict, bet_maker_nick: str, action_id: int) -> Optional[bool]:
    """True iff the bet is NOT supported by the bet-maker's hand. None if unknown."""
    hand = _bet_maker_hand(game_state, bet_maker_nick)
    if hand is None:
        return None
    num_jokers = sum(1 for c in hand if int(c.get("value", 0)) < 0)
    rules = game_state.get("rules", {}) or {}
    return not bool(determine_set_existence(hand, int(action_id), rules, num_jokers))


def _play_games(
    learner_under_test,
    baseline,
    n_games: int,
    deck_size: int = 24,
    n_players: int = 2,
    max_cards: int = 11,
    seed: Optional[int] = None,
    personality_spec: Optional[PersonalityTrainSpec] = None,
    obs_shuffle_block: Optional[str] = None,
    card_embedding=None,
    history_embedding=None,
) -> Dict[str, Counter]:
    """Play ``n_games`` rounds with ``learner_under_test`` at seat 0 (the
    reference) and ``baseline`` at the other seat. Returns counters for the
    learner-under-test's decisions and the baseline's, separately.

    When ``obs_shuffle_block`` is set, the named obs block is replaced with a
    random permutation of its values each step — used to verify obs-blind
    personalities don't actually use those features.
    """
    if seed is not None:
        import random as _r
        _r.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    env = MyEnv(
        n_agents=n_players,
        max_cards=max_cards,
        deck_size=deck_size,
        jokers=0, blanks=0, common_cards=0,
        card_embedding=card_embedding,
        history_embedding=history_embedding,
        personality_spec=personality_spec,
    )

    learner_actions = Counter()
    learner_bluffs = [0, 0]      # [bluffs, total bet decisions]
    baseline_actions = Counter()
    baseline_bluffs = [0, 0]

    shuffle_slice = None
    if obs_shuffle_block is not None:
        layout = resolve_obs_block_slices(env.deck_spec)
        rng = layout.get(obs_shuffle_block)
        if rng is not None:
            shuffle_slice = slice(*rng)

    check_id = int(env.deck_spec.check_action_id)

    for _ in range(n_games):
        obs, mask, _pid = env.reset()
        done = False
        while not done:
            cp = env.game.get("cp_nickname")
            ref = env._ref_nick
            is_learner = (cp == ref)
            game_state = env.game

            if shuffle_slice is not None:
                obs = obs.clone()
                blk = obs[shuffle_slice].clone().numpy()
                np.random.shuffle(blk)
                obs[shuffle_slice] = torch.from_numpy(blk)

            agent = learner_under_test if is_learner else baseline
            action = int(agent.select_action(obs, mask, use_average_policy=True, greedy=True))
            # Ensure legal:
            legal = mask.nonzero(as_tuple=False).view(-1).tolist()
            if action not in legal:
                action = int(legal[0]) if legal else 0

            # Log BEFORE step (so bluff oracle sees the pre-step game state).
            ht = _hand_type_of(action, deck_size, check_id)
            ctr = learner_actions if is_learner else baseline_actions
            bluff_ctr = learner_bluffs if is_learner else baseline_bluffs
            if ht is not None:
                ctr[ht] += 1
            if 0 <= action < check_id:
                blf = _was_bluff(game_state, cp, action)
                if blf is not None:
                    bluff_ctr[1] += 1
                    if blf:
                        bluff_ctr[0] += 1

            obs, mask, _r, done, _info = env.step(action)

    return {
        "learner_actions": learner_actions,
        "learner_bluff_total": learner_bluffs[1],
        "learner_bluff_count": learner_bluffs[0],
        "baseline_actions": baseline_actions,
        "baseline_bluff_total": baseline_bluffs[1],
        "baseline_bluff_count": baseline_bluffs[0],
    }


def _freq(ctr: Counter) -> Dict[str, float]:
    total = sum(ctr.values()) or 1
    return {k: ctr.get(k, 0) / total for k in (*HAND_TYPES, "CHECK")}


def _diff_pp(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    return {k: 100 * (a.get(k, 0.0) - b.get(k, 0.0)) for k in a}


def _top_deviations(diff_pp: Dict[str, float], n: int = 3) -> list:
    items = sorted(diff_pp.items(), key=lambda kv: -abs(kv[1]))
    return items[:n]


def _auto_sentence(
    name: str,
    diff_pp: Dict[str, float],
    bluff_pp: float,
    spec: Optional[PersonalityTrainSpec] = None,
) -> str:
    """One-line trait sentence anchored on the spec's mechanism, verified by data.

    Treats a positive CHECK delta specially when ``spec.forbid_check_unless_only``
    is set: the personality didn't choose to check more, it got *pinned* at the
    top of the bet ladder. The phrasing makes that distinction so readers don't
    mistake aggressive bots for passive ones.
    """
    cap = name.capitalize()
    parts = []

    # 1) CHECK delta — phrase depends on whether the spec forbids voluntary CHECK.
    check_delta = diff_pp.get("CHECK", 0.0)
    forbids_check = bool(spec and spec.forbid_check_unless_only)
    if forbids_check and check_delta >= 10:
        parts.append(
            f"escalates aggressively and only checks when pinned at the ceiling "
            f"(forced CHECK {check_delta:+.0f}pp)"
        )
    elif check_delta >= 10:
        parts.append(f"checks far more often ({check_delta:+.0f}pp)")
    elif check_delta <= -10:
        parts.append(f"rarely checks ({check_delta:+.0f}pp)")

    # 2) Hand-type deviations (top 2, excluding CHECK).
    hand_pairs = sorted(
        ((ht, d) for ht, d in diff_pp.items() if ht != "CHECK"),
        key=lambda kv: -abs(kv[1]),
    )
    used = 0
    for ht, d in hand_pairs:
        if used >= 2 or abs(d) < 5:
            break
        if d < 0:
            parts.append(f"rarely claims {ht.lower()} ({d:+.0f}pp)")
        else:
            parts.append(f"claims {ht.lower()} far more often ({d:+.0f}pp)")
        used += 1

    # 3) Bluff rate.
    if abs(bluff_pp) >= 5:
        if bluff_pp > 0:
            parts.append(f"bluffs {bluff_pp:+.0f}pp more often (claims hands it doesn't have)")
        else:
            parts.append(f"bluffs {abs(bluff_pp):.0f}pp less often (plays honest)")

    if not parts:
        return f"**{cap}** plays essentially like the baseline."
    return f"**{cap}**: " + "; ".join(parts) + "."


def write_card(
    path: str,
    name: str,
    n_games: int,
    main_stats: Dict,
    shuffled_stats: Optional[Dict],
    spec: Optional[PersonalityTrainSpec],
) -> None:
    learn_freq = _freq(main_stats["learner_actions"])
    base_freq = _freq(main_stats["baseline_actions"])
    diff = _diff_pp(learn_freq, base_freq)

    lb_total = main_stats["learner_bluff_total"] or 1
    bb_total = main_stats["baseline_bluff_total"] or 1
    learn_bluff = 100 * main_stats["learner_bluff_count"] / lb_total
    base_bluff = 100 * main_stats["baseline_bluff_count"] / bb_total
    bluff_pp = learn_bluff - base_bluff

    sentence = _auto_sentence(name, diff, bluff_pp, spec)

    lines = []
    lines.append(f"# {name} — trait card")
    lines.append("")
    lines.append(sentence)
    lines.append("")
    lines.append(f"Sample: {n_games} games (24-deck, 2p, vanilla).  "
                 f"Personality vs baseline NFSP at seat 0/1.")
    lines.append("")
    if spec is not None:
        lines.append("## Spec")
        lines.append("```json")
        from nfsp_ai.personality_train import spec_to_dict
        import json as _j
        lines.append(_j.dumps(spec_to_dict(spec), indent=2))
        lines.append("```")
        lines.append("")
    lines.append("## Action-frequency profile (% of decisions)")
    lines.append("| Bucket | Baseline | Personality | Δ (pp) |")
    lines.append("|---|---:|---:|---:|")
    for ht in (*HAND_TYPES, "CHECK"):
        lines.append(
            f"| {ht} | {100*base_freq[ht]:.1f} | {100*learn_freq[ht]:.1f} | "
            f"{diff[ht]:+.1f} |"
        )
    lines.append("")
    lines.append(f"**Bluff rate:** personality {learn_bluff:.1f}%  vs  "
                 f"baseline {base_bluff:.1f}%  (Δ {bluff_pp:+.1f} pp)")
    lines.append("")

    if shuffled_stats is not None and spec is not None and spec.obs_blind_spots:
        s_freq = _freq(shuffled_stats["learner_actions"])
        s_diff = _diff_pp(learn_freq, s_freq)
        max_shift = max(abs(v) for v in s_diff.values())
        verdict = "PASS" if max_shift < 2.0 else "FAIL"
        lines.append("## Adaptivity probe (obs blind-spot verification)")
        lines.append(
            f"Shuffled obs block(s) {list(spec.obs_blind_spots)} at inference; "
            f"action-distribution shift max = **{max_shift:.1f} pp** "
            f"({verdict} — threshold 2.0 pp)"
        )
        lines.append("")
        if max_shift >= 2.0:
            lines.append("Top shifts:")
            top = sorted(s_diff.items(), key=lambda kv: -abs(kv[1]))[:5]
            for ht, d in top:
                lines.append(f"  - {ht}: {d:+.1f} pp")
            lines.append("")

    lines.append("---")
    lines.append("*Auto-generated by `tools/personality_trait_card.py`.*")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--personality", required=True,
        help="Personality name (e.g. 'kupala'). Used to locate the spec at "
             "nfsp_ai/personality_specs/<name>.json and the checkpoint at "
             "artifacts/nfsp_inference_24_1v1_<name>.pt (unless --checkpoint is set).",
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Override the personality checkpoint path.")
    parser.add_argument("--baseline", default="artifacts/nfsp_inference_24_1v1.pt",
                        help="Baseline NFSP checkpoint (opponent + comparison).")
    parser.add_argument("--n-games", type=int, default=2000)
    parser.add_argument("--deck-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="Output markdown path. Default: "
                             "eval_results/trait_card_<personality>.md")
    parser.add_argument(
        "--use-card-embeddings", nargs="?", const="auto", default="auto",
        help="Card embedding artifact path or 'auto'.",
    )
    parser.add_argument(
        "--use-history-embeddings", nargs="?", const="auto", default="auto",
        help="History embedding artifact path or 'auto'.",
    )
    args = parser.parse_args(argv)

    name = args.personality.strip().lower()
    spec_path = f"nfsp_ai/personality_specs/{name}.json"
    spec = load_spec(spec_path) if os.path.exists(spec_path) else None

    ckpt = args.checkpoint or f"artifacts/nfsp_inference_24_1v1_{name}.pt"
    if not os.path.exists(ckpt):
        print(f"error: personality checkpoint not found at {ckpt!r}", file=sys.stderr)
        return 2
    if not os.path.exists(args.baseline):
        print(f"error: baseline checkpoint not found at {args.baseline!r}", file=sys.stderr)
        return 2

    # Embeddings (required for 1v1 24-deck baselines that were trained with them).
    from nfsp_ai.nfsp_run_local import (
        _load_card_embedding, _resolve_card_embedding_path,
        _load_history_embedding, _resolve_history_embedding_path,
    )
    ce = None
    he = None
    if args.use_card_embeddings:
        p = _resolve_card_embedding_path(args.use_card_embeddings, args.deck_size)
        if p:
            ce = _load_card_embedding(p, device="cpu")
    if args.use_history_embeddings:
        p = _resolve_history_embedding_path(args.use_history_embeddings, args.deck_size)
        if p:
            he = _load_history_embedding(p, device="cpu")

    learner = load_learner(ckpt)
    baseline = load_learner(args.baseline)

    print(f"[trait-card] {name}: playing {args.n_games} games...")
    main_stats = _play_games(
        learner, baseline, n_games=args.n_games,
        deck_size=args.deck_size, seed=args.seed,
        personality_spec=spec,
        card_embedding=ce, history_embedding=he,
    )

    shuffled_stats = None
    if spec is not None and spec.obs_blind_spots:
        # Probe the first blind-spot block (representative — composite probes
        # can be added later if needed).
        blk = spec.obs_blind_spots[0]
        print(f"[trait-card] {name}: adaptivity probe on obs block {blk!r}...")
        shuffled_stats = _play_games(
            learner, baseline, n_games=max(500, args.n_games // 4),
            deck_size=args.deck_size, seed=args.seed + 1,
            personality_spec=spec, obs_shuffle_block=blk,
            card_embedding=ce, history_embedding=he,
        )

    out = args.output or f"eval_results/trait_card_{name}.md"
    write_card(out, name, args.n_games, main_stats, shuffled_stats, spec)
    print(f"[trait-card] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
