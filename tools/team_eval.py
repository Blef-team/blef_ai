#!/usr/bin/env python3
"""Team-aware evaluation of an NFSP learner against ConservativeAgent in
team-mode games.

The in-training eval (`_evaluate_policy` in `nfsp_ai/agent.py`) and
`tools/eval_ladder` both score from the *individual* learner perspective
("did the loss-card land on me?"). For a team specialist, the right
metric is whether the loser is on the LEARNER's team or the opponent
team. This script does that.

Reports per-config:
  - team_winrate:  (rounds where loser was on opp team) / (settled rounds)
  - team_avg_reward: mean of [+1 if opp-team loser, -1 if our-team loser]
  - indiv_winrate: same as the in-training metric, for cross-check
  - n_settled_rounds, n_games

Usage:
  python -m tools.team_eval \
      --checkpoint runs/.../checkpoints/nfsp_blef.pt \
      --deck-size 24 \
      --n-players 4 --n-teams 2 --max-cards 11 \
      --games 400 --seeds 5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Optional

import numpy as np
import torch

# Make the project tree importable regardless of how the script is run.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from conservative_ai.agent import ConservativeAgent  # type: ignore
from nfsp_ai.agent import NFSPAgent, NFSPConfig  # type: ignore
from nfsp_ai.nfsp_run_local import (  # type: ignore
    MyEnv,
    _build_deck_spec,
    _load_card_embedding,
    _load_history_embedding,
)


def _legal_or_random(action: Optional[int], mask: torch.Tensor) -> int:
    if action is not None and 0 <= int(action) < int(mask.numel()) and mask[int(action)] > 0:
        return int(action)
    legal = mask.nonzero(as_tuple=False).view(-1).tolist()
    return int(random.choice(legal)) if legal else 0


def _infer_dims_from_q_state(q_sd: dict) -> tuple[int, int, bool]:
    """Return (obs_dim, act_dim, factorize) by probing the Q-net state dict.

    Plain net: net.0.weight = [hidden, obs_dim], net.4.weight = [act_dim, hidden]
    Factorized: trunk.0.weight = [hidden, obs_dim], bet_head.weight = [act_dim-1, hidden],
                check_head.weight = [1, hidden]; act_dim = bet+1.
    """
    if "trunk.0.weight" in q_sd:
        obs_dim = int(q_sd["trunk.0.weight"].shape[1])
        bet = int(q_sd["bet_head.weight"].shape[0])
        act_dim = bet + 1
        return obs_dim, act_dim, True
    if "net.0.weight" in q_sd:
        obs_dim = int(q_sd["net.0.weight"].shape[1])
        # Find the last linear layer in net.* — its first dim is act_dim.
        last_idx = max(
            int(k.split(".")[1]) for k in q_sd
            if k.startswith("net.") and k.endswith(".weight")
        )
        act_dim = int(q_sd[f"net.{last_idx}.weight"].shape[0])
        return obs_dim, act_dim, False
    raise ValueError("could not infer obs_dim/act_dim from Q-net state dict")


def _build_learner_from_checkpoint(path: str, device: str = "cpu") -> NFSPAgent:
    """Load either an inference export OR a training-time checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_d = ckpt.get("cfg", {}) or {}
    base = NFSPConfig()
    q_sd = ckpt["q"]

    if "obs_dim" in ckpt and "act_dim" in ckpt:
        obs_dim = int(ckpt["obs_dim"])
        act_dim = int(ckpt["act_dim"])
        factorize_probe = "trunk.0.weight" in q_sd
    else:
        # Training checkpoint — infer from state dict.
        obs_dim, act_dim, factorize_probe = _infer_dims_from_q_state(q_sd)

    hidden = cfg_d.get("hidden")
    if hidden is None:
        for k in ("net.0.weight", "trunk.0.weight"):
            if k in q_sd:
                hidden = int(q_sd[k].shape[0])
                break
        if hidden is None:
            hidden = base.hidden

    factorize = cfg_d.get("factorize_action_head")
    if factorize is None:
        factorize = factorize_probe

    cfg = NFSPConfig(**{**base.__dict__, **{
        "hidden": int(hidden),
        "factorize_action_head": bool(factorize),
        "anticipatory_eta": cfg_d.get("anticipatory_eta", base.anticipatory_eta),
    }})
    cfg.rl_capacity = 1
    cfg.sl_capacity = 1
    cfg.batch_rl = 1
    cfg.batch_sl = 1
    cfg.train_rl_every = 1
    cfg.train_sl_every = 1
    cfg.warmup_steps = 0

    learner = NFSPAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=torch.device(device),
        cfg=cfg,
    )
    learner.q.load_state_dict(ckpt["q"])
    learner.pi.load_state_dict(ckpt["pi"])
    learner.q.eval()
    learner.pi.eval()
    learner.rl_buf = None
    learner.sl_buf = None
    return learner


def _team_of(players: list, nick: str) -> Optional[int]:
    for p in players:
        if p.get("nickname") == nick:
            t = p.get("team")
            return None if t is None else int(t)
    return None


def run_team_eval(
    checkpoint: str,
    *,
    deck_size: int,
    n_players: int,
    n_teams: int,
    max_cards: int,
    jokers: int = 0,
    blanks: int = 0,
    common_cards: int = 0,
    games: int = 400,
    seed: int = 0,
    card_embedding_path: Optional[str] = None,
    history_embedding_path: Optional[str] = None,
    device: str = "cpu",
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    learner = _build_learner_from_checkpoint(checkpoint, device=device)

    card_emb = _load_card_embedding(card_embedding_path, device=device) if card_embedding_path else None
    hist_emb = _load_history_embedding(history_embedding_path, device=device) if history_embedding_path else None

    env = MyEnv(
        n_agents=n_players,
        max_cards=max_cards,
        deck_size=deck_size,
        jokers=jokers,
        blanks=blanks,
        common_cards=common_cards,
        n_teams=n_teams,
        card_embedding=card_emb,
        history_embedding=hist_emb,
    )

    # Sanity: obs_dim match
    obs, _mask, _pid = env.reset()
    if int(obs.shape[-1]) != int(learner.obs_dim):
        raise ValueError(
            f"env obs_dim {int(obs.shape[-1])} != learner obs_dim {int(learner.obs_dim)} "
            f"(deck/jokers/blanks/common_cards / embeddings mismatch)"
        )

    env = MyEnv(
        n_agents=n_players,
        max_cards=max_cards,
        deck_size=deck_size,
        jokers=jokers,
        blanks=blanks,
        common_cards=common_cards,
        n_teams=n_teams,
        card_embedding=card_emb,
        history_embedding=hist_emb,
    )

    team_wins = team_losses = 0
    indiv_wins = indiv_losses = 0
    total_rounds = 0
    t0 = time.time()

    for _g in range(games):
        obs, mask, _pid = env.reset()
        ref = env._ref_nick
        ref_team = _team_of(env.game.get("players") or [], ref)
        done = False
        while not done:
            cp = env.game.get("cp_nickname")
            if cp == ref:
                a = learner.select_action(obs, mask, use_average_policy=True, greedy=True)
                a = _legal_or_random(a, mask)
            else:
                a = ConservativeAgent.determine_action(env.game)
                a = _legal_or_random(a, mask)
            obs, mask, _r, done, info = env.step(int(a))
            rr = (info or {}).get("round_result") or {}
            loser = rr.get("loser")
            if loser:
                total_rounds += 1
                # Individual perspective (matches in-training eval)
                if loser == ref:
                    indiv_losses += 1
                else:
                    indiv_wins += 1
                # Team perspective
                loser_team = _team_of(env.game.get("players") or [], loser)
                if ref_team is not None and loser_team is not None:
                    if loser_team == ref_team:
                        team_losses += 1
                    else:
                        team_wins += 1

    dur = time.time() - t0
    settled = team_wins + team_losses
    indiv_settled = indiv_wins + indiv_losses
    return {
        "checkpoint": checkpoint,
        "deck_size": deck_size,
        "n_players": n_players,
        "n_teams": n_teams,
        "max_cards": max_cards,
        "jokers": jokers,
        "blanks": blanks,
        "common_cards": common_cards,
        "games": games,
        "seed": seed,
        "n_rounds": total_rounds,
        "team_winrate": (team_wins / settled) if settled else float("nan"),
        "team_avg_reward": ((team_wins - team_losses) / settled) if settled else float("nan"),
        "indiv_winrate": (indiv_wins / indiv_settled) if indiv_settled else float("nan"),
        "team_settled_rounds": settled,
        "indiv_settled_rounds": indiv_settled,
        "duration_sec": round(dur, 2),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--deck-size", type=int, required=True)
    p.add_argument("--n-players", type=int, default=4)
    p.add_argument("--n-teams", type=int, default=2)
    p.add_argument("--max-cards", type=int, default=11)
    p.add_argument("--jokers", type=int, default=0)
    p.add_argument("--blanks", type=int, default=0)
    p.add_argument("--common-cards", type=int, default=0)
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--seeds", type=int, default=5, help="number of seeds to average over")
    p.add_argument("--card-embedding", default=None)
    p.add_argument("--history-embedding", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args(argv)

    rows = []
    for seed in range(args.seeds):
        r = run_team_eval(
            args.checkpoint,
            deck_size=args.deck_size,
            n_players=args.n_players,
            n_teams=args.n_teams,
            max_cards=args.max_cards,
            jokers=args.jokers,
            blanks=args.blanks,
            common_cards=args.common_cards,
            games=args.games,
            seed=seed,
            card_embedding_path=args.card_embedding,
            history_embedding_path=args.history_embedding,
            device=args.device,
        )
        rows.append(r)
        print(
            f"[seed={seed}] team_wr={r['team_winrate']:.4f} "
            f"team_rew={r['team_avg_reward']:+.4f} "
            f"indiv_wr={r['indiv_winrate']:.4f} "
            f"rounds={r['n_rounds']} dur={r['duration_sec']}s"
        )

    if rows:
        twrs = [r["team_winrate"] for r in rows if r["team_winrate"] == r["team_winrate"]]
        trew = [r["team_avg_reward"] for r in rows if r["team_avg_reward"] == r["team_avg_reward"]]
        iwrs = [r["indiv_winrate"] for r in rows if r["indiv_winrate"] == r["indiv_winrate"]]
        if twrs:
            mean = sum(twrs) / len(twrs)
            std = (sum((x - mean) ** 2 for x in twrs) / len(twrs)) ** 0.5
            stderr = std / (len(twrs) ** 0.5)
            print(
                f"[summary] team_wr={mean:.4f} ± {stderr:.4f} "
                f"team_rew={sum(trew)/len(trew):+.4f} "
                f"indiv_wr={sum(iwrs)/len(iwrs):.4f}  ({len(twrs)} seeds)"
            )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
