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
    # Cell breakdowns: (key) -> [team_wins, team_losses]
    by_cards: dict = {}        # key = (our_cards, opp_total_cards) at round start
    by_round_bin: dict = {}    # key = "early" | "mid" | "late"
    by_partition: dict = {}    # key = "ours_vs_opp1[_opp2...]" e.g., "2v2", "1v2", "1v1v1"
    t0 = time.time()

    def _bump(d: dict, key, win: bool):
        if key not in d:
            d[key] = [0, 0]
        d[key][0 if win else 1] += 1

    def _round_bin(rn: int) -> str:
        if rn <= 2:
            return "early"
        if rn <= 5:
            return "mid"
        return "late"

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
                # Team perspective + cell bookkeeping
                loser_team = _team_of(env.game.get("players") or [], loser)
                if ref_team is not None and loser_team is not None:
                    is_win = (loser_team != ref_team)
                    if is_win:
                        team_wins += 1
                    else:
                        team_losses += 1
                    # Cell keys
                    before = rr.get("before_counts") or {}
                    our_c = int(before.get(ref, 0))
                    opp_c = sum(int(v) for k, v in before.items() if k != ref)
                    rn = int(env.game.get("round_number", 1))
                    # Team-partition at ROUND START (not after elimination —
                    # otherwise auto-game-ending rounds get binned as 1v0/2v0
                    # which can't actually be played). Read from before_counts.
                    team_map = {p.get("nickname"): p.get("team")
                                for p in (env.game.get("players") or [])}
                    by_team_active: dict = {}
                    for nick, n_before in before.items():
                        if int(n_before) <= 0:
                            continue
                        t = team_map.get(nick)
                        if t is None:
                            continue
                        by_team_active[t] = by_team_active.get(t, 0) + 1
                    our_size = by_team_active.pop(ref_team, 0)
                    opp_sizes = sorted(by_team_active.values(), reverse=True)
                    partition_key = f"{our_size}v" + "v".join(str(x) for x in opp_sizes) if opp_sizes else f"{our_size}v0"
                    _bump(by_cards, (our_c, opp_c), is_win)
                    _bump(by_round_bin, _round_bin(rn), is_win)
                    _bump(by_partition, partition_key, is_win)

    dur = time.time() - t0
    settled = team_wins + team_losses
    indiv_settled = indiv_wins + indiv_losses

    def _winrates(d: dict) -> dict:
        out = {}
        for key, (w, l) in sorted(d.items(), key=lambda kv: str(kv[0])):
            n = w + l
            out[str(key)] = {"team_wr": (w / n) if n else None, "n": n}
        return out

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
        "by_cards": _winrates(by_cards),
        "by_round_bin": _winrates(by_round_bin),
        "by_partition": _winrates(by_partition),
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

        # Cell breakdown: aggregate counts across seeds before computing rates
        # so per-cell stats reflect the full sample, not seed-mean-of-rates.
        def _agg(field: str) -> dict:
            agg: dict = {}
            for r in rows:
                for key, v in (r.get(field) or {}).items():
                    if v.get("team_wr") is None:
                        continue
                    n = int(v["n"])
                    w = int(round(float(v["team_wr"]) * n))
                    a = agg.setdefault(key, [0, 0])
                    a[0] += w
                    a[1] += n - w
            return agg

        def _print_cells(label: str, agg: dict, sort_key=None):
            if not agg:
                return
            print(f"[breakdown:{label}]")
            items = list(agg.items())
            if sort_key:
                items.sort(key=sort_key)
            for k, (w, l) in items:
                n = w + l
                if n == 0:
                    continue
                wr = w / n
                # Wilson-ish stderr for binomial
                se = (wr * (1 - wr) / n) ** 0.5 if n else 0.0
                print(f"  {label}={k!s:<12s} team_wr={wr:.4f} ± {se:.4f}  (n={n})")

        # Round bin (early/mid/late) — sort early < mid < late.
        rb_agg = _agg("by_round_bin")
        rb_order = {"'early'": 0, "'mid'": 1, "'late'": 2}
        _print_cells("round", rb_agg, sort_key=lambda kv: rb_order.get(kv[0], 99))

        # Team partition (e.g., "2v2", "1v2", "2v1", "1v1v1"): from ref's
        # POV, our_team_size + sorted opponent team sizes.
        _print_cells("partition", _agg("by_partition"), sort_key=lambda kv: kv[0])

        # (our_cards, opp_cards) — print most-frequent cells only, top 12.
        cards_agg = _agg("by_cards")
        if cards_agg:
            top = sorted(cards_agg.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:12]
            print(f"[breakdown:cards] (top {len(top)} by frequency)")
            for k, (w, l) in top:
                n = w + l
                wr = w / n
                se = (wr * (1 - wr) / n) ** 0.5
                print(f"  cards={k!s:<14s} team_wr={wr:.4f} ± {se:.4f}  (n={n})")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
