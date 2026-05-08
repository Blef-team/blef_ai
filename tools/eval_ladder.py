"""
Head-to-head evaluation harness for the NFSP Blef agent.

Plays a learner agent against a fixed opponent across N games and reports
winrate, mean reward, and 95% CI from the learner's perspective. Supports
four opponent types (the "metric ladder"):

  random      : uniform random over legal actions (floor)
  conservative: rule-based ConservativeAgent (current default baseline)
  cfr         : tabular CFR strategy (strong baseline; requires CFR CSVs
                under cfr_ai/outputs/ — see cfr_ai/agent.py:28)
  snapshot    : another NFSP checkpoint loaded for opposing play
                (self-improvement signal)

Run as a module from the repository root:

  python -m tools.eval_ladder \\
    --checkpoint runs/<id>/checkpoints/nfsp_blef.pt \\
    --opponents random,conservative \\
    --n-games 200 \\
    --deck-size 24 --n-players 2 --max-cards 11

Writes a CSV of one row per (opponent, config) to --output (default stdout).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple

import torch

# Project imports — assume repo root is on sys.path (run as `python -m tools.eval_ladder`).
from conservative_ai.agent import ConservativeAgent
from nfsp_ai.agent import NFSPAgent, NFSPConfig
from nfsp_ai.nfsp_run_local import MyEnv, _compute_obs_dim


# --------------------------------------------------------------------------
# Opponents
# --------------------------------------------------------------------------

class Opponent:
    """Minimal interface: pick a legal action from the current env state."""

    name: str = "abstract"

    def act(self, game_state: Dict, obs: torch.Tensor, mask: torch.Tensor) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        """Optional per-game reset hook."""
        pass


def _legal_or_random(action: Optional[int], mask: torch.Tensor) -> int:
    """Validate `action` against `mask`; fall back to a random legal action."""
    legal = mask.nonzero(as_tuple=False).view(-1).tolist()
    if action is None or not legal:
        return int(legal[0]) if legal else 0
    a = int(action)
    if 0 <= a < mask.shape[-1] and mask[a] > 0:
        return a
    return int(random.choice(legal)) if legal else 0


class RandomOpponent(Opponent):
    name = "random"

    def act(self, game_state, obs, mask):
        legal = mask.nonzero(as_tuple=False).view(-1).tolist()
        return int(random.choice(legal)) if legal else 0


class ConservativeOpponent(Opponent):
    name = "conservative"

    def act(self, game_state, obs, mask):
        try:
            a = ConservativeAgent.determine_action(game_state)
        except Exception:
            a = None
        return _legal_or_random(a, mask)


class CFROpponent(Opponent):
    """
    Loads CFR strategy CSVs lazily. If the strategy files aren't present,
    `available()` returns False and the ladder skips this opponent.

    See cfr_ai/agent.py for the CSV layout. The CSVs live under
    cfr_ai/outputs/<hand_sizes>/<key0>/<key1>.csv plus a top-level metadata.csv.
    """

    name = "cfr"

    def __init__(self):
        self._available = self._probe()

    def available(self) -> bool:
        return self._available

    @staticmethod
    def _probe() -> bool:
        # CFR's determine_action requires both metadata.csv and an outputs/ dir
        # populated for the configs we'll ask about. Cheapest probe: check that
        # cfr_ai/outputs exists and is non-empty, AND metadata.csv exists.
        if not os.path.exists("metadata.csv"):
            return False
        outputs = "cfr_ai/outputs"
        if not os.path.isdir(outputs):
            return False
        try:
            return any(os.scandir(outputs))
        except OSError:
            return False

    def act(self, game_state, obs, mask):
        if not self._available:
            return _legal_or_random(None, mask)
        try:
            from cfr_ai.agent import determine_action as cfr_determine
            a = cfr_determine(game_state)
        except (FileNotFoundError, KeyError, IndexError, ValueError, Exception):
            a = None
        return _legal_or_random(a, mask)


class SnapshotOpponent(Opponent):
    """Loads an NFSP checkpoint for inference-only opposition."""

    def __init__(self, checkpoint_path: str, obs_dim: int, act_dim: int, device: Optional[torch.device] = None, label: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self.name = label or f"snapshot:{os.path.basename(checkpoint_path)}"
        self.device = device or torch.device("cpu")
        # Tiny buffers — we don't train, but NFSPAgent's __init__ allocates them.
        cfg = NFSPConfig(rl_capacity=1, sl_capacity=1)
        self.agent = NFSPAgent(obs_dim, act_dim, device=self.device, cfg=cfg)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        # Use the inference-relevant subset; tolerate both full and exported payloads.
        if "q" in ckpt:
            self.agent.q.load_state_dict(ckpt["q"])
        if "pi" in ckpt:
            self.agent.pi.load_state_dict(ckpt["pi"])

    def act(self, game_state, obs, mask):
        return self.agent.select_action(obs, mask, use_average_policy=True, greedy=True)


class BRSelfOpponent(Opponent):
    """Best-response opponent using the *same* checkpoint's Q-net — i.e. the
    learner's own Q. Used for exploitability estimation: if BR-vs-avg wins
    significantly above 0.5, the avg policy is exploitable (sub-Nash).

    NashConv lower bound = 2 * (BR_winrate - 0.5), reaching 0 at Nash.
    """

    def __init__(self, checkpoint_path: str, obs_dim: int, act_dim: int,
                 device: Optional[torch.device] = None, label: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self.name = label or "br-self"
        self.device = device or torch.device("cpu")
        # Detect hidden width to match the learner's architecture.
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        try:
            hidden = _detect_hidden_from_checkpoint(state)
        except Exception:
            hidden = 128
        cfg = NFSPConfig(rl_capacity=1, sl_capacity=1, hidden=hidden)
        # Sniff factorize_action_head from the saved cfg if present.
        saved_cfg = state.get("cfg") or {}
        cfg.factorize_action_head = bool(saved_cfg.get("factorize_action_head", False))
        self.agent = NFSPAgent(obs_dim, act_dim, device=self.device, cfg=cfg)
        if "q" in state:
            self.agent.q.load_state_dict(state["q"])
        if "pi" in state:
            self.agent.pi.load_state_dict(state["pi"])

    def act(self, game_state, obs, mask):
        # Best-response: argmax over Q-values on legal actions.
        return self.agent.select_action(obs, mask, use_average_policy=False, greedy=True)


# --------------------------------------------------------------------------
# Match runner
# --------------------------------------------------------------------------

@dataclass
class MatchResult:
    opponent: str
    config: str
    n_games: int
    wins: int
    losses: int
    winrate: float
    winrate_ci_95: float
    mean_reward: float
    mean_game_length_actions: float
    elapsed_seconds: float
    metadata: Dict = field(default_factory=dict)


def _winrate_ci_95(wins: int, n: int) -> float:
    if n <= 0:
        return 0.0
    p = wins / n
    return 1.96 * math.sqrt(p * (1.0 - p) / n)


def head_to_head(
    learner: NFSPAgent,
    opponent: Opponent,
    *,
    n_games: int,
    deck_size: int = 24,
    n_players: int = 2,
    max_cards: int = 11,
    jokers: int = 0,
    blanks: int = 0,
    common_cards: int = 0,
    seed: Optional[int] = None,
    greedy_learner: bool = True,
    use_average_policy: bool = True,
    card_embedding=None,
    history_embedding=None,
) -> MatchResult:
    """Run `n_games` between learner and opponent. Learner plays the env's
    randomly-chosen reference seat each game; opponent plays all other seats.

    Counts per-round outcomes from the *learner's* perspective by inspecting
    ``info["round_result"]["loser"]`` at each terminal — same convention as
    the trainer's in-loop ``_evaluate_policy``. The env's raw step reward is
    actor-relative (positive when the actor wins), which is misleading when
    the opponent is the actor at a round terminal.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    env = MyEnv(
        n_agents=n_players,
        max_cards=max_cards,
        deck_size=deck_size,
        jokers=jokers,
        blanks=blanks,
        common_cards=common_cards,
        card_embedding=card_embedding,
        history_embedding=history_embedding,
    )

    # Sanity: env obs_dim must match the learner's expected input.
    # Learners trained with different rules / embeddings have different
    # obs_dims that aren't recorded in the checkpoint, so configuration
    # has to be supplied at eval time. Catch the mismatch with a clear
    # error rather than letting torch raise "mat1 and mat2 shapes ...".
    probe_obs, _probe_mask, _ = env.reset()
    if int(probe_obs.shape[-1]) != int(learner.obs_dim):
        raise ValueError(
            f"Env obs_dim {int(probe_obs.shape[-1])} does not match learner obs_dim "
            f"{int(learner.obs_dim)} for checkpoint. The checkpoint was trained with "
            f"different rules (deck_size, jokers, blanks, common_cards) or with embeddings "
            f"enabled. Re-run with matching --deck-size / --jokers / --blanks / --common-cards "
            f"flags."
        )
    # Reset env so the first game starts cleanly.
    env = MyEnv(
        n_agents=n_players,
        max_cards=max_cards,
        deck_size=deck_size,
        jokers=jokers,
        blanks=blanks,
        common_cards=common_cards,
        card_embedding=card_embedding,
        history_embedding=history_embedding,
    )

    rounds_won = rounds_lost = 0
    total_actions = 0
    t0 = time.time()
    # Per-round-bin counters keyed by (learner_n_cards, opp_n_cards) at the
    # START of the round (read from round_result.before_counts when the round
    # ends). Lets callers separate "good late-game" from "compounded small
    # per-round edge across many rounds" — the latter masquerades as the
    # former in raw winrate when games start at 1 card and accumulate up to
    # max_cards. The 2-D form (our_cards × opp_cards) lets us report a true
    # late-game-conditional table; the 1-D collapse over the off-axis is
    # written to MatchResult.metadata for backwards compat.
    wins_2d: Dict[Tuple[int, int], int] = {}
    losses_2d: Dict[Tuple[int, int], int] = {}

    def _ref_seat() -> Optional[str]:
        players = env.game.get("players", []) or []
        for i, p in enumerate(players):
            if p.get("nickname") == env._ref_nick:
                return str(i)
        return None

    for _ in range(n_games):
        opponent.reset()
        obs, mask, _pid = env.reset()
        actions_this_game = 0
        done = False

        while not done:
            cp = env.game.get("cp_nickname")
            ref = env._ref_nick
            game_state = env.game
            if cp == ref:
                action = learner.select_action(
                    obs, mask,
                    use_average_policy=use_average_policy,
                    greedy=greedy_learner,
                )
                action = _legal_or_random(action, mask)
            else:
                action = opponent.act(game_state, obs, mask)
                action = _legal_or_random(action, mask)

            obs, mask, _reward, done, info = env.step(int(action))
            actions_this_game += 1

            # Per-round bookkeeping from the learner's perspective.
            # Mirrors `_evaluate_policy` in nfsp_ai/agent.py.
            if info and isinstance(info, dict):
                rr = info.get("round_result") or {}
                loser = rr.get("loser")
                if loser:
                    seat = _ref_seat()
                    before = (rr.get("before_counts") or {})
                    our_c = int(before.get(seat, 0)) if seat is not None else 0
                    # Opponent count = sum of every other seat's before_count.
                    opp_c = 0
                    for k, v in before.items():
                        if k != seat:
                            try:
                                opp_c += int(v)
                            except Exception:
                                pass
                    key = (our_c, opp_c)
                    if loser == ref:
                        rounds_lost += 1
                        losses_2d[key] = losses_2d.get(key, 0) + 1
                    else:
                        rounds_won += 1
                        wins_2d[key] = wins_2d.get(key, 0) + 1

        total_actions += actions_this_game

    elapsed = time.time() - t0
    settled = rounds_won + rounds_lost
    winrate = rounds_won / settled if settled else 0.0
    return MatchResult(
        opponent=opponent.name,
        config=f"deck{deck_size}_p{n_players}_mc{max_cards}_j{jokers}_b{blanks}_cc{common_cards}",
        n_games=n_games,
        wins=rounds_won,
        losses=rounds_lost,
        winrate=winrate,
        winrate_ci_95=_winrate_ci_95(rounds_won, settled),
        mean_reward=(rounds_won - rounds_lost) / max(1, n_games),
        mean_game_length_actions=total_actions / max(1, n_games),
        elapsed_seconds=elapsed,
        metadata={
            "seed": seed,
            "greedy": greedy_learner,
            "wins_2d": dict(wins_2d),    # keyed by (our_n_cards, opp_n_cards)
            "losses_2d": dict(losses_2d),
        },
    )


# --------------------------------------------------------------------------
# Loading the learner
# --------------------------------------------------------------------------

def _detect_dims_from_checkpoint(ckpt_path: str) -> Tuple[int, int]:
    """Inspect a saved NFSP checkpoint and return (obs_dim, act_dim)."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pi = state.get("pi") or state.get("q") or {}
    # PolicyNet first linear weight has shape [hidden, obs_dim]; output head [act_dim, hidden].
    weights = list(pi.values()) if isinstance(pi, dict) else []
    obs_dim = act_dim = None
    for k, v in (pi.items() if isinstance(pi, dict) else []):
        if k.endswith("enc.0.weight") or k.endswith("net.0.weight"):
            obs_dim = int(v.shape[1])
        if k.endswith("pi.weight") or (k.endswith("net.4.weight") and act_dim is None):
            act_dim = int(v.shape[0])
    if obs_dim is None or act_dim is None:
        # Fallback: use the explicit "obs_dim"/"act_dim" if exported-inference shape
        obs_dim = obs_dim or int(state.get("obs_dim") or 0)
        act_dim = act_dim or int(state.get("act_dim") or 0)
    if not (obs_dim and act_dim):
        raise ValueError(f"Could not infer obs_dim/act_dim from checkpoint {ckpt_path}")
    return obs_dim, act_dim


def _detect_hidden_from_checkpoint(state: dict) -> int:
    """Infer the hidden width from the Q net's first-layer output dim. Falls
    back to 128 if not present (older checkpoints)."""
    q = state.get("q") or {}
    for k, v in (q.items() if isinstance(q, dict) else []):
        if k.endswith("net.0.weight"):
            return int(v.shape[0])
    return 128


def load_learner(checkpoint_path: str, device: Optional[torch.device] = None) -> NFSPAgent:
    device = device or torch.device("cpu")
    obs_dim, act_dim = _detect_dims_from_checkpoint(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden = _detect_hidden_from_checkpoint(state)
    cfg = NFSPConfig(rl_capacity=1, sl_capacity=1, hidden=hidden)  # don't allocate large buffers for inference
    agent = NFSPAgent(obs_dim, act_dim, device=device, cfg=cfg)
    if "q" in state:
        agent.q.load_state_dict(state["q"])
    if "pi" in state:
        agent.pi.load_state_dict(state["pi"])
    return agent


# --------------------------------------------------------------------------
# Ladder driver
# --------------------------------------------------------------------------

OPPONENT_REGISTRY: Dict[str, Callable[[int, int], Opponent]] = {
    "random": lambda obs_dim, act_dim: RandomOpponent(),
    "conservative": lambda obs_dim, act_dim: ConservativeOpponent(),
    "cfr": lambda obs_dim, act_dim: CFROpponent(),
}


def build_opponents(
    spec: str,
    obs_dim: int,
    act_dim: int,
    snapshot_paths: Optional[List[str]] = None,
    self_checkpoint_path: Optional[str] = None,
) -> List[Opponent]:
    out: List[Opponent] = []
    for name in (spec.split(",") if spec else []):
        name = name.strip()
        if not name:
            continue
        if name == "snapshot":
            for p in (snapshot_paths or []):
                out.append(SnapshotOpponent(p, obs_dim, act_dim))
            continue
        if name == "br-self":
            if self_checkpoint_path is None:
                raise ValueError("br-self opponent requires --checkpoint to be provided")
            out.append(BRSelfOpponent(self_checkpoint_path, obs_dim, act_dim))
            continue
        builder = OPPONENT_REGISTRY.get(name)
        if not builder:
            raise ValueError(f"Unknown opponent: {name}")
        opp = builder(obs_dim, act_dim)
        if isinstance(opp, CFROpponent) and not opp.available():
            print(f"[warn] CFR opponent unavailable (cfr_ai/outputs/ missing or empty); skipping", file=sys.stderr)
            continue
        out.append(opp)
    return out


def run_ladder(
    learner: NFSPAgent,
    opponents: List[Opponent],
    *,
    n_games: int,
    deck_size: int = 24,
    n_players: int = 2,
    max_cards: int = 11,
    jokers: int = 0,
    blanks: int = 0,
    common_cards: int = 0,
    seed: Optional[int] = None,
    card_embedding=None,
    history_embedding=None,
) -> List[MatchResult]:
    results: List[MatchResult] = []
    for opp in opponents:
        r = head_to_head(
            learner, opp,
            n_games=n_games,
            deck_size=deck_size,
            n_players=n_players,
            max_cards=max_cards,
            jokers=jokers,
            blanks=blanks,
            common_cards=common_cards,
            seed=seed,
            card_embedding=card_embedding,
            history_embedding=history_embedding,
        )
        results.append(r)
    return results


def write_per_bin_csv(path: str, results: List[MatchResult]) -> None:
    """Per-(our_n_cards, opp_n_cards) winrate breakdown across all results."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = ["opponent", "config", "our_n_cards", "opp_n_cards",
                  "wins", "losses", "winrate", "winrate_ci_95"]
    has_existing_data = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a" if has_existing_data else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not has_existing_data:
            w.writeheader()
        for r in results:
            wins_2d = r.metadata.get("wins_2d", {}) or {}
            losses_2d = r.metadata.get("losses_2d", {}) or {}
            keys = set(wins_2d) | set(losses_2d)
            for k in sorted(keys):
                wn = int(wins_2d.get(k, 0))
                ls = int(losses_2d.get(k, 0))
                tot = wn + ls
                if tot == 0:
                    continue
                wr = wn / tot
                w.writerow({
                    "opponent": r.opponent,
                    "config": r.config,
                    "our_n_cards": k[0],
                    "opp_n_cards": k[1],
                    "wins": wn,
                    "losses": ls,
                    "winrate": f"{wr:.4f}",
                    "winrate_ci_95": f"{_winrate_ci_95(wn, tot):.4f}",
                })


def write_results_csv(path: str, results: List[MatchResult]) -> None:
    fieldnames = [
        "opponent", "config", "n_games", "wins", "losses",
        "winrate", "winrate_ci_95", "mean_reward",
        "mean_game_length_actions", "elapsed_seconds",
    ]
    new_file = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a" if not new_file else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in fieldnames}
            w.writerow(row)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Path to .pt NFSP checkpoint.")
    p.add_argument(
        "--opponents",
        default="random,conservative",
        help=(
            "Comma-separated subset of "
            "{random, conservative, cfr, snapshot, br-self}. "
            "`br-self` uses the same checkpoint's Q-net as a best-response "
            "opponent — winrate above 0.5 = avg policy is exploitable, "
            "i.e. NashConv lower bound = 2*(BR-vs-avg-winrate - 0.5)."
        ),
    )
    p.add_argument("--snapshot-paths", default="", help="Comma-separated checkpoint paths (used when opponents includes 'snapshot').")
    p.add_argument("--n-games", type=int, default=200)
    p.add_argument("--deck-size", type=int, default=24, choices=[24, 32])
    p.add_argument("--n-players", type=int, default=2)
    p.add_argument("--max-cards", type=int, default=11)
    p.add_argument("--jokers", type=int, default=0)
    p.add_argument("--blanks", type=int, default=0)
    p.add_argument("--common-cards", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="", help="CSV output path. Empty = stdout.")
    p.add_argument("--bin-output", default="", help="Per-(our_n_cards, opp_n_cards) breakdown CSV path. Empty = skip.")
    p.add_argument(
        "--use-card-embeddings",
        nargs="?",
        const="auto",
        default=None,
        help="Path to a pretrained card-embedding artifact (.pt). Required when "
             "the learner checkpoint was trained with embeddings; otherwise the "
             "env obs_dim won't match the learner's input dim.",
    )
    p.add_argument(
        "--use-history-embeddings",
        nargs="?",
        const="auto",
        default=None,
        help="Path to a pretrained history-embedding artifact (.pt).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    learner = load_learner(args.checkpoint)
    snap_paths = [p for p in args.snapshot_paths.split(",") if p.strip()]
    # Load embeddings if requested. Required for checkpoints trained with
    # --use-card-embeddings / --use-history-embeddings, since the env obs_dim
    # depends on whether embedding features are appended.
    card_embedding = None
    history_embedding = None
    if args.use_card_embeddings:
        from nfsp_ai.nfsp_run_local import (
            _resolve_card_embedding_path,
            _load_card_embedding,
        )
        path = _resolve_card_embedding_path(args.use_card_embeddings, args.deck_size)
        if path is None:
            print(f"warning: --use-card-embeddings {args.use_card_embeddings!r} did not resolve to a file", file=sys.stderr)
        else:
            card_embedding = _load_card_embedding(path, device="cpu")
    if args.use_history_embeddings:
        from nfsp_ai.nfsp_run_local import (
            _resolve_history_embedding_path,
            _load_history_embedding,
        )
        path = _resolve_history_embedding_path(args.use_history_embeddings, args.deck_size)
        if path is None:
            print(f"warning: --use-history-embeddings {args.use_history_embeddings!r} did not resolve to a file", file=sys.stderr)
        else:
            history_embedding = _load_history_embedding(path, device="cpu")

    opponents = build_opponents(args.opponents, learner.obs_dim, learner.act_dim,
                                  snapshot_paths=snap_paths,
                                  self_checkpoint_path=args.checkpoint)
    if not opponents:
        print("No opponents constructed — exiting.", file=sys.stderr)
        return 2
    results = run_ladder(
        learner, opponents,
        n_games=args.n_games,
        deck_size=args.deck_size,
        n_players=args.n_players,
        max_cards=args.max_cards,
        jokers=args.jokers,
        blanks=args.blanks,
        common_cards=args.common_cards,
        seed=args.seed,
        card_embedding=card_embedding,
        history_embedding=history_embedding,
    )
    if args.output:
        write_results_csv(args.output, results)
        print(f"Wrote {len(results)} rows to {args.output}")
    if args.bin_output:
        write_per_bin_csv(args.bin_output, results)
        print(f"Wrote per-bin breakdown to {args.bin_output}")
    else:
        w = csv.DictWriter(
            sys.stdout,
            fieldnames=[
                "opponent", "config", "n_games", "wins", "losses",
                "winrate", "winrate_ci_95", "mean_reward",
                "mean_game_length_actions", "elapsed_seconds",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k) for k in w.fieldnames})
    return 0


if __name__ == "__main__":
    sys.exit(main())
