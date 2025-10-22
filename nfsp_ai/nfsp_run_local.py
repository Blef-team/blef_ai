# nfsp_run_local.py

# RUN FROM REPOSITORY ROOT /

# Adapter for NFSP <-> simpleschema_local_manager with legality, jokers, and common cards.
import argparse
import os
import random
from datetime import datetime
import math
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import torch

import shared.api.simpleschema_local_manager as gm                 # local manager
from shared.probabilities.dynamic_probabilities import get_bet_probabilities, get_generic_bet_probabilities
from nfsp_ai.agent import NFSPAgent, NFSPConfig          # your NFSP implementation
from nfsp_ai.control_plane import JsonControlPlane, build_control_snapshot, write_control_file
from shared.game_utils import GameRules


@dataclass
class DeckSpec:
    deck_size: int
    num_values: int
    num_actions: int
    check_action_id: int
    hand_vec_dim: int
    hist_dim: int
    obs_dim: int


def _compute_obs_dim(hand_vec_dim: int, hist_dim: int) -> int:
    return (
        hand_vec_dim
        + 1  # rules jokers
        + 1  # private jokers
        + 1  # common jokers
        + hand_vec_dim
        + MAX_PLAYERS
        + 1  # round scaling
        + hist_dim
        + 1  # last bet prob
        + hist_dim  # private priors
        + hist_dim  # public priors
        + 1  # total blanks
    )


def _build_deck_spec(rules: Dict) -> DeckSpec:
    deck_size = int(rules.get("deck_size", 24))
    if deck_size % 4 != 0:
        raise ValueError(f"Unsupported deck_size {deck_size}; must be divisible by four.")
    gr = GameRules(deck_size)
    num_values = deck_size // 4
    num_actions = gr.num_actions
    check_action_id = gr.check_action_id
    hand_vec_dim = num_values * 4
    hist_dim = check_action_id
    obs_dim = _compute_obs_dim(hand_vec_dim, hist_dim)
    return DeckSpec(
        deck_size=deck_size,
        num_values=num_values,
        num_actions=num_actions,
        check_action_id=check_action_id,
        hand_vec_dim=hand_vec_dim,
        hist_dim=hist_dim,
        obs_dim=obs_dim,
    )

MAX_PLAYERS = 8


# ---------- Utilities ----------

def _nickname_to_idx(players: List[dict], nick: str) -> int:
    return next((i for i, p in enumerate(players) if p["nickname"] == nick), -1)


def _deck_spec_from_game(game: dict) -> DeckSpec:
    rules = game.get("rules", {}) or {}
    return _build_deck_spec(rules)


def _current_hand(game: dict, nick: str) -> List[dict]:
    for h in game.get("hands", []):
        if h["nickname"] == nick:
            return h["hand"]
    return []


def _count_jokers(cards: List[dict]) -> int:
    return sum(1 for c in cards if int(c.get("value", 0)) < 0)


def _last_bet_action_id(game: dict, spec: DeckSpec) -> int:
    """Return the last *bet* action id (< check_action_id) if any; else -1."""
    hist = game.get("history", []) or []
    if not hist:
        return -1
    check = spec.check_action_id
    for ev in reversed(hist):
        try:
            act = int(ev.get("action_id", -1))
        except Exception:
            continue
        if 0 <= act < check:
            return act
    return -1


def _legal_action_mask(game: dict, spec: DeckSpec) -> torch.Tensor:
    """Compute legality exactly as enforced by manager.play()."""
    mask = np.zeros((spec.num_actions,), dtype=np.float32)
    check = spec.check_action_id

    hist = game.get("history", []) or []
    if not hist:
        mask[:check] = 1.0
        return torch.from_numpy(mask)

    try:
        last = int(hist[-1].get("action_id", -1))
    except Exception:
        last = -1

    if last == check:
        return torch.from_numpy(mask)

    next_min = min(check, last + 1) if last >= 0 else 0
    if next_min < check:
        mask[next_min:check] = 1.0
    mask[check] = 1.0
    return torch.from_numpy(mask)


# ---------- Observation encoding ----------

def _seat_index(players: List[dict], nick: str) -> int:
    for i, p in enumerate(players or []):
        if p.get("nickname") == nick:
            return i
    return 0

def _rotate_list(xs: List[int], k: int) -> List[int]:
    if not xs:
        return xs
    k %= len(xs)
    return xs[k:] + xs[:k]

def _rules_total_jokers(rules: dict) -> int:
    """
    Total possible jokers per rules (not just seen on table).
    Accepts either:
      - rules["jokers"] as int, or
      - rules["jokers"] as dict with "num"
    Fallback: 0
    """
    j = rules.get("jokers", 0)
    if isinstance(j, dict):
        return int(j.get("num", 0))
    return int(j)

def _rules_total_blanks(rules: dict) -> int:
    """Total possible blanks per rules (not just seen on table)."""
    return int(rules.get("blanks", 0))

def _card_index(value: int, colour: int, spec: DeckSpec) -> int:
    if value < 0 or colour < 0:
        return -1
    if value >= spec.num_values or colour >= 4:
        return -1
    return value * 4 + colour


def _multi_hot_cards(cards: List[dict], spec: DeckSpec) -> np.ndarray:
    vec = np.zeros((spec.hand_vec_dim,), dtype=np.float32)
    for card in cards or []:
        v = int(card.get("value", -2))
        c = int(card.get("colour", -2))
        idx = _card_index(v, c, spec)
        if idx >= 0:
            vec[idx] = 1.0
    return vec

def _count_jokers_from_ints(cards: List[dict]) -> int:
    return sum(1 for c in (cards or []) if int(c.get("value", -2)) == -1)

def _count_blanks_from_ints(cards: List[dict]) -> int:
    return sum(1 for c in (cards or []) if int(c.get("value", -2)) == -2)

def _current_hand_int(game: dict, nick: str) -> List[dict]:
    for h in game.get("hands", []) or []:
        if h.get("nickname") == nick:
            return h.get("hand", []) or []
    return []

def _common_hand_int(game: dict) -> List[dict]:
    return game.get("common_hand", []) or []

def _round_history_action_ids(game: dict, hist_dim: int) -> List[int]:
    ids = []
    for ev in game.get("history", []) or []:
        try:
            a = int(ev.get("action_id"))
            if 0 <= a < hist_dim:
                ids.append(a)
        except Exception:
            continue
    return ids


def vectorize_obs(game: dict, nick: str, spec: DeckSpec) -> torch.Tensor:
    """
    Updated observation for current player (nick) with minimal allocations.
    """
    rules = game.get("rules", {}) or {}
    players = game.get("players", []) or []

    obs = np.zeros((spec.obs_dim,), dtype=np.float32)
    idx = 0

    # Private hand multi-hot
    my_hand = _current_hand_int(game, nick)
    obs[idx:idx + spec.hand_vec_dim] = _multi_hot_cards(my_hand, spec)
    idx += spec.hand_vec_dim

    # Totals & joker counts
    obs[idx] = float(_rules_total_jokers(rules)); idx += 1
    obs[idx] = float(_count_jokers_from_ints(my_hand)); idx += 1
    common_hand = _common_hand_int(game)
    obs[idx] = float(_count_jokers_from_ints(common_hand)); idx += 1

    # Common cards multi-hot
    obs[idx:idx + spec.hand_vec_dim] = _multi_hot_cards(common_hand, spec)
    idx += spec.hand_vec_dim

    # Seat counts rotated so current player is seat 0
    my_seat = _seat_index(players, game.get("cp_nickname", nick))
    counts = [int(p.get("n_cards", 0)) for p in players]
    counts = (counts + [0] * (MAX_PLAYERS - len(counts)))[:MAX_PLAYERS]
    counts = _rotate_list(counts, my_seat)
    obs[idx:idx + MAX_PLAYERS] = counts[:MAX_PLAYERS]
    idx += MAX_PLAYERS

    # Round scaling
    cur_round = int(game.get("round_number", 1))
    max_rounds = int(rules.get("max_rounds", 4))
    obs[idx] = float(min(max(cur_round / max(1, max_rounds), 0.0), 1.0))
    idx += 1

    # Action history multi-hot
    hist_ids = _round_history_action_ids(game, spec.hist_dim)
    if hist_ids:
        obs[idx + np.unique(hist_ids)] = 1.0
    idx += spec.hist_dim

    # Last bet probability
    last_bet_id = _last_bet_action_id(game, spec)
    prob_last_bet_exists = get_bet_probabilities(
        game_state=game,
        for_betting=False,
        specific_action_id=last_bet_id,
    )
    obs[idx] = float(prob_last_bet_exists)
    idx += 1

    # Private priors (depends on my hand)
    pvt_slice = slice(idx, idx + spec.hist_dim)
    pvt_prior = get_bet_probabilities(
        game_state=game,
        for_betting=True,
        last_bet=last_bet_id,
    )
    obs[pvt_slice] = pvt_prior
    idx += spec.hist_dim

    # Public priors
    pub_slice = slice(idx, idx + spec.hist_dim)
    pub_prior = get_generic_bet_probabilities(
        game_state=game,
        last_bet=last_bet_id,
    )
    obs[pub_slice] = pub_prior
    idx += spec.hist_dim

    # Total blanks
    obs[idx] = float(_rules_total_blanks(rules))
    idx += 1

    if idx != spec.obs_dim:
        raise ValueError(f"vectorize_obs produced dim {idx}, expected {spec.obs_dim}")

    # Validations on probability slices
    if len(pvt_prior) != spec.hist_dim:
        raise ValueError(
            f"get_bet_probabilities() returned length {len(pvt_prior)}, expected {spec.hist_dim}"
        )
    if len(pub_prior) != spec.hist_dim:
        raise ValueError(
            f"get_generic_bet_probabilities() returned length {len(pub_prior)}, expected {spec.hist_dim}"
        )

    if not np.isfinite(obs[pvt_slice]).all():
        raise ValueError("get_bet_probabilities() returned non-finite values")
    if not np.isfinite(obs[pub_slice]).all():
        raise ValueError("get_generic_bet_probabilities() returned non-finite values")

    if not ((0.0 <= obs[pvt_slice]).all() and (obs[pvt_slice] <= 1.0).all()):
        raise ValueError("get_bet_probabilities() returned values outside [0, 1]")
    if not ((0.0 <= obs[pub_slice]).all() and (obs[pub_slice] <= 1.0).all()):
        raise ValueError("get_generic_bet_probabilities() returned values outside [0, 1]")

    return torch.from_numpy(obs)

# ---------- Environment Adapter ----------

class MyEnv:
    """
    Minimal TurnEnvAdapter for nfsp_complex.NFSPAgent:
      reset() -> (obs, mask, pid)
      step(action) -> (obs, mask, reward, done, pid)

    reward: ±1 on CHECK resolution, measured from a fixed reference player's perspective.
            0 otherwise.
    done: True only when the game finishes or after the reference player is eliminated.
    """

    def __init__(
        self,
        n_agents: int = 2,
        max_cards: int = 11,
        deck_size: int = 24,
        jokers: int = 0,
        blanks: int = 0,
        verbose: bool = False,
        illegal_penalty: float = -0.01,
        game_save_dir: Optional[str] = None,
        save_sample_rate: int = 5000,
    ):
        if n_agents < 2 or n_agents > 8:
            raise ValueError("n_agents must be in [2, 8]")
        self.n_agents = n_agents
        self.max_cards = max_cards
        self.rules = {"deck_size": int(deck_size), "jokers": int(jokers), "blanks": int(blanks)}
        self.deck_spec = _build_deck_spec(self.rules)
        self.verbose = verbose
        self.illegal_penalty = float(illegal_penalty)
        self.game_save_dir = game_save_dir
        self.save_sample_rate = max(1, int(save_sample_rate))
        self.game: Dict = {}
        self._last_obs = None
        self._last_mask = None
        self._ref_nick: Optional[str] = None
        self.rounds_since_reset = 0
        self._save_counter = 0

    def _pid(self) -> int:
        return _nickname_to_idx(self.game["players"], self.game.get("cp_nickname", ""))

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        self.game = gm.create_game(
            self.n_agents,
            deck_size=self.rules["deck_size"],
            max_cards=self.max_cards,
            jokers=self.rules["jokers"],
            blanks=self.rules["blanks"],
            verbose=self.verbose,
        )
        self.rules = dict(self.game.get("rules", self.rules))
        self.rules = dict(self.game.get("rules", self.rules))
        self.deck_spec = _deck_spec_from_game(self.game)
        self.rounds_since_reset = 0
        players = self.game.get("players", []) or []
        if not players:
            raise RuntimeError("Game manager returned no players")
        self._ref_nick = random.choice([p["nickname"] for p in players])
        cp = self.game["cp_nickname"]
        obs = vectorize_obs(self.game, cp, self.deck_spec).float()
        mask = _legal_action_mask(self.game, self.deck_spec).float()
        pid = self._pid()
        self._last_obs, self._last_mask = obs, mask
        return obs, mask, pid

    def step(self, action: int):
        """
        Apply action for the current player.
        - If action is illegal (shouldn't happen if mask is used), return same state + small penalty.
        - If the chosen action is the check action: the manager resolves the round internally; we compute reward by
          comparing players' n_cards before vs after the move.
        """
        actor_nick = self.game.get("cp_nickname")
        before_counts = {p["nickname"]: int(p["n_cards"]) for p in self.game.get("players", [])}
        status_before = self.game.get("status", "Running")
        history_before = [
            {"player": ev.get("player"), "action_id": int(ev.get("action_id", -1))}
            for ev in (self.game.get("history") or [])
        ]

        # Try to act; catch and handle illegal attempts gracefully.
        save_dir = None
        should_save = False
        if self.game_save_dir:
            if self._save_counter >= self.save_sample_rate - 1:
                save_dir = self.game_save_dir
                should_save = True
        try:
            action_int = int(action)
        except Exception:
            obs = self._last_obs.clone()
            mask = self._last_mask.clone()
            pid = self._pid()
            info = {"next_pid": pid, "illegal": 1}
            return obs, mask, float(self.illegal_penalty), False, info
        is_check = action_int == self.deck_spec.check_action_id
        try:
            gm.play(self.game, action_int, save_dir=save_dir, verbose=self.verbose)
            if should_save:
                self._save_counter = 0
            elif self.game_save_dir:
                self._save_counter += 1
        except Exception:
            # Return same obs/mask/pid with a penalty; do NOT advance player.
            obs = self._last_obs.clone()
            mask = self._last_mask.clone()
            pid = self._pid()
            info = {"next_pid": pid, "illegal": 1}
            return obs, mask, float(self.illegal_penalty), False, info

        self.deck_spec = _deck_spec_from_game(self.game)
        # New observation / mask / pid after the move (may be a new round if the last action was a check)
        cp = self.game.get("cp_nickname")
        obs = vectorize_obs(self.game, cp, self.deck_spec).float()
        mask = _legal_action_mask(self.game, self.deck_spec).float()
        pid = self._pid()

        # Determine terminal and reward
        done = False
        reward = 0.0
        round_result = None
        history_for_log = None

        if is_check:
            # Round has been resolved by the manager, and a new round likely started.
            # Identify loser by delta in n_cards (one player +1, possibly -> 0 on elimination).
            self.rounds_since_reset += 1
            after_counts = {p["nickname"]: int(p["n_cards"]) for p in self.game.get("players", [])}
            loser_candidates = []
            for nick, before in before_counts.items():
                after = after_counts.get(nick, before)
                if after == 0 and before > 0:
                    # Could be elimination at max; treat as increased then zeroed
                    loser_candidates.append(nick)
                elif after > before:
                    loser_candidates.append(nick)

            loser = loser_candidates[0] if loser_candidates else None
            ref = self._ref_nick
            if ref is None:
                reward = 0.0
            elif loser is None:
                reward = 0.0
            elif loser == ref:
                reward = -1.0
            else:
                reward = 1.0
            round_result = {
                "actor": actor_nick,
                "loser": loser,
                "ref": ref,
                "before_counts": before_counts,
                "after_counts": after_counts,
            }
            history_for_log = history_before

        # Game finished? Mark terminal regardless of action.
        if self.game.get("status") == "Finished" and status_before != "Finished":
            done = True
        else:
            # If the reference player has been eliminated from the table, we treat the episode as done.
            players_now = {p["nickname"] for p in self.game.get("players", []) if p["n_cards"] > 0}
            if self._ref_nick is not None and self._ref_nick not in players_now:
                done = True

        # safety cap to avoid non-terminating matches
        MAX_ROUNDS = 50
        assert done or self.rounds_since_reset < MAX_ROUNDS

        self._last_obs, self._last_mask = obs, mask
        info = {
            "next_pid": pid,
            "illegal": 0,
            "action": action_int,
            "history": history_for_log,
            "round_result": round_result,
            "reward": float(reward),
        }
        return obs, mask, float(reward), bool(done), info


def main():
    parser = argparse.ArgumentParser(description="Run NFSP Blef self-play locally.")
    parser.add_argument(
        "--resume",
        dest="resume_path",
        type=str,
        default=None,
        help="Optional path to an NFSP checkpoint (.pt) to resume from.",
    )
    parser.add_argument(
        "--total-steps",
        dest="total_steps",
        type=int,
        default=25_000_000,
        help="Number of environment steps to train for (default: 25,000,000).",
    )
    parser.add_argument(
        "--n-agents",
        dest="n_agents",
        type=int,
        default=2,
        help="Number of seated agents in self-play (default: 2).",
    )
    parser.add_argument(
        "--max-cards",
        dest="max_cards",
        type=int,
        default=11,
        help="Maximum cards per player for Blef variant (default: 3).",
    )
    parser.add_argument(
        "--deck-size",
        dest="deck_size",
        type=int,
        default=24,
        help="Deck size to use (24 or 32).",
    )
    parser.add_argument(
        "--jokers",
        dest="jokers",
        type=int,
        default=0,
        help="Number of jokers to include in the deck (default: 0).",
    )
    parser.add_argument(
        "--blanks",
        dest="blanks",
        type=int,
        default=0,
        help="Number of blanks to include in the deck (default: 0).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging from the game manager.",
    )
    parser.add_argument(
        "--control-plane",
        dest="control_plane",
        type=str,
        default=None,
        help="Path to control-plane JSON file for runtime overrides.",
    )
    parser.add_argument(
        "--control-plane-cooldown",
        dest="control_plane_cooldown",
        type=int,
        default=50_000,
        help="Minimum env steps between control-plane applications (default: 50k).",
    )
    parser.add_argument(
        "--history-sample-every",
        dest="history_sample_every",
        type=int,
        default=100_000,
        help="Env steps between stored action-history samples (default: 100k).",
    )
    parser.add_argument(
        "--history-sample-limit",
        dest="history_sample_limit",
        type=int,
        default=1_000,
        help="Maximum number of history samples to record (default: 1000, <=0 disables limit).",
    )
    parser.add_argument(
        "--history-sample-path",
        dest="history_sample_path",
        type=str,
        default=None,
        help="Optional override for the action-history sample JSONL output.",
    )
    parser.add_argument(
        "--save-game-every",
        dest="save_game_every",
        type=int,
        default=5000,
        help="Persist one full game out of this many (default: 5000).",
    )
    args = parser.parse_args()

    postfix = datetime.now().strftime("%Y%m%d%H%M%S")
    model_save_path = f"nfsp_blef_{postfix}.pt"
    game_save_dir = f"games_{postfix}"
    os.makedirs(game_save_dir, exist_ok=True)
    history_sample_limit = None if args.history_sample_limit <= 0 else args.history_sample_limit
    history_sample_path = args.history_sample_path or os.path.join(
        "logs", f"action_samples_{postfix}.jsonl"
    )
    if history_sample_path:
        print(
            f"[history] samples -> {history_sample_path} (every {args.history_sample_every} steps)"
        )

    env = MyEnv(
        n_agents=args.n_agents,
        max_cards=args.max_cards,
        deck_size=args.deck_size,
        jokers=args.jokers,
        blanks=args.blanks,
        verbose=args.verbose,
        game_save_dir=game_save_dir,
        save_sample_rate=args.save_game_every,
    )
    obs0, mask0, _ = env.reset()

    agent = NFSPAgent(
        obs_dim=obs0.numel(),
        act_dim=mask0.numel(),
        cfg=NFSPConfig(
            anticipatory_eta=0.25,
            batch_rl=64,
            train_rl_every=32,
            batch_sl=256,
            train_sl_every=8,
            lr_q=1e-4,
            lr_pi=3e-4,
            target_tau=0.01,
            hard_target_interval=0,
            warmup_steps=5_000,
            max_grad_norm=10.0,
            gamma=0.995,
            use_double_dqn=True,
            rl_capacity=200_000,
            sl_capacity=200_000,
            n_step=5,
            burst_rl_updates_on_reward=4,
            burst_reward_threshold=0.5,
            #hidden=16
        ),
    )

    if args.resume_path:
        agent.load(args.resume_path)

    control_plane = None
    if args.control_plane:
        cp_path = os.path.abspath(args.control_plane)
        if not os.path.exists(cp_path):
            snapshot = build_control_snapshot(agent, env, cooldown_steps=args.control_plane_cooldown)
            write_control_file(cp_path, snapshot)
            print(f"[control] bootstrap control-plane file written to {cp_path}")
        control_plane = JsonControlPlane(
            cp_path,
            cooldown_steps=args.control_plane_cooldown,
            verbose=True,
        )

    agent.train_from_selfplay(
        env,
        total_steps=args.total_steps,
        log_every=50_000,
        save_path=model_save_path,
        game_save_dir=game_save_dir,
        eval_env_factory=lambda: MyEnv(
            n_agents=env.n_agents,
            max_cards=env.max_cards,
            deck_size=env.rules["deck_size"],
            jokers=env.rules["jokers"],
            blanks=env.rules["blanks"],
            verbose=env.verbose,
            illegal_penalty=env.illegal_penalty,
            game_save_dir=None,
        ),
        control_plane=control_plane,
        history_sample_path=history_sample_path,
        history_sample_every=args.history_sample_every,
        history_sample_limit=history_sample_limit,
    )


if __name__ == "__main__":
    main()
