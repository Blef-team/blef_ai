# nfsp_run_local.py

# RUN FROM REPOSITORY ROOT /

# Adapter for NFSP <-> simpleschema_local_manager with legality, jokers, and common cards.
import argparse
import os
import random
from datetime import datetime
import math
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch

import shared.api.simpleschema_local_manager as gm                 # local manager
from shared.probabilities.dynamic_probabilities import get_bet_probabilities, get_generic_bet_probabilities
from nfsp_ai.agent import NFSPAgent, NFSPConfig          # your NFSP implementation
from nfsp_ai.control_plane import JsonControlPlane, build_control_snapshot, write_control_file

CHECK = gm.CHECK
MAX_PLAYERS = 8

# Deck: values 0..5, colours 0..3 => 24 cards. Jokers value == -1.
N_VALUES = 6
N_COLOURS = 4
HAND_VEC_DIM = N_VALUES * N_COLOURS  # 24
ACT_DIM = CHECK + 1                  # 0..87 plus CHECK(88) => 89


# ---------- Utilities ----------

def _nickname_to_idx(players: List[dict], nick: str) -> int:
    return next((i for i, p in enumerate(players) if p["nickname"] == nick), -1)


def _current_hand(game: dict, nick: str) -> List[dict]:
    for h in game.get("hands", []):
        if h["nickname"] == nick:
            return h["hand"]
    return []


def _count_jokers(cards: List[dict]) -> int:
    return sum(1 for c in cards if int(c.get("value", 0)) < 0)


def _last_bet_action_id(game: dict) -> int:
    """Return the last *bet* action id (0..87) if any; else -1."""
    hist = game.get("history", [])
    if not hist:
        return -1
    last = hist[-1]["action_id"]
    if last == CHECK:
        # If last is CHECK, the round should be resolved already in this manager.
        # But just in case we ever see it in-flight, look one step earlier.
        for ev in reversed(hist[:-1]):
            if 0 <= int(ev["action_id"]) < CHECK:
                return int(ev["action_id"])
        return -1
    return int(last)


def _legal_action_mask(game: dict) -> torch.Tensor:
    """Compute legality exactly as enforced by manager.play()."""
    mask = np.zeros((ACT_DIM,), dtype=np.float32)

    hist = game.get("history", [])
    if not hist:
        # First action in a round: any bet 0..87 is legal; CHECK is illegal.
        mask[:CHECK] = 1.0
        mask[CHECK] = 0.0
        return torch.from_numpy(mask)

    last = int(hist[-1]["action_id"])
    if last == CHECK:
        # According to the manager: cannot act immediately after unresolved CHECK.
        # In practice, the manager resolves and resets the round, so we shouldn't land here.
        # But keep this guard: nothing legal until the round is reset.
        return torch.from_numpy(mask)  # all zeros => forces resample upstream if ever hit.

    # Otherwise last is a bet in [0..87]; legal bets are strictly greater, and CHECK is legal.
    next_min = min(CHECK, last + 1)
    if next_min < CHECK:
        mask[next_min:CHECK] = 1.0
    mask[CHECK] = 1.0
    return torch.from_numpy(mask)


# ---------- Observation encoding ----------

HIST_DIM = ACT_DIM - 1  # 88 (0..87), exclude CHECK channel for history & priors per your spec

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

def _card_idx_24(value: int, colour: int) -> int:
    """
    Map (value, colour) to [0..23] = value*4 + colour.
    Ignore jokers (-1,-1) and blanks (-2,-2) by returning -1.
    """
    if value < 0 or colour < 0:
        return -1
    if value >= N_VALUES or colour >= N_COLOURS:
        return -1
    return value * N_COLOURS + colour

def _multi_hot_cards24_from_ints(cards: List[dict]) -> np.ndarray:
    vec = np.zeros((HAND_VEC_DIM,), dtype=np.float32)
    for c in cards or []:
        v = int(c.get("value", -2))
        s = int(c.get("colour", -2))
        idx = _card_idx_24(v, s)
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

def _round_history_action_ids_88(game: dict) -> List[int]:
    """
    Return action_ids in this round, clipped to [0..87].
    We assume game['history'] contains current round only (as in your example).
    """
    ids = []
    for ev in game.get("history", []) or []:
        try:
            a = int(ev.get("action_id"))
            if 0 <= a < HIST_DIM:
                ids.append(a)
        except Exception:
            continue
    return ids

def vectorize_obs(game: dict, nick: str) -> torch.Tensor:
    """
    Updated observation for current player (nick):
      - 24-dim multi-hot for private hand ranks*suits (ignores jokers & blanks)
      - 1-dim total joker count (possible by rules, not just seen)
      - 1-dim private joker count (in my hand)
      - 1-dim common-hand joker count
      - 24-dim multi-hot common-hand ranks*suits (ignores jokers & blanks)
      - 8-dim public n_cards per seat (padded/rotated so this player is seat 0)
      - 1-dim round number (scaled by rules.get('max_rounds', 4))
      - 88-dim multi-hot of actions taken in THIS round
      - 88-dim prior probabilities: p(action exists | private hand, rules)
      - 88-dim prior probabilities: p(action exists | public counts/common hand/rules)
      - 1-dim total blanks (possible by rules, not just seen)

    Total dim = 24 + 1 + 1 + 1 + 24 + 8 + 1 + 88 + 88 + 88 + 1 = 325
    """
    rules = game.get("rules", {}) or {}
    players = game.get("players", []) or []

    # Hands
    my_hand = _current_hand_int(game, nick)
    common  = _common_hand_int(game)

    my_vec24      = _multi_hot_cards24_from_ints(my_hand)     # 24
    common_vec24  = _multi_hot_cards24_from_ints(common)      # 24
    my_jokers     = float(_count_jokers_from_ints(my_hand))   # 1
    common_jokers = float(_count_jokers_from_ints(common))    # 1
    total_jokers  = float(_rules_total_jokers(rules))         # 1
    total_blanks  = float(_rules_total_blanks(rules))         # 1

    # Seat counts rotated so current player is seat 0
    my_seat = _seat_index(players, game.get("cp_nickname", nick))
    counts  = [int(p.get("n_cards", 0)) for p in players]
    counts  = (counts + [0] * (MAX_PLAYERS - len(counts)))[:MAX_PLAYERS]
    counts  = _rotate_list(counts, my_seat)
    counts8 = np.asarray(counts[:MAX_PLAYERS], dtype=np.float32)     # 8

    # Round scaling (fallback to 1 if not present)
    cur_round  = int(game.get("round_number", 1))
    max_rounds = int(rules.get("max_rounds", 4))
    round_scaled = np.float32(min(max(cur_round / max(1, max_rounds), 0.0), 1.0))  # 1

    # History (88)
    hist_ids = _round_history_action_ids_88(game)
    hist88 = np.zeros((HIST_DIM,), dtype=np.float32)
    if hist_ids:
        hist88[np.unique(hist_ids)] = 1.0

    # Last bet probability, to evaluate checking it
    prob_last_bet_exists = get_bet_probabilities(
        game_state=game,
        for_betting=False,
        specific_action_id=_last_bet_action_id(game),
    )

    # Private prior: depends on my hand and rules (for_betting=True)
    pvt_prior = get_bet_probabilities(
        game_state=game,
        for_betting=True,
        last_bet=_last_bet_action_id(game),
    )
    pvt_prior = np.asarray(pvt_prior, dtype=np.float32)

    # Public prior: depends only on public information
    pub_prior = get_generic_bet_probabilities(
        game_state=game,
        last_bet=_last_bet_action_id(game),
    )
    pub_prior = np.asarray(pub_prior, dtype=np.float32)

    # --- Strict checks ---
    if pvt_prior.shape != (HIST_DIM,):
        raise ValueError(
            f"get_bet_probabilities() returned shape {pvt_prior.shape}, expected {(HIST_DIM,)}"
        )
    if pub_prior.shape != (HIST_DIM,):
        raise ValueError(
            f"get_generic_bet_probabilities() returned shape {pub_prior.shape}, expected {(HIST_DIM,)}"
        )

    # Validate that all values are finite and in [0, 1]
    if not np.isfinite(pvt_prior).all():
        raise ValueError("get_bet_probabilities() returned non-finite values")
    if not np.isfinite(pub_prior).all():
        raise ValueError("get_generic_bet_probabilities() returned non-finite values")

    if not ((0.0 <= pvt_prior).all() and (pvt_prior <= 1.0).all()):
        raise ValueError("get_bet_probabilities() returned values outside [0, 1]")
    if not ((0.0 <= pub_prior).all() and (pub_prior <= 1.0).all()):
        raise ValueError("get_generic_bet_probabilities() returned values outside [0, 1]")


    parts = [
        my_vec24,                                 # 24
        np.asarray([total_jokers], dtype=np.float32),    # 1
        np.asarray([my_jokers], dtype=np.float32),       # 1
        np.asarray([common_jokers], dtype=np.float32),   # 1
        common_vec24,                             # 24
        counts8,                                  # 8
        np.asarray([round_scaled], dtype=np.float32),    # 1
        hist88,                                   # 88
        np.asarray([prob_last_bet_exists], dtype=np.float32),    # 1
        pvt_prior,                                # 88
        pub_prior,                                # 88
        np.asarray([total_blanks], dtype=np.float32),    # 1
    ]

    obs = np.concatenate(parts, axis=0).astype(np.float32)

    # Hard assert schema to catch drift early
    if obs.shape[0] != 326:
        raise ValueError(f"vectorize_obs produced dim {obs.shape[0]}, expected 325")

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

    def __init__(self, n_agents: int = 2, max_cards: int = 11, verbose: bool = False, illegal_penalty: float = -0.01, game_save_dir: Optional[str] = None, save_sample_rate: int = 5000):
        if n_agents < 2 or n_agents > 8:
            raise ValueError("n_agents must be in [2, 8]")
        self.n_agents = n_agents
        self.max_cards = max_cards
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
        self.game = gm.create_game(self.n_agents, max_cards=self.max_cards, verbose=self.verbose)
        self.rounds_since_reset = 0
        players = self.game.get("players", []) or []
        if not players:
            raise RuntimeError("Game manager returned no players")
        self._ref_nick = random.choice([p["nickname"] for p in players])
        cp = self.game["cp_nickname"]
        obs = vectorize_obs(self.game, cp).float()
        mask = _legal_action_mask(self.game).float()
        pid = self._pid()
        self._last_obs, self._last_mask = obs, mask
        return obs, mask, pid

    def step(self, action: int):
        """
        Apply action for the current player.
        - If action is illegal (shouldn't happen if mask is used), return same state + small penalty.
        - If action == CHECK: the manager resolves the round internally; we compute reward by
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
            gm.play(self.game, int(action), save_dir=save_dir, verbose=self.verbose)
            if should_save:
                self._save_counter = 0
            elif self.game_save_dir:
                self._save_counter += 1
        except Exception as e:
            # Return same obs/mask/pid with a penalty; do NOT advance player.
            obs = self._last_obs.clone()
            mask = self._last_mask.clone()
            pid = self._pid()
            info = {"next_pid": pid, "illegal": 1}
            return obs, mask, float(self.illegal_penalty), False, info

        # New observation / mask / pid after the move (may be a new round if CHECK)
        cp = self.game.get("cp_nickname")
        obs = vectorize_obs(self.game, cp).float()
        mask = _legal_action_mask(self.game).float()
        pid = self._pid()

        # Determine terminal and reward
        done = False
        reward = 0.0
        round_result = None
        history_for_log = None

        if int(action) == CHECK:
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
            "action": int(action),
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
