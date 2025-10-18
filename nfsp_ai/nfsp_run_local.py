# nfsp_run_local.py

# RUN FROM REPOSITORY ROOT /

# Adapter for NFSP <-> simpleschema_local_manager with legality, jokers, and common cards.
import math
from typing import Tuple, Dict, List

import numpy as np
import torch

import shared.api.simpleschema_local_manager as gm                 # local manager
from nfsp_ai.agent import NFSPAgent, NFSPConfig          # your NFSP implementation

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

def vectorize_obs(game: dict, nick: str) -> torch.Tensor:
    """
    Observation for current player:
      - 24-dim one-hot for private hand ranks*suits (ignores jokers)
      - 1-dim private joker count
      - 1-dim common-hand size
      - 1-dim common-hand joker count
      - 8-dim public n_cards per seat (padded)
      - 1-dim round number (scaled)
      - 1-dim max_cards (scaled)
      - 1-dim deck_size (scaled; 24 default)
      - 1-dim current-player seat index (scaled)
      - 1-dim last bet action_id (scaled -1..87 -> 0..1)
      - 1-dim my hand size / 11
    Total dim = 24 + 1 + 1 + 1 + 8 + 6 = 41
    """
    rules = game.get("rules", {"deck_size": 24})
    deck_size = float(rules.get("deck_size", 24))

    # Private hand (24 ignoring jokers)
    hand = _current_hand(game, nick)
    hand_vec = np.zeros((HAND_VEC_DIM,), dtype=np.float32)
    for c in hand:
        v, col = int(c["value"]), int(c["colour"])
        if v >= 0:
            hand_vec[v * N_COLOURS + col] = 1.0

    # Joker & common stats
    my_jokers = float(_count_jokers(hand))
    common = game.get("common_hand", []) or []
    common_size = float(len(common))
    common_jokers = float(_count_jokers(common))

    # Public per-seat counts (padded to 8)
    counts = [int(p["n_cards"]) for p in game.get("players", [])]
    counts = (counts + [0] * (MAX_PLAYERS - len(counts)))[:MAX_PLAYERS]
    counts = np.array(counts, dtype=np.float32)

    # Meta scalars
    round_num = float(game.get("round_number", 1))
    max_cards = float(game.get("max_cards", 11))
    cp_idx = float(_nickname_to_idx(game["players"], game.get("cp_nickname", "")))
    last_bet = _last_bet_action_id(game)  # -1 if none
    meta = np.array([
        my_jokers / 4.0,                 # safe small scale
        common_size / 6.0,
        common_jokers / 4.0,
        round_num / 40.0,
        max_cards / 20.0,
        deck_size / 40.0,                # 24 -> 0.6
        cp_idx / max(1, len(game.get("players", [])) - 1),
        (last_bet + 1.0) / ACT_DIM,      # -1..88 -> 0..1
        float(len(hand)) / 11.0,
    ], dtype=np.float32)

    obs = np.concatenate([hand_vec, counts, meta], axis=0)
    return torch.from_numpy(obs)


# ---------- Environment Adapter ----------

class MyEnv:
    """
    Minimal TurnEnvAdapter for nfsp_complex.NFSPAgent:
      reset() -> (obs, mask, pid)
      step(action) -> (obs, mask, reward, done, pid)

    reward: ±1 only on CHECK resolution (from perspective of the actor who issued CHECK);
            0 otherwise.
    done: True when a ROUND ends (i.e., when the actor plays CHECK), or if game finishes.
    """

    def __init__(self, n_agents: int = 2, verbose: bool = False, illegal_penalty: float = -0.25):
        if n_agents < 2 or n_agents > 8:
            raise ValueError("n_agents must be in [2, 8]")
        self.n_agents = n_agents
        self.verbose = verbose
        self.illegal_penalty = float(illegal_penalty)
        self.game: Dict = {}
        self._last_obs = None
        self._last_mask = None

    def _pid(self) -> int:
        return _nickname_to_idx(self.game["players"], self.game.get("cp_nickname", ""))

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        self.game = gm.create_game(self.n_agents, verbose=self.verbose)
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

        # Try to act; catch and handle illegal attempts gracefully.
        try:
            gm.play(self.game, int(action), verbose=self.verbose)
        except Exception as e:
            # Return same obs/mask/pid with a penalty; do NOT advance player.
            obs = self._last_obs.clone()
            mask = self._last_mask.clone()
            pid = self._pid()
            return obs, mask, float(self.illegal_penalty), False, pid

        # New observation / mask / pid after the move (may be a new round if CHECK)
        cp = self.game.get("cp_nickname")
        obs = vectorize_obs(self.game, cp).float()
        mask = _legal_action_mask(self.game).float()
        pid = self._pid()

        # Determine terminal and reward
        done = False
        reward = 0.0

        if int(action) == CHECK:
            # Round has been resolved by the manager, and a new round likely started.
            # Identify loser by delta in n_cards (one player +1, possibly -> 0 on elimination).
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
            reward = 1.0 if (loser is not None and loser != actor_nick) else -1.0
            done = True

        # Game finished? Mark terminal regardless of action.
        if self.game.get("status") == "Finished" and status_before != "Finished":
            done = True

        self._last_obs, self._last_mask = obs, mask
        return obs, mask, float(reward), bool(done), pid


# ---------- Train (example) ----------
# RUN FROM REPOSITORY ROOT /
if __name__ == "__main__":
    env = MyEnv(n_agents=2, verbose=False)
    obs0, mask0, _ = env.reset()

agent = NFSPAgent(
    obs_dim=obs0.numel(),
    act_dim=mask0.numel(),
    cfg=NFSPConfig(
        # You can tune these; below are conservative to get learning started
        anticipatory_eta=0.1,
        rl_capacity=200_000,
        sl_capacity=200_000,
        batch_rl=256,   # Use batch_rl for RL batch size
        batch_sl=512,   # Use batch_sl for supervised learning batch size
        train_rl_every=4,
        train_sl_every=4,
        warmup_steps=5_000,
    ),
)

agent.train_from_selfplay(
    env,
    total_steps=2_000,  # 200,000   # start smaller to validate the loop
    log_every=5,      # 5,000
    save_path="nfsp_blef.pt",
)