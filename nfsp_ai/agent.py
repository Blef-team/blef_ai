# nfsp_blef.py
# NFSP for imperfect-information, turn-based games (e.g., Blef abstraction)
# - Anticipatory dynamics: act with BR (Q) w.p. eta, else average policy (pi)
# - RL buffer: BR transitions only (Double DQN, masked)
# - SL reservoir: empirical one-hot action from behavior policy (both BR and pi)
# - Action masking everywhere; random tie-break on argmax
# - Online updates during self-play (no separate offline batch phase)

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Any, TYPE_CHECKING

from collections import deque
import csv, os, math, time, json
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

import torch
import torch.nn as nn
import torch.nn.functional as F

# Baseline strategy for evaluation
from conservative_ai.agent import ConservativeAgent

if TYPE_CHECKING:
    from nfsp_ai.control_plane import JsonControlPlane

# Logging, metrics

def _masked_entropy(logits: torch.Tensor, mask: torch.Tensor) -> float:
    # logits: [B, A], mask: [B, A]
    # returns scalar mean entropy over batch after masked softmax
    mlog = masked_softmax_logits(logits, mask)
    probs = torch.softmax(mlog, dim=-1)
    ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    return float(ent.mean().item())

@torch.no_grad()
@torch.no_grad()
def _evaluate_policy(
    agent,
    env_source,
    max_cards: int = 11,
    episodes: int = 200,
    *,
    save_dir: Optional[str] = None,
    save_games: int = 0,
) -> dict:
    wins = losses = total_r = total_len = 0
    managed_env = callable(env_source)

    for _ in range(episodes):
        env = env_source() if managed_env else env_source
        env.max_cards = max_cards

        if save_dir and save_games > 0 and hasattr(env, "game_save_dir"):
            env.game_save_dir = save_dir
            env.save_sample_rate = max(1, episodes // max(1, save_games))
            env._save_counter = env.save_sample_rate - 1
            if hasattr(env, "_saved_games"):
                env._saved_games = 0

        obs, mask, pid = env.reset()
        done, steps = False, 0

        while not done:
            cp = env.game["cp_nickname"]
            if cp == env._ref_nick:
                a = agent.select_action(obs, mask, use_average_policy=True, greedy=True)
            else:
                # baseline acts
                a = ConservativeAgent.determine_action(env.game)
                # fallback for illegal/None
                if a is None or mask[int(a)] == 0:
                    legal = mask.nonzero(as_tuple=False).view(-1).tolist()
                    a = random.choice(legal) if legal else 0

            obs, mask, r, done, info = env.step(int(a))

            loser = None
            if info and info.get("round_result", {}) and info.get("round_result", {}).get("loser"): #Please don't "simplify", you'll break it
                loser = info.get("round_result", {}).get("loser")
                if loser == env._ref_nick:
                    losses += 1
                    total_r += -1
                else:
                    wins += 1
                    total_r += 1
            steps += 1

        total_len += steps

        if managed_env:
            del env

    return {
        "avg_reward": total_r / episodes,
        "avg_len": total_len / episodes,
        "win_rate": wins / (wins + losses),
    }


class _CsvLogger:
    def __init__(self, path: str, header: list[str]):
        self.path = path
        self._exists = os.path.exists(path)
        self.header = header
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not self._exists:
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.header)
    def row(self, values: list):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(values)

def plot_csv_progress(csv_path="./logs/nfsp_blef.csv"):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots()
    ax.plot(df["step"], df["avg_reward"], label="avg_reward")
    if "q_loss" in df.columns:
        ax.plot(df["step"], df["q_loss"], label="Q loss")
    if "sl_loss" in df.columns:
        ax.plot(df["step"], df["sl_loss"], label="SL loss")
    if "policy_entropy" in df.columns:
        ax.plot(df["step"], df["policy_entropy"], label="policy entropy")
    ax.set_xlabel("env steps")
    ax.set_title("NFSP Self-Play Progress")
    ax.legend()
    plt.show()


# =========================
# Utils
# =========================
def masked_softmax_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set invalid action logits to -inf so softmax/argmax ignore them."""
    neg_inf = torch.finfo(logits.dtype).min
    return logits.masked_fill(mask == 0, neg_inf)
    neg_inf = float("-inf")
    # Support float or bool masks robustly
    if mask.dtype != torch.bool:
        illegal = (mask <= 0)
    else:
        illegal = ~mask

    return logits.masked_fill(illegal, neg_inf)

def masked_random_argmax(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Argmax over legal actions with uniform random tie-break.
    q, mask: [B, A]
    returns: [B] long
    """
    neg_inf = torch.finfo(q.dtype).min
    q_masked = q.masked_fill(mask == 0, neg_inf)          # [B, A]
    maxv = q_masked.max(dim=1, keepdim=True).values       # [B, 1]
    ties = (q_masked == maxv) & (mask > 0)                # [B, A] bool
    # sample uniformly among ties per batch row
    actions = []
    for row in range(q.shape[0]):
        idxs = torch.nonzero(ties[row], as_tuple=False).squeeze(1)
        # guard: if mask was empty (shouldn't happen), pick argmax of raw q
        if idxs.numel() == 0:
            actions.append(q[row].argmax().item())
        else:
            j = torch.randint(0, idxs.numel(), (1,)).item()
            actions.append(int(idxs[j].item()))
    return torch.tensor(actions, dtype=torch.long, device=q.device)

def epsilon_valid_sample(mask: torch.Tensor) -> int:
    """Uniform sample among valid actions given [1, A] mask."""
    valid_idx = torch.nonzero(mask[0] > 0.0, as_tuple=False).squeeze(1)
    j = torch.randint(0, valid_idx.numel(), (1,)).item()
    return int(valid_idx[j].item())


# =========================
# Models (separate nets)
# =========================
class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
    def forward(self, x):  # returns unmasked Q-values [B, A]
        return self.net(x)

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, act_dim)
    def forward(self, obs):  # returns logits [B, A] (unmasked)
        return self.pi(self.enc(obs))


# =========================
# Buffers
# =========================
class ReplayBuffer:
    """For BR transitions only (DQN)."""
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device):
        self.device, self.capacity = device, capacity
        self.ptr, self.size = 0, 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs  = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.mask = torch.zeros((capacity, act_dim), dtype=torch.float32, device=device)
        self.act  = torch.zeros((capacity,), dtype=torch.long, device=device)
        self.rew  = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.nobs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.nmsk = torch.zeros((capacity, act_dim), dtype=torch.float32, device=device)
        self.done = torch.zeros((capacity,), dtype=torch.float32, device=device)
        # NEW: store per-sample gamma_power (γ^k for n-step)
        self.gpow  = torch.ones((capacity,), dtype=torch.float32, device=device)

    def add(self, obs, mask, act, rew, nobs, nmask, done, gamma_power):
        self.obs[self.ptr]  = obs
        self.mask[self.ptr] = mask
        self.act[self.ptr]  = act
        self.rew[self.ptr]  = rew
        self.nobs[self.ptr] = nobs
        self.nmsk[self.ptr] = nmask
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.gpow = gamma_power

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (self.obs[idx], self.mask[idx], self.act[idx], self.rew[idx],
                self.nobs[idx], self.nmsk[idx], self.done[idx], self.gpow)

    def state_dict(self) -> dict:
        filled = int(self.size)
        return {
            "capacity": self.capacity,
            "ptr": int(self.ptr),
            "size": filled,
            "obs": self.obs[:filled].cpu(),
            "mask": self.mask[:filled].cpu(),
            "act": self.act[:filled].cpu(),
            "rew": self.rew[:filled].cpu(),
            "nobs": self.nobs[:filled].cpu(),
            "nmsk": self.nmsk[:filled].cpu(),
            "done": self.done[:filled].cpu(),
        }

    def load_state_dict(self, state: dict):
        self.ptr = int(state.get("ptr", 0))
        self.size = min(int(state.get("size", 0)), self.capacity)
        filled = self.size
        for name, tensor in (
            ("obs", self.obs),
            ("mask", self.mask),
            ("act", self.act),
            ("rew", self.rew),
            ("nobs", self.nobs),
            ("nmsk", self.nmsk),
            ("done", self.done),
        ):
            src = state.get(name)
            if src is None or filled == 0:
                tensor.zero_()
            else:
                tensor.zero_()
                tensor[:filled] = src.to(self.device)

    def resize(self, new_capacity: int):
        if new_capacity <= self.capacity:
            return
        def _expand(tensor, new_shape):
            new_t = torch.zeros(new_shape, dtype=tensor.dtype, device=self.device)
            new_t[:self.size] = tensor[:self.size]
            return new_t
        self.obs = _expand(self.obs, (new_capacity, self.obs_dim))
        self.mask = _expand(self.mask, (new_capacity, self.act_dim))
        self.act = _expand(self.act, (new_capacity,))
        self.rew = _expand(self.rew, (new_capacity,))
        self.nobs = _expand(self.nobs, (new_capacity, self.obs_dim))
        self.nmsk = _expand(self.nmsk, (new_capacity, self.act_dim))
        self.done = _expand(self.done, (new_capacity,))
        self.capacity = new_capacity
        self.ptr = self.size % self.capacity

class ReservoirSL:
    """True reservoir sampling (Vitter) for empirical average policy (one-hot actions)."""
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device):
        self.device, self.capacity = device, capacity
        self.size = 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs  = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.mask = torch.zeros((capacity, act_dim), dtype=torch.float32, device=device)
        self.ta   = torch.zeros((capacity, act_dim), dtype=torch.float32, device=device)  # one-hot action

    def add(self, obs, mask, one_hot_action):
        self.size += 1
        if self.size <= self.capacity:
            i = self.size - 1
        else:
            j = random.randint(1, self.size)
            if j > self.capacity:
                return
            i = j - 1
        self.obs[i] = obs.to(self.device)
        self.mask[i] = mask.to(self.device)
        self.ta[i] = one_hot_action.to(self.device)

    def sample(self, batch_size: int):
        if self.size == 0:
            raise RuntimeError("SL reservoir is empty")
        n = min(self.size, self.capacity)
        idx = torch.randint(0, n, (batch_size,), device=self.device)
        return self.obs[idx], self.mask[idx], self.ta[idx]

    def state_dict(self) -> dict:
        filled = min(self.size, self.capacity)
        return {
            "capacity": self.capacity,
            "seen": int(self.size),
            "filled": filled,
            "obs": self.obs[:filled].cpu(),
            "mask": self.mask[:filled].cpu(),
            "ta": self.ta[:filled].cpu(),
        }

    def load_state_dict(self, state: dict):
        self.size = int(state.get("seen", 0))
        filled = min(int(state.get("filled", 0)), self.capacity)
        for name, tensor in (("obs", self.obs), ("mask", self.mask), ("ta", self.ta)):
            tensor.zero_()
            src = state.get(name)
            if src is not None and filled > 0:
                tensor[:filled] = src.to(self.device)

    def resize(self, new_capacity: int):
        if new_capacity <= self.capacity:
            return
        filled = min(self.capacity, self.size, new_capacity)
        def _expand(tensor, dim):
            new_t = torch.zeros((new_capacity, dim), dtype=tensor.dtype, device=self.device)
            new_t[:filled] = tensor[:filled]
            return new_t
        self.obs = _expand(self.obs, self.obs_dim)
        self.mask = _expand(self.mask, self.act_dim)
        self.ta = _expand(self.ta, self.act_dim)
        self.capacity = new_capacity


# =========================
# Env Adapter (plug your engine)
# =========================
class TurnEnvAdapter:
    """
    Implement for your engine:
      - reset() -> (obs, mask, current_player_id)
      - step(action) -> (obs, mask, reward, done, next_player_id)

    obs:  torch.float32 [obs_dim]
    mask: torch.float32 [act_dim] (1=valid, 0=invalid). Ensure at least one valid action.
    reward: float from perspective of the player who JUST acted (terminal ±1; 0 otherwise).
    done: bool. Make 'forced-end' states terminal (zero bootstrap).
    """
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        raise NotImplementedError
    def step(self, action: int) -> Tuple[torch.Tensor, torch.Tensor, float, bool, int]:
        raise NotImplementedError


# =========================
# NFSP Agent
# =========================
@dataclass
class NFSPConfig:
    gamma: float = 0.666
    lr_q: float = 5e-5                 # q net learning rate
    lr_pi: float = 3e-4                # pi net learning rate
    batch_rl: int = 1024
    batch_sl: int = 2048
    target_tau: float = 0.005          # Polyak; set to 0 for hard updates
    hard_target_interval: int = 0      # if >0, do hard copy every N steps (overrides Polyak on that step)
    eps_start: float = 0.10
    eps_end: float = 0.05
    eps_decay_steps: int = 2_000_000
    anticipatory_eta: float = 0.10     # probability to act with BR
    rl_capacity: int = 1_000_000
    sl_capacity: int = 1_000_000
    train_rl_every: int = 1            # Q update cadence
    train_sl_every: int = 10           # pi update cadence
    warmup_steps: int = 10_000
    max_grad_norm: float = 10.0        # For nn.utils.clip_grad_norm_
    hidden: int = 256
    use_double_dqn: bool = True
    n_step: int = 1
    burst_rl_updates_on_reward: int = 2
    burst_reward_threshold: float = 0.5

class NFSPAgent:
    def __init__(self, obs_dim: int, act_dim: int, device: Optional[torch.device] = None, cfg: NFSPConfig = NFSPConfig(), debug=False):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim, self.act_dim, self.cfg = obs_dim, act_dim, cfg

        # Separate networks
        self.q = QNet(obs_dim, act_dim, cfg.hidden).to(self.device)
        self.q_tgt = QNet(obs_dim, act_dim, cfg.hidden).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=cfg.lr_q)

        self.pi = PolicyNet(obs_dim, act_dim, cfg.hidden).to(self.device)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr_pi)

        self.rl_buf = ReplayBuffer(cfg.rl_capacity, obs_dim, act_dim, self.device)
        self.sl_buf = ReservoirSL(cfg.sl_capacity, obs_dim, act_dim, self.device)

        self.total_env_steps = 0
        self._pinned_overrides: dict[str, Any] = {}
        self._pinned_env_overrides: dict[str, Any] = {}

        # near __init__
        self.stats = getattr(self, "stats", {})
        self.stats.setdefault("sl_skipped_exploration", 0)

        self._took_epsilon_action = False
        self._took_forced_check = False

        self.debug = debug


    # -------- Acting (returns sampled action only) --------
    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor, use_br: bool, epsilon: float) -> int:
        self.q.eval(); self.pi.eval()
        obs = obs.unsqueeze(0).to(self.device)   # [1, D]
        mask = mask.unsqueeze(0).to(self.device) # [1, A]

        # Reset flags each decision
        self._took_forced_check = False
        self._took_epsilon_action = False

        # Manual override: here we force it to pick CHECK (for early exposure)
        check_prob = getattr(self, "_check_explore_prob", 0.0)
        if check_prob > 0.0:
            check_idx = mask.shape[-1] - 1
            if mask[0, check_idx] > 0 and random.random() < check_prob:
                self._took_forced_check = True
                return int(check_idx)

        if use_br:
            q = self.q(obs)                      # [1, A]
            # epsilon-greedy over legal actions
            if random.random() < epsilon:
                self._took_epsilon_action = True
                return epsilon_valid_sample(mask)
            a = int(masked_random_argmax(q, mask)[0].item())
            return a
        else:
            logits = self.pi(obs)                # [1, A]
            mlog = masked_softmax_logits(logits, mask)
            dist = torch.distributions.Categorical(logits=mlog)
            return int(dist.sample().item())


    # -------- RL --------
    def _train_rl_step(self):
        """
        Optimized RL update for round-based (fully terminal) environments.

        - Each transition in rl_buf corresponds to a complete round outcome.
        - Rewards already include all temporal shaping (from _flush_nstep).
        - There are no non-terminal steps to bootstrap from (done=True for all).
        """

        # --- 1. Skip if buffer too small ---
        if self.rl_buf.size < self.cfg.batch_rl:
            return

        # --- 2. Sample batch ---
        obs, mask, act, rew, _, _, _, _ = self.rl_buf.sample(self.cfg.batch_rl)

        act = act.view(-1, 1).long()
        rew = rew.float().view(-1)

        # --- 3. Compute predicted Q(s,a) ---
        q = self.q(obs)
        q_sa = q.gather(1, act).squeeze(1)

        # --- 4. Terminal-only target = reward (no bootstrap) ---
        target = rew

        # --- 5. Compute Huber loss and apply update ---
        loss = F.smooth_l1_loss(q_sa, target)

        self.opt_q.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.opt_q.step()

        # --- 6. Update target network ---
        self.rl_updates = getattr(self, "rl_updates", 0) + 1
        if self.cfg.hard_target_interval and (self.rl_updates % self.cfg.hard_target_interval == 0):
            self.q_tgt.load_state_dict(self.q.state_dict())
        else:
            tau = self.cfg.target_tau
            if tau and tau > 0:
                with torch.no_grad():
                    for p, pt in zip(self.q.parameters(), self.q_tgt.parameters()):
                        pt.data.mul_(1 - tau).add_(tau * p.data)

        # --- 7. Return loss metric ---
        return {"q_loss": float(loss.detach().item())}


    # -------- SL (policy imitation of empirical average) --------
    def _train_sl_step(self):
        try:
            obs, mask, one_hot = self.sl_buf.sample(self.cfg.batch_sl)
        except RuntimeError:
            print(RuntimeError in _train_sl_step)
            return

        logits = self.pi(obs)  # [B, A]

        # Build legality (tolerate float masks)
        legal = mask if mask.dtype == torch.bool else (mask > 0)          # [B, A]
        has_legal = legal.any(dim=-1)                                     # [B]

        # Filter: logits must be finite on legal actions
        legal_logits_finite = (torch.isfinite(logits) | (~legal)).all(dim=-1)  # [B]

        # Filter: target must have zero mass on illegal actions AND some mass on legal actions
        illegal_mass = (one_hot * (~legal).to(one_hot.dtype)).sum(dim=1)       # [B]
        no_illegal_mass = illegal_mass == 0
        legal_mass = (one_hot * legal.to(one_hot.dtype)).sum(dim=1)            # [B]
        has_legal_mass = legal_mass > 0

        valid = has_legal & legal_logits_finite & no_illegal_mass & has_legal_mass
        if not valid.any():
            return  # skip step cleanly

        # Slice to valid rows (no in-place on graph tensors)
        logits  = logits[valid]
        legal   = legal[valid]
        one_hot = one_hot[valid]

        # Mask illegals to a large negative FINITE value (safer than -inf for grads)
        NEG_LARGE = -1e9
        masked_logits = logits.masked_fill(~legal, NEG_LARGE)  # out-of-place

        # Stable log-softmax over legal actions
        denom = torch.logsumexp(masked_logits, dim=1, keepdim=True)       # [B,1]
        log_probs = masked_logits - denom                                  # [B,A]

        # Renormalize the empirical target over LEGAL actions only
        tgt = one_hot * legal.to(one_hot.dtype)                            # zero-out illegals
        tgt_sum = tgt.sum(dim=1, keepdim=True)                             # > 0 by construction
        tgt = tgt / tgt_sum

        # Cross-entropy
        loss = -(tgt * log_probs).sum(dim=1).mean()

        # Final finite check (no in-place sanitization)
        if not torch.isfinite(loss):
            return  # skip this step rather than poisoning grads

        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pi.parameters(), self.cfg.max_grad_norm)
        self.opt_pi.step()

        return {"sl_loss": float(loss.detach().item())}

    # -------- Orchestration (online self-play) --------
    def _epsilon(self) -> float:
        if hasattr(self, "_eps_current"):
            return float(self._eps_current)
        s = self.total_env_steps
        frac = min(1.0, s / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_end + (1.0 - frac) * (self.cfg.eps_start - self.cfg.eps_end)

    def _ensure_override_state(self):
        if not hasattr(self, "_pinned_overrides"):
            self._pinned_overrides = {}
        if not hasattr(self, "_pinned_env_overrides"):
            self._pinned_env_overrides = {}

    def _apply_pin(self, key: str, value: Any, pin: bool):
        self._ensure_override_state()
        if not pin:
            self._pinned_overrides.pop(key, None)
            return
        self._pinned_overrides[key] = value

    def _apply_env_pin(self, key: str, value: Any, pin: bool):
        self._ensure_override_state()
        if value is None or not pin:
            self._pinned_env_overrides.pop(key, None)
            return
        self._pinned_env_overrides[key] = value

    def _is_pinned(self, key: str) -> bool:
        self._ensure_override_state()
        return key in self._pinned_overrides

    def _get_env_override(self, key: str, default: Any = None) -> Any:
        self._ensure_override_state()
        return self._pinned_env_overrides.get(key, default)

    def set_eta(self, value: Optional[float], *, pin: bool = True) -> Optional[float]:
        key = "anticipatory_eta"
        if value is None:
            self._pinned_overrides.pop(key, None)
            return None
        val = float(max(0.0, min(1.0, value)))
        self.cfg.anticipatory_eta = val
        self._apply_pin(key, val, pin)
        return val

    def set_epsilon(self, value: Optional[float], *, pin: bool = True) -> Optional[float]:
        key = "epsilon"
        if value is None:
            self._pinned_overrides.pop(key, None)
            return None
        val = float(max(0.0, min(1.0, value)))
        self._eps_current = val
        self._apply_pin(key, val, pin)
        return val

    def set_lr_q(self, value: Optional[float], *, pin: bool = True) -> Optional[float]:
        key = "lr_q"
        if value is None:
            self._pinned_overrides.pop(key, None)
            return None
        val = float(max(1e-8, min(1e-2, value)))
        self._set_lr(self.opt_q, val)
        self.cfg.lr_q = val
        self._apply_pin(key, val, pin)
        return val

    def set_lr_pi(self, value: Optional[float], *, pin: bool = True) -> Optional[float]:
        key = "lr_pi"
        if value is None:
            self._pinned_overrides.pop(key, None)
            return None
        val = float(max(1e-8, min(1e-2, value)))
        self._set_lr(self.opt_pi, val)
        self.cfg.lr_pi = val
        self._apply_pin(key, val, pin)
        return val

    def set_train_rl_every(self, value: Optional[int], *, pin: bool = True) -> Optional[int]:
        key = "train_rl_every"
        if value is None:
            self._pinned_overrides.pop(key, None)
            return None
        val = int(max(1, min(512, value)))
        self.cfg.train_rl_every = val
        self._apply_pin(key, val, pin)
        return val

    def set_batch_rl(self, value: Optional[int], *, pin: bool = True) -> Optional[int]:
        key = "batch_rl"
        if value is None:
            self._pinned_overrides.pop(key, None)
            return None
        val = int(max(16, min(4096, value)))
        if val % 2 != 0:
            val += 1
        self.cfg.batch_rl = val
        self._apply_pin(key, val, pin)
        return val

    def set_train_sl_every(self, value: Optional[int], *, pin: bool = True) -> Optional[int]:
        key = "train_sl_every"
        if value is None:
            self._pinned_overrides.pop(key, None)
            return None
        val = int(max(1, min(512, value)))
        self.cfg.train_sl_every = val
        self._apply_pin(key, val, pin)
        return val

    def set_batch_sl(self, value: Optional[int], *, pin: bool = True) -> Optional[int]:
        key = "batch_sl"
        if value is None:
            self._pinned_overrides.pop(key, None)
            return None
        val = int(max(32, min(8192, value)))
        if val % 32 != 0:
            val = ((val // 32) + 1) * 32
        self.cfg.batch_sl = val
        self._apply_pin(key, val, pin)
        return val

    def set_target_tau(self, value: Optional[float], *, pin: bool = True) -> Optional[float]:
        key = "tau"
        if value is None:
            self._pinned_overrides.pop(key, None)
            return None
        val = float(max(0.0, min(0.5, value)))
        self.cfg.target_tau = val
        self._apply_pin(key, val, pin)
        return val

    def set_hard_target_interval(self, value: Optional[int], *, pin: bool = True) -> Optional[int]:
        key = "hard_target_interval"
        if value is None:
            self._pinned_overrides.pop(key, None)
            return None
        val = int(max(0, min(100_000, value)))
        self.cfg.hard_target_interval = val
        self._apply_pin(key, val, pin)
        return val

    def set_n_step(self, value: Optional[int], *, pin: bool = True) -> Optional[int]:
        key = "n_step"
        if value is None:
            self._pinned_overrides.pop(key, None)
            return None
        val = int(max(1, min(32, value)))
        if val != getattr(self, "_active_n_step", val):
            self._flush_nstep(force=True)
            self._active_n_step = val
        self.cfg.n_step = val
        self._apply_pin(key, val, pin)
        return val

    def set_check_explore_prob(self, value: Optional[float], *, pin: bool = True) -> Optional[float]:
        key = "check_prob"
        if value is None:
            self._pinned_overrides.pop(key, None)
            return None
        val = float(max(0.0, min(1.0, value)))
        self._check_explore_prob = val
        self._apply_pin(key, val, pin)
        return val

    def set_burst_rl_updates(self, value: Optional[int], *, pin: bool = True) -> Optional[int]:
        key = "burst_rl_updates_on_reward"
        if value is None:
            self._pinned_overrides.pop(key, None)
            return None
        val = int(max(0, min(16, value)))
        self.cfg.burst_rl_updates_on_reward = val
        self._apply_pin(key, val, pin)
        return val

    def set_burst_reward_threshold(self, value: Optional[float], *, pin: bool = True) -> Optional[float]:
        key = "burst_reward_threshold"
        if value is None:
            self._pinned_overrides.pop(key, None)
            return None
        val = float(max(-1.0, min(1.0, value)))
        self.cfg.burst_reward_threshold = val
        self._apply_pin(key, val, pin)
        return val

    def set_max_cards(self, value: Optional[int], *, env: Optional["TurnEnvAdapter"] = None, pin: bool = True) -> Optional[int]:
        key = "max_cards"
        if value is None:
            self._apply_env_pin(key, None, pin=False)
            return None
        val = int(max(1, min(11, value)))
        self._apply_env_pin(key, val, pin)
        if env is not None:
            self._sync_env_max_cards(env, val)
        return val

    def _ensure_schedule_state(self):
        self._ensure_override_state()
        if not hasattr(self, "_schedule_flags"):
            self._schedule_flags: set[str] = set()
        if not hasattr(self, "_nstep_queue"):
            self._nstep_queue: deque = deque()
        if not hasattr(self, "_eps_current"):
            self._eps_current = float(self.cfg.eps_end)
        if self._is_pinned("epsilon"):
            self._eps_current = float(self._pinned_overrides["epsilon"])
        if not hasattr(self, "_active_n_step"):
            self._active_n_step = max(1, int(self.cfg.n_step))
        if self._is_pinned("n_step"):
            pinned_n = int(self._pinned_overrides["n_step"])
            if pinned_n != self._active_n_step:
                self._flush_nstep(force=True)
                self._active_n_step = pinned_n
            self.cfg.n_step = pinned_n
        if not hasattr(self, "_last_checkpoint_million"):
            self._last_checkpoint_million = self.total_env_steps // 1_000_000
        if self.rl_buf.capacity >= 1_000_000:
            self._schedule_flags.add("buffers_1m")
        if not hasattr(self, "_nstep_weight_scale"):
            self._nstep_weight_scale = 0.5
        if not hasattr(self, "_nstep_terminal_boost"):
            self._nstep_terminal_boost = 0.5
        if not hasattr(self, "_check_explore_prob"):
            self._check_explore_prob = 0.0
        if self._is_pinned("check_prob"):
            self._check_explore_prob = float(self._pinned_overrides["check_prob"])
        if not hasattr(self, "_control_plane_events"):
            self._control_plane_events: list[dict[str, Any]] = []
        if not hasattr(self, "_last_control_plane_override_flag"):
            self._last_control_plane_override_flag = 0
        if not hasattr(self, "_last_control_plane_override_summary"):
            self._last_control_plane_override_summary = ""

    def _set_lr(self, optimizer, lr: float):
        lr = float(lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

    def _sync_env_max_cards(self, env: Optional["TurnEnvAdapter"], target: int):
        if env is None:
            return
        if getattr(env, "max_cards", None) != target:
            env.max_cards = target
        if hasattr(env, "game") and env.game:
            env.game["max_cards"] = target
            rules = env.game.get("rules")
            if isinstance(rules, dict):
                rules["max_cards"] = target

    def _interp(self, step: int, start: int, end: int, v_start: float, v_end: float) -> float:
        if step <= start:
            return v_start
        if step >= end:
            return v_end
        frac = (step - start) / max(1, end - start)
        return v_start + frac * (v_end - v_start)

    def _apply_phase_schedules(self, step: int, env: Optional["TurnEnvAdapter"] = None):
        self._ensure_schedule_state()

        # ---------------------------
        # Looping schedules, if reset point set
        # ---------------------------
        if env and env.reset_schedules_at > env.reset_schedules_to and self.total_env_steps >= env.reset_schedules_at:
            self.total_env_steps = env.reset_schedules_to

        # ---------------------------
        # Helper: set n_step safely
        # ---------------------------
        def set_nstep(n):
            n = int(max(1, min(5, n)))  # hard cap at 5 given round length
            if not self._is_pinned("n_step") and n != self._active_n_step:
                self._flush_nstep(force=True)
                self._active_n_step = n
            if not self._is_pinned("n_step"):
                self.cfg.n_step = n

        # ---------------------------
        # Early curriculum: 0–5M
        # Goal: learn to CHECK, stabilize Q/SL targets, avoid SL pollution
        # ---------------------------
        if step < 5_000_000:
            # Anticipatory eta (prob of acting as Q/best-response)
            if not self._is_pinned("anticipatory_eta"):
                # Start high to collect BR data, taper a bit to calm drift
                if step < 2_000_000:
                    eta = 0.25
                else:
                    eta = self._interp(step, 2_000_000, 5_000_000, 0.25, 0.15)
                self.cfg.anticipatory_eta = max(0.10, eta)

            # Epsilon: keep healthy exploration; smooth decay
            if not self._is_pinned("epsilon"):
                if step < 1_500_000:
                    eps = self._interp(step, 0, 1_500_000, 0.20, 0.12)
                elif step < 3_500_000:
                    eps = self._interp(step, 1_500_000, 3_500_000, 0.12, 0.07)
                else:
                    eps = self._interp(step, 3_500_000, 5_000_000, 0.07, 0.05)
                self._eps_current = max(0.03, min(1.0, eps))  # floor 0.03

            # Q optimizer / targets
            if not self._is_pinned("lr_q"):
                self._set_lr(self.opt_q, 1e-4); self.cfg.lr_q = 1e-4
            if not self._is_pinned("tau"):
                self.cfg.target_tau = 0.01           # keep Polyak updates
            if not self._is_pinned("hard_target_interval"):
                self.cfg.hard_target_interval = 0     # disable hard updates

            # RL cadence
            if not self._is_pinned("train_rl_every"):
                self.cfg.train_rl_every = 32
            if not self._is_pinned("batch_rl"):
                self.cfg.batch_rl = 64
            if step >= 10_000_000 and not self._is_pinned("batch_rl"):
                self.cfg.batch_rl = 256

            # n-step: align to round length (fixed)
            set_nstep(3)

            # SL optimizer / cadence
            if not self._is_pinned("lr_pi"):
                self.cfg.lr_pi = 1e-4; self._set_lr(self.opt_pi, self.cfg.lr_pi)
            if not self._is_pinned("train_sl_every"):
                self.cfg.train_sl_every = 12   # a bit more frequent than your 16
            if not self._is_pinned("batch_sl"):
                self.cfg.batch_sl = 256

            # Forced check curriculum (SMOOTH; NEVER cliff to 0)
            if not self._is_pinned("check_prob"):
                # 0 → 5M: 0.25 → 0.05
                self._check_explore_prob = self._interp(step, 0, 5_000_000, 0.25, 0.05)
            return

        # ---------------------------
        # Mid curriculum: 5–20M
        # Goal: phase out forced checks smoothly; steady BR/avg learning
        # ---------------------------
        if not self._is_pinned("check_prob"):
            # 5M → 10M: 0.05 → 0; 10M+: 0
            if step < 10_000_000:
                self._check_explore_prob = self._interp(step, 5_000_000, 10_000_000, 0.05, 0.0)
            else:
                self._check_explore_prob = 0.0

        if step >= 5_000_000 and "buffers_1m" not in self._schedule_flags:
            self.rl_buf.resize(1_000_000)
            self.sl_buf.resize(1_000_000)
            self._schedule_flags.add("buffers_1m")

        # η: taper earlier to reduce teacher drift; stabilize SL
        if not self._is_pinned("anticipatory_eta"):
            if step < 10_000_000:
                eta = 0.15
            elif step < 20_000_000:
                eta = self._interp(step, 10_000_000, 20_000_000, 0.15, 0.10)
            else:
                eta = self._interp(step, 20_000_000, 40_000_000, 0.10, 0.08)
            self.cfg.anticipatory_eta = max(0.06, eta)

        # ε: gentle decay; never below 0.0075 overall
        if not self._is_pinned("epsilon"):
            if step < 15_000_000:
                eps = self._interp(step, 5_000_000, 15_000_000, 0.05, 0.02)
            elif step < 50_000_000:
                eps = self._interp(step, 15_000_000, 50_000_000, 0.02, 0.01)
            else:
                eps = self._interp(step, 50_000_000, 100_000_000, 0.01, 0.0075)
            self._eps_current = max(0.0, min(1.0, eps))

        # lr_q: smooth, conservative taper
        if not self._is_pinned("lr_q"):
            if step < 40_000_000:
                lr_q = self._interp(step, 5_000_000, 40_000_000, 1e-4, 3e-5)
            elif step < 70_000_000:
                lr_q = self._interp(step, 40_000_000, 70_000_000, 3e-5, 2e-5)
            else:
                lr_q = self._interp(step, 70_000_000, 100_000_000, 2e-5, 1.5e-5)
            self._set_lr(self.opt_q, lr_q); self.cfg.lr_q = lr_q

        # RL cadence: modest increases only
        if not self._is_pinned("train_rl_every"):
            if step < 20_000_000:
                self.cfg.train_rl_every = 32
            elif step < 50_000_000:
                self.cfg.train_rl_every = 48
            else:
                self.cfg.train_rl_every = 64

        # n-step: keep fixed (don’t increase with step)
        set_nstep(3)

        # SL: a bit higher LR early, taper later
        if not self._is_pinned("lr_pi"):
            if step < 40_000_000:
                lr_pi = 3e-4
            elif step < 70_000_000:
                lr_pi = 2e-4
            else:
                lr_pi = 1.5e-4
            self.cfg.lr_pi = lr_pi; self._set_lr(self.opt_pi, self.cfg.lr_pi)

        if not self._is_pinned("train_sl_every"):
            if step < 60_000_000:
                self.cfg.train_sl_every = 8
            else:
                self.cfg.train_sl_every = 12

        if not self._is_pinned("batch_sl"):
            self.cfg.batch_sl = max(256, self.cfg.batch_sl)


    def _store_transition(
        self,
        obs,
        mask,
        action_idx: int,
        reward: float,
        nobs,
        nmask,
        done: bool,
        is_br: bool,
        *,
        actor=None,          # <-- seat index or nickname of the actor for this step
        loser=None           # <-- loser id ONLY on terminal step (None otherwise)
    ):
        if not hasattr(self, "_nstep_queue"):
            raise ValueError("_nstep_queue not in self!")
            self._nstep_queue = collections.deque()  # or list, your choice

        if self.debug:
            #DEBUG
            print("_store_transition DEBUG:")
            print(f"action_idx: {action_idx}")
            print(f"reward: {reward}")
            print(f"done: {done}")
            print(f"actor: {actor}")
            print(f"loser: {loser}")
            print("_store_transition DEBUG ------ END")
            #DEUBG


        action_tensor = torch.tensor(action_idx, dtype=torch.long, device=self.device)
        reward_val = float(reward)

        # --- 1-step fast path (TD(0)) ---
        if self._active_n_step <= 1:
            if is_br:
                # For TD(0), write exactly what the env emitted.
                # gamma_power: keep your current convention; often 1.0 if done else gamma.
                gamma_power = 1.0 if done else float(self.cfg.gamma)
                self.rl_buf.add(
                    obs.clone(),
                    mask.clone(),
                    action_tensor,
                    torch.tensor(reward_val, dtype=torch.float32, device=self.device),
                    nobs.clone(),
                    nmask.clone(),
                    bool(done),
                    torch.tensor(gamma_power, dtype=torch.float32, device=self.device),
                )
            return

        # --- n-step path: queue raw transition; _flush_nstep will compute shaped R for non-terminals
        entry = {
            "obs":   obs.clone(),
            "mask":  mask.clone(),
            "action": action_tensor,
            "reward": reward_val,            # terminal step already carries true env reward; earlier steps are 0
            "nobs":  nobs.clone(),
            "nmask": nmask.clone(),
            "done":  bool(done),
            "is_br": bool(is_br),
            "actor": actor,                  # REQUIRED for all steps (2–8 players)
            "loser": loser if done else None # ONLY present on terminal step
        }
        self._nstep_queue.append(entry)

        # Flush only when the last appended transition is terminal (or when forced elsewhere)
        self._flush_nstep(force=False)


    def _flush_nstep(self, force: bool = False):
        """
        Custom n-step flush for round-based multi-player games (2–8 players).

        - Each round ends with exactly one terminal ('check') step that determines the loser.
        - Non-terminal actions get shaped rewards propagated back from the terminal result:
              if actor == loser:  R = -gamma**distance
              else:               R = +gamma**distance
          where distance = 0 for penultimate (γ⁰ = 1), 1 for one before (γ¹), etc.
        - The terminal step itself already has its true environment reward and is written as-is.
        """

        # --- Safety and trivial cases ---
        if not hasattr(self, "_nstep_queue") or not self._nstep_queue:
            return

        n = max(1, self._active_n_step)
        if n <= 1:
            self._nstep_queue.clear()
            return

        # Only flush when the round actually ended (last transition done=True)
        if not force and not self._nstep_queue[-1]["done"]:
            return

        seq = list(self._nstep_queue)
        self._nstep_queue.clear()

        # Identify terminal and loser
        term = seq[-1]
        assert term["done"], "Expected last transition in round to be terminal."
        loser_id = term.get("loser", None)
        total_steps = len(seq)
        gamma = float(self.cfg.gamma)

        if self.debug:
            #DEBUG
            print(f"term['action']: {term['action']}")
            print(f"term['reward']: {term['reward']}")
            #DEUBG

        # --- 1) Store the terminal step exactly as emitted by env ---
        if term.get("is_br", False):
            self.rl_buf.add(
                term["obs"],
                term["mask"],
                term["action"],
                torch.tensor(term["reward"], dtype=torch.float32, device=self.device),
                term["nobs"],
                term["nmask"],
                True,
                torch.tensor(1.0, dtype=torch.float32, device=self.device),
            )

        # --- 2) Back-propagate credit/blame to earlier steps ---
        for i, step in enumerate(seq[:-1]):  # skip terminal
            # distance to terminal: penultimate=0, earlier=1,2,...
            distance = max(0, (total_steps - 2) - i)
            g = gamma ** distance

            actor_id = step.get("actor", None)
            if actor_id is None or loser_id is None:
                raise ValueError(f"actor and or loser info missing! actor: {str(actor)}, loser: {str(loser)}")
                R = 0.0
            else:
                sign = -1.0 if actor_id == loser_id else +1.0
                R = sign * g

            if self.debug:
                #DEBUG
                print(f"step['action']: {step['action']}")
                print(f"R: {R}")
                print(f"step.get('is_br'): {step.get('is_br')}")
                #DEUBG

            if not step.get("is_br", False):
                continue

            done_flag = True  # every round ends at terminal
            gamma_power = gamma ** max(1, distance + 1)  # not really used, but kept for consistency

            # Keep your weighting / oversampling logic
            weight = 1.0 + max(0.0, getattr(self, "_nstep_weight_scale", 0.0)) * max(0, n - (distance + 1))
            weight += max(0.0, getattr(self, "_nstep_terminal_boost", 0.0)) * max(0, n - (distance + 1))
            repeats = max(1, int(weight))
            residual = max(0.0, weight - repeats)

            if self.debug:
                #DEBUG
                print(f"step['action']: {step['action']}")
                print(f"R: {R}")
                #DEUBG

            for _ in range(repeats):
                self.rl_buf.add(
                    step["obs"],
                    step["mask"],
                    step["action"],
                    torch.tensor(R, dtype=torch.float32, device=self.device),
                    term["nobs"],
                    term["nmask"],
                    done_flag,
                    torch.tensor(gamma_power, dtype=torch.float32, device=self.device),
                )
            if residual > 0.0 and random.random() < residual:
                self.rl_buf.add(
                    step["obs"],
                    step["mask"],
                    step["action"],
                    torch.tensor(R, dtype=torch.float32, device=self.device),
                    term["nobs"],
                    term["nmask"],
                    done_flag,
                    torch.tensor(gamma_power, dtype=torch.float32, device=self.device),
                )

        # --- 3) Cleanup ---
        if force:
            self._nstep_queue.clear()

    def train_from_selfplay(
        self,
        env: "TurnEnvAdapter",
        total_steps: int = 2_000_000,
        log_every: int = 10_000,
        save_path: str = "nfsp_blef.pt",
        game_save_dir: str = "games",
        # NEW:
        tb_logdir: str | None = "./runs/nfsp_blef",
        csv_path: str | None = "./logs/nfsp_blef.csv",
        eval_every: int = 50_000,
        eval_episodes: int = 200,
        eval_env_factory: Optional[Callable[[], "TurnEnvAdapter"]] = None,
        eval_save_dir: Optional[str] = None,
        eval_save_games: int = 0,
        apply_phase_schedules_every: int = 1_000,
        loop_schedules_to: int = 100_000,
        reset_schedules_at: int = 0,
        save_checkpoint_every: int = 50_000,
        control_plane: Optional["JsonControlPlane"] = None,
        history_sample_path: Optional[str] = "./logs/action_history_samples.jsonl",
        history_sample_every: int = 100_000,
        history_sample_limit: Optional[int] = 1_000,
    ):
        # --- setup ---
        obs, mask, pid = env.reset()
        obs, mask = obs.to(self.device), mask.to(self.device)

        # environment curriculum for max_cards
        def _target_max_cards(step: int) -> int:
            if step < 1_000_000:
                return 1
            if step < 3_000_000:
                return 2
            if step < 5_000_000:
                return 3
            extra = (step - 5_000_000) // 3_000_000
            return int(max(1, min(11, 3 + extra)))
        def _effective_max_cards(step: int) -> int:
            override = self._get_env_override("max_cards")
            if override is not None:
                return int(override)
            return _target_max_cards(step)

        self._ensure_schedule_state()
        self._nstep_queue.clear()
        self._active_n_step = max(1, int(self.cfg.n_step))
        self._last_checkpoint_million = max(
            self._last_checkpoint_million, self.total_env_steps // 1_000_000
        )
        initial_max_cards = _effective_max_cards(self.total_env_steps)
        self._sync_env_max_cards(env, initial_max_cards)

        # track update counters across runs
        self.rl_updates = getattr(self, "rl_updates", 0)
        self.sl_updates = getattr(self, "sl_updates", 0)

        # rolling gameplay stats
        recent_rewards = deque(maxlen=5_000)
        recent_winloss_results = deque(maxlen=5_000)
        recent_lens    = deque(maxlen=5_000)
        recent_illegal  = deque(maxlen=20_000)  # if env reports illegal flags in info

        # rolling training stats (store last N non-None values)
        recent_q_loss  = deque(maxlen=5_000)
        recent_sl_loss = deque(maxlen=5_000)
        recent_pi_ent  = deque(maxlen=5_000)

        # TensorBoard
        writer = None
        if tb_logdir and SummaryWriter is not None:
            os.makedirs(tb_logdir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_logdir)

        # CSV
        csv_header = [
            "step","avg_reward","win_rate","avg_len",
            "q_loss","sl_loss","policy_entropy",
            "illegal_rate","epsilon","anticipatory_eta","lr_q","n_step","train_rl_every","check_explore_prob","max_cards",
            "rl_buf","sl_buf","control_override","control_changes","pinned_overrides","pinned_env"
        ]
        csv_logger = _CsvLogger(csv_path, csv_header) if csv_path else None

        # episode accumulators
        ep_reward, ep_len = 0.0, 0

        history_sample_next = None
        history_samples_written = 0
        if history_sample_path:
            os.makedirs(os.path.dirname(history_sample_path) or ".", exist_ok=True)
            history_sample_every = max(1, int(history_sample_every))
            history_sample_next = history_sample_every
            history_sample_limit = (
                None if history_sample_limit is None else max(0, int(history_sample_limit))
            )
        else:
            history_sample_limit = None

        # main loop
        while self.total_env_steps < total_steps:
            if self.total_env_steps % apply_phase_schedules_every == 0:
                self._apply_phase_schedules(self.total_env_steps, env=env)
            eps = self._epsilon()
            if control_plane is not None:
                self._ensure_override_state()
                pinned_override_keys = sorted(self._pinned_overrides.keys())
                pinned_env_keys = sorted(self._pinned_env_overrides.keys())
                metrics_snapshot = {
                    "step": self.total_env_steps,
                    "eta": float(self.cfg.anticipatory_eta),
                    "epsilon": float(eps),
                    "lr_q": float(self.opt_q.param_groups[0]["lr"]),
                    "lr_pi": float(self.opt_pi.param_groups[0]["lr"]),
                    "train_rl_every": int(self.cfg.train_rl_every),
                    "batch_rl": int(self.cfg.batch_rl),
                    "train_sl_every": int(self.cfg.train_sl_every),
                    "batch_sl": int(self.cfg.batch_sl),
                    "tau": float(self.cfg.target_tau),
                    "hard_target_interval": int(self.cfg.hard_target_interval),
                    "n_step": int(self._active_n_step),
                    "max_cards": int(getattr(env, "max_cards", _effective_max_cards(self.total_env_steps))),
                    "burst_rl_updates_on_reward": int(self.cfg.burst_rl_updates_on_reward),
                    "burst_reward_threshold": float(self.cfg.burst_reward_threshold),
                    "check_prob": float(getattr(self, "_check_explore_prob", 0.0)),
                    "pinned_overrides": pinned_override_keys,
                    "pinned_env": pinned_env_keys,
                    "rho_est": float(
                        self.cfg.batch_rl
                        / max(1.0, self.cfg.train_rl_every)
                        / max(self.cfg.anticipatory_eta, 1e-9)
                    ),
                }
                try:
                    control_plane.tick(self, env, metrics_snapshot)
                except Exception as exc:
                    print(f"[control] failed to apply overrides: {exc}")
                eps = self._epsilon()
            desired_max_cards = _effective_max_cards(self.total_env_steps)
            if getattr(env, "max_cards", None) != desired_max_cards:
                self._sync_env_max_cards(env, desired_max_cards)
            use_br = (random.random() < self.cfg.anticipatory_eta)

            action = self.act(obs, mask, use_br=use_br, epsilon=eps)
            nobs, nmask, reward, done, info = env.step(action)
            nobs, nmask = nobs.to(self.device), nmask.to(self.device)

            info_dict = info if isinstance(info, dict) else {}
            if info_dict:
                illegal = int(info_dict.get("illegal", 0))
                recent_illegal.append(illegal)
            else:
                illegal = 0

            # Extract actor every step; loser only on terminal
            actor = info_dict.get("actor", None) if info_dict else None
            loser = None
            rr = info_dict.get("round_result")
            if rr is not None:
                loser = rr.get("loser", None)

            # Pass actor/loser so _flush_nstep can distribute rewards
            self._store_transition(
                obs, mask, action, reward, nobs, nmask, done, is_br=use_br,
                actor=actor, loser=loser
            )

            # Ensure n-step buffer flushes at terminals (round end)
            if done and self._active_n_step > 1:
                self._flush_nstep(force=True)

            # SL reservoir: empirical one-hot from behavior (prefer BR-only), but skip exploration/forced/illegal
            one_hot = torch.zeros(self.act_dim, dtype=torch.float32)
            one_hot[action] = 1.0

            # Flags set by self.act(...) in this step
            took_forced_check   = self._took_forced_check
            took_epsilon_action = self._took_epsilon_action

            # Skip if forced check, epsilon action, or illegal
            skip_sl_log = took_forced_check or took_epsilon_action or (illegal == 1)

            # Safe guard in case stats wasn't initialized somewhere else
            if not hasattr(self, "stats"):
                self.stats = {}
            self.stats.setdefault("sl_skipped_exploration", 0)

            if skip_sl_log:
                self.stats["sl_skipped_exploration"] += 1
            else:
                # NFSP: SL should clone average-policy (not BR) behavior
                if not use_br:
                    self.sl_buf.add(obs.detach().cpu(), mask.detach().cpu(), one_hot)


            if history_sample_path and history_sample_next is not None and info_dict:
                history_payload = info_dict.get("history")
                if history_payload:
                    if self.total_env_steps >= history_sample_next:
                        if history_sample_limit is None or history_samples_written < history_sample_limit:
                            self._ensure_override_state()
                            record = {
                                "step": self.total_env_steps,
                                "history": history_payload,
                                "round_result": info_dict.get("round_result"),
                                "reward": float(reward),
                                "eta": float(self.cfg.anticipatory_eta),
                                "epsilon": float(eps),
                                "check_prob": float(getattr(self, "_check_explore_prob", 0.0)),
                                "max_cards": int(getattr(env, "max_cards", desired_max_cards)),
                                "pins": {
                                    "overrides": sorted(self._pinned_overrides.keys()),
                                    "env": sorted(self._pinned_env_overrides.keys()),
                                },
                            }
                            with open(history_sample_path, "a", encoding="utf-8") as hf:
                                hf.write(json.dumps(record, sort_keys=True) + "\n")
                            history_samples_written += 1
                            history_sample_next += history_sample_every
                            if (
                                history_sample_limit is not None
                                and history_samples_written >= history_sample_limit
                            ):
                                history_sample_next = None
                        else:
                            history_sample_next = None

            # gameplay accumulators
            ep_reward += float(reward)
            ep_len    += 1

            # Increment step counter
            self.total_env_steps += 1

            # per-million checkpointing
            if save_path and self.total_env_steps % save_checkpoint_every == 0:
                current_million = self.total_env_steps // 1_000_000
                if current_million > self._last_checkpoint_million:
                    base, ext = os.path.splitext(save_path)
                    if not ext:
                        ext = ".pt"
                    million_path = f"{base}_{current_million}M{ext}"
                    self.save(million_path)
                    self._last_checkpoint_million = current_million

            # Online updates (after warmup)
            if self.total_env_steps > self.cfg.warmup_steps:
                if self.total_env_steps % self.cfg.train_rl_every == 0:
                    # Expect dict or scalar; handle None gracefully.
                    out = self._train_rl_step()
                    performed_rl_update = out is not None
                    if isinstance(out, dict):
                        q_loss = out.get("q_loss", None)
                    else:
                        q_loss = float(out) if out is not None else None
                    if q_loss is not None and math.isfinite(q_loss):
                        recent_q_loss.append(q_loss)
                    if performed_rl_update:
                        self.rl_updates += 1

                if self.total_env_steps % self.cfg.train_sl_every == 0:
                    out = self._train_sl_step()
                    performed_sl_update = out is not None
                    if isinstance(out, dict):
                        sl_loss = out.get("sl_loss", None)
                    else:
                        sl_loss = float(out) if out is not None else None
                    if sl_loss is not None and math.isfinite(sl_loss):
                        recent_sl_loss.append(sl_loss)

                    # Track policy entropy from current π on the current state (cheap proxy)
                    with torch.no_grad():
                        logits = self.pi(obs.unsqueeze(0).to(self.device))
                        ent = _masked_entropy(logits, mask.unsqueeze(0).to(self.device))
                        if math.isfinite(ent):
                            recent_pi_ent.append(ent)
                    if performed_sl_update:
                        self.sl_updates += 1

            # Episode handling
            if done:
                self._flush_nstep(force=True)
                if loser:
                    recent_winloss_results.append(loser!=env._ref_nick)
                if loser:
                    recent_rewards.append(1 if loser!=env._ref_nick else -1)
                recent_lens.append(ep_len)
                ep_reward, ep_len = 0.0, 0
                obs, mask, pid = env.reset()
                obs, mask = obs.to(self.device), mask.to(self.device)
            else:
                obs, mask = nobs, nmask

            # --- Logging / checkpoints ---
            if self.total_env_steps % log_every == 0:
                avg_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
                avg_len    = float(np.mean(recent_lens)) if recent_lens else 0.0
                win_rate   = float(np.mean(recent_winloss_results)) if recent_winloss_results else float("nan")
                ql         = float(np.mean(recent_q_loss)) if recent_q_loss else float("nan")
                sll        = float(np.mean(recent_sl_loss)) if recent_sl_loss else float("nan")
                pent       = float(np.mean(recent_pi_ent)) if recent_pi_ent else float("nan")
                illegal_rt = float(np.mean(recent_illegal)) if recent_illegal else 0.0
                episodes_logged = len(recent_rewards)
                eta_val = float(self.cfg.anticipatory_eta)
                lr_q_val = float(self.opt_q.param_groups[0]["lr"])
                n_step_active = int(self._active_n_step)
                rl_every = int(self.cfg.train_rl_every)
                check_prob = float(getattr(self, "_check_explore_prob", 0.0))
                current_max_cards = int(getattr(env, "max_cards", desired_max_cards))
                self._ensure_override_state()
                pinned_override_keys = sorted(self._pinned_overrides.keys())
                pinned_env_keys = sorted(self._pinned_env_overrides.keys())
                pin_summary = ",".join(pinned_override_keys) if pinned_override_keys else "-"
                env_pin_summary = ",".join(pinned_env_keys) if pinned_env_keys else "-"
                control_events = getattr(self, "_control_plane_events", [])
                control_override_flag = 1 if control_events else 0
                control_changes = []
                if control_events:
                    for event in control_events:
                        step_mark = event.get("step", self.total_env_steps)
                        for msg in event.get("changes", []):
                            control_changes.append(f"{step_mark}:{msg}")
                control_change_summary = ";".join(control_changes)

                print(
                    f"[steps={self.total_env_steps}] len={avg_len:.1f} "
                    f"avgR={avg_reward:.4f} win={win_rate:.3f} "
                    f"Qloss={ql:.5f} SLloss={sll:.5f} H(pi)={pent:.3f} "
                    f"illegal={illegal_rt:.3f} eps={eps:.3f} "
                    f"RL_buf={self.rl_buf.size} SL_buf≈{min(self.sl_buf.size, self.sl_buf.capacity)} "
                    f"RL_upd={self.rl_updates} SL_upd={self.sl_updates} "
                    f"eta={eta_val:.3f} lr_q={lr_q_val:.2e} n_step={n_step_active} rl_every={rl_every} "
                    f"chk_p={check_prob:.3f} max_cards={current_max_cards} "
                    f"pins={pin_summary} env_pins={env_pin_summary} "
                    f"ctrl={control_override_flag} "
                    f"episodes_tracked={episodes_logged}"
                )

                if writer:
                    writer.add_scalar("Game/avg_reward", avg_reward, self.total_env_steps)
                    writer.add_scalar("Game/win_rate",  win_rate,  self.total_env_steps)
                    writer.add_scalar("Game/avg_len",   avg_len,   self.total_env_steps)
                    writer.add_scalar("Loss/Q",         ql,        self.total_env_steps)
                    writer.add_scalar("Loss/SL",        sll,       self.total_env_steps)
                    writer.add_scalar("Policy/entropy", pent,      self.total_env_steps)
                    writer.add_scalar("Env/illegal_rate", illegal_rt, self.total_env_steps)
                    writer.add_scalar("Exploration/epsilon", eps,   self.total_env_steps)
                    writer.add_scalar("Exploration/check_prob", check_prob, self.total_env_steps)
                    writer.add_scalar("Exploration/eta", eta_val, self.total_env_steps)
                    writer.add_scalar("Optimization/lr_q", lr_q_val, self.total_env_steps)
                    writer.add_scalar("Optimization/n_step", n_step_active, self.total_env_steps)
                    writer.add_scalar("Optimization/train_rl_every", rl_every, self.total_env_steps)
                    writer.add_scalar("Env/max_cards", current_max_cards, self.total_env_steps)
                    writer.add_scalar("Buffers/RL_size", self.rl_buf.size, self.total_env_steps)
                    writer.add_scalar("Buffers/SL_size", min(self.sl_buf.size, self.sl_buf.capacity), self.total_env_steps)
                    writer.add_scalar("Control/pinned_override_count", len(pinned_override_keys), self.total_env_steps)
                    writer.add_scalar("Control/pinned_env_count", len(pinned_env_keys), self.total_env_steps)
                    writer.add_scalar("Control/override_applied", control_override_flag, self.total_env_steps)

                if csv_logger:
                    csv_logger.row([
                        self.total_env_steps, avg_reward, win_rate, avg_len,
                        ql, sll, pent, illegal_rt, eps,
                        eta_val, lr_q_val, n_step_active, rl_every, check_prob, current_max_cards,
                        self.rl_buf.size, min(self.sl_buf.size, self.sl_buf.capacity),
                        control_override_flag, control_change_summary,
                        "|".join(pinned_override_keys) if pinned_override_keys else "",
                        "|".join(pinned_env_keys) if pinned_env_keys else "",
                    ])
                self._last_control_plane_override_flag = control_override_flag
                self._last_control_plane_override_summary = control_change_summary
                self._control_plane_events = []

            if save_path and (self.total_env_steps % (10 * log_every) == 0):
                self.save(save_path)

            # Periodic evaluation (no exploration, greedy avg policy)
            if eval_every and (self.total_env_steps % eval_every == 0):
                eval_source = eval_env_factory if eval_env_factory is not None else env
                eval_stats = _evaluate_policy(
                    self,
                    eval_source,
                    max_cards=env.max_cards,
                    episodes=eval_episodes,
                    save_dir=eval_save_dir,
                    save_games=eval_save_games,
                )
                print(
                    "EVALUATION:\n"
                    f"[steps={self.total_env_steps}] len={eval_stats["avg_len"]:.3f} "
                    f"avgR={eval_stats['avg_reward']:.4f} win={eval_stats['win_rate']:.3f} "
                    f"illegal={illegal_rt:.3f}"
                )
                if writer:
                    writer.add_scalar("Eval/avg_reward", eval_stats["avg_reward"], self.total_env_steps)
                    writer.add_scalar("Eval/win_rate",   eval_stats["win_rate"],  self.total_env_steps)
                    writer.add_scalar("Eval/avg_len",    eval_stats["avg_len"],   self.total_env_steps)
                if csv_logger:
                    self._ensure_override_state()
                    pinned_override_keys = sorted(self._pinned_overrides.keys())
                    pinned_env_keys = sorted(self._pinned_env_overrides.keys())
                    # Write a “synthetic” row carrying eval stats (loss fields left as NaN)
                    csv_logger.row([
                        self.total_env_steps,
                        eval_stats["avg_reward"],
                        eval_stats["win_rate"],
                        eval_stats["avg_len"],
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        eps,
                        float(self.cfg.anticipatory_eta),
                        float(self.opt_q.param_groups[0]["lr"]),
                        int(self._active_n_step),
                        int(self.cfg.train_rl_every),
                        float(getattr(self, "_check_explore_prob", 0.0)),
                        int(getattr(env, "max_cards", desired_max_cards)),
                        self.rl_buf.size,
                        min(self.sl_buf.size, self.sl_buf.capacity),
                        int(getattr(self, "_last_control_plane_override_flag", 0)),
                        getattr(self, "_last_control_plane_override_summary", ""),
                        "|".join(pinned_override_keys) if pinned_override_keys else "",
                        "|".join(pinned_env_keys) if pinned_env_keys else "",
                    ])
                if eval_env_factory is None:
                    # Evaluation mutated the training env; reset so gameplay resumes cleanly
                    ep_reward, ep_len = 0.0, 0
                    obs, mask, pid = env.reset()
                    obs, mask = obs.to(self.device), mask.to(self.device)

        self._flush_nstep(force=True)
        if save_path:
            self.save(save_path)
        if writer:
            writer.flush(); writer.close()

    # -------- Inference (usually average policy) --------
    @torch.no_grad()
    def select_action(self, obs_vec: torch.Tensor, action_mask: torch.Tensor, use_average_policy: bool = True, greedy: bool = False) -> int:
        if use_average_policy:
            logits = self.pi(obs_vec.unsqueeze(0).to(self.device))
            mask = action_mask.unsqueeze(0).to(self.device)
            mlog = masked_softmax_logits(logits, mask)
            if greedy:
                return int(mlog.argmax(dim=-1).item())
            dist = torch.distributions.Categorical(logits=mlog)
            return int(dist.sample().item())
        else:
            q = self.q(obs_vec.unsqueeze(0).to(self.device))
            mask = action_mask.unsqueeze(0).to(self.device)
            return int(masked_random_argmax(q, mask)[0].item())

    # -------- Persistence --------
    def save(self, path: str):
        torch.save({
            "q": self.q.state_dict(),
            "q_tgt": self.q_tgt.state_dict(),
            "pi": self.pi.state_dict(),
            "cfg": self.cfg.__dict__,
            "steps": self.total_env_steps,
            "opt_q": self.opt_q.state_dict(),
            "opt_pi": self.opt_pi.state_dict(),
            "rl_buf": self.rl_buf.state_dict(),
            "sl_buf": self.sl_buf.state_dict(),
            "rl_updates": getattr(self, "rl_updates", 0),
            "sl_updates": getattr(self, "sl_updates", 0),
            "pinned_overrides": dict(getattr(self, "_pinned_overrides", {})),
            "pinned_env_overrides": dict(getattr(self, "_pinned_env_overrides", {})),
            "eps_current": float(getattr(self, "_eps_current", self.cfg.eps_end)),
            "check_explore_prob": float(getattr(self, "_check_explore_prob", 0.0)),
        }, path)

    def load(self, path: str, map_location=None, *, reset_schedules: bool = False):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.q.load_state_dict(ckpt["q"])
        self.q_tgt.load_state_dict(ckpt["q_tgt"])
        self.pi.load_state_dict(ckpt["pi"])
        self.total_env_steps = 0 if reset_schedules else ckpt.get("steps", 0)
        if reset_schedules:
            self.opt_q = torch.optim.Adam(self.q.parameters(), lr=self.cfg.lr_q)
            self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=self.cfg.lr_pi)
            self.rl_buf = ReplayBuffer(self.cfg.rl_capacity, self.obs_dim, self.act_dim, self.device)
            self.sl_buf = ReservoirSL(self.cfg.sl_capacity, self.obs_dim, self.act_dim, self.device)
            self.rl_updates = 0
            self.sl_updates = 0
            self._pinned_overrides = {}
            self._pinned_env_overrides = {}
            self._nstep_queue = deque()
            self._eps_current = None
            self._check_explore_prob = 0.0
        else:
            # Resize buffers etc. to what the checkpoint was at
            self._apply_phase_schedules(self.total_env_steps)
            if "opt_q" in ckpt:
                self.opt_q.load_state_dict(ckpt["opt_q"])
            if "opt_pi" in ckpt:
                self.opt_pi.load_state_dict(ckpt["opt_pi"])
            if "rl_buf" in ckpt:
                self.rl_buf.load_state_dict(ckpt["rl_buf"])
            if "sl_buf" in ckpt:
                self.sl_buf.load_state_dict(ckpt["sl_buf"])
            self.rl_updates = ckpt.get("rl_updates", getattr(self, "rl_updates", 0))
            self.sl_updates = ckpt.get("sl_updates", getattr(self, "sl_updates", 0))
            self._pinned_overrides = dict(ckpt.get("pinned_overrides", getattr(self, "_pinned_overrides", {})))
            self._pinned_env_overrides = dict(ckpt.get("pinned_env_overrides", getattr(self, "_pinned_env_overrides", {})))
            if "eps_current" in ckpt and ckpt["eps_current"] is not None:
                self._eps_current = float(ckpt["eps_current"])
            if "check_explore_prob" in ckpt and ckpt["check_explore_prob"] is not None:
                self._check_explore_prob = float(ckpt["check_explore_prob"])
        self._ensure_schedule_state()


# =========================
# Wiring example (pseudo)
# =========================
# class MyEnv(TurnEnvAdapter):
#     def __init__(self, engine): self.engine = engine
#     def reset(self):
#         s = self.engine.reset()
#         obs  = torch.from_numpy(self.engine.vectorize(s)).float()
#         mask = torch.from_numpy(self.engine.action_mask(s)).float()
#         pid  = self.engine.current_player_id()
#         return obs, mask, pid
#     def step(self, action: int):
#         s, r, done = self.engine.step(action)
#         obs  = torch.from_numpy(self.engine.vectorize(s)).float()
#         mask = torch.from_numpy(self.engine.action_mask(s)).float()
#         pid  = self.engine.current_player_id()
#         return obs, mask, float(r), bool(done), pid
#
# # Train:
# env = MyEnv(engine)
# obs0, mask0, _ = env.reset()
# agent = NFSPAgent(obs_dim=obs0.numel(), act_dim=mask0.numel(), cfg=NFSPConfig())
# agent.train_from_selfplay(env, total_steps=2_000_000, log_every=20000, save_path="nfsp_blef.pt")
#
# # Inference (average policy pi):
# action = agent.select_action(obs_vec, action_mask, use_average_policy=True, greedy=False)
