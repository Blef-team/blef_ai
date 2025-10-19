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
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Utils
# =========================
def masked_softmax_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set invalid action logits to -inf so softmax/argmax ignore them."""
    neg_inf = torch.finfo(logits.dtype).min
    return logits.masked_fill(mask == 0, neg_inf)

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
        self.obs  = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.mask = torch.zeros((capacity, act_dim), dtype=torch.float32, device=device)
        self.act  = torch.zeros((capacity,), dtype=torch.long, device=device)
        self.rew  = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.nobs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.nmsk = torch.zeros((capacity, act_dim), dtype=torch.float32, device=device)
        self.done = torch.zeros((capacity,), dtype=torch.float32, device=device)

    def add(self, obs, mask, act, rew, nobs, nmask, done):
        self.obs[self.ptr]  = obs
        self.mask[self.ptr] = mask
        self.act[self.ptr]  = act
        self.rew[self.ptr]  = rew
        self.nobs[self.ptr] = nobs
        self.nmsk[self.ptr] = nmask
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (self.obs[idx], self.mask[idx], self.act[idx], self.rew[idx],
                self.nobs[idx], self.nmsk[idx], self.done[idx])

class ReservoirSL:
    """True reservoir sampling (Vitter) for empirical average policy (one-hot actions)."""
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device):
        self.device, self.capacity = device, capacity
        self.size = 0
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
        self.obs[i], self.mask[i], self.ta[i] = obs, mask, one_hot_action

    def sample(self, batch_size: int):
        if self.size == 0:
            raise RuntimeError("SL reservoir is empty")
        n = min(self.size, self.capacity)
        idx = torch.randint(0, n, (batch_size,), device=self.device)
        return self.obs[idx], self.mask[idx], self.ta[idx]


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
    gamma: float = 0.995
    lr_q: float = 1e-3
    lr_pi: float = 5e-4
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
    max_grad_norm: float = 1.0
    hidden: int = 256
    use_double_dqn: bool = True

class NFSPAgent:
    def __init__(self, obs_dim: int, act_dim: int, device: Optional[torch.device] = None, cfg: NFSPConfig = NFSPConfig()):
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

    # -------- Acting (returns sampled action only) --------
    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor, use_br: bool, epsilon: float) -> int:
        self.q.eval(); self.pi.eval()
        obs = obs.unsqueeze(0).to(self.device)   # [1, D]
        mask = mask.unsqueeze(0).to(self.device) # [1, A]

        if use_br:
            q = self.q(obs)                      # [1, A]
            # epsilon-greedy over legal actions
            if random.random() < epsilon:
                return epsilon_valid_sample(mask)
            a = int(masked_random_argmax(q, mask)[0].item())
            return a
        else:
            logits = self.pi(obs)                # [1, A]
            mlog = masked_softmax_logits(logits, mask)
            dist = torch.distributions.Categorical(logits=mlog)
            return int(dist.sample().item())

    # -------- RL (Double DQN with masking & tie-breaks) --------
    def _train_rl_step(self):
        if self.rl_buf.size < self.cfg.batch_rl:
            return
        obs, mask, act, rew, nobs, nmask, done = self.rl_buf.sample(self.cfg.batch_rl)

        q = self.q(obs)                          # [B, A]
        q_sa = q.gather(1, act.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            q_next_main = self.q(nobs)           # [B, A]
            q_next_tgt  = self.q_tgt(nobs)       # [B, A]
            if self.cfg.use_double_dqn:
                next_a = masked_random_argmax(q_next_main, nmask).view(-1, 1)
                q_next = q_next_tgt.gather(1, next_a).squeeze(1)
            else:
                neg_inf = torch.finfo(q.dtype).min
                q_next = q_next_tgt.masked_fill(nmask == 0, neg_inf).max(dim=1).values

            # zero bootstrap at terminal or forced-end states (done=1)
            target = rew + (1.0 - done) * self.cfg.gamma * q_next

        loss = F.smooth_l1_loss(q_sa, target)    # Huber
        self.opt_q.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.opt_q.step()

        # Target updates
        if self.cfg.hard_target_interval and (self.total_env_steps % self.cfg.hard_target_interval == 0):
            self.q_tgt.load_state_dict(self.q.state_dict())
        else:
            with torch.no_grad():
                tau = self.cfg.target_tau
                if tau > 0:
                    for p, pt in zip(self.q.parameters(), self.q_tgt.parameters()):
                        pt.data.mul_(1 - tau).add_(tau * p.data)

    # -------- SL (policy imitation of empirical average) --------
    def _train_sl_step(self):
        try:
            obs, mask, one_hot = self.sl_buf.sample(self.cfg.batch_sl)
        except RuntimeError:
            return
        logits = self.pi(obs)                    # [B, A]
        mlog = masked_softmax_logits(logits, mask)
        log_probs = mlog - torch.logsumexp(mlog, dim=1, keepdim=True)
        loss = -(one_hot * log_probs).sum(dim=1).mean()  # CE on empirical action
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.pi.parameters(), self.cfg.max_grad_norm)
        self.opt_pi.step()

    # -------- Orchestration (online self-play) --------
    def _epsilon(self) -> float:
        s = self.total_env_steps
        frac = min(1.0, s / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_end + (1.0 - frac) * (self.cfg.eps_start - self.cfg.eps_end)

    def train_from_selfplay(self, env: TurnEnvAdapter, total_steps: int = 2_000_000, log_every: int = 10000, save_path: str = "nfsp_blef.pt", game_save_dir="games"):
        obs, mask, pid = env.reset()
        obs, mask = obs.to(self.device), mask.to(self.device)

        while self.total_env_steps < total_steps:
            use_br = (random.random() < self.cfg.anticipatory_eta)
            eps = self._epsilon()

            action = self.act(obs, mask, use_br=use_br, epsilon=eps)
            nobs, nmask, reward, done, _ = env.step(action, game_save_dir=game_save_dir)
            nobs, nmask = nobs.to(self.device), nmask.to(self.device)

            # RL buffer: BR transitions only
            if use_br:
                self.rl_buf.add(
                    obs, mask, torch.tensor(action, device=self.device),
                    torch.tensor(reward, dtype=torch.float32, device=self.device),
                    nobs, nmask, done
                )

            # SL reservoir: empirical one-hot from behavior (BR or pi)
            one_hot = torch.zeros(self.act_dim, dtype=torch.float32)
            one_hot[action] = 1.0
            self.sl_buf.add(obs.detach().cpu(), mask.detach().cpu(), one_hot)

            self.total_env_steps += 1

            # Online updates (after warmup)
            if self.total_env_steps > self.cfg.warmup_steps:
                if self.total_env_steps % self.cfg.train_rl_every == 0:
                    self._train_rl_step()
                if self.total_env_steps % self.cfg.train_sl_every == 0:
                    self._train_sl_step()

            # Episode handling
            if done:
                obs, mask, pid = env.reset()
                obs, mask = obs.to(self.device), mask.to(self.device)
            else:
                obs, mask = nobs, nmask

            # Logging / checkpoints
            if self.total_env_steps % log_every == 0:
                print(f"[steps={self.total_env_steps}] RL_buf={self.rl_buf.size} SL_buf≈{min(self.sl_buf.size, self.sl_buf.capacity)} eps={eps:.3f}")
            if save_path and (self.total_env_steps % (10 * log_every) == 0):
                self.save(save_path)

        if save_path:
            self.save(save_path)

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
            "steps": self.total_env_steps
        }, path)

    def load(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.q.load_state_dict(ckpt["q"])
        self.q_tgt.load_state_dict(ckpt["q_tgt"])
        self.pi.load_state_dict(ckpt["pi"])
        self.total_env_steps = ckpt.get("steps", 0)


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
