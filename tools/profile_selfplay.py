#!/usr/bin/env python3
"""
Lightweight profiler for NFSP self-play runs.

Usage examples:
    PYTHONPATH=. python3 tools/profile_selfplay.py --steps 200000
    PYTHONPATH=. python3 tools/profile_selfplay.py --steps 500000 --profile-out profile.prof
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import random
import sys
import tempfile
import time
from contextlib import contextmanager

import numpy as np
import torch

from nfsp_ai.nfsp_run_local import MyEnv
from nfsp_ai.agent import NFSPAgent, NFSPConfig


@contextmanager
def maybe_profile(enabled: bool, profile_out: str | None):
    if not enabled:
        yield None
        return
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield profiler
    finally:
        profiler.disable()
        if profile_out:
            profiler.dump_stats(profile_out)


def build_agent_env(seed: int, max_cards: int) -> tuple[NFSPAgent, MyEnv]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = MyEnv(n_agents=2, max_cards=max_cards, verbose=False)
    obs0, mask0, _ = env.reset()
    cfg = NFSPConfig(
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
        rl_capacity=200_000,
        sl_capacity=200_000,
        n_step=5,
        burst_rl_updates_on_reward=2,
        burst_reward_threshold=0.5,
        eps_start=0.2,
        eps_end=0.05,
        eps_decay_steps=1_000_000,
    )
    agent = NFSPAgent(
        obs_dim=obs0.numel(),
        act_dim=mask0.numel(),
        cfg=cfg,
    )
    return agent, env


def run_profile(
    total_steps: int,
    log_dir: str,
    profile_enabled: bool,
    profile_out: str | None,
    sort_by: str,
    top_n: int,
    max_cards: int,
    seed: int,
):
    agent, env = build_agent_env(seed, max_cards)

    log_every = total_steps + 1
    os.makedirs(log_dir, exist_ok=True)
    game_save_dir = os.path.join(log_dir, "games_profile")
    os.makedirs(game_save_dir, exist_ok=True)

    start = time.perf_counter()
    with maybe_profile(profile_enabled, profile_out) as profiler:
        agent.train_from_selfplay(
            env,
            total_steps=total_steps,
            log_every=log_every,
            save_path=None,
            game_save_dir=game_save_dir,
            tb_logdir=None,
            csv_path=None,
            eval_every=0,
            history_sample_path=None,
            control_plane=None,
        )
    elapsed = time.perf_counter() - start
    steps_per_second = total_steps / max(elapsed, 1e-9)
    print(f"[profile] steps={total_steps} elapsed={elapsed:.2f}s throughput={steps_per_second:.2f} steps/s")

    if profile_enabled and not profile_out and profiler is not None:
        stats = pstats.Stats(profiler)
        stats.strip_dirs().sort_stats(sort_by).print_stats(top_n)


def main():
    parser = argparse.ArgumentParser(description="Profile NFSP self-play loop (CPU-friendly).")
    parser.add_argument("--steps", type=int, default=200_000, help="Environment steps to simulate (default: 200k).")
    parser.add_argument("--profile-out", type=str, default=None, help="Optional file to dump cProfile stats.")
    parser.add_argument("--sort", type=str, default="tottime", help="Sort key for pstats output (default: tottime).")
    parser.add_argument("--top", type=int, default=40, help="Rows to print from pstats when not dumping to file.")
    parser.add_argument("--max-cards", type=int, default=3, help="Initial max_cards in the env (default: 3).")
    parser.add_argument("--seed", type=int, default=12345, help="Seed for reproducibility.")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory for temporary artifacts.")
    args = parser.parse_args()

    total_steps = max(1, int(args.steps))
    top_n = max(1, int(args.top))
    profile_out = args.profile_out
    profile_enabled = True
    log_dir = args.log_dir or tempfile.mkdtemp(prefix="nfsp_profile_")

    try:
        run_profile(
            total_steps=total_steps,
            log_dir=log_dir,
            profile_enabled=profile_enabled,
            profile_out=profile_out,
            sort_by=args.sort,
            top_n=top_n,
            max_cards=max(1, int(args.max_cards)),
            seed=int(args.seed),
        )
    except KeyboardInterrupt:
        print("\n[profile] interrupted", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
