# NFSP Blef AI

This folder contains our Blef self-play agent built on Neural Fictitious Self-Play (NFSP), the
approach introduced by Heinrich & Silver (NeurIPS 2016). NFSP blends reinforcement learning with
running averages of best responses, and it’s our workhorse for training a policy that can bluff,
call, and adapt to different deck sizes in Blef.

## Fast start (5 minutes)
1. **Create a virtualenv** and install deps (from the repo root):
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **(Optional) Drop pretrained embeddings in `artifacts/`**:
   - `card_embedding_pretrain.pt`
   - `history_embedding_pretrain.pt`
3. **Launch a run**:
   ```bash
   PYTHONPATH=. python3 nfsp_ai/nfsp_run_local.py \
       --deck-size 24 \
       --use-card-embeddings auto \
       --use-history-embeddings auto \
       --total-steps 100000000 \
       --control-plane control_plane_run.json
  ```
  This prints progress, writes checkpoints under `nfsp_blef_<timestamp>.pt`, keeps logs in
  `logs/`, and stores sample evaluation games under `games_<timestamp>_eval/`.
   To warm-start from an existing checkpoint but follow schedules from step 0, use
   `--initialise-with path/to/checkpoint.pt` instead of `--resume`.

Stop here if you just want a working self-play job. The defaults (two agents, 24-card deck, no
jokers) are decent for smoke testing.

## Learn the knobs (30 minutes)
- **Embedding flags**: `--use-card-embeddings` / `--use-history-embeddings` fall back gracefully if
  the artifacts aren’t present. Pretrain them with the helpers in `tools/`:
  - `tools/pretrain_card_embeddings.py`
  - `tools/pretrain_history_embeddings.py`

- **Control plane**: edit `control_plane_run.json` mid-training to pin or adjust schedules (ε, η,
  learning rates, curriculum `max_cards`). Edits are gated by a cooldown (50k steps by default) to
  keep runs stable. The console and CSV log explicitly show what’s pinned.

- **Schedule extension**: `_apply_phase_schedules` in `agent.py` is already tuned for 0–100 M steps.
  If you need custom gating (e.g., freeze `max_cards` until CHECK metrics improve), look at
  `docs/training_schedule_plan.md` for TODOs.

- **Saved artifacts**: besides checkpoints, the runner keeps:
  - `games_<timestamp>/`: occasional training games.
  - `games_<timestamp>_eval/`: 10 stored games from every evaluation sweep.
  - `logs/action_samples_<timestamp>.jsonl`: snippets for quick debugging.

- **Evaluation**: by default every 50k steps the agent plays 200 deterministic matches against the
  conservative baseline. Tweaking `--eval-every` moves that cadence; the runner now produces a
  handful of eval games you can replay by hand.

- **Profiling**: want to know where compute goes? `python -m nfsp_ai.profile_selfplay --steps 500000`
  will drop a `profile.prof` you can open in SnakeViz or pstats.

## Where to tweak deeper
- **Network architecture** (`agent.py`): current policy/Q nets are simple MLPs. If you want to blend
  the embeddings directly into the model (instead of the observation pipeline), this is the place.
- **Curriculum / masks**: gating and validation code lives at the top of `train_from_selfplay`; the
  roadmap’s “Training Foundations” section outlines upcoming hardening tasks.
- **Game manager**: `shared/api/simpleschema_local_manager.py` mirrors production logic; if you’re
  hunting legality bugs, it’s your friend.


Most of the mechanics follow Heinrich & Silver (2016), with additions for Blef-specific quirks—
variable deck sizes, live control-plane tweaks, and the embedding system we keep expanding. If you
notice the agent folding where it should bluff (or vice versa), check the evaluation games—with ten
samples per sweep it’s much easier to see what’s happening.
