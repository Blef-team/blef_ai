# NFSP Blef Bot — Phase 4+ — empirical updates (2026-05-08)

This document extends `docs/phase_plan.md` with phases derived from
overnight empirical work. The original plan stops at Phase 3; this
adds 4 onwards based on what we learned by training and ablating.

## What we now know that the original plan didn't

The original plan §111 hypothesized buffer-distribution starvation as
the cause of the late-game generalization failure. Tonight's diagnostics
(`tools/diagnose_buffer.py` on `mc-baseline-10M @ 9M`) confirmed it:

```
RL buffer (1M sampled): 1c 27.79%, 2c 33.22%, 3c 27.75%, 4c 11.24%, >4c 0.00%
```

Zero entries above 4 cards despite `--max-cards 11`. **Root cause is the
hardcoded `_target_max_cards` curriculum in `agent.py:1287` (1→11 ramp
over 14M training steps; only reaches 8 by 10M training steps).**

We also tried two interventions and measured:

- **RIH (random initial hand sizes from [1, max_cards])**: aggressive
  buffer fix. At 1M steps, B = `rih+pin no-emb` reaches mean 0.538 vs
  Conservative across 8 cells (vs baseline 0.498). Late-game lift
  (mc=11: 0.295→0.500). Cost: -0.215 at mc=1.
- **Pin only (control-plane override of `max_cards` from curriculum to
  11) + embeddings**: gentler buffer fix. At 1M steps, D = `pin+emb` at
  mean 0.606. Late-game lift (mc=11: 0.295→0.525) without losing early
  game (mc=1: 0.535→0.615). At 1.5M, mean rises to 0.621.
- **Embeddings alone (curriculum still active)**: emb-5M-final mean
  = 0.556. Embeddings help (+0.058 over baseline) but don't fix
  late-game (mc=11: 0.375→0.390).

**The strongest finding**: pin + embeddings (no RIH) cleanly improves on
baseline at every cell. RIH's distribution shift trades early-game for
late-game — expensive.

## Phase 4 — Ablations and stronger interventions (1–2 days)

### 4.1 Triangular RIH

Uniform [1, max_cards] gives only ~9% one-card games — the agent
rarely sees mc=1. Plan §216 warned about this. Replace uniform with a
distribution biased toward low-card games (e.g. `1/k` weighting,
geometric with p=0.5, or triangular descending). Goal: keep RIH's
late-game data while preserving early-game.

Tunable via a new flag `--rih-distribution {uniform,triangular,1/k}`.

### 4.2 Pin η and ε alongside max_cards

The control-plane pin only sets `max_cards=11`. The η/ε/n_step
schedules continue to anneal on a step-based clock. By mid-training
(say 5M), ε is at floor (~0.05) — the agent under-explores the
late-game states it's now finally seeing.

Add: pin η at e.g. 0.20 and ε at e.g. 0.10 for the full duration of a
curriculum-bypassing run. Use a longer `cp_aggressive.json` template.

### 4.3 Curriculum redesign

Replace the hardcoded 1→11 ramp with a configurable schedule. Specifically:

- `--max-cards-curriculum=fixed:N` — pin to N from step 0 (current behavior with control-plane override).
- `--max-cards-curriculum=ramp:start,end,duration` — explicit ramp.
- `--max-cards-curriculum=uniform-sample:[1,N]` — sample uniformly per game (the equivalent of permanent pin + RIH).

Make this a first-class CLI arg, not buried in agent.py.

### 4.4 Snapshot-league self-play

Plan §147 — defer no longer. Given that pin+emb is empirically working,
the next limitation is that NFSP's symmetric self-play has no
opponent diversity. With shared weights, the agent can collapse to a
single equilibrium and not be robust against off-distribution play.

Add `--snapshot-pool-dir DIR --snapshot-prob P`: with prob P at every
non-learner-seat decision, sample a frozen snapshot from a rolling pool
and use its policy. Essential for n>2-player Blef.

## Phase 5 — Multi-deck and multi-player (1 week)

Tonight's experiments were 24-deck, 2-player only. The user has noted
two production models (24-deck and 32-deck), trained separately. The
plan should bless one of:

### 5.1 Train 32-deck with the same protocol

Run pin+emb on 32-deck for 12M steps. Compare against
`nfsp_inference_32.pt` (production deployment).

### 5.2 Multi-player retraining

Use `--n-agents 3` and `--n-agents 4` with `--pick-n-agents-in-range`
and pin+emb. Plan §147 notes NFSP convergence isn't guaranteed for
n>2. Combined with snapshot-league (§4.4), this is the right setup.

### 5.3 Special-card variants

`--jokers J --blanks B --common-cards C` and the `--pick-*-in-range`
flags exist. Train with mixed configs (e.g. pick jokers in [0, 2],
blanks in [0, 2]) so the model handles all rule sets the user runs.

## Phase 6 — Architecture improvements (2–4 weeks)

### 6.1 Action-head factorization (originally 2.1)

Plan §144. Plan §240 raised real correctness concerns
(parametric_detail incompatibility across set-types). With pin+emb
already a strong baseline, factorization can be tried at small scale
without staking a long training run on it.

### 6.2 Width sweep on top of pin+emb (originally 2.3)

{64, 128, 256, 512} hidden width × fixed compute. Plan §218 noted that
narrower might generalize better with limited data. Pin+emb has fixed
the data limitation, so larger nets may now actually win.

## Phase 7 — Robustness and exploration (1–3 weeks)

### 7.1 Two-seed evals on every claim

Single-seed runs + 200-game evals = ±0.07 CI. Most claims tonight
need Δ ≥ 0.10 to be statistically defensible. Adopt 2-seed minimum
for any retrained variant; 4-seed for headline results. Use the
already-existing `--seed` flag to tools/eval_ladder.py and have a
helper that aggregates.

### 7.2 Phase 0.3 (eval-during-training)

Wire `tools/eval_ladder.py` into the trainer at a low cadence (e.g.
every 1M steps, 100 games per opponent, only random+conservative).
Adds ~30s overhead per million steps; well worth it for visibility into
convergence-vs-CFR.

### 7.3 CFR ladder up to (11, 11)

Tonight I fetched all CFR hand-pair strategies up to (11, 11). Total
1.7 GB. Confirmed CFR may be poorly converged at high mc (baseline-9M
beat CFR @ mc=11 at 0.59 — outside Nash range). Treat CFR as a
near-Nash benchmark only at mc ≤ 6.

## Phase 8 — Production deploy (1 week)

### 8.1 `--export-inference` for promotion

`agent.py` already has `export_inference()`. Use it at the end of every
final training run to produce a small (~1 MB) inference checkpoint
suitable for Lambda. Document the contract (`obs_dim`, `act_dim`, `q`,
`pi`, `cfg`, `version`).

### 8.2 Migration from `nfsp_inference_24.pt`

Tonight's eval shows the deployed `nfsp_inference_24.pt` scores 0.21 vs
Conservative @ mc=1 — significantly worse than the recent 9M baseline
(0.535) at that cell. Training-time distribution looks different from
mc-baseline-10M. Once the Phase 4–5 pin+emb candidate is validated,
promote it to production and retire `nfsp_inference_24.pt`.

### 8.3 Two-deck deployment harness

Maintain separate 24-deck and 32-deck inference checkpoints (per current
production setup). Or — explore a unified model with the factorized
action head from Phase 6.1, contingent on factorization working.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code) overnight session 2026-05-08
