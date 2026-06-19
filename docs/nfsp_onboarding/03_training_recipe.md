# Training recipe — what's load-bearing, and what was never built

## The load-bearing "control" recipe (reaches Nash parity)
- **Net:** W256 hidden (chosen over W128). Card + history **pretrained embeddings**
  (`artifacts/card_embedding_pretrain_{24,32}.pt` + history encoder). Keep params ~0.5–1.5M.
- **Q head:** **Monte-Carlo regression, NO bootstrap / target net** (`fbbd720`). Leftover
  TD/Polyak/`hard_target_interval` knobs are **vestigial** — don't reintroduce bootstrapping.
- **π head:** average policy via true **reservoir** sampling (uniform over history). Do
  **not** prioritize SL. Store **behavioral one-hots** in the reservoir (no on-policy leakage).
- **Episodes:** each **round** is its own episode (`d627754`); reward ±1 at CHECK with
  `γ^d` shaping, **γ=0.666**. Fixed-perspective from a `ref_nick` (never actor-relative).
- **Curriculum:** `max_cards` 1→11 over ~29M steps. Short runs never see deep rounds.
- **Routing:** **specialists + most-specific-first**, not one generalist. Models ~1.5MB;
  loading many is free.
- Reaches ~0.499 vs CFR (Nash parity) by ~9M steps; beats Conservative everywhere.

## Credit assignment (sparse terminal reward)
Reward fires only at round end, so pre-CHECK bets look like noise. Use one (or A+B):
- **(A) n-step TD**, n≈5–10 (median moves/round). Cheapest.
- **(B) round-level MC backfill** — overwrite the last K=5–10 transitions with the final
  fixed-perspective reward; earlier steps keep 0. Works well for sparse round games.
- **(C) TD(λ)**, λ≈0.9 — principled, more work; skip unless traces already exist.
- **Precondition:** reward must be fixed-perspective first (see dead-end A1/A2).

## Schedules (starting points / sanity bands; live-tune via control plane)
Canonical spec: `docs/nfsp_training_playbook.yaml`. Typical bands:

| knob | schedule |
|---|---|
| η (anticipatory) | 0.25 (0–10M) → 0.15 → 0.12 → 0.10 (25M). More BR labels early, bias π later. |
| ε (BR branch only) | 0.20→0.05 over 0–5M; drift to 0.02 by 25M. **Floor 0.05 during early training.** |
| Q LR | 1e-4 → 7e-5 → 5e-5 → 3e-5 (last ~2M). |
| π LR | ~1e-4–3e-4 (notes conflict; π is supervised, keep responsive). |
| replay reuse ρ | target **10–20** (25 temporarily OK, **never >30**). `ρ ≈ batch_rl/(η·train_rl_every)`; steer via `train_rl_every`. |
| n_step | 5 → 7 (~12–20M) → 10 (deep rounds). On change, `_flush_nstep(force=True)`. |
| buffers | grow RL & SL to ~1M each by 5M steps. |

## Tripwires (sanity bands — know these)
- **Q-loss** healthy 0.15–0.6. MA >0.8 for >200k steps → halve Q LR or bump `train_rl_every`.
- **SL-loss** ~1.8–2.2 in the ~80-action space is fine.
- **Entropy H(π)** should decline *slowly*. **Collapse to <~0.6 too early = bad** (this is
  the same failure mode as the kupala reward-shaping collapse). If H(π)<0.6 while
  SLloss>1.8, add a tiny SL entropy bonus (`+0.001·entropy`) for 1–2M steps.
- **illegal-action rate** should stay ≈ 0. **avgR ≈ 0 in balanced self-play is correct.**

## Metrics glossary
`avgR` (≈0 balanced = good, ≈1 = the actor-credit bug) · `win` (reward>0 fraction, ref POV)
· `len` (steps/episode; too short = degenerate early checks) · `Qloss` · `SLloss` ·
`H(pi)` (low/fast crash = collapse) · `illegal` (≈0) · `eps` · `RL_buf`/`SL_buf` fill.
Note: `metrics.csv` has **two rows per step** (BR + behavior); avg_reward/win_rate stderr
~0.035 — wait for 3+ consecutive rows before reacting.

## Built but check before assuming — and NOT built
**Shipped:** control plane (JSON watcher, 50k cooldown, pinned overrides persist across
resume, atomic writes), embeddings (pretrained + wired), profiling toolkit (~470→~4400
steps/s, `tools/profile_selfplay.py`), team mode (engine + trainer, PR #64).

**Aspirational / NOT implemented (don't assume these exist):**
- **CHECK-focused curriculum** (`docs/check_focus_training_plan.md`) — terminal-pair
  buffer, contrastive Q-penalty, aux CHECK-outcome head, CHECK exploration floor. Largely
  not in code. COA/CAC CHECK metrics are planned, not wired.
- **Schedule-gating / assertion infra / control-plane CI smoke**
  (`docs/training_schedule_plan.md`) — mostly not built; schedules trusted by convention.
- **32-deck:** manager/probabilities/runner/embeddings done, but `probability_table.py`
  is still 24-only and `prompt_env_32.txt` regeneration is flagged. **CFR has no 32-deck
  strategies** → morana is useless on 32-deck (recurring blocker).
- Sparsity-audit / feature-ablation / dueling-head / gradient-diagnostic ideas listed in
  `docs/model_architecture_plan.md` — not pursued.
