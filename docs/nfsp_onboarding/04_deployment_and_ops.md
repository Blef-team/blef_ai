# Deployment, serving, prod-safety, and ops hygiene

## Serving chain
`deployment/lambda_function.py` / `aiagent.py` → root `agent.py` →
`nfsp_ai/production_agent.py`. Env vars `NFSP_MODEL_PATH / NFSP_GREEDY / …` select
profiles. **DynamoDB = game-state source of truth**; SQS carries move deadlines; CFR runs
on separate per-hand-size workers. Live Lambda (eu-west-2):
`blef-nfsp-lambda:lambda-compatible` (verify current digest with `aws lambda get-function`).

## Routing (`production_agent.py`)
- **Most-specific-first:** `24_1v1_kupala` → `24_1v1` → `24` → … Personality suffix is
  optional in the inference-filename regex (`[A-Za-z0-9]+` segments; **hyphenated names
  are rejected and never loaded**).
- 1v1 specialists **canonicalize eliminated players away** (n_active collapse).
- **Cross-variant fallback within a deck, NEVER across decks.**
- Trained ckpt present → honor stamped `cfg["personality"]` (`greedy`, `head`), **skip
  sculpting**. No ckpt → fall through to `<deck>_<variant>` + sculpt overlay.
- PR #71 adds an **inference-only public-priors mask filter**: zeros bet actions whose
  public-prior ≈ 0 (e.g. Full house with 4 cards in play). CHECK is never touched; a
  safety rail returns the original mask if filtering empties it. (Trained NFSP was
  occasionally betting provably-impossible sets — a bug, not a personality.)

## Prod-safety rules (each one is a scar)
- **No prod swap without full deployment-surface coverage.** Full surface = **3p+,
  jokers, blanks, common cards.** A candidate must be **trained** (not just evaluated/
  assumed-to-generalize) on every axis: n_agents (2p/3p/4p), deck 24 AND 32, j/b/cc. If any
  axis was held at default/zero, **refuse to call it SOTA or swap** until trained or the
  user explicitly accepts the regression.
- **"SOTA" requires the slice named.** Never bare "SOTA" — say e.g. "best on
  (2p, deck=24, j=0, b=0, cc=0)". 2p→3p generalization is a hypothesis, not a metric.
- **Verify the tested ckpt IS the exported artifact.** Prod serves
  `artifacts/nfsp_inference_<deck>_<variant>.pt`, **not** run-dir
  `runs/<id>/checkpoints/nfsp_blef.pt` (often a different snapshot — one observed
  first-layer maxdiff ~2.86). Compare `q`/`pi` state-dicts with `torch.allclose` before
  saying "production model". Keep non-prod files OUT of `artifacts/` (deploy rsyncs the
  whole dir) — stage backups under `runs/.distinct_pivot/artifacts_*`.
- **Don't overgate authorized prod actions.** When the user says "deploy"/"run it" and
  `aws sts get-caller-identity` resolves, just run it (incl. `SWAP=1`). The deploy script
  already has the rails (SWAP=0 default, auth pre-flight). Don't ladder more confirmations.
- **No parallel dependent FS/upload ops** — sequence reorg → swap → upload, one step per
  turn; confirm counts are stable (all mutators dead) before the next step.

## Training hygiene
- **Check disk before AND after every training.** `ls runs/` for a reusable match (export
  from an existing final ckpt via `eval_ladder.load_learner` + `agent.export_inference` —
  seconds vs hours); `df -h .` for headroom (runs are 1–8 GB). **Prune run-dir
  `checkpoints/` (buffers ~5G each) immediately after a verified export.** Surface if free
  space drops below ~20 GB.
- **One control-plane file per training.** Never share `--control-plane` across concurrent
  runs — live tuning couples experiments. Name `cp_<experiment-key>.json`; swap the path
  when copy-pasting a prior `cmd.txt`. Whitelist: `eta/eps/lr_q/lr_pi/cadence/batch/tau/
  n_step/max_cards/check_prob/burst`. 50k-step cooldown; pinned overrides persist across resume.
- **Actively monitor; don't fire-and-forget.** Arm a metric-anomaly watcher (win_rate
  plateau, avg_reward regression, sl_loss >3.0 or rising, q_loss reversal) with an
  adjustment policy ready (sl_loss climbing → pulse `sl_learning_off`; plateau → ε→0.05,
  η→0.40; reward regression → revert; sustained reward <0.45 past 6M of 12M → abort).

## Scope / ownership (do not cross)
- **`cfr_ai/` is READ-ONLY** (external maintainer). Never `git add cfr_ai/…` or edit it;
  root `metadata.csv` is also CFR territory. Reading blueprints / calling
  `cfr_ai.agent.determine_action` is fine. **Report CFR bugs to the maintainer; never patch.**
- **dazhbog → Conservative, porevit → ConservativeCrawling** (delegated rule bots).
  **Exclude from all trained/sculpted personality work**, even if a theme fits perfectly.
- **domovoi shipped sculpt-only** (trained ckpts retired 2026-06-13). The other 11 trained
  bogs are live. Don't resurrect trained domovoi without re-clearing the "too strong for
  the character" judgment.
