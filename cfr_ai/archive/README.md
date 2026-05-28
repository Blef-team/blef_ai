# CFR archive

Two kinds of entries live here:

* **Strategy snapshots** — self-contained model folders consumable by
  `cfr_ai/analysis/head_to_head.py` via `--model1 <tag>` / `--model2 <tag>`.
  Created by `python -m cfr_ai.archive_tool --tag <name>` from the project
  root, which snapshots the current `cfr_ai/outputs/`.
* **Shelved code experiments** — directories that contain a `README.md`
  describing the experiment, why it was shelved, and what's preserved.
  These are research records, not active code; nothing in production
  `cfr_ai/` imports from them.

## Strategy snapshots

| Tag | Date | Setups | Note |
|---|---|---|---|
| `v0_baseline` | 2026-05-24 | all (66) | Production CFR baseline: rounds 1-21, MCCFR external sampling, 5M iter per setup. Reference point for any algorithm/abstraction changes. |
| `v1_post_speedup` | 2026-05-25 | 1_1, 1_2, 1_3, 2_2 | Rounds 1-3 retrained after precompute-set-existence (2x), encoding-round (unbiased), and clear-lows-non-mutating fixes. LBR-1 and game values match v0_baseline within noise; cleaner diagnostic files. |
| `v0` | 2026-05-27 | all (66) | Pre-Hetzner-retrain baseline (Python trainer prod CFR). Same strategies as the deleted v0_baseline, but in NPZ format. Diagnostics excluded; rerun with --include-diagnostics to add them. |

## Shelved code experiments

| Directory | Shelved | Note |
|---|---|---|
| `subgame/` | 2026-05-28 | Depth-1 subgame solver + analysis scripts. Online use case ruled out by Phase 2 LBR-1 cost evidence. See `subgame/README.md` for full rationale and revival path. |
