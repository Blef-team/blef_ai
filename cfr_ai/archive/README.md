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
| `v1` | 2026-05-27 | all (66) | Pre-Hetzner-retrain baseline (Python trainer prod CFR). Diagnostics excluded; rerun with --include-diagnostics to add them. |

## Shelved code experiments

| Directory | Shelved | Note |
|---|---|---|
| `subgame/` | 2026-05-28 | Depth-1 subgame solver + analysis scripts. Online use case ruled out by Phase 2 LBR-1 cost evidence. See `subgame/README.md` for full rationale and revival path. |
