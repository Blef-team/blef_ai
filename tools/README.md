# tools/

Operational tooling for evaluating, comparing, and visualising NFSP runs.

## Eval ladder — head-to-head metric harness

`tools/eval_ladder.py` plays a checkpointed NFSP agent against a fixed opponent
and reports winrate / mean reward / 95% CI. Four opponent types in the "metric
ladder," ordered floor → strong:

| Opponent       | Rationale                                                 |
| -------------- | --------------------------------------------------------- |
| `random`       | Floor. Anything weaker than this is broken.               |
| `conservative` | Current default baseline (rule-based).                    |
| `cfr`          | Tabular CFR, near-Nash at low cardinality. Requires the strategy CSVs under `cfr_ai/outputs/` plus a top-level `metadata.csv`. Skipped automatically if missing. |
| `snapshot`     | Another NFSP checkpoint (for self-improvement signal).    |

Run from the repository root:

```sh
python -m tools.eval_ladder \
  --checkpoint nfsp_blef_<timestamp>_<N>M.pt \
  --opponents random,conservative \
  --n-games 200 \
  --deck-size 24 --n-players 2 --max-cards 11
```

The `--deck-size`, `--jokers`, `--blanks`, `--common-cards` flags must match the
configuration the checkpoint was trained with. Mismatched rules raise a clear
`ValueError` rather than producing nonsense (the trained network's input shape
won't match the env's observation shape).

## Backfill — run the ladder over historical checkpoints

`tools/backfill_eval.py` iterates a glob of `.pt` checkpoints and writes one
`eval_results/<checkpoint-stem>/ladder.csv` per checkpoint. Resumable: skips
checkpoints whose `ladder.csv` already exists.

```sh
python -m tools.backfill_eval \
  --pattern 'nfsp_blef_*_*M.pt' \
  --opponents random,conservative \
  --n-games 200 \
  --deck-size 24 --n-players 2 --max-cards 11
```

## Optional dependencies

The dashboard requires `streamlit` and `pandas`. Install with:

```sh
pip install -r tools/requirements.txt
```

The eval ladder and backfill scripts use only stdlib + project deps; no extra installs.

## Dashboard — Streamlit live view

`tools/run_dashboard.py` is a Streamlit app that auto-discovers run dirs under
`runs/` and ladder backfills under `eval_results/`. Multi-select runs to overlay
their training metrics; multi-select backfills to compare ladder outcomes
across checkpoints.

```sh
streamlit run tools/run_dashboard.py
```

The browser tab stays open; new runs appear automatically as their dirs
materialise. Auto-refresh interval is configurable in the sidebar.

Conventions the dashboard expects:

- Runs:        `runs/<YYYYMMDD-HHMMSS>__<experiment-name>/metrics.csv`
- Backfills:   `eval_results/<checkpoint-stem>/ladder.csv`

A future PR will rewire `nfsp_run_local.py` to write into the timestamped
`runs/` convention by default. For now, the dashboard works on backfilled
results out of the box.
