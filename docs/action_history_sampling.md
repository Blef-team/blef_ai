# Action History Sampling

The NFSP runner now emits lightweight action-history snippets so you can inspect representative rounds without retaining millions of full game files.

## Output
- Default path: `logs/action_samples_<timestamp>.jsonl` (configurable via `--history-sample-path`).
- Format: JSON Lines; each entry contains:
  ```json
  {
    "step": 150000,
    "history": [
      {"player": "0", "action_id": 12},
      {"player": "1", "action_id": 34},
      ...
    ],
    "round_result": {
      "actor": "1",
      "loser": "0",
      "ref": "1",
      "before_counts": {"0": 3, "1": 3},
      "after_counts": {"0": 4, "1": 2}
    },
    "reward": 1.0,
    "eta": 0.18,
    "epsilon": 0.04,
    "check_prob": 0.12,
    "max_cards": 5,
    "pins": {
      "overrides": ["n_step"],
      "env": []
    }
  }
  ```
- Entries appear at the first CHECK resolution after each `history_sample_every` interval (default: 100k env steps). Set `--history-sample-limit` to control how many snippets are recorded (≤0 disables the limit).

## CLI Flags
```
--history-sample-every <steps>
--history-sample-limit <count>
--history-sample-path <file>
```
Example:
```
python nfsp_run_local.py --control-plane control_plane.json \
    --history-sample-every 50000 \
    --history-sample-limit 200 \
    --history-sample-path logs/history_samples_run42.jsonl
```

## Notes
- Histories capture the actions taken *before* the CHECK that resolved the round so you can reconstruct the decisive sequence.
- The `round_result` field summarizes the outcome from the reference player's perspective (which player lost, before/after card counts).
- Each record also includes the current exploration and control-plane pin state, enabling correlation with schedule overrides.
