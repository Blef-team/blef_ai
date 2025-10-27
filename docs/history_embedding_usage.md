# History Embedding Usage Guide

Use this note when you want NFSP to replace the legacy one-hot action-history features with the pretrained history encoder that ships with the project.

## 1. Pretrain (or refresh) the encoder
- Script: `tools/pretrain_history_embeddings.py`
- Example run (CPU friendly defaults shown):
  ```bash
  PYTHONPATH=. python3 tools/pretrain_history_embeddings.py \
      --epochs 10 \
      --train-batches-per-epoch 800 \
      --val-batches 200 \
      --batch-size 512 \
      --history-len 8 \
      --embedding-dim 64 \
      --deck-size 24 \
      --min-k 1 \
      --max-k 6 \
      --save-path artifacts/history_embedding_pretrain.pt
  ```
- What the dataset does: it samples `k` recent bet action IDs (CHECK is excluded), builds the minimal card sets that realise those actions, merges the cards, and labels every action ID that the merged cards make legal. Each sample is an action sequence (most recent first) padded to `history_len` with -1s plus a mask.
- Controls you may tweak:
  - `--deck-size {24,32}` to match the deck you train on.
  - `--history-len` to keep in sync with the downstream observer (NFSP currently expects length 8).
  - `--embedding-dim` for the width of the learned embedding.
  - `--min-k/--max-k` to widen or restrict how many historical moves appear in a sample.
  - `--no-positional` / `--no-layer-norm` if you want to ablate positional encodings or LayerNorm.
  - `--print-val-samples N` to print a few validation examples with the model’s top predictions each epoch.

## 2. Artifact contents
Running the script writes a `.pt` file. It always contains:

| Key | Meaning |
| --- | --- |
| `version` | Format bump guard (currently `1`). |
| `encoder_config` | Dict used to rebuild `HistoryEmbeddingConfig` (num actions, history length, embedding dim, flags). |
| `encoder_state_dict` | Weights for `HistoryEmbeddingEncoder` (all you need at run time). |
| `model_state_dict` | Full classifier weights (encoder + head) in case you want to resume pretraining. |
| `optimizer_state_dict` | AdamW state for warm restarts. |
| `hyperparams` | CLI values (history length, deck size, hidden sizes, dropout, etc.). |
| `metrics` | Best validation loss/accuracy reached so far. |

You can reload the encoder in Python with:
```python
artifact = torch.load("artifacts/history_embedding_pretrain.pt", map_location="cpu")
cfg = HistoryEmbeddingConfig(**artifact["encoder_config"])
encoder = HistoryEmbeddingEncoder(cfg)
encoder.load_state_dict(artifact["encoder_state_dict"])
encoder.eval()
```

## 3. Plug it into NFSP self-play
1. Launch your run with the flag:
   ```bash
   python nfsp_ai/nfsp_run_local.py --use-history-embeddings auto
   ```
   - `auto` loads `artifacts/history_embedding_pretrain.pt` if it exists.
   - Provide an explicit path (e.g., `--use-history-embeddings artifacts/expA_history.pt`) to override.
   - Pass `--history-embedding-device cuda:0` if you want the encoder on GPU; CPU is the default.
   - Omit the flag to fall back to the legacy per-player multi-hot history grid.
2. `nfsp_run_local.py` wraps the artifact in a `HistoryEmbeddingRuntime`. It assumes:
   - `history_len` matches the encoder config (the default curriculum uses 8).
   - Eight slots (`HISTORY_SLOTS = 8`) are reserved per observation. Slot order is fixed relative to the current player: seats immediately to your left/right occupy the adjacent slots, with outer slots filled as the table grows (see `_HISTORY_SLOT_MAP` in `nfsp_run_local.py`).
   - Action IDs must stay below the CHECK action index (the encoder ignores CHECK, just like the dataset).
3. During observation construction (`vectorize_obs`):
   - The collector walks the round history backwards, gathering each player’s most recent bets up to `history_len`.
   - The encoder is invoked once per observation (`runtime.encoder(actions, mask)`), outputting a `(slots × embedding_dim)` matrix that is flattened into the final observation vector.
   - If the artifact is missing or mismatched (wrong history length / action count), the runner prints a warning and reverts to multi-hot features.

## 4. Validation checklist
- Sanity checks before long runs:
  - Run the pretrainer’s validation printouts (`--print-val-samples 3`) to spot obvious mislabelling.
  - From Python, feed a hand-crafted sequence (e.g., `[12, 27, 45, -1, …]`) through the encoder and confirm the output shape is `embedding_dim × 8`.
  - Start a short self-play session (≤200k env steps) with and without the embeddings and compare:
    - `history_feature_dim` reported in the console banner.
    - SL loss curve (should not flatline to zero immediately).
    - Illegal-action rate (embedding mistakes often surface here).
- Unit tests are available:
  ```bash
  pytest tests/test_history_embedding_dataset.py tests/test_history_embedding_pretraining.py
  ```

## 5. When to retrain or adjust
- Deck rules change (switching to 32-card deck, enabling jokers/blanks): rerun the pretrainer so the label oracle matches the live environment.
- You change `history_len` or the number of seats you model: regenerate artifacts and restart runs; checkpoints store the feature dimensionality.
- You observe drift when fine-tuning: try freezing the encoder for a warm-up window or reducing its learning rate in your NFSP optimisers (the runtime wrapper currently keeps the encoder in eval mode, so no gradients flow unless you explicitly hook it into the model).
- Need richer behaviour? Increase `embedding_dim`, widen the classifier MLP (`--hidden-sizes`), or add dropout for regularisation.

Keeping this doc beside `card_embedding_usage.md` should make it easy to manage both embedding families consistently.
