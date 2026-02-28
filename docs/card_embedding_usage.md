# Card Embedding Usage Guide

This note explains how to pretrain card embeddings and plug them into the NFSP Blef agent. These embeddings can be used to build any neural network-based agent for Blef.

## 1. Pretraining
- Run `tools/pretrain_card_embeddings.py` with the desired coverage. Example:
  ```bash
  PYTHONPATH=. python3 tools/pretrain_card_embeddings.py \
      --epochs 20 \
      --train-batches-per-epoch 800 \
      --val-batches 200 \
      --batch-size 512 \
      --max-private-cards 12 \
      --max-common-cards 12 \
      --num-jokers 2 \
      --num-blanks 1 \
      --save-path artifacts/card_embedding_pretrain.pt
  ```
- The script samples private/common hands uniformly from the specified ranges, consults `determine_set_existence` for labels, and stores the best checkpoint in `artifacts/card_embedding_pretrain.pt`.
- Use `--print-val-samples N` to inspect validation hands and the model’s top predicted actions each epoch.

## 2. Artifact Contents
The saved checkpoint contains:
- `encoder_config`: rank/suit dimensions, joker/blank id counts, embedding-table sizes, aggregation type, LayerNorm flag.
- `encoder_state_dict`: weights for the rank/suit (and optional joker/blank) embedding tables plus the aggregator LayerNorm.
- `model_state_dict`: encoder + classifier MLP (handy if you want to reuse the head).
- `hyperparams`: CLI settings used to train, including card-count ranges and joker count.
- `metrics`: best validation loss/accuracy.

## 3. Integrating With NFSP
1. Load the artifact and initialise `CardEmbeddingEncoder` with `encoder_config`, then load `encoder_state_dict`.
   ```python
   artifact = torch.load("artifacts/card_embedding_pretrain.pt", map_location="cpu")
   cfg = CardEmbeddingConfig(**artifact["encoder_config"])
   encoder = CardEmbeddingEncoder(cfg)
   encoder.load_state_dict(artifact["encoder_state_dict"])
   ```
2. Update the observation pipeline to emit padded card ID tensors (`-1` for padding, `>=24` for jokers) alongside the existing scalar features.
   - Reserve sequential ids for jokers (`24..24+num_jokers-1`) followed by blanks (`24+num_jokers .. 24+num_jokers+num_blanks-1`) so the encoder can map them correctly.
3. In the NFSP agent, wrap the encoder behind a feature flag (e.g., `use_card_embeddings`). When enabled, feed the encoder output into the Q/π networks instead of the 24-dim multi-hot vectors.
4. Store the flag (and embedding config) in checkpoints so resumes stay consistent. New runs can default to embeddings once validated.

## 4. Fine-Tuning Tips
- Freeze embeddings for the first few million env steps or apply a smaller LR to prevent catastrophic drift.
- Log embedding/gradient norms (e.g., `priv_emb_norm`, `common_emb_norm`) to catch instability.
- Run short self-play smoke tests (≤200k steps) with and without embeddings; compare CHECK metrics, illegal-rate, and training losses before fully switching over.

## 5. Regeneration / Variants
- Rerun the pretrainer whenever deck rules change (e.g., enabling blanks) or the curriculum extends beyond the current card-count range.
- For faster experiments, reduce `--max-private-cards`, `--max-common-cards`, or epochs; for exhaustive coverage, increase the batch counts and epochs.
