#!/usr/bin/env python3
"""
Pretrain action-history embeddings for NFSP Blef.

Training task: Given a sequence of recent action IDs (excluding CHECK) sampled from the
closure dataset, predict which action IDs are achievable for the merged card set
constructed from those actions. The encoder is later transferred into NFSP to embed
per-player histories.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from nfsp_ai.embedding import HistoryEmbeddingConfig, HistoryEmbeddingEncoder
from tools.history_embedding_dataset import ClosureIterableDataset


class HistoryClassifier(nn.Module):
    """Wraps the history embedding encoder with a classification head."""

    def __init__(
        self,
        encoder: HistoryEmbeddingEncoder,
        hidden_sizes: Sequence[int],
        dropout: float,
        num_actions: int,
    ):
        super().__init__()
        self.encoder = encoder
        layers: List[nn.Module] = []
        in_dim = encoder.config.embedding_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, num_actions))
        self.head = nn.Sequential(*layers)

    def forward(self, actions: torch.LongTensor, mask: torch.BoolTensor) -> torch.Tensor:
        emb = self.encoder(actions, mask)
        return self.head(emb)


def make_loader(dataset: IterableDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False)


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).to(dtype=targets.dtype)
        accuracy = (preds == targets).float().mean().item()
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="mean").item()
    return loss, accuracy


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = ClosureIterableDataset(
        deck_size=args.deck_size,
        num_samples=args.train_batches_per_epoch * args.batch_size,
        history_len=args.history_len,
        min_k=args.min_k,
        max_k=args.max_k,
        seed=args.seed,
    )
    val_dataset = ClosureIterableDataset(
        deck_size=args.deck_size,
        num_samples=args.val_batches * args.batch_size,
        history_len=args.history_len,
        min_k=args.min_k,
        max_k=args.max_k,
        seed=args.seed + 5_000,
    )
    num_actions = dataset.num_actions

    embedding_dim = max(1, int(args.embedding_dim))
    args.embedding_dim = embedding_dim
    encoder_cfg = HistoryEmbeddingConfig(
        num_actions=num_actions,
        history_len=args.history_len,
        embedding_dim=embedding_dim,
        use_positional=not args.no_positional,
        use_layer_norm=not args.no_layer_norm,
    )
    encoder = HistoryEmbeddingEncoder(encoder_cfg)
    model = HistoryClassifier(
        encoder,
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        num_actions=num_actions,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = make_loader(dataset, args.batch_size, args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, args.num_workers)

    best_val_loss = float("inf")
    save_path = args.save_path
    if save_path == "auto":
        save_path = f"artifacts/history_embedding_pretrain_{args.deck_size}.pt"
    artifact_path = Path(save_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        dataset.set_epoch(epoch)
        running_loss = 0.0
        batches = 0
        for actions, mask, labels in train_loader:
            actions = actions.to(device=device, dtype=torch.long)
            mask = mask.to(device=device, dtype=torch.bool)
            labels = labels.to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            logits = model(actions, mask)
            loss = criterion(logits, labels)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        avg_train_loss = running_loss / max(1, batches)

        model.eval()
        val_dataset.set_epoch(epoch)
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        samples_to_print = args.print_val_samples
        printed = 0
        with torch.no_grad():
            for actions, mask, labels in val_loader:
                actions = actions.to(device=device, dtype=torch.long)
                mask = mask.to(device=device, dtype=torch.bool)
                labels = labels.to(device=device, dtype=torch.float32)

                logits = model(actions, mask)
                loss, acc = compute_metrics(logits, labels)
                val_loss += loss
                val_acc += acc
                val_batches += 1

                if printed < samples_to_print:
                    probs = torch.sigmoid(logits).cpu().numpy()
                    actions_cpu = actions.cpu().numpy()
                    mask_cpu = mask.cpu().numpy()
                    labels_cpu = labels.cpu().numpy()
                    batch = probs.shape[0]
                    for i in range(batch):
                        if printed >= samples_to_print:
                            break
                        top_preds = probs[i].argsort()[::-1][:20]
                        active_actions = actions_cpu[i][mask_cpu[i]]
                        print(
                            json.dumps(
                                {
                                    "sample": printed + 1,
                                    "history": active_actions.tolist(),
                                    "legal_actions": np.flatnonzero(labels_cpu[i] > 0.5).tolist(),
                                    "top_predictions": [
                                        {"action": int(a), "prob": float(probs[i][a])} for a in top_preds.tolist()
                                    ],
                                }
                            ),
                            flush=True,
                        )
                        print()
                        printed += 1

        avg_val_loss = val_loss / max(1, val_batches)
        avg_val_acc = val_acc / max(1, val_batches)

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(avg_train_loss, 6),
                    "val_loss": round(avg_val_loss, 6),
                    "val_accuracy": round(avg_val_acc, 6),
                }
            ),
            flush=True,
        )

        if avg_val_loss < best_val_loss - args.early_stop_delta:
            best_val_loss = avg_val_loss
            artifact = {
                "version": 1,
                "encoder_config": encoder_cfg.to_dict(),
                "encoder_state_dict": encoder.state_dict(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "hyperparams": {
                    "deck_size": args.deck_size,
                    "history_len": args.history_len,
                    "embedding_dim": args.embedding_dim,
                    "hidden_sizes": list(args.hidden_sizes),
                    "dropout": args.dropout,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "grad_clip": args.grad_clip,
                    "train_batches_per_epoch": args.train_batches_per_epoch,
                    "val_batches": args.val_batches,
                    "min_k": args.min_k,
                    "max_k": args.max_k,
                },
                "metrics": {
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": avg_val_acc,
                },
            }
            torch.save(artifact, artifact_path)

    print(f"Saved best weights to {artifact_path}", flush=True)


def parse_hidden_sizes(raw: str) -> List[int]:
    if not raw:
        return [256, 256]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-batches-per-epoch", type=int, default=800)
    parser.add_argument("--val-batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=16,
        help="History embedding width (default: 16).",
    )
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--hidden-sizes", type=parse_hidden_sizes, default=[256, 256])
    parser.add_argument("--deck-size", type=int, choices=[24, 32], default=24)
    parser.add_argument("--no-positional", action="store_true")
    parser.add_argument("--no-layer-norm", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save-path", type=str, default="auto")
    parser.add_argument("--print-val-samples", type=int, default=10)
    parser.add_argument("--early-stop-delta", type=float, default=1e-4)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
