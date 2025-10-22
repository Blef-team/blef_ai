#!/usr/bin/env python3
"""
Pretrain card embeddings for input vectorisation for a neural network based agent.

The task: given a single hand snapshot (private or common cards sampled from the
Blef deck, default 24-card but optionally 32-card) represented as 32 padded card
ids plus aggregate counts (ranks, suits, jokers, blanks), predict which bet actions
are valid. Labels use `shared.api.simpleschema_local_manager.determine_set_existence`,
matching runtime legality (including jokers and blanks).

The output: `artifacts/card_embedding_pretrain.pt` — a pretrained checkpoint with
encoder configuration, encoder/head weights, training hyperparameters, and best
validation metrics ready for NFSP integration or fine-tuning.
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from nfsp_ai.embedding import CardEmbeddingEncoder, CardEmbeddingConfig
from shared.api.simpleschema_local_manager import determine_set_existence
from shared.game_utils import GameRules


RANK_LABELS = {
    6: ["9", "T", "J", "Q", "K", "A"],
    8: ["7", "8", "9", "T", "J", "Q", "K", "A"],
}


@dataclass
class DeckDomain:
    deck_size: int
    num_values: int
    rank_labels: List[str]
    check_action_id: int  # exclusion of CHECK action itself

    @property
    def num_actions(self) -> int:
        return self.check_action_id


def build_deck_domain(deck_size: int) -> DeckDomain:
    if deck_size % 4 != 0:
        raise ValueError("deck_size must be divisible by 4")
    rules = GameRules(deck_size)
    num_values = deck_size // 4
    rank_labels = RANK_LABELS.get(num_values, [str(v) for v in range(num_values)])
    return DeckDomain(
        deck_size=deck_size,
        num_values=num_values,
        rank_labels=rank_labels,
        check_action_id=rules.check_action_id,
    )


def parse_int_list(raw: str) -> List[int]:
    vals = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("List must contain at least one integer")
    return vals


def parse_card_count_list(raw: str) -> List[int]:
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            step = 1 if end >= start else -1
            values.extend(list(range(start, end + step, step)))
        else:
            values.append(int(token))
    if not values:
        raise argparse.ArgumentTypeError("List must contain at least one integer or range segment")
    deduped: List[int] = []
    for v in values:
        if v not in deduped:
            deduped.append(v)
    return deduped


def to_manager_cards(card_ids: np.ndarray, domain: DeckDomain, num_jokers: int, num_blanks: int) -> Tuple[List[dict], int, int]:
    cards = []
    jokers = 0
    blanks = 0
    base = domain.deck_size
    joker_limit = base + max(0, num_jokers)
    blank_limit = joker_limit + max(0, num_blanks)
    for card in card_ids.tolist():
        c_int = int(card)
        if c_int < base:
            cards.append({"value": c_int // 4, "colour": c_int % 4})
        elif c_int < joker_limit:
            cards.append({"value": -1, "colour": -1})
            jokers += 1
        elif c_int < blank_limit:
            cards.append({"value": -2, "colour": -2})
            blanks += 1
        else:
            raise ValueError("Card id outside configured deck range.")
    return cards, jokers, blanks


def compute_existence_vector(card_ids: np.ndarray, domain: DeckDomain, num_jokers: int, num_blanks: int) -> np.ndarray:
    cards, joker_count, blank_count = to_manager_cards(card_ids, domain, num_jokers=num_jokers, num_blanks=num_blanks)
    labels = np.zeros(domain.num_actions, dtype=np.float32)
    rules = {"deck_size": domain.deck_size}
    for action_id in range(domain.num_actions):
        labels[action_id] = 1.0 if determine_set_existence(
            cards,
            action_id,
            rules,
            num_jokers=joker_count,
            num_blanks=blank_count,
        ) else 0.0
    return labels


def decode_card(card_id: int, domain: DeckDomain, num_jokers: int, num_blanks: int) -> str:
    if card_id < 0:
        return "PAD"
    base = domain.deck_size
    joker_start = base
    joker_end = joker_start + max(0, num_jokers)
    blank_start = joker_end
    blank_end = blank_start + max(0, num_blanks)
    if card_id < base:
        rank = card_id // 4
        suit = card_id % 4
        ranks = domain.rank_labels
        suits = ["C", "D", "H", "S"]
        return f"{ranks[rank]}{suits[suit]}"
    if card_id < joker_end:
        return f"Joker{card_id - joker_start + 1}"
    if card_id < blank_end:
        return f"Blank{card_id - blank_start + 1}"
    return f"Card{card_id}"


def cards_to_str(card_ids: Sequence[int], domain: DeckDomain, num_jokers: int, num_blanks: int) -> str:
    visible = [
        decode_card(int(cid), domain, num_jokers=num_jokers, num_blanks=num_blanks)
        for cid in card_ids
        if cid >= 0
    ]
    return "[" + ", ".join(visible) + "]"


class CardExistenceIterableDataset(IterableDataset):
    """Generates synthetic (hand_ids, aux_features, labels) tuples for the classifier."""

    def __init__(
        self,
        num_samples: int,
        private_card_options: Sequence[int],
        common_card_options: Sequence[int],
        seed: int,
        domain: DeckDomain,
        rank_feature_dim: int,
        suit_feature_dim: int,
        max_hand_cards: int,
        num_jokers: int = 0,
        num_blanks: int = 0,
    ):
        super().__init__()
        self.num_samples = int(num_samples)
        self.private_card_options = tuple(int(x) for x in private_card_options)
        self.common_card_options = tuple(int(x) for x in common_card_options)
        self.base_seed = int(seed)
        self.domain = domain
        self.rank_feature_dim = int(rank_feature_dim)
        self.suit_feature_dim = int(suit_feature_dim)
        self.extra_feature_dim = self.rank_feature_dim + self.suit_feature_dim + 2
        self.num_jokers = max(0, int(num_jokers))
        self.num_blanks = max(0, int(num_blanks))
        base_deck = np.arange(domain.deck_size, dtype=np.int64)
        deck_parts = [base_deck]
        if self.num_jokers:
            joker_ids = np.arange(domain.deck_size, domain.deck_size + self.num_jokers, dtype=np.int64)
            deck_parts.append(joker_ids)
        if self.num_blanks:
            blank_start = domain.deck_size + self.num_jokers
            blank_ids = np.arange(blank_start, blank_start + self.num_blanks, dtype=np.int64)
            deck_parts.append(blank_ids)
        self.deck = np.concatenate(deck_parts).astype(np.int64)
        self.deck_size = self.deck.size
        self.max_private = max(self.private_card_options) if self.private_card_options else 0
        self.max_common = max(self.common_card_options) if self.common_card_options else 0
        self.max_hand_cards = int(max_hand_cards)
        if self.max_hand_cards <= 0:
            raise ValueError("max_hand_cards must be positive")
        if self.max_private > self.max_hand_cards or self.max_common > self.max_hand_cards:
            raise ValueError(
                f"Hand size exceeds supported maximum ({self.max_hand_cards}); "
                "adjust --max-private-cards/--max-common-cards or limit options."
            )
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _spawn_rng(self, worker_id: int) -> np.random.Generator:
        seed = self.base_seed + 9973 * self._epoch + 53 * worker_id
        return np.random.default_rng(seed)

    def __iter__(self) -> Iterable[Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        stride = worker.num_workers if worker else 1
        rng = self._spawn_rng(worker_id)

        private_opts = self.private_card_options
        common_opts = self.common_card_options

        start_index = worker_id
        produced = 0
        for sample_idx in range(start_index, self.num_samples, stride):
            private_count = int(rng.choice(private_opts))
            common_count = int(rng.choice(common_opts))
            total_cards = private_count + common_count

            if total_cards > self.deck_size:
                raise ValueError("Requested more cards than deck contains")

            draw = rng.choice(self.deck, size=total_cards, replace=False)
            rng.shuffle(draw)  # Break correlation between private/common slices
            private_cards = np.sort(draw[:private_count]) if private_count else np.array([], dtype=np.int64)
            common_cards = np.sort(draw[private_count:]) if common_count else np.array([], dtype=np.int64)

            # Choose which partition to emit this iteration (default uniform choice).
            use_private = True
            if private_cards.size and common_cards.size:
                use_private = bool(rng.integers(0, 2))
            elif private_cards.size == 0 and common_cards.size > 0:
                use_private = False
            hand_cards = private_cards if use_private else common_cards

            existence = compute_existence_vector(
                hand_cards,
                self.domain,
                num_jokers=self.num_jokers,
                num_blanks=self.num_blanks,
            ).astype(np.float32)

            hand_pad = np.full(self.max_hand_cards, -1, dtype=np.int64)
            if hand_cards.size:
                if hand_cards.size > self.max_hand_cards:
                    raise ValueError("Hand size exceeds maximum supported card slots.")
                hand_pad[:hand_cards.size] = hand_cards
            rank_counts = np.zeros(self.rank_feature_dim, dtype=np.float32)
            suit_counts = np.zeros(self.suit_feature_dim, dtype=np.float32)
            joker_count = 0.0
            blank_count = 0.0
            for cid in hand_cards:
                cid_int = int(cid)
                if cid_int < self.domain.deck_size:
                    rank_idx = cid_int // self.suit_feature_dim
                    suit_idx = cid_int % self.suit_feature_dim
                    if 0 <= rank_idx < self.rank_feature_dim:
                        rank_counts[rank_idx] += 1.0
                    else:
                        raise ValueError(f"Rank index {rank_idx} exceeds configured feature dim.")
                    suit_counts[suit_idx] += 1.0
                elif cid_int < self.domain.deck_size + self.num_jokers:
                    joker_count += 1.0
                elif cid_int < self.domain.deck_size + self.num_jokers + self.num_blanks:
                    blank_count += 1.0
                else:
                    raise ValueError("Card id outside configured deck range.")

            aux_features = np.concatenate(
                (
                    rank_counts,
                    suit_counts,
                    np.array([joker_count, blank_count], dtype=np.float32),
                )
            ).astype(np.float32, copy=False)

            yield (
                torch.from_numpy(hand_pad),
                torch.from_numpy(aux_features),
                torch.from_numpy(existence),
            )
            produced += 1
            if produced >= (self.num_samples + stride - 1) // stride:
                break


class CardExistenceModel(nn.Module):
    """Wraps the encoder with a lightweight classifier head."""

    def __init__(
        self,
        encoder: CardEmbeddingEncoder,
        hidden_sizes: Sequence[int],
        dropout: float,
        num_actions: int,
        extra_features: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.extra_features = extra_features
        features = encoder.output_dim + extra_features

        layers: List[nn.Module] = []
        in_dim = features
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, num_actions))
        self.head = nn.Sequential(*layers)

    def forward(self, hand_ids: torch.LongTensor, aux_features: torch.Tensor) -> torch.Tensor:
        emb, _ = self.encoder(hand_ids)
        if aux_features.dtype != emb.dtype:
            aux_features = aux_features.to(dtype=emb.dtype)
        if aux_features.dim() != 2 or aux_features.shape[1] != self.extra_features:
            raise ValueError(
                f"Expected auxiliary features with shape [B, {self.extra_features}], got {tuple(aux_features.shape)}"
            )
        features = torch.cat((emb, aux_features), dim=1)
        return self.head(features)


def make_loader(
    dataset: CardExistenceIterableDataset,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )


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

    domain = build_deck_domain(args.deck_size)

    encoder_cfg = CardEmbeddingConfig(
        rank_dim=args.rank_dim,
        suit_dim=args.suit_dim,
        aggregator=args.aggregator,
        use_layer_norm=not args.no_layer_norm,
        num_ranks=domain.num_values,
        num_joker_ids=args.num_jokers,
        num_blank_ids=args.num_blanks,
        num_joker_embeddings=args.num_joker_embeddings,
        num_blank_embeddings=args.num_blank_embeddings,
        base_deck_size=domain.deck_size,
    )
    encoder = CardEmbeddingEncoder(encoder_cfg)

    train_samples = args.train_batches_per_epoch * args.batch_size
    val_samples = args.val_batches * args.batch_size

    private_options = args.private_card_options or list(range(0, args.max_private_cards + 1))
    common_options = args.common_card_options or list(range(0, args.max_common_cards + 1))
    args.private_card_options = private_options
    args.common_card_options = common_options

    rank_feature_dim = domain.num_values
    suit_feature_dim = 4
    extra_feature_dim = rank_feature_dim + suit_feature_dim + 2
    max_hand_cards = max(
        domain.deck_size + args.num_jokers + args.num_blanks,
        max(private_options) if private_options else 0,
        max(common_options) if common_options else 0,
    )
    if max_hand_cards <= 0:
        max_hand_cards = domain.deck_size + args.num_jokers + args.num_blanks

    model = CardExistenceModel(
        encoder,
        args.hidden_sizes,
        args.dropout,
        num_actions=domain.num_actions,
        extra_features=extra_feature_dim,
    )
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_dataset = CardExistenceIterableDataset(
        num_samples=train_samples,
        private_card_options=private_options,
        common_card_options=common_options,
        seed=args.seed,
        domain=domain,
        rank_feature_dim=rank_feature_dim,
        suit_feature_dim=suit_feature_dim,
        max_hand_cards=max_hand_cards,
        num_jokers=args.num_jokers,
        num_blanks=args.num_blanks,
    )
    val_dataset = CardExistenceIterableDataset(
        num_samples=val_samples,
        private_card_options=private_options,
        common_card_options=common_options,
        seed=args.seed + 10_000,
        domain=domain,
        rank_feature_dim=rank_feature_dim,
        suit_feature_dim=suit_feature_dim,
        max_hand_cards=max_hand_cards,
        num_jokers=args.num_jokers,
        num_blanks=args.num_blanks,
    )

    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, args.num_workers)

    best_val_loss = float("inf")
    artifact_path = Path(args.save_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_dataset.set_epoch(epoch)
        running_loss = 0.0
        total_batches = 0
        for hand_ids, aux_features, labels in train_loader:
            hand_ids = hand_ids.to(device=device, dtype=torch.long)
            aux_features = aux_features.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            logits = model(hand_ids, aux_features)
            loss = criterion(logits, labels)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        avg_train_loss = running_loss / max(1, total_batches)

        model.eval()
        val_dataset.set_epoch(epoch)
        val_loss_accum = 0.0
        val_accuracy_accum = 0.0
        val_batches = 0
        samples_to_print = args.print_val_samples
        printed_samples = 0
        with torch.no_grad():
            for hand_ids, aux_features, labels in val_loader:
                hand_ids = hand_ids.to(device=device, dtype=torch.long)
                aux_features = aux_features.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.float32)

                logits = model(hand_ids, aux_features)
                loss, acc = compute_metrics(logits, labels)
                val_loss_accum += loss
                val_accuracy_accum += acc
                val_batches += 1

                if printed_samples < samples_to_print:
                    probs = torch.sigmoid(logits).cpu().numpy()
                    hand_cpu = hand_ids.cpu().numpy()
                    aux_cpu = aux_features.cpu().numpy()
                    labels_cpu = labels.cpu().numpy()
                    batch = probs.shape[0]
                    for i in range(batch):
                        if printed_samples >= samples_to_print:
                            break
                        top_predictions = probs[i].argsort()[::-1][:20]
                        legal_actions = np.flatnonzero(labels_cpu[i] > 0.5)
                        print(
                            json.dumps(
                                {
                                    "sample": printed_samples + 1,
                                    "hand": cards_to_str(
                                        hand_cpu[i],
                                        domain,
                                        num_jokers=args.num_jokers,
                                        num_blanks=args.num_blanks,
                                    ),
                                    "rank_counts": aux_cpu[i][:rank_feature_dim].tolist(),
                                    "suit_counts": aux_cpu[i][rank_feature_dim:rank_feature_dim + suit_feature_dim].tolist(),
                                    "joker_count": float(aux_cpu[i][-2]),
                                    "blank_count": float(aux_cpu[i][-1]),
                                    "legal_actions": legal_actions.tolist(),
                                    "top_predictions": [
                                        {"action": int(a), "prob": float(probs[i][a])}
                                        for a in top_predictions.tolist()
                                    ],
                                }
                            ),
                            flush=True,
                        )
                        print("\n")
                        printed_samples += 1

        avg_val_loss = val_loss_accum / max(1, val_batches)
        avg_val_acc = val_accuracy_accum / max(1, val_batches)

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
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "hyperparams": {
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "hidden_sizes": list(args.hidden_sizes),
                    "dropout": args.dropout,
                    "weight_decay": args.weight_decay,
                    "grad_clip": args.grad_clip,
                    "train_batches_per_epoch": args.train_batches_per_epoch,
                    "val_batches": args.val_batches,
                    "deck_size": domain.deck_size,
                    "num_actions": domain.num_actions,
                    "private_card_options": list(private_options),
                    "common_card_options": list(common_options),
                    "num_jokers": args.num_jokers,
                    "num_joker_embeddings": args.num_joker_embeddings,
                    "num_blanks": args.num_blanks,
                    "num_blank_embeddings": args.num_blank_embeddings,
                    "max_hand_cards": int(max_hand_cards),
                    "rank_feature_dim": int(rank_feature_dim),
                    "suit_feature_dim": int(suit_feature_dim),
                    "aux_feature_dim": int(extra_feature_dim),
                },
                "metrics": {
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": avg_val_acc,
                },
            }
            torch.save(artifact, artifact_path)

    print(f"Saved best weights to {artifact_path}", flush=True)


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
    parser.add_argument("--rank-dim", type=int, default=16)
    parser.add_argument("--suit-dim", type=int, default=8)
    parser.add_argument("--num-joker-embeddings", type=int, default=1)
    parser.add_argument("--num-blank-embeddings", type=int, default=1)
    parser.add_argument("--aggregator", choices=("mean", "sum"), default="mean")
    parser.add_argument(
        "--deck-size",
        type=int,
        choices=[24, 32],
        default=24,
        help="Deck size to sample from (24 or 32).",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=parse_int_list,
        default=[256, 256],
        help="Comma-separated hidden layer sizes (e.g., '256,128').",
    )
    parser.add_argument("--no-layer-norm", action="store_true", help="Disable LayerNorm on aggregated embeddings.")
    parser.add_argument(
        "--private-card-options",
        type=parse_card_count_list,
        default=None,
        help="Comma-separated counts or ranges (e.g., '0-12,14') for private cards; defaults to 0..max-private-cards.",
    )
    parser.add_argument(
        "--common-card-options",
        type=parse_card_count_list,
        default=None,
        help="Comma-separated counts or ranges (e.g., '0-12') for common cards; defaults to 0..max-common-cards.",
    )
    parser.add_argument("--max-private-cards", type=int, default=12)
    parser.add_argument("--max-common-cards", type=int, default=0)
    parser.add_argument("--num-jokers", type=int, default=2)
    parser.add_argument("--num-blanks", type=int, default=2)
    parser.add_argument("--print-val-samples", type=int, default=10, help="Print N validation samples with predicted probabilities each epoch.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save-path", type=str, default="artifacts/card_embedding_pretrain.pt")
    parser.add_argument("--early-stop-delta", type=float, default=1e-4)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if isinstance(args.hidden_sizes, list) and args.hidden_sizes and isinstance(args.hidden_sizes[0], str):
        args.hidden_sizes = [int(h) for h in args.hidden_sizes]
    train(args)


if __name__ == "__main__":
    main()
