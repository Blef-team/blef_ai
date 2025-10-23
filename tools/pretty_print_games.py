#!/usr/bin/env python3
"""Pretty-print saved Blef games.

Given a directory of saved game snapshots (24 card-deck games), this script emits
one easy-to-read line per file so you can skim hands and action history without
opening each JSON. It recognises both `.json` files and the extension-less dumps
produced by the NFSP runner.

Examples:

    python tools/pretty_print_games.py games_20260101010101_eval/
    find games_20260101010101_eval -type f | xargs python tools/pretty_print_games.py

The output looks like:

    0: 9♠,K♥,A♠  1: J♠  ///  0: HC-J  1: CHECK  1: LOST

Values and actions are decoded using the 24-card deck; for 32-card variants,
update the tables below as needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

# --- Card metadata (24-card deck: ranks 9..A) ---
CARD_VALUES = {
    0: "9",
    1: "10",
    2: "J",
    3: "Q",
    4: "K",
    5: "A",
}

CARD_SUITS = {
    0: "♣",
    1: "♦",
    2: "♥",
    3: "♠",
}

# --- Action names (matches shared/game_utils mapping for deck size 24) ---
ACTION_NAMES = {
    0:  "HC-9",
    1:  "HC-10",
    2:  "HC-J",
    3:  "HC-Q",
    4:  "HC-K",
    5:  "HC-A",
    6:  "P-9",
    7:  "P-10",
    8:  "P-J",
    9:  "P-Q",
    10: "P-K",
    11: "P-A",
    12: "2P-10/9",
    13: "2P-J/9",
    14: "2P-J/10",
    15: "2P-Q/9",
    16: "2P-Q/10",
    17: "2P-Q/J",
    18: "2P-K/9",
    19: "2P-K/10",
    20: "2P-K/J",
    21: "2P-K/Q",
    22: "2P-A/9",
    23: "2P-A/10",
    24: "2P-A/J",
    25: "2P-A/Q",
    26: "2P-A/K",
    27: "SmlSt 9-K",
    28: "BigSt 10-A",
    29: "GSt 9-A",
    30: "3x9",
    31: "3x10",
    32: "3xJ",
    33: "3xQ",
    34: "3xK",
    35: "3xA",
    36: "FH 9/10",
    37: "FH 9/J",
    38: "FH 9/Q",
    39: "FH 9/K",
    40: "FH 9/A",
    41: "FH 10/9",
    42: "FH 10/J",
    43: "FH 10/Q",
    44: "FH 10/K",
    45: "FH 10/A",
    46: "FH J/9",
    47: "FH J/10",
    48: "FH J/Q",
    49: "FH J/K",
    50: "FH J/A",
    51: "FH Q/9",
    52: "FH Q/10",
    53: "FH Q/J",
    54: "FH Q/K",
    55: "FH Q/A",
    56: "FH K/9",
    57: "FH K/10",
    58: "FH K/J",
    59: "FH K/Q",
    60: "FH K/A",
    61: "FH A/9",
    62: "FH A/10",
    63: "FH A/J",
    64: "FH A/Q",
    65: "FH A/K",
    66: "Fl-♣",
    67: "Fl-♦",
    68: "Fl-♥",
    69: "Fl-♠",
    70: "4x9",
    71: "4x10",
    72: "4xJ",
    73: "4xQ",
    74: "4xK",
    75: "4xA",
    76: "SSFl 9-K♣",
    77: "SSFl 9-K♦",
    78: "SSFl 9-K♥",
    79: "SSFl 9-K♠",
    80: "BSFl 10-A♣",
    81: "BSFl 10-A♦",
    82: "BSFl 10-A♥",
    83: "BSFl 10-A♠",
    84: "GSFl 9-A♣",
    85: "GSFl 9-A♦",
    86: "GSFl 9-A♥",
    87: "GSFl 9-A♠",
    88: "CHECK",
    89: "LOST",
}


def to_action_name(action_id: int | str) -> str:
    try:
        aid = int(action_id)
    except Exception:
        return str(action_id)
    return ACTION_NAMES.get(aid, str(aid))


def value_to_str(value: int) -> str:
    return CARD_VALUES.get(value, str(value))


def suit_to_str(suit: int) -> str:
    return CARD_SUITS.get(suit, str(suit))


def format_hand(hand: dict) -> str:
    nickname = hand.get("nickname")
    cards = [
        f"{value_to_str(c.get('value'))}{suit_to_str(c.get('colour'))}"
        for c in hand.get("hand", [])
    ]
    return f"{nickname}: {','.join(cards)}"


def load_game(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"[warn] failed to read {path}: {exc}", file=sys.stderr)
        return None


def iter_game_files(paths: Iterable[Path]) -> Iterable[Path]:
    for entry in paths:
        if not entry.exists():
            print(f"[warn] path not found: {entry}", file=sys.stderr)
            continue
        if entry.is_dir():
            yield from iter_game_files(entry.iterdir())
        elif entry.is_file():
            suffix = entry.suffix.lower()
            if suffix in {"", ".json"}:
                yield entry


def pretty_print(path: Path) -> None:
    data = load_game(path)
    if not data:
        return

    hands = [format_hand(hand) for hand in data.get("hands", [])]
    history = [
        f"{event.get('player')}: {to_action_name(event.get('action_id'))}"
        for event in data.get("history", [])
    ]
    if not history:
        return

    print("     ".join(hands + ["///"] + history))


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretty-print Blef game JSON files for quick inspection.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more files/directories containing saved games.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Show at most this many most-recent games (default: 25).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(list(argv) if argv is not None else sys.argv[1:])
    paths = [Path(p) for p in args.paths]

    files = list(iter_game_files(paths))
    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)

    found = False
    for game_path in files[: max(0, args.limit)]:
        pretty_print(game_path)
        found = True

    if not found:
        print("No JSON games found.", file=sys.stderr)


if __name__ == "__main__":
    main()
