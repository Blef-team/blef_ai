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

Values and actions are decoded using the deck size reported in each game’s rules
(`rules["deck_size"]`, default 24). The tables below include both 24- and
32-card mappings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

# --- Card metadata ---
CARD_VALUES = {
    24: {
        -2: "-",
        -1: "Joker",
        0: "9",
        1: "10",
        2: "J",
        3: "Q",
        4: "K",
        5: "A",
    },
    32: {
        -2: "-",
        -1: "Joker",
        0: "7",
        1: "8",
        2: "9",
        3: "10",
        4: "J",
        5: "Q",
        6: "K",
        7: "A"
    }
}

CARD_SUITS = {
    -1: "",
    0: "♣",
    1: "♦",
    2: "♥",
    3: "♠",
}

# --- Action names (matches shared/game_utils mapping) ---
ACTION_NAMES = {
    24: {
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
    },
    32: {
        0:  "HC-7",
        1:  "HC-8",
        2:  "HC-9",
        3:  "HC-10",
        4:  "HC-J",
        5:  "HC-Q",
        6:  "HC-K",
        7:  "HC-A",
        8:  "P-7",
        9:  "P-8",
        10: "P-9",
        11: "P-10",
        12: "P-J",
        13: "P-Q",
        14: "P-K",
        15: "P-A",
        16: "2P-8/7",
        17: "2P-9/7",
        18: "2P-9/8",
        19: "2P-10/7",
        20: "2P-10/8",
        21: "2P-10/9",
        22: "2P-J/7",
        23: "2P-J/8",
        24: "2P-J/9",
        25: "2P-J/10",
        26: "2P-Q/7",
        27: "2P-Q/8",
        28: "2P-Q/9",
        29: "2P-Q/10",
        30: "2P-Q/J",
        31: "2P-K/7",
        32: "2P-K/8",
        33: "2P-K/9",
        34: "2P-K/10",
        35: "2P-K/J",
        36: "2P-K/Q",
        37: "2P-A/7",
        38: "2P-A/8",
        39: "2P-A/9",
        40: "2P-A/10",
        41: "2P-A/J",
        42: "2P-A/Q",
        43: "2P-A/K",
        44: "St 7-J",
        45: "St 8-Q",
        46: "St 9-K",
        47: "St 10-A",
        48: "3x7",
        49: "3x8",
        50: "3x9",
        51: "3x10",
        52: "3xJ",
        53: "3xQ",
        54: "3xK",
        55: "3xA",
        56: "FH 7/8",
        57: "FH 7/9",
        58: "FH 7/10",
        59: "FH 7/J",
        60: "FH 7/Q",
        61: "FH 7/K",
        62: "FH 7/A",
        63: "FH 8/7",
        64: "FH 8/9",
        65: "FH 8/10",
        66: "FH 8/J",
        67: "FH 8/Q",
        68: "FH 8/K",
        69: "FH 8/A",
        70: "FH 9/7",
        71: "FH 9/8",
        72: "FH 9/10",
        73: "FH 9/J",
        74: "FH 9/Q",
        75: "FH 9/K",
        76: "FH 9/A",
        77: "FH 10/7",
        78: "FH 10/8",
        79: "FH 10/9",
        80: "FH 10/J",
        81: "FH 10/Q",
        82: "FH 10/K",
        83: "FH 10/A",
        84: "FH J/7",
        85: "FH J/8",
        86: "FH J/9",
        87: "FH J/10",
        88: "FH J/Q",
        89: "FH J/K",
        90: "FH J/A",
        91: "FH Q/7",
        92: "FH Q/8",
        93: "FH Q/9",
        94: "FH Q/10",
        95: "FH Q/J",
        96: "FH Q/K",
        97: "FH Q/A",
        98: "FH K/7",
        99: "FH K/8",
        100: "FH K/9",
        101: "FH K/10",
        102: "FH K/J",
        103: "FH K/Q",
        104: "FH K/A",
        105: "FH A/7",
        106: "FH A/8",
        107: "FH A/9",
        108: "FH A/10",
        109: "FH A/J",
        110: "FH A/Q",
        111: "FH A/K",
        112: "Fl-♣",
        113: "Fl-♦",
        114: "Fl-♥",
        115: "Fl-♠",
        116: "4x7",
        117: "4x8",
        118: "4x9",
        119: "4x10",
        120: "4xJ",
        121: "4xQ",
        122: "4xK",
        123: "4xA",
        124: "StFl 7-J♣",
        125: "StFl 7-J♦",
        126: "StFl 7-J♥",
        127: "StFl 7-J♠",
        128: "StFl 8-Q♣",
        129: "StFl 8-Q♦",
        130: "StFl 8-Q♥",
        131: "StFl 8-Q♠",
        132: "StFl 9-K♣",
        133: "StFl 9-K♦",
        134: "StFl 9-K♥",
        135: "StFl 9-K♠",
        136: "StFl 10-A♣",
        137: "StFl 10-A♦",
        138: "StFl 10-A♥",
        139: "StFl 10-A♠",
        140: "CHECK",
        141: "LOST"
    }
}

def _resolve_deck_size(raw_deck_size) -> int:
    try:
        deck = int(raw_deck_size)
    except Exception:
        deck = 24
    return 32 if deck >= 32 else 24


def to_action_name(action_id: int | str, deck_size: int) -> str:
    try:
        aid = int(action_id)
    except Exception:
        return str(action_id)
    names = ACTION_NAMES.get(deck_size, ACTION_NAMES[24])
    return names.get(aid, str(aid))


def value_to_str(value: int, deck_size: int) -> str:
    table = CARD_VALUES.get(deck_size, CARD_VALUES[24])
    return table.get(value, str(value))


def suit_to_str(suit: int) -> str:
    return CARD_SUITS.get(suit, str(suit))


def format_hand(hand: dict, deck_size: int) -> str:
    nickname = hand.get("nickname")
    cards = [
        f"{value_to_str(c.get('value'), deck_size)}{suit_to_str(c.get('colour'))}"
        for c in hand.get("hand", [])
    ]
    return f"{nickname}: {','.join(cards)}"


def format_common_hand(hand: dict, deck_size: int) -> str:
    if not hand:
        return ""
    cards = [
        f"{value_to_str(c.get('value'), deck_size)}{suit_to_str(c.get('colour'))}"
        for c in hand
    ]
    return f"COMMON: {','.join(cards)}"


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
    deck_size = _resolve_deck_size(data.get("rules", {}).get("deck_size", 24))
    hands = [format_hand(hand, deck_size) for hand in data.get("hands", [])]
    common_hand = format_common_hand(data.get("common_hand", []), deck_size)
    if common_hand:
        hands.extend([common_hand])
    history = [
        f"{event.get('player')}: {to_action_name(event.get('action_id'), deck_size)}"
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
