"""Snapshot trained CFR strategies into a versioned archive for later head-to-head comparison.

Each archive is a self-contained model folder consumable by
``cfr_ai/analysis/head_to_head.py`` -- it bundles the strategy CSVs from
``cfr_ai/outputs/<setup>/`` together with the ``information_set.py`` and
``history.csv`` snapshots needed to interpret them.

Usage from the project root:

    python -m cfr_ai.archive_tool --tag v0_baseline
    python -m cfr_ai.archive_tool --tag v1_dcfr --setups 1_1 1_2 2_2 --note "DCFR retrain"

``*_diagnostic/`` folders inside ``outputs/`` are skipped by default (large; not
needed for play). Pass ``--include-diagnostics`` to copy them too.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

ARCHIVE_ROOT = Path("cfr_ai") / "archive"
SOURCE_ROOT = Path("cfr_ai")
SNAPSHOT_FILES = ("information_set.py", "history.csv")


def _all_setups() -> List[str]:
    """Return every setup folder name (e.g. '1_1', '2_3') under cfr_ai/outputs/."""
    out = SOURCE_ROOT / "outputs"
    if not out.is_dir():
        return []
    setups = []
    for d in out.iterdir():
        if not d.is_dir():
            continue
        # Setup folders look like '1_1', '2_3'; archive/ and other top-level
        # dirs under outputs/ should not be treated as setups.
        parts = d.name.split("_")
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            setups.append(d.name)
    return sorted(setups, key=lambda s: tuple(int(p) for p in s.split("_")))


def _copy_setup(setup: str, dest_outputs: Path, include_diagnostics: bool) -> None:
    """Copy outputs/<setup>/ into the archive, skipping diagnostic data by default.

    Diagnostic data was previously a `<hand_size>_diagnostic/` folder of
    CSVs; it now lives in a single `diagnostic.npz` next to `strategy.npz`.
    We skip either form unless `include_diagnostics` is set."""
    src = SOURCE_ROOT / "outputs" / setup
    dst = dest_outputs / setup
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            if item.name == "diagnostic.npz" and not include_diagnostics:
                continue
            shutil.copy2(item, dst / item.name)
        elif item.is_dir():
            if item.name.endswith("_diagnostic") and not include_diagnostics:
                continue
            shutil.copytree(item, dst / item.name, dirs_exist_ok=True)


def _ensure_index(readme: Path) -> None:
    if readme.exists():
        return
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        "# CFR strategy archive\n"
        "\n"
        "Snapshotted strategies for head-to-head comparison. Each tag is a\n"
        "self-contained model folder consumable by\n"
        "`cfr_ai/analysis/head_to_head.py` via `--model1 <tag>` / `--model2 <tag>`.\n"
        "\n"
        "Run `python -m cfr_ai.archive_tool --tag <name>` from the project root to\n"
        "snapshot the current `cfr_ai/outputs/` into a new entry.\n"
        "\n"
        "| Tag | Date | Setups | Note |\n"
        "|---|---|---|---|\n",
        encoding="utf-8",
    )


def _format_setups_cell(setups: List[str], archived_all: bool) -> str:
    """Compact rendering for the index: 'all (N)' when archiving the full sweep."""
    if archived_all:
        return f"all ({len(setups)})"
    return ", ".join(setups)


def _append_index_entry(tag: str, setups: List[str], note: str, archived_all: bool) -> None:
    readme = ARCHIVE_ROOT / "README.md"
    _ensure_index(readme)
    safe_note = (note or "").replace("|", "\\|").replace("\n", " ").strip()
    line = (
        f"| `{tag}` | {datetime.now().strftime('%Y-%m-%d')} | "
        f"{_format_setups_cell(setups, archived_all)} | {safe_note} |\n"
    )
    with open(readme, "a", encoding="utf-8") as f:
        f.write(line)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tag", required=True,
                   help="Archive tag; becomes the folder name under cfr_ai/archive/.")
    p.add_argument("--setups", nargs="*", default=None,
                   help="Setup folder names to archive (e.g. 1_1 2_3). Default: all under outputs/.")
    p.add_argument("--note", default="",
                   help="One-line note appended to the archive index.")
    p.add_argument("--include-diagnostics", action="store_true",
                   help="Also copy *_diagnostic/ folders (large; needed only for analysis).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing archive tag.")
    args = p.parse_args(argv)

    if not SOURCE_ROOT.is_dir():
        print(f"Error: must be run from a project root containing {SOURCE_ROOT}/.",
              file=sys.stderr)
        return 1

    if "/" in args.tag or "\\" in args.tag or args.tag.startswith("."):
        print(f"Error: tag '{args.tag}' looks like a path; pick a plain folder name.",
              file=sys.stderr)
        return 1

    archived_all = args.setups is None
    setups = args.setups if args.setups else _all_setups()
    if not setups:
        print("Error: no setups found under cfr_ai/outputs/.", file=sys.stderr)
        return 1

    dest = ARCHIVE_ROOT / args.tag
    if dest.exists():
        if not args.force:
            print(f"Error: archive '{args.tag}' already exists at {dest}. "
                  f"Use --force to overwrite.", file=sys.stderr)
            return 1
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # Snapshot the abstraction/history files so the archive is interpretable
    # even after these files evolve in the working tree.
    for fname in SNAPSHOT_FILES:
        src = SOURCE_ROOT / fname
        if not src.exists():
            print(f"Warning: {src} not found; skipping.", file=sys.stderr)
            continue
        shutil.copy2(src, dest / fname)

    dest_outputs = dest / "outputs"
    archived: List[str] = []
    for s in setups:
        src = SOURCE_ROOT / "outputs" / s
        if not src.is_dir():
            print(f"Warning: setup '{s}' not found under outputs/; skipping.",
                  file=sys.stderr)
            continue
        _copy_setup(s, dest_outputs, args.include_diagnostics)
        archived.append(s)

    if not archived:
        print("Error: nothing was archived.", file=sys.stderr)
        shutil.rmtree(dest)
        return 1

    _append_index_entry(args.tag, archived, args.note, archived_all)
    print(f"Archived {len(archived)} setup(s) to {dest}/:")
    for s in archived:
        print(f"  - {s}")
    print(f"Index: {ARCHIVE_ROOT / 'README.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
