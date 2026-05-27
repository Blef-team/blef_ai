"""Stage the cfr_ai Lambda payload into a directory ready for `docker build`.

Replaces the rsync-based staging step from the nfsp_ai deploy script with a
cross-platform Python equivalent, so the deploy works from any shell that
has python (git-bash on Windows, WSL, Linux, macOS).

Layout produced under `<dest>/`:
    cfr_ai/              - package, minus dev cruft (analysis/, archive/,
                           deployment/, scripts/, __pycache__, diagnostic.npz,
                           metadata.csv, *.md)
    lambda_function.py   - entry point (copied from cfr_ai/)
    Dockerfile           - copied from cfr_ai/deployment/Dockerfile.lambda
    requirements.txt     - copied from cfr_ai/deployment/requirements.txt

Usage:
    python -m cfr_ai.scripts.stage_for_docker <dest_dir>
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


# Patterns we drop from cfr_ai/ on the way in. Names match anywhere in the
# path tree (handled by `_should_skip`).
EXCLUDE_DIRS = {
    "__pycache__",
    "analysis",
    "archive",
    "deployment",
    "scripts",
    "_bench_formats",  # ~1.9 GB of strategy-format benchmark artifacts
}
EXCLUDE_FILE_NAMES = {
    "diagnostic.npz",
    "metadata.csv",
    "SESSION_NOTES.md",
    "README.md",
    "README_Appendix_A.md",
    "README_Appendix_B.md",
    # Tracking CSVs and visualisations not needed at runtime:
    "bench_5M_results.csv",
    "lbr_summary.csv",
    "lbr_summary_old.csv",
    "strategy_format_bench.csv",
    "subgame_h2h.csv",
    "subgame_sweep.csv",
    "summary_of_all_runs.csv",
}
EXCLUDE_FILE_SUFFIXES = (".pyc", ".png")


def _should_skip_dir(name: str) -> bool:
    return name in EXCLUDE_DIRS


def _should_skip_file(name: str) -> bool:
    if name in EXCLUDE_FILE_NAMES:
        return True
    return any(name.endswith(s) for s in EXCLUDE_FILE_SUFFIXES)


def _copy_tree(src: Path, dest: Path) -> None:
    """Like shutil.copytree, but applies our exclusion rules."""
    dest.mkdir(parents=True, exist_ok=True)
    for entry in os.scandir(src):
        if entry.is_dir(follow_symlinks=False):
            if _should_skip_dir(entry.name):
                continue
            _copy_tree(Path(entry.path), dest / entry.name)
        else:
            if _should_skip_file(entry.name):
                continue
            shutil.copy2(entry.path, dest / entry.name)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dest", help="Directory to stage the Docker build context into.")
    p.add_argument(
        "--root",
        default=None,
        help="Repo root (defaults to two levels up from this file).",
    )
    args = p.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parents[2]
    cfr_src = root / "cfr_ai"
    if not cfr_src.is_dir():
        print(f"[error] cfr_ai/ not found at {cfr_src}", file=sys.stderr)
        return 1

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[stage] copying {cfr_src} -> {dest / 'cfr_ai'} (with exclusions)", flush=True)
    _copy_tree(cfr_src, dest / "cfr_ai")

    # Top-level files the Dockerfile expects.
    shutil.copy2(cfr_src / "lambda_function.py", dest / "lambda_function.py")
    shutil.copy2(cfr_src / "deployment" / "Dockerfile.lambda", dest / "Dockerfile")
    shutil.copy2(cfr_src / "deployment" / "requirements.txt", dest / "requirements.txt")

    # Quick sanity report.
    n_npz = sum(1 for _ in (dest / "cfr_ai" / "outputs").rglob("strategy.npz"))
    n_abs = sum(1 for _ in (dest / "cfr_ai" / "outputs").rglob("strategy.abs.json"))
    total_bytes = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(
        f"[stage] staged: {n_npz} strategy.npz, {n_abs} strategy.abs.json, "
        f"total {total_bytes / 1e6:.1f} MB",
        flush=True,
    )
    if n_npz != 66 or n_abs != 66:
        print(
            f"[warn] expected 66 strategies, got {n_npz} npz / {n_abs} abs.json. "
            "Continuing, but the image will be incomplete.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
