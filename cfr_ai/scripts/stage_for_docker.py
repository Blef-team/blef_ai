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

    # Convert each setup's compressed strategy.npz into the mmap-friendly
    # split layout (uncompressed probs_flat.npy + small strategy_meta.npz +
    # sidecar abs.json). The agent's load_strategy_for_agent prefers this
    # layout and mmap's the probs file → ~50 MB peak resident regardless
    # of strategy size.
    print("[stage] converting strategies to mmap-friendly layout...", flush=True)
    from cfr_ai.strategy_io import write_mmap_layout
    outputs = dest / "cfr_ai" / "outputs"
    n_converted = 0
    for setup_dir in sorted(outputs.iterdir()):
        if not setup_dir.is_dir():
            continue
        if not (setup_dir / "strategy.npz").exists():
            continue
        # Write the mmap layout into the SAME setup_dir, then drop the
        # original compressed file (we don't ship it — the agent will
        # only ever read the split layout).
        write_mmap_layout(str(setup_dir), str(setup_dir))
        (setup_dir / "strategy.npz").unlink()
        n_converted += 1
    print(f"[stage] converted {n_converted} setups to mmap layout", flush=True)

    # Quick sanity report.
    n_idx = sum(1 for _ in (dest / "cfr_ai" / "outputs").rglob("probs_sparse_indices.npy"))
    n_val = sum(1 for _ in (dest / "cfr_ai" / "outputs").rglob("probs_sparse_values.npy"))
    n_meta = sum(1 for _ in (dest / "cfr_ai" / "outputs").rglob("strategy_meta.npz"))
    n_abs = sum(1 for _ in (dest / "cfr_ai" / "outputs").rglob("strategy.abs.json"))
    total_bytes = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(
        f"[stage] staged: {n_idx} probs_sparse_indices.npy + {n_val} probs_sparse_values.npy, "
        f"{n_meta} strategy_meta.npz, {n_abs} strategy.abs.json, "
        f"total {total_bytes / 1e6:.1f} MB",
        flush=True,
    )
    if n_idx != 66 or n_val != 66 or n_meta != 66:
        print(
            f"[warn] expected 66 strategies, got {n_idx} indices / {n_val} values / "
            f"{n_meta} meta. Continuing, but the image will be incomplete.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
