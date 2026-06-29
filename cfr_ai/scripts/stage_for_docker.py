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
    "p3",              # 3-player raw outputs; staged explicitly (mmap-quantised) below
    "outputs",         # 2-player raw outputs; staged explicitly (mmap-quantised) into a
                       # fresh dest dir below, so stale in-place mmap files don't tag along
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
    if name in EXCLUDE_DIRS:
        return True
    # Skip data/experiment trees that aren't part of the runtime package — only
    # cfr_ai code + outputs/ (quantised in place) belong in the image. This covers
    # training-box cruft (experiments/, v2.1/, exp_*/, scratch_*/, grouped_*/) and
    # the raw 3-player runs (p3, p3_2x — staged separately, mmap-quantised below).
    # outputs/, abstraction/, etc. don't match these prefixes and are kept.
    return name.startswith(("p3", "exp", "v2", "scratch", "grouped"))


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
    # split layout (uncompressed probs_sparse_indices.npy + probs_sparse_values.npy
    # + small strategy_meta.npz + the abs.json companion). The agent's
    # load_strategy_for_agent prefers this layout and mmap's the sparse probs
    # files → ~50 MB peak resident regardless of strategy size.
    # Stage every setup OUT-OF-PLACE: read its compressed strategy.npz from the
    # source and write the mmap-friendly split layout (uncompressed sparse probs +
    # small strategy_meta.npz + the abs.json companion) into a FRESH dest dir. The
    # agent's load_strategy_for_agent prefers this layout and mmap's the sparse
    # probs → ~50 MB peak resident regardless of strategy size. uint8 quantised
    # (strength-neutral vs uint16, H2H 0.5018 over 20k games). Out-of-place means a
    # cluttered source setup dir (stale mmap files, diagnostics) can't tag along.
    print("[stage] converting strategies to mmap-friendly layout...", flush=True)
    from cfr_ai.strategy_io import write_mmap_layout
    outputs = dest / "cfr_ai" / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    n_converted = 0
    for setup_dir in sorted((cfr_src / "outputs").iterdir()):
        if not setup_dir.is_dir() or not (setup_dir / "strategy.npz").exists():
            continue
        dst = outputs / setup_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        write_mmap_layout(str(setup_dir), str(dst), value_bits=8)
        n_converted += 1
    print(f"[stage] converted {n_converted} two-player setups to mmap layout", flush=True)

    # Also stage the 176 directed 3-player setups into the SAME image outputs/ dir
    # as the two-player ones — no name collision (2-part "a_b" vs 3-part "a_b_c"),
    # so the agent's player-count routing + setup-string resolution finds either.
    # write_mmap_layout reads strategy.npz from the source and copies
    # strategy.abs.json into the destination itself. CFR_3P_SUBDIR selects which
    # 3-player run to ship (default "p3"; set "p3_2x" to ship the 2x-iters run).
    p3_src = root / "cfr_ai" / os.environ.get("CFR_3P_SUBDIR", "p3") / "outputs"
    n_p3 = 0
    if p3_src.is_dir():
        for setup_dir in sorted(p3_src.iterdir()):
            if not setup_dir.is_dir() or not (setup_dir / "strategy.npz").exists():
                continue
            dst = outputs / setup_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            write_mmap_layout(str(setup_dir), str(dst), value_bits=8)
            n_p3 += 1
    print(f"[stage] converted {n_p3} three-player setups to mmap layout", flush=True)

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
    EXPECTED = 66 + 176  # 66 two-player + 176 directed three-player setups
    if n_idx != EXPECTED or n_val != EXPECTED or n_meta != EXPECTED:
        print(
            f"[warn] expected {EXPECTED} strategies, got {n_idx} indices / {n_val} values "
            f"/ {n_meta} meta. Continuing, but the image will be incomplete.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
