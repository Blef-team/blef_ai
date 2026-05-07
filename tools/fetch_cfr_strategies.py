"""
Download CFR strategy CSVs from the production CFR worker Lambdas into
`cfr_ai/outputs/`.

The deployed CFR agent for Blef lives across many `blef-aiagent-cfr-worker-X-Y`
Lambdas (one per (sorted hand-size pair)). Each Lambda's deployment package
ships the corresponding `cfr_ai/outputs/X_Y/` strategy tree inside it. This
script downloads each Lambda's code zip, extracts the `cfr_ai/outputs/X_Y/`
subdirectory, and merges into the local `cfr_ai/outputs/`.

CFR is strongest at low cardinality (early-round hand sizes). The default is
to fetch pairs whose maximum hand size is ≤ 4 — about 13 MB total. Larger
pairs are available with `--max-cards`, at much greater download size.

Run from the repository root:

  python -m tools.fetch_cfr_strategies               # default: max-cards=4
  python -m tools.fetch_cfr_strategies --max-cards 6
  python -m tools.fetch_cfr_strategies --pairs 1-1,2-2,3-3

Requires AWS CLI configured locally (the script shells out to `aws lambda
get-function`). Also writes a top-level `metadata.csv` with `Minimum bet,0`
if one isn't already present — the local CFR agent reads it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from typing import List, Optional, Sequence, Tuple


DEFAULT_OUTPUT_DIR = "cfr_ai/outputs"
DEFAULT_METADATA_PATH = "metadata.csv"
WORKER_PREFIX = "blef-aiagent-cfr-worker-"


def _gen_pairs(max_cards: int) -> List[Tuple[int, int]]:
    """All sorted (a, b) pairs with 1 ≤ a ≤ b ≤ max_cards."""
    return [(a, b) for a in range(1, max_cards + 1) for b in range(a, max_cards + 1)]


def _parse_pairs(spec: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" not in token:
            raise ValueError(f"pair must be 'A-B', got {token!r}")
        a, b = token.split("-", 1)
        a_i, b_i = int(a), int(b)
        if a_i > b_i:
            a_i, b_i = b_i, a_i
        out.append((a_i, b_i))
    return out


def _aws_lambda_code_url(function_name: str) -> Optional[str]:
    proc = subprocess.run(  # noqa: S603,S607 — controlled invocation, args are static
        ["aws", "lambda", "get-function", "--function-name", function_name,
         "--query", "Code.Location", "--output", "text"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    url = proc.stdout.strip()
    return url or None


def _fetch_pair(
    pair: Tuple[int, int],
    output_dir: str,
    *,
    force: bool = False,
) -> Tuple[bool, str]:
    """Returns (ok, message)."""
    a, b = pair
    pair_dir = os.path.join(output_dir, f"{a}_{b}")
    if os.path.exists(pair_dir) and not force:
        return True, f"{a}-{b}: already present, skipping (use --force to redownload)"

    function_name = f"{WORKER_PREFIX}{a}-{b}"
    url = _aws_lambda_code_url(function_name)
    if url is None:
        return False, f"{a}-{b}: aws lambda get-function failed (function may not exist)"

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "lambda.zip")
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(zip_path, "wb") as fh:
                shutil.copyfileobj(resp, fh)
        except Exception as exc:  # noqa: BLE001
            return False, f"{a}-{b}: download failed: {exc!r}"

        try:
            with zipfile.ZipFile(zip_path) as zf:
                prefix = f"cfr_ai/outputs/{a}_{b}/"
                members = [m for m in zf.namelist() if m.startswith(prefix)]
                if not members:
                    return False, f"{a}-{b}: zip did not contain {prefix!r}"
                os.makedirs(output_dir, exist_ok=True)
                # Extract into a staging dir, then move into place atomically.
                staging = os.path.join(tmp, "extract")
                zf.extractall(path=staging, members=members)
                src = os.path.join(staging, "cfr_ai", "outputs", f"{a}_{b}")
                if os.path.exists(pair_dir):
                    shutil.rmtree(pair_dir)
                shutil.move(src, pair_dir)
        except (zipfile.BadZipFile, OSError) as exc:
            return False, f"{a}-{b}: extract failed: {exc!r}"

    file_count = sum(len(files) for _, _, files in os.walk(pair_dir))
    return True, f"{a}-{b}: ok ({file_count} strategy files)"


def _ensure_metadata_csv(path: str) -> None:
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Minimum bet,0\n")
    print(f"[meta] wrote {path} (Minimum bet=0)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    p.add_argument("--max-cards", type=int, default=4, help="Fetch pairs with max(a,b) ≤ N (default: 4).")
    p.add_argument("--pairs", default="", help="Comma-separated explicit pair list (e.g. '1-1,2-3'). Overrides --max-cards.")
    p.add_argument("--force", action="store_true", help="Redownload pairs already present.")
    args = p.parse_args(argv)

    if args.pairs:
        pairs = _parse_pairs(args.pairs)
    else:
        if args.max_cards <= 0:
            print("error: --max-cards must be ≥ 1 (or use --pairs)", file=sys.stderr)
            return 2
        pairs = _gen_pairs(args.max_cards)

    print(f"[fetch] {len(pairs)} pair(s) into {args.output_dir}/")
    n_ok = n_skipped = n_failed = 0
    for pair in pairs:
        ok, msg = _fetch_pair(pair, args.output_dir, force=args.force)
        print(f"  {msg}")
        if ok:
            if "already present" in msg:
                n_skipped += 1
            else:
                n_ok += 1
        else:
            n_failed += 1
    _ensure_metadata_csv(args.metadata_path)
    print(f"[fetch] {n_ok} downloaded, {n_skipped} skipped, {n_failed} failed")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
