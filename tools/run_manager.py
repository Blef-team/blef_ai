"""
Lightweight CLI for managing NFSP self-play training runs.

Subcommands:
  start       Launch a run in the background.
  status      Show latest metrics for one run, or summary for all.
  override    Write to a run's control-plane JSON.
  stop        Send SIGTERM to a running run; waits for clean exit.
  list        List all run dirs and their status.
  compare     Print last-row metric snapshots side-by-side across runs.

Run directory convention
------------------------
Each run lives at:

    runs/<YYYYMMDD-HHMMSS>__<experiment-name>/
        ├── checkpoints/
        ├── games/
        ├── games_eval/
        ├── metrics.csv
        ├── control_plane.json
        ├── action_samples.jsonl
        ├── pid          (process id of the trainer; absent if not running)
        └── cmd.txt      (the launch command, for reproducibility)

`pid` is written by `nfsp_run_local.py`. The trainer registers a SIGTERM
handler so `stop` causes a clean checkpoint flush before exit.

Examples
--------

  # Launch a 5M-step run in the background.
  python -m tools.run_manager start \\
      --experiment-name mc-baseline \\
      --total-steps 5000000 --deck-size 24 --max-cards 11

  # Latest metrics:
  python -m tools.run_manager status --experiment-name mc-baseline

  # Override eta mid-run (cooldown logic still applies):
  python -m tools.run_manager override --experiment-name mc-baseline --eta 0.05

  # All runs:
  python -m tools.run_manager list
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple


RUN_DIRNAME_RE = re.compile(r"^(?P<ts>\d{8}-\d{6})(?:__(?P<name>.+))?$")
DEFAULT_RUNS_ROOT = os.environ.get("BLEF_RUNS_DIR", "runs")
NFSP_MODULE = "nfsp_ai.nfsp_run_local"


# --------------------------------------------------------------------------
# Run discovery
# --------------------------------------------------------------------------

def _list_run_dirs(runs_root: str) -> List[Dict]:
    if not os.path.isdir(runs_root):
        return []
    out = []
    for entry in os.listdir(runs_root):
        full = os.path.join(runs_root, entry)
        if not os.path.isdir(full):
            continue
        m = RUN_DIRNAME_RE.match(entry)
        if not m:
            continue
        out.append({
            "name": m.group("name") or "auto",
            "timestamp": m.group("ts"),
            "dir": full,
            "label": entry,
        })
    out.sort(key=lambda r: r["timestamp"], reverse=True)
    return out


def _find_run(runs_root: str, experiment_name: str) -> Optional[Dict]:
    """Return the most recent run dir whose experiment name matches."""
    matches = [r for r in _list_run_dirs(runs_root) if r["name"] == experiment_name]
    return matches[0] if matches else None


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_pid(run_dir: str) -> Optional[int]:
    pid_path = os.path.join(run_dir, "pid")
    try:
        with open(pid_path, "r", encoding="utf-8") as fh:
            return int(fh.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def _run_status(run_dir: str) -> str:
    pid = _read_pid(run_dir)
    if pid is None:
        return "no-pid"
    return "running" if _process_alive(pid) else "stopped"


def _read_latest_metrics(run_dir: str) -> Optional[Dict]:
    """Return the latest metrics row, merging training + eval rows that share
    the same `step` value (the trainer writes one of each per log cycle, with
    different fields populated).
    """
    csv_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        return None
    rows: List[Dict] = []
    try:
        with open(csv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)
    except OSError:
        return None
    if not rows:
        return None
    last_step = rows[-1].get("step")
    same_step = [r for r in rows if r.get("step") == last_step]
    if not same_step:
        return rows[-1]
    merged: Dict[str, str] = dict(same_step[0])
    for r in same_step[1:]:
        for k, v in r.items():
            if v not in (None, "", "nan"):
                merged[k] = v
    return merged


# --------------------------------------------------------------------------
# Subcommand implementations
# --------------------------------------------------------------------------

def cmd_start(args) -> int:
    """Launch nfsp_run_local in the background.

    Pass through any extra flags after `--` (e.g. `--total-steps`, `--deck-size`).
    """
    if not args.experiment_name:
        print("error: --experiment-name is required for start", file=sys.stderr)
        return 2

    pre_existing = _find_run(args.runs_root, args.experiment_name)
    if pre_existing and _run_status(pre_existing["dir"]) == "running":
        print(
            f"refusing to start: '{args.experiment_name}' is already running at {pre_existing['dir']}",
            file=sys.stderr,
        )
        return 3

    # Strip leading '--' separator if caller used `start --experiment-name X -- ...`
    extras = list(args.extra)
    while extras and extras[0] == "--":
        extras.pop(0)

    cmd: List[str] = [
        sys.executable, "-m", NFSP_MODULE,
        "--experiment-name", args.experiment_name,
        "--runs-root", args.runs_root,
    ] + extras

    log_dir = os.path.join(args.runs_root, ".launcher_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{args.experiment_name}-{int(time.time())}.log")
    log_fh = open(log_path, "wb")  # noqa: SIM115 — handed to the child process
    print(f"[start] {args.experiment_name}: {' '.join(shlex.quote(x) for x in cmd)}")
    print(f"[start] launcher log: {log_path}")
    proc = subprocess.Popen(  # noqa: S603 — controlled invocation
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"[start] pid={proc.pid}; the trainer will overwrite this with its own pid file.")
    return 0


def cmd_status(args) -> int:
    runs_root = args.runs_root
    if args.experiment_name:
        run = _find_run(runs_root, args.experiment_name)
        if not run:
            print(f"no run found for '{args.experiment_name}'", file=sys.stderr)
            return 4
        latest = _read_latest_metrics(run["dir"]) or {}
        print(f"{run['label']}  status={_run_status(run['dir'])}")
        if latest:
            keys = ("step", "avg_reward", "win_rate", "q_loss", "sl_loss", "policy_entropy")
            for k in keys:
                if k in latest:
                    print(f"  {k:>16}: {latest[k]}")
        else:
            print("  (no metrics yet)")
        return 0

    runs = _list_run_dirs(runs_root)
    if not runs:
        print(f"no runs under {runs_root}/")
        return 0
    width = max(len(r["label"]) for r in runs)
    print(f"{'run':<{width}}  status     step           avg_reward  win_rate")
    for r in runs:
        latest = _read_latest_metrics(r["dir"]) or {}
        step = latest.get("step", "")
        avg_r = latest.get("avg_reward", "")
        wr = latest.get("win_rate", "")
        print(f"{r['label']:<{width}}  {_run_status(r['dir']):<10} {str(step):<14} {str(avg_r):<11} {str(wr)}")
    return 0


def cmd_override(args) -> int:
    run = _find_run(args.runs_root, args.experiment_name)
    if not run:
        print(f"no run found for '{args.experiment_name}'", file=sys.stderr)
        return 4
    cp_path = os.path.join(run["dir"], "control_plane.json")
    if not os.path.exists(cp_path):
        print(f"no control_plane.json at {cp_path}; the run may not have started writing it yet", file=sys.stderr)
        return 5

    with open(cp_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    overrides = payload.get("overrides")
    if not isinstance(overrides, dict):
        overrides = {}
        payload["overrides"] = overrides

    pairs: Dict[str, Optional[float]] = {}
    for kv in args.set or []:
        if "=" not in kv:
            print(f"error: --set entry must be KEY=VALUE, got {kv!r}", file=sys.stderr)
            return 2
        k, v = kv.split("=", 1)
        if v.lower() in {"none", "null", ""}:
            pairs[k.strip()] = None
            continue
        try:
            pairs[k.strip()] = float(v)
        except ValueError:
            pairs[k.strip()] = v.strip()

    overrides.update(pairs)
    with open(cp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"[override] {cp_path}: {pairs}")
    return 0


def cmd_stop(args) -> int:
    run = _find_run(args.runs_root, args.experiment_name)
    if not run:
        print(f"no run found for '{args.experiment_name}'", file=sys.stderr)
        return 4
    pid = _read_pid(run["dir"])
    if pid is None or not _process_alive(pid):
        print(f"[stop] no live pid; nothing to stop")
        return 0
    print(f"[stop] sending SIGTERM to pid={pid} ({run['label']})")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return 0
    if args.wait_seconds <= 0:
        return 0
    deadline = time.time() + args.wait_seconds
    while time.time() < deadline:
        if not _process_alive(pid):
            print(f"[stop] pid={pid} exited cleanly")
            return 0
        time.sleep(1)
    print(f"[stop] pid={pid} did not exit within {args.wait_seconds}s; you may need SIGKILL", file=sys.stderr)
    return 6


def cmd_list(args) -> int:
    return cmd_status(argparse.Namespace(runs_root=args.runs_root, experiment_name=None))


def cmd_compare(args) -> int:
    names: List[str] = [n.strip() for n in (args.experiments or "").split(",") if n.strip()]
    if not names:
        print("error: --experiments is required (comma-separated)", file=sys.stderr)
        return 2
    rows: List[Tuple[str, Dict]] = []
    for n in names:
        run = _find_run(args.runs_root, n)
        if not run:
            rows.append((n, {}))
            continue
        rows.append((n, _read_latest_metrics(run["dir"]) or {}))
    keys = ("step", "avg_reward", "win_rate", "q_loss", "sl_loss", "policy_entropy")
    width = max(len(n) for n in names)
    header = f"{'experiment':<{width}} | " + " | ".join(f"{k:<14}" for k in keys)
    print(header)
    print("-" * len(header))
    for n, m in rows:
        cells = [str(m.get(k, "")) for k in keys]
        print(f"{n:<{width}} | " + " | ".join(f"{c:<14}" for c in cells))
    return 0


# --------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_start = sub.add_parser("start", help="Launch nfsp_run_local in the background.")
    sp_start.add_argument("--experiment-name", required=True)
    sp_start.add_argument("extra", nargs=argparse.REMAINDER, help="Pass-through args for nfsp_run_local.")
    sp_start.set_defaults(func=cmd_start)

    sp_status = sub.add_parser("status", help="Show metrics for one run, or summary across all.")
    sp_status.add_argument("--experiment-name", default=None)
    sp_status.set_defaults(func=cmd_status)

    sp_override = sub.add_parser("override", help="Write to a run's control_plane.json.")
    sp_override.add_argument("--experiment-name", required=True)
    sp_override.add_argument("--set", action="append", help="KEY=VALUE; repeatable. VALUE='none' clears an override.")
    sp_override.set_defaults(func=cmd_override)

    sp_stop = sub.add_parser("stop", help="SIGTERM the run's pid; wait for clean exit.")
    sp_stop.add_argument("--experiment-name", required=True)
    sp_stop.add_argument("--wait-seconds", type=int, default=60, help="0 = don't wait.")
    sp_stop.set_defaults(func=cmd_stop)

    sp_list = sub.add_parser("list", help="List run dirs.")
    sp_list.set_defaults(func=cmd_list)

    sp_compare = sub.add_parser("compare", help="Compare last-row metrics across runs.")
    sp_compare.add_argument("--experiments", required=True, help="Comma-separated experiment names.")
    sp_compare.set_defaults(func=cmd_compare)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
