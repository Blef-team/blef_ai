"""Parallel CFR retraining + LBR orchestrator for a Hetzner CCX-class box.

Designed for the canonical retrain workflow (16-vCPU dedicated, 64 GB RAM):
  Phase 1: train all 66 setups (with periodic LBR-1 snapshots during training).
  Phase 2: production-quality LBR-1 (writes summary + per-setup metadata).
  Phase 3: LBR-2 (run as many setups as the remaining budget allows).

Per-setup hyperparameters (penalty, min-bet, pruning range) are read from the
EXISTING `metadata.csv` files so we replicate the historical grid faithfully;
nothing about the algorithm itself changes vs. the V2 code already on disk.

Concurrency: keeps `--workers` child processes alive at all times via
`subprocess.Popen`, scheduling the next job from the queue when any slot
frees up. Per-setup-per-phase log files live at
`cfr_ai/outputs/<setup>/<phase>.log`; the current job list and progress
counters are dumped to `cfr_ai/outputs/_progress.json` every few seconds.

Resumable: a job whose expected output already exists is skipped. If the
orchestrator gets SIGTERM'd or the instance reboots, restarting it picks up
where it left off without re-running completed work.

Usage on the Hetzner box (in tmux):
    python -m cfr_ai.scripts.hetzner_run \\
        --workers 16 --iter 5000000 --phases train,lbr1,lbr2

Monitor from outside:
    cat cfr_ai/outputs/_progress.json | python -m json.tool
    tail -f cfr_ai/outputs/<setup>/<phase>.log
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# Default save root is production (`cfr_ai/outputs/`, updates the shared summary
# CSV). `--out-root cfr_ai/v2.1` isolates a retrain so the live V2 models stay
# intact for validation before promotion.
DEFAULT_OUT_ROOT = "cfr_ai"


def _outputs_dir(out_root: str) -> str:
    return os.path.join(out_root, "outputs")


def _progress_path(out_root: str) -> str:
    return os.path.join(_outputs_dir(out_root), "_progress.json")


# ---------------------------------------------------------------------------
# Per-setup config from existing metadata.csv
# ---------------------------------------------------------------------------

from cfr_ai.scripts._setup_configs import SETUP_CONFIGS


def _config_for(setup: Tuple[int, int]) -> Dict[str, str]:
    """Per-setup hyperparams from the baked-in `_setup_configs.py` table
    (the historical values from the v0 production runs). Falls back to safe
    defaults if a setup is missing from the table, but every (x, y) for
    1 <= x <= y <= 11 should be present."""
    if setup in SETUP_CONFIGS:
        return SETUP_CONFIGS[setup]
    return {"penalty": "0.0", "min_bet": "0", "pruning": "-20", "min_regret": "-22"}


# ---------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------

@dataclass
class Job:
    setup: Tuple[int, int]
    phase: str  # "train", "lbr1", "lbr2"
    cmd: List[str]
    log_path: str
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    returncode: Optional[int] = None

    @property
    def setup_label(self) -> str:
        return f"{self.setup[0]},{self.setup[1]}"

    @property
    def cost_estimate(self) -> int:
        """Rough relative cost for queue ordering — sum of hand sizes squared
        is a reasonable proxy for tree size / training cost."""
        return self.setup[0] ** 2 + self.setup[1] ** 2


def _setup_dir(setup: Tuple[int, int], out_root: str) -> str:
    return os.path.join(_outputs_dir(out_root), f"{setup[0]}_{setup[1]}")


def _train_output_exists(setup: Tuple[int, int], out_root: str) -> bool:
    d = _setup_dir(setup, out_root)
    return os.path.exists(os.path.join(d, "strategy.npz")) and \
           os.path.exists(os.path.join(d, "diagnostic.npz"))


def _lbr_done_at_depth(setup: Tuple[int, int], depth_label: str,
                       out_root: str) -> bool:
    """Inspect summary_of_all_runs.csv for a non-empty LBR-<depth_label> expl
    cell. Cheaper than re-parsing metadata.csv blocks."""
    setup_key = f"{setup[0]},{setup[1]}"
    path = os.path.join(_outputs_dir(out_root), "summary_of_all_runs.csv")
    if not os.path.exists(path):
        return False
    col = f"LBR-{depth_label} expl"
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("Setup") == setup_key:
                    return bool(r.get(col, "").strip())
    except Exception:
        pass
    return False


def _build_train_job(setup: Tuple[int, int], iters: int, log_points: int,
                     out_root: str, overrides: Dict[str, Optional[str]]) -> Job:
    # min_bet is ALWAYS per-setup (it defines the legal betting tree). penalty,
    # pruning, and min_regret come from the per-setup config UNLESS overridden
    # (the V2.1 consolidation overrides penalty->0, pruning->-10/-12 for all 66).
    cfg = _config_for(setup)
    penalty = overrides.get("penalty") or cfg["penalty"]
    pruning = overrides.get("pruning") or cfg["pruning"]
    min_regret = overrides.get("min_regret") or cfg["min_regret"]
    cmd = [
        sys.executable, "-m", "cfr_ai.training",
        "--hand-sizes", str(setup[0]), str(setup[1]),
        "--iter", str(iters),
        "--penalty", penalty,
        "--min-bet", cfg["min_bet"],
        "--pruning-range", pruning, min_regret,
        "--log-points", str(log_points),
        # Periodic exploitability OFF for the bulk retrain — the sampling
        # caps at our defaults make each snapshot's LBR as expensive as a
        # full final-LBR run, which would inflate per-setup wall by 3-10×.
        # Convergence judgement falls back to the utility log + the
        # post-training LBR-1 phase. See `--get-exploitability` in
        # cfr_ai/training.py for the per-setup opt-in path.
    ]
    # An isolated root needs the explicit --out-root (training.py copies the
    # supporting files there and leaves the shared summary CSV untouched).
    if os.path.normpath(out_root) != DEFAULT_OUT_ROOT:
        cmd += ["--out-root", out_root]
    log_path = os.path.join(_setup_dir(setup, out_root), "train.log")
    return Job(setup=setup, phase="train", cmd=cmd, log_path=log_path)


def _build_lbr_job(setup: Tuple[int, int], depth: int, out_root: str) -> Job:
    cmd = [
        sys.executable, "-m", "cfr_ai.lbr",
        "--hand-sizes", str(setup[0]), str(setup[1]),
        "--depth", str(depth),
        "--n-belief-samples", "300",
        "--n-lbr-hand-samples", "500",
    ]
    # Production root: write the strategy's metadata + the shared summary CSV.
    # Isolated root: target the variant's own setup dir, leave the summary alone.
    if os.path.normpath(out_root) == DEFAULT_OUT_ROOT:
        cmd += ["--update-summary"]
    else:
        cmd += ["--setup-dir", _setup_dir(setup, out_root)]
    log_path = os.path.join(_setup_dir(setup, out_root), f"lbr{depth}.log")
    return Job(setup=setup, phase=f"lbr{depth}", cmd=cmd, log_path=log_path)


def _all_setups() -> List[Tuple[int, int]]:
    return [(x, y) for x in range(1, 12) for y in range(x, 12)]


def _build_phase_queue(phase: str, iters: int, log_points: int,
                       out_root: str,
                       overrides: Dict[str, Optional[str]]) -> List[Job]:
    """Return jobs for `phase` in the right order.

    Train phase: biggest-first — the longest training runs need to start
    immediately so they fully overlap with the rest, otherwise they tail-end
    the wall time.

    LBR phases: smallest-first — LBR cost grows EXPLOSIVELY with setup size
    (big-setup LBR-1 can take 20+ hours each). Starting small means we ship
    most of the matrix's worth of LBR data within a few hours, then can
    decide whether the budget allows running the expensive big setups.
    """
    jobs: List[Job] = []
    for setup in _all_setups():
        if phase == "train":
            if _train_output_exists(setup, out_root):
                continue  # resume: skip if strategy + diagnostic already present
            jobs.append(_build_train_job(setup, iters, log_points,
                                         out_root, overrides))
        elif phase == "lbr1":
            if _lbr_done_at_depth(setup, "1", out_root):
                continue
            jobs.append(_build_lbr_job(setup, depth=1, out_root=out_root))
        elif phase == "lbr2":
            if _lbr_done_at_depth(setup, "2", out_root):
                continue
            jobs.append(_build_lbr_job(setup, depth=2, out_root=out_root))
        else:
            raise ValueError(f"unknown phase: {phase}")
    reverse = (phase == "train")  # biggest-first only for train
    jobs.sort(key=lambda j: j.cost_estimate, reverse=reverse)
    return jobs


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

@dataclass
class Runner:
    workers: int
    progress_path: str = _progress_path(DEFAULT_OUT_ROOT)
    progress_every_s: float = 5.0

    completed: List[Job] = field(default_factory=list)
    running: List[Tuple[Job, subprocess.Popen]] = field(default_factory=list)
    failed: List[Job] = field(default_factory=list)
    queue: List[Job] = field(default_factory=list)
    last_progress_write: float = 0.0
    shutdown: bool = False

    def _start_one(self, job: Job) -> None:
        os.makedirs(os.path.dirname(job.log_path), exist_ok=True)
        log_f = open(job.log_path, "a", buffering=1)
        log_f.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} START "
                    f"{job.phase} {job.setup_label}: {' '.join(job.cmd)} =====\n")
        log_f.flush()
        proc = subprocess.Popen(
            job.cmd, stdout=log_f, stderr=subprocess.STDOUT,
            close_fds=True,
        )
        job.started_at = time.time()
        self.running.append((job, proc))
        print(f"[start] {job.phase} {job.setup_label} (pid={proc.pid})", flush=True)

    def _poll_running(self) -> None:
        still_running: List[Tuple[Job, subprocess.Popen]] = []
        for job, proc in self.running:
            rc = proc.poll()
            if rc is None:
                still_running.append((job, proc))
                continue
            job.finished_at = time.time()
            job.returncode = rc
            duration = (job.finished_at - (job.started_at or job.finished_at))
            if rc == 0:
                self.completed.append(job)
                print(f"[done ] {job.phase} {job.setup_label} "
                      f"in {duration:.0f}s", flush=True)
            else:
                self.failed.append(job)
                print(f"[FAIL ] {job.phase} {job.setup_label} "
                      f"rc={rc} after {duration:.0f}s — see {job.log_path}",
                      flush=True)
        self.running = still_running

    def _write_progress(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self.last_progress_write) < self.progress_every_s:
            return
        self.last_progress_write = now
        snapshot = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "running": [
                {"setup": j.setup_label, "phase": j.phase,
                 "pid": p.pid,
                 "elapsed_s": round(now - (j.started_at or now), 1)}
                for j, p in self.running
            ],
            "queue_remaining": len(self.queue),
            "completed_count": len(self.completed),
            "failed_count": len(self.failed),
            "failed": [
                {"setup": j.setup_label, "phase": j.phase,
                 "rc": j.returncode}
                for j in self.failed
            ],
        }
        os.makedirs(os.path.dirname(self.progress_path), exist_ok=True)
        tmp = self.progress_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        os.replace(tmp, self.progress_path)

    def run_phase(self, jobs: List[Job], phase_label: str) -> None:
        if not jobs:
            print(f"[phase] {phase_label}: nothing to do", flush=True)
            return
        print(f"[phase] {phase_label}: {len(jobs)} jobs queued", flush=True)
        self.queue = list(jobs)
        while self.queue or self.running:
            if self.shutdown:
                self._terminate_running()
                break
            # Fill empty slots
            while self.queue and len(self.running) < self.workers:
                self._start_one(self.queue.pop(0))
            # Poll for completions
            self._poll_running()
            self._write_progress()
            time.sleep(1.0)
        self._write_progress(force=True)

    def _terminate_running(self) -> None:
        print(f"[shutdown] terminating {len(self.running)} running jobs", flush=True)
        for job, proc in self.running:
            try:
                proc.terminate()
            except Exception:
                pass
        for job, proc in self.running:
            try:
                proc.wait(timeout=30)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=16,
                    help="Parallel processes (default 16 — match CCX43 vCPUs).")
    # Accepted-and-ignored: an earlier build had a --max-heavy RAM scheduler
    # (removed — the OOM was a save-prep spike, fixed in trainer.py, not
    # concurrency). A systemd unit launched with the old flag would otherwise
    # fail argparse on restart and could thrash; tolerate it as a no-op.
    ap.add_argument("--max-heavy", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--iter", type=int, default=5_000_000,
                    help="MCCFR iterations per setup (default 5M).")
    ap.add_argument("--log-points", type=int, default=20)
    ap.add_argument("--phases", type=str, default="train,lbr1,lbr2",
                    help="Comma-separated phases to run, in order.")
    ap.add_argument("--out-root", type=str, default=DEFAULT_OUT_ROOT,
                    help="Save root. Default 'cfr_ai' = production "
                         "(cfr_ai/outputs/, updates the shared summary CSV). "
                         "Pass e.g. 'cfr_ai/v2.1' to isolate a retrain: outputs "
                         "land under <root>/outputs/, the summary CSV is left "
                         "untouched, and the live V2 models stay intact for "
                         "validation before promotion.")
    ap.add_argument("--penalty", type=str, default=None,
                    help="Override the per-setup penalty for ALL setups "
                         "(e.g. '0.0' for the V2.1 consolidation). Default: "
                         "use each setup's value from _setup_configs.")
    ap.add_argument("--pruning", type=str, default=None,
                    help="Override the per-setup pruning threshold for ALL "
                         "setups (e.g. '-10' for V2.1). Pair with --min-regret.")
    ap.add_argument("--min-regret", type=str, default=None,
                    help="Override the per-setup minimum regret for ALL setups "
                         "(e.g. '-12' for V2.1).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the per-setup train commands and exit.")
    args = ap.parse_args()

    out_root = args.out_root
    overrides: Dict[str, Optional[str]] = {
        "penalty": args.penalty,
        "pruning": args.pruning,
        "min_regret": args.min_regret,
    }

    if args.dry_run:
        phases = [p.strip() for p in args.phases.split(",") if p.strip()]
        for phase in phases:
            jobs = _build_phase_queue(phase, args.iter, args.log_points,
                                      out_root, overrides)
            print(f"[dry-run] phase={phase}: {len(jobs)} jobs", flush=True)
            for j in jobs:
                print(f"  {j.setup_label:>6} -> {' '.join(j.cmd)}", flush=True)
        return 0

    runner = Runner(workers=args.workers,
                    progress_path=_progress_path(out_root))

    def _sigterm_handler(signum, frame):
        runner.shutdown = True
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    for phase in phases:
        if runner.shutdown:
            break
        jobs = _build_phase_queue(phase, args.iter, args.log_points,
                                  out_root, overrides)
        runner.run_phase(jobs, phase_label=phase)

    print(f"\n[summary] completed={len(runner.completed)} "
          f"failed={len(runner.failed)}", flush=True)
    if runner.failed:
        print("FAILED jobs:", flush=True)
        for j in runner.failed:
            print(f"  - {j.phase} {j.setup_label} rc={j.returncode}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
