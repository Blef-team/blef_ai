"""Parameter-experiment orchestrator for the Blef CFR AI (per-setup DAG).

Runs hyperparameter sweeps over a fixed cross-section of setups, isolating
each variant's artifacts under `cfr_ai/experiments/<variant_tag>/` and
ranking variants against the current production baseline
(`cfr_ai/outputs/<setup>/`) via head-to-head.

Unlike `hetzner_run.py` (phase-based: all train, then all LBR), this is a
per-(variant, setup) DAG:

    train ──┬─→ h2h vs baseline  (10k MC deals; primary ranking signal)
            └─→ lbr battery      (only where affordable; per-setup depths
                                  from LBR_DEPTHS — inside-view)

`h2h` and the `lbr` jobs become eligible the moment their `train` completes, and
are independent of each other. The scheduler keeps `--workers` slots full
with whatever jobs are eligible across all pipelines, so small-setup
trainings finish early and free cores for h2h/lbr while big-setup
trainings are still running.

The BASELINE is not retrained — it already exists in `cfr_ai/outputs/`
from the V2 retrain. We train only the new variants and compare each to
the baseline.

Metrics are parsed from each job's stdout (captured in the per-job log)
and written to `cfr_ai/experiments/<variant>/experiment_summary.csv` plus
a machine-readable `cfr_ai/experiments/_results.json` (also used for
resume — a job whose result is already recorded, or whose training output
already exists, is skipped on restart).

Three independent sweeps (pick with --sweep):
  * pruning (default): vary the pruning range at the baseline 5M iters.
  * iters:             vary the iteration count at the baseline -20/-22
                       pruning. Does NOT depend on the pruning sweep's winner.
  * penalty:           override the per-setup penalty (lower it) at the
                       baseline iters + pruning. Each variant only applies to
                       setups whose prod penalty it lowers, so no-ops never run.
All H2H each variant against the V2 baseline (5M, -20/-22, prod penalties) in
cfr_ai/outputs/.

Usage on the Hetzner box (in tmux), AFTER LBR-1 has been stopped:
    python -m cfr_ai.scripts.experiments_run --workers 8                 # Sweep 1
    python -m cfr_ai.scripts.experiments_run --workers 8 --sweep iters   # Sweep 2
    python -m cfr_ai.scripts.experiments_run --workers 8 --sweep penalty # Sweep 3

Notes
-----
* Periodic in-training LBR (`--get-exploitability`) is OFF by default and
  gated behind `--train-exploitability`. Phase 2 showed a single LBR-1 on
  a sum>=10 setup costs hours, so 5 snapshots per training run would be
  catastrophic on the bigger cross-section setups. The post-train `lbr`
  stage (small setups only) plus the utility log are the convergence
  signal instead.
* RAM: a big-setup training peaks ~2.5-3.5 GB. With `--workers 8` and the
  three big setups (9,11 / 5,11 / 3,9) potentially training concurrently
  across variants, peak can approach the box's 30 GB. Drop to `--workers
  6` if you see memory pressure.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from cfr_ai.scripts._setup_configs import SETUP_CONFIGS


EXPERIMENTS_ROOT = os.path.join("cfr_ai", "experiments")
PROGRESS_PATH = os.path.join(EXPERIMENTS_ROOT, "_progress.json")
RESULTS_PATH = os.path.join(EXPERIMENTS_ROOT, "_results.json")
BASELINE_OUTPUTS = os.path.join("cfr_ai", "outputs")


# ---------------------------------------------------------------------------
# Experiment definition
# ---------------------------------------------------------------------------

# Cross-section that saturates the 8 vCPUs (sums 4..20).
CROSS_SECTION: List[Tuple[int, int]] = [
    (2, 2), (3, 3), (4, 5), (5, 7), (6, 6), (3, 9), (5, 11), (9, 11),
]

# A variant specifies the axes a training run can move along: iteration count,
# pruning range, and (optionally) a penalty override. Each sweep below fixes
# the others and varies one. `penalty=None` (the default) means "use the
# setup's per-setup prod penalty from SETUP_CONFIGS"; a penalty sweep sets it
# to a fixed value that overrides that. In all sweeps, h2h compares against the
# V2 baseline (5M iters, pruning -20/-22, prod penalties) already sitting in
# cfr_ai/outputs/, which is therefore never a variant here.
Variant = namedtuple("Variant", ["tag", "iters", "pruning", "min_regret", "penalty"],
                     defaults=(None,))

BASELINE_ITERS = 5_000_000
BASELINE_PRUNING = (-20, -22)   # what all 66 setups were (re-)trained with

# Sweep 1 — pruning range, at the baseline 5M iters. Varies pruning only.
PRUNING_VARIANTS: List[Variant] = [
    Variant("prune-10", BASELINE_ITERS, -10, -12),
    Variant("prune-5",  BASELINE_ITERS, -5,  -7),
    Variant("prune-2",  BASELINE_ITERS, -2,  -4),
]

# Sweep 2 — iteration count, at the baseline pruning. INDEPENDENT of Sweep 1
# (does NOT use its winner). 5M = the existing V2 baseline, so only 10M is
# trained up front. `iters-3M` is a CONDITIONAL follow-up — add it here only
# after the 10M H2H shows 5M is already converged (then 3M tests whether
# training cheaper hurts).
ITERS_VARIANTS: List[Variant] = [
    Variant("iters-10M", 10_000_000, *BASELINE_PRUNING),
    # Variant("iters-3M", 3_000_000, *BASELINE_PRUNING),  # conditional; see above
]

# Sweep 3 — penalty, at the baseline iters + pruning. Each variant OVERRIDES
# the per-setup prod penalty. A variant only applies where it LOWERS a setup's
# prod penalty (we're testing reduced regularization): penalty-0.05 hits the
# prod-0.1 setups, penalty-0 hits every penalty>0 setup. (variant, setup) pairs
# where the override equals or exceeds the prod penalty are skipped in
# build_graph, so we never train a model identical to / heavier than baseline.
# All cross-section penalty>0 setups have sum >= 9 (> the LBR cap 6), so this
# sweep is H2H-only — no LBR jobs.
PENALTY_VARIANTS: List[Variant] = [
    Variant("penalty-0",    BASELINE_ITERS, *BASELINE_PRUNING, penalty="0.0"),
    Variant("penalty-0.05", BASELINE_ITERS, *BASELINE_PRUNING, penalty="0.05"),
]

SWEEPS = {"pruning": PRUNING_VARIANTS, "iters": ITERS_VARIANTS,
          "penalty": PENALTY_VARIANTS}


# Post-train LBR battery, per setup. LBR is a LOWER bound on exploitability
# and tightens (rises) with depth; cost is ~88^depth per LBR-active node, so
# depth >1 is only affordable on round-1 setups. These depths match the
# manually validated battery (Sweep 2) that showed the 10M model strictly
# less exploitable than 5M at every depth, the gap widening with depth. Any
# other LBR-eligible setup (sum <= --lbr-max-sum) defaults to LBR-1 only.
# NB: only sum<=6 setups clear --lbr-max-sum's default, and those are exactly
# (2,2)/(3,3) — so a sweep whose cross-section excludes the small setups
# (e.g. a 0-penalty sweep, where penalty>0 setups all have sum>=7) runs no
# LBR at all, deep or shallow.
LBR_DEPTHS: Dict[Tuple[int, int], Tuple[int, ...]] = {
    (2, 2): (1, 2, 3),
    (3, 3): (1, 2),
}
DEFAULT_LBR_DEPTHS: Tuple[int, ...] = (1,)


def _lbr_depths_for(setup: Tuple[int, int]) -> Tuple[int, ...]:
    return LBR_DEPTHS.get(tuple(sorted(setup)), DEFAULT_LBR_DEPTHS)


def _setup_label(setup: Tuple[int, int]) -> str:
    return f"{setup[0]},{setup[1]}"


def _setup_dirname(setup: Tuple[int, int]) -> str:
    return f"{setup[0]}_{setup[1]}"


def _variant_root(variant: str) -> str:
    return os.path.join(EXPERIMENTS_ROOT, variant)


def _variant_setup_dir(variant: str, setup: Tuple[int, int]) -> str:
    return os.path.join(_variant_root(variant), "outputs", _setup_dirname(setup))


# ---------------------------------------------------------------------------
# Job model + DAG
# ---------------------------------------------------------------------------

@dataclass
class Job:
    job_id: str           # "<variant>/<setup>/<kind>"
    kind: str             # "train" | "h2h" | "lbr<depth>" (e.g. lbr1/lbr2/lbr3)
    variant: str
    setup: Tuple[int, int]
    cmd: List[str]
    log_path: str
    deps: List[str] = field(default_factory=list)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    returncode: Optional[int] = None
    attempts: int = 0     # incremented on each failed run; bounded by max_attempts


# Parsers: pull the scalar metric(s) out of a finished job's log text.
_H2H_RE = re.compile(
    r"advantage when player with (\d+) cards starts:\s*([+-]?\d*\.\d+)")
_LBR_RE = re.compile(
    r"starting_player=(\d+): exploitability=([+-]?\d*\.\d+)%"
    r"(?:\s*(?:\+/-|±)\s*([\d.]+)pp)?")  # accept ASCII "+/-" and legacy "±"


def _parse_h2h(log_text: str) -> Dict[str, float]:
    """Return {'h2h_p0': adv, 'h2h_p1': adv?} — advantage of variant (B) vs
    baseline (A), in [-1, 1]. P1 present only for asymmetric setups."""
    out: Dict[str, float] = {}
    for m in _H2H_RE.finditer(log_text):
        # Two matches at most; first is the smaller-hand starter (P0 seat).
        key = "h2h_p0" if "h2h_p0" not in out else "h2h_p1"
        out[key] = float(m.group(2))
    return out


def _parse_lbr_value(log_text: str) -> str:
    """Return the '<sp0>[ | <sp1>]' exploitability string (percent, with SE
    if sampled), or '' if none found. Depth-agnostic — the caller keys it by
    the job's LBR depth (lbr<depth>_expl)."""
    parts = []
    for m in _LBR_RE.finditer(log_text):
        expl = m.group(2)
        se = m.group(3)
        parts.append(f"{expl}%" + (f" +/- {se}pp" if se else ""))  # ASCII, no mojibake
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Job-graph construction
# ---------------------------------------------------------------------------

def _config_for(setup: Tuple[int, int]) -> Dict[str, str]:
    if setup in SETUP_CONFIGS:
        return SETUP_CONFIGS[setup]
    return {"penalty": "0.0", "min_bet": "0", "pruning": "-20", "min_regret": "-22"}


def _train_done(variant: str, setup: Tuple[int, int]) -> bool:
    return os.path.exists(os.path.join(_variant_setup_dir(variant, setup), "strategy.npz"))


def _build_train_job(v: "Variant", setup: Tuple[int, int],
                     log_points: int, train_expl: bool) -> Job:
    # min_bet is always per-setup. penalty is per-setup UNLESS the variant
    # overrides it (penalty sweep); iters and pruning come from the Variant.
    cfg = _config_for(setup)
    penalty = v.penalty if v.penalty is not None else cfg["penalty"]
    cmd = [
        sys.executable, "-m", "cfr_ai.training",
        "--hand-sizes", str(setup[0]), str(setup[1]),
        "--iter", str(v.iters),
        "--penalty", penalty,
        "--min-bet", cfg["min_bet"],
        "--pruning-range", str(v.pruning), str(v.min_regret),
        "--log-points", str(log_points),
        "--out-root", _variant_root(v.tag),
    ]
    if train_expl:
        cmd += ["--get-exploitability", "--exploitability-points", "5",
                "--exploitability-n-belief", "300",
                "--exploitability-n-lbr-hand", "500"]
    log_path = os.path.join(_variant_setup_dir(v.tag, setup), "train.log")
    return Job(job_id=f"{v.tag}/{_setup_dirname(setup)}/train", kind="train",
               variant=v.tag, setup=setup, cmd=cmd, log_path=log_path)


def _build_h2h_job(variant: str, setup: Tuple[int, int], num_deals: int,
                   train_id: str) -> Job:
    cmd = [
        sys.executable, "-m", "cfr_ai.analysis.head_to_head",
        "--hand-sizes", str(setup[0]), str(setup[1]),
        "--monte-carlo", "--num-deals", str(num_deals),
        "--model1", "current",                       # baseline = working tree (A)
        "--model2-folder", _variant_root(variant),    # variant (B)
    ]
    log_path = os.path.join(_variant_setup_dir(variant, setup), "h2h.log")
    return Job(job_id=f"{variant}/{_setup_dirname(setup)}/h2h", kind="h2h",
               variant=variant, setup=setup, cmd=cmd, log_path=log_path,
               deps=[train_id])


def _build_lbr_job(variant: str, setup: Tuple[int, int], depth: int,
                   train_id: str) -> Job:
    kind = f"lbr{depth}"
    cmd = [
        sys.executable, "-m", "cfr_ai.lbr",
        "--hand-sizes", str(setup[0]), str(setup[1]),
        "--depth", str(depth),
        "--n-belief-samples", "300",
        "--n-lbr-hand-samples", "500",
        "--setup-dir", _variant_setup_dir(variant, setup),
        # NB: NO --update-summary — that writes the PRODUCTION summary,
        # ignoring --setup-dir. We parse stdout into the variant's own CSV.
    ]
    log_path = os.path.join(_variant_setup_dir(variant, setup), f"{kind}.log")
    return Job(job_id=f"{variant}/{_setup_dirname(setup)}/{kind}", kind=kind,
               variant=variant, setup=setup, cmd=cmd, log_path=log_path,
               deps=[train_id])


def build_graph(variants: List["Variant"], setups, log_points, num_deals,
                lbr_max_sum, train_expl,
                results: Dict[str, dict]) -> Tuple[Dict[str, Job], set]:
    """Return ({job_id: Job}, preset_done). A job whose output/result already
    exists is omitted and its id added to `preset_done` so dependents are
    immediately eligible on resume. Each Variant carries its own iters +
    pruning, so this is sweep-agnostic."""
    jobs: Dict[str, Job] = {}
    preset_done = set()
    for v in variants:
        for setup in setups:
            # A penalty-override variant only applies where it LOWERS the
            # setup's prod penalty (testing reduced regularization). Skip
            # no-ops (override == prod) and raises (override > prod) so we
            # never train a model identical to / heavier than the baseline.
            if v.penalty is not None and \
                    float(v.penalty) >= float(_config_for(setup)["penalty"]):
                continue
            sd = _setup_dirname(setup)
            train_id = f"{v.tag}/{sd}/train"
            h2h_id = f"{v.tag}/{sd}/h2h"
            rec = results.get(f"{v.tag}/{sd}", {})

            if _train_done(v.tag, setup):
                preset_done.add(train_id)
            else:
                jobs[train_id] = _build_train_job(v, setup, log_points, train_expl)

            if "h2h_p0" in rec:
                preset_done.add(h2h_id)
            else:
                jobs[h2h_id] = _build_h2h_job(v.tag, setup, num_deals, train_id)

            if sum(setup) <= lbr_max_sum:
                for depth in _lbr_depths_for(setup):
                    lbr_id = f"{v.tag}/{sd}/lbr{depth}"
                    if f"lbr{depth}_expl" in rec:
                        preset_done.add(lbr_id)
                    else:
                        jobs[lbr_id] = _build_lbr_job(v.tag, setup, depth, train_id)
    return jobs, preset_done


# ---------------------------------------------------------------------------
# Results store
# ---------------------------------------------------------------------------

def _load_results() -> Dict[str, dict]:
    if not os.path.exists(RESULTS_PATH):
        return {}
    try:
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_results(results: Dict[str, dict]) -> None:
    os.makedirs(EXPERIMENTS_ROOT, exist_ok=True)
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp, RESULTS_PATH)


def _write_variant_summaries(variants: List["Variant"], setups,
                             results: Dict[str, dict]) -> None:
    """One experiment_summary.csv per variant, columns aggregating h2h + lbr."""
    import csv
    for v in variants:
        variant = v.tag
        rows = []
        for setup in setups:
            key = f"{variant}/{_setup_dirname(setup)}"
            rec = results.get(key, {})
            rows.append({
                "Setup": _setup_label(setup),
                "H2H P0 (variant-baseline)": rec.get("h2h_p0", ""),
                "H2H P1 (variant-baseline)": rec.get("h2h_p1", ""),
                "LBR-1 expl": rec.get("lbr1_expl", ""),
                "LBR-2 expl": rec.get("lbr2_expl", ""),
                "LBR-3 expl": rec.get("lbr3_expl", ""),
            })
        out_dir = _variant_root(variant)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "experiment_summary.csv")
        tmp = path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "Setup", "H2H P0 (variant-baseline)",
                "H2H P1 (variant-baseline)",
                "LBR-1 expl", "LBR-2 expl", "LBR-3 expl"])
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

@dataclass
class Scheduler:
    jobs: Dict[str, Job]
    results: Dict[str, dict]
    workers: int
    max_attempts: int = 2     # a failed job is re-queued until it hits this
    progress_every_s: float = 5.0

    done: set = field(default_factory=set)        # real jobs completed this run
    failed: set = field(default_factory=set)
    skipped: set = field(default_factory=set)
    preset: set = field(default_factory=set)      # deps already satisfied on resume
                                                  # (NOT in self.jobs — eligibility only)
    running: List[Tuple[Job, subprocess.Popen]] = field(default_factory=list)
    last_progress_write: float = 0.0
    shutdown: bool = False

    def _terminal(self, jid: str) -> bool:
        return jid in self.done or jid in self.failed or jid in self.skipped

    def _running_ids(self) -> set:
        return {j.job_id for j, _ in self.running}

    def _eligible(self) -> List[Job]:
        """Jobs not yet started/terminal whose deps are all done, ordered
        biggest-setup-first. Jobs whose any dep failed/was skipped are moved
        to `skipped` (dead branch).

        Cost-aware ordering: free slots get the largest pending setup first
        (across ALL variants), so the long-pole trainings — the three 9,11s,
        then 5,11s — start as early as possible and run in parallel rather
        than one variant's copy trailing the others. This packs the tail
        tighter (vs. plain insertion order, which is variant-major and would
        defer prune-2's big setups). A big setup's h2h/lbr only become
        eligible after its train, so this never starves downstream work."""
        out = []
        running_ids = self._running_ids()
        for jid, job in self.jobs.items():
            if jid in running_ids or self._terminal(jid):
                continue
            if any(d in self.failed or d in self.skipped for d in job.deps):
                self.skipped.add(jid)
                continue
            if all(d in self.done or d in self.preset for d in job.deps):
                out.append(job)
        out.sort(key=lambda j: (-sum(j.setup), j.job_id))
        return out

    def _start(self, job: Job) -> None:
        os.makedirs(os.path.dirname(job.log_path), exist_ok=True)
        log_f = open(job.log_path, "a", buffering=1)
        log_f.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} START "
                    f"{job.job_id}: {' '.join(job.cmd)} =====\n")
        log_f.flush()
        proc = subprocess.Popen(job.cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                close_fds=True)
        job.started_at = time.time()
        self.running.append((job, proc))
        print(f"[start] {job.job_id} (pid={proc.pid})", flush=True)

    def _record(self, job: Job) -> None:
        """Parse a finished job's log and merge metrics into results."""
        if job.kind == "train":
            return
        try:
            with open(job.log_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            text = ""
        if job.kind == "h2h":
            parsed = _parse_h2h(text)
        else:  # lbr<depth> — key the exploitability string by the job's depth
            val = _parse_lbr_value(text)
            parsed = {f"{job.kind}_expl": val} if val else {}
        if not parsed:
            print(f"[warn ] {job.job_id} finished rc=0 but no metric parsed",
                  flush=True)
            return
        key = f"{job.variant}/{_setup_dirname(job.setup)}"
        self.results.setdefault(key, {}).update(parsed)
        _save_results(self.results)

    def _poll(self) -> None:
        still = []
        for job, proc in self.running:
            rc = proc.poll()
            if rc is None:
                still.append((job, proc))
                continue
            job.finished_at = time.time()
            job.returncode = rc
            dur = job.finished_at - (job.started_at or job.finished_at)
            if rc == 0:
                self.done.add(job.job_id)
                if job.kind != "train":
                    self._record(job)
                print(f"[done ] {job.job_id} in {dur:.0f}s", flush=True)
            else:
                job.attempts += 1
                if job.attempts < self.max_attempts:
                    # Re-queue: clear run state but keep it OUT of `failed`, so
                    # _eligible() picks it up again next pass (deps still hold,
                    # dependents stay alive instead of being dead-branched).
                    job.started_at = None
                    job.finished_at = None
                    job.returncode = None
                    print(f"[retry] {job.job_id} rc={rc} after {dur:.0f}s "
                          f"(attempt {job.attempts}/{self.max_attempts}); re-queuing",
                          flush=True)
                else:
                    self.failed.add(job.job_id)
                    print(f"[FAIL ] {job.job_id} rc={rc} after {dur:.0f}s, "
                          f"{job.attempts} attempts — see {job.log_path}", flush=True)
        self.running = still

    def _write_progress(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self.last_progress_write) < self.progress_every_s:
            return
        self.last_progress_write = now
        total = len(self.jobs)
        snapshot = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "running": [
                {"job": j.job_id, "kind": j.kind,
                 "elapsed_s": round(now - (j.started_at or now), 1)}
                for j, _ in self.running
            ],
            "done": len(self.done), "failed": len(self.failed),
            "skipped": len(self.skipped), "total": total,
            "remaining": total - len(self.done) - len(self.failed) - len(self.skipped),
            "failed_ids": sorted(self.failed),
            "skipped_ids": sorted(self.skipped),
        }
        os.makedirs(EXPERIMENTS_ROOT, exist_ok=True)
        tmp = PROGRESS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        os.replace(tmp, PROGRESS_PATH)

    def run(self) -> None:
        total = len(self.jobs)
        if total == 0:
            print("[experiments] nothing to do — all outputs/results present",
                  flush=True)
            return
        print(f"[experiments] {total} jobs across "
              f"{len({j.variant for j in self.jobs.values()})} variants", flush=True)
        while True:
            n_terminal = len(self.done) + len(self.failed) + len(self.skipped)
            if n_terminal >= total and not self.running:
                break
            if self.shutdown:
                self._terminate()
                break
            for job in self._eligible():
                if len(self.running) >= self.workers:
                    break
                self._start(job)
            self._poll()
            self._write_progress()
            time.sleep(1.0)
        self._write_progress(force=True)

    def _terminate(self) -> None:
        print(f"[shutdown] terminating {len(self.running)} running jobs", flush=True)
        for _job, proc in self.running:
            try:
                proc.terminate()
            except Exception:
                pass
        for _job, proc in self.running:
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
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel processes (default 8 — match CCX33 vCPUs).")
    ap.add_argument("--sweep", choices=sorted(SWEEPS), default="pruning",
                    help="Which variant set to run: 'pruning' (Sweep 1, varies "
                         "pruning at 5M iters), 'iters' (Sweep 2, varies "
                         "iteration count at the baseline -20/-22 pruning), or "
                         "'penalty' (Sweep 3, lowers the per-setup penalty at "
                         "baseline iters+pruning). Independent sweeps; default "
                         "'pruning'.")
    ap.add_argument("--iter", type=int, default=None,
                    help="Override every variant's iteration count (e.g. for a "
                         "low-iter smoke test). Default: use each variant's own "
                         "iters (5M for the pruning sweep, per-variant for iters).")
    ap.add_argument("--log-points", type=int, default=20)
    ap.add_argument("--num-deals", type=int, default=10_000,
                    help="Monte-Carlo deals per head-to-head (default 10k; "
                         "SE ~0.01, so margin >=0.02 is signal).")
    ap.add_argument("--lbr-max-sum", type=int, default=6,
                    help="Run the post-train LBR stage only for setups whose "
                         "hand-size sum is <= this (default 6 = (2,2)/(3,3) "
                         "only — these run in seconds/minutes). Each eligible "
                         "setup runs the depth battery in LBR_DEPTHS ((2,2): "
                         "LBR-1/2/3, (3,3): LBR-1/2, else LBR-1). (4,5) at sum 9 "
                         "took 3.5-5.5h for LBR-1 alone in Sweep 1, so it is "
                         "excluded by default; raise this only if you can afford "
                         "multi-hour LBR on the bigger setups.")
    ap.add_argument("--train-exploitability", action="store_true",
                    help="Enable periodic in-training LBR-1 snapshots. OFF by "
                         "default — expensive on big setups (see module docs).")
    ap.add_argument("--max-attempts", type=int, default=2,
                    help="Times a failed job is run before giving up (default 2). "
                         "A transient failure (e.g. a missing-dep deploy gap) is "
                         "re-queued so its dependents aren't dead-branched.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build and print the job graph, then exit.")
    args = ap.parse_args()

    variants = SWEEPS[args.sweep]
    if args.iter is not None:
        variants = [v._replace(iters=args.iter) for v in variants]

    results = _load_results()
    jobs, preset_done = build_graph(
        variants, CROSS_SECTION, args.log_points,
        args.num_deals, args.lbr_max_sum, args.train_exploitability, results)

    if args.dry_run:
        print(f"[dry-run] sweep={args.sweep} variants="
              f"{[(v.tag, v.iters, v.pruning, v.min_regret) for v in variants]}")
        print(f"[dry-run] {len(jobs)} jobs to run, "
              f"{len(preset_done)} already complete (skipped):")
        for jid in sorted(jobs):
            deps = jobs[jid].deps
            dep_note = ""
            if deps:
                ready = all(d in preset_done for d in deps)
                dep_note = f"  deps={deps} ({'ready' if ready else 'waits'})"
            print(f"  {jid:35s} [{jobs[jid].kind}]{dep_note}")
        # also surface what was skipped as already-done
        if preset_done:
            print(f"[dry-run] preset-done ids: {sorted(preset_done)}")
        return 0

    sched = Scheduler(jobs=jobs, results=results, workers=args.workers,
                      max_attempts=args.max_attempts)
    # Already-complete prerequisites (omitted from the graph) go in `preset`,
    # NOT `done` — so termination/progress counts stay over real jobs only,
    # while dependents still see their deps as satisfied.
    sched.preset = preset_done

    def _sigterm(signum, frame):
        sched.shutdown = True
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    sched.run()
    _write_variant_summaries(variants, CROSS_SECTION, results)

    print(f"\n[summary] done={len(sched.done)} failed={len(sched.failed)} "
          f"skipped={len(sched.skipped)}", flush=True)
    if sched.failed:
        print("FAILED:", flush=True)
        for jid in sorted(sched.failed):
            print(f"  - {jid} (log: {sched.jobs[jid].log_path})", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
