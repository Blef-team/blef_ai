"""V2.1 (or any new CFR version) validation bundle.

Runs the three independent yardsticks comparing a NEW version folder against
the current working-tree models (cfr_ai/outputs/, the existing V2 baseline):

  1. LBR exploitability on the cheap asymmetric probes (1,8 and 2,7), depths
     1..--lbr-depth, for BOTH the new version and the baseline. LBR is a lower
     bound on exploitability that tightens with depth; asymmetric setups
     collapse the enumeration so deep LBR is affordable. Lower = better.
     This is the PRIMARY, intrinsic instrument (CFR convergence quality).

  2. Head-to-head new-vs-baseline (Monte-Carlo), per setup. >0.5 in the
     "new" column = the new version wins. Direct A/B of the two model sets.

  3. vs NFSP/Perun in GREEDY mode (production-realistic — deployed Perun runs
     greedy by default), for both the new version and the baseline. A coarse
     cross-paradigm pass/fail ("do we beat the agent users face"), NOT a fine
     ranking tool: CFR doesn't best-respond, so equilibrium gains don't show up
     strongly here. Reported as confirmation, not selection.

This selects versions on (1)+(2); (3) is independent confirmation. NFSP is a
held-out yardstick ONLY because CFR never trains against it — don't optimize
toward it.

Usage (on the box, AFTER the retrain finishes; full box free then):
  python -m cfr_ai.scripts.validate_version --new cfr_ai/v2.1 \
      --lbr-depth 3 --h2h-deals 10000 --perun-deals 4000 --workers 6
"""
import argparse
import os
import re
import subprocess
import sys
import time

PROBES = [(1, 8), (2, 7)]           # cheap asymmetric LBR probes
H2H_SETUPS = [(2, 2), (3, 3), (4, 5), (5, 7), (6, 6), (3, 9), (5, 11), (9, 11)]

_LBR_RE = re.compile(r"starting_player=(\d+): exploitability=([+-]?\d*\.\d+)%"
                     r"(?:\s*(?:\+/-|±)\s*([\d.]+)pp)?")
_H2H_RE = re.compile(r"advantage when player with (\d+) cards starts:\s*"
                     r"([+-]?\d*\.\d+)")


def run(cmd, timeout=None):
    """Run a subprocess, return (rc, stdout+stderr text)."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout


def lbr_one(hand_sizes, setup_dir, depth):
    """Return {sp: (expl_pct, se_pp)} for one setup dir at one depth."""
    cmd = [sys.executable, "-m", "cfr_ai.lbr",
           "--hand-sizes", str(hand_sizes[0]), str(hand_sizes[1]),
           "--depth", str(depth),
           "--n-belief-samples", "300", "--n-lbr-hand-samples", "500",
           "--setup-dir", setup_dir]
    rc, out = run(cmd)
    res = {}
    for m in _LBR_RE.finditer(out):
        res[int(m.group(1))] = (float(m.group(2)),
                                float(m.group(3)) if m.group(3) else 0.0)
    return res


def section_lbr(new_root, base_root, max_depth):
    print("\n" + "=" * 72)
    print("1. LBR EXPLOITABILITY  (lower = less exploitable = better)")
    print("=" * 72)
    print(f"{'setup':>6} {'depth':>5} {'sp':>3} {'baseline(V2)':>14} "
          f"{'new':>14} {'delta(new-base)':>16}")
    for setup in PROBES:
        sd = f"{setup[0]}_{setup[1]}"
        for depth in range(1, max_depth + 1):
            base = lbr_one(setup, os.path.join(base_root, "outputs", sd), depth)
            new = lbr_one(setup, os.path.join(new_root, "outputs", sd), depth)
            for sp in sorted(set(base) | set(new)):
                b = base.get(sp); n = new.get(sp)
                bs = f"{b[0]:+.3f}%" if b else "   -  "
                ns = f"{n[0]:+.3f}%" if n else "   -  "
                d = (f"{n[0]-b[0]:+.3f}pp" if (b and n) else "")
                flag = ""
                if b and n:
                    flag = "  better" if n[0] < b[0] - 1e-9 else \
                           ("  worse" if n[0] > b[0] + 1e-9 else "  even")
                print(f"{sd:>6} {depth:>5} {sp:>3} {bs:>14} {ns:>14} {d:>16}{flag}")


def section_h2h(new_root, deals, setups):
    print("\n" + "=" * 72)
    print("2. HEAD-TO-HEAD  new vs baseline(V2)   (>0 = new wins)")
    print("=" * 72)
    print(f"{'setup':>6} {'adv (new-baseline), per starting hand size':>50}")
    for setup in setups:
        cmd = [sys.executable, "-m", "cfr_ai.analysis.head_to_head",
               "--hand-sizes", str(setup[0]), str(setup[1]),
               "--monte-carlo", "--num-deals", str(deals),
               "--model1", "current",            # baseline = working tree (V2)
               "--model2-folder", new_root]      # new version
        rc, out = run(cmd)
        advs = [f"{m.group(1)}c:{float(m.group(2)):+.3f}"
                for m in _H2H_RE.finditer(out)]
        print(f"{setup[0]},{setup[1]:<3} {'  '.join(advs) if advs else '(parse failed)':>50}")
    print("  NB advantage is for model2 (new) over model1 (baseline V2).")


def section_perun(new_root, deals, workers):
    print("\n" + "=" * 72)
    print("3. vs NFSP/PERUN (GREEDY = production)   (>0.5 = CFR wins)")
    print("=" * 72)
    for label, args_cfr in (("baseline V2", ["--cfr", "current"]),
                            ("new version", ["--cfr-folder", new_root])):
        out_csv = os.path.join("/tmp", f"perun_{label.split()[0]}.csv")
        cmd = [sys.executable, "-m", "cfr_ai.analysis.cfr_vs_nfsp",
               *args_cfr, "--setups", "auto", "--nfsp-greedy",
               "--deals-per-cell", str(deals), "--workers", str(workers),
               "--out", out_csv]
        print(f"\n--- {label} vs greedy Perun ---", flush=True)
        rc, out = run(cmd)
        # echo just the per-setup summary block the tool prints
        keep = False
        for line in out.splitlines():
            if "CFR win-rate vs NFSP" in line:
                keep = True
            if keep:
                print("  " + line)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--new", required=True,
                    help="New version root (contains outputs/<setup>/), e.g. cfr_ai/v2.1")
    ap.add_argument("--baseline", default="cfr_ai",
                    help="Baseline root (default cfr_ai = working-tree V2).")
    ap.add_argument("--lbr-depth", type=int, default=3,
                    help="Max LBR depth on the asymmetric probes (default 3).")
    ap.add_argument("--h2h-deals", type=int, default=10000)
    ap.add_argument("--perun-deals", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip", default="",
                    help="Comma list of sections to skip: lbr,h2h,perun")
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    t0 = time.time()
    print(f"[validate] new={args.new}  baseline={args.baseline}  "
          f"lbr-depth={args.lbr_depth}", flush=True)
    if "lbr" not in skip:
        section_lbr(args.new, args.baseline, args.lbr_depth)
    if "h2h" not in skip:
        section_h2h(args.new, args.h2h_deals, H2H_SETUPS)
    if "perun" not in skip:
        section_perun(args.new, args.perun_deals, args.workers)
    print(f"\n[validate] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
