"""Benchmark a CFR model version against the NFSP agent (Perun), per setup.

Standard CFR-analysis tool: when you train a new CFR version for a setup,
run this to see how it does head-to-head against the fixed NFSP opponent,
broken down by (starting-player hand size, non-starting-player hand size)
with alternating agent roles. The aggregate "who wins more games" number
hides where each agent is stronger; this surfaces it.

The *setup* is the CFR-centric unit (a sorted pair of hand sizes, one
trained strategy each), so the tool lives in cfr_ai/ and selects the CFR
version to evaluate the same way head_to_head.py does:

    --cfr current                 working tree (cfr_ai/outputs/)        [default]
    --cfr <archive-tag>           cfr_ai/archive/<tag>/outputs/
    --cfr-folder <path>           <path>/outputs/  (e.g. an experiments variant)

NFSP is the fixed benchmark: the deployed 1v1 deck-24 specialist
(nfsp_inference_24_1v1.pt) via nfsp_ai.production_agent, playing its average
policy. Both agents share `determine_action(game_state) -> int`, and the
Blef engine (shared.api.simpleschema_local_manager) deals/resolves rounds;
`create_game(init_card_dist=[s, t])` forces seat 0 to hold s cards (seat 0
always starts) and seat 1 to hold t.

By default it evaluates exactly the setups the chosen CFR version has
trained (auto-detected). `--setups all` does the full 11x11; `--setups
"5,7 3,3"` a chosen few.

Run from the repo root:
    python -m cfr_ai.analysis.cfr_vs_nfsp --cfr current --setups all
    python -m cfr_ai.analysis.cfr_vs_nfsp --cfr-folder cfr_ai/experiments/prune-10
"""

import argparse
import csv
import multiprocessing as mp
import os
import random
import sys
import time
from typing import List, Optional, Tuple

DECK_SIZE = 24
MAX_CARDS = 11

# NFSP (Perun) benchmark artifacts, extracted from the blef-nfsp-lambda image.
DEF_NFSP_MODEL = "nfsp_ai/artifacts/nfsp_inference_24_1v1.pt"
DEF_NFSP_CARD = "nfsp_ai/artifacts/card_embedding_pretrain_24.pt"
DEF_NFSP_HIST = "nfsp_ai/artifacts/history_embedding_pretrain_24.pt"


# --------------------------------------------------------------------------
# CFR version resolution
# --------------------------------------------------------------------------

def _cfr_outputs_base(tag: Optional[str], folder: Optional[str]) -> str:
    """Resolve the outputs dir (containing <setup>/strategy.npz) for the CFR
    version to evaluate. Mirrors head_to_head.py's model resolution."""
    if folder:
        model_dir = folder
    elif tag in (None, "current", "."):
        model_dir = "cfr_ai"
    else:
        model_dir = os.path.join("cfr_ai", "archive", tag)
    return os.path.join(model_dir, "outputs")


def _has_strategy(setup_dir: str) -> bool:
    return (os.path.exists(os.path.join(setup_dir, "strategy_meta.npz"))
            or os.path.exists(os.path.join(setup_dir, "strategy.npz")))


def _detect_setups(outputs_base: str) -> List[Tuple[int, int]]:
    """All sorted (a,b) setups present (with a strategy) under outputs_base."""
    found = []
    if not os.path.isdir(outputs_base):
        return found
    for name in os.listdir(outputs_base):
        parts = name.split("_")
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            a, b = int(parts[0]), int(parts[1])
            if a <= b and _has_strategy(os.path.join(outputs_base, name)):
                found.append((a, b))
    return sorted(found, key=lambda st: (sum(st), st))


def _configs_for_setups(setups: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Expand sorted setups to ordered (start, nonstart) configs. Asymmetric
    setups yield both starting orders; symmetric ones a single config."""
    cfgs = []
    for a, b in setups:
        cfgs.append((a, b))
        if a != b:
            cfgs.append((b, a))
    return cfgs


# --------------------------------------------------------------------------
# Round playout (one deal) and per-cell tally
# --------------------------------------------------------------------------

def _round_loser(start_size, nonstart_size, cfr_seat, cfr_act, nfsp_act, guard=400):
    """One round at (seat0=start_size, seat1=nonstart_size); seat 0 starts.
    `cfr_seat` in {"0","1"} is the seat CFR plays. Returns losing seat or None."""
    import shared.api.simpleschema_local_manager as gm
    game = gm.create_game(2, deck_size=DECK_SIZE, max_cards=MAX_CARDS,
                          init_card_dist=[start_size, nonstart_size])
    before = {p["nickname"]: int(p["n_cards"]) for p in game["players"]}
    for _ in range(guard):
        cp = game["cp_nickname"]
        try:
            action = cfr_act(game) if cp == cfr_seat else nfsp_act(game)
            gm.play(game, int(action), save_dir=None)
        except Exception:
            return None
        after = {p["nickname"]: int(p["n_cards"]) for p in game["players"]}
        if after != before:
            for nick, b in before.items():
                af = after.get(nick, b)
                if af > b or (b > 0 and af == 0):
                    return nick
            return None
    return None


def run_cell(start_size, nonstart_size, deals_per_role, cfr_act, nfsp_act):
    """CFR-as-starter and NFSP-as-starter, deals_per_role each. CFR win counts."""
    res = {"cfr_starts_w": 0, "cfr_starts_n": 0,
           "nfsp_starts_w": 0, "nfsp_starts_n": 0, "errors": 0}
    for cfr_seat, wkey, nkey in (("0", "cfr_starts_w", "cfr_starts_n"),
                                 ("1", "nfsp_starts_w", "nfsp_starts_n")):
        nfsp_seat = "1" if cfr_seat == "0" else "0"
        for _ in range(deals_per_role):
            loser = _round_loser(start_size, nonstart_size, cfr_seat, cfr_act, nfsp_act)
            if loser is None:
                res["errors"] += 1
                continue
            res[nkey] += 1
            if loser == nfsp_seat:
                res[wkey] += 1
    return res


# --------------------------------------------------------------------------
# Parallel workers
# --------------------------------------------------------------------------
_CFR_ACT = None
_NFSP_ACT = None


def _init_worker(cfr_base, nfsp_model, nfsp_card, nfsp_hist, nfsp_greedy, base_seed):
    import torch
    torch.set_num_threads(1)
    import cfr_ai.agent as cfr_agent
    cfr_agent.set_outputs_base(cfr_base)
    from nfsp_ai.production_agent import load_agent, determine_action as nfsp
    load_agent(model_path=nfsp_model, card_embedding_path=nfsp_card,
               history_embedding_path=nfsp_hist, greedy=nfsp_greedy)
    seed = base_seed + os.getpid()
    random.seed(seed)
    torch.manual_seed(seed)
    global _CFR_ACT, _NFSP_ACT
    _CFR_ACT, _NFSP_ACT = cfr_agent.determine_action, nfsp


def _cell_task(arg):
    s, t, dpr = arg
    return (s, t, run_cell(s, t, dpr, _CFR_ACT, _NFSP_ACT))


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--cfr", default="current",
                   help="CFR version: 'current' (working tree) or an archive tag.")
    g.add_argument("--cfr-folder", default=None,
                   help="Explicit CFR model folder (its outputs/ is used). E.g. "
                        "cfr_ai/experiments/prune-10.")
    ap.add_argument("--setups", default="auto",
                    help="'auto' (setups the CFR version has trained; default), "
                         "'all' (full 11x11), or a space-separated list like "
                         "'5,7 3,3 9,11'.")
    ap.add_argument("--deals-per-cell", type=int, default=10000,
                    help="Deals per (start,nonstart) config; split between the "
                         "two role assignments. SE per cell ~1/sqrt(deals).")
    ap.add_argument("--workers", type=int,
                    default=min(8, max(1, (os.cpu_count() or 2) - 2)),
                    help="Worker processes. Each holds one CFR strategy (~0.3-0.5 "
                         "GB for big setups), so keep modest if RAM-bound.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nfsp-greedy", action="store_true",
                    help="NFSP plays argmax of its average policy. Default OFF = "
                         "sample (matches CFR's stochastic mixed play).")
    ap.add_argument("--nfsp-model", default=DEF_NFSP_MODEL)
    ap.add_argument("--nfsp-card-embed", default=DEF_NFSP_CARD)
    ap.add_argument("--nfsp-history-embed", default=DEF_NFSP_HIST)
    ap.add_argument("--out", default=None,
                    help="Output CSV. Default cfr_ai/analysis/cfr_vs_nfsp_<label>.csv")
    ap.add_argument("--heatmap", action="store_true",
                    help="Also render an 11x11 PNG heatmap (only useful for a "
                         "full/large grid).")
    args = ap.parse_args()

    cfr_base = _cfr_outputs_base(args.cfr, args.cfr_folder)
    label = (os.path.basename(args.cfr_folder.rstrip("/\\")) if args.cfr_folder
             else args.cfr)
    if not os.path.isdir(cfr_base):
        print(f"[error] CFR outputs dir not found: {cfr_base}", file=sys.stderr)
        return 1
    for p in (args.nfsp_model, args.nfsp_card_embed, args.nfsp_history_embed):
        if not os.path.exists(p):
            print(f"[error] missing NFSP artifact: {p}", file=sys.stderr)
            return 1

    # Resolve setups -> configs.
    if args.setups == "auto":
        setups = _detect_setups(cfr_base)
        if not setups:
            print(f"[error] no trained setups under {cfr_base}", file=sys.stderr)
            return 1
    elif args.setups == "all":
        setups = [(a, b) for a in range(1, 12) for b in range(a, 12)]
    else:
        setups = []
        for tok in args.setups.split():
            a, b = (int(x) for x in tok.split(","))
            setups.append(tuple(sorted((a, b))))
    configs = _configs_for_setups(setups)
    out = args.out or os.path.join("cfr_ai", "analysis", f"cfr_vs_nfsp_{label}.csv")

    print(f"[cfr_vs_nfsp] CFR='{label}' ({cfr_base}) vs NFSP "
          f"({os.path.basename(args.nfsp_model)}, "
          f"{'greedy' if args.nfsp_greedy else 'sampled'})", flush=True)
    print(f"[cfr_vs_nfsp] {len(setups)} setups -> {len(configs)} configs, "
          f"{args.deals_per_cell} deals/config, {args.workers} workers -> {out}",
          flush=True)

    deals_per_role = args.deals_per_cell // 2
    cell_args = [(s, t, deals_per_role) for (s, t) in configs]
    results = {}
    t0 = time.time()
    total = 0
    done = 0
    fields = ["start_size", "nonstart_size", "deals", "cfr_winrate",
              "cfr_as_starter_winrate", "cfr_as_nonstarter_winrate", "errors"]

    def _flush():
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        tmp = out + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            for (s, t) in configs:
                if (s, t) not in results:
                    continue
                r = results[(s, t)]
                cs, ns = r["cfr_starts_n"], r["nfsp_starts_n"]
                n = cs + ns
                w = r["cfr_starts_w"] + r["nfsp_starts_w"]
                wr.writerow({
                    "start_size": s, "nonstart_size": t, "deals": n,
                    "cfr_winrate": round(w / n, 4) if n else "",
                    "cfr_as_starter_winrate": round(r["cfr_starts_w"] / cs, 4) if cs else "",
                    "cfr_as_nonstarter_winrate": round(r["nfsp_starts_w"] / ns, 4) if ns else "",
                    "errors": r["errors"],
                })
        os.replace(tmp, out)

    def _record(s, t, r):
        nonlocal total, done
        results[(s, t)] = r
        total += r["cfr_starts_n"] + r["nfsp_starts_n"] + r["errors"]
        done += 1
        n = r["cfr_starts_n"] + r["nfsp_starts_n"]
        w = r["cfr_starts_w"] + r["nfsp_starts_w"]
        el = time.time() - t0
        print(f"[{done}/{len(configs)}] ({s},{t}) cfr_winrate="
              f"{round(w/n,4) if n else 'NA'} | {total/el:.0f} deals/s | {el:.0f}s",
              flush=True)
        _flush()

    if args.workers <= 1:
        import cfr_ai.agent as cfr_agent
        cfr_agent.set_outputs_base(cfr_base)
        from nfsp_ai.production_agent import load_agent, determine_action as nfsp_act
        load_agent(model_path=args.nfsp_model, card_embedding_path=args.nfsp_card_embed,
                   history_embedding_path=args.nfsp_history_embed,
                   greedy=bool(args.nfsp_greedy))
        random.seed(args.seed)
        for (s, t, dpr) in cell_args:
            _record(s, t, run_cell(s, t, dpr, cfr_agent.determine_action, nfsp_act))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=(cfr_base, args.nfsp_model, args.nfsp_card_embed,
                                args.nfsp_history_embed, bool(args.nfsp_greedy),
                                args.seed)) as pool:
            for (s, t, r) in pool.imap_unordered(_cell_task, cell_args):
                _record(s, t, r)

    _print_setup_summary(setups, results)
    print(f"\nWrote {len(results)} configs to {out} in {time.time()-t0:.0f}s "
          f"({total} deals).", flush=True)
    if args.heatmap:
        _render_heatmap(out, label)
    return 0


def _print_setup_summary(setups, results):
    """Collapse the (up to two) configs per setup into one CFR win-rate."""
    print("\n  setup |  CFR win-rate vs NFSP  (>0.5 = CFR wins)", flush=True)
    for (a, b) in setups:
        w = n = 0
        for cfg in ([(a, b)] if a == b else [(a, b), (b, a)]):
            r = results.get(cfg)
            if not r:
                continue
            n += r["cfr_starts_n"] + r["nfsp_starts_n"]
            w += r["cfr_starts_w"] + r["nfsp_starts_w"]
        print(f"   {a},{b:<3}|  {w/n:.3f}" if n else f"   {a},{b:<3}|  (no data)",
              flush=True)


def _render_heatmap(csv_path, label):
    import csv as _csv
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[heatmap] skipped ({e})", flush=True)
        return
    Z = np.full((11, 11), np.nan)
    for r in _csv.DictReader(open(csv_path)):
        if r["cfr_winrate"] == "":
            continue
        Z[int(r["start_size"]) - 1, int(r["nonstart_size"]) - 1] = float(r["cfr_winrate"])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(Z, origin="lower", extent=[0.5, 11.5, 0.5, 11.5],
                   cmap="RdBu_r", vmin=0.30, vmax=0.70)
    ax.set_xlabel("non-starting hand size")
    ax.set_ylabel("starting hand size")
    ax.set_title(f"CFR '{label}' win-rate vs NFSP (red=CFR wins)")
    ax.set_xticks(range(1, 12))
    ax.set_yticks(range(1, 12))
    for i in range(11):
        for j in range(11):
            if not np.isnan(Z[i, j]):
                ax.text(j + 1, i + 1, f"{Z[i,j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, label="CFR win-rate")
    plt.tight_layout()
    png = os.path.splitext(csv_path)[0] + ".png"
    plt.savefig(png, dpi=120)
    print(f"[heatmap] wrote {png}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
