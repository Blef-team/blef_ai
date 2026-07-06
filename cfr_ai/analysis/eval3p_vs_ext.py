"""3-player A/B vs an EXTERNAL opponent: how do shipped (p3) and 2x (p3_2x)
CFR each fare when 1 CFR seat plays 2 identical external opponents
(conservative_bayesian or nfsp), seat rotated. Metric = CFR round-loss rate
(baseline 1/3; lower = stronger). Reports both versions per setup + the delta,
so we see whether 2x's self-A/B high-total edge also shows vs the yardsticks.

  python -m cfr_ai.analysis.eval3p_vs_ext --opp conservative --n 300 --out vs_cons.csv
  python -m cfr_ai.analysis.eval3p_vs_ext --opp nfsp --n 400 --out vs_nfsp.csv
"""
import argparse
import csv
import math
import random
from multiprocessing import Pool

import shared.api.simpleschema_local_manager as gm
from cfr_ai.analysis.eval3p_serve import serve

OPP = None
BASE_SHIP = BASE_2X = None


def _init(opp_kind, base_ship, base_2x):
    global OPP, BASE_SHIP, BASE_2X
    BASE_SHIP, BASE_2X = base_ship, base_2x
    if opp_kind == "conservative":
        import conservative_bayesian_ai.agent as a
        OPP = a.determine_action
    elif opp_kind == "nfsp":
        import nfsp_ai.production_agent as pa
        pa.load_agent(
            model_path="nfsp_ai/artifacts/nfsp_inference_24.pt",
            card_embedding_path="nfsp_ai/artifacts/card_embedding_pretrain_24.pt",
            history_embedding_path="nfsp_ai/artifacts/history_embedding_pretrain_24.pt",
            greedy=False,
        )
        OPP = pa.determine_action
    else:
        raise ValueError(opp_kind)


def one_round(setup, cfr_seat, base, cache, seed):
    random.seed(seed)
    g = gm.create_game(3, deck_size=24, max_cards=8, init_card_dist=list(setup))
    before = {p["nickname"]: int(p["n_cards"]) for p in g["players"]}
    for _ in range(80):
        cp = int(g["cp_nickname"])
        a = int(serve(g, base, cache) if cp == cfr_seat else OPP(g))
        gm.play(g, a, save_dir=None)
        after = {p["nickname"]: int(p["n_cards"]) for p in g["players"]}
        if after != before:
            changed = [k for k in before if after.get(k, 0) != before[k]]
            return int(changed[0]) if changed else None
    return None


def _ver_loss(setup, base, n):
    cache = {}
    losses = rounds = unresolved = 0
    for cfr_seat in range(3):
        for i in range(n):
            loser = one_round(setup, cfr_seat, base, cache, seed=cfr_seat * 1_000_003 + i)
            if loser is None:
                unresolved += 1
                continue
            rounds += 1
            if loser == cfr_seat:
                losses += 1
    return losses, rounds, unresolved


def eval_one(task):
    setup, n = task
    sl, sr, su = _ver_loss(setup, BASE_SHIP, n)
    xl, xr, xu = _ver_loss(setup, BASE_2X, n)
    return {"setup": "_".join(map(str, setup)), "total": sum(setup),
            "ship_loss": sl, "ship_rounds": sr, "x2_loss": xl, "x2_rounds": xr,
            "unresolved": su + xu}


DEFAULT_SETUPS = [
    (1, 2, 4), (2, 3, 4), (3, 3, 3), (2, 2, 5),      # total 7,9,9,9
    (3, 4, 5), (2, 4, 6),                            # total 12
    (3, 6, 7), (4, 6, 7),                            # total 16,17
    (5, 7, 8), (5, 8, 8), (6, 7, 8),                 # total 20,21,21  (2x's big-edge zone)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", required=True, choices=["conservative", "nfsp"])
    ap.add_argument("--base-ship", default="cfr_ai/p3/outputs")
    ap.add_argument("--base-2x", default="cfr_ai/p3_2x/outputs")
    ap.add_argument("--n", type=int, default=300, help="rounds per CFR seat (x3)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--setups", default="", help="semicolon a_b_c list; blank=DEFAULT_SETUPS")
    args = ap.parse_args()

    if args.setups:
        setups = [tuple(int(x) for x in s.split("_")) for s in args.setups.split(";")]
    else:
        setups = DEFAULT_SETUPS
    tasks = [(s, args.n) for s in setups]
    rows = []
    with Pool(args.procs, initializer=_init, initargs=(args.opp, args.base_ship, args.base_2x)) as pool:
        for i, r in enumerate(pool.imap_unordered(eval_one, tasks), 1):
            sl = r["ship_loss"] / r["ship_rounds"] if r["ship_rounds"] else float("nan")
            xl = r["x2_loss"] / r["x2_rounds"] if r["x2_rounds"] else float("nan")
            print(f"[{i}/{len(tasks)}] {r['setup']} (t{r['total']}): ship {sl*100:.2f}%  2x {xl*100:.2f}%  d {(sl-xl)*100:+.2f}pp", flush=True)
            rows.append(r)
    rows.sort(key=lambda r: (r["total"], r["setup"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    SL = sum(r["ship_loss"] for r in rows); SR = sum(r["ship_rounds"] for r in rows)
    XL = sum(r["x2_loss"] for r in rows); XR = sum(r["x2_rounds"] for r in rows)
    rs, rx = SL / SR, XL / XR
    se = math.sqrt(rs * (1 - rs) / SR + rx * (1 - rx) / XR)
    z = (rs - rx) / se if se else float("nan")
    print(f"\n==== vs {args.opp} OVERALL (baseline 33.33%, lower=stronger CFR) ====")
    print(f"  shipped CFR loss {rs*100:.3f}%  ({SR:,} rounds)")
    print(f"  2x      CFR loss {rx*100:.3f}%  ({XR:,} rounds)")
    better = "2x STRONGER" if rx < rs else ("shipped STRONGER" if rs < rx else "TIE")
    print(f"  -> {better}: 2x cuts CFR loss by {(rs-rx)*100:+.3f}pp vs {args.opp}  (z={z:+.2f})")
    print(f"\n  by total: " + "  ".join(
        f"t{t}:{'+' if (sum(rr['ship_loss'] for rr in rows if rr['total']==t)/max(1,sum(rr['ship_rounds'] for rr in rows if rr['total']==t)) - sum(rr['x2_loss'] for rr in rows if rr['total']==t)/max(1,sum(rr['x2_rounds'] for rr in rows if rr['total']==t)))>=0 else ''}{(sum(rr['ship_loss'] for rr in rows if rr['total']==t)/max(1,sum(rr['ship_rounds'] for rr in rows if rr['total']==t)) - sum(rr['x2_loss'] for rr in rows if rr['total']==t)/max(1,sum(rr['x2_rounds'] for rr in rows if rr['total']==t)))*100:.1f}pp"
        for t in sorted(set(r['total'] for r in rows))))


if __name__ == "__main__":
    main()
