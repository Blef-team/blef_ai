"""3-player A/B strength: CFR version A (e.g. p3_2x) vs version B (shipped p3).

Mixed 3p tables: one version is "lone" vs two of the other. The lone position is
rotated over all 3 seats and which-version-is-lone is swapped, so across the full
design each version appears equally in each seat and equally as lone/field. The
two versions only ever face each other, so it's a clean relative measure.

Metric per version = round-loss rate (losses / seat-appearances); lower = stronger.
Equal strength -> both ~1/3 (the two rates sum to 2/3 by construction). Per-setup
CSV + aggregate, broken down by total cards (the regime where extra iters might
matter most is the high totals = deep third-seat infosets).

  python -m cfr_ai.analysis.eval3p_ab --base-a cfr_ai/p3_2x/outputs --base-b cfr_ai/p3/outputs --n 800 --out ab.csv
"""
import argparse
import csv
import math
import random
from multiprocessing import Pool

import shared.api.simpleschema_local_manager as gm
from cfr_ai.analysis.eval3p_serve import serve


def directed_setups():
    seen, out = set(), []
    for a in range(1, 9):
        for b in range(1, 9):
            for c in range(1, 9):
                rep = min((a, b, c), (b, c, a), (c, a, b))
                if rep in seen:
                    continue
                seen.add(rep)
                out.append((a, b, c))
    out.sort(key=lambda t: (sum(t), t))
    return out


def one_round(setup, seat_base, seat_cache, seed):
    random.seed(seed)
    g = gm.create_game(3, deck_size=24, max_cards=8, init_card_dist=list(setup))
    before = {p["nickname"]: int(p["n_cards"]) for p in g["players"]}
    for _ in range(80):
        cp = int(g["cp_nickname"])
        a = int(serve(g, seat_base[cp], seat_cache[cp]))
        gm.play(g, a, save_dir=None)
        after = {p["nickname"]: int(p["n_cards"]) for p in g["players"]}
        if after != before:
            changed = [k for k in before if after.get(k, 0) != before[k]]
            return int(changed[0]) if changed else None
    return None


def eval_setup(task):
    setup, base_a, base_b, n = task
    base = {"A": base_a, "B": base_b}
    cache = {"A": {}, "B": {}}          # one per base; serve() keys cache by setup
    loss = {"A": 0, "B": 0}
    app = {"A": 0, "B": 0}
    rounds = unresolved = 0
    for lone_pos in range(3):
        for lone_ver in ("A", "B"):
            field_ver = "B" if lone_ver == "A" else "A"
            seat_ver = {s: (lone_ver if s == lone_pos else field_ver) for s in range(3)}
            seat_base = {s: base[seat_ver[s]] for s in range(3)}
            seat_cache = {s: cache[seat_ver[s]] for s in range(3)}
            for i in range(n):
                loser = one_round(setup, seat_base, seat_cache,
                                  seed=lone_pos * 7_000_003 + (0 if lone_ver == "A" else 3_500_001) + i)
                if loser is None:
                    unresolved += 1
                    continue
                rounds += 1
                for s in range(3):
                    app[seat_ver[s]] += 1
                loss[seat_ver[loser]] += 1
    return {"setup": "_".join(map(str, setup)), "total": sum(setup),
            "lossA": loss["A"], "appA": app["A"], "lossB": loss["B"], "appB": app["B"],
            "rounds": rounds, "unresolved": unresolved}


def _rate(loss, app):
    return (loss / app) if app else float("nan")


def summarize(rows, label_a, label_b):
    LA = sum(r["lossA"] for r in rows); AA = sum(r["appA"] for r in rows)
    LB = sum(r["lossB"] for r in rows); AB = sum(r["appB"] for r in rows)
    rA, rB = _rate(LA, AA), _rate(LB, AB)
    se = math.sqrt(rA * (1 - rA) / AA + rB * (1 - rB) / AB)
    z = (rB - rA) / se if se else float("nan")
    print(f"\n==== OVERALL ({sum(r['rounds'] for r in rows):,} rounds) ====")
    print(f"  A={label_a}: loss-rate {rA*100:.3f}%  ({AA:,} appearances)")
    print(f"  B={label_b}: loss-rate {rB*100:.3f}%  ({AB:,} appearances)")
    verdict = "A STRONGER" if rA < rB else ("B STRONGER" if rB < rA else "TIE")
    print(f"  -> {verdict}: A beats B by {(rB-rA)*100:+.3f}pp   (z={z:+.2f}; |z|>2 ~ significant)")

    print(f"\n==== BY TOTAL CARDS (A loss%% / B loss%% / A-edge pp) ====")
    bands = {}
    for r in rows:
        bands.setdefault(r["total"], []).append(r)
    for tot in sorted(bands):
        rs = bands[tot]
        la = sum(r["lossA"] for r in rs); aa = sum(r["appA"] for r in rs)
        lb = sum(r["lossB"] for r in rs); ab = sum(r["appB"] for r in rs)
        ra, rb = _rate(la, aa), _rate(lb, ab)
        sed = math.sqrt(ra*(1-ra)/aa + rb*(1-rb)/ab) if aa and ab else float("nan")
        zz = (rb - ra) / sed if sed else float("nan")
        flag = " *" if abs(zz) > 2 else ""
        print(f"  total {tot:2d} ({len(rs):2d} setups): A {ra*100:5.2f}%  B {rb*100:5.2f}%  edge {(rb-ra)*100:+5.2f}pp  z={zz:+5.2f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-a", required=True)
    ap.add_argument("--base-b", required=True)
    ap.add_argument("--n", type=int, default=800, help="rounds per (lone_pos x lone_ver); 6n per setup")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only first N setups (smoke)")
    ap.add_argument("--label-a", default="2x")
    ap.add_argument("--label-b", default="shipped")
    args = ap.parse_args()

    setups = directed_setups()
    if args.limit:
        setups = setups[:args.limit]
    tasks = [(s, args.base_a, args.base_b, args.n) for s in setups]
    rows = []
    with Pool(args.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(eval_setup, tasks), 1):
            rows.append(r)
            print(f"[{i}/{len(tasks)}] {r['setup']} rounds={r['rounds']} unresolved={r['unresolved']}", flush=True)
    rows.sort(key=lambda r: (r["total"], r["setup"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} setups)")
    summarize(rows, args.label_a, args.label_b)


if __name__ == "__main__":
    main()
