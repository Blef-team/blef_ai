# Subgame solver — shelved 2026-05-28

Depth-1 subgame solver exploration. Lives here as a research record, not
as active code. Nothing in `cfr_ai/` (production trainer / lbr / agent
paths) imports anything from this directory.

## Why shelved

Phase 2 LBR-1 cost evidence (collected 2026-05-26 to 2026-05-28) made
the online-subgame use case infeasible for any non-trivial setup:

| Setup | LBR-1 at (500, 300) sampling |
|---|---|
| (3,5) | 41 min |
| (4,5) | 3 h |
| (5,5) / (4,6) / (3,7) | 5-6 h (sum=10 setups in progress at shelving time) |
| (2,8) | 3 h 32 min |
| (5,7) / (6,6) | many hours |
| (5,11) / (9,11) | days |

A subgame solver does CFR over a posterior belief — a different game-tree
shape than LBR, but combinatorial cost in the same neighbourhood. Even a
1000x speedup vs. full LBR (very aggressive sampling + shallow rollouts)
only fits tiny setups in a 2-second decision budget, and those are
already near-Nash, so the marginal quality gain is small.

The original justification for keeping the strategy-load path fast
(sub-second NPZ load via mmap) was subgame's online budget. That
justification is now weak, BUT the lazy mmap layout shipped anyway for
its own benefits (256 MB Lambda tier, 30-50 ms warm decisions). So no
production code needs to be torn out for the shelving to be clean.

## What's here

```
subgame.py                          - depth-1 solver core
analysis/subgame_h2h.py             - subgame seat vs. blueprint seat
analysis/subgame_h2h_sweep.py       - parameter sweep over h2h
analysis/subgame_latency.py         - decision-time latency measurement
analysis/subgame_setup_sweep.py     - subgame quality across setups
analysis/subgame_validate.py        - posterior + best-response sanity tests
analysis/subgame_vs_subgame.py      - two subgame agents head-to-head
```

The `cfr_ai/archive/subgame/analysis/*.py` files import
`cfr_ai.subgame`. Reviving the code means either:

- moving `subgame.py` back to `cfr_ai/subgame.py` and the analysis files
  back to `cfr_ai/analysis/`, OR
- adjusting the imports to `cfr_ai.archive.subgame.subgame`.

## When to revisit

Promising directions if subgame becomes interesting again:

1. **Offline KL-regularised refinement**. Precompute refined strategies
   for posterior belief classes the agent commonly encounters, store in
   the same `strategy.npz`-shaped format, look up at decision time. No
   online compute. Storage scales with the number of distinct beliefs
   we choose to refine.
2. **Online subgame on tiny setups only** (sum <= 6). Cost a few hundred
   ms; preserves the option without claiming general subgame solving.
3. **Drop subgame, double down on abstraction quality**. The path the
   experiment plan is currently following: pruning / iteration / penalty
   sweeps to push LBR-1 lower without changing the deployment shape.
