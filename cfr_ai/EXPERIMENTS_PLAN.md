# Post-V2 Parameter Experiments

Planning doc + results record for the experiment phase after the V2
retrain. Keeps the plan, the per-sweep results, and the conclusions in one
place (deliberately out of the main README, which only needs the final
adopted settings, not the full experiment tables).

## Status (updated 2026-05-29)

* **Phase 1 retrain**: complete. All 66 setups retrained V2, validated
  against v0 archive (game values within noise, structural metrics within
  ~1%, head-to-head on 11,11 = +0.006 = noise). 5.7-9× per-setup
  speedup over the historical Python production trainer.
* **Phase 2 LBR-1**: stopped at **22 setups** (sums 2-10) when the
  trajectory stalled (sum-10 setups taking 6h+). Recovered to local
  `summary_of_all_runs.csv` (ASCII). Pivoted to the parameter sweeps.
* **Sweep 1 (pruning)**: **COMPLETE — see results below.** Verdict:
  **adopt prune-10 (-10/-12)** — ~+8.6% faster training, no quality cost.
* **Sweep 2 (iterations, 10M @ baseline -20/-22)**: **COMPLETE — see
  results below.** Verdict: **10M strictly beats 5M** — H2H positive on
  13/13 non-trivial setups (sign-test p ≈ 1e-4), and less exploitable at
  LBR-1/2/3. **3M arm skipped** (5M isn't saturated, so fewer iters would
  only hurt). Production stays 5M for now; 10M banked as a "throw-compute"
  lever for when cheaper ideas are exhausted.
* **Sweep 3 (penalty)**: **COMPLETE — see results below.** H2H *slightly
  favours removal* (penalty-0 mean +0.0074 vs baseline, 5/6 setups positive;
  penalty-0.05 +0.0039), and training times show a speedup or at least no
  degradation — so the bet-penalty isn't clearly earning its keep on the two
  axes it was built for (speed + `temporary_value` soundness). Effects are
  small / near-noise and the speed read is contention-confounded; penalty=0 is
  a candidate to drop, not a prod change yet.
* **CFR-vs-NFSP evaluation**: tooling built & validated; analysis **PAUSED**
  — opponent too weak (see section below).
* **Lambda deployment**: refactored to sparse-mmap + uint16. Lambda at
  **256 MB tier**, cold ~2.5-3.5s, warm ~30-50ms, peak memory ~115-170 MB.
  Validated live in production (Morana traffic across setups up to sum 18,
  0 errors; synthetic (11,11) at 133 MB). The retired multi-Lambda
  comparison was dropped from the README. See `cfr_ai/README.md`.

## Sweep 1 — Results & Conclusions (2026-05-29)

3 pruning variants vs the V2 baseline (-20/-22, 5M iters), 8 cross-section
setups. Method: H2H = 10k MC deals (variant minus baseline payoff, [-1,1];
per-value SE ~0.01, so a single setup is signal only at |val| >= ~0.02).
Aggregate quality per variant via mean H2H, t = mean/SE_mean, and a sign
test (robust to the SE assumption). Speedup vs the V2 training duration.

**Training speedup** (clean signal = min_bet-0 mid setups, all ran 8-way):

| variant | 4,5 | 5,7 | 6,6 | 3,9 | **mean** |
|---|--:|--:|--:|--:|--:|
| prune-10 | +9.3% | +10.6% | +8.0% | +6.5% | **+8.6%** |
| prune-5  | +14.1% | +16.1% | +11.5% | +11.1% | **+13.2%** |
| prune-2  | +15.6% | +18.9% | +27.2%* | +15.1% | **+19.2%** |

min_bet-27 setups (5,11/9,11): ~0-5%, noise (tiny action space, little to
prune). Tiny 2,2/3,3: excluded — recovered under light load, contention
artifact, not comparable. *6,6/prune-2 a likely co-runner-noise outlier.

**Quality** (H2H vs baseline, n=13 per variant; negative = worse):

| variant | mean H2H | EV cost | t | sign neg/pos | sign-test p | verdict |
|---|--:|--:|--:|--:|--:|---|
| prune-10 | -0.0015 | -0.15% | -0.5 | 8/5 | 0.58 | no degradation |
| prune-5  | -0.0054 | -0.54% | -1.9 | 11/2 | 0.022 | small, real |
| prune-2  | -0.0101 | -1.01% | -3.7 | 12/1 | 0.003 | clear |

LBR-1 (sum<=9) corroborates: exploitability on 4,5 creeps up with
loosening (baseline mean ~4.45% → prune-10 ~4.68% → prune-5 ~4.81%).

**Conclusions:**
1. **Adopt prune-10 (-10/-12) for future training / productionization** —
   ~8-9% cheaper training on every full-action-space setup with
   statistically zero quality cost (sign test p=0.58). A strict
   improvement over the -20 baseline.
2. prune-5 (+13% speed, ~-0.5% EV, p=0.022) and prune-2 (+19% speed,
   ~-1.0% EV, p=0.003) trade increasing speed for increasing quality cost;
   not worth it unless training cost dominates.
3. The pruning threshold barely affects *speed* on small-action-space
   setups (min_bet-27, or few-card), and barely affects *quality* until
   pushed aggressively (prune-2). It's a mild knob, not a sensitive one.

## Sweep 2 — Results & Conclusions (2026-05-29)

1 iteration variant (10M) vs the V2 baseline (5M), both at pruning -20/-22,
across the 8 cross-section setups — isolating the pure effect of iteration
count. Same H2H method as Sweep 1 (10k MC deals, variant minus baseline
payoff; positive = 10M stronger). LBR-1 came from the sweep; LBR-2/3 were
added afterward as a deeper-adversary check on the two tiny setups (6 paired
runs, 6 cores, ~7 min wall).

**H2H, 10M vs 5M baseline** (positive = 10M better; SE ~0.01, |val| ≥ 0.02 = signal):

| Setup | H2H P0 | H2H P1 |
|---|--:|--:|
| (2,2) | +0.0008 | — |
| (3,3) | +0.0170 | — |
| (4,5) | +0.0088 | +0.0239 |
| (5,7) | +0.0108 | +0.0200 |
| (6,6) | +0.0218 | — |
| (3,9) | +0.0095 | +0.0113 |
| (5,11) | +0.0112 | +0.0145 |
| (9,11) | +0.0156 | +0.0071 |

Every non-trivial number is positive (13/13; (2,2) is solved → dead even).
Sign-test p ≈ 1e-4 — not noise. Margins +1 to +2.4 pp.

**LBR exploitability, baseline 5M vs 10M** (lower = less exploitable; LBR is a
*lower* bound on true exploitability and tightens — rises — with depth).
(2,2) is exact (population < sampling cap → no SE); (3,3) is sampled K=500 but
**paired** (shared seed → identical sampled hands/beliefs across the two
models, so the delta is far tighter than the marginal SE ≈ 0.91 pp):

| Setup | depth | baseline 5M | 10M | 10M advantage |
|---|--:|--:|--:|--:|
| (2,2) | LBR-1 | +1.029% | +0.880% | 0.149 pp |
| (2,2) | LBR-2 | +2.131% | +1.973% | 0.158 pp |
| (2,2) | LBR-3 | +2.907% | +2.624% | **0.283 pp** |
| (3,3) | LBR-1 | -0.312% | -0.571% | 0.259 pp |
| (3,3) | LBR-2 | +2.129% | +1.308% | **0.821 pp** |

The LBR-1 baseline is the V2 production value from Phase 2
(`outputs/summary_of_all_runs.csv`); the 10M LBR-1 came from the sweep — both
at the same (500 lbr-hand, 300 belief) sampling and seed 42, as are the
LBR-2/3 runs, so every row is paired/comparable. Run times (1 core each):
(2,2) LBR-2 ~96s, LBR-3 ~225-262s; (3,3) LBR-2 ~440-449s. Raw LBR-2/3 logs on
the box at `cfr_ai/experiments/_lbr_extra/` (gitignored).

**Conclusions:**
1. **10M is strictly stronger and less exploitable than 5M**, confirmed by
   three independent lenses (H2H, LBR-1, and the deeper LBR-2/3). The
   exact-(2,2) gap *widens* with LBR depth (0.15 → 0.16 → 0.28 pp) — the
   signature of a better-converged strategy, not a self-play artifact.
2. **Skip the conditional 3M arm.** Its only purpose was to test whether 5M
   is overkill; since 5M is, if anything, mildly under-trained (10M still
   gains), 3M would land below baseline. Nothing to learn.
3. **Production stays 5M for now.** 10M is a one-time sub-2× training cost
   (~1.75× measured, but optimistic — see cost note) for a durable +1-2 pp /
   ~0.2-0.3 pp-exploitability edge — banked as the
   "throw-compute" lever, to pull only after cheaper ideas (penalty,
   multi-visit, abstraction) are exhausted.
4. **The orchestrator now runs this LBR battery by default** ((2,2):
   LBR-1/2/3, (3,3): LBR-1/2 — `LBR_DEPTHS` in `experiments_run.py`), so any
   future sweep that includes the small setups re-measures the deeper
   exploitability automatically. **Caveat — a penalty sweep won't trigger it:**
   every penalty>0 setup has sum ≥ 7 (above the LBR cap of 6), and the small
   setups that afford deep LBR already train at penalty 0, so a 0-penalty
   variant is a no-op on them — they drop out of that sweep's cross-section
   entirely. For a penalty sweep, H2H is the only ranking signal unless we
   raise `--lbr-max-sum` and accept multi-hour LBR on a mid setup.

**Cost note — 10M vs 5M training wall (evidence, not a clean measurement):**
We have evidence the 10M runs are **meaningfully under 2×** the 5M time —
per-setup ratios averaged ~1.75× (median ~1.81×), and the saving concentrates
on full-action-space (min_bet 0) setups (5,7: 1.28×, 3,9: 1.31×, 6,6: 1.68×,
4,5: 1.81×), where late-stage `prune_feast` trims the most low-regret branches
as training matures; min_bet-27 setups with little to prune stay ~2× (9,11:
1.97×, 5,11: 2.08×). This mirrors Sweep 1 (pruning only speeds full-action
setups). **But these ratios are optimistic:** the 5M baseline times come from
the 66-setup V2 retrain (heavier, sustained contention) while the 10M times
come from this 8-setup sweep (lighter, decongesting as setups finished), so
the 10M wall is understated relative to baseline — the true compute ratio sits
somewhat above ~1.75× while still below 2×. The action-space *pattern* is
robust regardless (contention can't explain why min_bet-27 stays ~2× while
min_bet-0 drops to ~1.3×; both ran under the same sweep). Tiny 2,2/3,3 ratios
(~2×) are overhead- and HH:MM-rounding-dominated — ignore them.

## Sweep 3 — penalty (Results & Conclusions, 2026-05-29)

2 penalty-override variants vs the V2 baseline, holding iters at 5M and pruning
at -20/-22, varying ONLY the per-setup penalty: `penalty-0` (remove it) over
all 6 penalty>0 cross-section setups, and `penalty-0.05` (halve it) over the 4
setups at prod-0.1. H2H = 10k MC deals (variant minus baseline; positive =
lower-penalty model beats the penalized prod model). No LBR (every penalty>0
setup is sum ≥ 9). 20 jobs, 0 failed.

The penalty (per the designer) taxes bet-lines — actions that hand the opponent
a decision — vs checks, intended to (a) speed training by shortening sampled
sequences + trimming the traverser's surviving branches, and (b) cut repeated
node-reaches, alleviating the theoretical unsoundness of the `temporary_value`
cache. So H2H + training time are the on-target metrics — this is a
speed/soundness knob, NOT an exploitability/robustness one.

**H2H vs baseline** (positive = lower penalty stronger; SE ~0.01/cell):

`penalty-0` (remove):
| Setup | change | p0 | p1 |
|---|---|--:|--:|
| 4_5 | 0.05→0 | +0.0074 | +0.0113 |
| 5_11 | 0.05→0 | +0.0083 | +0.0119 |
| 5_7 | 0.1→0 | +0.0137 | +0.0058 |
| 9_11 | 0.1→0 | +0.0088 | +0.0071 |
| 3_9 | 0.1→0 | -0.0027 | +0.0127 |
| 6_6 | 0.1→0 | -0.0032 | — |

`penalty-0.05` (halve the 0.1 setups):
| Setup | change | p0 | p1 |
|---|---|--:|--:|
| 5_7 | 0.1→0.05 | +0.0125 | -0.0026 |
| 3_9 | 0.1→0.05 | -0.0009 | +0.0122 |
| 9_11 | 0.1→0.05 | +0.0010 | +0.0075 |
| 6_6 | 0.1→0.05 | -0.0027 | — |

penalty-0: mean **+0.0074** (11 values), 5/6 setups positive. penalty-0.05:
mean **+0.0039** (7 values), 3/4 positive. Lower-is-better, monotonic to 0
(full removal ≥ halving on every 0.1 setup). 6_6 (lone symmetric full-action
setup) is the only resister — slightly negative both arms.

**Training time (minutes), by resulting penalty:**

| Setup | min_bet | baseline | →0.05 | →0 |
|---|--:|--:|--:|--:|
| 5_7 | 0 | 143 (p0.1) | 126 | 129 |
| 6_6 | 0 | 125 (p0.1) | 137 | 111 |
| 3_9 | 0 | 124 (p0.1) | 116 | 108 |
| 9_11 | 27 | 91 (p0.1) | 104 | 91 |
| 4_5 | 0 | 120 (p0.05) | — | 94 |
| 5_11 | 27 | 92 (p0.05) | — | 108 |

**Contention caveat:** the three columns are from three different load regimes
— baseline (V2 66-setup retrain, 8-wide), →0.05 (this sweep's early 6-wide
phase), →0 (largely the decongested tail, down to 1 training) — so cross-column
wall-times are NOT a clean speed test. The one clean same-contention pair is
9_11 (both arms ran concurrently from launch): penalty-0 **91 min** vs
penalty-0.05 **104 min** → penalty-0 ~12% faster.

**Conclusions:**
1. **H2H slightly favours removal** — penalty-0 beats prod by ~+0.7 pp (5/6
   setups), penalty-0.05 by ~+0.4 pp; lower-is-better, monotonic to 0. Small
   (mostly within the ±0.01 noise floor); the signal is the consistent sign +
   aggregate, not any single setup.
2. **Times show a speedup, or at least no degradation.** penalty-0 lands at or
   below baseline on every full-action setup, and the clean 9_11 pair has it
   ~12% faster than the 0.05 arm. So the bet-penalty does NOT visibly speed
   training here — if anything the reverse (keeping marginal bets alive above
   the prune threshold costs more than the sequence-shortening saves).
3. **The penalty isn't clearly earning its keep** on either built-for axis:
   removing it slightly *improves* H2H and does not cost training time. Its one
   untested rationale is `temporary_value` soundness — penalty=0 has *more*
   repeated node-reaches yet shows no H2H quality symptom, so any theoretical
   degradation isn't manifesting in play.
4. **Not a prod change yet** (effects small/near-noise, speed read
   contention-limited), but penalty=0 is now a live candidate to drop. Clean
   confirmations if wanted: a controlled single-core speed run at matched
   concurrency, and/or an exploitability check (LBR on the smallest penalty>0
   setup, or vs NFSP once that eval is unpaused).

## Goal

Identify hyperparameter changes that improve strategy quality without
catastrophically blowing up compute. Each experiment outputs a single
H2H scalar (vs V2 baseline) + LBR-1 where affordable + utility-log
convergence features.

## Cross-section setups (8 — saturates the 8 vCPUs)

| Setup | sum | role |
|---|---|---|
| (2,2) | 4 | Tiny sanity; affords LBR-2 and LBR-3 |
| (3,3) | 6 | Small symmetric; affords LBR-2 |
| (4,5) | 9 | Mid asymmetric, penalty=0.05 |
| (5,7) | 12 | Mid-big asymmetric, penalty=0.1 |
| (6,6) | 12 | Mid symmetric, penalty=0.1, min_bet=4 |
| (3,9) | 12 | Asymmetric, large card differential |
| (5,11) | 16 | Big asymmetric, min_bet=27 |
| (9,11) | 20 | Biggest feasible; long pole |

Training pole (9,11) is ~1.6h at 5M iters. Smaller setups finish earlier
and free their cores for LBR/H2H pipelining (below).

## Variants to test

### Sweep 1: Pruning range (cheapest, run first)

| Variant tag | (pruning_threshold, min_regret) | Rationale |
|---|---|---|
| `baseline` | (-20, -22) | Current V2; **already trained, no retrain needed** |
| `prune-10` | (-10, -12) | Mild loosening |
| `prune-5` | (-5, -7) | CFR+ with MC variance headroom |
| `prune-2` | (-2, -4) | Near pure CFR+ |

3 new training runs per setup × 8 setups + LBR-1 + LBR-2 + LBR-3 (where
affordable) + H2H rounds vs baseline ≈ **~7.5h total wall** on 8 vCPUs ≈ €1.

### Sweep 2: Iteration count (INDEPENDENT of Sweep 1)

This sweep is **independent** of the pruning sweep — it does NOT use Sweep
1's winner. Pruning is held **fixed at the V2 retrain value (-20, -22)**,
the same pruning all 66 setups were (re-)trained with, so we isolate the
pure effect of iteration count.

| Variant tag | iters | pruning | Rationale |
|---|---|---|---|
| `baseline` (= V2) | 5M | (-20, -22) | **already trained** in `cfr_ai/outputs/`; the comparison point |
| `iters-10M` | 10M | (-20, -22) | does training longer obviously help? |
| `iters-3M` | 3M | (-20, -22) | **conditional** — only if 10M does NOT obviously help (i.e. we're already converged at 5M). Tests whether *fewer* iters hurt much, i.e. whether we can train cheaper. |

Only the 10M variant is trained up front (5M = the existing V2 baseline).
H2H each new variant against the V2 baseline → isolates the iteration
effect at constant pruning. `iters-3M` is a follow-up decided after seeing
the 10M result, not queued up front.

8 setups × 1 new training (10M = ~2× the 5M time) ≈ **~6h wall** on 8 vCPUs
≈ €0.8 (plus ~€0.5 for the conditional 3M arm if run).

### Future sweeps (deferred unless Sweep 1 or 2 motivates them)

- **Lower penalty + multi-visit**: requires gating the
  `temporary_value`/`last_touched` skip mechanism behind a flag in
  `_traverse_jit`. Promising but riskier. Run after we know if 5M is
  enough (Sweep 2).
- **Probabilistic 80% pruning**: replace binary `prune_feast` with
  per-action stochastic retention. Bigger code change in `_traverse_jit`;
  defer unless above sweeps suggest exploration matters.

## Per-variant pipeline (per setup)

The orchestrator (`cfr_ai/scripts/experiments_run.py`) is a per-setup DAG
runner (built; validated end-to-end on a throwaway (2,2) variant):

```
train ─────┬─→ H2H vs V2 baseline (10k MC deals)   ← primary ranking signal
           └─→ LBR battery  (affordable sums ≤ --lbr-max-sum, default 6 =
                             2,2/3,3 only; per-setup depths from LBR_DEPTHS:
                             2,2 → LBR-1/2/3, 3,3 → LBR-1/2, else LBR-1.
                             4,5 LBR-1 took 3.5-5.5h, so excluded)
```

`H2H` and `lbr1` become eligible the moment their `train` completes and
are independent of each other. The scheduler keeps `--workers` (default 8)
slots filled with whatever is eligible across all pipelines, so small-setup
training finishes early and frees cores for H2H/LBR while big-setup
training continues. Resumable: a job whose training output already exists,
or whose metric is already in `experiments/_results.json`, is skipped.

H2H is the primary ranking signal. LBR-1 is the inside view, gated by an
affordability sum (`--lbr-max-sum 9` → (2,2)/(3,3)/(4,5) only). The
experiment LBR runs **without** `--update-summary` (that writer is
hardcoded to the production summary and ignores `--setup-dir`); metrics
are parsed from stdout into each variant's `experiment_summary.csv`.

**Periodic in-training LBR is OFF by default** (`--train-exploitability`
to enable). The Phase 2 evidence overturned the earlier "cost negligible"
assumption: a single LBR-1 on a sum≥10 setup costs *hours*, so 5 snapshots
per training run would be catastrophic on (5,7)/(6,6)/(3,9)/(5,11)/(9,11).
Convergence judgement falls back to the utility log + the post-train LBR-1
stage on the small setups.

Isolation: training writes each variant to `cfr_ai/experiments/<tag>/`
via the new `--out-root` flag (copies `history.csv` + `information_set.py`
so H2H can load it; leaves the production summary untouched).

### LBR-K affordability (measured at the orchestrator's (300, 500) sampling)

| Setup | LBR-1 | LBR-2 | LBR-3 |
|---|---|---|---|
| (2,2) | ~80s ✓ | ~96s ✓ (measured) | ~4-4.5 min ✓ (measured) |
| (3,3) | ~250s ✓ | ~7.5 min ✓ (measured) | (deeper than the wired battery; not run) |
| (4,5) | (~100+ min at (500,1000); fall back to (300,500) → ~30 min) | (skip — expensive) | — |
| (5,7), (6,6), (3,9) | (300,500) ≈ ~hours | (skip) | — |
| (5,11), (9,11) | (300,500) ≈ many hours; **skip** | — | — |

Rule of thumb: run LBR-1 at the sampling that finishes in <1h. For
(5,11) and (9,11) we rely entirely on the H2H scalar for ranking.

### H2H methodology

- 10k Monte Carlo deals (`head_to_head.py --monte-carlo --num-deals 10000`)
- Model A = V2 baseline (current `cfr_ai/outputs/<setup>/`)
- Model B = variant being tested
- Reported number: average payoff to Model B per deal, in [-1, +1]
- SE ≈ 1/√10000 = 0.01, so margin ≥ 0.02 is detectable signal
- Run on Hetzner once we pivot (8 vCPUs free, fan out 8 setups in parallel)

## Output storage

Per-variant isolated directory:

```
cfr_ai/experiments/<variant_tag>/
  outputs/<setup>/
    strategy.npz
    strategy.abs.json
    diagnostic.npz
    metadata.csv               # includes utility log + LBR Exploitability block
  experiment_summary.csv       # aggregated: setup, h2h_vs_baseline, lbr1_expl,
                               #             lbr2_expl, lbr3_expl, utility features
```

`experiments/` should be gitignored locally; mirror to local from Hetzner
after each variant completes — **but skip `diagnostic.npz` on the rsync**
(local drive is tight; experiments don't need diagnostics on local since
the analysis happens against the metadata + strategy artifacts only).
Diagnostic files stay on the Hetzner box and get deleted with it.

## Utility-log retrospective (post-experiment analysis)

After all variants run, for each (variant, setup) pair extract from the
saved utility log:

| Feature | Computation |
|---|---|
| `mean_30` | Mean of (P0 + P1) chunk utilities over last 30% of iters |
| `std_30` | StdDev of the same chunks |
| `slope_50` | Linear regression slope over last 50% of iters (drift detection) |
| `burn_in_delta` | Mean P0+P1 in first 30% minus mean in last 30% |

Cross-correlate these against the variant's H2H + LBR-1 result. If
features predict the outcomes, utility log becomes a cheap proxy for
"did this variant converge?" — useful for cheaper future sweeps. If they
don't predict, treat utility log as a sanity check only.

## Budget

| Item | Wall | Cost |
|---|---|---|
| Phase 2 LBR-1 to pivot (sums ≤ 14) | ~20h | ~€2.5 |
| Sweep 1 (3 pruning variants) | ~7.5h | ~€1 |
| Sweep 2 (1 iter variant) | ~3h | ~€0.4 |
| Buffer | — | ~€1 |
| **Total estimated** | **~30h after pivot** | **~€5** |
| Productionization run (if winning variant emerges) | TBD | up to €5-10 (separate decision) |

## Productionization

Reserved for a final tightly-scoped run if any experiment shows clear
improvement. Would be:
- Apply winning hyperparams to all 66 setups
- Full LBR-1 + LBR-2 on the affordable subset
- Deploy via Docker → ECR → Lambda (with the np.searchsorted refactor
  in place; see lambda-side notes below)

## Subgame solver — shelved 2026-05-28

Phase 2's actual LBR-1 timings ruled out online subgame at any
non-trivial setup (sum=10 setups taking 5-6h+ for LBR-1 at production
sampling; subgame is a different game-tree shape but in the same cost
neighbourhood). Code moved to `cfr_ai/archive/subgame/`; see that
directory's README for the cost evidence, the revival path, and the
three directions that remain open (offline KL-regularised refinement,
tiny-setup-only online subgame, abstraction-quality work) if subgame
becomes interesting again.

The strategy-load fast path that was originally motivated by subgame's
online budget shipped anyway for its own benefits (256 MB Lambda tier,
30-50 ms warm decisions). No production code needs to be torn out for
the shelving to be clean.

## Parallel lambda-side work (does not block experiments)

**This is now done as of the second compaction.** Final state:

- `strategy_io.load_strategy_for_agent()` returns a `FlatStrategyAgent`
  with `np.searchsorted` lookup. No numba on the agent path.
- Sparse-mmap storage layout (uint16 quantised values, ~10× smaller
  than dense int16, ~30× smaller than dense fp32 per setup).
- `cfr_ai/scripts/stage_for_docker.py` converts compressed `strategy.npz`
  → sparse layout at build time only; local `cfr_ai/outputs/` stays
  compact.
- Lambda: **256 MB** (was 2048 MB before refactor, 1536 MB intermediate).
  Cold ~2.5-3.5s, warm ~30-50ms, peak memory ~115-170 MB.
- ECR storage: ~$0.11/month for the 1.13 GB image (was $0.19 at 1.85 GB
  before sparse format).
- All test cases pass including (1,1) opening-move and (7,8) worst-case.

## CFR-vs-NFSP evaluation (tools built; analysis PAUSED 2026-05-29)

A second evaluation axis: benchmark CFR against the deployed NFSP agent
(Perun). NFSP artifacts were extracted from the `blef-nfsp-lambda` ECR
image (friend's acct, eu-west-2) into `nfsp_ai/artifacts/` (gitignored);
the 1v1 deck-24 model is `nfsp_inference_24_1v1.pt`. Both agents share
`determine_action(game_state)`; the Blef engine
(`shared.api.simpleschema_local_manager`) deals/resolves. Needs torch
(installed locally) — runs local, not on the box.

**Tools (all in `cfr_ai/analysis/`, outputs gitignored):**
- `cfr_vs_nfsp.py` — per-setup + full 11×11 matrix, role-alternated,
  CFR win-rate per (start-size, non-start-size). The bread-and-butter for
  evaluating a CFR *version* (`--cfr <tag>` / `--cfr-folder`, via the new
  backward-compatible `agent.set_outputs_base`). `--setups auto|all|<list>`.
- `cfr_vs_nfsp_games.py` — whole-game generator (real mechanic: 1v1 start,
  loser +1 & opens next, eliminate at 11). One ~123-byte row per round
  (deal[24] + history[88] + sizes/seats + loser + existence) → `.npy`
  structured array. CFR opens exactly half (`cfr_seat = game_id % 2`).
- `diagnose_games.py` — Phase-1 analyzer: loss attribution, check
  calibration vs ground-truth existence, bet/bluff & response by category.

**Supporting:** all 66 setups in `cfr_ai/outputs/` were staged to the
sparse-mmap layout **in place** (kept `strategy.npz`; +~1 GB, gitignored)
so the local agent reads via mmap — cheap per-round reloads, low + shared
RAM across workers. This also de-risked the matrix tool's RAM.

**Findings (validated harness; existence cross-check 361/361):**
- Full 11×11 matrix: CFR beats this NFSP in 120/121 configs, mean
  win-rate 0.603, edge largest at small hands, → ~parity only at the
  biggest symmetric configs ((11,11)=0.465). Saved at
  `cfr_nfsp_matrix_10k.csv`/`.png`.
- Whole games (partial, 4000 of 10000 stored in
  `cfr_vs_nfsp_games_current.npy`): CFR game win-rate **0.762** — internally
  consistent with the matrix but contradicts the friend's even 1004–996.

**PAUSE reason:** 0.76 means we're studying a mismatch, not Morana's real
weaknesses. The harness is sound; the *opponent* is too weak (sampled 1v1
avg-policy Perun). The diagnostic (`diagnose_games.py`) DID show a coherent
story — CFR over-challenges senior bets (flush+) vs ground truth, Perun
badly under-challenges — but conclusions are unreliable vs a weak opponent.

**To resume:** identify the NFSP config matching the friend's 1004–996
(most likely `--nfsp-greedy`, or a different/stronger checkpoint — confirm
with the friend). Re-run the SAME tools against it; if win-rate ≈ even, the
loss-attribution / calibration tables become a real read on Morana. Phase 2
(full-information oracle correctness) stays shelved until then.
