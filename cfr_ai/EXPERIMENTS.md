# Blef CFR — Experiments

The single authoritative record of every experiment run on the Blef CFR
trainer: the plan, the per-experiment results, the conclusions, and — crucially
— the **surprises vs. what theory/experience predicted** and the process
lessons. This consolidates the former `EXPERIMENTS_PLAN.md` (forward plan +
early parameter sweeps) and `EXPERIMENTS_LOG.md` (post-V2.1 chronological log).
The main `README.md` carries only the final *adopted* settings; this doc carries
the full experiment tables and reasoning.

Numbers are **measured** unless explicitly labelled an estimate/hypothesis. When
in doubt, content was preserved rather than dropped.

## Purpose & key locations

- **V2.1 (current anchor; next deploy):** `cfr_ai/outputs` — the consolidation
  retrain, swapped in here 2026-05-31 (validated ≈ V2 whole-game, less exploitable
  on probes). `summary_of_all_runs.csv` (training + LBR-1/LBR-2 cols) lives *inside*
  `outputs/`.
- **V2 (previous live; archived):** `cfr_ai/archive/v2/outputs`
- **Shared package files:** `cfr_ai/information_set.py`, `cfr_ai/history.csv` — one
  copy used by whatever model is in `outputs/`; **not** duplicated per snapshot.
- **min_bet=27 experiment:** `cfr_ai/exp_mb27` (on box)
- **Per-variant sweep outputs:** `cfr_ai/experiments/<variant_tag>/` (gitignored
  locally; mirror from box)
- **Raw LBR-2/3 logs:** on the box at `cfr_ai/experiments/_lbr_extra/` (gitignored)
- **Box:** Hetzner CCX33 (8 vCPU, ~30.6 GiB RAM), venv
  `/root/blef_ai/.venv/bin/python`

Each experiment outputs a single **H2H scalar** (vs the relevant baseline) +
**LBR-1** where affordable + utility-log convergence features. **Overall goal:**
identify hyperparameter changes that improve strategy quality without
catastrophically blowing up compute.

### Cross-section setups (8 — saturates the 8 vCPUs)

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

Training pole (9,11) is ~1.6h at 5M iters. Smaller setups finish earlier and
free their cores for LBR/H2H pipelining (see Methodology section).

### Phase status snapshot (as of 2026-05-30)

- **Phase 1 retrain:** complete. All 66 setups retrained V2, validated against
  v0 archive (game values within noise, structural metrics within ~1%,
  head-to-head on 11,11 = +0.006 = noise). 5.7–9× per-setup speedup over the
  historical Python production trainer.
- **Phase 2 LBR-1:** stopped at **22 setups** (sums 2-10) when the trajectory
  stalled (sum-10 setups taking 6h+). Recovered to local
  `summary_of_all_runs.csv` (ASCII). Pivoted to the parameter sweeps.
- **Sweep 1 (pruning):** COMPLETE — **adopt prune-10 (-10/-12)**, ~+8.6% faster
  training, no quality cost.
- **Sweep 2 (iterations, 10M):** COMPLETE — **10M strictly beats 5M**, 3M arm
  skipped, production stays 5M (10M banked as a throw-compute lever).
- **Sweep 3 (penalty) + exp3b probe:** COMPLETE — **DROP the penalty, KEEP
  temporary_value.**
- **Abstraction speedup:** `get_hand_abstraction` rewritten pure-Python,
  byte-identical, 5.7× faster on the function, ~2.3× faster retrains. Committed
  (6b21a8a).
- **V2.1 consolidation retrain:** COMPLETE — all 66 setups retrained at the
  banked wins; ≈ V2 strength, ~40% faster.
- **CFR-vs-NFSP evaluation:** tooling built & validated; **evaluations run** vs
  both Perun modes. Only **greedy** Perun is competitive (~0.50 vs CFR, matches
  the friend's even 1004–996) but is **not** Nash-approximating (pure,
  exploitable by adaptive play); the sampled avg-policy loses ~24% (CFR punishes
  its off-policy tails, ≈9pp inflation). Greedy = the realistic yardstick; never
  train against it.
- **Lambda deployment:** refactored to sparse-mmap + uint16; **256 MB tier**,
  validated live in production.

---

## Chronological experiments — conclusions & surprises

### Sweep 1 — pruning range (2026-05-29) — adopt prune-10

3 pruning variants vs the V2 baseline (-20/-22, 5M iters), 8 cross-section
setups. Method: H2H = 10k MC deals (variant minus baseline payoff, [-1,1];
per-value SE ~0.01, so a single setup is signal only at |val| >= ~0.02).
Aggregate quality per variant via mean H2H, t = mean/SE_mean, and a sign test
(robust to the SE assumption). Speedup vs the V2 training duration.

Variants tested:

| Variant tag | (pruning_threshold, min_regret) | Rationale |
|---|---|---|
| `baseline` | (-20, -22) | Current V2; already trained, no retrain needed |
| `prune-10` | (-10, -12) | Mild loosening |
| `prune-5` | (-5, -7) | CFR+ with MC variance headroom |
| `prune-2` | (-2, -4) | Near pure CFR+ |

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

LBR-1 (sum<=9) corroborates: exploitability on 4,5 creeps up with loosening
(baseline mean ~4.45% → prune-10 ~4.68% → prune-5 ~4.81%).

**Conclusions:**
1. **Adopt prune-10 (-10/-12) for future training / productionization** —
   ~8-9% cheaper training on every full-action-space setup with statistically
   zero quality cost (sign test p=0.58). A strict improvement over the -20
   baseline.
2. prune-5 (+13% speed, ~-0.5% EV, p=0.022) and prune-2 (+19% speed, ~-1.0% EV,
   p=0.003) trade increasing speed for increasing quality cost; not worth it
   unless training cost dominates.
3. The pruning threshold barely affects *speed* on small-action-space setups
   (min_bet-27, or few-card), and barely affects *quality* until pushed
   aggressively (prune-2). It's a mild knob, not a sensitive one.

### Sweep 2 — iteration count (2026-05-29) — 10M strictly beats 5M

1 iteration variant (10M) vs the V2 baseline (5M), both at pruning -20/-22,
across the 8 cross-section setups — isolating the pure effect of iteration
count. This sweep is **independent** of Sweep 1 — it does NOT use Sweep 1's
winner; pruning is held fixed at the V2 retrain value (-20, -22), the same
pruning all 66 setups were (re-)trained with. Same H2H method as Sweep 1 (10k
MC deals, variant minus baseline payoff; positive = 10M stronger). LBR-1 came
from the sweep; LBR-2/3 were added afterward as a deeper-adversary check on the
two tiny setups (6 paired runs, 6 cores, ~7 min wall).

Variants (the conditional 3M arm was a follow-up decided after seeing 10M, not
queued up front):

| Variant tag | iters | pruning | Rationale |
|---|---|---|---|
| `baseline` (= V2) | 5M | (-20, -22) | already trained in `cfr_ai/outputs/`; the comparison point |
| `iters-10M` | 10M | (-20, -22) | does training longer obviously help? |
| `iters-3M` | 3M | (-20, -22) | **conditional** — only if 10M does NOT obviously help (i.e. we're already converged at 5M). Tests whether *fewer* iters hurt much. |

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
**paired** (shared seed → identical sampled hands/beliefs across the two models,
so the delta is far tighter than the marginal SE ≈ 0.91 pp):

| Setup | depth | baseline 5M | 10M | 10M advantage |
|---|--:|--:|--:|--:|
| (2,2) | LBR-1 | +1.029% | +0.880% | 0.149 pp |
| (2,2) | LBR-2 | +2.131% | +1.973% | 0.158 pp |
| (2,2) | LBR-3 | +2.907% | +2.624% | **0.283 pp** |
| (3,3) | LBR-1 | -0.312% | -0.571% | 0.259 pp |
| (3,3) | LBR-2 | +2.129% | +1.308% | **0.821 pp** |

The LBR-1 baseline is the V2 production value from Phase 2
(`outputs/summary_of_all_runs.csv`); the 10M LBR-1 came from the sweep — both at
the same (500 lbr-hand, 300 belief) sampling and seed 42, as are the LBR-2/3
runs, so every row is paired/comparable. Run times (1 core each): (2,2) LBR-2
~96s, LBR-3 ~225-262s; (3,3) LBR-2 ~440-449s. Raw LBR-2/3 logs on the box at
`cfr_ai/experiments/_lbr_extra/` (gitignored).

**Conclusions:**
1. **10M is strictly stronger and less exploitable than 5M**, confirmed by three
   independent lenses (H2H, LBR-1, and the deeper LBR-2/3). The exact-(2,2) gap
   *widens* with LBR depth (0.15 → 0.16 → 0.28 pp) — the signature of a
   better-converged strategy, not a self-play artifact.
2. **Skip the conditional 3M arm.** Its only purpose was to test whether 5M is
   overkill; since 5M is, if anything, mildly under-trained (10M still gains),
   3M would land below baseline. Nothing to learn.
3. **Production stays 5M for now.** 10M is a one-time sub-2× training cost
   (~1.75× measured, but optimistic — see cost note below) for a durable +1-2 pp
   / ~0.2-0.3 pp-exploitability edge — banked as the "throw-compute" lever, to
   pull only after cheaper ideas (penalty, multi-visit, abstraction) are
   exhausted.
4. **The orchestrator now runs this LBR battery by default** ((2,2): LBR-1/2/3,
   (3,3): LBR-1/2 — `LBR_DEPTHS` in `experiments_run.py`), so any future sweep
   that includes the small setups re-measures the deeper exploitability
   automatically. **Caveat — a penalty sweep won't trigger it:** every penalty>0
   setup has sum ≥ 7 (above the LBR cap of 6), and the small setups that afford
   deep LBR already train at penalty 0, so a 0-penalty variant is a no-op on them
   — they drop out of that sweep's cross-section entirely. For a penalty sweep,
   H2H is the only ranking signal unless we raise `--lbr-max-sum` and accept
   multi-hour LBR on a mid setup.

**Cost note — 10M vs 5M training wall (evidence, not a clean measurement):**
We have evidence the 10M runs are **meaningfully under 2×** the 5M time —
per-setup ratios averaged ~1.75× (median ~1.81×), and the saving concentrates
on full-action-space (min_bet 0) setups (5,7: 1.28×, 3,9: 1.31×, 6,6: 1.68×,
4,5: 1.81×), where late-stage `prune_feast` trims the most low-regret branches
as training matures; min_bet-27 setups with little to prune stay ~2× (9,11:
1.97×, 5,11: 2.08×). This mirrors Sweep 1 (pruning only speeds full-action
setups). **But these ratios are optimistic:** the 5M baseline times come from
the 66-setup V2 retrain (heavier, sustained contention) while the 10M times come
from this 8-setup sweep (lighter, decongesting as setups finished), so the 10M
wall is understated relative to baseline — the true compute ratio sits somewhat
above ~1.75× while still below 2×. The action-space *pattern* is robust
regardless (contention can't explain why min_bet-27 stays ~2× while min_bet-0
drops to ~1.3×; both ran under the same sweep). Tiny 2,2/3,3 ratios (~2×) are
overhead- and HH:MM-rounding-dominated — ignore them.

### Sweep 3 — penalty (2026-05-29) — H2H favours removal

2 penalty-override variants vs the V2 baseline, holding iters at 5M and pruning
at -20/-22, varying ONLY the per-setup penalty: `penalty-0` (remove it) over all
6 penalty>0 cross-section setups, and `penalty-0.05` (halve it) over the 4
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
mean **+0.0039** (7 values), 3/4 positive. Lower-is-better, monotonic to 0 (full
removal ≥ halving on every 0.1 setup). 6_6 (lone symmetric full-action setup) is
the only resister — slightly negative both arms.

**Training time (minutes), by resulting penalty:**

| Setup | min_bet | baseline | →0.05 | →0 |
|---|--:|--:|--:|--:|
| 5_7 | 0 | 143 (p0.1) | 126 | 129 |
| 6_6 | 0 | 125 (p0.1) | 137 | 111 |
| 3_9 | 0 | 124 (p0.1) | 116 | 108 |
| 9_11 | 27 | 91 (p0.1) | 104 | 91 |
| 4_5 | 0 | 120 (p0.05) | — | 94 |
| 5_11 | 27 | 92 (p0.05) | — | 108 |

**Contention caveat:** the three columns are from three different load regimes —
baseline (V2 66-setup retrain, 8-wide), →0.05 (this sweep's early 6-wide phase),
→0 (largely the decongested tail, down to 1 training) — so cross-column
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
   training here — if anything the reverse (keeping marginal bets alive above the
   prune threshold costs more than the sequence-shortening saves).
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

### exp3b — penalty + temporary_value probe (2026-05-30) — DROP penalty, KEEP temp_value

Follow-up to Sweep 3 (`run_exp3b.sh`, two parallel groups):
- **Probe** (penalty-0, normal temp_value): trained 1,8 & 2,7 — their asymmetry
  collapses the LBR enumeration, so LBR-1 is cheap (1,8 ~4.5 min, 2,7 ~1 h), and
  both are prod-penalty-0.1 so this tests the full 0.1→0 removal on the inside
  view (H2H + LBR-1 vs baseline).
- **No-temp-value** (penalty-0 + `--no-temporary-value`): the 6 penalty>0
  cross-section setups, isolating `temporary_value` (same seed/penalty as the
  Sweep-3 penalty-0 models — only the cache differs).

**Verdict: DROP the penalty, KEEP temporary_value.**

#### Penalty → drop (less exploitable, no time cost)

LBR-1 (penalty-0 vs baseline; lower = less exploitable):

| setup | seat | penalty-0 | baseline | Δ |
|---|---|--:|--:|--:|
| 1,8 | sp0 | 5.06% | 5.11% | ~0 |
| 1,8 | sp1 | **3.03%** | 5.40% | **−2.37 pp (~3σ)** |
| 2,7 | sp0 | **3.81%** | 6.26% | **−2.46 pp** |
| 2,7 | sp1 | **2.78%** | 4.54% | **−1.76 pp** |

Every seat with a real change is *less* exploitable under penalty-0, never
worse; H2H ≈ even. **Duration-neutral**: 1,8 96 min vs baseline 94; 2,7 108 vs
108 — despite penalty-0 touching ~1.5× more nodes (no bet-tax → longer lines),
wall is unchanged because abs_ids dominates the clock, not node count. The
bet-penalty was *distorting* the strategy for no speed gain.

#### temporary_value → keep (earns its keep on speed + quality)

No-temp-value vs Sweep-3 penalty-0 (clean isolation):

| setup | nodes notv / tv | ratio | infosets | H2H notv | H2H temp-on |
|---|---|--:|---|---|---|
| 4,5 | 11.79B / 1.91B | **6.2×** | ≈ (+1%) | +0.0026 / +0.0000 | +0.0074 / +0.0113 |
| 5,11 | 12.99B / 1.47B | **8.8×** | ≈ (+1%) | −0.0099 / +0.0012 | +0.0083 / +0.0119 |

- **Speed:** the cache avoids **85–89% of node visits** — memoization that stops
  transposed sub-trees being re-traversed (multiplier compounds down the tree).
  Infosets unchanged → pure recompute, not extra exploration.
- **Quality:** removing it made H2H *worse* on both (~0.005–0.018/seat, all same
  direction) → temp_value is a small net *positive*.
- **Mechanism (open, doesn't matter):** the regret update is *not* reach-weighted
  (`regret += cf − node_value`), so an infoset reached K times via transpositions
  in one iteration gets K updates (no-temp-value) vs 1 (temp_value). Whether that
  K is a correct reach-summation (→ no-temp-value faithful) or a structural
  artifact (→ temp_value removes spurious weighting) is genuinely ambiguous; data
  leans temp_value-helps but ~1–2σ / could be variance. **Unresolved, and it need
  not be: action abstraction kills the transpositions, so temp_value becomes a
  near-no-op and any residual influence vanishes.** Keep it; the redesign retires
  the concern for free.

Caveats: probe LBR is on small/asymmetric setups; the no-temp-value setups can't
be LBR'd at sum ≥ 9; midgame no-temp-value (5,7/6,6/3,9/9,11) were **killed at
25–40%** on 2026-05-30 (at ~86–124 it/s they'd have needed ~12 h more, for
confirmatory H2H only — verdict already settled by 4,5/5,11 + the 1,8/2,7 LBR).
The no-temp-value wall cost (1.7–2.4×) is *smaller* than its 6–9× node cost
because abs_ids dominates today — but it will *grow* once the abstraction speedup
shifts the bottleneck to the traversal.

### Abstraction: the dominant cost, a 5.7× speedup, and the redesign (2026-05-30)

**Profiling.** Per-iteration time = a Python prelude (deal, existence,
`_build_abs_ids_for_iter`) + the JIT traversal. `precompute_set_existence` is
already fast (~1.5%); the dominant cost on round-6+ setups is **abstraction
interning** (`get_hand_abstraction`), recomputed every iteration: **68–81% of
training time**. It's **round-gated** — ~25 µs/call for rounds 1-5 (cheap branch)
vs ~770–816 µs/call for round 6+, a ~32× cliff exactly where the abstraction
representation changes (sum 6→7). That's why training time clusters by round, not
node count / true complexity (which lives in the mostly-unexplored infoset space
— also why bigger setups are more exploitable at fixed iters). The
penalty/temp_value hacks cut node visits, but JIT made nodes cheap, so they
barely move the abstraction-dominated clock — which is why they "stopped helping"
vs the old pure-Python trainer.

**Speedup shipped (commit 6b21a8a).** `get_hand_abstraction` rewritten in pure
Python (no numpy on tiny arrays, no `np.delete`-in-loops), **byte-identical**
(verified over all hands of sizes 2-5 + sampled 6-11), **5.7× faster** on the
function (362→64 µs/call). End-to-end ~**2.3× faster** on round-6+ setups
(in-situ 4,5 1049→1957 it/s; abs_ids 81%→34%; jit_traverse now dominant) →
roughly **halves the ~106-CPUh full retrain**, identical model. A result-cache
was rejected: recurrence collapses for big hands (an 11-card hand recurs ~2× over
5M), so a memo costs ~GBs RAM for little gain exactly where it's needed. Profiler
un-staled at `analysis/profile_trainer.py` (+ `--no-temporary-value`).

**The redesign (the real frontier — not yet built).** The hand+action
abstraction is hand-crafted without rigour and is the elephant. Directions:
1. **Action abstraction / composite actions** — collapse strategically
   equivalent bets (10s-over-9s/-Js/-Ks all serve "bluff") into a few
   representatives: a "truthful" action = the set with the highest
   P(exists | my cards)/P(exists | no info), plus a few bluff tiers by
   disprovability. Shrinks the tree → fewer transpositions (temp_value → no-op)
   → faster, better-converged, less exploitable. Highest leverage.
2. **Richer hand summary** — encode *weaknesses* / bluff-credibility (what you
   can credibly claim but not back, to induce a profitable check), not just
   strengths. Hardest, biggest ceiling.
3. **JIT the abstraction** — return integer codes (numba can't build strings),
   a 1-1 mapping of the current partition; ~10–100× more, on top. Fold into (1).

Blef framing: every bet is a claim about ALL cards on the table combined; you
escalate by *recombining* the standing claim with your cards (truthfully or as a
bluff); the *check* (does the claimed set exist?) is the pivotal action — unlike
poker, there's no opponent-equity estimation. We now have the rigour to design
empirically: profiler + H2H + the LBR-1/2/3 battery + the (paused) vs-NFSP
harness can score any abstraction change.

### Action abstraction — three generations (B/G/H), all shelved (2026-05-31 → 06-02)

Direction 1 above, built and benchmarked. All three modify **only the action list
per infoset** — the composite key / infoset delineation is untouched, so the infoset
count is ~unchanged (0.9–0.98× baseline). **All three shelved**; V2.1's full
contiguous action set stays. The code was untracked scaffolding under
`cfr_ai/abstraction/` (since removed — reproducible from the designs here); saved
artifacts on the box at `cfr_ai/{scratch_v21,grouped_outputs,experiments}`.

**Gen-B — curated probability menu. SHELVED: structural coverage gap.** A
fixed-semantics **slot** layout (~W=32) where each slot resolves to a *different
concrete bet per hand*, driven by that hand's claim-existence probabilities: `p` =
a-posteriori (hypergeometric: given my cards + the unknown opponent cards) and `g` =
a-priori (given only the card total, no hand info). Slots, all restricted to claims
above the standing bet:
- **value** — the hand's top claims by `p − g` (how much more likely a claim is true
  given *my* hand than the baseline): the bets this hand most credibly supports. (v1
  ranked by raw `p`; the v2 "structural" menu by the `p − g` diff.)
- **escalation** — build-ons of the *standing* claim via the ESCALATION graph
  (recombine the current claim with your cards into a higher claim that contains it),
  closest first. (v2 only — the natural raises.)
- **checkpoint** — a-priori probability cliffs (`g`-driven, hand-independent landmarks
  in the claim ordering; squeeze / bluff-jump targets).
- **min-raise** — the contiguous smallest legal claims just above the standing bet.
- **check.**

Only *which* bet each slot maps to varies per hand, so the resolve table is a function
of (hand, standing-bet) and is built once per distinct hand (cached by an exact
count/suit signature) — never per node. CFR trains over the slots; at play each slot
resolves to its concrete bet. Near-lossless + resource-positive in the mid-T mb0 band,
but **lost 9_11 H2H at 44.6% (hit 91.6%)**. Structural to any *selection* menu: claims
it never bets are never reached in self-play → responses never train → default-to-check
vs full-action V2.1, exploited. Don't revive.

**Gen-G — descriptor-salient grouping. SHELVED: gap-free + ties V2.1, but no resource
win (memory-bound).** A claim is its own action iff its rank/suit is referenced by the
hand OR history abstraction, + fixed ANCHORS (top of each set type + 3 straights);
unreferenced members collapse into per-type groups resolved hand-aware at traversal
(BLUFF_LOW=argmin held-support / BLUFF_HIGH=argmax / RANDOM, by group size). The random
sentinel gives full support → **gap-free (hit ~99.8%)**, fixing B's high-T collapse
(9_11 ~50.6%). But compute **2.3–4.7× wall, 1.2–1.9× RAM**: grouping cuts nodes to
0.55–0.73× yet per-node cost rises to 2.2–2.8×, because a slot is a looked-up
`(kind,arg)` → 3 per-row arrays read/node → 2–3× cache lines on random row access.
**Surprise:** the membership *computation* was only ~2% (a member-pool optimization
that removed the per-visit scan moved wall ~2%) — the cost is memory traffic from the
menu indirection, which is *architectural* (grouping breaks the contiguous action
layout). **Caveat:** its "ties V2.1" used unpruned `[-300,-1e9]` vs pruned-V2.1
`[-10,-12]` (penalty matched at 0) → hyperparameter-confounded, never re-verified matched.

**Gen-H — deterministic-hash subset. SHELVED: closest to keeping, and the only clean
comparison.** Every action concrete, bet-indexed (no menu indirection): per-row legal
bitmask = referenced ∪ anchors ∪ a deterministic ⌈k/2⌉ of each type's unreferenced
members (hash of the infoset key, frozen). Output is a standard FlatStrategy. Run
**matched** (penalty 0, pruning `[-10,-12]`, min_bet per setup, 8-wide contended)
through the production `training.py` — the only apples-to-apples comparison in this line.
- *Play:* ties V2.1 on all 8 (49.3–50.4%, ~49.7% mean = a hair weaker). Does **not**
  collapse at high-T (9_11 49.7%) unlike Gen-B — **surprise: a uniformly-random subset
  is more robust than a curated one**, because random drops LOW-EV actions while
  curation drops critical ones (Gen-H's hit% is 86–90%, *lower* than B's 91.6%, yet it
  holds). The hoped-for TV-revisit-bias improvement did not appear.
- *Resource (sum over 8):* wall **424.6 vs 445 min** (~5% faster ≈ equal within
  minute-resolution + contention-width caveats); RAM **15.81 vs 15.42 GB** (+2.5% ≈
  equal — the +16 B/row legal masks offset by ~3–5% fewer rows); nodes **0.78×** (22%
  fewer). Roughly resource-neutral.
- *Verdict:* ties + resource-neutral + a real coverage gap + much more code → not worth keeping.

**Cross-cutting conclusions.**
1. Action abstraction **can't shrink the infoset table** (hand/history-bound) → ~no RAM
   win; it changes node/branching count only.
2. The traverse is **memory-bound**: breaking the baseline's contiguous
   `regrets[row, lo:hi]` / `action = lower+i` layout (Gen-G's menu arrays) adds per-node
   cache traffic that swamps node savings; staying bet-indexed (Gen-H) stays neutral.
3. **Random > curated** for coverage robustness (B collapsed at high-T, H didn't despite
   a bigger gap).
4. **Correction to the "68–81%" above:** that was the *pre*-5.7×-rewrite cost. Post-
   rewrite, `get_hand_abstraction` is ~34% solo (this section's "abs_ids 81%→34%") and at
   8-wide contention only ~9–19% of total — the **JIT traverse dominates (~70–86%)**. So
   "abstraction is the wall-clock driver" holds only pre-rewrite; post-rewrite the
   node-traversal is the driver. A solo profile would pin the exact post-rewrite split.
5. **Match hyperparameters** or the H2H is confounded (V2.1 = penalty 0 / pruning
   `[-10,-12]`); use `cfr_ai/outputs/<setup>/metadata.csv` for V2.1's recorded RAM/nodes/duration.

**Net:** at cross-section scale the game fits in RAM and the infoset table is
hand/history-bound, so action abstraction yields no resource win. The capability/resource
lever is the **hand+history abstraction** (direction 2 above), not the action list.

### V2.1 consolidation retrain (2026-05-30) — bank the wins, ~free

**Version bump: V2 → V2.1** (hyperparameter adoption; **V3** reserved for the
abstraction redesign).

**Goal:** one consistent hyperparameter set for all 66 setups, banking the
validated wins with no structural change: **penalty=0** (less exploitable),
**pruning -10/-12** (prune-10, ~8.6% faster), the **fast `get_hand_abstraction`**
(byte-identical → strategy unchanged, ~2.3× faster training), **temporary_value
kept**, **5M iters** (10M banked). Set `VERSION="V2.1"` in `training.py`. Applied
penalty=0 + prune-10 to ALL 66 (override `_setup_configs` penalties; keep its
per-setup min_bet). Expected ~50 CPUh (~half of V2's ~106, from the abstraction
speedup) → ~6–7 h wall at 8-wide. Isolated to `cfr_ai/v2.1` (live V2 untouched).

Safe workflow that was followed: deploy the fast `information_set.py` to the box
→ train to an **isolated** dir (NOT `cfr_ai/outputs`) → validate vs current V2 by
H2H (expect ≈even-or-better; penalty=0 was ≤baseline everywhere) + the cheap-LBR
probes (1,8/2,7) → **promote** to `cfr_ai/outputs` → redeploy Docker→ECR→Lambda.

**Result:** 66/66 trained, 0 failures, all `Version=V2.1`. **CPU-h ≈ 63.7, GB-h
≈ 153.3** (settled; slightly inflated by an 8-setup redo — see the V2.1 OOM
root-cause and the process-lesson on the erroneous `systemctl stop`).
**vs V2:** explored infosets median **1.00×**, settled RAM median **0.99×**,
total train wall **0.60×** (≈40% faster, from the looser pruning).
**Conclusion:** consolidation is ~free on memory/infosets and faster to train —
banked the validated wins while preserving playing quality (see the whole-game
and per-setup H2H below).

**Process to bake in:** define the success metric *before* the redesign
(exploitability ↓ on the cheap-LBR probes; H2H ≥ current; ≥ even vs humans), and
gate every abstraction change through retrain → H2H-vs-old → redeploy.

### "penalty=0 explodes infosets ~3×" — FALSE (fabricated, then measured)

**Claimed (by me, unverified):** penalty=0 would triple infosets → 11–13 GB.
**Measured:** infoset ratio V2.1/V2 median **1.003** (range 0.98–1.08), RAM
median **0.99×**. The claim was fabricated.
**Lesson:** never assert a number before measuring it. (Worst lapse of the
project — see Process lessons §a.)

### V2.1 OOM root-cause

**Symptom:** one OOM during the V2.1 run.
**Diagnosis (data-backed):** a transient **save-prep array-copy spike**
coinciding with 8 large setups at settled concurrency near the 30.6 GiB ceiling
— **not** penalty, **not** npz writing, **not** a V2.1 regression. V2 had run the
same 8-wide/~24 GB; the OOM was timing luck.
**Fix:** in `trainer.get_diagnostic_arrays()` reorder each big array via one
fancy-index on the `[:n]` view then null the member immediately; in
`training.save_strategies()` `del fs`/`del diag` after save. A/B: old +4203 MB vs
new +1089 MB at 3M rows.
**Dead hypothesis:** "V2 had more RAM (CCX43)" — user never owned a CCX43;
everything was CCX33. Killed the regression theory.

### NFSP (Perun) reference: greedy vs sampled

- Checkpoint files verified against friend's checksums.
- **greedy** (argmax of avg policy) = production default; a *pure* strategy,
  **not Nash-approximating** and exploitable by adaptive/best-response play —
  but the only *competitive* yardstick (~0.50 vs CFR, matching the friend's even
  1004–996). Never train against it (held-out).
- **sampled** (full mixed avg policy) is the fuller, more Nash-like policy, but
  its off-policy tail errors are exactly what near-Nash CFR *does* punish → CFR
  wins ~0.76 (≈9pp above greedy); not the realistic bar.
- **Why CFR only draws ~0.50 vs greedy Perun (rather than beating it):** in a
  2-player zero-sum game a near-Nash strategy guarantees **≥ the game value vs
  *any* opponent** (never loses in expectation) and *beats* genuinely bad
  opponents (Nash play punishes gross/dominated errors). What it does **not** do
  is *maximally* exploit a competent-but-imperfect opponent — capturing greedy
  Perun's subtle residual errors needs best-response, which CFR doesn't do. So
  the ~0.50 means "doesn't best-respond to a competitive opponent", **not**
  "draws everyone regardless of quality" (a near-Nash strategy still beats an awful one).

### V2.1 vs V2 / Perun — whole-game win-rates (PRIMARY; 10k games each, ±1 SE)

| matchup | V2 | V2.1 | Δ |
|---|---|---|---|
| vs V2 (direct H2H) | 0.5000 | **0.5033 ±0.0050** | +0.33pp (<0.7σ, tie) |
| vs greedy Perun | 0.5024 ±0.0050 | **0.4864 ±0.0050** | −1.6pp (~2.3σ) |
| vs sampled Perun | 0.7558 ±0.0043 | **0.7452 ±0.0044** | −1.06pp (~1.7σ, NS) |

**Conclusion:** V2.1 ≈ V2 on whole-game strength — consolidation preserved
playing quality. The small Perun dips are not equilibrium regressions (win-rate
vs a fixed opponent isn't an equilibrium measure); LBR is the right instrument.

### Per-setup H2H, V2.1 vs V2 — the removed-penalty hypothesis (CONFIRMED)

**Hypothesis (user):** small improvement on most setups from removing the
bet-penalty.
**Result (10k deals/setup, all 66):**
- **Penalized 54 setups: 47 positive / 7 negative, sign-test p ≈ 1e-8**, mean
  +0.32%.
- **Unpenalized 12 (control): 6/6, p = 1.0**, mean ≈ 0.

**Conclusion:** confirmed. The control (which got every *other* V2.1 change but
no penalty change) shows the effect tracks **penalty removal specifically**.
**Surprise:** the whole-game H2H was a *tie* (0.5033) yet the per-setup sign test
is overwhelming — the aggregate masked a real, systematic signal because a 54-way
sign test has far more power than one whole-game win-rate.
**Disaggregation surprise:** the *largest* setups (sum≥16, min_bet=27) improved
*most* and uniformly (16/16); the weak group was **min_bet=0 (the smallest)
setups** — they held all 7 negatives. The guess "largest min_bet=0 setups do
worse" was inverted: there are *no* large min_bet=0 setups, and large setups did
best. Benefit grows with setup size.

### Repository behaviour (V2.1 vs greedy Perun, 10k)

- V2.1 ends **53.6%** of rounds with its own check (challenge) vs Perun's 46.4%
  — it challenges more.
- But its challenges are **lower quality**: 45.1% catch a bluff vs Perun's 55.5%.
  This over-calling is the source of the ~1.4pp greedy-Perun deficit — expected
  "near-Nash over-calls a fixed deterministic opponent" behaviour, not a defect.

### LBR feasibility — two surprises, one corrected mistake

- **Depth-3 LBR is INFEASIBLE.** Depth-2 on `1,8` = ~53 min/call; depth-3 ~88× →
  days/call. My "asymmetric setups (1,8/2,7) collapse the enumeration so deep LBR
  is cheap" was true at depth 1, **false at depth 3**.
- **`1,8` LBR-1:** V2.1 less exploitable than V2 on *both* seats
  (+5.044/+3.815% vs +5.111/+5.403%) — supports the removed-penalty →
  closer-to-Nash story.
- **"2,7 LBR will take 5 hours" was wrong.** I extrapolated from a slow *laptop*
  run instead of checking V2's recorded durations (box: 2,7 LBR-1 = 1562/2340 s
  ≈ 26/39 min/seat). The laptop is ~3× slower for this workload, and it was
  *redundantly recomputing the already-recorded V2 baseline*. Fix: kill it,
  recompute only V2.1's 2,7 on the box.
  **Lesson:** check recorded data before declaring something infeasible.

### min_bet=27 experiment (sum-13–15 setups) — the standout result

**Idea (user):** raise min_bet to 27 for the 14 sum-13–15 setups — faster and
fewer same-iteration node revisits (problematic under the algorithm). Train at
V2.1 params, isolate to `cfr_ai/exp_mb27`, then H2H each vs V2.1.
**H2H result:** min_bet=27 **beats V2.1 in 13 of 14** (mean **+0.64%**, sign-test
p ≈ 0.0018); edge grows with size (sum13 +0.30% → sum15 +1.16%); only `7,7` (the
lone symmetric setup) regressed (−0.76%).
**Speedup:** **~2.8× wall** (1359 → 486 min for the 14), **~2.2× fewer explored
infosets**. Per setup 70–123 min → 27–41 min.
**Surprise:** min_bet=27 isn't just *faster*, it's *better*. Likely because in
card-rich setups (13–15 cards) sub-straight claims (high card/pair/two-pair) are
almost always true → strategically near-trivial *and* exactly where the
node-revisiting hurts convergence; pruning them sharpens the meaningful play.
**Implication:** the min_bet schedule for sum-13–15 is mis-set in V2/V2.1 —
raising to 27 is a free speedup *and* a small quality gain. Candidate for a future
version. (Caveat: H2H isn't a pure-game comparison — the mb27 agent treats V2.1's
sub-27 opens as fresh opens; but "V2.1 wastes EV on trivial low claims" is itself
a real reason it's worse here.)
**RAM:** min_bet=27 also cut training RAM **~50%** (42.7 → 20.9 GB total across the
14; per-setup 44–58%, scaling with size) — pruning the low-claim subtree drops a
large slice of infosets.
**vs greedy Perun (per-setup, paired 10k/cell, seed 0):** **no degradation.** Mean
diff (mb27 − V2.1) = **+0.0006**, median +0.0007; 18/27 configs within ±0.005; the
3 small dips (`8,5` −0.016, `4,11` −0.015, `3,10` −0.011) are offset by gains
(`6,8` +0.026, `7,8` +0.015). **Verdict: adopt min_bet=27 for sum-13–15** —
~2.2–2.8× faster, ~50% less RAM, 13/14 H2H wins, flat vs the production (greedy
Perun) opponent.

### LBR-2 cost & exact-extrapolation (sum-6 setups) — measure-don't-assume

**Sampled LBR-2** (n_belief=300, n_lbr_hand=500, depth 2): `1,5`=113 s, `2,4`=606
s, `3,3`=388 s.
**LBR-2 reveals exploitability LBR-1 missed:** `1,5` +3.25/+3.67% (LBR-1 was
+1.2/−0.14%), `2,4` +4.25/+4.10%, `3,3` +2.13% (LBR-1 was −0.31%). So at 5M iters
these "easy" small setups are **not** as solved as LBR-1 suggested — real
motivation for the undertraining experiment.
**Cost model (verified from code):** cost = (LBR hands) × (beliefs), **linear in
both**; depth-2 double-sampling processes S1/S2 in separate loops (not a
cross-product → not quadratic). "Exact/no-sampling" uses full raw populations
`C(24,lbr)` hands × `C(24−lbr,opp)` beliefs.
**Exact factors:** `1,5` ≈97×, `2,4` ≈23×, `3,3` ≈18× → **exact LBR-2 ≈ 3,3
1.5–2 h, 1,5 2.5–3 h, 2,4 3.5–4 h**.
**Surprise:** the deal-combination "absolute cost" argument (1,5 cheapest, fewest
deals) is **refuted by data** — measured per-(hand,belief) cost varies ~5× (`1,5`
6.8 ms vs `3,3` 1.3 ms; the 5-card best-responder is far heavier). So `3,3` is
the cheapest exact, not `1,5`. Per-pair cost dominates the deal count.
**Lesson:** even a clean combinatorial cost argument can invert once measured.

**V2.1 baseline LBR-2 @ (1000,1000), depth-2 — the 5M anchor** (measured on all 9
rounds-1–5 setups; values in the local V2.1 `summary_of_all_runs.csv`).
Exploitability (start-p0 | start-p1): `1,1` +0.06% · `1,2` +0.55/+0.66% · `1,3`
−1.35/+0.95% · `2,2` +2.12% · `1,4` +1.37/+3.93% · `2,3` +1.80/+2.82% · `1,5`
+3.70/+5.82% · `2,4` +5.12/+5.48% · `3,3` +3.51%. **Real headroom at 5M:** the
bigger early-game setups carry **3.5–5.8%** depth-2 exploitability (not solved);
the tiniest (`1,1` ~0%) is solved. Durations validated the (1000,1000) cost
extrapolation (`3,3` ~40 min, `2,4` ~25 min vs estimates ~43/~27 → full sweep
~3–5 h). Penalty-removal signal also shows at depth-1 (both probes, both seats): V2.1 `1,8`
+5.04/+3.82% (V2 +5.11/+5.40%) and `2,7` +3.76/+3.05% (V2 +6.26/+4.54%) — V2.1
**less exploitable on every seat** (closer to Nash, as hypothesised). NB V2.1's
`2,7` LBR-1 ran ~2.6× slower than V2's (~2.8 h vs ~65 min) — likely penalty=0 →
more betting → deeper LBR tree, a measurable side-effect of the removal. This row
is the baseline the **10/20/40M early-game undertraining
sweep** compares against (the one clean substrate where iteration→LBR-2 is
affordable and abstraction-unconfounded).

### Early-game iteration sweep (rounds 1–5, 10/20/40M) — RESULTS (2026-05-31)
**Setup:** trained the 9 sum-2–6 setups at **10M/20M/40M** (V2.1 params: penalty 0,
pruning −10/−12, temp_value on, min_bet 0, seed 42), then LBR-2 @ **(1000,1000)** at
each, vs the stored 5M baseline. 27 variants, **0 failures**. The one clean
substrate where iteration→LBR-2 is affordable and abstraction-unconfounded.

LBR-2 (depth-2) exploitability % by iteration count (per seat; SE ≈ 0.3–0.5 pp;
full table in `scratch/_lbr2_curve.txt`):
```
setup p     5M       10M      20M      40M
1,1  p0   +0.055   +0.024   +0.028   +0.017
1,2  p0   +0.553   +0.202   +0.097   +0.093
1,2  p1   +0.656   +0.508   +0.400   +0.322
1,3  p1   +0.952   +0.429   +0.167   +0.091
2,2  p0   +2.119   +1.710   +1.884   +1.525
1,4  p1   +3.931   +3.451   +2.607   +2.429
2,3  p0   +1.800   +1.127   +0.592   +0.264
2,3  p1   +2.824   +2.491   +1.764   +1.457
1,5  p0   +3.703   +2.806   +2.417   +1.714
1,5  p1   +5.821   +4.355   +3.391   +3.254
2,4  p0   +5.118   +3.571   +3.015   +2.768
2,4  p1   +5.481   +4.573   +3.918   +3.593
3,3  p0   +3.510   +2.913   +2.340   +2.050
```
(`1,3` p0 / `1,4` p0 omitted: near-zero/noise — `1,3` p0 is negative, `1,4` p0
non-monotone within ~2×SE.)

**Findings:**
1. **5M badly undertrains the early game** — every non-trivial setup roughly
   *halves* its LBR-2 from 5M→40M (`2,4` p0 5.12→2.77, `1,5` p1 5.82→3.25).
2. **20M clearly beats 10M everywhere** — the headline question, confirmed.
3. **Even 40M hasn't converged on the bigger early-game setups** — `2,3`/`1,5`/`2,4`/`3,3`
   still drop meaningfully 20M→40M (Δ 0.25–0.70 pp ≫ noise); doubling **still pays
   past 40M** there. Only the tiny `1,1`/`1,2`/`1,3` flatten by ~20M.
4. Two non-monotone wiggles (`1,4` p0, `2,2` p0) sit within ~2× the LBR-2 SE → noise.
5. **Cost:** 40M ≈ 1 h/setup → **9.73 CPU-h** for the 9 (~1.5–2 h wall 8-wide); cheap
   for the early game, but the big setups are 10–20× heavier even at 5M.
6. **The gain is opponent-dependent:** vs *greedy Perun*, 5M→40M is **+0.23 pp (flat)**
   — a fixed non-adaptive opponent can't see lower exploitability. But the combined
   hybrid (V2.1 with rounds-1–5 → 40M and sum-13–15 → mb27) beats V2.1 **0.5100 ±0.0050**
   (+1.0 pp, ~2σ) whole-game. The exploitability win is real, but cashes against a
   *best-responder*, not against Perun.

**Implication:** the early game is far more undertrained than the 5M production budget
assumes, and per-iteration cost is cheap there — so more iterations are a high-ROI
quality lever for rounds 1–5 (squares with eval-by-resource: cheap iters → lower true
exploitability), the bigger early-game setups wanting **>40M**. Judge the payoff by
LBR / per-setup H2H, not Perun win-rate.

---

## Methodology & affordability reference

### Per-variant pipeline (per setup)

The orchestrator (`cfr_ai/scripts/experiments_run.py`) is a per-setup DAG runner
(built; validated end-to-end on a throwaway (2,2) variant):

```
train ─────┬─→ H2H vs V2 baseline (10k MC deals)   ← primary ranking signal
           └─→ LBR battery  (affordable sums ≤ --lbr-max-sum, default 6 =
                             2,2/3,3 only; per-setup depths from LBR_DEPTHS:
                             2,2 → LBR-1/2/3, 3,3 → LBR-1/2, else LBR-1.
                             4,5 LBR-1 took 3.5-5.5h, so excluded)
```

`H2H` and `lbr1` become eligible the moment their `train` completes and are
independent of each other. The scheduler keeps `--workers` (default 8) slots
filled with whatever is eligible across all pipelines, so small-setup training
finishes early and frees cores for H2H/LBR while big-setup training continues.
Resumable: a job whose training output already exists, or whose metric is already
in `experiments/_results.json`, is skipped.

H2H is the primary ranking signal. LBR-1 is the inside view, gated by an
affordability sum (`--lbr-max-sum 9` → (2,2)/(3,3)/(4,5) only). The experiment
LBR runs **without** `--update-summary` (that writer is hardcoded to the
production summary and ignores `--setup-dir`); metrics are parsed from stdout into
each variant's `experiment_summary.csv`.

**Periodic in-training LBR is OFF by default** (`--train-exploitability` to
enable). The Phase 2 evidence overturned the earlier "cost negligible" assumption:
a single LBR-1 on a sum≥10 setup costs *hours*, so 5 snapshots per training run
would be catastrophic on (5,7)/(6,6)/(3,9)/(5,11)/(9,11). Convergence judgement
falls back to the utility log + the post-train LBR-1 stage on the small setups.

Isolation: training writes each variant to `cfr_ai/experiments/<tag>/` via the
new `--out-root` flag (copies `history.csv` + `information_set.py` so H2H can load
it; leaves the production summary untouched).

### H2H methodology

- 10k Monte Carlo deals (`head_to_head.py --monte-carlo --num-deals 10000`)
- Model A = V2 baseline (current `cfr_ai/outputs/<setup>/`)
- Model B = variant being tested
- Reported number: average payoff to Model B per deal, in [-1, +1]
- SE ≈ 1/√10000 = 0.01, so margin ≥ 0.02 is detectable signal
- Run on Hetzner once we pivot (8 vCPUs free, fan out 8 setups in parallel)
- `head_to_head` reads `strategy.npz` directly (no sparse-mmap staging);
  `strategy_io.load_strategy` is functionally identical across the V2/V2.1 code
  versions, so the box (older strategy_io) gives identical H2H results.

### LBR-K affordability (measured at the orchestrator's (300, 500) sampling)

| Setup | LBR-1 | LBR-2 | LBR-3 |
|---|---|---|---|
| (2,2) | ~80s ✓ | ~96s ✓ (measured) | ~4-4.5 min ✓ (measured) |
| (3,3) | ~250s ✓ | ~7.5 min ✓ (measured) | (deeper than the wired battery; not run) |
| (4,5) | (~100+ min at (500,1000); fall back to (300,500) → ~30 min) | (skip — expensive) | — |
| (5,7), (6,6), (3,9) | (300,500) ≈ ~hours | (skip) | — |
| (5,11), (9,11) | (300,500) ≈ many hours; **skip** | — | — |

Rule of thumb: run LBR-1 at the sampling that finishes in <1h. For (5,11) and
(9,11) we rely entirely on the H2H scalar for ranking.

**Additional measured LBR timings (from the post-V2.1 log):**
- Sweep 2 small-setup runs (1 core each, (300,500) sampling, seed 42): (2,2)
  LBR-2 ~96s, LBR-3 ~225-262s; (3,3) LBR-2 ~440-449s.
- exp3b probe LBR-1 (asymmetric setups collapse the enumeration): 1,8 ~4.5 min;
  2,7 ~1 h. (Recorded box V2 baseline durations: 2,7 LBR-1 = 1562/2340 s ≈ 26/39
  min/seat.)
- **Depth-2 on `1,8` = ~53 min/call; depth-3 ~88× → days/call → depth-3
  INFEASIBLE.** The "asymmetric setups collapse the enumeration so deep LBR is
  cheap" intuition holds at depth 1 but breaks at depth 3.
- **Sampled LBR-2** (n_belief=300, n_lbr_hand=500, depth 2): `1,5`=113 s,
  `2,4`=606 s, `3,3`=388 s.

**LBR-2 cost model (verified from code):** cost = (LBR hands) × (beliefs),
**linear in both**; depth-2 double-sampling processes S1/S2 in separate loops
(not a cross-product → not quadratic). "Exact/no-sampling" uses full raw
populations `C(24,lbr)` hands × `C(24−lbr,opp)` beliefs. Exact-vs-sampled
factors: `1,5` ≈97×, `2,4` ≈23×, `3,3` ≈18× → **exact LBR-2 ≈ 3,3 1.5–2 h, 1,5
2.5–3 h, 2,4 3.5–4 h**. Note the per-(hand,belief) cost varies ~5× (`1,5` 6.8 ms
vs `3,3` 1.3 ms — the 5-card best-responder is far heavier), so `3,3` is the
cheapest *exact*, not `1,5`; per-pair cost dominates the raw deal count.

(LBR is a *lower* bound on true exploitability and tightens — rises — with depth.)

### Output storage

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

`experiments/` should be gitignored locally; mirror to local from Hetzner after
each variant completes — **but skip `diagnostic.npz` on the rsync** (local drive
is tight; experiments don't need diagnostics on local since the analysis happens
against the metadata + strategy artifacts only). Diagnostic files stay on the
Hetzner box and get deleted with it.

### Utility-log retrospective (post-experiment analysis)

After all variants run, for each (variant, setup) pair extract from the saved
utility log:

| Feature | Computation |
|---|---|
| `mean_30` | Mean of (P0 + P1) chunk utilities over last 30% of iters |
| `std_30` | StdDev of the same chunks |
| `slope_50` | Linear regression slope over last 50% of iters (drift detection) |
| `burn_in_delta` | Mean P0+P1 in first 30% minus mean in last 30% |

Cross-correlate these against the variant's H2H + LBR-1 result. If features
predict the outcomes, utility log becomes a cheap proxy for "did this variant
converge?" — useful for cheaper future sweeps. If they don't predict, treat
utility log as a sanity check only.

### Budget

| Item | Wall | Cost |
|---|---|---|
| Phase 2 LBR-1 to pivot (sums ≤ 14) | ~20h | ~€2.5 |
| Sweep 1 (3 pruning variants) | ~7.5h | ~€1 |
| Sweep 2 (1 iter variant) | ~3h | ~€0.4 |
| Buffer | — | ~€1 |
| **Total estimated** | **~30h after pivot** | **~€5** |
| Productionization run (if winning variant emerges) | TBD | up to €5-10 (separate decision) |

Per-sweep estimates as originally scoped:
- **Sweep 1:** 3 new training runs per setup × 8 setups + LBR-1 + LBR-2 + LBR-3
  (where affordable) + H2H rounds vs baseline ≈ **~7.5h total wall** on 8 vCPUs ≈
  €1.
- **Sweep 2:** 8 setups × 1 new training (10M = ~2× the 5M time) ≈ **~6h wall**
  on 8 vCPUs ≈ €0.8 (plus ~€0.5 for the conditional 3M arm if run).

---

## Forward roadmap / open experiments

### Prioritized roadmap (post-V2.1)

1. **Action abstraction — TRIED (Gen-B/G/H), all shelved** (see "Action abstraction —
   three generations" above). The "group actions + shorten arrays" thesis did NOT
   hold: action abstraction can't shrink the infoset table (it's hand/history-bound →
   ~no RAM win), and the traverse is memory-bound, so a per-row menu layout *adds* cost
   (Gen-G hit 2.3–4.7×). The one **surviving live idea** is the *augmenting* bluff/value
   **macro** — a node-resolved, hand-aware action ADDED to the full set (see "Decided
   next steps" below) — which can *raise* capability rather than just shrink the tree.
   Don't re-propose grouping / array-shortening without reading the three-generations record.
2. **Hand-abstraction quality** — encode *weaknesses* / bluff-credibility (not
   just strengths), steered empirically by the human-games analysis below.
   History abstraction is more robust → lower priority.
3. **Evaluation (run it to *lead* the abstraction work):**
   - Pull **human-vs-Morana games (1k+) from DynamoDB** and analyse — the
     empirical weakness diagnosis that should steer the hand-abstraction redesign.
     Access: needs perms in the friend's account (IAM there is ECR-only — clear
     first).
   - **Perun yardstick (resolved):** use **greedy** (`--nfsp-greedy`) — the
     competitive ~0.50-vs-CFR opponent (matches the friend's even result);
     sampled is a secondary inflated view. A genuinely *stronger* held-out
     opponent for adaptive-exploitability testing remains a nice-to-have.
   - Reusable: the **asymmetric-setup cheap-LBR trick** (1,8/2,7 → LBR in
     minutes) as an exploitability probe for any abstraction change.
4. **Skip C++** — numba ≈ C on the hot loops (~1.3–2×, expected-revert); and
   post-redesign the bottleneck is the JIT'd abstraction, not numba.
5. **Expansion (long-term)** — 3+ players (NB: not 2-player zero-sum, so CFR loses
   its equilibrium guarantee — research-grade) and non-standard rules. After the
   core 1v1 AI is strong + well-evaluated.

### Decided next steps (2026-06-02) — priority order

After Gen-B/G/H (all shelved — see "Action abstraction — three generations"), the
agreed plan, in order. Morana = CFR agent (V2.1 lineage); Perun = NFSP agent.

**1. Bluff macro — try FIRST. An *augmenting* action, not a lossy abstraction.**
Add ONE abstract action, **"bet the bluffiest legal set"** = argmin over legal claims
of `(p − g)` (hand-aware minus a-priori existence prob), resolved at the node from the
actual hand's `p`/`g`. **ADDED** to the full action set (coverage untouched), **NOT** in
the infoset key. (Truthful macro = argmax `p` is the natural follow-on if the bluff one helps.)
- *Why:* the abstraction buckets many hands into one `abs_id`, so the shared per-bucket
  strategy can't pick *this hand's* bluffiest/most-credible claim — the node-time
  resolution injects that discarded info back. A transposition in the tree but a new
  policy capability → it can **beat** V2.1, not just tie (unlike B/G/H). Highest
  upside-per-effort.
- *Impl:* reuse `cfr_ai/abstraction/probs.py` (`p_vector`, `g_vector`). Precompute per
  hand per iteration `bluff_resolve[last_bet]` = argmin `(p−g)` over claims `> last_bet`
  (one O(88) backward sweep; cache `p` by hand-signature à la old Gen-B). Keep regrets
  **bet-indexed** (do NOT use a Gen-G-style menu layout — that's what made Gen-G
  memory-bound at 2.3–4.7×); the macro is one extra column whose expansion uses the
  precomputed lookup. The old trainers were deleted → build a small fresh trainer (or
  re-derive from the Gen-G/H designs above); kept modules (`probs`, `action_menu`,
  `structure`, `referenced`) supply the primitives.
- *Cost:* expect **<2×, ~baseline+ε**. Only real new cost is `p_vector` prep (a prep-side
  minority term; cache it). Extra revisits = +1/node (transposition; short-circuits on the
  TV cache).
- *Verdict test:* train MATCHED to V2.1 (penalty 0, pruning [-10,-12], min_bet per setup,
  5M) via the production `training.py` path; H2H + hit% + (where affordable) LBR-1 vs V2.1
  on a few setups incl high-T. WIN = less exploitable / >50% H2H.

**2. Hand-abstraction diagnostic (roadmap #2/#3) — descriptive; steers the redesign.**
Per (responder ∈ {Morana, Perun, humans}, set-type of the standing bet), aggregate:
truthful% / bluff% (via `p − g`: report the distribution or a threshold; it measures
hand-**justification**, not whether the claim turned out true), check/challenge%, and
**reference-overlap** with the last / 2nd-last / 3rd-last bet (shares a rank or suit, via
`referenced.py` `CLAIM_RMASK`/`SMASK`). Also condition on round/total (`p`,`g` depend on it).
- *Data:* Morana-vs-Perun via self-play transcripts (log per-decision hand + history +
  action; Perun = greedy NFSP). Humans = **DynamoDB** (1k+ human-vs-Morana games; needs
  IAM perms in the friend's account, currently ECR-only — clear first, per roadmap #3).
- *Tools:* `probs.py` (p/g), `referenced.py` (rank/suit masks), `action_menu.py` (set-type
  ranges), `structure.py` (claim decomposition).
- *Caveat:* descriptive — a difference is a *candidate* leak, not a proven one; confirm with
  exploitability / H2H.

**3. CFR-without-MC (CFR+ + pruning) — phase 2; primarily a YARDSTICK.**
Full-traversal **CFR+** (regret-matching⁺ + linear averaging) + **regret-based pruning**.
Feasible because most legal actions are rubbish → pruning collapses effective branching →
cheap full sweeps; gains variance-free convergence (the Cepheus recipe).
- *Primary use:* a near-optimal **exploitability yardstick on the small/shallow setups**
  (abstraction near-lossless there) to measure true abstraction loss → feeds #2.
- *Caveats:* per-iteration cost ~ `b^(opp-depth)` → setup-dependent (great small/shallow +
  high-min_bet; harder deep mb0). **Check the DCFR precedent first** (closed at 10–18× slower
  vs ES — see closed chapters): confirm whether that DCFR was CFR+/pruned or naive; if naive,
  the CFR⁺-with-pruning version is genuinely untested. More RAM (explores opponent branches).

*Contested:* the "abstraction = 68–81% of training time" figure above is pre-`get_hand_abstraction`-rewrite;
post-rewrite, and at 8-wide, the **JIT traverse dominates (~70–86%)** — a solo profile would settle it.
*Kept reusable code:* `cfr_ai/abstraction/{probs,probs_jit,structure,referenced,action_menu}.py`
(UNTRACKED — commit before relying on it).

### Open / planned experiments

- **Action abstraction** — the active next experiment (see roadmap #1). Big topic.
- **Next retrain (a "V3"): fold in the validated improvements** (unless a better
  per-round mod supersedes them): V2.1 params + **min_bet=27 for sum-13–15** +
  **more iterations for the early game** (rounds 1–5 want ≫5M; the bigger ones
  still gain past 40M) + whatever the action-abstraction work yields. Big setups
  dominate cost, so budget iterations accordingly.

*(DONE — see the chronological log above: the iteration-count × LBR-2 undertraining
sweep, and the min_bet=27 evaluation. Both validated; the action work + V3 retrain
will incorporate them.)*

### Future sweeps (deferred unless a sweep motivates them)

- **Lower penalty + multi-visit:** requires gating the
  `temporary_value`/`last_touched` skip mechanism behind a flag in
  `_traverse_jit`. Promising but riskier. Run after we know if 5M is enough
  (Sweep 2). (Partly addressed by exp3b's no-temp-value isolation.)
- **Probabilistic 80% pruning:** replace binary `prune_feast` with per-action
  stochastic retention. Bigger code change in `_traverse_jit`; defer unless the
  sweeps suggest exploration matters.

### Productionization

Reserved for a final tightly-scoped run if any experiment shows clear
improvement. Would be:
- Apply winning hyperparams to all 66 setups
- Full LBR-1 + LBR-2 on the affordable subset
- Deploy via Docker → ECR → Lambda (with the np.searchsorted refactor in place;
  see lambda-side notes below)

### Lambda-side work (done; does not block experiments)

**This is done as of the second compaction.** Final state:
- `strategy_io.load_strategy_for_agent()` returns a `FlatStrategyAgent` with
  `np.searchsorted` lookup. No numba on the agent path.
- Sparse-mmap storage layout (uint16 quantised values, ~10× smaller than dense
  int16, ~30× smaller than dense fp32 per setup).
- `cfr_ai/scripts/stage_for_docker.py` converts compressed `strategy.npz` →
  sparse layout at build time only; local `cfr_ai/outputs/` stays compact.
- Lambda: **256 MB** (was 2048 MB before refactor, 1536 MB intermediate). Cold
  ~2.5-3.5s, warm ~30-50ms, peak memory ~115-170 MB. Validated live in production
  (Morana traffic across setups up to sum 18, 0 errors; synthetic (11,11) at 133
  MB). The retired multi-Lambda comparison was dropped from the README.
- ECR storage: ~$0.11/month for the 1.13 GB image (was $0.19 at 1.85 GB before
  sparse format).
- All test cases pass including (1,1) opening-move and (7,8) worst-case.

### CFR-vs-NFSP evaluation (tools built; 0.76 mismatch RESOLVED → greedy = even yardstick; full suite run)

A second evaluation axis: benchmark CFR against the deployed NFSP agent (Perun).
NFSP artifacts were extracted from the `blef-nfsp-lambda` ECR image (friend's
acct, eu-west-2) into `nfsp_ai/artifacts/` (gitignored); the 1v1 deck-24 model is
`nfsp_inference_24_1v1.pt`. Both agents share `determine_action(game_state)`; the
Blef engine (`shared.api.simpleschema_local_manager`) deals/resolves. Needs torch
(installed locally) — runs local, not on the box.

**Tools (all in `cfr_ai/analysis/`, outputs gitignored):**
- `cfr_vs_nfsp.py` — per-setup + full 11×11 matrix, role-alternated, CFR win-rate
  per (start-size, non-start-size). The bread-and-butter for evaluating a CFR
  *version* (`--cfr <tag>` / `--cfr-folder`, via the new backward-compatible
  `agent.set_outputs_base`). `--setups auto|all|<list>`.
- `cfr_vs_nfsp_games.py` — whole-game generator (real mechanic: 1v1 start, loser
  +1 & opens next, eliminate at 11). One ~123-byte row per round (deal[24] +
  history[88] + sizes/seats + loser + existence) → `.npy` structured array. CFR
  opens exactly half (`cfr_seat = game_id % 2`).
- `diagnose_games.py` — Phase-1 analyzer: loss attribution, check calibration vs
  ground-truth existence, bet/bluff & response by category.

**Supporting:** all 66 setups in `cfr_ai/outputs/` were staged to the sparse-mmap
layout **in place** (kept `strategy.npz`; +~1 GB, gitignored) so the local agent
reads via mmap — cheap per-round reloads, low + shared RAM across workers. This
also de-risked the matrix tool's RAM.

**Early findings (vs *sampled* Perun; validated harness, existence cross-check 361/361):**
- Full 11×11 matrix: CFR beats this NFSP in 120/121 configs, mean win-rate 0.603,
  edge largest at small hands, → ~parity only at the biggest symmetric configs
  ((11,11)=0.465). Saved at `cfr_nfsp_matrix_10k.csv`/`.png`.
- Whole games (partial, 4000 of 10000 stored in `cfr_vs_nfsp_games_current.npy`):
  CFR game win-rate **0.762** — internally consistent with the matrix but
  contradicts the friend's even 1004–996.

**Resolution (later):** the 0.76 was the **sampled** avg-policy Perun — CFR
punishes its off-policy tails. Switching Perun to **greedy** (`--nfsp-greedy`,
the production default) reproduced the friend's even result: V2 vs greedy =
**0.5024**, matching 1004–996. So greedy is the competitive, production-realistic
yardstick (though *not* Nash-approximating); sampled is a secondary, ~9pp-inflated
view. The full whole-game suite was then run vs both (see "V2.1 vs V2 / Perun —
whole-game win-rates"). The `diagnose_games.py` story (CFR over-challenges senior
bets; Perun under-challenges) holds against greedy too (see "Repository
behaviour"). Phase 2 (full-information oracle correctness) stays shelved.

### Subgame solver — shelved 2026-05-28

Phase 2's actual LBR-1 timings ruled out online subgame at any non-trivial setup
(sum=10 setups taking 5-6h+ for LBR-1 at production sampling; subgame is a
different game-tree shape but in the same cost neighbourhood). Code moved to
`cfr_ai/archive/subgame/`; see that directory's README for the cost evidence, the
revival path, and the three directions that remain open (offline KL-regularised
refinement, tiny-setup-only online subgame, abstraction-quality work) if subgame
becomes interesting again.

The strategy-load fast path that was originally motivated by subgame's online
budget shipped anyway for its own benefits (256 MB Lambda tier, 30-50 ms warm
decisions). No production code needs to be torn out for the shelving to be clean.

---

## Process & operational lessons

These are the operational discipline rules earned during the experiment phase, so
we don't repeat the mistakes.

**a. Fabrication is the cardinal sin.** Two instances: the "3× infosets" claim
(see the FALSE penalty=0 infosets entry above) and writing A/B numbers (+55 vs
+4140 MB) into a cron *before the test ran*. Both corrected by measuring. Rule:
label measured-vs-hypothesis; never state a number you haven't computed.

**b. Don't mutate on a hunch in a check turn.** An erroneous `systemctl stop`
killed 8 in-flight trainings (batched a mutation into a "maybe it's done" check) —
this is the 8-setup redo that inflated the V2.1 CPU-h. Rule: check turns are
read-only; only `done==N` triggers a stop.

**c. Investigate before declaring infeasible** (see LBR feasibility entry: the "5
hours" that was ~1 h on the right machine — extrapolated from a slow laptop run
instead of checking V2's recorded box durations; check recorded data first).

**d. Even a clean combinatorial cost argument can invert once measured** (the
LBR-2 "1,5 is cheapest" deal-count argument was refuted by the ~5× per-(hand,
belief) cost variance; `3,3` is actually the cheapest exact). Measure, don't
assume.

**e. Periodic in-training LBR costs hours — overturned the earlier "cost
negligible" assumption.** The Phase 2 evidence showed a single LBR-1 on a sum≥10
setup costs *hours*, so 5 in-training snapshots would be catastrophic on the mid/
big setups. Hence periodic in-training LBR is OFF by default
(`--train-exploitability` to enable); convergence judgement falls back to the
utility log + post-train LBR-1 on the small setups.

**f. IPv6 SSH is flaky** — exactly one ssh call per message; `cd /root/blef_ai`
first; explicit `.venv/bin/python`; `tr -d '\r'`; collapse checks to a single
`echo RESULT ...` line.

**g. Be precise when killing processes** — a psutil kill matching module-name
*substrings* also matched bash wrappers whose command text contained those
strings. Match exact pids/cmdlines; build needles by concatenation to avoid
self-matching the `-c` snippet.

**h. Long runs on the box use `systemd-run` transient units** (survive SSH
teardown; `-p MemoryMax=` as an OOM backstop), not tmux.

**i. head_to_head reads `strategy.npz` directly** (no sparse-mmap staging);
`strategy_io.load_strategy` is functionally identical across the V2/V2.1 code
versions, so the box (older strategy_io) gives identical H2H results.
