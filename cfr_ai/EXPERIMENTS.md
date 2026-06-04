# Blef CFR — Experiment Record

The authoritative record of every experiment run on the Blef CFR trainer: the
design, the measured results, the conclusions, and — where they were instructive —
the surprises versus what theory predicted. The `README.md` carries only the final
adopted settings; this document carries the full reasoning. Numbers are **measured**
unless explicitly labelled an estimate or hypothesis.

Blef is a 1v1 imperfect-information bluffing game. Every bet is a claim about all
cards on the table combined; a player escalates by recombining the standing claim
with their own cards (truthfully or as a bluff), and the pivotal action is the
*check* — "does the claimed set exist?". Unlike poker there is no opponent-equity
estimation, which makes the existence probability of each claim the natural object
the abstraction and the action set are built around. The trainer is external-sampling
MCCFR, JIT-compiled with numba over flat arrays.

**Box:** Hetzner CCX33 (8 vCPU, ~30.6 GiB RAM), `/root/blef_ai/.venv/bin/python`.
**Cross-section setups** used for most sweeps (8, to saturate the 8 vCPUs): (2,2),
(3,3), (4,5), (5,7), (6,6), (3,9), (5,11), (9,11) — spanning tiny-symmetric (afford
deep LBR) to the (9,11) training pole (~1.6 h at 5M iters).

---

## 1. Version history

Each version is the previous one plus a specific, validated change. The line runs
V1 → V2 → V2.1 → V3; every experiment below is tagged with the version it fed.

| Version | What changed | Headline result | Location |
|---|---|---|---|
| **V1** (baseline) | Original production CFR: pure external-sampling MCCFR, per-setup penalty, pruning −20/−22, 5M iters, all 66 setups. The first deployed model. | The reference point. | `cfr_ai/archive/v1` (npz) |
| **V2** | Full 66-setup retrain on the new numba JIT trainer (replacing the historical pure-Python trainer). Same hyperparameters as V1. | **5.7–9× faster** training; game values, structural metrics, and (11,11) H2H all within noise of V1 — a faithful re-implementation. | `cfr_ai/archive/v2` |
| **V2.1** | Hyperparameter *consolidation*: penalty **0**, pruning **−10/−12**, the byte-identical fast `get_hand_abstraction`. No structural change. | ≈ V2 strength, **~40% faster** to train and slightly *less* exploitable (penalty removal moved it closer to Nash). | `cfr_ai/outputs` (until V3) |
| **V3** | V2.1 + two independent additions: **augmenting action macros** (3 per infoset, for total ≥ 7) and **min_bet = 27 for total ≥ 13**. | Beats V2.1 head-to-head (**0.535** whole-game) and **flips greedy Perun from a loss to a win** (0.489 → **0.538**). Current anchor. | `cfr_ai/outputs` |

V1 was historically tagged `v0`/`v0_baseline`; the archive is now consolidated to a
single `v1` snapshot. V2's hyperparameters (pruning −20/−22, per-setup penalty
0.05/0.1) are the signature distinguishing it from V2.1 — the box working tree
`cfr_ai/outputs` is V2, while `cfr_ai/v2.1` is the V2.1 anchor.

---

## 2. Hyperparameter consolidation (the V2 → V2.1 sweeps)

Four experiments, all run against the V2 baseline, established the V2.1 settings.
Method throughout: **H2H** = 10k Monte-Carlo deals, variant-minus-baseline payoff in
[−1, +1] (per-setup SE ≈ 0.01, so a single setup is signal only at |val| ≥ ~0.02);
ranked in aggregate by mean, t-statistic, and a sign test. **LBR-K** (local best
response) is a lower bound on true exploitability that tightens — rises — with depth.

### 2.1 Pruning range (Sweep 1) → adopt −10/−12

Three pruning variants vs V2 (−20/−22) over the 8 cross-section setups.

| variant (threshold, min_regret) | training speedup (min_bet-0 setups) | mean H2H | sign test | verdict |
|---|--:|--:|---|---|
| `prune-10` (−10, −12) | **+8.6%** | −0.0015 | 8/5, p=0.58 | **adopt — free speedup** |
| `prune-5` (−5, −7) | +13.2% | −0.0054 | 11/2, p=0.022 | small real quality cost |
| `prune-2` (−2, −4) | +19.2% | −0.0101 | 12/1, p=0.003 | clear quality cost |

Looser pruning speeds only full-action-space (min_bet-0) setups; min_bet-27 setups
have little to prune (~0–5%, noise). prune-10 buys ~8–9% cheaper training at
statistically zero quality cost (LBR-1 on 4,5 corroborates: 4.45% → 4.68%, within
noise). prune-5 and prune-2 trade increasing speed for increasing exploitability and
were rejected. **Adopted into V2.1.**

### 2.2 Iteration count (Sweep 2) → 10M is strictly better, but banked

10M vs the 5M V2 baseline, pruning held at −20/−22. 10M won on every non-trivial
setup (13/13 H2H positive, +1 to +2.4 pp, sign-test p ≈ 1e-4) and was less
exploitable on all measured seats. Critically, the **exact-(2,2) LBR gap widens with
depth** (LBR-1 0.15 pp → LBR-2 0.16 → LBR-3 0.28), the signature of genuinely better
convergence rather than a self-play artifact; (3,3) LBR-2 improved 0.82 pp.

The conditional 3M arm was dropped (5M is, if anything, mildly under-trained, so 3M
could only be worse). **Production stays 5M:** 10M is a one-time ~1.75–2× training
cost for a durable +1–2 pp edge, **banked as the "throw-compute" lever** to pull only
after cheaper ideas are exhausted. The saving from 10M concentrates on min_bet-0
setups (5,7: 1.28×, 3,9: 1.31×) where late-stage pruning trims maturing branches,
while min_bet-27 setups stay ~2× — the same action-space pattern as Sweep 1.

### 2.3 Penalty and temporary-value (Sweep 3 + probe exp3b) → drop the penalty, keep temp-value

The per-setup **bet penalty** taxes bet-lines vs checks, intended to speed training
(shorter sampled sequences) and to relieve the theoretical unsoundness of the
`temporary_value` (TV) transposition cache. Removing it (penalty-0) won H2H by
~+0.7 pp (5/6 setups), monotonically to zero (full removal ≥ halving on every 0.1
setup), and — decisively — was **less exploitable** on every seat of the cheap-LBR
probes (1,8: −2.37 pp on sp1; 2,7: −2.46/−1.76 pp), never worse. It was also
duration-neutral despite touching ~1.5× more nodes, because abstraction interning
(§2.4) dominated the clock, not node count. The penalty was distorting the strategy
for no speed gain. **Adopted (penalty = 0) into V2.1.**

The companion `--no-temporary-value` isolation kept TV: removing it re-traversed
85–89% of node visits (4,5: 6.2×, 5,11: 8.8× node blow-up, infoset count unchanged
→ pure recompute) and made H2H slightly *worse* on both setups. So TV is a small net
positive on both speed and quality. Its theoretical soundness question (the
non-reach-weighted regret update double-counts transposed infosets) is genuinely
ambiguous but moot: action abstraction kills the transpositions, so TV approaches a
no-op. **Kept in V2.1.**

### 2.4 Abstraction-interning speedup → byte-identical, shipped

Profiling showed `get_hand_abstraction`, recomputed every iteration, was **68–81% of
training time** on round-6+ setups — a ~32× cost cliff exactly at sum 6→7 where the
hand-abstraction representation changes, which is why training time clusters by round
rather than by node count. (Caveat: the 68–81% figure is *pre-rewrite*; post-rewrite
and at 8-wide contention the JIT traverse dominates at ~70–86% — a solo profile would
settle the exact split.) The function was rewritten in pure Python (no numpy on tiny
arrays, no `np.delete` in loops), **byte-identical** over all hands of sizes 2–5 plus
sampled 6–11, **5.7× faster** (362 → 64 µs/call), giving ~2.3× faster round-6+
retrains for an *identical* model (commit 6b21a8a). A result cache was rejected:
big-hand recurrence collapses (an 11-card hand recurs ~2× over 5M iters), so a memo
would cost GBs for little gain exactly where it is needed. **Adopted into V2.1.**

### 2.5 The V2.1 consolidation retrain (result)

All 66 setups retrained at the banked settings (penalty 0, prune-10, fast
abstraction, TV on, 5M), `VERSION="V2.1"`. **66/66, 0 failures**; CPU-h ≈ 63.7,
GB-h ≈ 153.3. Versus V2: explored-infoset median **1.00×**, settled-RAM median
**0.99×**, total train wall **0.60×** (~40% faster, from the looser pruning).
Whole-game H2H vs V2 was a tie (0.5033 ± 0.0050, < 0.7σ) — consolidation preserved
playing quality — yet the per-setup sign test was overwhelming (penalized 54 setups:
47+/7−, p ≈ 1e-8; unpenalized 12-setup control: 6/6, mean ≈ 0), confirming the gain
tracks penalty removal specifically and that a 54-way sign test has far more power
than one aggregate win-rate. The benefit *grew* with setup size (the 16 largest
setups improved uniformly), inverting the prior guess that large setups would suffer.

Two notable corrections from this retrain are recorded as process lessons (§10): the
fabricated "penalty=0 triples infosets" claim (measured ratio 1.003), and the V2.1
OOM, diagnosed as a transient save-prep array-copy spike near the 30.6 GiB ceiling
(not a regression) and fixed by reordering diagnostic arrays in place plus dropping
references after save (A/B: +4203 → +1089 MB at 3M rows).

---

## 3. Action-abstraction experiments (post-V2.1)

The hand+action abstraction is the project's hand-crafted "elephant", and §2.4
identified it as the dominant cost. Two lines were pursued. The first — *lossy
grouping* of the action list — was tried in three generations and **all shelved**.
The second — *augmenting* the action set with hand-resolved macros — **succeeded and
became V3**. All of this work originates from the V2.1 anchor and was benchmarked
against it.

### 3.1 The lossy-grouping line: Gen-B, Gen-G, Gen-H (all shelved)

All three modify only the per-infoset action *list*; the composite key / infoset
delineation is untouched, so the infoset count stays ~0.9–0.98× baseline.

- **Gen-B — curated probability menu.** A fixed-semantics slot layout (~32 slots:
  value by `p−g`, escalation build-ons, a-priori checkpoints, min-raises, check),
  each slot resolving to a different concrete bet per hand. Near-lossless and
  resource-positive in the mid-T band, but it **lost 9_11 H2H at 44.6%** (hit only
  91.6%): a *selection* menu never bets some claims, so in self-play their responses
  never train and default to check, which full-action V2.1 exploits. **Shelved —
  structural coverage gap.**

- **Gen-G — descriptor-salient grouping.** A claim is its own action iff its
  rank/suit is *referenced* by the hand or history abstraction, plus fixed anchors;
  unreferenced members of each set type collapse into BLUFF_LOW / BLUFF_HIGH / RANDOM
  sentinels resolved hand-aware at the node. The random sentinel restores full
  support → **gap-free (hit ~99.8%)**, fixing Gen-B's high-T collapse (9_11 ~50.6%).
  But compute was **2.3–4.7× wall / 1.2–1.9× RAM**: grouping cut nodes to 0.55–0.73×,
  yet per-node cost rose to 2.2–2.8× because a slot is a looked-up `(kind, arg)` →
  three per-row array reads → 2–3× cache lines on random row access. A controlled
  A/B/C profile **refuted** the natural hypothesis that membership computation was the
  cost (a member-pool optimization moved wall only ~2–4%): the cost is memory traffic
  from breaking the baseline's contiguous action layout, which is architectural.
  **Shelved — gap-free but memory-bound, no resource win.**

- **Gen-H — deterministic-hash subset.** Every action concrete and bet-indexed (no
  menu indirection): per-row legal bitmask = referenced ∪ anchors ∪ a deterministic
  ⌈k/2⌉ of each type's unreferenced members (frozen hash of the key). Run *matched*
  through the production trainer — the only clean apples-to-apples comparison in this
  line. It **ties V2.1** (49.3–50.4%, ~49.7% mean) and does *not* collapse at high-T
  (9_11 49.7%), a surprise: a uniformly-random subset is more robust than Gen-B's
  curated one, because random drops low-EV actions while curation drops critical ones.
  Resource-neutral (wall ~equal, RAM +2.5%, nodes 0.78×). **Shelved — ties +
  resource-neutral + a real coverage gap + much more code = not worth it.**

**Cross-cutting conclusions.** (1) Action abstraction *cannot* shrink the infoset
table (it is hand/history-bound) → no RAM win; it changes only branching/node count.
(2) The traverse is memory-bound: a per-row menu layout adds cache traffic that
swamps node savings (Gen-G); staying bet-indexed (Gen-H) stays neutral. (3) Random >
curated for coverage robustness. The capability lever is therefore the *hand+history*
abstraction, not the action list — **do not re-propose grouping or array-shortening.**

The Gen-G/H scaffolding (`structure.py`, `action_menu.py`, `referenced.py`) has been
deleted; its algorithm is preserved here so it can be re-derived. `structure`'s
`claim_ranks_suits(b)` mapped each of the 88 claims to the {ranks},{suits} it involves
(high/pair/trips/quads = one rank; two-pair/full-house = two ranks; flush = one suit;
straight = its rank run; straight-flush = suit + rank run), and its `ESCALATION[s]`
graph listed the higher claims sharing a rank or suit with `s` — the hand-independent
"build-ons"/natural raises that top-p value menus (Gen-B) systematically missed.
`referenced` decided which claims stay individual: the **hand** side mirrored
`get_hand_abstraction` feature-by-feature (a function of `abs_id`, so CFR-consistent),
and the **history** side was `claim_ranks_suits(last_bet)` ∪ the *intersection over
members* of each earlier bet's history code — a precise code pins a rank/suit, but an
imprecise 'Z'-type code whose members share nothing references nothing. A claim stayed
individual iff both its rank- and suit-masks were subsets of the referenced masks.
`action_menu`'s sentinels resolved via a probability-free `support_vector(hand)` (held
cards matching the claim, full house weighted trips-first as `10·min(trips,3) +
min(pair,2)`): BLUFF_LOW = argmin support, BLUFF_HIGH = argmax, RANDOM = uniform.

### 3.2 The augmenting line: action macros (Gen-I) → V3

The surviving idea, and the V3 feature. A **macro** is an extra action *added* to the
full action set (coverage untouched, not in the infoset key), which at the node
resolves to a concrete legal bet `b*` by the argmax/argmin of a per-hand score over
legal claims. Because the lossy hand abstraction buckets many hands into one `abs_id`,
the shared per-bucket strategy cannot pick *this hand's* most-credible or bluffiest
claim; a macro injects that discarded per-hand information back. It is a transposition
in the tree but a genuinely new policy capability, so unlike the grouping line it can
**beat** V2.1 rather than only tie it. No new nodes are created: `b*` is an existing
legal bet, force-traversed once and value-copied (exact under the TV cache).

Three macros were defined over the existence probabilities `p` = P(claim exists | my
cards) (hypergeometric a-posteriori) and `g` = P(claim exists | card total only)
(a-priori), argmax'd over legal claims with random tie-break:

- **value** = argmax `p` — the most-probable legal set.
- **difftruthy** = argmax `(p − g)` — the set this hand most *over*-supports relative
  to baseline (an axis orthogonal to raw `p`).
- **bluff** = argmin `(p − g)` = argmax `(g − p)` — the bluffiest legal set.

Each was trained matched to V2.1 (penalty 0, pruning −10/−12, min_bet by total, 5M)
and evaluated in self-play vs V2.1 by paired, symmetrised 10k-deal MC (per-setup SE
~0.010) over six setups (4_5, 5_7, 6_6, 3_9, 5_11, 9_11).

| config | macro vs V2.1 (clean retrain) | verdict |
|---|--:|---|
| bluff alone | ~0 | inert alone |
| value alone | +0.0169 | helps on deep setups |
| **difftruthy alone** | **+0.0217 (6/6)** | best single |
| difftruthy + bluff | +0.0190 | below difftruthy-alone (bluff hurts) |
| value + difftruthy | +0.0193 | below all-three |
| **all three** | **+0.0256 (6/6, ~6σ agg)** | **adopted → V3** |

The macros interact **non-additively**: adding value *or* bluff singly to difftruthy
slightly *hurts* (both 2-combos fall below difftruthy-alone), yet adding *both* gives
the best point estimate. The all-three margin (+0.0256) over difftruthy-alone
(+0.0217) is within ~1 aggregate SE, so the choice of all-three over the cheaper
single macro was a deliberate cost/quality call (3 macros ≈ 1.4× wall, see §5.4), not
a statistically forced one.

**Serving is a top-up, not a gate.** Of the +0.0256, only +0.0124 comes from
*resolving* the macro per-hand at play; the remaining ~+0.013 comes from
training-with-macros improving the underlying *concrete* strategy and survives even if
the deployed agent never resolves a macro (it would just serve the macro-trained
concrete distribution). So shipping the macro-trained model captures most of the gain
regardless; the per-hand fold is the worthwhile remainder.

**Gate to total ≥ 7.** A small-setup study (2_2, 3_3, 2_4 near-lossless; 4_4 the lossy
boundary) trained no-macro vs all-three. H2H was within noise everywhere, but the
decisive signal — invisible to a single-opponent H2H — was LBR-2 on the concrete
strategy: macros **raised** exploitability 2.3–4.2× on the near-lossless setups (2_2
2.12% → 6.93%, 3_3 1.91% → 7.95%, 2_4 ~4.3% → ~10%). In the near-lossless regime there
is no abstraction loss to recover, so a macro only adds exploitable distortion plus
~1.65–1.96× compute. **Macros are therefore enabled only for total ≥ 7.**

---

## 4. Per-round training tweaks

Two further results from the V2.1 lineage. The first was adopted into V3; the second
informs the planned final retrain.

### 4.1 min_bet = 27 for total 13–15 → adopted into V3

In card-rich setups the sub-straight claims (high card / pair / two-pair) are almost
always true, so they are strategically near-trivial *and* exactly where same-iteration
node-revisiting hurts convergence. Raising min_bet to 27 prunes that subtree. Trained
at V2.1 params and H2H'd against V2.1 over the 14 sum-13–15 setups, min_bet=27 **beat
V2.1 in 13 of 14** (mean +0.64%, sign-test p ≈ 0.0018; edge growing with size, sum13
+0.30% → sum15 +1.16%; only the lone symmetric 7,7 regressed). It was simultaneously
**~2.8× faster** to train, used **~50% less RAM**, and was flat against greedy Perun
(mean diff +0.0006). So it is not merely cheaper — pruning the trivial low claims
*sharpens* the meaningful play. **Adopted into V3** (V3 uses min_bet 27 for total ≥ 13;
≥ 16 was already 27 under V2.1's 0/4/27 schedule).

### 4.2 Early-game iteration scaling (10/20/40M, rounds 1–5) → forward plan, not V3

The rounds-1–5 setups (sums 2–6) are the one substrate where abstraction is
near-lossless and LBR-2 is affordable, so iteration count can be probed cleanly. At
V2.1 params, the 9 small setups were trained at 10/20/40M and LBR-2'd vs the 5M
baseline. **5M badly under-trains the early game** — every non-trivial setup roughly
halves its LBR-2 from 5M→40M (2,4 sp0 5.12% → 2.77%; 1,5 sp1 5.82% → 3.25%). 20M
clearly beats 10M everywhere, and **even 40M has not converged** on the bigger
early-game setups (2,3 / 1,5 / 2,4 / 3,3 still drop 0.25–0.70 pp from 20M→40M ≫ noise),
so they want > 40M. The win is best-responder-facing: vs greedy Perun the same change
is flat (+0.23 pp), but a hybrid V2.1 (rounds-1–5 → 40M, sum-13–15 → mb27) beat V2.1
whole-game 0.5100 ± 0.0050. **Not in V3** (V3 is 5M everywhere); this is the basis for
the planned final retrain's early-round iteration budget (§8).

---

## 5. V3 — the macro anchor

### 5.1 Design

V3 = the V2.1 hyperparameters (penalty 0, pruning −10/−12, fp32, 5M iters) plus, per
setup: **min_bet = 27 for total ≥ 13** (else 0, §4.1), and the **all-three macros for
total ≥ 7** (else the plain trainer, §3.2 + the ≥7 gate). 66 setups, trained 8-wide,
biggest-total first.

### 5.2 Implementation (and why it is more than "three numbers")

The deployed artifact is exactly what the user's intuition expects: the macros are
**three extra columns on `strategy.npz`** (`masses[n,3]` + `kinds`), resolved
dynamically at serve. There is no separate file and no new node type. The supporting
code is the *machinery* that computes, trains, and validated those three columns:

- `probs.py` / `probs_jit.py` — the existence-probability kernels (`p_vector`,
  `g_vector`, the numba `p_vector_fast`). The macros are *defined* by `p` and `g`, so
  these are used at **both** training (to find each macro's per-hand `b*`) and serving
  (the agent folds the macro's mass onto `b* = argmax` of `p` / `p−g` / `g−p`). They
  are the only abstraction files the Lambda imports, and the reason the macro serve
  path pulls in numba (lazily, only for macro setups; concrete setups stay numba-free).
- `trainer_macros.py` — the `Trainer` subclass that adds the N macro columns and
  force-traverses each to its `b*`. Required to produce a macro model at all.
- `run_macros_training.py` — a ~20-line adapter that swaps the macro trainer into the
  production `training.py` and writes the single unified `strategy.npz`. It exists so
  the core trainer stays macro-agnostic (it trains both concrete and macro setups).
- `selftest_macros.py` + `selftest_bluff.py` — the H2H *experiment* harness that
  produced the §3.2 table and chose all-three for V3 (`selftest_bluff` supplies the
  standard-model loader and is the single-macro precursor). These are decision tooling,
  not deployment code, and not needed at serve.

So the runtime cost of the feature is small (three columns + a ~25-line fold in
`agent.py`); the file count reflects the *math* (p/g kernels), the *training path*
(trainer + adapter), and the *validation* (selftest harness) behind those columns.

### 5.3 Results

Validated on the box, 10k whole games per matchup (win-rate from the new model's
view, ±0.005):

| matchup | win-rate | verdict |
|---|--:|---|
| **V3 vs V2.1** | **0.5348** | V3 wins ~53.5% |
| **V3 vs greedy Perun** | **0.5381** | V3 **wins** |
| V2.1 vs greedy Perun (matched re-run) | 0.4889 | V2.1 **loses** |

The headline is the **+4.9 pt swing against greedy Perun** — V2.1 lost to it (0.489),
V3 beats it (0.538) — while also beating the V2.1 anchor directly. Per-setup vs V2.1
(`selftest_macros`, 10k deals over the 57 macro setups) was **+0.0200 mean, 52/57
positive**: flat on the near-lossless small setups (as the ≥7 gate predicts) and
+0.02 to +0.049 on the lossy mid/large setups. The whole-game margin is larger than
the per-round edge (round-level ~0.507) because a consistent small per-round advantage
compounds over the ~19 rounds of a full game.

### 5.4 Cost / KPIs

Clean across all 66 setups: min_bet rule 100% correct, all 5M iters, 0 errors, no OOM,
peak RAM 3.96 GB. Total train wall **1.24× V2.1** — the macros cost ~1.6–1.8× on the
min_bet-0 setups (a real memory-bandwidth cost of the extra fp32 columns on the
memory-bound traverse: node-work is only ~1.09× and RAM ~1.06×, but throughput dropped
from ~868 it/s 8-wide to ~548 it/s 6-wide), *offset* by min_bet-27 being 0.43–0.77× of
V2.1 on the total-13–15 setups (V2.1 used min_bet 0/4 there). Node-revisit magnitude
from the diagnostic touch counters is ~600–900× per infoset.

---

## 6. Evaluation instruments

- **H2H** (`analysis/head_to_head.py`, `--monte-carlo`): 10k MC deals, variant payoff
  in [−1, +1], SE ≈ 0.01. The primary per-setup ranking signal. Reads `strategy.npz`
  directly, so it is identical across code versions.
- **Whole-game** (`analysis/cfr_vs_cfr_games.py`, `cfr_vs_nfsp_games.py`): plays full
  real-mechanic games (1v1 start, round loser gains a card and opens next, eliminate
  at 11 cards), logging one compact row per round to a `.npy`; aggregated to a
  whole-game win-rate. This is the primary *version-vs-version* and *vs-Perun* signal.
- **LBR-K** (`lbr.py`): local best response, a lower bound on exploitability that rises
  with depth. The inside view, but affordability-bound — LBR-1 finishes in < 1 h only
  on small/asymmetric setups; LBR-2 is feasible on sums ≤ 6; LBR-3 is infeasible
  (depth-3 on 1,8 ≈ days/call). Asymmetric setups (1,8 / 2,7) collapse the depth-1
  enumeration and are the reusable cheap probe. Cost = (LBR hands) × (beliefs), linear
  in both; per-(hand,belief) cost varies ~5×, so the cheapest *exact* setup is 3,3, not
  the fewest-deals 1,5 — a clean combinatorial argument that inverted once measured.
- **NFSP / Perun yardstick.** Perun is the deployed NFSP agent. Its **greedy** mode
  (argmax of the average policy) is a *pure*, non-Nash-approximating strategy but the
  only *competitive* yardstick (~0.50 vs near-Nash CFR, matching the friend's even
  1004–996 result); never train against it. Its **sampled** mode is fuller but its
  off-policy tails are exactly what near-Nash CFR punishes, inflating CFR's win-rate
  ~9 pp (~0.76) — a secondary view, not the realistic bar. A near-Nash strategy
  guarantees ≥ the game value against *any* opponent and beats genuinely bad ones, but
  does not *maximally* exploit a competent-but-imperfect opponent (that needs
  best-response), which is why CFR only draws greedy Perun rather than beating it — and
  why V3's macros, by adding a capability, can push past 0.50.
- **Games diagnostic** (`analysis/MORANA_GAMES_DIAGNOSTIC.md`). V1 beat humans 54.4%
  but lost to greedy Perun 48.4% (V2.1 48.6% — V1 ≈ V2.1 vs greedy, so the *abstraction*
  was the binding constraint, motivating V3 and beyond). Morana over-challenges and
  over-bluffs versus strong opponents and bets the least-hand-supported claims. The
  sharpest, triple-evidenced hole is **flush response**: her raises over a flush are
  true only 48.7% (callable) vs Perun's 54.7%, and she defaults to checking flushes
  (60.6%) — the hand abstraction lacks suit/flush weakness information. Challenge
  accuracy also degrades with card count as the abstraction coarsens. This directly
  motivates the next capability lever (§8).

---

## 7. Closed chapters — do not revisit

- **DCFR / CFR+ as external-sampling variants**: an `--algorithm {es,cfr_plus,dcfr}`
  switch added to the then pure-Python trainer (commit `123a601`). This was **not**
  canonical DCFR/CFR+: the opponent action and the deal were still *sampled*, so it was
  external-sampling MCCFR with a CFR+/DCFR discount (clip regrets at 0; α=1.5, β=0, γ=1)
  bolted on — the commit's own README concedes "MCCFR external sampling rather than full
  traversal" — and it ran with **pruning disabled**. On rounds 1–3 at 5M it was
  statistically indistinguishable from ES (the *abstraction* is the exploitability floor
  there; both saturate it), and the "10–18× slower wall" headline is **confounded**: the
  README attributes the slowdown to the disabled pruning plus the α-discount lookup, not
  to the update rule. So this closes the *sampled* hybrid only; it does **not** test or
  rule out a *proper full-traversal* CFR+/DCFR, which is the §8 forward-plan yardstick.
  The code was dropped in the numba rewrite and survives only at `123a601`.
- **Outcome Sampling**: ~400–500× more exploitable than external sampling on Blef's
  88-action shallow trees (variance blows up). Closed; code kept only as a VR-MCCFR
  starting point.
- **C++ port**: closed on a cost/benefit judgement, *not* on a benchmark. The hot
  path is already the numba-JIT'd traverse kernel (§2.4), so a port could only buy
  whatever constant factor separates numba from hand-written C on these loops. That
  gap is **widely assumed small but was never measured for this trainer** — any
  specific ratio (an earlier draft of this doc guessed "1.3–2×"; that figure was never
  tested) is speculation. Weeks of porting work is unjustified for, at best, a constant
  factor, absent a ~100× scale-up need, which does not exist.
- **Action-list grouping / array-shortening** (Gen-B/G/H, §3.1): shelved; the infoset
  table is hand/history-bound and the traverse is memory-bound, so it yields no
  resource win.
- **Online subgame solver**: shelved 2026-05-28 — Phase-2 LBR-1 timings (sum-10 setups
  5–6 h+) ruled it out at any non-trivial setup. Code in `cfr_ai/archive/subgame/`;
  its README has the revival path. The strategy-load fast path it motivated shipped
  anyway for its own benefits (the small Lambda tier).

---

## 8. Forward plan

1. **Hand-abstraction enhancement** — the #1 capability lever, steered by the games
   diagnostic: a compact, rank-agnostic *weakness* / bluff-quality signal at the high
   bet bands, especially the missing suit/flush information that makes Morana's flush
   responses incredible. Validate by retrain → H2H-vs-V3 → LBR on the cheap probes.
2. **CFR+** (regret-matching⁺ + linear averaging, full-traversal, made feasible by
   pruning's collapse of effective branching) — primarily a variance-free
   exploitability *yardstick* on the small/shallow setups, where the abstraction is
   near-lossless, to measure true abstraction loss. The closed "DCFR" chapter (§7) does
   **not** count against this: that was external-sampling with pruning *off*, so its
   10–18× cost was the no-pruning penalty, not the update rule — a genuine full-traversal
   CFR+ is an untested regime. A win retrains **V4** (H2H vs V3 + greedy Perun).
3. **Final retrain** — ~40M iters for the early rounds (the clean convergence
   substrate; §4.2 shows they want ≫ 5M) / ~10M for the later rounds → deploy.

The human-vs-Morana games have already been pulled from DynamoDB and analysed; that
work produced the games diagnostic (`analysis/MORANA_GAMES_DIAGNOSTIC.md`, §6), which
is what now steers item 1.

Longer-term: 3+ players (no longer 2-player zero-sum, so CFR loses its equilibrium
guarantee — research-grade) and non-standard rules, only after the 1v1 AI is strong.

---

## 9. Deployment (Lambda)

The agent loads via `strategy_io.load_strategy_for_agent` (a `FlatStrategyAgent` with
`np.searchsorted` lookup) and `stage_for_docker.py` converts each `strategy.npz` into
a sparse-mmap layout (uint16-quantised values, ~10× smaller than dense int16) at build
time, so resident memory stays ~tens of MB regardless of strategy size. V2.1 ran on
the **256 MB** tier (cold ~2.5–3.5 s), with no numba on the agent path.

For V3 the macro columns ride **inside** `strategy.npz` (and the mmap `strategy_meta`),
so the macro masses are carried through the same compact path; the agent folds them at
serve, lazily importing the numba `p_vector_fast` only for macro setups (concrete
setups remain numba-free). The Lambda memory was raised to **512 MB**. Deploying V3 is
the immediate next operational step (stage → ECR → update function → verify a few
setups including a big one → retire the dangling images).

---

## 10. Process & operational lessons

These are the discipline rules earned during the experiment phase.

- **a. Fabrication is the cardinal sin.** The "penalty=0 triples infosets" claim
  (measured ratio 1.003) and writing A/B numbers into a cron before the test ran were
  both wrong. Label measured-vs-hypothesis; never state a number you have not computed.
- **b. Check turns are read-only.** An erroneous `systemctl stop` batched into a
  "maybe it's done" check killed 8 in-flight trainings (the redo that inflated V2.1's
  CPU-h). Only `done == N` triggers a stop.
- **c. Investigate before declaring infeasible.** A "5 hour" LBR was ~1 h on the right
  machine — extrapolated from a slow laptop instead of checking the box's recorded
  durations.
- **d. Even a clean combinatorial cost argument can invert once measured** (the LBR-2
  "1,5 is cheapest" deal-count argument, refuted by the ~5× per-(hand,belief) variance).
- **e. Periodic in-training LBR costs hours** — overturning the "negligible"
  assumption; a single LBR-1 on a sum ≥ 10 setup costs hours, so it is OFF by default.
- **f. IPv6 SSH to the box is flaky** — exactly one ssh per message, `cd` first,
  explicit `.venv/bin/python`, strip `\r`, collapse checks to one echo line.
- **g. Be precise when killing processes** — a substring-matching kill also matched
  bash wrappers; match exact pids/cmdlines.
- **h. Long box runs use `systemd-run` transient units** (survive SSH teardown;
  `-p MemoryMax=` as an OOM backstop), not tmux.
</content>
</invoke>
