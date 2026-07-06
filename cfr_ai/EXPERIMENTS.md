# Blef CFR — Experiment Record

The authoritative record of every experiment run on the Blef CFR trainer: the design, the measured results, the conclusions, and — where they were instructive — the surprises versus what theory predicted. The `README.md` carries only the final adopted settings; this document carries the full reasoning. Numbers are **measured** unless explicitly labelled an estimate or hypothesis.

Blef is an imperfect-information bluffing game (1v1 throughout §§1–5.10; the 3-player extension is §5.11). Every bet is a claim about all cards on the table combined; a player escalates by recombining the standing claim with their own cards (truthfully or as a bluff), and the pivotal action is the *check* — "does the claimed set exist?". Unlike poker there is no opponent-equity estimation, which makes the existence probability of each claim the natural object the abstraction and the action set are built around. The trainer is external-sampling MCCFR, JIT-compiled with numba over flat arrays.

**Named agents used throughout:** **Morana** = the deployed CFR agent (this project's product; the version under test unless stated). **Perun** = the deployed NFSP agent (NFSP = Neural Fictitious Self-Play, a deep-RL self-play baseline trained separately; "greedy" = argmax of its average policy, "sampled" = drawing from it — see §6). **conservative_bayesian** ("Dazhbog") = the strongest hand-crafted heuristic agent (Bayesian claim-probability model). Perun and conservative_bayesian are *external* yardsticks: they share no code or abstraction with the CFR line.

**Box:** Hetzner CCX33 (8 vCPU, ~30.6 GiB RAM), `/root/blef_ai/.venv/bin/python`. **Cross-section setups** used for most sweeps (8, to saturate the 8 vCPUs): (2,2), (3,3), (4,5), (5,7), (6,6), (3,9), (5,11), (9,11) — spanning tiny-symmetric (afford deep LBR) to the (9,11) training pole (~1.6 h at 5M iters).

---

## 1. Version history

Each version is the previous one plus a specific, validated change. The line runs V1 → V2 → V2.1 → V3 → V3.1 → V3.2; every experiment below is tagged with the version it fed.

| Version | What changed | Headline result | Location |
|---|---|---|---|
| **V1** (baseline) | Original production CFR: pure external-sampling MCCFR, per-setup penalty, pruning −20/−22, 5M iters, all 66 setups. The first deployed model. | The reference point. | `cfr_ai/archive/v1` (npz) |
| **V2** | Full 66-setup retrain on the new numba JIT trainer (replacing the historical pure-Python trainer). Same hyperparameters as V1 — an **efficiency rewrite**, not a strategy change. | **~6–10× faster** training (measured: (1,2) 5M iters 63 → 6.4 min; see README "Performance", where the JIT kernels are validated bit-equivalent to the pure-Python reference within fp32 noise). Equivalence of the *retrained strategies* was checked at the time via game values, structural metrics, and an (11,11) H2H, all read as within-noise — but those comparison numbers were **not preserved**; the kernel-level bit-equivalence is the surviving hard evidence. | `cfr_ai/archive/v2` |
| **V2.1** | Hyperparameter *consolidation*: penalty **0**, pruning **−10/−12**, the byte-identical fast `get_hand_abstraction`. No structural change. | ≈ V2 strength, **~40% faster** to train and slightly *less* exploitable (penalty removal moved it closer to Nash). | `cfr_ai/outputs` (until V3) |
| **V3** | V2.1 + two independent additions: **augmenting action macros** (3 per infoset, for total ≥ 7) and **min_bet = 27 for total ≥ 13**. | Beats V2.1 head-to-head (**0.535** whole-game) and **flips greedy Perun from a loss to a win** (0.489 → **0.538**). Anchor for the V3.1 round. | `cfr_ai/outputs` |
| **V3.1** | V3 + total-7 → **value-only @ 10M** (concrete) + **suits@root for total ≥ 17**; ranks@root dropped, flush-refinement deferred. | Validated no-regression refinement of V3: whole-game vs **greedy Perun 0.550** (V3 0.538), vs **V3 0.507** (no-regression), vs **V1 0.534** (V3 0.525) — **+0.7–1.2 pt over V3 across all three independent yardsticks**. Gains localized to total-7 (value-only) and total-22 / 11,11 (suits@root). **Anchor for the V3.2 round** (since superseded). | `cfr_ai/experiments/v31` (box), with its own info_set snapshot (pre the V3.2 suit-flag drop) |
| **V3.2** (anchor) | V3.1 + the §5.7/§5.8 ingredients: **11-v-11 opening floor `min_bet = 65`** (last full house) and **dropping the round-6 suit-flag** on the total-7 setups (`1_6`/`2_5`/`3_4` → the plain value multiset; the decompressed-flush-band probe was shelved). V3.2 info_set snapshot (md5 `9595e646`). | Net-neutral-to-positive vs V3.1 on the aggregate while fixing the 11-v-11 endgame. Its iteration-scaled instances keep paying: **V3.2x2** (2×) reached 0.564 vs greedy Perun / 0.541 vs V1; **V3.2x4** (4×: macros 20M, value-only 40M) is the strongest model on every axis — vs greedy Perun **0.572**, vs V3.2x2 **0.518**, vs V3.1 **0.531**, vs V1 **0.554** — and is **deployed** (2026-06-11, uint8-quantised serving payload; §5.10). **Current anchor.** | box `experiments/v32` (config) / `v32x2` / `v32x4` (deployed, = local `cfr_ai/outputs`); live on Lambda |

V1 was historically tagged `v0`/`v0_baseline`; the archive is now consolidated to a single `v1` snapshot. V2's hyperparameters (pruning −20/−22, per-setup penalty 0.05/0.1) are the signature distinguishing it from V2.1 — the box working tree `cfr_ai/outputs` is V2, while `cfr_ai/v2.1` is the V2.1 anchor.

---

## 2. Hyperparameter consolidation (the V2 → V2.1 sweeps)

Four experiments, all run against the V2 baseline, established the V2.1 settings. Method throughout: **H2H** = 10k Monte-Carlo deals, variant-minus-baseline payoff in [−1, +1] (per-setup SE ≈ 0.01, so a single setup is signal only at |val| ≥ ~0.02); ranked in aggregate by mean, t-statistic, and a sign test. **LBR-K** (local best response) is a lower bound on true exploitability that tightens — rises — with depth.

### 2.1 Pruning range (Sweep 1) → adopt −10/−12

Three pruning variants vs V2 (−20/−22) over the 8 cross-section setups.

| variant (threshold, min_regret) | training speedup (min_bet-0 setups) | mean H2H | sign test | verdict |
|---|--:|--:|---|---|
| `prune-10` (−10, −12) | **+8.6%** | −0.0015 | 8/5, p=0.58 | **adopt — free speedup** |
| `prune-5` (−5, −7) | +13.2% | −0.0054 | 11/2, p=0.022 | small real quality cost |
| `prune-2` (−2, −4) | +19.2% | −0.0101 | 12/1, p=0.003 | clear quality cost |

(Sign tests are over n = 13 seat-results from the 8 cross-section setups: the 3 symmetric setups contribute one result each, the 5 asymmetric ones one per seat = 3 + 10.) Looser pruning speeds only full-action-space (min_bet-0) setups; min_bet-27 setups have little to prune (~0–5%, noise). prune-10 buys ~8–9% cheaper training at statistically zero quality cost (LBR-1 on 4,5 corroborates: 4.45% → 4.68%, within noise). prune-5 and prune-2 trade increasing speed for increasing exploitability and were rejected. **Adopted into V2.1.**

### 2.2 Iteration count (Sweep 2) → 10M is strictly better, but banked

10M vs the 5M V2 baseline, pruning held at −20/−22. 10M won on every non-trivial setup (13/13 H2H positive, +1 to +2.4 pp, sign-test p ≈ 1e-4) and was less exploitable on all measured seats. Critically, the **exact-(2,2) LBR gap widens with depth** (LBR-1 0.15 pp → LBR-2 0.16 → LBR-3 0.28), the signature of genuinely better convergence rather than a self-play artifact; (3,3) LBR-2 improved 0.82 pp.

The conditional 3M arm was dropped (5M is, if anything, mildly under-trained, so 3M could only be worse). **Production stays 5M:** 10M is a one-time ~1.75–2× training cost for a durable +1–2 pp edge, **banked as the "throw-compute" lever** to pull only after cheaper ideas are exhausted. The saving from 10M concentrates on min_bet-0 setups (5,7: 1.28×, 3,9: 1.31×) where late-stage pruning trims maturing branches, while min_bet-27 setups stay ~2× — the same action-space pattern as Sweep 1.

### 2.3 Penalty and temporary-value (Sweep 3 + probe exp3b) → drop the penalty, keep temp-value

The per-setup **bet penalty** taxes bet-lines vs checks, intended to speed training (shorter sampled sequences) and to relieve the theoretical unsoundness of the `temporary_value` (TV) transposition cache. Removing it (penalty-0) won H2H by ~+0.7 pp (5/6 setups), monotonically to zero (full removal ≥ halving on every 0.1 setup), and — decisively — was **less exploitable** on every seat of the cheap-LBR probes (1,8: −2.37 pp on sp1; 2,7: −2.46/−1.76 pp), never worse. It was also duration-neutral despite touching ~1.5× more nodes, because abstraction interning (§2.4) dominated the clock, not node count. The penalty was distorting the strategy for no speed gain. **Adopted (penalty = 0) into V2.1.**

The companion `--no-temporary-value` isolation kept TV: removing it re-traversed 85–89% of node visits (4,5: 6.2×, 5,11: 8.8× node blow-up, infoset count unchanged → pure recompute) and did not help strength. The original H2H deltas were not preserved; re-measured 2026-07-06 on the archived variant models (`experiments/ penalty-0` vs `penalty-0-notv`, 40k symmetrised deals/setup, SE 0.005): no-TV advantage **+0.002 (4,5 — statistical zero)** and **−0.022 (5,11 — ~4.5σ worse)**. So TV is speed-positive everywhere and strength-positive on the deeper setup. Its theoretical soundness question (the non-reach-weighted regret update double-counts transposed infosets) is genuinely ambiguous but moot: action abstraction kills the transpositions, so TV approaches a no-op. **Kept in V2.1.**

### 2.4 Abstraction-interning speedup → byte-identical, shipped

Profiling showed `get_hand_abstraction`, recomputed every iteration, was **68–81% of training time** on round-6+ setups — a ~32× cost cliff exactly at sum 6→7 where the hand-abstraction representation changes, which is why training time clusters by round rather than by node count. (Caveat: the 68–81% figure is *pre-rewrite*; post-rewrite and at 8-wide contention the JIT traverse dominates at ~70–86% — a solo profile would settle the exact split.) The function was rewritten in pure Python (no numpy on tiny arrays, no `np.delete` in loops), **byte-identical** over all hands of sizes 2–5 plus sampled 6–11, **5.7× faster** (362 → 64 µs/call), giving ~2.3× faster round-6+ retrains for an *identical* model (commit 6b21a8a). A result cache was rejected: big-hand recurrence collapses (an 11-card hand recurs ~2× over 5M iters), so a memo would cost GBs for little gain exactly where it is needed. **Adopted into V2.1.**

### 2.5 The V2.1 consolidation retrain (result)

All 66 setups retrained at the banked settings (penalty 0, prune-10, fast abstraction, TV on, 5M), `VERSION="V2.1"`. **66/66, 0 failures**; CPU-h ≈ 63.7, GB-h ≈ 153.3. Versus V2: explored-infoset median **1.00×**, settled-RAM median **0.99×**, total train wall **0.60×** (~40% faster, from the looser pruning). Whole-game H2H vs V2 was a tie (0.5033 ± 0.0050, < 0.7σ) — consolidation preserved playing quality — yet the per-setup sign test was overwhelming (penalized 54 setups: 47+/7−, p ≈ 1e-8; unpenalized 12-setup control: 6/6, mean ≈ 0), confirming the gain tracks penalty removal specifically and that a 54-way sign test has far more power than one aggregate win-rate. The benefit *grew* with setup size (the 16 largest setups improved uniformly), inverting the prior guess that large setups would suffer.

Two notable corrections from this retrain are recorded as process lessons (§10): the fabricated "penalty=0 triples infosets" claim (measured ratio 1.003), and the V2.1 OOM, diagnosed as a transient save-prep array-copy spike near the 30.6 GiB ceiling (not a regression) and fixed by reordering diagnostic arrays in place plus dropping references after save (A/B: +4203 → +1089 MB at 3M rows).

---

## 3. Action-abstraction experiments (post-V2.1)

The hand+action abstraction is the project's hand-crafted "elephant", and §2.4 identified it as the dominant cost. Two lines were pursued. The first — *lossy grouping* of the action list — was tried in three generations and **all shelved**. The second — *augmenting* the action set with hand-resolved macros — **succeeded and became V3**. All of this work originates from the V2.1 anchor and was benchmarked against it.

### 3.1 The lossy-grouping line: Gen-B, Gen-G, Gen-H (all shelved)

All three modify only the per-infoset action *list*; the composite key / infoset delineation is untouched, so the infoset count stays ~0.9–0.98× baseline.

- **Gen-B — curated probability menu.** A fixed-semantics slot layout (~32 slots: value by `p−g`, escalation build-ons, a-priori checkpoints, min-raises, check), each slot resolving to a different concrete bet per hand. Near-lossless and resource-positive in the mid-T band, but it **lost 9_11 H2H at 44.6%** (lookup hit-rate only 91.6% — "hit" throughout §3.1 = the share of in-play infoset lookups that find a trained policy row, the rest falling back to the check-100% default): a *selection* menu never bets some claims, so in self-play their responses never train and default to check, which full-action V2.1 exploits. **Shelved — structural coverage gap.**

- **Gen-G — descriptor-salient grouping.** A claim is its own action iff its rank/suit is *referenced* by the hand or history abstraction, plus fixed anchors; unreferenced members of each set type collapse into BLUFF_LOW / BLUFF_HIGH / RANDOM sentinels resolved hand-aware at the node. The random sentinel restores full support → **gap-free (hit ~99.8%)**, fixing Gen-B's high-T collapse (9_11 ~50.6%). But compute was **2.3–4.7× wall / 1.2–1.9× RAM**: grouping cut nodes to 0.55–0.73×, yet per-node cost rose to 2.2–2.8× because a slot is a looked-up `(kind, arg)` → three per-row array reads → 2–3× cache lines on random row access. A controlled A/B/C profile **refuted** the natural hypothesis that membership computation was the cost (a member-pool optimization moved wall only ~2–4%): the cost is memory traffic from breaking the baseline's contiguous action layout, which is architectural. **Shelved — gap-free but memory-bound, no resource win.**

- **Gen-H — deterministic-hash subset.** Every action concrete and bet-indexed (no menu indirection): per-row legal bitmask = referenced ∪ anchors ∪ a deterministic ⌈k/2⌉ of each type's unreferenced members (frozen hash of the key). Run *matched* through the production trainer — the only clean apples-to-apples comparison in this line. It **ties V2.1** (49.3–50.4%, ~49.7% mean) and does *not* collapse at high-T (9_11 49.7%), a surprise: a uniformly-random subset is more robust than Gen-B's curated one, because random drops low-EV actions while curation drops critical ones. Resource-neutral (wall ~equal, RAM +2.5%, nodes 0.78×). **Shelved — ties + resource-neutral + a real coverage gap + much more code = not worth it.**

**Cross-cutting conclusions.** (1) Action abstraction *cannot* shrink the infoset table (it is hand/history-bound) → no RAM win; it changes only branching/node count. (2) The traverse is memory-bound: a per-row menu layout adds cache traffic that swamps node savings (Gen-G); staying bet-indexed (Gen-H) stays neutral. (3) Random > curated for coverage robustness. The capability lever is therefore the *hand+history* abstraction, not the action list — **do not re-propose grouping or array-shortening.**

The Gen-G/H scaffolding (`structure.py`, `action_menu.py`, `referenced.py`) has been deleted; its algorithm is preserved here so it can be re-derived. `structure`'s `claim_ranks_suits(b)` mapped each of the 88 claims to the {ranks},{suits} it involves (high/pair/trips/quads = one rank; two-pair/full-house = two ranks; flush = one suit; straight = its rank run; straight-flush = suit + rank run), and its `ESCALATION[s]` graph listed the higher claims sharing a rank or suit with `s` — the hand-independent "build-ons"/natural raises that top-p value menus (Gen-B) systematically missed. `referenced` decided which claims stay individual: the **hand** side mirrored `get_hand_abstraction` feature-by-feature (a function of `abs_id`, so CFR-consistent), and the **history** side was `claim_ranks_suits(last_bet)` ∪ the *intersection over members* of each earlier bet's history code — a precise code pins a rank/suit, but an imprecise 'Z'-type code whose members share nothing references nothing. A claim stayed individual iff both its rank- and suit-masks were subsets of the referenced masks. `action_menu`'s sentinels resolved via a probability-free `support_vector(hand)` (held cards matching the claim, full house weighted trips-first as `10·min(trips,3) + min(pair,2)`): BLUFF_LOW = argmin support, BLUFF_HIGH = argmax, RANDOM = uniform.

### 3.2 The augmenting line: action macros (Gen-I) → V3

The surviving idea, and the V3 feature. A **macro** is an extra action *added* to the full action set (coverage untouched, not in the infoset key), which at the node resolves to a concrete legal bet `b*` by the argmax/argmin of a per-hand score over legal claims. Because the lossy hand abstraction buckets many hands into one `abs_id`, the shared per-bucket strategy cannot pick *this hand's* most-credible or bluffiest claim; a macro injects that discarded per-hand information back. It is a transposition in the tree but a genuinely new policy capability, so unlike the grouping line it can **beat** V2.1 rather than only tie it. No new nodes are created: `b*` is an existing legal bet, force-traversed once and value-copied (exact under the TV cache).

Three macros were defined over the existence probabilities `p` = P(claim exists | my cards) (hypergeometric a-posteriori) and `g` = P(claim exists | card total only) (a-priori), argmax'd over legal claims with random tie-break:

- **value** = argmax `p` — the most-probable legal set.
- **difftruthy** = argmax `(p − g)` — the set this hand most *over*-supports relative to baseline (an axis orthogonal to raw `p`).
- **bluff** = argmin `(p − g)` = argmax `(g − p)` — the bluffiest legal set.

Each was trained matched to V2.1 (penalty 0, pruning −10/−12, min_bet by total, 5M) and evaluated in self-play vs V2.1 by paired, symmetrised 10k-deal MC (per-setup SE ~0.010) over six setups (4_5, 5_7, 6_6, 3_9, 5_11, 9_11).

| config | macro vs V2.1 (clean retrain) | verdict |
|---|--:|---|
| bluff alone | ~0 | inert alone |
| value alone | +0.0169 | helps on deep setups |
| **difftruthy alone** | **+0.0217 (6/6)** | best single |
| difftruthy + bluff | +0.0190 | below difftruthy-alone (bluff hurts) |
| value + difftruthy | +0.0193 | below all-three |
| **all three** | **+0.0256 (6/6, ~6σ agg)** | **adopted → V3** |

The macros interact **non-additively**: adding value *or* bluff singly to difftruthy slightly *hurts* (both 2-combos fall below difftruthy-alone), yet adding *both* gives the best point estimate. The all-three margin (+0.0256) over difftruthy-alone (+0.0217) is within ~1 aggregate SE, so the choice of all-three over the cheaper single macro was a deliberate cost/quality call (3 macros ≈ 1.4× wall, see §5.4), not a statistically forced one.

**Serving is a top-up, not a gate.** Measured by a serve-time A/B on the *same* macro-trained model — one side folding the macro masses onto their per-hand `b*` at play, the other serving only the renormalised concrete slice (the harness's "macro-folder on/off" toggle; 40k deals/setup): of the +0.0256, only +0.0124 comes from *resolving* the macro per-hand at play; the remaining ~+0.013 (= fold-off vs the non-macro-trained baseline) comes from training-with-macros improving the underlying *concrete* strategy and survives even if the deployed agent never resolves a macro (it would just serve the macro-trained concrete distribution). So shipping the macro-trained model captures most of the gain regardless; the per-hand fold is the worthwhile remainder.

**Gate to total ≥ 7.** A small-setup study (2_2, 3_3, 2_4 near-lossless; 4_4 the lossy boundary) trained no-macro vs all-three. H2H was within noise everywhere, but the decisive signal — invisible to a single-opponent H2H — was LBR-2 on the concrete strategy: macros **raised** exploitability 2.3–4.2× on the near-lossless setups (2_2 2.12% → 6.93%, 3_3 1.91% → 7.95%, 2_4 ~4.3% → ~10%). In the near-lossless regime there is no abstraction loss to recover, so a macro only adds exploitable distortion plus ~1.65–1.96× compute. **Macros are therefore enabled only for total ≥ 7.**

---

## 4. Per-round training tweaks

Two further results from the V2.1 lineage. The first was adopted into V3; the second informs the planned final retrain.

### 4.1 min_bet = 27 for total 13–15 → adopted into V3

In card-rich setups the sub-straight claims (high card / pair / two-pair) are almost always true, so they are strategically near-trivial *and* exactly where same-iteration node-revisiting hurts convergence. Raising min_bet to 27 prunes that subtree. Trained at V2.1 params and H2H'd against V2.1 over the 14 sum-13–15 setups, min_bet=27 **beat V2.1 in 13 of 14** (mean +0.64%, sign-test p ≈ 0.0018; edge growing with size, sum13 +0.30% → sum15 +1.16%; only the lone symmetric 7,7 regressed). It was simultaneously **~2.8× faster** to train, used **~50% less RAM**, and was flat against greedy Perun (mean diff +0.0006). So it is not merely cheaper — pruning the trivial low claims *sharpens* the meaningful play. **Adopted into V3** (V3 uses min_bet 27 for total ≥ 13; ≥ 16 was already 27 under V2.1's 0/4/27 schedule).

### 4.2 Early-game iteration scaling (10/20/40M, rounds 1–5) → forward plan, not V3

The rounds-1–5 setups (sums 2–6) are the one substrate where abstraction is near-lossless and LBR-2 is affordable, so iteration count can be probed cleanly. At V2.1 params, the 9 small setups were trained at 10/20/40M and LBR-2'd vs the 5M baseline. **5M badly under-trains the early game** — every non-trivial setup roughly halves its LBR-2 from 5M→40M (2,4 sp0 5.12% → 2.77%; 1,5 sp1 5.82% → 3.25%). 20M clearly beats 10M everywhere, and **even 40M has not converged** on the bigger early-game setups (2,3 / 1,5 / 2,4 / 3,3 still drop 0.25–0.70 pp from 20M→40M ≫ noise), so they want > 40M. The win is best-responder-facing: vs greedy Perun the same change is flat (+0.23 pp), but a hybrid V2.1 (rounds-1–5 → 40M, sum-13–15 → mb27) beat V2.1 whole-game 0.5100 ± 0.0050. **Not in V3** (V3 is 5M everywhere); this is the basis for the planned final retrain's early-round iteration budget (§8).

---

## 5. V3 — the macro anchor

### 5.1 Design

V3 = the V2.1 hyperparameters (penalty 0, pruning −10/−12, fp32, 5M iters) plus, per setup: **min_bet = 27 for total ≥ 13** (else 0, §4.1), and the **all-three macros for total ≥ 7** (else the plain trainer, §3.2 + the ≥7 gate). 66 setups, trained 8-wide, biggest-total first.

### 5.2 Implementation (and why it is more than "three numbers")

The deployed artifact is exactly what the user's intuition expects: the macros are **three extra columns on `strategy.npz`** (`masses[n,3]` + `kinds`), resolved dynamically at serve. There is no separate file and no new node type. The supporting code is the *machinery* that computes, trains, and validated those three columns:

- `probs.py` / `probs_jit.py` — the existence-probability kernels (`p_vector`, `g_vector`, the numba `p_vector_fast`). The macros are *defined* by `p` and `g`, so these are used at **both** training (to find each macro's per-hand `b*`) and serving (the agent folds the macro's mass onto `b* = argmax` of `p` / `p−g` / `g−p`). They are the only abstraction files the Lambda imports, and the reason the macro serve path pulls in numba (lazily, only for macro setups; concrete setups stay numba-free).
- `trainer_macros.py` — the `Trainer` subclass that adds the N macro columns and force-traverses each to its `b*`. Required to produce a macro model at all.
- `run_macros_training.py` — a ~20-line adapter that swaps the macro trainer into the production `training.py` and writes the single unified `strategy.npz`. It exists so the core trainer stays macro-agnostic (it trains both concrete and macro setups).
- `selftest_macros.py` + `selftest_bluff.py` — the H2H *experiment* harness that produced the §3.2 table and chose all-three for V3 (`selftest_bluff` supplies the standard-model loader and is the single-macro precursor). These are decision tooling, not deployment code, and not needed at serve.

So the runtime cost of the feature is small (three columns + a ~25-line fold in `agent.py`); the file count reflects the *math* (p/g kernels), the *training path* (trainer + adapter), and the *validation* (selftest harness) behind those columns.

### 5.3 Results

Validated on the box, 10k whole games per matchup (win-rate from the new model's view, ±0.005):

| matchup | win-rate | verdict |
|---|--:|---|
| **V3 vs V2.1** | **0.5348** | V3 wins ~53.5% |
| **V3 vs greedy Perun** | **0.5381** | V3 **wins** |
| V2.1 vs greedy Perun (matched re-run) | 0.4889 | V2.1 **loses** |

The headline is the **+4.9 pt swing against greedy Perun** — V2.1 lost to it (0.489), V3 beats it (0.538) — while also beating the V2.1 anchor directly. Per-setup vs V2.1 (`selftest_macros`, 10k deals over the 57 macro setups) was **+0.0200 mean, 52/57 positive**: flat on the near-lossless small setups (as the ≥7 gate predicts) and +0.02 to +0.049 on the lossy mid/large setups. The whole-game margin is larger than the per-round edge (round-level ~0.507) because a consistent small per-round advantage compounds over the ~19 rounds of a full game.

### 5.4 Cost / KPIs

Clean across all 66 setups: min_bet rule 100% correct, all 5M iters, 0 errors, no OOM, peak RAM 3.96 GB. Total train wall **1.24× V2.1** — the macros cost ~1.6–1.8× on the min_bet-0 setups (a real memory-bandwidth cost of the extra fp32 columns on the memory-bound traverse: node-work is only ~1.09× and RAM ~1.06×, but throughput dropped from ~868 it/s 8-wide to ~548 it/s 6-wide), *offset* by min_bet-27 being 0.43–0.77× of V2.1 on the total-13–15 setups (V2.1 used min_bet 0/4 there). Node-revisit magnitude from the diagnostic touch counters is ~600–900× per infoset.

---

## 5.5 V3.1 -- the hand-abstraction tuning round (post-V3)

**What was tested.** The games diagnostic (§6) flagged flush response and the decay of challenge accuracy with card count as the abstraction's weak points. Four candidate `get_hand_abstraction` changes were trained against V3 to probe them:
- *value-only at total 7-8* -- the rounds-1-5 sorted-multiset abstraction extended upward (dropping the suit machinery, and the macros, for those totals), at both 5M and 10M iters;
- *ranks@root* -- the distinct-rank count appended to the opening (last_bet=88) token;
- *suits@root* -- the per-suit count shape appended to the opening token;
- *flush-band refinement* -- replacing the flush token `(count[claimed suit], global max strength)`, which doubles the claimed-suit information (the global max is usually that suit), with `(augmented claimed-suit sf, max strength excluding that suit)`.

Each variant was retrained at matched V3 params and scored by paired symmetrised MC H2H. Because these are the first changes to the *abstraction* (not the action set), a per-model-abstraction harness was needed: `selftest_macros` keys both models with the global `information_set`, so `scratch/h2h_macro.py` was written, in which each model loads its own snapshot info_set + macro masses and acts by the agent's exact deployed fold; it reads ~0 on V3-vs-V3.

**Results.**

*value-only, total 7-8* (concrete; 100k-deal H2H, per-setup SE ~0.0032). At 5M it was a wash-to-behind V3 (total-7 ~flat; total-8 down to -0.007); at 10M it moved clearly positive against both its own 5M self and V3:

| value-only @ 10M | vs same abstraction @ 5M | vs V3 |
|---|--:|--:|
| total 7 (1_6, 2_5, 3_4) | +0.0080 (3/3, 4.4σ) | +0.0066 (3/3) |
| total 8 (1_7, 2_6, 3_5, 4_4) | +0.0140 (4/4, 8.8σ) | +0.0050 (4/4) |

Cost (settled training RSS): the value-only table is 1.1-4.3x the main abstraction's infosets, scaling with the largest hand -- ~3 GB at total-7's 1_6 (2.6x) but **6.3 GB at 1_7** (4.3x, total 8). Wall-clock at 10M concrete was below V3's 5M macro.

*root-token features* (15-setup asymmetric spread, one per total 8->22; 50k-deal H2H vs V3, SE ~0.0045):

| feature | mid-N (total 11-13) | high-N | mean (15 setups) |
|---|--:|--:|--:|
| suits@root | -0.012 to -0.019 (to 4.2σ) | +0.009 (17-18), +0.018 (21), +0.028 (22) | +0.0011 |
| ranks@root | ~0 | ~0 | -0.0001 (no trend) |

An earlier 8-setup read of ranks@root was +0.0010 (7/8); it did not replicate on the asymmetric spread.

*flush-band refinement* (8 flush-relevant setups, total 14-22; 50k-deal H2H vs V3): mean **+0.0001 (4/8)**, with only a faint mid-N-positive / high-N-negative gradient.

**Conclusions.**

1. *5M under-trains the richer abstractions.* The value-only result is the clean demonstration -- identical abstraction, 10M beats 5M by +0.008 (total 7) / +0.014 (total 8), and only at 10M does it overtake V3. A finer abstraction has more infosets, so at a fixed budget each is less converged and a 5M H2H under-states it. This is the H2H counterpart of §4.2's LBR-2 result (rounds 1-5 halve LBR-2 from 5M->40M). Rule: judge a richer abstraction at ~2x iters.
2. *value-only is adopted for total 7 only.* It wins at total 7 and 8, but 1_7's 6.3 GB exceeds a 4 GB/CPU training budget, so V3.1 takes it at total 7 (<=3 GB), at 10M.
3. *a root feature has outsized leverage and must be gated.* The opening is ~half the first moves, so suits@root helps strongly where the suit shape is informative (high-N) and hurts where it is not (mid-N -- it fragments the opening into undertrained buckets). Adopt suits@root gated to total >= 17; drop ranks@root (null).
4. *the flush refinement cannot be judged by H2H.* The flush band is ~8% of decisions (mostly responses) and self-play cannot see exploitability, so its wash is uninformative; deferred to an LBR round.

**V3.1** is the resulting synthesis:

| total | abstraction | trainer | iters |
|---|---|---|---|
| <= 6 | value-only | concrete | 10M |
| 7 | value-only *(was main+macro)* | concrete | 10M |
| 8-16 | main | macro | 5M |
| >= 17 | main + suits@root | macro | 5M |

It is monotonic over V3 (the changed regimes improve, the rest is unchanged, the suits mid-N harm is gated out). The whole-game gain over V3 is expected small-but-positive (the changes touch only total-7 and total->=17 rounds), pending cfr_vs_cfr (vs V3, no-regression) and cfr_vs_nfsp (vs greedy Perun) before V3.1 anchors.

---

## 5.6 V3.1 — whole-game validation and behavioural diagnostic (vs Perun / V3 / V1)

**What was run.** After the 66-setup V3.1 retrain (value-only @ 10M for total ≤ 7; main + macros @ 5M for total ≥ 8; suits@root for total ≥ 17), V3.1 was put through full real-mechanic whole-game H2H on the box (20k games per matchup, ~385k rounds, 8-wide). Because V3.1 is the first version to change the *hand abstraction* (not just the action set), the existing whole-game tools — which key both models through one global `information_set` — could not host two abstractions at once. `scratch/h2h_wholegame.py` was written for it: it reuses the deployed decision path (`cfr_ai.agent`, one independent module instance per model) but **rebinds** each instance's make_key / get_hand_abstraction / get_possible_actions to that model's own snapshot, and stages the shared sparse-mmap layout so 8 workers stay low-RAM. vs-Perun uses the established `cfr_vs_nfsp_games --nfsp-greedy` (only V3.1 carries an abstraction there). A V3-vs-V3 run is the harness null.

**Results — whole-game win-rate (B = the newer model; SE 0.0035 at 20k games):**

| matchup | win-rate | note |
|---|--:|---|
| gate: V3 vs V3 (harness null) | 0.5037 | identical models → ≈ 0.500; harness sound |
| **V3.1 vs greedy Perun** | **0.5502** | V3 was 0.538; V2.1 lost at 0.489 |
| **V3.1 vs V3** | **0.5068** | no-regression (≈ even, direct) |
| **V3.1 vs V1** | **0.5342** | — |
| V3 vs V1 (triangulation) | 0.5248 | — |

The three independent yardsticks **triangulate to V3.1 ≈ +0.7–1.2 pt over V3**: +1.2 pt vs Perun (0.550 vs 0.538), +0.94 pt vs the common opponent V1 (0.534 vs 0.525 — both measured here, so V1's coverage gaps cancel; ~1.9σ), and +0.7 pt direct over parity. The direct V3.1-vs-V3 is small in aggregate because the abstraction changes touch a minority of game-states; the per-total-cards breakdown localises the gain exactly where designed — **total-7** (value-only) slightly positive, and **total-22 / 11,11** (suits@root) strongly positive: V3 was actually *behind* V1 there (round win 0.495) while V3.1 is *ahead* (0.514), and the V3.1-vs-V3 total-22 cell reads 0.531 (~3.9σ). CFR-vs-CFR matchups otherwise hug 0.5 (both equilibrium-seeking), which is why the margin over greedy-NFSP Perun (off-equilibrium, punishable) is the larger one.

**Behavioural diagnostic vs greedy Perun** (`perun_games_diag.py` on the 20k V3.1 log — the same instrument as the V1/V2.1 MORANA diagnostic). Game win-rate reads **55.0%** (independent confirmation of the 0.5502 above). Metric definitions (all percentages of decisions at the named claim band, full spec in `analysis/MORANA_GAMES_DIAGNOSTIC.md`): **chkT / chkF** = the side's check-(challenge-)rate when the *faced* bet is true / false — low chkT + high chkF is good, and the **discrimination gap = chkF − chkT** (points); **tru%** = share of the side's own raises whose claimed set is actually true ("response credibility" — below 50% means the raise is profitably callable); **blf%** = share of the side's raises that are bluffs (claimed set not held, `p−g < 0`); **ru** = share of the side's raises left unchallenged (a proxy for the raise being respected). Against the diagnostic's two named V1 defects, at the flush band (rounds-6+):

| signal | V1 vs Perun | **V3.1 vs Perun** | Perun's own |
|---|--:|--:|--:|
| discrimination gap (chkF − chkT) | 2.8 | **16.9** | 16.7 |
| raise credibility (tru%) | 48.7 (callable) | **52.1** (credible) | 54.2 |
| over-bluff rate (blf%) | 30.6 | **20.4** | 14.9 |
| raises left unchallenged (ru) | 8.6 | **14.8** | 27.5 |

The flush hole — the diagnostic's "primary capability target," where V1 could neither challenge discriminately nor raise credibly — is **largely closed**: V3.1's flush discrimination now matches Perun's and its flush raises are credible. The second V1 defect, the per-round edge **eroding with card count** (V1 drifted 53% → 48%, losing from N ≈ 6), is also fixed: V3.1's symmetric-NvN round win-rate **holds 51.5–52.7% through N = 10**, dipping below 50% only at N = 11 (48.2%, the 11v11 elimination edge). V3.1 still over-bluffs relative to Perun across bands, but far less than V1 did.

**Conclusions.**
1. V3.1 is a **validated, no-regression refinement of V3** (+0.7–1.2 pt, triangulated), adopted as the **anchor** for the V3.2 round (since superseded by V3.2 — §5.7–5.9); the model lives at `cfr_ai/experiments/v31` (box) and the info_set is its own snapshot, distinct from the now-canonical V3.2 `information_set.py` (which drops the round-6 suit-flag).
2. Relative to the original V1/V2.1, the **flush hole is largely closed and the high-N erosion is fixed** — the two capability targets the MORANA diagnostic named. This is the cumulative payoff of the V3 macros + the V3.1 abstraction.
3. **Caveat / open.** The diagnostic measures V3.1 against the *V1* baseline, so the flush and high-N gains are cumulative (macros + abstraction); cleanly isolating the V3.1 abstraction's *own* contribution needs the same diagnostic on a V3-vs-greedy-Perun log (a fresh generation — not yet run). N = 11 (11v11) remains the one sub-50% per-round cell vs Perun, and the flush-band refinement (deferred from §5.5) still awaits an LBR round.

---

## 5.7 V3.2 ingredient — scaling the opening floor with card count (11-v-11)

**What / why.** V3.1's only sub-0.50 cell vs greedy Perun was the 11-v-11 endgame (0.482). At 22/24 cards almost every low/mid claim trivially exists, so opening low hands the opponent a free, safe escalation; V3's `min_bet = 27` (total ≥ 13) is too low for that extreme. The fix is to raise the *opening floor* (min_bet) — but how high, and for which setups? Two high floors were swept against the V3.1 default on the top-4 setups (action ids: flush = 66–69 for colours ♣♦♥♠, so **last full house = 65**, **last flush / flush-spades = 69**). Each variant is a clean min_bet-only retrain (macros, suits@root, 5M, pruning −10/−12).

**Results — vs greedy Perun** (per-setup forced N-v-N, 40k deals, SE ≈ 0.0025):

| setup (total) | mb27 (V3.1) | **mb65** (last FH) | **mb69** (last flush) |
|---|--:|--:|--:|
| **11_11** (22) | 0.484 | **0.502** | 0.496 |
| 10_11 (21) | 0.497 | 0.500 | 0.502 |
| 10_10 (20) | 0.511 | 0.515 | 0.515 |
| **9_11** (20) | **0.516** | 0.510 | 0.508 |

(H2H corroboration: the mb69 11_11 model also beat V3.1's own 11_11 head-to-head by +0.0207 (~4σ) and is ~9× leaner — 208k infosets / 6 MB vs 54 MB — since the trivially-true low subtree collapses; mb65 is a hair larger, same idea.)

**Conclusions.**
1. **11_11 → min_bet 65 (last full house).** It takes the 11-v-11 cell from a clear loss (0.484, −6.4σ below even) to **statistically even** (0.502, within ~1σ of 0.50): +1.8 pt over mb27 (~5σ) and **+0.6 pt over mb69** (~1.7σ). mb69 (flush-spades-only) was *too* restrictive — allowing the full-house + all-flush band plays better. **This supersedes the earlier mb69 banking.** Artifact: box `cfr_ai/experiments/v32_mb65/outputs/11_11`.
2. **No blanket high-N floor rule.** Raising the floor clearly helps only at the *symmetric extreme*. At 10_11/10_10 it is marginal (+0.4–0.5 pt, ~1–1.4σ); at **9_11 it HURTS** (mb27 0.516 is best, ~1.7σ) — the 9-card side makes flush/FH opens too aggressive. So the floor should rise only when *both* hands are near-maximal (total 22), not by total alone; 10_11 / 10_10 / 9_11 keep mb27.
3. **Principle (refined):** the opening floor scales with card count, and the right floor is the **highest *meaningless* (trivially-true) claim** given *both* hands — set min_bet at the top of the always-true range so an open can't sit on a claim the opponent would never challenge. At 22/24 full houses are still trivially true, so the floor sits at the **top full house (65)**; flushes (66+) are the first claims whose existence is actually uncertain. flush-spades (69) was *too high* — it excluded the still-useful lower flushes. **Caveats:** mb65 reaches even, not a win, vs Perun (the near-full-table endgame stays hard); H2H / vs-Perun only, not exploitability; a raised floor also blocks *challenging* a sub-floor opening bet (the min_bet quirk, true of V3's 27).

**V3.2 ingredient:** 11_11 → min_bet 65 (others unchanged).

---

## 5.8 V3.2 probes — decompressed flush band (wash) and the round-6 suit-flag (dropped)

Two abstraction questions on the V3.1 anchor, both via per-model H2H (`h2h_macro`, each model keyed with its own snapshot; run per-setup-parallel across the box cores).

**Decompressed flush band — comprehensive wash.** The §5.5 flush refinement (flush token `(count[claimed suit], global max)` → `(augmented claimed-suit sf, max strength EXCLUDING that suit)`, removing the "doubling") was retested on ALL main totals (8–22) vs V3 (the `v4_flush` `FLUSH_AUGMENTED` variant; 50k deals/setup, SE 0.0045):

| total (setup) | Δ | total (setup) | Δ |
|---|--:|---|--:|
| 8 (4_4) | +0.0027 | 16 (8_8) | +0.0071 |
| 9 (4_5) | −0.0040 | 17 (8_9) | +0.0015 |
| 10 (5_5) | +0.0028 | 18 (9_9) | +0.0013 |
| 11 (5_6) | +0.0045 | 19 (9_10) | −0.0031 |
| 12 (6_6) | +0.0002 | 20 (9_11) | −0.0014 |
| 13 (6_7) | +0.0028 | 21 (10_11) | −0.0053 |
| 14 (7_7) | +0.0015 | 22 (11_11) | −0.0006 |
| 15 (7_8) | +0.0012 | | |

No setup reaches 2σ. Faint gradient: total 8–18 slightly positive (agg +0.0020, ~1.4σ), total 19–22 slightly negative (agg −0.0026, ~1.2σ); best single 8_8 +0.0071 (~1.6σ). Self-play cannot see the flush band's value (a response-side / exploitability effect, ~8% of decisions) → **not adopted**; adjudication still needs LBR.

**Round-6 (total-7) max-suit-count flag → DROPPED.** V3.1's value-only total-7 token appends ` f<msuit>` when msuit ≥ 4 (a coarse flush-draw flag). It fires on 33% / 21% / 7.6% of tokens (1_6 / 2_5 / 3_4) and splits 181 / 66 / 15 value-multisets — heavily used. But a no-flag retrain (10M) H2H vs V3.1 shows the split *hurts* where it is most active and is neutral elsewhere (advantage = no-flag − V3.1):

| setup | 100k (SE 0.0032) | 300k (SE 0.0018) |
|---|--:|--:|
| 1_6 | +0.0057 | **+0.0041 (~2.3σ)** |
| 2_5 | −0.0014 | −0.0003 |
| 3_4 | −0.0001 | +0.0006 |

The flag never helps and costs infosets — fragmenting the table at fixed 10M iters outweighs the coarse signal (the §5.5 / small-setup over-resolution effect). **Dropped for V3.2** (total-7 value-only = the plain sorted multiset).

**V3.2 ingredients** (validated, monotonic over V3.1): (1) **min_bet = 65 (last full house)** for 11_11 (§5.7); (2) drop the round-6 suit-flag. The decompressed flush band is shelved pending an LBR round. **V3.2 is assembled** at box `cfr_ai/experiments/v32` (= V3.1 with those two swaps, V3.2-labelled, V3.2 info_set snapshot, diagnostics dropped, 4.5 GB).

**Whole-game validation** (20k games, box, 8-wide): **vs V3.1 = 0.5087** (no-regression; the gain sits at total-22, +1 pt from the mb65 floor) and **vs greedy Perun = 0.5502** (SE 0.0035) — matched to V3.1's 0.550, so the two swaps are net-neutral-to-positive on the whole-game aggregate (each touches only a thin slice of the deal mix). The **targeted fix landed**: the 11-v-11 round-win cell rose **0.482 → 0.504** (mb65 closed V3.1's only sub-0.50 endgame); by-total win-rate is ≥0.50 across the board except total-21 (0.4948, ~1σ), peaking ~0.52 at total-9/18. The by-total round-win rates (V3.2 vs greedy Perun, 385,042 rounds over 20k games; n ≈ 20k rounds/total through 14, tapering to 5,093 at total-22):

| total | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| win | .522 | .498 | .496 | .508 | .513 | .508 | .507 | **.520** | .517 | .516 | .509 |

| total | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| win | .506 | .508 | .511 | .509 | .515 | **.519** | .518 | .514 | .495 | .504 | |

Whole-game 0.5502 splits 0.5533 opener / 0.5471 non-opener; median game 20 rounds (13–21). (The raw report also carries the full starter × non-starter grid; it adds colour but nothing load-bearing beyond the tables above.) **Validated but NOT deployed** — retrained fresh at 2× iterations as V3.2x2 (§5.9).

## 5.9 V3.2x2 — doubled-iteration retrain (deployed)

Rather than deploy the assembled V3.2, retrain the **same V3.2 abstraction at 2× iterations** and deploy that: macros 5M → **10M** (total ≥ 8), value-only 10M → **20M** (total ≤ 7); min_bet unchanged (0 / 27, **11_11 → 65**); 8-wide, box `cfr_ai/experiments/v32x2`. Launched 01:07 UTC 2026-06-08 via an autonomous systemd chain (`scratch/chain_v32x2.sh` waits for the Perun unit to free the 8 cores, then trains; survives a dropped connection). MemoryMax 27 GiB cgroup backstop — never tripped (8-wide peaked ~13 GB at 10M; V3.1 was ~17 GB at 5M).

**Iteration cost scales ~1.66×, not 2×.** Measured per-setup over the 17 biggest setups (total 16–22): doubling iterations raised training time by a **median 1.66×** (range 1.59–1.81), not the naive 2×. Cause: **regret-based pruning** (`pruning_range [−10, −12]`) — by the time the first 5M iterations finish, most branches sit below the prune threshold, so the second 5M re-touch far fewer active branches (cheaper per iteration); fixed JIT-warm-up + save overhead pulls the ratio down further. 11_11 is the outlier at **0.58×** (73 → 42 min) because the mb65 floor slashes its legal-action space, an effect orthogonal to iter-doubling. So V3.1's **86 CPUh** projects to **≈144 CPUh** for V3.2x2 (≈18 h ideal 8-wide wall → ~20:00 UTC finish) — not the ~176 CPUh a linear 2× would imply, nor the ~96 CPUh a 13:00 finish would (that ETA was an unforced under-estimate).

The whole-game validation, run mmap-staged at 8-wide, confirmed V3.2x2 as the strongest model on every axis: **0.5240** against V3.1, **0.5261** against V3.2, **0.5638** against greedy Perun (clearing the 0.550 gate by +1.4 pt, with the 11-v-11 cell at 0.510 and every high-N total now ≥ 0.50, so the endgame erosion is gone), and **0.5413** against the deployed V1. Doubling the iterations is therefore a real, significant gain everywhere, overturning the earlier worry that iterations might be exhausted: more compute still buys strength. One process lesson came at a price — the automated test OOM-killed the box three times before it became clear that a big-model whole-game head-to-head must mmap-stage the strategies first. `write_mmap_layout` lets the spawn workers share one copy, keeping an 8-wide run low on RAM; without it each worker loads its own ~13 GB anonymous copy.

V3.2x2 was deployed to the production Lambda on 2026-06-09 (function `blef-aiagent-cfr-container` in `eu-west-2`, the 1024 MB tier, a 4.17 GB arm64 image, via `scratch/finish_deploy.sh`). CloudWatch logs from two played games were healthy: no errors, a peak memory of 316 MB, and a cold-start init of ~3 s on the first invocation (the image pull) falling to ~0.7 s thereafter. The finding worth carrying forward is serving latency. The first load of each *new* setup costs ~1–2.5 s, because the larger 2× strategies make `strategy_meta.npz` expensive to inflate: its `keys`, `masses`, and `probs_offset` arrays scale with the infoset count and were being decompressed eagerly (profiling with `scratch/profile_load.py` confirmed that `abs.json` and the sparse probabilities, already mmap'd, are essentially free). The fix — implemented in `strategy_io.py` as the mmap-meta layout (`write_mmap_layout` writes `keys`, `probs_offset`, `lower`/`upper` and the macro `masses` as separate *uncompressed* `.npy` files that `_load_mmap` memory-maps and pages in lazily, with `probs_offset` narrowed int64→int32, both losslessly and with no retrain) — was deployed to the live Lambda on 2026-06-09 and measured against the pre-fix game still in CloudWatch. The result was one clear win and one disappointment. **Memory** fell from a peak of 316 MB (still climbing at game end, as each setup's meta was decompressed into anonymous RAM) to a flat **166 MB (−47%)**, since the meta now stays as shared file-backed pages — a real robustness gain, since a long-lived container can serve far more setups before nearing the 1024 MB cap. **Latency**, however, improved only modestly: the worst first-load fell from ~2.6 s to ~1.9 s (~−25%), *not* the order-of-magnitude the laptop microbenchmark (197→5 ms) had implied. That benchmark measured only the meta-decompress CPU with a *warm* page cache; on Lambda the big-setup first-load is dominated by **I/O — page-faulting tens of MB of `.npy` from the read-only overlay filesystem on a ~0.58-vCPU (1024 MB) container** — and un-compressing the meta (0.37 → 1.08 GB, image to ~4.9 GB, still under half the 10 GB cap) *adds* bytes to fault, so the net is just the saved decompress CPU. The fix is kept deployed (half the memory, ~25% faster, no errors), but the warm-cache projection was an over-claim worth recording as such.

The real lever for the residual 1–2 s first-load is to shrink the bytes that get faulted, which points at **8-bit quantisation** of the served probabilities. Measured across all 66 setups (49 M infosets), dropping uint16 → uint8 cuts `probs_sparse_values` by **68%** (1049 → 334 MB — half the width, and the coarser 1/255 grid also drops 36% more near-zero entries) and the total faulted payload by **56%**, for a mean served-distribution shift of only **0.47% total-variation** (vs uint16's 0.003%). The cost is in the tail: ~13% of infosets shift
> 1% TV, the most-likely action flips in 1.25%, and uint8 truncates the rare-action mass uint16 keeps (mean 0.15%, up to 8.9% in a few hundred infosets) — a concern for *exploitability* (the rare mixing that resists a best-responder) more than for win-rate against greedy Perun. The head-to-head gate was run on 2026-06-10 against V3.2x4: a uint8-quantised staging of the model played its uint16 twin over 20k whole games (the macro masses rescaled by the same per-row factor, so the serve-time fold stays on one scale) and scored **0.5018 ± 0.0035** — statistically dead even, so the quantisation is strength-neutral against a sibling. It was adopted for the serving image: `write_mmap_layout` gained a `value_bits` parameter (16 remains the full-fidelity default used for evaluation staging; 8 is the serving option) and `stage_for_docker.py` stages the image at 8 bits, cutting the payload that first-loads page-fault by ~a third. The one caveat a sibling H2H cannot resolve — exploitability risk from the truncated rare mixing — was closed 2026-07-06 by an LBR pass on a uint8 staging of V3.2x4 (identical sampling to the recorded campaign: n_belief 500, n_lbr_hand 1000, seed 42; the 12-setup LBR-2 band, dequantised-served exactly as the Lambda serves): **no exploitability increase detected** — LBR-2 band mean +1.70% for uint8 vs +2.48% recorded for the fp original, and the uint8 point estimate is at-or-below fp on 11 of 12 setups (e.g. 3_4: 3.31/3.41 vs 4.79/5.46; 2_5: 3.32/5.26 vs 4.23/6.27; LBR-1 differences are inside the ±0.5–1.4 pp SEs). The coarser 1/255 grid drops thin near-zero mixing that the best-responder could otherwise lean on — quantisation behaves like a mild extra `clear_lows`, not an information leak. (Per-setup values: the uint8 column of the table in §5.12's exploitability-gate bullet — the two runs are one campaign.) Both the mmap layout and the quantisation matter most for V3.2x4, whose larger strategies would otherwise load more slowly still.

## 5.10 V3.2x4 — 4× retrain (deployed)

V3.2x4 retrains the same V3.2 abstraction at four times the iterations — 20M for the macro setups (total ≥ 8) and 40M for the value-only ones (total ≤ 7) — with the min-bet scheme unchanged (11-v-11 at 65). It was launched on the box under `experiments/v32x4` on 2026-06-09. The 20M macro runs need more memory per worker than V3.2x2's 10M, so this round trains 6-wide rather than 8-wide to stay under the box's 30 GB, with a 28 GiB cgroup backstop. Two transient units drive it: `v32x4` does the training, and `v32x4eval` waits for all 66 strategies, mmap-stages every model, then runs the 8-wide whole-game suite — against V3.2x2 (the central question of whether 4× beats 2×), V3.1, V1, and greedy Perun.

**Results (2026-06-10; 20k games per matchup, SE 0.0035).** Training finished after ~40 h wall = 237 CPUh (sum of per-setup training durations from metadata — the same figure §5.12's footprint table uses); the measured x4/x2 cost ratio is **1.64** CPU-weighted — the same pruning-driven sublinearity as the first doubling's 1.66. The eval confirmed V3.2x4 as the strongest model on every axis: **0.5184 vs V3.2x2** (the central question — ≈ 5σ), 0.5310 vs V3.1, 0.5544 vs V1, and **0.5716 vs greedy Perun** (V3.2x2 scored 0.5638). The second doubling bought +1.8 pt directly where the first bought +2.4–2.6: diminishing returns, but compute is not exhausted. (Operationally, the eval itself cost ~1 h: once the strategies are mmap-staged and cached, the CFR-vs-CFR games run at lookup speed — 20k whole games in ~30–45 s — and only the Perun leg, bound by NFSP inference, takes its usual ~hour.)

**Where the gains occur — the undertraining map** (per-setup decomposition of both 20k-game round logs, ~385k rounds each, rounds pooled by unordered setup; within a cell rounds come from distinct games, so the per-cell binomial test is exact). The **value-only setups (total ≤ 7) are statistical zero in both doublings** — round-win 0.5012 (z = +0.8) then 0.5018 (z = +1.3) on n = 120k each. They were already converged at vanilla V3.2's iteration level, at least as far as head-to-head strength can detect (§4.2's LBR view shows exploitability still improving there, but a near-equal sibling cannot punish it). The **macro setups (total ≥ 8) carry the entire gain**: 0.5060 (z = +6.2) for the first doubling, 0.5042 (z = +4.3) for the second, on n ≈ 266k each. Within the macro class the biggest movers shift toward the endgame as iterations grow — the first doubling's stars were 7_8 (0.5153, z = 3.4), 11_11 (0.5198, z = 2.9) and 10_11 (z = 2.4); the second's were **9_11 (0.5208, z = 3.9** — the one cell that survives a strict ×66 Bonferroni correction**)**, 10_10 (0.5202, z = 3.0) and again 7_8 (0.5135, z = 3.0). The per-round edges are thin (~+0.4–0.6 pp class-wide) but compound over a ~19-round game into the whole-game margins. The conclusion for any further compute: **the large-total macro setups remain undertrained at 20M iterations** — 7_8 absorbed both doublings productively and the totals-18–21 endgame keeps gaining — while the value-only band is done; further iteration spend should go to the macro setups only, biggest first.

V3.2x4 was deployed to the production Lambda on 2026-06-11 with the **uint8-quantised** serving payload (§5.9's gate having passed at 0.5018 over 20k games). Although the raw training outputs grew ~30% over V3.2x2 (almost entirely diagnostics, which never ship), the served payload is capped by `clear_lows` and the sparse format — non-checking rows grew only +7% and nonzeros +5% — so at 8 bits the staged payload came to **1.68 GB** (vs 2.66 GB for the uint16 V3.2x2 image) and the image *shrank* relative to its predecessor. Deployed via the usual cycle (stage → arm64 build → ECR push → digest-pinned cutover with the prior image captured as the rollback target); the in-container check confirmed uint8 values/masses, int32 offsets, and the x4 row counts on all 66 setups.

---

## 5.11 The 3-player line (1v1v1) — build, evaluation, deployment

Standard-rules 3-player Blef (24-card deck, max 8 cards per player, round loser gains a card, eliminated at the cap; the endgame after one elimination is served by the 2-player strategies). Built 2026-06-25 → deployed 2026-06-27; the current production 3p model is the 4× retrain (2026-07-04). §5.12 A/B-tests these models; this section is their record.

**Trainer generalization (n-player, traverser-centric).** `_traverse_jit` was reformed to return the *traverser's* payoff rather than the active player's, so values propagate with no sign flip — the 2-player negation trick generalizes to N. Three localized edits: terminal nodes compute the round loser from rotation position (checker = `(active+n−1)%n`, last bettor = `(active+n−2)%n`; loser = checker if the claimed set exists, else the bettor) and return −1 to the loser / +1/(n−1) to the others; opponent stepping `1−active → (active+1)%n`; both negations removed. The reward model is myopic single-round, constant-sum — like the 2p ±1 abstraction, it cannot express "I'd rather *this* opponent lose". The hand/action abstraction stack needed no change (set existence unions all hands; `get_hand_abstraction` keys on total cards; the macro probability machinery keys on (own hand, #unseen) — how unseen cards split among 1 vs 2 opponents is irrelevant). For n = 2 the reform reduces exactly to ±1: 2-player training is **byte-identical**, and an LBR-1 regression on the retrained code read +0.088% at 2M iters (unchanged). One reporting change followed later (2026-06-29): the n-player trainer initially recorded *position-averaged* per-player values (≈ 0 for symmetric setups); it now records the opener value (value to the player opening the round), and the shipped 2×-run's metadata was backfilled from 8k-deal self-play per opener (symmetric `1_1_1` ≈ +0.19 opener advantage, decaying toward 0.5/0 by high totals).

**Directed setups: 176, not 120.** Rotating the opener does not cover play *direction*: `2→3→4` and `2→4→3` are different games. Distinct directed cyclic orders over hand sizes 1–8 = (8³ + 2·8)/3 = **176 directed necklaces** (vs 120 multisets). Direction is worth real value — measured at 5M iters, the 1-card seat's value swings ~0.14 by direction (`1_2_3` −0.200 vs `1_3_2` −0.060; `1_2_4` −0.252 vs `1_4_2` −0.114). Serving selects the strategy file by the live table's cyclic order from the acting seat (min-rotation canonical, `agent.py::_directed_setup`; the players-list order is the engine's turn order). The V3.2 recipe carries over by total: concrete/value-only through total 7, main + macros from total 8, min_bet graduated 0 (≤12) / 27 (13–21) / 65 (22–23) / 87 (24).

**Cost.** 3p is *cheaper* than 2p at matched total (infoset count is dominated by the largest hand, and splitting a fixed total over 3 hands shrinks it: total-8 `2_3_3` 0.76M infosets/0.79 GB vs 2p `3_5` 2.50M/2.2 GB). Budgets trained: **1× = 10M/5M** (concrete/ macro; 176 setups, ~130 core-h, 0 failures), **2× = 20M/10M** (30h wall on 8 cores, also moved pruning −20/−22 → the V3.2-standard −10/−12 — a deliberately bundled config upgrade, so the 2×-vs-1× A/B measures the pair), **4× = 40M/20M** (55.6h wall on 8 cores). Deployable strategy.npz: 1× 4.29 GB → 2× 5.09 GB (1.19×) → 4× 5.65 GB (+16% over 2×, infosets +4.5% — the set is near-saturated; training RAM flat).

**Strength (all AI-vs-AI; the 3p line has no LBR instrument).**
* **vs the heuristic yardsticks** (1 CFR seat vs 2 identical opponents, seat-rotated, round-loss rate vs the fair 33.3%; 2026-06-26, 1× models): vs **conservative_bayesian** (the strong yardstick) CFR is stronger through total 9 (+0.8 to +4.9 pp, e.g. `3_3_3` +4.9 pp at 5.9σ), dips at the boundary total-8 band, and degrades at totals ≥ 10 — the **last-to-act seat's deep infosets are undertrained at 5M** (lookup hit-rate falls to ~70% at total 10; misses default to check and lose to true claims). vs NFSP CFR wins everywhere (+6 to +14 pp). Whole-game (1_1_1 → first elimination, 900 games): **first-out rate CFR 12.4% < conservative 15.1% < NFSP 72.4%**.
* **2× vs 1×** (mixed-table A/B, 633,600 rounds): 2× stronger by **+0.98 pp round-loss (z = 14.3)**, textbook undertraining signature — neutral through total 8, +0.4 to +4.2 pp at totals 9–22 (total-21 +4.2 pp, z = 9.3). The gain transfers to a real opponent: vs conservative_bayesian 2× beats 1× by +1.33 pp (z = 3.35), monotone in total (t20 +3.4 pp). Whole-game first-out 11.8% → 10.4% (~1σ — the whole-game metric dilutes a high-total edge). Re-shipped 2026-06-29.
* **4× vs 2×** (2026-07-04, 844,800 rounds + 18k games): 4× stronger by **+0.86 pp per-round (z = 14.6)** and +3.65 pp whole-game first-out (z = 9.0) — the 2× was still badly undertrained at high totals. Shipped 2026-07-04 (66 2p + 176 3p in one arm64 image, uint8 mmap serving; smoke + live-play validated, 0 errors over 79 live decisions, warm decisions 32 ms floor).

**Theory caveat.** 3-player is not 2-player zero-sum: CFR has no equilibrium guarantee here, and a Nash profile would anyway not be a safety guarantee against off-equilibrium opponents. The empirical gate above (beats both heuristic yardsticks and NFSP; wins real games) is the whole claim — "research-grade" in §8's original sense.

**Artifacts.** Model families `cfr_ai/p3` (1×), `p3_2x`, `p3x4`, plus the depth-2 twins (§5.12); each with per-setup `metadata.csv` (opener values) and a rebuilt `summary_of_all_runs.csv`. All six families are backed up off-box (OneDrive zips, 2026-06-29 → 07-06); the evaluation harnesses are `analysis/eval3p_*.py`, `analysis/h2h_persetup_mc.py` and `analysis/lbr_campaign.py` (promoted from scratch 2026-07-06).

---

## 5.12 History depth — dropping the third bet (`h_m2`)

The history abstraction (README §*History abstraction*) keeps the **last three** bets: `last_bet` exact, plus two independently-compressed codes `h_m1` (second-last) and `h_m2` (third-last). A `--history-depth {2,3}` knob (default 3) makes the depth a **per-model property**: depth-2 forces `h_m2` to absent everywhere, at both train and serve time. It is stored in `strategy.npz` / `strategy_meta.npz` / `metadata.csv` and read back per model, so a depth-2 and a depth-3 model serve correctly in one process (required for the H2H below). Default 3 is **byte-identical** to the prior trainer — a same-seed `2_2` pair at 200k iters produced equal `keys` and `probs` arrays.

**Footprint — the keyspace collapse.** Dropping `h_m2` merges ≈6 infosets into one. Measured across every setup of each deployed budget (metadata `Explored infosets` / `RAM taken (MB)` / `Training duration`, and `strategy.npz` bytes):

| | infosets | `strategy.npz` | train CPUh | peak RAM/setup |
|---|---|---|---|---|
| **2p V3.2x4** (66 setups) | 135.1M → **20.0M** (−85%) | 3.24 → **0.65 GB** (−80%) | 237 → **159** (−33%) | 4.05 → **0.80 GB** (−80%) |
| **3p p3_2x** (176 setups) | 335.9M → **57.9M** (−83%) | 5.09 → **1.66 GB** (−67%) | 233 → **196** (−16%) | 3.35 → **0.81 GB** (−76%) |

Every footprint metric drops **~5–7×**, proportional to the keyspace collapse; **training compute drops only 1.2–1.5×**, because iteration count is fixed and each iteration traverses the *same* game tree — depth-2 shrinks only the per-infoset bookkeeping, not the traversal (3p saves least: its 3-player traversal has the most infoset-independent work).

**2p strength — `h_m2` carries real but small value, and only once converged.** Per-setup H2H (`analysis/h2h_persetup_mc.py`: 10k symmetrised deals/setup, SE 0.010, depth-2 vs depth-3 at matched iters), plus whole-game (`cfr_vs_cfr_games.py`).
* **At 1× (V3.2, 10M/5M): strength-neutral.** Cross-section mean advantage **+0.0000 ± 0.0035** (95% CI ±0.007); no setup significant (largest 1.5σ), 3 positive / 5 negative.
* **At 4× (V3.2x4, 40M/20M): depth-3 wins, narrowly and macro-only.** Over all 66 setups, mean **−0.0063 ± 0.0012 (z = −5.1)**, median −0.0040, 44/66 negative, 7 significant-negative + 1 significant-positive. The loss is entirely in the **macro** class: concrete (total ≤ 7, n = 12) mean **−0.0001 ± 0.0036** (dead neutral — short betting rarely reaches a third-last bet), macro (total ≥ 8, n = 54) **−0.0077 ± 0.0017** (z ≈ −4.5). (Class = whether the model was trained with macros, read from each setup's metadata; the V3.2 boundary is value-only/concrete through total 7, macro from total 8.) It concentrates in a few setups — 3_9 **−0.051 (5σ)**, 7_10 −0.037, 11_11 −0.032, 4_4 −0.029, 8_9 −0.026, 4_8 −0.025, 3_8 −0.024 — with one reverse outlier, 2_4 **+0.034 (3.4σ)**, a tiny setup where depth-2's convergence beats `h_m2` noise. (The 8-setup cross-section alone read −0.0117; it happened to include 3_9 and overstated the board-wide mean.)
* **Whole-game (10k games, avg 19.3 rounds): depth-2 48.39% ± 0.50% vs depth-3 51.61% = −3.2 pp (3.2σ).** The ~−0.6 pp per-round edge compounds ~5× over the escalation.
* **Reading:** at 1× the keyspace is under-converged, so depth-2's ~7× more visits-per-infoset offsets the lost context (neutral); at 4× both converge and `h_m2`'s context pays off. The clean measurement of `h_m2`'s value is therefore the 4× result: **real, but small (~0.6 pp/setup, ~3 pp whole-game) and macro-concentrated.**

**3p strength — depth-2 *wins*, which measures 3p undertraining (cf. §5.10).** At the deployed 3p budget (p3_2x, 20M/10M) the sign flips. *Sign convention for this whole 3p block: every quoted edge is **depth-2's advantage in pp, positive = depth-2 stronger** (per-round = depth-3's loss-rate minus depth-2's; whole-game = depth-3's first-out rate minus depth-2's; both rates are "lower = stronger").*
* **Per-setup (`analysis/eval3p_ab.py`, 844,800 rounds): depth-2 loses rounds 33.00% vs depth-3 33.66% → depth-2 +0.66 pp (z = 11).** By total cards: neutral at low totals, depth-2 slightly *behind* at totals 10–11 (−0.4 to −0.6 pp), then winning increasingly at high totals — 13 +0.6, 14 +1.3, 15 +1.1, 17 +1.2, 18 +2.2, **19 +3.5 (z = 12)**, 21 +3.0 pp.
* **Whole-game (`analysis/eval3p_wg_ab.py`, 18k games, first-out rate): depth-2 31.14% vs depth-3 35.52% → depth-2 +4.38 pp (z = 11).** Depth-2 sits *below* the fair 33.3% (survives longer), depth-3 *above* it (busts out first); the per-round edge compounds ~6.6×.
* **Reading:** this is **not** evidence depth-2 is a better abstraction — it is a +4.4 pp measurement of how undertrained the deployed depth-3 3p is. §5.10 already flagged the large-total macro setups as undertrained at 20M iterations; here a *strictly less-informative* model beats depth-3 by 4.4 pp whole-game, exactly at the high totals where depth-3 holds the most infosets and visits each the fewest times. At 10M iters over 336M infosets, depth-3's `h_m2` splits are mostly unresolved regret noise; depth-2 pools ≈6 of them and averages the noise out. `h_m2`'s value only materialises once converged (the 2p-4× result), so the 3p-at-budget comparison is confounded by undertraining.
* **Depth-3 3p at 4× (40M/20M, the deployed-2p budget, box unit `p3x4`) — measured 2026-07-04.** Two questions, both from 844,800-round per-setup + 18k-game whole-game A/Bs:
  * **Does 4× substantially beat the 2× depth-3?** Yes, decisively — the deployed 2× depth-3 is badly undertrained. p3x4 (4×) vs p3_2x (2×), *both depth-3*: per-round **+0.86 pp (z = +14.6)**, whole-game first-out 35.16% → 31.51% = +3.65 pp (z = +9.0) for p3x4. The gain concentrates at high totals (per-round +1.0 to +3.2 pp for totals 9–21), exactly where §5.10 flagged the macro setups as compute-starved.
  * **Does the converged depth-3 retake the lead over depth-2, as at 4× in 2p?** No — it closes most of the gap but not all. p3x4 (depth-3, 4×) vs p3_2x_hist2 (depth-2, still 2×): depth-2 **+0.18 pp per-round (z = 3.0)**, whole-game **32.85% vs 33.82% → depth-2 +0.96 pp (z = 2.4)**. So doubling depth-3's compute shrinks depth-2's whole-game lead from +4.38 → +0.96 pp (~78% closed) — confirming undertraining as the dominant factor — yet depth-2, trained at **half the iterations**, still edges it. The transitive whole-game order is depth-2@2× ≻ depth-3@4× ≻ depth-3@2×.
  * **The clean matched-budget test — depth-2 @ 4× vs depth-3 @ 4× (both 40M/20M) — measured 2026-07-06 (box unit `p3x4_hist2`).** Depth-2 *still wins* at equal compute: per-round depth-2 **+0.38 pp (z = 6.4)**, whole-game first-out 32.37% vs 34.30% → depth-2 +1.93 pp (z = 4.8). So — unlike 2p, where converged depth-3 retook the lead at 4× — **`h_m2` never pays off in 3p at any budget tested.** The by-total breakdown says why: depth-2's aggregate win is carried by the high totals (+2.75 pp at total 19 / +1.75 at 20 / +2.17 at 21, all z > 4.9), exactly where 3p's much larger keyspace is *still* undertrained at 40M/20M and pooling the noise wins; only at low totals (7–8), where depth-3 does converge, does `h_m2` edge ahead (depth-2 −0.68 pp at total 7, z ≈ 1.9) — the 2p pattern in miniature, but drowned out by the high-total band. (4× also helps depth-2 itself: depth-2@4× over depth-2@2× = +0.48 pp/round z = 8.1, +1.96 pp whole-game — so both depths still had convergence headroom at 2×.) 3p is simply so much more infoset-hungry that the h_m2 crossover 2p reaches at 4× would need ≥8× here, if ever. Bottom line so far: in self-play, depth-2 is the stronger 3p abstraction at every affordable budget, with ~85% fewer infosets / ~80% smaller storage / half the compute — but the exploitability gate below reverses the production reading.
  * **The exploitability gate (measured 2026-07-06): depth-2 fails it.** LBR on the 2p depth-2 4× models (the matched-budget twins; identical sampling to the recorded campaign, 12-setup LBR-2 band): depth-2 is **~2.8× more exploitable** than depth-3 — band-mean LBR-2 +6.92% vs +2.48% — and worse on every setup in the band, from the tiniest up (1_1: +6.85% vs +0.02%; 2_2: +8.41 vs +1.52; 2_3: 4.56/10.08 vs 1.27/1.62; 3_4: 10.92/11.94 vs 4.79/5.46; even LBR-1, usually too loose to see anything, flags 1_1 at +2.97% and 2_2 at +1.83%). Crucially, on these same 12 setups the matched-budget H2H is a dead tie — band mean −0.0001 ± 0.0036, 5/12 positive — so where both instruments run, depth-2 buys *no* head-to-head benefit while paying the full exploitability bill; its H2H wins live in 3p (three-hand setups), where no LBR instrument exists. The mechanism is the one h_m2 exists to prevent: with the third-last bet pooled away, a best-responder crafts raise *sequences* whose distinct contexts the pooled policy must answer identically, and at least one of them is answered badly. AI-vs-AI H2H never finds these lines (a near-Nash sibling doesn't steer into them) — the fixed-opponent trap (§10-n), previously seen on parameter tuning (the alpha-4.5 rejection in the rules-based record), now measured on abstraction granularity. Production verdict: **depth-3 stays in prod for both 2p and 3p** (the 3p reading is a labelled inference — the pooling mechanism is player-count-independent, but the measurement is 2p). The full per-setup record — the H2H tie and the exploitability split, side by side (H2H = depth-2's round-win advantage at matched 4×, 10k deals, SE ±0.010; LBR-2 in % payoff units, sp0 / sp1 for asymmetric setups, recorded SEs ±0.5–1.4 pp; harness `analysis/lbr_campaign.py`, sampling (500, 1000), seed 42):

    | setup | H2H (d2 adv) | LBR-2: depth-3 fp | depth-3 uint8 | depth-2 4× |
    |---|--:|--:|--:|--:|
    | 1_1 | +0.006 | +0.02 | +0.02 | +6.85 |
    | 1_2 | −0.018 | +0.09 / +0.32 | +0.09 / +0.33 | +1.16 / +1.64 |
    | 1_3 | −0.007 | +1.24 / +0.53 | −2.14 / +0.17 | −0.39 / +5.76 |
    | 2_2 | −0.001 | +1.52 | +1.54 | +8.41 |
    | 1_4 | −0.004 | +0.35 / +2.69 | +1.36 / +1.81 | +2.86 / +10.51 |
    | 2_3 | −0.003 | +1.27 / +1.62 | −0.02 / +0.97 | +4.56 / +10.08 |
    | 1_5 | −0.011 | +2.12 / +3.02 | +1.75 / +2.66 | +4.06 / +7.03 |
    | 2_4 | +0.034 | +2.80 / +3.95 | +2.51 / +2.53 | +6.88 / +10.60 |
    | 3_3 | +0.002 | +2.43 | +1.17 | +7.79 |
    | 1_6 | +0.001 | +2.31 / +5.09 | +1.99 / +3.64 | +4.34 / +11.31 |
    | 2_5 | −0.004 | +4.23 / +6.27 | +3.32 / +5.26 | +8.50 / +10.53 |
    | 3_4 | +0.003 | +4.79 / +5.46 | +3.31 / +3.41 | +10.92 / +11.94 |
    | **band mean** | **−0.0001** | **+2.48** | **+1.70** | **+6.92** |

**Takeaways.** (i) `h_m2` carries **real but small strategic value that only pays off at convergence** (2p 4×: −0.6 pp/setup, −3.2 pp whole-game). (ii) The 3p result is a **diagnostic of undertraining** — a coarser model beats the shipped 3p by +4.4 pp whole-game, consistent with §5.10. (iii) The two eval lenses split, and the adversarial one decides. In *self-play*, depth-2 is the stronger 3p abstraction at every budget — it beats depth-3 outspent 2:1 (+0.18 pp/round, +0.96 pp whole-game) and at the equal 4× budget (+0.38 pp/round, +1.93 pp whole-game, 2026-07-06; signs per the block's depth-2-advantage convention), because `h_m2` only helps at the low totals that converge while the high totals stay undertrained even at 40M/20M and pooling wins there. But under *best response* the merged infosets are a liability everywhere: **~2.8× the LBR-2 exploitability** at the same matched budget (+6.92% vs +2.48% band mean, worse on 12/12 setups) — so the ~67–80% footprint saving is priced in exploitability, and depth-2 was not shipped for either player count (rule §10-n). It remains a legitimate lever only where an adversarial opponent is out of scope or the footprint constraint is binding (e.g. a future 2v2 model against the 10 GB image ceiling would have to re-run this gate).

---

## 6. Evaluation instruments

- **H2H** (`analysis/head_to_head.py`, `--monte-carlo`): 10k MC deals, variant payoff in [−1, +1], SE ≈ 0.01. The primary per-setup ranking signal. Reads `strategy.npz` directly, so it is identical across code versions.
- **Whole-game** (`analysis/cfr_vs_cfr_games.py`, `cfr_vs_nfsp_games.py`): plays full real-mechanic games (1v1 start, round loser gains a card and opens next, eliminate at 11 cards), logging one compact row per round to a `.npy`; aggregated to a whole-game win-rate. This is the primary *version-vs-version* and *vs-Perun* signal.
- **LBR-K** (`lbr.py`): local best response, a lower bound on exploitability that rises with depth. The inside view, but affordability-bound — LBR-1 finishes in < 1 h only on small/asymmetric setups; LBR-2 is feasible on sums ≤ 6; LBR-3 is infeasible (depth-3 on 1,8 ≈ days/call). Asymmetric setups (1,8 / 2,7) collapse the depth-1 enumeration and are the reusable cheap probe. Cost = (LBR hands) × (beliefs), linear in both; per-(hand,belief) cost varies ~5×, so the cheapest *exact* setup is 3,3, not the fewest-deals 1,5 — a clean combinatorial argument that inverted once measured. **Caveat (measured on V3.2x4, 2026-06): LBR-1 is a *loose* floor on Blef — depth-1→2 multiplies the recovered exploitability by 2–25× (3,3: −0.004% → +2.43%; 2,5: 0.10% → 6.27%), so an LBR-1 value can understate by an order of magnitude.** The duration wall is also steeper than "< 1 h": at lbr-hands 500 / beliefs 1000, near-symmetric total-9 setups take 1.5–3 days each (4,5 = 43 h, 3,6 = 73 h) and total-10+ is days→weeks, so a full LBR-1 sweep is infeasible; LBR-2 reaches the whole value-only band (total ≤ 7, max ~2.8 h), not just sums ≤ 6.
- **NFSP / Perun yardstick.** Perun is the deployed NFSP agent. Its **greedy** mode (argmax of the average policy) is a *pure*, non-Nash-approximating strategy but the only *competitive* yardstick (~0.50 round-win vs near-Nash CFR); never train against it. Its **sampled** mode is fuller but its off-policy tails are exactly what near-Nash CFR punishes: CFR's round-win rises ~9 pp (to ~0.59), which compounds over a many-round game to ~0.76 whole-game — a secondary view, not the realistic bar. A near-Nash strategy guarantees ≥ the game value against *any* opponent and beats genuinely bad ones, but does not *maximally* exploit a competent-but-imperfect opponent (that needs best-response), which is why CFR only draws greedy Perun rather than beating it — and why V3's macros, by adding a capability, can push past 0.50.
- **Games diagnostic** (`analysis/MORANA_GAMES_DIAGNOSTIC.md`). V1 beat humans 54.4% but lost to greedy Perun 48.4% (V2.1 48.6% — V1 ≈ V2.1 vs greedy, so the *abstraction* was the binding constraint, motivating V3 and beyond). Morana over-challenges and over-bluffs versus strong opponents and bets the least-hand-supported claims. The sharpest, triple-evidenced hole is **flush response**: her raises over a flush are true only 48.7% (callable) vs Perun's 54.7%, and she defaults to checking flushes (60.6%) — the hand abstraction lacks suit/flush weakness information. Challenge accuracy also degrades with card count as the abstraction coarsens. This directly motivates the next capability lever (§8).

---

## 7. Closed chapters — do not revisit

- **DCFR / CFR+ as external-sampling variants**: an `--algorithm {es,cfr_plus,dcfr}` switch added to the then pure-Python trainer (commit `123a601`). This was **not** canonical DCFR/CFR+: the opponent action and the deal were still *sampled*, so it was external-sampling MCCFR with a CFR+/DCFR discount (clip regrets at 0; α=1.5, β=0, γ=1) bolted on — the commit's own README concedes "MCCFR external sampling rather than full traversal" — and it ran with **pruning disabled**. On rounds 1–3 at 5M it was statistically indistinguishable from ES (the *abstraction* is the exploitability floor there; both saturate it), and the "10–18× slower wall" headline is **confounded**: the README attributes the slowdown to the disabled pruning plus the α-discount lookup, not to the update rule. So this closes the *sampled* hybrid only; it does **not** test or rule out a *proper full-traversal* CFR+/DCFR, which is the §8 forward-plan yardstick. The code was dropped in the numba rewrite and survives only at `123a601`.
- **Full-traversal CFR+ (the §8 yardstick — now closed 2026-06-08).** RM⁺ needs *exact* counterfactual regrets, which in Blef requires deterministic full traversal — and that is **intractable at any hand size**: every node branches over the 88 strictly-increasing claims for *both* players, so the betting tree has ≥ 2⁸⁸ lines, *independent of card count* (small **cards** ≠ small **tree**). Sampling is what tames the **betting** tree (not just the deal), so Blef has **no sample-free regime**. Two independent walls: **(a)** sampled RM⁺ is biased — flooring a noisy cumulative regret at 0 overweights bad actions (`E[max(X,0)] ≥ max(E[X],0)`); Brown & Sandholm 2019 (*Discounted Regret Minimization*) state CFR+ is incompatible with both sampling *and* pruning, which is the motivation for DCFR/LCFR. **(b)** full traversal is intractable as above. An RM⁺-on-MCCFR probe confirmed the pruning incompatibility empirically: floored regrets never cross the −10 threshold, so **pruning goes inert** → ~2.7× slower/iter and ~50 % more infosets on 2_2. The §8 premise below ("made feasible by pruning's collapse of effective branching") was wrong on its own terms — pruning is *inert* under RM⁺, so it cannot collapse anything. **CFR+ is structurally inapplicable to Blef; the variance-free exploitability yardstick is LBR (§6).** (Probe code was reverted; canonical trainer is unchanged.)
- **Outcome Sampling**: measured at the time as ~400–500× more exploitable than external sampling on Blef's 88-action shallow trees (variance blows up). The harness and per-setup numbers were **not preserved** — treat the figure as an order-of-magnitude recollection; the qualitative conclusion (OS variance is hopeless at this branching factor) is what the closure rests on. Closed; code kept only as a VR-MCCFR starting point.
- **C++ port**: closed on a cost/benefit judgement, *not* on a benchmark. The hot path is already the numba-JIT'd traverse kernel (§2.4), so a port could only buy whatever constant factor separates numba from hand-written C on these loops. That gap is **widely assumed small but was never measured for this trainer** — any specific ratio (an earlier draft of this doc guessed "1.3–2×"; that figure was never tested) is speculation. Weeks of porting work is unjustified for, at best, a constant factor, absent a ~100× scale-up need, which does not exist.
- **Action-list grouping / array-shortening** (Gen-B/G/H, §3.1): shelved; the infoset table is hand/history-bound and the traverse is memory-bound, so it yields no resource win.
- **Online subgame solver**: shelved 2026-05-28 — Phase-2 LBR-1 timings (sum-10 setups 5–6 h+) ruled it out at any non-trivial setup. Code in `cfr_ai/archive/subgame/`; its README has the revival path. The strategy-load fast path it motivated shipped anyway for its own benefits (the small Lambda tier).

---

## 8. Forward plan

1. **Hand-abstraction enhancement** — the #1 capability lever, steered by the games diagnostic: a compact, rank-agnostic *weakness* / bluff-quality signal at the high bet bands, especially the missing suit/flush information that makes Morana's flush responses incredible. **Largely executed in the V3.1 round (§5.5)** -- suits@root (gated >= 17) and value-only-total-7 (@10M) adopted, ranks@root dropped; the one piece H2H cannot resolve is the **flush-band refinement**, which needs the LBR round (item 2 below).
2. **CFR+ — closed (see §7), 2026-06-08.** The hoped-for full-traversal CFR+ yardstick is **infeasible for Blef**: RM⁺ needs exact regrets → deterministic full traversal → a betting tree of ≥ 2⁸⁸ lines (intractable at *any* hand size, since both players branch over all 88 claims at every node), and pruning — the very thing this item assumed would "collapse branching" — is itself *inert* under RM⁺ (floored regrets never cross the threshold). Blef has no sample-free regime, so CFR+ is structurally inapplicable; the variance-free exploitability yardstick is **LBR (§6)** instead.
3. **Final retrain — executed and superseded (§5.9–§5.10).** V3.2x2 (20M/10M) and then V3.2x4 (40M/20M value-only band, 20M macro band) were trained, gated, and deployed 2026-06-11 with the uint8 serving payload. §4.2's "early rounds want ≫ 5M" is what sized the 40M band.

The human-vs-Morana games have already been pulled from DynamoDB and analysed; that work produced the games diagnostic (`analysis/MORANA_GAMES_DIAGNOSTIC.md`, §6), which is what now steers item 1.

**3+ players — built and deployed (2026-06-25 → 07-04),** no longer "longer-term". The concern stated here originally (no 2-player zero-sum ⇒ no equilibrium guarantee) stands in theory, but empirically the n-player traverser-centric generalization of this trainer produced a 176-setup directed 1v1v1 model that beats the strongest heuristic yardstick (whole-game first-out: CFR 12.4% < conservative 15.1% < NFSP 72.4%) and is live in production at the 4× budget. Its build/eval record is §5.11, and §5.12 A/B-tests its history-depth variants. Non-standard rules remain future work.

---

## 9. Deployment (Lambda)

The agent loads via `strategy_io.load_strategy_for_agent` (a `FlatStrategyAgent` with `np.searchsorted` lookup) and `stage_for_docker.py` converts each `strategy.npz` into a sparse-mmap layout (quantised values via `value_bits` — **uint8 for the production image since V3.2x2**, §5.9; uint16 remains the full-fidelity staging option) at build time, so resident memory stays ~tens of MB regardless of strategy size. V2.1 ran on the **256 MB** tier (cold ~2.5–3.5 s), with no numba on the agent path.

For V3 the macro columns ride **inside** `strategy.npz` (and the mmap meta), so the macro masses are carried through the same compact path; the agent folds them onto the resolved per-hand bet at serve. The serve path stays **numba-free** — it uses the pure-Python `p_vector_factored`, with the numba `p_vector_fast` kept as a training-only kernel (the two are bit-exact). This V3 work was superseded operationally by the V3.2x2 deploy (§5.9): the live function runs on the **1024 MB** tier with the sparse-mmap-meta serving layout.

---

## 10. Process & operational lessons

These are the discipline rules earned during the experiment phase.

- **a. Fabrication is the cardinal sin.** The "penalty=0 triples infosets" claim (measured ratio 1.003) and writing A/B numbers into a cron before the test ran were both wrong. Label measured-vs-hypothesis; never state a number you have not computed.
- **b. Check turns are read-only.** An erroneous `systemctl stop` batched into a "maybe it's done" check killed 8 in-flight trainings (the redo that inflated V2.1's CPU-h). Only `done == N` triggers a stop.
- **c. Investigate before declaring infeasible.** A "5 hour" LBR was ~1 h on the right machine — extrapolated from a slow laptop instead of checking the box's recorded durations.
- **d. Even a clean combinatorial cost argument can invert once measured** (the LBR-2 "1,5 is cheapest" deal-count argument, refuted by the ~5× per-(hand,belief) variance).
- **e. Periodic in-training LBR costs hours** — overturning the "negligible" assumption; a single LBR-1 on a sum ≥ 10 setup costs hours, so it is OFF by default.
- **f. IPv6 SSH to the box is flaky** — exactly one ssh per message, `cd` first, explicit `.venv/bin/python`, strip `\r`, collapse checks to one echo line.
- **g. Be precise when killing processes** — a substring-matching kill also matched bash wrappers; match exact pids/cmdlines.
- **h. Long box runs use `systemd-run` transient units** (survive SSH teardown; `-p MemoryMax=` as an OOM backstop), not tmux.
- **i. Judge a richer abstraction at convergence, not at 5M.** A finer abstraction has more infosets and is more under-trained at a fixed budget, so the 5M H2H *under-credits* it: value-only at total 7-8 flipped from a wash to a +0.006-0.014 win at 10M (§5.5). Re-run promising-but-marginal richer abstractions at 2x before judging.
- **j. A root-token feature has outsized leverage -- gate it.** The opening is ~half the first moves, so a root feature helps a lot where its signal is real and hurts where it is not: suits@root was +0.028 at total 22 but -0.019 at total 11 (§5.5). Apply root features only where informative.
- **k. §5.12 is the sharpest test of rule (i).** Dropping `h_m2` (a *coarser* abstraction) is neutral at 1× 2p, loses once depth-3 converges at 4× 2p (depth-2 −0.6 pp/setup), yet *wins* by +4.4 pp whole-game at the undertrained 2× 3p budget — the finer model's extra infosets are unresolved regret noise there. A coarser model beating a finer one is *prima facie* a measurement of undertraining, not of the abstraction; judge abstraction richness only at convergence. The 3p coda sharpens the rule: at the matched 4× budget depth-2 *still* wins (+0.38 pp/round) — so the converged judgment can land either way (2p: keep `h_m2`; 3p: drop it), and only the matched-budget test distinguishes "undertrained artifact" from "genuinely better abstraction" — after which rule (n) still applies: the matched-budget H2H winner (depth-2) failed the best-response gate.
- **l. Version-vs-version whole-game evals must load each strategy once.** Loading dominates the policy lookup, so a fixed-size cache that reloads as games escalate through setups is catastrophic — **1.7 s/game vs 0.005 s/game (~340×)** on the 3p whole-game A/B. Mmap-stage the strategies (≈50 MB lazy-paged vs ~22 GB padded), use an *unbounded* cache, raise `ulimit -n` (≈7 mmap FDs per cached setup), and `os._exit` past the numba+fork multiprocessing-`Pool` shutdown deadlock (`analysis/eval3p_wg_ab.py`; 18k games in 39 s once fixed).
- **m. `RuntimeMaxSec` is a backstop, not a schedule.** A 25 h cap killed a 3p training at 169/176 when it overran; set a completion-critical run's cap several-fold beyond the estimate (or omit it) and keep the scheduler resume-safe, so a premature kill costs only the in-flight setups.
- **n. Self-play H2H cannot gate an abstraction change — only best-response can.** §5.12's depth-2 won *every* AI-vs-AI comparison at the matched budget (+0.38 pp/round, +1.93 pp whole-game) and still turned out ~2.8× more exploitable under LBR-2 (+6.92% vs +2.48% band mean, worse on 12/12 setups). Near-Nash opponents never walk into the lines a merged infoset misplays; a best-responder lives there. Corollary: an H2H win for a *coarser* abstraction is compatible with a large hidden exploitability bill, so any abstraction-coarsening ship decision needs an LBR (or equivalent adversarial) pass at matched budget — the same reason the rules-based record rejected alpha 4.5 on human-exploitability despite fixed-opponent gains.
