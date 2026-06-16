# Rules-based Blef agents: tuning experiments and results

This document records experiments on the project's two rules-based agents, with their methods,
results, and conclusions. Results are kept here rather than in code comments, which should describe
mechanism only.

## Agents and methodology

**Agents under study.** Two heuristic agents share a common probability layer:

- the **conservative agent** (`conservative_ai`), a simple agent that bets and checks using only its
  own-hand set-existence probabilities;
- the **conservative-crawling agent** (`conservative_crawling_ai`), which additionally conditions on
  opponents' bets by estimating P(set *s* exists | an opponent's bet *B* is true) and blending that
  estimate into bet selection.

**Reference opponents.** Two trained agents serve as fixed benchmarks: a **CFR** agent
(counterfactual-regret model, checkpoint v32x4) and an **NFSP** agent (neural fictitious self-play,
evaluated with its greedy policy, which is stronger than its sampling policy).

**Metric.** Whole-game win-rate. In 1v1 this is the head-to-head win-rate; in 4-player free-for-all
(FFA, two of each type) we report a *type win-share* — the fraction of games won by a given agent type,
where >0.5 means that type is stronger. Seats are rotated across games for fairness. Uncertainty is
given as ±SE or in units of σ.

**Rule sets.** "Standard" = 24-card deck, no jokers. "Pro" = 32-card deck with one joker, four blanks,
and common cards (approximated as a fixed two).

**Eval harnesses** (in `scratch/`, gitignored): `config_eval.py` (crawling configs vs self or CFR, 1v1),
`perun_eval.py` (crawling vs NFSP across standard/pro × 1v1/4p), `dazhbog_eval.py` (conservative agent
vs self/CFR/crawling), `opening_profile.py` and `cliff_probe.py` (descriptive opening profiles),
`joker_attr_probe.py` and `validate_conditional.py` (probability-layer validation).

---

## 1. Conservative-crawling agent

### 1.1 Opponent-conditional probabilities and constant tuning

The agent's distinctive mechanism is the opponent-bet conditional P(s | B), blended into bet selection
with weight `LAM_OPP`. With the conditional enabled (`LAM_OPP = 0.75`) and the sampling/decision
constants tuned (`alpha = 4`, `beta = 2`, `check_mult = 1.5`, `check_exp = 4`), the agent beat both
reference opponents across all tested settings, improving on the original constants by roughly +13 to
+25 percentage points against NFSP. Reference win-rates for this configuration:

- vs CFR (1v1): ≈ 0.362–0.366 (run-to-run noise ≈ 0.005);
- vs NFSP: standard-1v1 0.388, standard-4p 0.386, pro-1v1 0.527, pro-4p 0.422.

A one-at-a-time sweep of the constants around this configuration (evaluated vs self and CFR, and vs
NFSP where promising) found it close to a local optimum:

| change | result | conclusion |
|---|---|---|
| `alpha` 4.0 → 4.5 | small but consistent gain across self, CFR, and all four NFSP settings | mild improvement; not adopted |
| `alpha` 4.5 / 4.75 / 5.0 ladder | self-play monotonically up (an artifact of decisiveness against a fixed timid opponent); CFR weak and non-monotonic | inconclusive |
| `beta` 2.0 → 2.5 | +1.4σ vs CFR but −4σ vs NFSP pro-1v1 | CFR-overfit; rejected |
| `check_exp` 4 → 3.5 | −2.5σ vs CFR | do not lower |

### 1.2 Second-meaningful-bet conditional

The blend reserves a `1 − LAM_OPP` (= 0.25) "self" slice that originally used only own-hand
probabilities. We tested filling this slice with a second conditional, on the *second-most-recent
meaningful bet* — "meaningful" meaning a bet not immediately followed by the bettor's own teammate (a
teed-up team raise being a weaker standalone signal). By position, this second bet is the agent's own
previous bet in 1v1 and a different opponent's bet in games of three or more players. The blend becomes
0.75·P(s | last opponent bet) + 0.25·P(s | second meaningful bet); its weight is `LAM_SECOND`.

- In 4-player FFA the second bet is a distinct opponent's, and conditioning on it helped: NFSP
  standard-4p type-share rose from 0.386 to 0.425 (+0.039, ≈ 3.5σ); pro-4p was neutral (0.421).
- In 1v1 the second bet is the agent's own previous bet; this was neutral vs CFR (0.368 ≈ baseline) and
  within the neutral band vs NFSP. Conditioning instead on a *different opponent* in 1v1 was worse
  (0.358), which is why the rule keys on bet position rather than identity.

Conclusion: neutral in 1v1, a real gain in standard 4-player FFA. Adopted (`LAM_SECOND = 1.0`).

### 1.3 Opening-shape investigation (negative result)

**Motivation.** Descriptively, the crawling agent spreads its opening bet across many sets, whereas both
equilibrium reference agents concentrate on the highest *safe, non-signaling* set, walking a ladder
Ace-high → great straight (mid-game 60–80% for the references vs ≈ 2% for the crawling agent) → full
house → flush → straight flush (measured with `opening_profile.py`).

**Method.** We added a "frontier/cliff" term that boosts sets sitting far above the opponent's best
higher set. An initial multiplicative, safety-gated form was neutral against the reference agents
because it rarely changed the chosen bet (a safe high set is usually already the top pick). To actually
reshape the opening we then tried additive forms in several functional shapes — a gap term
`generic − best_higher`; `generic·(½ − bh)`; `(generic − ½)(½ − bh)`; a three-exponent
`generic^p·(τ − bh)^q·gap^r`; and a multiplicative `exp(γ·gapᵖ)`. These reproduced the equilibrium
opening ladder descriptively — one form matched CFR's great-straight onset, and conservative settings
kept the agent's probability of an unsafe opening below CFR's own (CFR's P(unsafe) is 0.65–0.71 at
mid hand totals).

**Result.** Applied to all bet selection, the term *hurt* monotonically with its strength (self-play /
vs-CFR win-rates of 0.237/0.267, 0.379/0.326, 0.485/0.340 for three settings — all below baseline).
Confining it to the opening bet only recovered the loss, identifying the harm as *mid-round
contamination*: the term had reshaped raises and the check-vs-bet decision, not just the opening. The
opening-only variant was at best marginally positive (≈ 0.507 self / 0.371–0.374 vs CFR for the
strongest form, only 1–2σ).

**Conclusion.** Opening shape is not a win-rate lever for this agent; all frontier machinery was removed.
The durable lesson is that the agent's mid-round raise and check-vs-bet logic is sensitive and co-tuned,
so reshaping bet selection globally perturbs it.

### 1.4 Multiplayer joker attribution (correctness)

**Defect.** When estimating P(s | B), the conditional drew all opponents' unknown cards as one pool and
counted *every* joker in that pool as a wild usable by the conditioning bet B. By the rules only the
bettor's own jokers (plus common jokers) may fill B, so in games with ≥2 opponents and a joker in the
deck this let B borrow a joker actually held by a different opponent, inflating P(B).

**Correction.** B now uses only the bettor's own jokers plus common. Because the draw is pooled, the
bettor's share of the pooled jokers is marginalized analytically — hypergeometrically over which of the
k pooled cards are the bettor's n_b — which is exact and adds no sampling variance. Real cards continue
to pool across all players (a set exists from all cards on the table, which is correct). The heads-up
case (n_b = k) and the jokerless case reduce exactly to the previous pooled computation, so the change
is a no-op for the 24-card deck and for all heads-up play, affecting only ≥3-player pro games.

**Magnitude.** On an exact brute-force case (bettor holding 2 of 3 opponent cards, B = a pair), the old
pooling inflated P(B) by 14.8% and shifted the conditional P(s | B) by ≈ 0.09 absolute (≈ 15% relative);
the error grows as the bettor's share of the pool shrinks.

**Validation.** The corrected output matches an independent per-opponent brute-force oracle exactly, the
heads-up enumeration path is exact (error 0.00e+00), and the disabled-conditional parity is preserved. A
secondary issue — the analytic fractional weights starved the sampler's positive-count stopping rule,
inflating worst-case multiplayer latency — was fixed by separating an integer support count (for the
stop rule) from the fractional weight sum (for the estimate), restoring prior latency.

**Win-rate effect.** Measured by an A/B (pooled vs corrected, n = 3000 per cell, 32-card deck):

| setting | pooled | corrected | Δ |
|---|---|---|---|
| pro 4-player (affected) | 0.411 | 0.393 | −0.018 (≈ 1.4σ) |
| pro 1v1 (no-op control) | 0.521 | 0.511 | −0.010 (≈ 0.8σ) |

The heads-up control is a no-op for the fix, yet still differs by −0.010 from RNG alone (the runs are
unpaired and the policy is stochastic); this sets the noise floor, and the 4-player Δ barely exceeds it.
The correction is therefore win-rate-neutral — a correctness change, not a performance lever. The old
pooling had mildly inflated P(B) and so slightly over-trusted opponent bets; removing it costs nothing
measurable. (The blend weight `LAM_OPP` was tuned under the biased probabilities, so a small re-tune
specific to ≥3-player pro games is possible but not expected to matter.)

### 1.5 Cumulative effect vs the original agent

A head-to-head self-play comparison (24-card, 1v1, n = 4000, ±0.0079 SE) against the original agent
(no conditional; original constants `alpha 3, check_mult 1.2, check_exp 3`) decomposes the total
improvement:

| configuration | win-rate vs original | marginal contribution |
|---|---|---|
| tuned constants, conditional off | 0.660 | constant retuning alone: +16 pts |
| + last-opponent conditional | 0.714 | conditional: +5.4 pts |
| + second-meaningful-bet conditional | 0.720 | second conditional: +0.6 pt (1v1-neutral) |

The current agent beats the original 72.0% head-to-head in 24-card 1v1. The constant retuning is the
largest single contributor, the opponent-conditional adds a clear ≈ 5 points, and the second conditional
is neutral in 1v1 (consistent with its gain being confined to 4-player FFA).

### 1.6 Computational cost

Per-decision cost is dominated by the conditional estimate, which is bounded by a draw cap of ≈ 100k
hands on both the exact-enumeration and sampling paths. Worst-case multiplayer cost is ≈ 140–180 ms
locally, and the 32-card deck is no more expensive than the 24-card deck (the cap binds first). The
second-meaningful-bet conditional adds one more such estimate, roughly doubling the conditional cost in
mid/late-game multiplayer decisions where a second meaningful bet exists (the opening and first response
are unaffected). A possible safeguard, not yet implemented, is a wall-clock budget inside the conditional
that returns no estimate — degrading to hand-only play — before approaching any deadline.

---

## 2. Conservative agent

### 2.1 Check-aggression constant

The conservative agent is intentionally simpler and weaker than the crawling agent. We tested making it
check less by raising its bet-vs-check multiplier `check_mult`:

| check_mult | self vs 1.2 | vs CFR | vs crawling agent |
|---|---|---|---|
| 1.2 | 0.498 | 0.288 | 0.274 |
| 1.35 | 0.524 | 0.290 | 0.325 |
| 1.5 | 0.535 | 0.306 | 0.354 |

1.5 was chosen: it is a clean modest gain (1.35 barely moved the result vs CFR), and the agent remains
distinctly weaker than the crawling agent (0.354 ≪ 0.5), as intended. Its sampling exponent `ALPHA` (= 3)
and check exponent `CHECK_EXP` (= 3) were also exposed as parameters for symmetry with the crawling
agent, without changing their values.
