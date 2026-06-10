## Appendix B: Outcome sampling investigation (negative result)

We implemented and evaluated outcome-sampling MCCFR (OS) as a potential alternative to the default external-sampling (ES) algorithm. OS is theoretically appealing for Blef: it samples a single trajectory per iteration instead of enumerating all of the traverser's actions at each decision node, which would dramatically lower the per-iteration cost. The hope was that the larger iteration budget per second of wall-clock would outweigh the higher per-iteration variance.

It didn't. This appendix records why we tried it, what we found, and the diagnosis. The implementation is kept in `cfr_ai/trainer_os.py` as a starting point for any future variance-reduced variant. The supporting harness lives in `cfr_ai/analysis/sampling_comparison.py`, `cfr_ai/analysis/os_trace.py`, and `cfr_ai/analysis/os_sweep.py`.

### Implementation

We followed the canonical OS formulation from [Lanctot 2009](https://papers.nips.cc/paper/3713-monte-carlo-sampling-for-regret-minimization-in-extensive-games) / OpenSpiel. At each iteration, the traverser samples actions via ε-on-policy sampling (mix uniform exploration with σ to guarantee coverage); the opponent samples on-policy. Regret update at the traverser's infoset *I* with sampled action *a\**:

```
util       = u_traverser(terminal) / sample_reach_at_terminal
Δr(a*)     = util · (1 − σ(a*)) · opp_reach · tail
Δr(a≠a*)   = util · (−σ(a*))    · opp_reach · tail
```

where `opp_reach` is the opponent's σ-reach to *I* and `tail` is the traverser's σ from below *I* to the terminal, both computed during the trajectory walk.

We tested three variants of the regret update:
* **plain** — canonical OS as above;
* **cfr_plus** — clip cumulative regrets at 0 after each update;
* **dcfr_linear** — `cfr_plus` plus a per-iteration α-discount on positive regrets (α=1.5), with linear strategy averaging.

### Hyperparameter sweep on setup (1,1)

We ran a 15-config sweep at 100k iterations (single seed). LBR-1 values (best in bold):

| ε | plain | cfr_plus | dcfr_linear |
|---|---|---|---|
| 0.05 | 87.0% | 85.5% | 79.3% |
| 0.1 | 77.2% | 75.1% | 74.1% |
| 0.3 | 84.6% | 72.8% | 72.4% |
| 0.6 | 85.4% | 70.1% | 62.8% |
| 0.9 | **53.5%** | 82.1% | 79.0% |

We then re-ran the most promising configurations at 1M iterations:

| ε | plain @ 1M | dcfr_linear @ 1M |
|---|---|---|
| 0.1 | 59.8% | 70.4% |
| 0.3 | 89.6% | 60.4% |
| 0.6 | 82.1% | **55.6%** |
| 0.9 | 79.5% | 72.2% |

The `dcfr_linear` variant improved monotonically with more iterations at every ε. Plain OS improved at low ε but was RNG-volatile at high ε (the 53.5% at ε=0.9 / 100k iter swung to 79.5% at 1M iter from the same seed).

### Multi-seed validation

We re-ran the best config (ε=0.6, `dcfr_linear`, 1M iter) across 6 seeds:

| Seed | LBR-1 |
|---|---|
| 0 | 55.6% |
| 1 | 60.1% |
| 2 | 65.8% |
| 3 | 52.5% |
| 4 | 72.2% |
| 5 | 80.9% |

Mean **64.5%**, std **10.7pp**. The earlier seed-0 result of 55.6% was a lucky draw; the expected LBR across seeds is closer to 65% with a ±10pp spread.

### Does more compute help?

Same config, 10M iterations on seed 0: **53.3% LBR-1**, taking 42 minutes of training plus 7 minutes of LBR. Compared to seed 0 at 1M iter (55.6%), this is roughly a 2pp improvement for 10× more compute, indicating that OS at this configuration has nearly plateaued on (1,1). To match ES's LBR-1 of 0.13% at the same setup and iteration count would require many further orders of magnitude — almost certainly more compute than is practical.

### Comparison to ES baseline

At 100k iterations on the same setup, ES gives **LBR-1 = 0.13%**. The best OS config (1M iter, ε=0.6, `dcfr_linear`) is therefore approximately **400-500× more exploitable than ES** on (1,1).

### Diagnosis

`cfr_ai/analysis/os_trace.py` instruments visits to a target infoset and logs the per-action regret evolution. Running it on (1,1) at the root infoset for hand=Queen revealed three compounding mechanisms:

1. **Path-dependent σ-concentration.** Once any non-default action gains positive regret, σ concentrates on it and σ̃ for that action gets large. Because `coef_sampled = (1 − σ(a*)) · util` approaches 0 as σ(a*) → 1, the dominant action stops receiving its own updates, while other actions only get small `−σ(a*) · util · scale` contributions when the dominant action is sampled. σ becomes very sticky once concentrated, and which action wins the race-to-concentrate is essentially RNG-determined.

2. **Co-evolution of σ_traverser and σ_opp into bad local equilibria.** Even on seeds where the analytically-correct opening (e.g. Q-exists with hand=Q) wins the race at the root, the traverser still loses 91% of the time, because the opponent has learned to raise effectively and the deeper response infosets in the traverser's policy are poorly trained.

3. **88-action infosets amplify OS variance.** With σ̃(a) ≈ ε/n ≈ 0.007 for non-σ-concentrated actions, the IS-corrected `util = u/sample_reach` can reach ±150 for those rare samples. In Blef most random bets lose, so these huge updates are almost always strongly negative for whichever action got sampled — and per-iteration variance dominates the expected drift by a factor of ~12 standard deviations.

### Conclusion

OS is not competitive with ES on Blef's structure (88-action infosets, shallow trees) at any reasonable compute budget. We expect this could in principle be redeemed by variance-reduced MCCFR variants (baseline subtraction, VR-MCCFR, MIX-MCCFR), but those are larger research projects, and the marginal compute savings would not justify them against the alternative of either (a) cheaper cloud compute on the existing ES trainer or (b) algorithmic improvements that compound on top of ES (DCFR, subgame solving at runtime).

The OS code is left in place as a starting point for any future variance-reduced variant.
