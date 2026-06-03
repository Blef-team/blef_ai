# Morana games diagnostic — humans (V1), V1- and V2.1-vs-greedy-Perun — 2026-06-03

This is a behavioural diagnostic of the CFR AI, written to characterise the weakness in its hand abstraction.

## Version lineage
- **V1** is the version that has been playing humans between July 2025 and May 2026. It was trained with a penalty of **0.05–0.1** on most setups, which "hurried" the AI.
- **V2** is a speed-up of V1 (precompute fixes and the like) with the same quality.
- **V2.1** removes the penalty and pulls pruning and minimum-regret close to zero; it is the anchor for the macro experiments.

## Datasets

| dataset | Morana | opponent | games | rounds |
|---|---|---|---|---|
| **human** | V1 | real humans | 561 (qualifying) | ~11k (6+) |
| **V1-vs-greedy** | V1 | NFSP **greedy** (deployed) | 10,000 | 192,307 |
| **V2.1-vs-greedy** | V2.1 | NFSP **greedy** | 10,000 | 192,698 |

The `cfr_vs_nfsp_games_v1_greedy.npy` set uses the same Morana version (V1) as the human games. Sampled NFSP is not deployed, so the benchmark is greedy NFSP, which best-responds (not specifically to Morana, but using pure policies) and is the harder opponent.
All game-record artifacts live under **`cfr_ai/analysis/data/`**, which is gitignored.

## Method
The analysis starts from a read-only DynamoDB dump written to
`cfr_ai/analysis/data/games_current.jsonl.gz`. Human games are filtered to 1v1 matches with `ai_agent=="cfr"`, `max_cards==11`, status `Finished`, and `last_modified ≥ 2025-07-10` (epoch 1752105600). 

Actions `0–87` are bets, `88` is a check, and `89` marks the lost round. Ground truth comes from two sources: the round's full set-existence vector,
`Game.precompute_set_existence(both hands)` (both hands are stored in finished game logs), and the `89` rule (when `89.player == 88.player`, the challenged set existed). 

We define `p−g` as the a-posteriori minus the a-priori existence probability, `bluff%` as the fraction of bets with `p−g < 0` (a measure of intent), and a set as "true" when the existence vector confirms it. 

Decisions are bucketed by the band being responded to (the context, `ctx`, is the last bet before the decision). "Rounds 6+" denotes `sum(n_cards) ≥ 7`, the regime in which the abstraction is lossy.

To reproduce the three analyses:
- `python -m cfr_ai.analysis.morana_games_diag --cutoff-epoch 1752105600`
- `python -m cfr_ai.analysis.perun_games_diag [--npy cfr_ai/analysis/data/cfr_vs_nfsp_games_*.npy]`
- `python -m cfr_ai.analysis.infoset_allocation` — the memory audit, which counts infosets in the V1 CSV archive `archive/v1_csv` (the `*_diagnostic` directories hold all infosets, i.e. RAM; the final directories hold only the non-pure-check infosets, i.e. storage) and compares them against the human-games faced-frequency.

## Winrate

| matchup | winrate |
|---|---|
| V1 vs **greedy** Perun | **48.4%** (loses) |
| V2.1 vs **greedy** Perun | **48.6%** (loses) |
| V1 vs humans | **54.4%** |

Greedy Perun beats V1 and V2.1 by the same margin (~48.5%), so the penalty difference between the two versions is immaterial here.

## Behavioural signature by responded-to band (rounds 6+)

Each row corresponds to the band of the bet being responded to (the context). The letters are used consistently: **T** means the relevant set is **true** (it exists) and **F** means it is **false** (whether a bluff or a good-faith bet that did not come in).
- **M %T / O %T** — the share of the faced bets (the other side's band-X bets) that are true: `M %T` is how often the opponent's band-X bet is true, and `O %T` how often Morana's is. Every column reads measure-then-state, so `%T` is the true-share and `chkT`/`chkF` (below) are check-rates *within* the true/false faced bets — **T** and **F** always denote the faced bet's truth.
- **M chkT / M chkF** — Morana's check (challenge) rate when the faced bet is true and false respectively. A low `chkT` and a high `chkF` are both good. The **discrimination gap** is `chkF − chkT`.
- **O chkT / O chkF** — the same two quantities for the opponent.
- **M/O blf%** — the percentage of that side's raises that are bluffs (`p−g < 0`, a measure of intent).
- **M/O tru%** — the percentage of that side's raises whose own set is true. This is the **response credibility**: below 50% means the raise is callable.
- **nM/nO** — the decision counts for each side.

**V1 vs greedy Perun:**

| band | M %T | M chkT | M chkF | O %T | O chkT | O chkF | M blf% | O blf% | M tru% | O tru% | nM/nO |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ROOT | — | — | — | — | — | — | 30.1 | 22.6 | 70.1 | 70.6 | 71629/70678 |
| high | 88.9 | 0.4 | 1.0 | 83.9 | 0.0 | 0.0 | 27.8 | 18.6 | 67.4 | 74.8 | 6172/7447 |
| pair | 75.3 | 15.6 | 28.4 | 66.4 | 16.5 | 27.2 | 24.3 | 15.8 | 59.1 | 56.5 | 16588/14647 |
| 2pair | 75.8 | 29.1 | 43.4 | 64.0 | 25.2 | 41.2 | 25.8 | 20.2 | 67.2 | 63.8 | 8259/9592 |
| straight | 60.3 | 33.8 | 55.2 | 68.3 | 19.1 | 36.1 | 19.7 | 16.8 | 59.1 | 59.2 | 22480/21593 |
| trips | 64.4 | 46.1 | 64.8 | 63.5 | 49.5 | 63.4 | 26.6 | 15.5 | 62.3 | 63.6 | 17551/20550 |
| fullhouse | 67.8 | 38.6 | 57.7 | 66.2 | 27.0 | 40.1 | 22.5 | 20.7 | 65.3 | 62.0 | 36432/36261 |
| **flush** | 56.8 | **59.4** | **62.2** | 65.7 | 35.9 | 40.9 | 30.6 | 15.1 | **48.7** | 54.7 | 20381/11471 |
| quads | 58.4 | 55.1 | 70.9 | 60.1 | 49.1 | 57.4 | 14.4 | 13.4 | 53.5 | 52.7 | 23202/23239 |
| sflush | 57.0 | 68.7 | 80.7 | 48.7 | 80.1 | 85.9 | 44.7 | 18.1 | 58.8 | 48.9 | 8233/7196 |
| great_sf | 55.6 | 83.1 | 87.4 | 55.4 | 95.2 | 97.5 | 6.7 | 24.1 | 45.1 | 35.7 | 7338/6384 |

**V1 vs humans:**

| band | M %T | M chkT | M chkF | O %T | O chkT | O chkF | M blf% | O blf% | M tru% | O tru% | nM/nO |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ROOT | — | — | — | — | — | — | 29.0 | 14.4 | 70.9 | 73.7 | 3910/3998 |
| high | 92.3 | 0.6 | 1.9 | 81.3 | 0.6 | 1.2 | 29.1 | 11.1 | 63.3 | 77.4 | 676/428 |
| pair | 77.7 | 16.7 | 30.8 | 63.9 | 13.1 | 21.8 | 24.9 | 7.1 | 56.7 | 62.1 | 757/888 |
| 2pair | 71.7 | 26.3 | 61.7 | 63.9 | 20.2 | 34.3 | 28.6 | 12.1 | 60.4 | 63.8 | 499/573 |
| straight | 66.3 | 22.2 | 54.5 | 67.8 | 12.8 | 23.8 | 24.0 | 9.3 | 59.8 | 62.4 | 1340/1241 |
| trips | 63.2 | 41.3 | 72.5 | 64.6 | 30.5 | 44.5 | 25.8 | 9.9 | 62.1 | 64.9 | 1510/1162 |
| fullhouse | 72.5 | 35.9 | 69.3 | 65.9 | 24.5 | 39.3 | 24.0 | 12.7 | 64.1 | 62.3 | 1946/2276 |
| **flush** | 60.8 | 54.1 | 65.2 | 63.8 | 35.6 | 49.6 | 31.9 | 11.4 | **48.0** | 51.9 | 1079/679 |
| quads | 56.7 | 50.4 | 78.9 | 59.2 | 50.9 | 63.2 | 12.9 | 7.3 | 50.4 | 53.3 | 1412/1278 |
| sflush | 53.0 | 68.3 | 76.1 | 48.9 | 66.8 | 72.0 | 45.4 | 22.7 | 51.5 | 48.4 | 464/419 |
| great_sf | 47.1 | 79.1 | 88.5 | 49.9 | 94.7 | 91.3 | 2.9 | 10.3 | 44.1 | 37.9 | 427/413 |

Morana over-bluffs at every band against Perun (for instance, 30.6% versus 15.1% at flush). The discrimination gap (`chkF − chkT`) runs at roughly 13–22 points against Perun and 28–35 against humans across the 2pair–quads bands, but it collapses at flush (2.8 against Perun, 11.1 against humans):
she challenges true and false flushes almost equally. Flush is also the only non-ceiling band where her `tru%` falls below 50% (48.7% against Perun and 48.0% against humans, compared with Perun's 54.7%), which makes her flush raises callable. In short, when she faces a flush she can neither raise credibly nor challenge discriminately, because the abstraction encodes no suit- or flush-weakness signal. This is the primary capability target for a deeper fix.

## Response outcomes by faced band — Morana vs opponent (rounds 6+)

Each decision in which a side faces a band-Y bet is split five ways, and each side's row sums to 100% of its `n`:
- `cl` (chk & lost): the side challenged a true bet and lost.
- `cw` (chk & won): the side challenged a false bet (the claimed set did not exist) and won.
- `rl` (rai & lost): the side raised, the opponent immediately challenged, and the side lost — i.e. its own set proved false.
- `rw` (rai & won): the side raised, the opponent immediately challenged, and the side won — a credible raise the opponent wrongly called.
- `ru` (rai & unchecked): the side raised and was not immediately challenged (the opponent re-raised or the round continued). This is a proxy for the raise being respected.

**V1 vs greedy Perun:**

| band | M cl | M cw | M rl | M rw | M ru | O cl | O cw | O rl | O rw | O ru | nM/nO |
|---|---|---|---|---|---|---|---|---|---|---|---|
| high | 0.3 | 0.1 | 7.7 | 11.9 | 80.0 | 0.0 | 0.0 | 9.3 | 16.3 | 74.4 | 6172/7447 |
| pair | 11.7 | 7.0 | 19.6 | 21.1 | 40.5 | 11.0 | 9.1 | 20.8 | 16.9 | 42.2 | 16588/14647 |
| 2pair | 22.0 | 10.5 | 12.2 | 21.0 | 34.3 | 16.2 | 14.8 | 13.7 | 15.3 | 40.0 | 8259/9592 |
| straight | 20.4 | 21.9 | 18.1 | 22.7 | 16.9 | 13.0 | 11.5 | 23.0 | 24.5 | 28.0 | 22480/21593 |
| trips | 29.7 | 23.1 | 12.4 | 17.0 | 17.8 | 31.4 | 23.1 | 11.5 | 13.0 | 21.0 | 17551/20550 |
| fullhouse | 26.2 | 18.5 | 13.0 | 19.4 | 22.8 | 17.9 | 13.5 | 18.6 | 20.5 | 29.5 | 36432/36261 |
| **flush** | 33.8 | 26.9 | 17.1 | 13.7 | **8.6** | 23.6 | 14.0 | 18.5 | 16.3 | **27.6** | 20381/11471 |
| quads | 32.2 | 29.5 | 15.8 | 17.1 | 5.4 | 29.5 | 22.9 | 16.9 | 15.5 | 15.2 | 23202/23239 |
| sflush | 39.2 | 34.7 | 8.9 | 12.2 | 5.0 | 39.0 | 44.1 | 7.7 | 5.9 | 3.4 | 8233/7196 |
| great_sf | 46.2 | 38.8 | 8.1 | 6.6 | 0.2 | 52.7 | 43.5 | 2.4 | 1.3 | 0.0 | 7338/6384 |

**V1 vs humans:**

| band | M cl | M cw | M rl | M rw | M ru | O cl | O cw | O rl | O rw | O ru | nM/nO |
|---|---|---|---|---|---|---|---|---|---|---|---|
| high | 0.6 | 0.1 | 8.1 | 9.3 | 81.8 | 0.5 | 0.2 | 12.1 | 14.3 | 72.9 | 676/428 |
| pair | 12.9 | 6.9 | 17.3 | 18.0 | 44.9 | 8.3 | 7.9 | 22.5 | 16.6 | 44.7 | 757/888 |
| 2pair | 18.8 | 17.4 | 11.6 | 14.8 | 37.3 | 12.9 | 12.4 | 18.5 | 14.5 | 41.7 | 499/573 |
| straight | 14.7 | 18.4 | 14.9 | 15.6 | 36.4 | 8.7 | 7.7 | 25.1 | 26.3 | 32.2 | 1340/1241 |
| trips | 26.1 | 26.7 | 10.5 | 12.6 | 24.2 | 19.7 | 15.7 | 17.6 | 17.5 | 29.5 | 1510/1162 |
| fullhouse | 26.0 | 19.1 | 12.4 | 16.7 | 25.8 | 16.1 | 13.4 | 21.1 | 21.5 | 27.9 | 1946/2276 |
| flush | 32.9 | 25.6 | 16.2 | 11.4 | 13.9 | 22.7 | 18.0 | 20.0 | 16.3 | 23.0 | 1079/679 |
| quads | 28.5 | 34.2 | 14.6 | 14.0 | 8.6 | 30.1 | 25.8 | 16.8 | 13.5 | 13.8 | 1412/1278 |
| sflush | 36.2 | 35.8 | 11.0 | 12.7 | 4.3 | 32.7 | 36.8 | 13.4 | 11.2 | 6.0 | 464/419 |
| great_sf | 37.2 | 46.8 | 8.9 | 6.8 | 0.2 | 47.2 | 45.8 | 4.4 | 2.4 | 0.2 | 427/413 |

There is no natural 50% midpoint here, so the right comparison is Morana (M) against the opponent (O) directly. Three patterns stand out:
- **Her flush raises are less respected, at least against the best-responder.** Conditional on raising over a flush, she draws an immediate challenge about 78% of the time against Perun (`rl+rw` out of `rl+rw+ru`), versus about 56% for Perun's own flush raises — consistent with her sub-50% flush-raise credibility (`tru%` 48.7%). The raw `ru` column (raises left unchallenged, as a share of *all* flush-facing decisions) points the same way and more starkly (8.6% versus 27.6% against Perun; a milder 13.9% versus 23.0% against humans), but part of that raw gap is simply that she raises less often at flush, so the conditional rate is the cleaner read. At full-house the conditional challenge rates are roughly equal (~59% versus ~57%), so its raw `ru` gap is a raise-frequency artifact rather than reduced respect.
- **When called at flush, she loses.** `rw` is below `rl` at flush (13.7% versus 17.1%), so her called flush-raises win only about 44% of the time.
- **At flush her challenges net-lose — but from the base rate, not from mis-aimed challenging.** `cl` (33.8%) exceeds `cw` (26.9%), so more of her flush challenges lose than win. This is because the flushes she faces are genuinely true about 57% of the time (`M %T` 56.8%), not because she favours challenging true bets: her *conditional* discrimination in fact leans the right way (she checks false flushes slightly more than true ones, `chkF` 62.2% versus `chkT` 59.4%), just far too weakly to overcome the base rate. The real bind is that she cannot raise the flush credibly (`tru%` 48.7%), which leaves her challenging a band that is true more often than not. Against humans the raw `cl`/`cw` balance turns favourable at quads and great_sf (where humans' high-band bets are more often false), but flush stays unfavourable.

## Win-rate by symmetric setup (NvN) — Morana's round win-rate

Every cell is Morana's round win-rate against the given opponent in N-versus-N rounds.

| N | Morana vs Perun | Morana vs humans | n (Perun/human) |
|---|---|---|---|
| 1 | 53.2 | 51.7 | 10000/561 |
| 2 | 49.6 | 48.7 | 6329/343 |
| 3 | 50.1 | 57.4 | 5127/289 |
| 4 | 51.2 | 54.7 | 4513/245 |
| 5 | 50.6 | 55.6 | 4072/198 |
| 6 | 48.1 | 50.8 | 3627/185 |
| 7 | 49.2 | 50.6 | 3421/178 |
| 8 | 49.5 | 47.8 | 3263/184 |
| 9 | 50.6 | 52.5 | 2977/141 |
| 10 | 48.1 | 55.8 | 2756/138 |
| 11 | 48.0 | 53.8 | 2455/143 |

Against the best-responder, Morana's per-round win-rate drifts from about 53% at N=1 to about 48% at the highest N, so her abstraction edge erodes as the card count grows. Against humans she wins throughout (50–57%), though the sample is thin at high N.

## Memory and storage allocation by band (V1 table, rounds-6+ setups)

`faced%` is the share of Morana's decisions (in the human games) that respond to that band. `RAM%` is the share of all infosets with that `last_bet`, including pure-check nodes; this is the training footprint. `store%` is the share of non-pure-check infosets, which is what the deployed table keeps. 

Across the 57 setups with total ≥ 7, the table holds **152.3M infosets in RAM but only 48.3M in storage, so 104.0M (68%) are pure-check** nodes that training keeps in RAM and deployment discards.

| band | faced% | RAM% | store% | n_RAM | n_store |
|---|---|---|---|---|---|
| ROOT | 27.9 | 0.0 | 0.0 | 12,531 | 12,426 |
| high | 4.8 | 0.2 | 0.4 | 288,961 | 196,589 |
| pair | 5.4 | 1.3 | 2.5 | 1,956,483 | 1,200,124 |
| **2pair** | 3.6 | 13.7 | **21.5** | 20,928,009 | 10,364,988 |
| straight | 9.6 | 3.6 | 6.8 | 5,426,041 | 3,287,908 |
| trips | 10.8 | 1.9 | 3.4 | 2,913,525 | 1,623,541 |
| **fullhouse** | 13.9 | **46.6** | **52.3** | 71,025,569 | 25,264,489 |
| flush | 7.7 | 3.3 | 3.1 | 5,077,736 | 1,486,086 |
| quads | 10.1 | 5.5 | 3.2 | 8,380,291 | 1,562,571 |
| **sflush** | 3.3 | **15.5** | 5.5 | 23,551,549 | 2,643,963 |
| great_sf | 3.0 | 8.4 | 1.3 | 12,732,819 | 646,801 |

- **Storage** is dominated by full-house (52.3%) and two-pair (21.5%), which together take about 74% of the deployed table while accounting for only ~17.5% of faced decisions. This is driven by claim-space size: full-house has 30 claim ids and two-pair has 15.
- **RAM** shifts further toward the straight-flush ceiling: full-house is 46.6%, and sflush (15.5%) plus great_sf (8.4%) add roughly 24%. Facing a near-top bet you can usually only check, which creates large numbers of pure-check nodes; training holds these in RAM, and they make up 68% of all infosets.
- In both cases the allocation tracks the width of the betting tree rather than where the game actually happens: the frequently faced trips, quads, and flush bands each receive only about 3% of storage. This is a direct argument for an abstraction redesign that collapses the full-house and two-pair claim space along with the straight-flush pure-check tail.

## Synthesis

1. **Capability.** The abstraction is weakest at the suit-heavy bands and at high card counts. Morana over-challenges, over-bluffs, and has her raises under-respected, and her per-round edge erodes with N. The sharpest defect is flush, where the discrimination gap is 2.8, raise-credibility is 48.7%, and her flush raises draw an immediate challenge about 78% of the time against Perun (versus about 56% for Perun's own). Greedy NFSP best-responds to beat her (~48.5%), and V1 and V2.1 behave identically, so the shared abstraction is the binding constraint.
2. **Memory.** 68% of infosets are pure-check, so the RAM footprint far exceeds storage. Deployed storage is about 74% full-house and two-pair, while RAM additionally bloats on the straight-flush pure-check tail. Allocation follows claim-space width rather than play frequency.
3. **One lever addresses both.** A weakness- and suit-aware abstraction that collapses the full-house and two-pair claim space (and the straight-flush pure-check tail) would raise capability, by closing the flush hole, and reclaim memory at the same time. The per-hand macros are the cheaper interim win in the lossy regime: on the rounds-6+ setups the value + diff-truthy + bluff menu beats V2.1 by +0.0256.

## Caveats
- The opponents are not optimal, so this is a descriptive study rather than an exploitability verdict.
- Greedy Perun masks per-setup quality in the aggregate per-(N,N) figures.
- The human sample is thin in the high bands; consult `nM/nO`, and treat the human `sflush` and `great_sf` cells (n ≈ 400) and the `high`-band split as indicative only.
- The memory audit uses the V1 table together with the human faced-frequency. Perun's faced-frequency would differ slightly, but the infoset allocation on the model side is unchanged.
