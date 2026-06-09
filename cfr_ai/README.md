# Blef CFR AI

>An AI for Blef trained using Counterfactual Regret Minimisation

## Introduction

This AI is made using Counterfactual Regret Minimisation (CFR), an algorithm designed to solve fixed-sum imperfect-information games and the leading algorithm for building Poker AIs.

The repository contains all code needed to train, evaluate and deploy the AI. 

## Algorithm

 CFR is an iterative algorithm used to find the Nash equilibrium strategy, which, in the two-player version of the game, is the least exploitable strategy (i.e. when playing against a perfect opponent who knows our strategy, it will deliver the best expected value).

Many additions and modifications to the original CFR algorithm have been developed over the many years of its existence. We will now describe which ones we incorporated.

### Monte Carlo

We're using MCCFR - that is, instead of searching through the whole game tree in every iteration (all possible cards and moves), we are using sampling to only search a part of the tree each time. Specifically, we're using external-sampling MCCFR:

* Only one agent's regrets are being updated in each run
* The cards of the traverser are being sampled
* Opponent's cards are being sampled as well
* The traverser explores all its moves (except for regret-based pruning, which we describe below)
* The opponents' moves are being sampled according to their current strategy

### Regret-based pruning and minimum regret

Due to our implementation of the external-sampling MCCFR, nodes with 0 opponent's reach probability are never considered.

We also use regret-based pruning. Actions below a certain regret level (specified in the first `pruning-range` CLI argument) are only considered every 20th iteration (all in the same cycle). 

There is also a minimum regret, set by the second `pruning-range` CLI argument.

### Initial strategy

The initial strategy in every set is to take the last action with 100% probability (so in all situations apart from the start of the round, it will be a check). This is because a check with 100% probability is the best strategy in most infosets in the game and makes the initial iterations fast. It would be possible to start each infoset with a more sophisticated strategy, for example by engaging the next-best AI to compute it. However, it is impractical, as we need to make as many iterations (or tighten the abstraction so much) that the algorithm figures out a reasonable strategy well within the first 30% of iterations anyway.

### Strategy sum discounting

In CFR, the latest iteration's strategy may not be convergent towards the equilibrium, only the average strategy is. However, we can suspect that regret updates and strategies (and therefore entries to the strategy sum) get better in later iterations.

We are not discounting regrets (apart from imposing a minimum).

However, we are discounting the strategy sum contributions. Contributions from the first 30% of iterations are not taken into account. Later contributions are multiplied by linearly increasing discounts. For example: 
* Contributions from the 30-40% range of iterations weigh 40% as much as those from the 90-100% range
* Contributions from the 40-50% range of tierations weigh 50% as much as those from the 90-100% range

### Action thresholding

In the final (outputted) strategy, actions with probabilities of less than 1% are reset to 0%. This is primarily to save storage space.

### Redundant information sets

In most setups in Blef, most information sets will never be reached by good players, and if they are encountered, checking 100% of the time is the best strategy in them. As a consequence:
* to save on memory, we are only initialising information set objects when they are actually reached; 
* the first strategy (for informations sets with 0 regrets) is checking 100% of the time; and
* we do not record the strategy for information sets where the final strategy is to check 100% of the time.

We are then instructing the agent who uses the CFR strategy to check with 100% probability if it cannot find the strategy for a given information set during online play.

### Note: modifications considered

We have considered:
* ICFR — not implemented due to the expected effort/benefit ratio;
* variance-reduction techniques on opponent's sampled actions or cards — tried, no noticeable benefit;
* DCFR / CFR+ regret-matching variants — tried; on rounds 1-3 they converged to the same exploitability as `es` (the hand+history abstraction was the floor) but took 10-18× longer wall-clock because pruning has to be disabled and the α-discount adds per-visit cost. Strictly worse than `es` on a resource-adjusted basis, so removed from the codebase; and
* outcome-sampling MCCFR — implemented but **not** competitive with external sampling on Blef's structure. Removed from the codebase; see `README_Appendix_B.md` for the diagnosis.

## Resource limits and abstraction

Memory, storage and computation constraints force us to heavily abstract information sets. Unabstracted, they would contain an list of our cards and the list of all moves that have been played in the given round.

We train the AI separately for each sorted array of players' hand sizes (e.g. 3 cards vs 5 cards). There are 66 of those, which we call 'setups'.

When a player has 11 cards, there are 10^6.7 possible hands. When we consider just the last three elements from the history, we have 10^6.0 combinations.

One cannot compute this algorithm with 10^11.7 information sets. 

Instead, our abstraction is designed to have a limit of around 5 million (10^6.7) infosets. This has the following benefits: 
* it limits the memory consumption to around 4 GB;
* it keeps each setup's served strategy small enough that all 66 fit comfortably inside a single Lambda container image; and
* each setup's strategy can be calibrated to a reasonable extent in around 0.5 core-days.

The AI should then take around a core-month with 4GB of memory to train, which costs in the order of 30 USD when trained on on-demand AWS EC2 instances. It can also be reasonably trained on a personal machine.

### Memory consumption

Training state is held in flat 2D numpy arrays indexed by infoset row: one for regrets and one for strategy sums, each row covering that infoset's legal action range. These arrays default to single-precision (`--dtype fp32`).

An average information set will only have 30-40 possible actions. Therefore, in setups where a wide range of bets are viable and risky (not just e.g. the first 12), the average infoset weight in memory will be 1-2 kB.

### Penalty

We have a mechanism for imposing a penalty (reduction of payoff) for the traversing player for betting instead of checking. Used as a last resort to make the compute and memory requirements manageable

### History abstraction

History is abstracted in two steps. 

First, we only consider last 3 bets.

Then, the second and the third previous bet are being independently compressed according to the same rules, which take the last bet into account:
* If the last bet is a high card or a pair, they are recalled exactly
* If the last bet is a two pair:
    * Two pairs and pairs are recalled exactly
    * High cards are remembered exactly if they're relevant or higher than the higher value of the current two pair bet. Other high cards are merged into a 'Z'
* If the last bet is a straight:
    * Two pairs and pairs are recalled exactly
    * High cards are merged into one
* If the last bet is a three-of-a-kind:
    * Straights are recalled exactly
    * The relevant two pairs are recalled exactly, otherwise as 'A'
    * The relevant pair is recalled exactly, the rest as a 'B'
    * The relevant high card is recalled exactly, the rest as a 'Z'
* If the last bet is a full house:
    * Other full houses are categorised by whether they contain the same primary value, the same secondary value, the current first value as their second value, or otherwise denoted by their primary value
    * The two relevant three-of-a-kinds are recalled exactly, the rest as 'E'
    * Straights are recalled as an 'F'
    * Two pairs are categoried by whether they contain the primary value, the secondary value, both or neither
    * Pairs are recalled exactly if they contain on of the two values, otherwise as a 'J' 
    * High cards are merged into a 'Z'
* If the last bet is a flush:
    * Other flushes are recalled exactly
    * Full houses and represented by their primary value and the same as the relevant three-of-a-kind (e.g. all Aces over ... and three-of-a-kind Ace are all represented as 'A5')
    * Straights are recalled exactly
    * Two pairs, pairs and high cards are merged into a 'Z'
* If the last bet is a four-of-a-kind:
    * Other four-of-a-kinds and flushes are recalled exactly
    * The full houses with the relevant primary value are an 'A', the immediate next full house is a 'B', the other full houses are a 'C'
    * The relevant three-of-a-kind is a 'D', others a 'Z'
    * Relevant two pairs are an 'E', others a 'Z'
    * The relevant pair is an 'F', others a 'Z'
    * High cards are merged into a 'Z'
* If the last bet is a straight flush:
    * Other straight flushes, all flushes and four-of-a-kind of 9 or Ace are remembered exactly
    * Other four-of-a-kinds are recalled as an 'A'
    * All full houses are recalled as a 'B'
    * All other bets are recalled as a 'Z'

This abstraction tries not to differentiate between bets that are less relevant or very junior to the last one. It's stored in `history.csv`. For example, if the last bet was on set 55 (Full house, Queens over Aces), a bet on set 5 (High Card, Ace) is represented in the history abstraction as a Z (row 55 column 5 in `history.csv`). 

### Hand abstraction

The AI does not consider the exact cards in its hand. Instead, it uses a hand-crafted abstraction that extracts only the most strategically relevant features of the hand, which changes depending on the round and the last bet made. It's encoded in `get_hand_abstraction` in `information_set.py`.

While the players hold few cards between them (total hand size ≤ 7), the AI only looks at card values and completely ignores the suits.

In the later rounds, there is a more complex hand-crafted abstraction. First, as the AI, we extract features:
* we count the number of cards of each value and each suit it holds, for a total of 10 integers.
* we identify the strongest features of our hand among these 10 integers. N cards of a given value are considered stronger than N+1 cards of a given suit, but weaker than N+2 cards of a given suit (so that four-of-a-kind on hand is considered stronger than a flush on hand). There is also an augmented version of these strengths, where each suit strength also contains information about whether we have the 9 and whether we have the Ace of that suit. 

We then pick these numbers as identifiers of our hand, depending on the last bet (the one we are responding to):

* For last best of a high card, pair or two pair or at the beginning of the round:
    * If we have a flush or four of a kind on hand, only that feature is reported
    * Else if we have a three of a kind on hand, the top 2 value-related strengths are considered
    * Else, we look at card values and ignore suits
* For the last best of any straight:
    * If we have a flush or four of a kind on hand, only that feature is reported
    * Else, report if we have a 9, how many distinct values between 10 and King we have, if we have an Ace, and what our top strength is 
* For the last best of a three-of-a-kind:
    * If we have a flush or four of a kind on hand, only that feature is reported
    * Else, report how many of the value being bet on we have, and what our top strength is 
* For the last best of a full house:
    * If we have a flush or four of a kind on hand, only that feature is reported
    * Else, report how many of the two values being bet on we have, and what our top strength is 
* For the last best of a flush, report how many of the suit being bet on we have, and what our top augmented strength is 
* For the last best of a four-of-a-kind, report how many of the value being bet on we have, and what our top augmented strength is, but ignore strenghts related to values lower than the one being bet on
* For the last best of a small/big straight flush, report augmented information about the suit being bet on and our strongest suit 
* For the last best of a great straight flush, report augmented information about the suit being bet on and our strongest suit among those that can still be bet on

At the largest hand sizes (total ≥ 17 cards), the round-start token is additionally refined by the player's own per-suit card counts, which sharpens play in the suit-heavy endgame.

### Action abstraction

We have a mechanism for the AI to not acknowledge or make a specific number of the lowest bets (e.g. all high cards). If it encounters one of those bets during online play, it acts as if the round just started.

With 14 cards, any specific high card has 98% chance of existing (97% and 99% with 13 and 15 cards respectively).

With 16 cards on the table, the great straight has 96% chance of existing (88%, 93%, 98% and 99% for 14, 15, 17 and 18 cards respectively). The production models therefore raise this floor as the table fills: `min_bet` is 0 below 13 total cards, rises to 27 (small straight) from 13 cards onward, and is set to the top full house (65) for the 11-v-11 round, where almost every lower claim is trivially true. Discarding near-certain low bets apart from the last one cuts training time, memory and storage while *sharpening* the meaningful play.

### Augmenting action macros

From the mid-game onward (total hand size ≥ 8, where the suit-aware abstraction kicks in), each infoset's action menu is augmented with three *macro* actions — `value`, `difftruthy`, and `bluff`. Unlike a concrete bet, a macro does not name a fixed claim; it resolves at decision time to a specific bet drawn from the legal range by its own scoring rule.

If we define `p` as the vector of hand-aware probabilities and `g` as the vector of hand-unaware probabilities, then:

* `value` is `argmax(p)` 
* `difftruthy` is `argmax(p-g)` 
* `bluff` is `argmin(p-g)` 

This lets a single abstracted infoset express hand-dependent aggression. Each macro's probability mass is stored alongside the concrete-action probabilities and folded onto the resolved bet when the agent serves the strategy.

### Strategy storage format

Every infoset is keyed by a composite int64 encoding hand size, last bet, two history-code ids, and an abstraction id. Only non-checking infosets are written — a missing key at lookup time is interpreted as "check 100%".

Two on-disk layouts are supported, both produced from the same training run and both loaded by `strategy_io.load_strategy_for_agent`:

1. **Compressed `strategy.npz`** — what `training.py` writes; the resting format under `cfr_ai/outputs/<setup>/`. A single compressed numpy archive containing `keys` (int64, sorted), `lower`/`upper` action bounds (int16), a padded float32 `[N, 89]` probability table, and `min_bet`. Read eagerly into RAM. Convenient for local development.
2. **Sparse mmap layout** — what `scripts/stage_for_docker.py` produces for the Lambda image. Every large per-row array is stored as its own uncompressed `.npy` so the agent can `mmap` it and page it in lazily, with no eager decompression at load. The per-setup directory contains:
   * `meta_keys.npy` (int64, sorted) — the composite infoset keys.
   * `meta_offset.npy` (int32) — length-`N+1` CSR-style prefix sum into the sparse arrays.
   * `meta_lower.npy` / `meta_upper.npy` (int16) — per-row legal-action bounds.
   * `meta_masses.npy` (uint16 `[N, k]`, macro setups only) — each infoset's macro masses, on the same scale as its concrete probabilities.
   * `probs_sparse_indices.npy` (uint8) — non-zero positions within each row's legal-action slice.
   * `probs_sparse_values.npy` (uint16) — non-zero probability values, quantised with scale `1/65535`.
   * `strategy_meta.npz` (tiny, compressed) — just the scalars: format version, `min_bet`, and the macro `kinds`.

   The `.npy` files are `mmap`'d at load time, so peak resident memory per loaded strategy is a few tens of MB regardless of `N` (strategies are typically 5-10% dense after `clear_lows`). An older layout that packed `keys`/`lower`/`upper`/`probs_offset`/`masses` into the compressed `strategy_meta.npz` and read them eagerly is still recognised by the loader, so images staged before this change continue to load unchanged.

The agent loader returns a `FlatStrategyAgent` regardless of which layout is on disk: lookup is always `np.searchsorted` on the sorted keys, and `get_strategy(row)` reconstructs the dense legal-action slice from whichever representation was loaded. The loader does not import numba.

Files written per setup under `cfr_ai/outputs/<setup>/`:

* `strategy.npz` — the compressed layout described above.
* `strategy.abs.json` — abstraction-string → id table, also covering diagnostic-only abstractions so they remain interpretable.
* `diagnostic.npz` (optional) — touch counters + raw `regrets` + raw `strategy_sum` for every infoset, including check-only ones. Used by `analysis/` scripts, not by the deployed agent.
* `metadata.csv` — human-readable training params (iterations, penalty, duration, game values, utility log).

## History abstraction and convergence

Because we only remember the last 3 bets (and imperfectly, too), a node in the game 'tree' might have more than 1 parent, which makes it not a tree. 

Some nodes may have a very large number of possible paths from the root, which might be very similar from a strategic perspective, but will result in multiple updates to a node in an iteration. This is a problem for performance and/or convergence properties of the algorithm.

We have evaluated three ways of dealing with that problem:
* doing nothing, which means that a node will potentially get a large number of updates in an iteration. This will result in potentially much quicker changes of regret between the pruning threshold or the minimum value and 0. It will also result in much slower iterations
* caching the last value of a node and recording the last iteration it was touched, and if it's the second time we're touching it in a given iteration, returning the cached value instead of (1) getting the strategy, (2) traversing its children, (3) updating cumulative regrets and (4) updating strategy sum again . This is the 'temporary value' solution. However, that biases the regret updates. For example, if a node has 200 possible paths leading to it, and each of them has 1% opponent reach probability when opponent has hand X and each of them has 2% opponent reach probability when opponent has hand Y, this will result in the node having a near 100% chance of geting one full-sized regret update in the iteration conditional on hand X as well as Y. This means that the CFR strategy will be prepared for a distribution of hands with higher entropy than the one it should be.
* using the 'temporary values', but every point in history that was either (A) visited or (B) considered by the opponent but not visited (because of the external sampling algorithm) will be marked as having been last considered in this iteration. Downstream nodes are not marked as considered. When (and if) that node is finally visited in this iteration, its value is calculated and (some of) its downstream nodes are visited. However, if the node was not visited the first time it was considered in this iteration, it will not get regret updates or strategy sum updates. Thanks to this, regret updates will be lower the lower the opponent's chance of making the last move (on the first path from which we considered this node) that reaches this node. This is the 'considered nodes' solution.

Although we expected the 'temporary value' solution to yield the worst results, for the setups we tested, we found the two otehr solutions to not improve the strategy at the same number of training iterations. These other solutions are typically 2-5 time slower than the temporary value solution. 

That is why we have left the temporary solution as the only one available to use. However, to enable all three, follow the steps in `README_Appendix_A.md.

## Usage

Training is done per setup, where a setup is the ordered number of cards per player. To train the 1 card vs 1 card setup, run this from the project root:

```
python -m cfr_ai.training --hand-sizes 1 1
```

CLI flags:

* `--hand-sizes` (required) — number of cards per player.
* `--iter` (default: 5,000,000) — Monte Carlo iterations.
* `--penalty` (default: 0.0) — penalty for non-checking moves (see Penalty above).
* `--dtype` (default: `fp32`) — regret-array dtype; `fp32` halves memory at no measurable quality cost on production setups.
* `--min-bet` (default: 0) — minimum bet the AI will make or acknowledge.
* `--capacity` (default: 64,000) — initial row capacity; auto-grows by chunks as needed, so the default is fine.
* `--seed` (default: 42) — RNG seed for deals + opponent sampling.
* `--log-points` (default: 20) — number of evenly-spaced iter-rate / utility log lines.
* `--archive-tag X` — save into `cfr_ai/archive/X/outputs/<setup>/` (snapshot for `head_to_head.py`) instead of the production `cfr_ai/outputs/<setup>/`.
* `--high-priority` — bump the process to a higher OS priority (Windows: `HIGH_PRIORITY_CLASS`; POSIX: `nice -5`). Useful for shared machines.

You will see a `tqdm` progress bar during training.

### Training outputs

A training without the `--no-save` flag writes its strategy under `cfr_ai/outputs/<setup>/` — see [Strategy storage format](#strategy-storage-format) above for the file layout (`strategy.npz`, `strategy.abs.json`, `diagnostic.npz`, `metadata.csv`).

The deployment file `strategy.npz` contains only non-checking infosets — at lookup time, a missing key is interpreted as "check 100%". The companion `diagnostic.npz` (always written by `cfr_ai/training.py` alongside `strategy.npz`) contains *every* infoset, including check-only ones, plus the raw regrets, raw `strategy_sum`, and the iteration counters (`first_touched`, `last_touched`, `times_touched`) used by the analysis scripts in `analysis/`.

## Evaluation & analytics

Evaluation is key to informed development of the algorithm. The canonical metric for CFR is **exploitability** — how much an optimal opponent can win against the trained agent. Computing the exact exploitability (full best response) is tractable for the first few rounds but explodes in cost beyond ~round 4, so we use a more scalable proxy.

Along each setup's outputs, there's a training metadata file (`metadata.csv`), which notes:

* the time the training finished;
* training duration (Hours:Minutes);
* the number of iterations;
* the pruning threshold;
* the minimum regret;
* the penalty (for betting instead of checking);
* the number of nodes touched (incremented at maximum once per node per iteration);
* the number of explored infosets;
* the number of infosets in which the strategy is not a check with 100% chance;
* the amount of RAM taken by the training Python process (including the memory claimed by the code that saves the strategies);
* the game value of each player (e.g. if we're training the 2 cards vs 3 cards case, it's 1. the game value for the starting player when the 2-card player is starting and 2. the game value for the starting player when the 3-card player is starting); and
* the log of utilities along the training run.

### LBR-based exploitability (`lbr.py`)

The exploitability calculation is decoupled from training:

```
python -m cfr_ai.lbr --hand-sizes 3 3 --depth 2 --update-summary
```

Key flags:

* `--depth K` — LBR-K (number of LBR-optimised decisions per game before falling back to CFR-vs-CFR rollout). K=1 is the classic [Lisý-Bowling local best response](https://arxiv.org/abs/1612.07547). Default is `INF_DEPTH = 10^6` (effectively infinity), which means LBR plays optimally all the way to terminal — equivalent to **exact best response** when combined with `--n-belief-samples` and `--n-lbr-hand-samples` large enough to enumerate (CLI labels this as `inf (= BR)`). The JIT recursion structurally supports any K; the practical limit is the combinatorial blow-up of ~88^K per LBR-active node, so anything past K=2-3 is infeasible on round-4+ setups.
* `--n-belief-samples N` (default: 300) — sample N opponent hands without replacement instead of enumerating the whole posterior. Default works on all setups.
* `--n-lbr-hand-samples K` (default: 500) — sample K LBR hands without replacement instead of enumerating all C(24, h) of them. Unbiased; adds variance.
* `--starting-player {0,1}` — fix the starting player; default runs both seats for asymmetric setups.
* `--update-summary` — append/update this setup's LBR cells in `outputs/summary_of_all_runs.csv` (unified training + LBR summary). LBR rows are auto-cleared on a subsequent training run for that setup, so a populated LBR cell always corresponds to the current trained policy.

When opponent hands are enumerated, the result is the exact LBR-K value. When they are sampled, we use a **double-sampling** scheme: an independent belief sample S2 is used to value the action that another sample S1 chose. This produces a conservative *lower bound* on LBR-K — sometimes the chosen action is suboptimal, but its EV is computed without winner's-curse bias. The standard error reported in the summary combines lbr_hand sampling variance and opp belief sampling variance, with a chi-squared upper bound on the std err itself.

LBR-1 captures roughly 85% of full BR on our verified shallow setups; LBR-2 captures ~99% but takes 4× longer per setup, and quickly becomes infeasible past round 3.

### `summary_of_all_runs.csv`

`outputs/summary_of_all_runs.csv` is the unified per-setup summary. One row per setup; training and LBR write disjoint column sets, each writer fully owning its cols.

Training cols (written by `training.py` on save):
* **Setup**, **Finished**, **Iterations**, **Penalty**, **Min bet**, **Pruning threshold**, **Minimum regret**, **Duration**, **Nodes touched**, **Explored infosets**, **Non-checking infosets**, **RAM (MB)**, **P0 value**, **P1 value**, **Version**.

LBR cols (written by `lbr.py --update-summary`):
* **LBR-K expl** / **LBR-K duration** / **LBR-K sampling** — one triple per depth K evaluated. *expl* is the point estimate (and `+/- std_err_worst_case` if sampled; ASCII `+/-`, not `±`, so the CSV stays mojibake-free in Excel/cp1252 tools); asymmetric setups join two starting-player results with `|`. Expl is a percentage of game value, duration is seconds.
* **LBR-K sampling** — the caps used as `(lbr, opp)`, e.g. `(300, 500)` (populations under the cap are enumerated, else sampled without replacement). It is **per-depth** because a cheap LBR-1 can enumerate fully where LBR-2/inf must subsample, so a single shared value would mis-describe the others.

A training save **blanks that row's LBR cols**, so a populated LBR cell always corresponds to the current trained policy. The LBR depth-triple columns grow rightward as more depths get computed.

When LBR runs on a setup with no training row (e.g. an LBR'd archive snapshot), the row is created with empty training cols.

### Utility logging

To get an approximate idea of whether we are running enough iterations, we are logging utility at equal intervals across the training run. If there is no substantial trend beyond the first 30% of iterations, then the exploitability coming from insufficient iterations is likely to be low (however, exploitability coming from the abstraction may still be high).

### Periodic exploitability (optional)

Pass `--get-exploitability` to `cfr_ai.training` to compute LBR-K at several evenly-spaced points during training. Each measurement is appended to the setup's `metadata.csv` under a `--- Exploitability Log ---` block as e.g. `LBR-1 expl sp=0 at Iter 4000000, +0.083%`. The utility log gives a stability signal; this gives an actual exploitability trajectory — far more reliable for deciding "how many iterations does this setup need". Flags:

* `--exploitability-points N` (default 5) — number of snapshots, evenly spaced across the run.
* `--exploitability-depth K` (default 1) — LBR-K used at each snapshot. K=1 is fast and a good convergence proxy.
* `--exploitability-n-belief 300`, `--exploitability-n-lbr-hand 500` — sampling caps (matching the production LBR defaults).

Cost is modest: a few LBR-1 calls per training run. Off by default to keep the production training command fast.

`analysis/training_analytics.py` is an admin tool for **rebuilding** the unified summary from the per-setup `metadata.csv` files and **regenerating** the per-setup utility chart PNGs. metadata.csv is the source of truth: the rebuild reconstructs both training and LBR cols from each file's FINAL `--- LBR Exploitability ---` block (never the in-training `--- Exploitability Log ---`), so a setup's exploitability always matches its training run (a retrain rewrites metadata.csv, dropping the stale block until LBR is re-run). Day-to-day, `training.py` and `lbr.py` maintain the summary incrementally — this script is for migrations or recovering from a corrupted summary file.

```
python -m cfr_ai.analysis.training_analytics             # rebuild CSV + regenerate charts
python -m cfr_ai.analysis.training_analytics --no-charts # only rebuild CSV
python -m cfr_ai.analysis.training_analytics --only-charts # only regenerate charts
```

### Head-to-head comparison

`analysis/head_to_head.py` compares two versions of a strategy for a **single setup** by replaying that setup's deals against each other (Monte Carlo; ~10 minutes for 10,000 deals on most setups), reporting a per-seat advantage. It accepts explicit folder paths (`--model1-folder`, `--model2-folder`) or archive-tag shortcuts (`--model1 <tag>`, `--model2 <tag>`) resolving to `cfr_ai/archive/<tag>/`; the tag `current` (or `.`) refers to the working `cfr_ai/` tree.

Its scope is **non-macro setups** — value-only setups (total ≤ 7) and pre-V3 / V2.1 models. It reads only the concrete probability slice, which is sub-stochastic for an augmenting-action (V3+) model, so it **refuses macro setups** rather than silently comparing a policy the agent never plays. For the current macro methodology the primary version-vs-version signals are the whole-game harnesses — `cfr_vs_cfr_games.py` (CFR-vs-CFR) and `cfr_vs_nfsp_games.py` (CFR-vs-Perun) — and, per setup, `selftest_macros.py` / `lbr_macro.py`, all of which fold the macro masses through the real serving path. `head_to_head.py` nonetheless remains the shared foundation those macro tools build on: its key and legal-action helpers are imported by `selftest_macros.py` and `selftest_bluff.py`, and it is the consumer of the archive-tag snapshots produced by `archive_tool.py`.

### CFR-vs-NFSP benchmark (`analysis/cfr_vs_nfsp.py`)

Benchmarks a CFR model version against the deployed NFSP agent (the 1v1
deck-24 specialist), broken down by **(starting-player hand size,
non-starting-player hand size)** with alternating agent roles — so each
cell's win-rate reflects agent skill at that configuration, not the
first-mover advantage. Use it to see *where* a CFR version beats NFSP, not
just the aggregate. Both agents play through the real Blef engine via the
same `determine_action(game_state)` interface.

```
python -m cfr_ai.analysis.cfr_vs_nfsp --cfr current                 # auto: setups this version has trained
python -m cfr_ai.analysis.cfr_vs_nfsp --cfr current --setups all    # full 11x11 matrix (+ --heatmap for a PNG)
python -m cfr_ai.analysis.cfr_vs_nfsp --cfr-folder cfr_ai/experiments/prune-10 --setups "5,7 6,6"
```

CFR version selection mirrors `head_to_head.py` (`--cfr <tag>` →
`cfr_ai/archive/<tag>/outputs`, `--cfr-folder <path>` → `<path>/outputs`,
`current` → working tree). NFSP is the fixed opponent, playing its average
policy (`--nfsp-greedy` for argmax). Multiprocess (`--workers`); each worker
holds one CFR strategy, so keep `--workers` modest when RAM-bound.

Prerequisites (not in git): **PyTorch**, and the NFSP artifacts under
`nfsp_ai/artifacts/` (`nfsp_inference_24_1v1.pt` + the deck-24
card/history embeddings) — extracted from the `blef-nfsp-lambda` ECR image.
Result CSVs / heatmaps are written to `cfr_ai/analysis/cfr_vs_nfsp_<label>.*`
and are gitignored.

### Strategy archive

Trained strategies can be snapshotted into versioned tags for later head-to-head comparison. Run from the project root:

```
python -m cfr_ai.archive_tool --tag v0 --note "Pre-Hetzner-retrain baseline"
```

By default this archives every setup currently in `outputs/` and skips diagnostics (pass `--include-diagnostics` to include them). Each archive is a self-contained model folder containing snapshots of `information_set.py`, `history.csv`, and the relevant `outputs/<setup>/` subtrees (the `strategy.npz` / `strategy.abs.json` per setup, optionally `diagnostic.npz`), directly consumable by `analysis/head_to_head.py` via the tag shortcut. `cfr_ai/archive/` is gitignored — archives are local to each machine.

### Winning probabilities

Using the game values noted down for each setup in the `summary_of_all_runs.csv`, we can compute the probabilities of winning the game starting from a specific setup and with a specific starting player within that setup. To compute those, use the `analysis/win_probabilities.py` script.

## Deployment

The AI is deployed as a single Lambda function backed by a container image stored in Amazon ECR. The image bundles `cfr_ai/` (code) and the 66 per-setup strategy payloads in the sparse-mmap layout described in [Strategy storage format](#strategy-storage-format).

`cfr_ai/agent.py:_get_strategy(hand_sizes)` loads the relevant strategy lazily on first use per warm container, caches it as a `FlatStrategyAgent` (defined in `strategy_io.py`), then resolves every subsequent call by `np.searchsorted` on the sorted-keys array. No numba on the agent path.

`cfr_ai/scripts/stage_for_docker.py` converts each setup's local compressed `strategy.npz` into the sparse-mmap layout at Docker-build time only — the source tree stays compact.

**Cache policy**: size = 1. Game state progresses linearly through (hand_size_a, hand_size_b) configurations as cards are won/lost; the previous setup is unlikely to be useful again before the next one displaces it. `agent.py` evicts the current strategy BEFORE loading the new one, so peak memory through a setup transition is exactly one loaded strategy plus the base runtime.

Files involved:
* `cfr_ai/lambda_function.py` — Lambda entry point. Parses the game event, calls `agent.determine_action`, invokes `blef-play` asynchronously.
* `cfr_ai/deployment/Dockerfile.lambda` — `public.ecr.aws/lambda/python:3.12` base + `cfr_ai/` + `lambda_function.py`.
* `cfr_ai/deployment/requirements.txt` — `numpy` only.
* `cfr_ai/scripts/stage_for_docker.py` — assembles the build context (excludes `analysis/`, `archive/`, `deployment/`, `__pycache__`, `diagnostic.npz`, tracking CSVs, visualisation PNGs, `_bench_formats/`) and converts strategies to the sparse-mmap layout. Cross-platform; no rsync needed.
* `cfr_ai/scripts/deploy_lambda.sh` — orchestrates build → ECR login → ECR push → `update-function-code`. Idempotently creates the ECR repo. Skip the Lambda update with `SKIP_LAMBDA_UPDATE=1`.
* `cfr_ai/scripts/create_lambda.sh` — first-deploy only; `aws lambda create-function` with the right architecture / memory / timeout. Subsequent updates use `deploy_lambda.sh`.

### First-time setup

```bash
export ACCT=<account-id> REGION=<region> PROFILE=<aws-cli-profile>
export ROLE_ARN=<execution-role-arn>
# Build + push to ECR (creates the repo if missing):
SKIP_LAMBDA_UPDATE=1 bash cfr_ai/scripts/deploy_lambda.sh
# Create the function from the pushed image:
bash cfr_ai/scripts/create_lambda.sh
```

The ECR repo needs a resource policy granting `lambda.amazonaws.com` permission to pull (`ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer`) for first-time function creation in the account. This is set once per repo by an admin via the ECR Console or `aws ecr set-repository-policy`.

### Subsequent deploys

```bash
export ACCT=<account-id> REGION=<region> PROFILE=<aws-cli-profile>
bash cfr_ai/scripts/deploy_lambda.sh
```

### Runtime characteristics

Lambda is sized at **1024 MB** — chosen for the vCPU that tier provides, not the resident set, which is far smaller. Cold start total wall is ~3 s on the first invocation (dominated by the image pull) and ~0.7 s thereafter. A decision on an already-loaded setup takes ~30-260 ms; the first decision on a *new* setup costs ~1-2 s, because that setup's `.npy` arrays must be paged in from the container's read-only filesystem — an I/O cost, not computation. Peak resident memory is ~166 MB on the biggest setup, comfortably under the cap.

### Architecture choice

The image is built for `linux/arm64`. On an x86 host this means QEMU emulation during `docker build`, which slows the build but not the runtime.

## Performance

Training (`trainer.py`) and exploitability (`lbr.py`) are JIT-compiled with [numba](https://numba.pydata.org/). They use a composite int64 infoset key `[abs_id | h_m2 | h_m1 | last_bet | hand_size]` (replacing per-call string concatenation), store regrets and strategies in flat 2D numpy arrays indexed by row, and recurse inside a single `@njit` function. The Python orchestration around the JIT-ed core is kept minimal (deal cards, build per-iter abstraction-id lookup, drive the iteration loop). Memory is held in fp32 by default.

A previous implementation used a dict of `InformationSet` objects. The table below shows the speedups achieved when we moved to JIT:

| Workload | Setup | Old Python | Current JIT | Speedup |
|----------|-------|-----------:|------------:|--------:|
| Training 5M iter | (1,2) | ~63 min† | 6.4 min | **~10×** |
| Training 5M iter | (3,7) | 10h 3min | 1h 46 min | **~6x** |
| LBR-1 | (1,3) | 14.7 s | 0.6 s | **24×** |
| LBR-2 | (1,3) | 152 s | 3.8 s | **40×** |
| LBR-2 | (3,3) | 519 s | 34 s | **15×** |

†(1,2) old-Python time is extrapolated from the measured 5M-iter steady-state rate (~1,320 it/s). All JIT rows are direct wall-clock measurements.

However, an earlier round of numba experimentation showed that using a dict of `InformationSet` and JIT-ing the inner math regresses performance, as each per-node `info_set.regrets`/`info_set.strategy_sum` access crossed the JIT-Python boundary. The current implementation flattens the entire trainer state into typed numpy arrays + a `numba.typed.Dict[int64, int64]` index, so the JIT-ed recursion never touches Python objects.

`precompute_set_existence` is called once per training iteration to evaluate the truth of all 88 possible bets against the dealt hands. It has been optimised to build value-count and (value, suit)-presence tables in one pass over the deal, then resolve every bet via Python int comparisons. The whole call costs ~15 µs regardless of hand size — small fraction of the per-iteration cost.

## Other notes

We thank [Thomas Trenner](https://github.com/tt293) for his writings on the CFR algorithm, which inspired us to create this AI.

Strategy outputs for every setup are avaiable upon request.
