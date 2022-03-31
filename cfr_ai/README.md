# Blef CFR AI

>An AI for Blef trained using Counterfactual Regret Minimisation

## Algorithm

In comparison to the original CFR algorithm, this one has several modifications.

### Monte Carlo

We're using MCCFR - that is, instead of searching through the whole game tree in every iteration (all possible cards and moves), we are using sampling to only search a part of the tree each time.

Only one agent's regrets are being updated in each run.

Our cards: sampled
Opponent's cards: sampled
Moves: vectorised for the agent being updated, sampled for the other agent

### Discounting

In CFR, the latest iteration's strategy may not be convergent towards the equilibrium, only the average one is. However, we can suspect that regret updates and strategies (and therefore entries to the strategy sum) get better in later iterations.

We are not discounting regrets (apart from imposing a minimum).

However, we are discounting the strategy sum contributions. Contributions from the first 30% of iterations are not taken into account. Later contributions are multiplied by linearly increasing discounts. Those from the 30-40% range of iterations weigh 40% as much as those from the 90-100% range, those from the 40-50% range of tierations weigh 50% as much as those from the 90-100% range, and so on.

### Action thresholding

In the final (outputted) strategy, actions with probabilities of less than 1% are reset to 0%.

### Pruning nodes

Nodes with 0 own and opponent's reach probability at the same time are never considered.

We also use regret-based pruning. Actions below a certain regret level (specified in the first `pruning_range` CLI argument) are only considered every 20th iteration (all in the same cycle). The second `pruning_range` CLI argument is the minimum regret.

### Appendix: modifications considered but not used

We have not implemented ICFR because of the expected effort/benefit ratio.

We are not using variance-reduction techniques with respect to opponent's sampled actions, because of no noticeable benefit when trying them. This may be due to the fact that most variance comes from the sampling of cards, not opponent's actions.

## Information set abstraction

We have memory, storage and computation constraints that force us to abstract information sets. Unabstracted, they would contain an list of our cards and the list of all moves that have been played in the given round.

We train the AI separately for each sorted array of players' hand sizes (e.g. 3 cards vs 5 cards). There are 66 of those, which we call 'scenarios'.

When a player has 11 cards, there are 10^6.7 possible hands (though only 10^5.7 strategically-distint ones). When we consider just the last three elements from the history, we have 10^6.0 combinations.

As it's difficult to parallelise this algorithm, the abstraction is designed to use, for each of the 66 scenarios, 4 GB of RAM and a single core over a number of days. The final strategy should fit in an 50 MB zip file, to deploy on AWS Lambda. 

### Memory constraint

Each information set object stores a regret array of up to 88 doubles and a strategy sum array of up to 88 doubles. Ideally it would also store an array of possible actions, but we're computing it on each visit to the information set because it's computaitonally cheap and saves a lot fo memory.

We could do with single- instead of double-precision floats, but they compute slower and we seem to be more constrained on computation power.

An average information set will only have 30-40 possible actions. 