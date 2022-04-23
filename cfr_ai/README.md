# Blef CFR AI

>An AI for Blef trained using Counterfactual Regret Minimisation

## Algorithm

This AI is made using Counterfactual Regret Minimisation (CFR), an algorithm designed to solve fixed-sum imperfect-information games and the most accomplished algorithm for building Poker AIs. It is meant to find the Nash equilibrium strategy, which, in the two-player version of the game, is the least explotable strategy (i.e. when playing against a perfect opponent who knows our strategy, it will deliver the best expected value).

Many additions and modifications to the CFR algorithm have been developed over the many years of its existence. We will now describe the particular version we are using.

### Monte Carlo

We're using MCCFR - that is, instead of searching through the whole game tree in every iteration (all possible cards and moves), we are using sampling to only search a part of the tree each time. Specifically, we're using external-sampling MCCFR.

Only one agent's regrets are being updated in each run.
The cards of the updating agent are being sampled.
Opponent's cards are being sampled, too.
The updating agent explores all its moves (except for regret-based pruning, which we describe below).
The opponents' moves are being sampled according to their current strategy.

### Discounting

In CFR, the latest iteration's strategy may not be convergent towards the equilibrium, only the average strategy is. However, we can suspect that regret updates and strategies (and therefore entries to the strategy sum) get better in later iterations.

We are not discounting regrets (apart from imposing a minimum).

However, we are discounting the strategy sum contributions. Contributions from the first 30% of iterations are not taken into account. Later contributions are multiplied by linearly increasing discounts. Those from the 30-40% range of iterations weigh 40% as much as those from the 90-100% range, those from the 40-50% range of tierations weigh 50% as much as those from the 90-100% range, and so on.

### Action thresholding

In the final (outputted) strategy, actions with probabilities of less than 1% are reset to 0%.

### Pruning nodes

Due to our implementation of the external-sampling MCCFR, nodes with 0 opponent's reach probability are never considered.

We also use regret-based pruning. Actions below a certain regret level (specified in the first `pruning_range` CLI argument) are only considered every 20th iteration (all in the same cycle). 

The second `pruning_range` CLI argument is the minimum regret.

### Appendix: modifications considered but not used

We have not implemented ICFR because of the expected effort/benefit ratio.

We are not using variance-reduction techniques with respect to opponent's sampled actions, because of no noticeable benefit when trying them. This may be due to the fact that most variance comes from the sampling of cards, not opponent's actions.

## Evaluation

We have developed an algorithm to compute the exploitability of the AI in the unabstracted game.

However, we found it infeasible to compute the explotability beyond approximately the 3rd round. This means we have a limited idea as to whether we are using the best possible variation of the algorithm and a good abstraction of the information sets.

## Resource limits and abstraction

We have memory, storage and computation constraints that force us to abstract information sets. Unabstracted, they would contain an list of our cards and the list of all moves that have been played in the given round.

We train the AI separately for each sorted array of players' hand sizes (e.g. 3 cards vs 5 cards). There are 66 of those, which we call 'setups'.

When a player has 11 cards, there are 10^6.7 possible hands (though only 10^5.7 strategically-distint ones). When we consider just the last three elements from the history, we have 10^6.0 combinations.

One cannot compute this algorithm with 10^11.7 information sets. 

Instead, our abstraction is designed to have a limit of around 5 million (10^6.7) infosets. This has a few implications: 

* it limits the memory consumption to around 4 GB;
* it limits the strategy csv output files to less than 50 MB when compressed, which is a hard limit when we are deploying the AI with AWS lambda using one Lambda per setup; and
* each setup's strategy can be calibrated to a reasonable extent in around 0.5 core-days.

The AI should then take around a core-month with 4GB of memory to train, which costs in the order of 30 USD when trained on on-demand AWS EC2 instances. It can also be reasonably trained on a personal machine.

### Memory consumption

Each information set object stores a regret array of up to 88 double-precision floats and a strategy sum array of up to 88 single-precision floats. Ideally it would also store an array of possible actions, but we're computing it on each visit to the information set because it's computationally cheap and saves a lot of memory.

Single-precision float arrays have seemed to update slower in our experience, so we only used them for the strategy sum, even though we do not need the actual double precision for regret arrays either.

An average information set will only have 30-40 possible actions. Therefore, for a setups that has a similar number of infosets for low last bets and high last bets, the average infoset weight in memory will be slightly less than 1 kB.

### Penalty

In most setups, we impose a penalty (reduction of payoff) for betting instead of checking. It's regulated using the `penalty` CLI argument. It's a necessary intervention to shorten the number of moves and therefore vastly speed up the training.

### Effect of history abstraction on convergence

Because we only remember the last 3 bets (and imperfectly, too), a node in the game 'tree' might have more than 1 parent, which makes it not a tree. This will likely bias the algorithm (make it not converge to a good strategy in the abstracted strategy space). Whenever there are multiple ways to reach a node, we should be applying bigger regret updates to the node when we are reaching it via a way that's more likely to occur. 

However, the probability of each path depends not only on the opponent, whose reach probability we are reflecting by sampling, but also on us (the updating agent), and since we do not (and should not) make the regret update dependent on our own reach probability, we may apply too many regret updates to a node. 

Specifically, we may visit a node many times during an iteration (because the updating agent explores many possible moves, branching the tree) and update it as if its reach probability was more than 1, even though in actual gameplay an infoset can be only visited once during a round.

To alleviate this problem, we are skipping the downstream search and the regret update for a node when we visit it for the second and subsequent times in an iteration, and returning its value from the first time we visited it. 

This creates another problem. For each opponent hand, we give multiple chances to visit and update the node but capping the number of updates to 1, which might underemphasise updates from opponent hands with high reach probability for the node. However, it reduces the training time significantly.

Note: imposing a higher penalty on betting or using a more relaxed history abstraction will alleviate this problem.

### History abstraction

TBA

### Hand abstraction

TBA

### Encoding

TBA

## Diagnostics

Outside of the above mentioned strategy output files, there's a diagnostic version of them:

* it contains all infosets, including the ones where we only check;
* for every infoset, we note the iterations it was first tocuhed and last touched and the numebr fo times it was touched; and
* we include probabilities below 1%, which are reset to 0% in the normal output file.

For each setup, there's also a training metadata file (`metadata.csv`), which notes: 

* the finishing time;
* the number of iterations;
* the pruning threshold;
* the minimum regret;
* the penalty (for betting instead of checking);
* the number of nodes touched (incremented at maximum once per node per iteration);
* the number of explored infosets;
* the number of infosets in which the strategy is not a check with 100% chance;
* the amount of RAM taken by the training Python process (including the memory claimed by the code that saves the strategies); and 
* the game value of each player (e.g. if we're training the 2 cards vs 3 cards case, it's 1. the game value for the starting player when the 2-card player is starting and 2. the game value for the starting player when the 3-card player is starting).

The duration of training can be looked up in the `tqdm` progress bar in the command line.

## Deployment

```
aws lambda update-function-code --function-name blef-aiagent-cfr --zip-file fileb://cfr_ai/dispatcher/lambda_function.zip
```