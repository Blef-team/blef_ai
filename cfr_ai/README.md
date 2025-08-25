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

### Note: modifications considered but not used

We have considered but ultimately not implemented:
* ICFR, because of the expected effort/benefit ratio;
* some variance-reduction techniques with respect to opponent's sampled actions or cards, because of no noticeable benefit when trying them; and
* discounting regrets, as we haven't found benefits.

## Resource limits and abstraction

Memory, storage and computation constraints force us to heavily abstract information sets. Unabstracted, they would contain an list of our cards and the list of all moves that have been played in the given round.

We train the AI separately for each sorted array of players' hand sizes (e.g. 3 cards vs 5 cards). There are 66 of those, which we call 'setups'.

When a player has 11 cards, there are 10^6.7 possible hands. When we consider just the last three elements from the history, we have 10^6.0 combinations.

One cannot compute this algorithm with 10^11.7 information sets. 

Instead, our abstraction is designed to have a limit of around 5 million (10^6.7) infosets. This has the following benefits: 
* it limits the memory consumption to around 4 GB;
* it limits the strategy csv output files to less than 250 MB, so that they can be uploaded to AWS lambda using one Lambda per setup; and
* each setup's strategy can be calibrated to a reasonable extent in around 0.5 core-days.

The AI should then take around a core-month with 4GB of memory to train, which costs in the order of 30 USD when trained on on-demand AWS EC2 instances. It can also be reasonably trained on a personal machine.

### Memory consumption

Each information set object stores a regret array of up to 88 double-precision floats, a strategy sum array of up to 88 double-precision floats and an array of possible actions. The former two could easily fit into single-precision float arrays and possible actions could be recomputed, but both turn out to slow the code down substantially. 

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

In rounds 1-5, the AI only looks at card values and completely ignores the suits. 

In further rounds, there is a more complex hand-crafted abstraction. First, as the AI, we extract features:
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

### Action abstraction

We have a mechanism for the AI to not acknowledge or make a specific number of the lowest bets (e.g. all high cards). If it encounters one of those bets during online play, it acts as if the round just started.

With 14 cards, any specific high card has 98% chance of existing (97% and 99% with 13 and 15 cards respectively).

With 16 cards on the table, the great straight has 96% chance of existing (88%, 93%, 98% and 99% for 14, 15, 17 and 18 cards respectively). In an experiment we found that having the 11 11 setup discard all bets below straights results in an approximately twofold improvement in time, memory and strategy storage space used.

### Strategy encoding

There is an encoding that highly compresses strategies so that they can be deployed on platforms with limited storage, such as within AWS Lambda functions.

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

The training is done by setup, which is the ordered number of cards per player, no matter which player the AI is and who is starting. To train a model for a specific setup, execute `training.py`, specifying the number of hands. For example, the train the 1 card vs 1 card setup, run this from the project root:

```
python -m cfr_ai.training --hand-sizes 1 1
```

`--hand-sizes` (required) sets the number of cards per player.

`--num-iterations` (default: 5 million) specifies the number of Monte Carlo iterations to run. Within one iteration, each player gets one set of cards and there is only one traverser.

`--no-save` (default: no) doesn't save any outputs. Designed for trial runs where you measure performance.

`--min-bet` (default: 0) specifies the minimum bet the AI will make or acknowledge.

`--pruning-range` (default: -20 and -22) is a tuple that specifies the threshold for pruning and the minimum regret.

`--penalty` (default: 0) sets the penalty (see Penalty above).

`--log-points` (default: 25) specifies the number of points (at equal intervals) where utility will be measured.

`--get-exploitability` computes exact exploitability in the unabstracted game, disaggregated by which player starts.

You will see a `tqdm` progress bar during training and exploitability calculations.

### Training outputs

A training without the `--no-save` flag will output strategy files to the `outputs` folder. Each setup gets a different folder (e.g. `1_1` for 1 vs 1 card). The strategy for each player (depending on the number of cards) will be a separate folder inside that one (e.g. `1`). 

Then, infosets are stored in separate csv files depending on the last bet (88 if  there was none). The keys in the csv complete the abstraction key. For example, in a 1v1 setup, if the CFR player has an Ace and the only previous bet was a High card, Ace, you will find the strategy in `outputs/1_1/1/5.csv` under `k` of `5`. 

The `v` represents the strategy. A two-digit number symbolises the number of consecutive actions with 0% chance. Two-character fragments represent non-zero chances, with higher values representing higher chances and `Ya` being 100%.

As mentioned before, information sets where the strategy is to check 100% of the time are not recorded, in order to save on storage.

There is also a version of the strategy files with extra, diagnostic columns in a separate folder (e.g. `1_diagnostic` instead of `1`). The diagnostic version:

* contains all infosets, including the ones where we only check;
* for every infoset, it notes the iterations it was first and last touched and the number of times it was touched; and
* we include probabilities below 1%, which are reset to 0% in the normal output file.

## Evaluation & analytics

Evaluation is key to informed development of the algorithm. Usually in the case of CFR, it is done by computing exploitability. We have created optimised tools to compute the exploitability of this AI in the unabstracted game. However, we are unable to run them within 1 core-day beyond the 3rd round. Therefore, we use a suite of other tools to get a rough idea as to the performance of the algorithm. 

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
* the game value of each player (e.g. if we're training the 2 cards vs 3 cards case, it's 1. the game value for the starting player when the 2-card player is starting and 2. the game value for the starting player when the 3-card player is starting);
* the log of utilities along the training run; and
* exploitability, if applicable.

### Utility logging

To get an approximate idea of whether we are running enough iterations, we are logging utility at equal intervals across the training run. If there is no substantial trend beyond the first 30% of iterations, then the exploitability coming from insufficient iterations is likely to be low (however, exploitability coming from the abstraction may still be high).

There is an `analysis/training_analytics.py` script that makes:
* a summary table showing key data for each setup trained;
* charts of utility over time for each setup.

To use it, run `python -m cfr_ai.analysis.training_analytics`.

### Head-to-head comparison

There is a `analysis/head_to_head.py` script that can be used to compare two versions of strategies for a single setup by making them play against each other. Using the Monte Carlo sampling, it takes in the order of 10 minutes to run 10,000 games for most setups. It is the best tool for evaluating modifications to the core algorithm.

### Winning probabilities

Using the game values noted down for each setup in the `summary_of_all_runs.csv`, we can compute the probabilities of winning the game starting from a specific setup and with a specific starting player within that setup. To compute those, use the `analysis/win_probabilities.py` script.

## Deployment

The AI is meant to be deployed alongside the [game engine](https://github.com/Blef-team/blef_game_engine). The integration has two components:
* the dispatcher lambda. The code in dispatcher/lambda_function.py needs to be copied over to the `blef-aiagent-cfr` lambda. This can be done through the UI or by zipping the function and executing `aws lambda update-function-code --function-name blef-aiagent-cfr --zip-file fileb://cfr_ai/dispatcher/lambda_function.zip`; and
* a collection of agents, each serving a particular setup.

To create all necessary worker lambdas for the first time, use the `create_lambas` script in the deployment folder (needs configuring)

To deploy an individual setup, you need to run the `deploy` script. For example, for the 1 vs 1 card setup, run `python -m cfr_ai.deployment.deploy --hand-sizes 1 1`

To deploy all setups at once, run `python -m cfr_ai.deployment.deploy_all`

## Performance

The core of the program, including `information_set.py`, `get_node_value` and `precompute_set_existence`, have gone through many rounds of optimisation. However, they would probably be much faster if they were written in a language like C++. We have tried using the `numba` package to compile a C++ version of some functions, like `precompute_set_existence`, but this actually worsened the performance. This is likely due to the frequent interface between Python and C++ (at least once per iteration, of which there are usually tens or hundreds in every second).

## Other notes

We thank [Thomas Trenner](https://github.com/tt293) for his writings on the CFR algorithm, which inspired us to create this AI.

Strategy outputs for every setup are avaiable upon request.
