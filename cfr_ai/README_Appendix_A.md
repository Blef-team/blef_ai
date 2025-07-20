## Appendix A: different convergence resolution algorithms

To enable all three methods mentioned in the history abstraction and convergence section, change:

### `information_set.py`
```
    def __init__(self, history: List[int], iter: int, min_bet: int):
        self.possible_actions = get_possible_actions(history, min_bet)
        self.regrets = np.zeros(len(self.possible_actions))
        self.strategy_sum = np.zeros(len(self.possible_actions), dtype=np.float32)
        self.times_touched = 0
        self.first_touched = iter
        self.last_touched = 0
        self.temporary_value = 0.0

    def get_strategy(self, reach_probability: float) -> np.array:
        if any(self.regrets > 0):
            strategy = np.maximum(0, self.regrets)
            strategy /= sum(strategy)
        else:
            strategy = np.zeros(len(self.regrets))
            strategy[-1] = 1.0

        self.strategy_sum += reach_probability * strategy
        return strategy
```
to:
```
class InformationSet():
    def __init__(self, history: List[int], iter: int, min_bet: int, convergence_resolution: int):
        self.possible_actions = get_possible_actions(history, min_bet)
        self.regrets = np.zeros(len(self.possible_actions))
        self.strategy_sum = np.zeros(len(self.possible_actions), dtype=np.float32)
        self.times_touched = 0
        self.first_touched = iter
        self.last_touched = -1
        if convergence_resolution == 1 or convergence_resolution == 2:
            self.temporary_value = 0.0
            if convergence_resolution == 2:
                self.last_considered = -1

    def get_strategy(self) -> np.array:
        if any(self.regrets > 0):
            strategy = np.maximum(0, self.regrets)
            strategy /= sum(strategy)
        else:
            strategy = np.zeros(len(self.regrets))
            strategy[-1] = 1.0
        return strategy
```

### `trainer.py`
```
    def __init__(self, hand_sizes: List[int], min_bet: int, pruning_range: List[int], penalty: float, log_points: int):
        self.infoset_map: Dict[str, InformationSet] = {}
        self.hand_sizes = hand_sizes
        self.nodes_touched = 0
        self.pruning_threshold = pruning_range[0]
        self.min_regret = pruning_range[1]
        self.penalty = penalty
        self.log_points = log_points
        self.min_bet = min_bet

    def get_node_value(self, hands: List[np.ndarray], hand_abstractions: List[str], history: List[int], reach_probability: float, active_player: int, traverser: int, prune_feast: bool, existence_array: np.ndarray, iter: int) -> float:
        if Game.check_finish(history):
            return 1 if existence_array[history[-2]] else -1
        
        key = make_key(hands[active_player], hand_abstractions[active_player], history, self.min_bet)
        if key not in self.infoset_map:
            self.infoset_map[key] = InformationSet(history, iter, self.min_bet)
        info_set = self.infoset_map[key]

        if info_set.last_touched == iter:
            return info_set.temporary_value
        else:
            possible_actions = info_set.possible_actions
            counterfactual_values = np.zeros(len(possible_actions))
            opponent = (active_player + 1) % 2

            if active_player == traverser:
                strategy = info_set.get_strategy(reach_probability)
                for i, action in enumerate(possible_actions):
                    if info_set.regrets[i] >= self.pruning_threshold or prune_feast:
                        counterfactual_values[i] = -self.get_node_value(hands, hand_abstractions, history + [action], reach_probability * strategy[i], opponent, traverser, prune_feast, existence_array, iter)
                node_value = np.dot(counterfactual_values, strategy)
                if prune_feast:
                    info_set.regrets = np.maximum(info_set.regrets + counterfactual_values - node_value, self.min_regret)
                else:
                    to_update = info_set.regrets >= self.pruning_threshold
                    info_set.regrets[to_update] += counterfactual_values[to_update] - node_value

            else:
                strategy = info_set.get_strategy(0.0)
                action = random.choices(possible_actions, weights=strategy, k=1)[0]
                node_value = -self.get_node_value(hands, hand_abstractions, history + [action], reach_probability, opponent, traverser, prune_feast, existence_array, iter) + self.penalty
            info_set.times_touched += 1
            info_set.last_touched = iter
            info_set.temporary_value = node_value
            self.nodes_touched += 1
            return node_value
```
to:
```
    def __init__(self, hand_sizes: List[int], min_bet: int, convergence_resolution: int, pruning_range: List[int], penalty: float, log_points: int):
        self.infoset_map: Dict[str, InformationSet] = {}
        self.hand_sizes = hand_sizes
        self.nodes_touched = 0
        self.pruning_threshold = pruning_range[0]
        self.min_regret = pruning_range[1]
        self.penalty = penalty
        self.log_points = log_points
        self.min_bet = min_bet
        self.convergence_resolution = convergence_resolution

    def get_node_value(self, hands: List[np.ndarray], hand_abstractions: List[str], history: List[int], reach_probability: float, active_player: int, traverser: int, prune_feast: bool, existence_array: np.ndarray, iter: int) -> float:
        if Game.check_finish(history):
            return 1 if existence_array[history[-2]] else -1
        
        key = make_key(hands[active_player], hand_abstractions[active_player], history, self.min_bet)
        if key not in self.infoset_map:
            self.infoset_map[key] = InformationSet(history, iter, self.min_bet, self.convergence_resolution)
        info_set = self.infoset_map[key]

        if self.convergence_resolution != 0 and info_set.last_touched == iter:
            return info_set.temporary_value
        else:
            possible_actions = info_set.possible_actions
            counterfactual_values = np.zeros(len(possible_actions))
            opponent = (active_player + 1) % 2

            if active_player == traverser:
                strategy = info_set.get_strategy()
                for i, action in enumerate(possible_actions):
                    if info_set.regrets[i] >= self.pruning_threshold or prune_feast:
                        counterfactual_values[i] = -self.get_node_value(hands, hand_abstractions, history + [action], reach_probability * strategy[i], opponent, traverser, prune_feast, existence_array, iter)
                node_value = np.dot(counterfactual_values, strategy)
                # In CN, if the node was available to the opponent earlier but chosen against, do not update regrets or strategy sum (to fix bias in implicit reach probabilities due to multiple shots)
                if self.convergence_resolution != 2 or info_set.last_considered != iter:
                    info_set.strategy_sum += reach_probability * strategy
                    if prune_feast:
                        info_set.regrets = np.maximum(info_set.regrets + counterfactual_values - node_value, self.min_regret)
                    else:
                        to_update = info_set.regrets >= self.pruning_threshold
                        info_set.regrets[to_update] += counterfactual_values[to_update] - node_value
                    if self.convergence_resolution == 2:
                        info_set.last_considered = iter

            else:
                strategy = info_set.get_strategy()
                action = random.choices(possible_actions, weights=strategy, k=1)[0]
                node_value = -self.get_node_value(hands, hand_abstractions, history + [action], reach_probability, opponent, traverser, prune_feast, existence_array, iter) + self.penalty
                if self.convergence_resolution == 2:
                # Mark all other traverser's infosets reachable from here as considered
                    for discarded_action in possible_actions:
                        if discarded_action == action or discarded_action == 88:
                            continue
                        discarded_key = make_key(hands[opponent], hand_abstractions[opponent], history + [discarded_action], self.min_bet)
                        if discarded_key not in self.infoset_map:
                            self.infoset_map[discarded_key] = InformationSet(history + [discarded_action], iter, self.min_bet, self.convergence_resolution)
                        self.infoset_map[discarded_key].last_considered = iter
            info_set.times_touched += 1
            info_set.last_touched = iter
            if self.convergence_resolution != 0:
                info_set.temporary_value = node_value
            self.nodes_touched += 1
            return node_value
```

### `training.py`:

```
VERSION_CODE = 'TV-NR-SS-SD-MB'
```
to
```
VERSION_CODE = 'NR-SS-SD-MB'
```
Then
```
    cfr_trainer = Trainer(args.hand_sizes, args.min_bet, args.pruning_range, args.penalty, args.log_points)
```
to
```    
    cfr_trainer = Trainer(args.hand_sizes, args.min_bet, args.convergence_resolution, args.pruning_range, args.penalty, args.log_points)
```
Then
```
            csv_row = writer.writerow({"k": "Version code", "v": VERSION_CODE})
```
to
```
            if args.convergence_resolution == 0:
                csv_row = writer.writerow({"k": "Version code", "v": VERSION_CODE})
            elif args.convergence_resolution == 1:
                csv_row = writer.writerow({"k": "Version code", "v": f'TV-{VERSION_CODE}'})
            if args.convergence_resolution == 2:
                csv_row = writer.writerow({"k": "Version code", "v": f'CN-{VERSION_CODE}'})
```
