import random
from shared.ai import agent
from shared.probabilities import dynamic_probabilities

def normalise(arr):
    sum_arr = sum(arr)
    if sum_arr:
        return [i/sum_arr for i in arr]
    return arr

def elementwise_mul(first_array, second_array):
    return [a*b for a, b in zip(first_array, second_array)]

def compute_sampling_weights(bet_probs, bet_probs_generic):
    if len(bet_probs) != len(bet_probs_generic):
        raise ValueError("Bet probability arrays are not of equal length")
    return normalise([bet_probs[i] ** 3 * bet_probs_generic[i] ** 2 for i in range(len(bet_probs))])

class ConservativeCrawlingAgent(agent.Agent):
    """
        Autonomous AI Agent class to play Blef.
        A simple, conservative agent.
    """
    def __init__(self, base_url=None):
        super(ConservativeCrawlingAgent, self).__init__(base_url)
        self.nickname = "Porevit"

    @staticmethod
    def determine_action(game_state):
        rules = game_state.get("rules", {})
        game_rules = dynamic_probabilities.GameRules(rules.get("deck_size", 24))
        check_action_id = game_rules.check_action_id
        
        last_bet = None
        if game_state.get("history"):
            last_bet = game_state.get("history")[-1]["action_id"]

        bet_probs_generic = dynamic_probabilities.get_generic_bet_probabilities(game_state, last_bet=last_bet)
        bet_floor = 0
        for i, prob in enumerate(bet_probs_generic):
            if prob == 1.0:
                bet_floor = i
        
        effective_last_bet = max(last_bet if last_bet is not None else -1, bet_floor - 1)

        bet_probs_betting = dynamic_probabilities.get_bet_probabilities(game_state, for_betting=True, last_bet=effective_last_bet)
        sampling_weights = compute_sampling_weights(bet_probs_betting, bet_probs_generic)

        if last_bet is not None and last_bet < check_action_id:
            prob_last_bet_exists = dynamic_probabilities.get_bet_probabilities(game_state, for_betting=False, specific_action_id=last_bet)

            if prob_last_bet_exists == 0:
                return check_action_id

            success_prob_of_check = 1 - prob_last_bet_exists
            
            weighted_probs = elementwise_mul(sampling_weights, bet_probs_betting)
            success_prob_of_bet = sum(weighted_probs)

            check_vs_bet_probs = [success_prob_of_check, success_prob_of_bet * 1.2]
            check_vs_bet_probs = [i ** 3 for i in check_vs_bet_probs]  # Be conservative
            if sum(check_vs_bet_probs) == 0:
                return check_action_id

            check = random.choices([True, False], weights=normalise(check_vs_bet_probs), k=1)[0]
            if check:
                return check_action_id

        if not any(sampling_weights):
            return check_action_id if last_bet is not None else 0
            
        sampled_action = random.choices(range(len(sampling_weights)), weights=sampling_weights, k=1)[0]
        return sampled_action

    def run(self):
        """
            Play the game.
        """
        if not self.joined_game:
            print("I have not joined any game yet")
            return
        done = False
        while not done:
            succeeded, game_state = self.game_manager.get_game_state()
            if not succeeded:
                print("Can't get the game state.")
                continue

            if game_state.get("status") != "Running":
                if game_state.get("status") == "Not started":
                    print("Game not yet started")
                else:
                    print("Game finished")
                    break
                continue

            if game_state.get("cp_nickname") != self.nickname:
                continue

            sampled_action = self.determine_action(game_state)
            self.game_manager.play(sampled_action)

# Expose determine_action for import
determine_action = ConservativeCrawlingAgent.determine_action
