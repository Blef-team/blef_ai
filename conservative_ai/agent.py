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

def compute_sampling_weights(bet_probs):
    return normalise([i ** 3 for i in bet_probs]) # Be conservative

class ConservativeAgent(agent.Agent):
    """
        Autonomous AI Agent class to play Blef.
        A simple, conservative agent.
    """

    def __init__(self, base_url=None):
        super(ConservativeAgent, self).__init__(base_url)
        self.nickname = "Dazhbog"

    @staticmethod
    def determine_action(game_state):
        rules = game_state.get("rules", {})
        deck_size = rules.get("deck_size", 24)
        num_actions = 141 if deck_size == 32 else 89
        check_action_id = num_actions - 1

        # Probabilities are now calculated directly from the game_state
        bet_probs_betting = dynamic_probabilities.get_bet_probabilities(game_state, for_betting=True)
        bet_probs_checking = dynamic_probabilities.get_bet_probabilities(game_state, for_betting=False)

        last_bet = None
        if game_state.get("history"):
            last_bet = game_state.get("history")[-1]["action_id"]

        if last_bet is not None and last_bet < check_action_id:
            for i in range(last_bet + 1):
                bet_probs_betting[i] = 0.0

            if bet_probs_checking[last_bet] == 0:
                return check_action_id

            success_prob_of_check = 1 - bet_probs_checking[last_bet]
            sampling_weights = compute_sampling_weights(bet_probs_betting)
            weighted_probs = elementwise_mul(sampling_weights, bet_probs_betting)
            success_prob_of_bet = sum(weighted_probs)

            check_vs_bet_probs = [success_prob_of_check, success_prob_of_bet * 1.2]
            check_vs_bet_probs = [i ** 3 for i in check_vs_bet_probs]  # Be conservative
            if sum(check_vs_bet_probs) == 0:
                return check_action_id

            check = random.choices([True, False], weights=normalise(check_vs_bet_probs), k=1)[0]
            if check:
                return check_action_id

        sampling_weights = compute_sampling_weights(bet_probs_betting)
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
determine_action = ConservativeAgent.determine_action
