import random
from shared.probabilities import handler
from shared.ai import agent


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
        agent_nickname = game_state["cp_nickname"]
        matching_hands = [hand for hand in game_state.get("hands", []) if hand.get("nickname") == agent_nickname]
        if len(matching_hands) != 1:
            raise ValueError("Can't get my hand from the game state.")

        hand = [(card["value"], card["colour"]) for card in matching_hands[0]["hand"]]
        players = game_state.get("players", [])
        others_card_num = sum([player.get("n_cards") for player in players if player.get("nickname", agent_nickname) != agent_nickname])
        total_card_num = sum([player.get("n_cards") for player in players])
        if not players or not others_card_num:
            raise ValueError("Can't get my hand or other player's card numbers from the game state.")

        bet_probs = handler.Handler().get_probability_vector(hand, others_card_num)
        bet_probs_generic = handler.Handler().get_probability_vector([], total_card_num)
        last_bet = None
        if game_state.get("history"):
            last_bet = game_state.get("history")[-1]["action_id"]

        if last_bet is not None and last_bet in range(88):
            for i in range(last_bet):
                bet_probs[i] = 0.0

            if bet_probs[last_bet] == 0:
                return 88

            success_prob_of_check = 1 - bet_probs[last_bet]
            bet_probs[last_bet] = 0.0
            sampling_weights = compute_sampling_weights(bet_probs, bet_probs_generic)
            weighted_probs = elementwise_mul(sampling_weights, bet_probs)

            success_prob_of_bet = sum(weighted_probs)
            check_vs_bet_probs = [success_prob_of_check, success_prob_of_bet * 1.2]
            check_vs_bet_probs = [i ** 3 for i in check_vs_bet_probs]  # Be conservative
            if sum(check_vs_bet_probs) == 0:
                return 88

            check = random.choices([True, False], weights=normalise(check_vs_bet_probs), k=1)[0]
            if check:
                return 88

        sampling_weights = compute_sampling_weights(bet_probs, bet_probs_generic)
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

            sampled_action = determine_action(game_state, self.nickname)
            self.game_manager.play(sampled_action)


# Expose determine_action for import
determine_action = ConservativeCrawlingAgent.determine_action
