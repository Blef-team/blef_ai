from time import sleep
import numpy as np
from shared.api import game_manager
from shared.probabilities import handler
from shared.ai import agent


class ConservativeAgent(agent.Agent):
    """
        Autonomous AI Agent class to play Blef.
        A simple, conservative agent.
    """

    def __init__(self, base_url=None):
        super(ConservativeAgent, self).__init__(base_url)
        self.nickname = "Honorable_Gentleman"

    def run(self):
        """
            Play the game.
        """
        done = False
        while not done:
            sleep(1) # DEBUG / DEV
            succeeded, game_state = self.game_manager.get_game_state()
            if not succeeded:
                print("Can't get the game state.")
                continue
            if game_state.get("status") != "Running":
                if game_state.get("status") != "Finished":
                    print("Game not yet started")
                else:
                    print("Game finished")
                    break
                continue
            if game_state.get("cp_nickname") != self.nickname:
                print("It's not my turn")
                continue
            matching_hands = [hand for hand in game_state.get("hands", []) if hand.get("nickname") == self.nickname]
            if len(matching_hands) != 1:
                print("Can't get my hand from the game state.")
                continue
            hand = [(card["value"], card["colour"]) for card in matching_hands[0]["hand"]]
            players = game_state.get("players", [])
            others_card_num = sum([player.get("n_cards") for player in players if player.get("nickname", self.nickname) != self.nickname])
            if not players or not others_card_num:
                print("Can't get my hand or other player's card numbers from the game state.")
                continue
            bet_probs = self.prob_handler.get_probability_vector(hand, others_card_num)
            if game_state.get("history"):
                last_bet = game_state.get("history")[-1]["action_id"]
                if bet_probs[last_bet] == 0:
                    self.game_manager.play(88)
                    continue
                check_vs_bet_probs = np.array([1-bet_probs[last_bet], sum(bet_probs**2) - bet_probs[last_bet]**2])
                check_vs_bet_probs **= 2  # Be conservative
                check = np.random.choice([True, False], p=check_vs_bet_probs/sum(check_vs_bet_probs))
                if check:
                    self.game_manager.play(88)
                    continue
            bet_probs **= 2  # Be conservative
            sampled_action = np.random.choice(np.arange(len(bet_probs)), p=bet_probs / sum(bet_probs))
            self.game_manager.play(sampled_action)
