import random
from shared.ai import agent, actions
from shared.probabilities import dynamic_probabilities
from shared.api import gpt


base_prompt_24_cards = os,getenv("BASE_PROMPT_24_CARDS")
base_prompt_32_cards = os,getenv("BASE_PROMPT_32_CARDS")


def format_prompt_game_state(game_state):
    return "" #TODO implement

def format_prompt_bet_floor(bet_floor):
    return "" #TODO implement

def format_prompt_bet_probs_betting(probs):
    return "" #TODO implement

def format_prompt_bet_probs_generic(probs):
    return "" #TODO implement

def prompt_action(game_state, bet_probs_betting, bet_probs_generic, bet_floor=None, custom_prompt_postfix=""):
    """
        Prompt GPT and return a valid action ID.
        
        Add current game state information to a prepared base prompt and query the Open AI API.
        
        Returns:
        - int: The determined action id for the agent to execute
    """
    rules = game_state.get("rules", {})
    deck_size = rules.get("deck_size")

    prompt = prompt_24_cards
    if not prompt or deck_size == 32 and prompt_32_cards:
        prompt = prompt_32_cards

    prompt += format_prompt_game_state(game_state)
    prompt += format_prompt_bet_floor(bet_floor) if bet_floor else ""
    prompt += format_prompt_bet_probs_betting(bet_probs_betting)
    prompt += format_prompt_bet_probs_generic(bet_probs_generic)

    response = gpt.get_response(prompt)

    action_id = None
    try:
        action_id = int(response.strip(".")[-3:])
    except ValueError:
        action_id = actions.get_safe_default(game_state)
    if not actions.is_legal(action_id, game_state):
        action_id = actions.get_safe_default(game_state)

    return action_id

class GPTAgent(agent.Agent):
    """
        Autonomous AI Agent class to play Blef.
        Calls OpenAI's API to use an LLM to determine its action.
    """
    def __init__(self, base_url=None):
        super(GPTAgent, self).__init__(base_url)
        self.nickname = "Rusalka"

    @staticmethod
    def determine_action(game_state):
        rules = game_state.get("rules", {})
        game_rules = dynamic_probabilities.GameRules(rules.get("deck_size", 24))
        check_action_id = game_rules.check_action_id
        
        last_bet = game_state.get("history", [{}])[-1].get("action_id", None)

        # Set the bet floor - skip bets up to the next 100% certain bet, if any such higher bet exists
        bet_probs_generic = dynamic_probabilities.get_generic_bet_probabilities(game_state, last_bet=last_bet)
        bet_floor = next((i for i, prob in enumerate(bet_probs_generic) if prob == 1.0), 0)
        effective_last_bet = max(last_bet if last_bet is not None else -1, bet_floor - 1)

        bet_probs_betting = dynamic_probabilities.get_bet_probabilities(game_state, for_betting=True, last_bet=effective_last_bet)

        return prompt_action(game_state, bet_probs_betting, bet_probs_generic, bet_floor=bet_floor)

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
determine_action = GPTAgent.determine_action
