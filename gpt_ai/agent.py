import os
from shared.ai import agent, actions
from shared.probabilities import dynamic_probabilities
from shared.api import gpt
from shared.game_utils import GameRules, get_set_details_from_action_id


base_prompt_24_cards = os,getenv("BASE_PROMPT_24_CARDS")
base_prompt_32_cards = os,getenv("BASE_PROMPT_32_CARDS")

def format_card(card, deck_size=24):
    values_24 = {i: v for i, v in enumerate(["9", "10", "J", "Q", "K", "A"])}
    values_32 = {i: v for i, v in enumerate(["7", "8", "9", "10", "J", "Q", "K", "A"])}
    values = values_32 if deck_size == 32 else values_24
    suits = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
    return f"{values.get(card['value'], '?')}{suits.get(card['colour'], '?')}"

def format_set_details(action_id):
    set_details = get_set_details_from_action_id(action_id)
    return f"{set_details.get('set_type','')}, {set_details.get('detail_1','')}, {set_details.get('detail_2','')}"

def format_prompt_game_state(game_state):
    lines = []

    rules = game_state.get("rules", {})
    deck_size = rules.get("deck_size")
    lines.append(f"Game State:")
    lines.append(f"Deck Size: {rules.get('deck_size', '?')} (cards 9–A in 4 suits)")
    lines.append(f"Jokers: {rules.get('jokers', '?')}")
    lines.append(f"Blanks: {rules.get('blanks', '?')}")
    lines.append(f"Common Cards: {rules.get('common_cards', '?')}")

    common_hand = game_state.get("common_hand", [])
    lines.append("Common Hand: " + (', '.join(format_card(c, deck_size=deck_size) for c in common_hand) if common_hand else "Empty"))

    lines.append("Players:")
    for player in game_state.get("players", []):
        name = player["nickname"]
        n_cards = player["n_cards"]
        suffix = " (You)" if name == game_state.get("cp_nickname") else ""
        lines.append(f"{name} — {n_cards} card{'s' if n_cards != 1 else ''}{suffix}")

    your_hand = next((p["hand"] for p in game_state.get("hands", []) if p["nickname"] == game_state.get("cp_nickname")), [])
    lines.append(f"Your Hand ({game_state['cp_nickname']}): " + (', '.join(format_card(c, deck_size=deck_size) for c in your_hand) if your_hand else "Empty"))

    history = game_state.get("history", [])
    if history:
        lines.append("Action History:")
        for h in history:
            action_id = h["action_id"]
            action_details = format_set_details(action_id)
            lines.append(f"{h['player']} bet {action_details} (action_id {action_id})")
    else:
        lines.append("Action History: None")

    return "\n".join(lines)

def format_prompt_bet_floor(bet_floor):
    return f"Don't bet anything lower than {bet_floor}, which is {get_formatted_set_details(action_id)}\n"

def format_prompt_bet_probs(probabilities, generic=False):
    """
    Formats action probabilities into a string, excluding values < 0.0001.

    Returns:
        str: Formatted string of action IDs and their probabilities (in percentage).
    """
    formatted_string = "Probabilities of each legal action ID\n"
    if generic:
        formatted_string = "Probabilities of each legal action ID, if you don’t know your cards\n"
    for i, prob in enumerate(probabilities):
        if prob < 0.0001:
            continue
        formatted_string += f"{i} {prob*100:.2f}%\n"
    return formatted_string

def prompt_action(game_state, bet_probs_betting, bet_probs_generic, bet_floor=None, custom_prompt_postfix=""):
    """
        Prompt GPT and return a valid action ID.
        
        Add current game state information to a prepared base prompt and query the Open AI API.
        
        Returns:
        - int: The determined action id for the agent to execute
    """
    rules = game_state.get("rules", {})
    deck_size = rules.get("deck_size")

    prompt = base_prompt_24_cards
    if not prompt or deck_size == 32 and base_prompt_32_cards:
        prompt = base_prompt_32_cards

    prompt += format_prompt_game_state(game_state)
    prompt += format_prompt_bet_floor(bet_floor) if bet_floor else ""
    prompt += format_prompt_bet_probs(bet_probs_betting)
    prompt += format_prompt_bet_probs(bet_probs_generic, generic=True)

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
        game_rules = GameRules(rules.get("deck_size", 24))
        check_action_id = game_rules.check_action_id
        history = game_state.get("history", [])
        last_bet = history[-1].get("action_id", None) if history else None

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
