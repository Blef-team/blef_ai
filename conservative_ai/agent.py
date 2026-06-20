import random
from shared.ai import agent
from shared.probabilities import dynamic_probabilities
from shared.game_utils import GameRules

CHECK_MULT = 1.5   # bet-vs-check aggression (higher -> check less)
ALPHA = 3.0        # exponent sharpening the bet-selection weights (higher = more conservative/peaked)
CHECK_EXP = 3.0    # exponent sharpening the check-vs-bet choice

def normalise(arr):
    sum_arr = sum(arr)
    if sum_arr:
        return [i/sum_arr for i in arr]
    return arr

def elementwise_mul(first_array, second_array):
    return [a*b for a, b in zip(first_array, second_array)]

def compute_sampling_weights(bet_probs, alpha=3.0):
    return normalise([i ** alpha for i in bet_probs])  # conservative: peak on the most plausible bets

class ConservativeAgent(agent.Agent):
    """
        Autonomous AI Agent class to play Blef.
        A simple, conservative agent.
    """
    def __init__(self, base_url=None):
        super(ConservativeAgent, self).__init__(base_url)
        self.nickname = "Dazhbog"

    @staticmethod
    def determine_action(game_state, check_mult=None, alpha=None, check_exp=None):
        cm = CHECK_MULT if check_mult is None else check_mult
        a_exp = ALPHA if alpha is None else alpha
        c_exp = CHECK_EXP if check_exp is None else check_exp
        rules = game_state.get("rules", {})
        game_rules = GameRules(rules.get("deck_size", 24))
        check_action_id = game_rules.check_action_id

        last_bet = None
        if game_state.get("history"):
            last_bet = game_state.get("history")[-1]["action_id"]
        else:
            last_bet = -1

        # Filter eliminated players and determine adjacency role
        all_players = game_state.get("players", [])
        active_players = [p for p in all_players if p.get("n_cards", 0) > 0]
        cp_nickname = game_state.get("cp_nickname")
        cp_index = next((i for i, p in enumerate(active_players) if p.get("nickname") == cp_nickname), -1)
        
        N = 1
        role = "alone"
        last_ally_nickname = cp_nickname

        if cp_index != -1 and active_players[cp_index].get("team") is not None:
            cp_team = active_players[cp_index].get("team")
            
            # Count contiguous allies forward
            forward_allies = 0
            for i in range(1, len(active_players)):
                idx = (cp_index + i) % len(active_players)
                if active_players[idx].get("team") == cp_team:
                    forward_allies += 1
                else:
                    break
                    
            # Count contiguous allies backward
            backward_allies = 0
            for i in range(1, len(active_players)):
                idx = (cp_index - i) % len(active_players)
                if active_players[idx].get("team") == cp_team:
                    backward_allies += 1
                else:
                    break
            
            if forward_allies > 0 or backward_allies > 0:
                N = 1 + forward_allies + backward_allies
                if backward_allies == 0:
                    role = "first"
                    last_ally_idx = (cp_index + forward_allies) % len(active_players)
                    last_ally_nickname = active_players[last_ally_idx].get("nickname")
                elif forward_allies == 0:
                    role = "last"
                else:
                    role = "middle"

        # Extract Generic Probabilities and Bet Floor
        bet_probs_generic = dynamic_probabilities.get_generic_bet_probabilities(game_state, last_bet=last_bet)
        bet_floor = 0
        for i, prob in enumerate(bet_probs_generic):
            if prob == 1.0:
                bet_floor = i

        # Middle role: just raise by 1
        if role == "middle":
            return min(last_bet + 1, check_action_id)

        # First role: get weights
        if role == "first":
            # First ally looks ahead to the situation the last ally will face 
            effective_last_bet = max(last_bet + N - 1, bet_floor - 1)

            if effective_last_bet >= check_action_id - 1:
                return check_action_id if last_bet > -1 else 0

            # Temporarily pretend to be the last ally to calculate using their jokers
            original_cp_nickname = game_state["cp_nickname"]
            game_state["cp_nickname"] = last_ally_nickname
            bet_probs_betting = dynamic_probabilities.get_bet_probabilities(game_state, for_betting=True, last_bet=effective_last_bet)
            game_state["cp_nickname"] = original_cp_nickname
            sampling_weights = compute_sampling_weights(bet_probs_betting, a_exp)
            
        else:
            # role is "alone" or "last" -> Their immediate bet is strictly bound by the bet floor
            effective_last_bet = max(last_bet, bet_floor - 1)
            bet_probs_betting = dynamic_probabilities.get_bet_probabilities(game_state, for_betting=True, last_bet=effective_last_bet)
            sampling_weights = compute_sampling_weights(bet_probs_betting, a_exp)

        # Check/Bet Evaluation (alone and first can check; last cannot)
        if role in {"alone", "first"} and last_bet > -1 and last_bet < check_action_id:
            prob_last_bet_exists = dynamic_probabilities.get_bet_probabilities(game_state, for_betting=False, specific_action_id=last_bet)
            
            if prob_last_bet_exists == 0:
                return check_action_id

            success_prob_of_check = 1 - prob_last_bet_exists
            
            weighted_probs = elementwise_mul(sampling_weights, bet_probs_betting)
            success_prob_of_bet = sum(weighted_probs)

            check_vs_bet_probs = [success_prob_of_check, success_prob_of_bet * cm]
            check_vs_bet_probs = [i ** c_exp for i in check_vs_bet_probs]  # sharpen toward the better option
            if sum(check_vs_bet_probs) == 0:
                return check_action_id

            check = random.choices([True, False], weights=normalise(check_vs_bet_probs), k=1)[0]
            if check:
                return check_action_id

        # Fallback if there are absolutely zero valid remaining bets
        if not any(sampling_weights):
            return check_action_id if last_bet > -1 else 0

        # By this point we've chosen not to check, so we bet
        if role == "first":
            return min(last_bet + 1, check_action_id)
        else: # Alone or last role
            return random.choices(range(len(sampling_weights)), weights=sampling_weights, k=1)[0]

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
