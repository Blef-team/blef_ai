def get_check_id(game_state):
    # Check the game rules for deck size
    rules = game_state.get("rules", {})
    deck_size = rules.get("deck_size")

    return 140 if deck_size == 32 else 88


def get_safe_default(game_state):
    """
    Determines the safe default action ID based on the game state and history.
    If the history is empty, returns 0. Otherwise, returns the correct check action ID
    based on the deck size (24 or 32 cards).
    
    Args:
    - game_state (dict): The current state of the game.

    Returns:
    - int: The default action ID (0 if history is empty, or check action ID based on deck size).
    """
    # Check if game history is empty
    if not game_state.get("history"):
        return 0  # No action history, default to 0
    
    return get_check_id(game_state)


def is_legal(action, game_state):
    check = get_check_id(game_state)
    history = game_state.get("history", [])

    # First, ensure the action is within the valid range
    if action not in range(check + 1):
        return False

    # Next, if there's history, ensure the action hasn't already happened
    if history and action <= history[-1].get("action_id", float('-inf')):
        return False

    # Finally, if no history, ensure the action isn't the check action
    if not history and action == check:
        return False

    return True


