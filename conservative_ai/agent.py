from shared.api import game_manager
from shared.probabilities import handler


class Agent(object):
    """Autonomous AI Agent class to play Blef."""

    def __init__(self, base_url=None):
        super(Agent, self).__init__()
        self.nickname = "Honorable_Gentleman"
        if base_url is not None:
            self.game_manager = game_manager.GameManager(base_url)
        else:
            self.game_manager = game_manager.GameManager()
        self.prob_handler = handler.Handler()

    def join_game(self, game_uuid, nickname=None):
        """
            Join an existing game of Blef using GameManager.
            If successful, start playing the game.
            return: succeeded(bool)
        """
        if isinstance(nickname, str) and len(nickname) > 0:
            self.nickname = nickname
        succeeded, _ = self.game_manager.join_game(game_uuid, self.nickname)
        if succeeded:
            return self.run()
        return False

    def run(self):
        """
            Play the game.
        """
        pass
