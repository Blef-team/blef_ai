from time import sleep
import urllib3
import json
from uuid import UUID


class GameManager(object):
    """Manager of API calls to the Blef Game Engine Service."""

    def __init__(self, base_url="http://localhost:8002/v2.1/"):
        super(GameManager, self).__init__()
        if not isinstance(base_url, str):
            raise TypeError("base_url must be a string")
        self.base_url = base_url
        self.game_uuid = None
        self.player_uuid = None
        self.test_connection(base_url)
        self.playing_locally = base_url.strip("http://").startswith("localhost")

    @staticmethod
    def test_connection(base_url):
        """
            Test whether it's possible to reach a valid
            Game Engine Service at self.base_url
        """
        http = urllib3.PoolManager()
        response = http.request("GET", base_url+"games")
        if response.status_code != 200:
            ConnectionError("base_url is not a valid base path of a running game engine")

    def create_game(self):
        """
            Call the Game Engine Service /create endpoint
            return: succeeded(bool), game_uuid(uuid.UUID)
        """
        if not self.playing_locally:
            sleep(1)
        url = self.base_url + "games/create"
        succeeded = False
        game_uuid = None
        http = urllib3.PoolManager()
        response = http.request("GET", url)
        if response.status == 200:
            json_data = json.loads(response.data)
            uuid = json_data.get("game_uuid")
            if uuid:
                try:
                    game_uuid = UUID(uuid)
                    succeeded = True
                except:
                    pass
        return succeeded, game_uuid

    def join_game(self, game_uuid, nickname):
        """
            Call the Game Engine Service /games/{id}/join endpoint
            store uuids in the object.
            return: succeeded(bool), player_uuid(uuid.UUID)
        """
        if not self.playing_locally:
            sleep(1)
        self.update_game_uuid(game_uuid)
        url = self.base_url + "games/{}/join?nickname={}".format(str(self.game_uuid), nickname)
        succeeded = False
        player_uuid = None
        http = urllib3.PoolManager()
        response = http.request("GET", url)
        if response.status == 200:
            json_data = json.loads(response.data)
            uuid = json_data.get("player_uuid")
            if uuid:
                try:
                    player_uuid = UUID(uuid)
                    succeeded = True
                    self.game_uuid = game_uuid
                    self.player_uuid = player_uuid
                except (ValueError, AttributeError, TypeError):
                    pass
        return succeeded, player_uuid

    def start_game(self, game_uuid=None, player_uuid=None):
        """
            Call the Game Engine Service /games/{id}/start endpoint
            return: succeeded(bool)
        """
        if not self.playing_locally:
            sleep(1)
        self.update_game_uuid(game_uuid)
        self.update_player_uuid(player_uuid)
        url = self.base_url + "games/{}/start?admin_uuid={}".format(str(self.game_uuid), str(self.player_uuid))
        succeeded = False
        http = urllib3.PoolManager()
        response = http.request("GET", url)
        if response.status == 202:
            json_data = json.loads(response.data)
            message = json_data.get("message")
            if message == "Game started":
                succeeded = True
        return succeeded

    def play(self, action_id, game_uuid=None, player_uuid=None):
        """
            Call the Game Engine Service /games/{id}/play endpoint
            return: succeeded(bool)
        """
        if not self.playing_locally:
            sleep(3)
        self.update_game_uuid(game_uuid)
        self.update_player_uuid(player_uuid)

        url = self.base_url + "games/{}/play?player_uuid={}&action_id={}".format(str(self.game_uuid), str(self.player_uuid), action_id)
        succeeded = False
        http = urllib3.PoolManager()
        response = http.request("GET", url)
        if response.status == 200:
            succeeded = True
        return succeeded

    def get_game_state(self, game_uuid=None, player_uuid=None):
        """
            Call the Game Engine Service /games/{id} endpoint
            return: succeeded(bool), game_state(dict)
        """
        if not self.playing_locally:
            sleep(1)
        self.update_game_uuid(game_uuid)
        self.update_player_uuid(player_uuid)

        url = self.base_url + "games/{}?player_uuid={}".format(str(self.game_uuid), str(self.player_uuid))
        succeeded = False
        game_state = None
        http = urllib3.PoolManager()
        response = http.request("GET", url)
        if response.status == 200:
            response_obj = json.loads(response.data)
            if response_obj:
                game_state = response_obj
                succeeded = True
        return succeeded, game_state

    def update_game_uuid(self, game_uuid):
        """
            Handle validating and storing game_uuid
        """
        if game_uuid is not None:
            if isinstance(game_uuid, UUID):
                self.game_uuid = game_uuid
            else:
                print("Provided game_uuid is not valid and will be ignored.")
        if self.game_uuid is None or not isinstance(self.game_uuid, UUID):
            raise TypeError("game_uuid is required and must be type uuid.UUID")

    def update_player_uuid(self, player_uuid):
        """
            Handle validating and storing player_uuid
        """
        if player_uuid is not None:
            if isinstance(player_uuid, UUID):
                self.player_uuid = player_uuid
            else:
                print("Provided player_uuid is not valid and will be ignored.")
        if self.player_uuid is None or not isinstance(self.player_uuid, UUID):
            raise TypeError("player_uuid is required and must be type uuid.UUID")
