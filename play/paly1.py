
from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.game_state_utils import restore_game_state

#from mymodule.player_model import PlayerModelAI

class FishPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
        return action, amount   # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
    
class RLPLayer(BasePokerPlayer):

    # Setup Emulator object by registering game information
    def receive_game_start_message(self, game_info):
        player_num = game_info["player_num"]
        max_round = game_info["rule"]["max_round"]
        small_blind_amount = game_info["rule"]["small_blind_amount"]
        ante_amount = game_info["rule"]["ante"]
        blind_structure = game_info["rule"]["blind_structure"]
        
        self.emulator = Emulator()
        self.emulator.set_game_rule(player_num, max_round, small_blind_amount, ante_amount)
        self.emulator.set_blind_structure(blind_structure)
        
        # Register algorithm of each player which used in the simulation.
        for player_info in game_info["seats"]["players"]:
            self.emulator.register_player(player_info["uuid"], FishPlayer())

    def declare_action(self, valid_actions, hole_card, round_state):
        game_state = restore_game_state(round_state)
        # decide action by using some simulation result
        updated_state, events = self.emulator.apply_action(game_state, "fold")
        updated_state, events = self.emulator.run_until_round_finish(game_state)
        updated_state, events = self.emulator.run_until_game_finish(game_state)
        if self.is_good_simulation_result(updated_state):
            return # you would declare CALL or RAISE action
        else:
            return "fold", 0
    
     
from pypokerengine.api.game import setup_config, start_poker

config = setup_config(max_round=10, initial_stack=100, small_blind_amount=5)
config.register_player(name="p1", algorithm=RLPLayer())
config.register_player(name="p2", algorithm=FishPlayer())
config.register_player(name="p3", algorithm=FishPlayer())

game_result = start_poker(config, verbose=2)

print(game_result)