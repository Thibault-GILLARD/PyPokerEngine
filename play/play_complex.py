from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.game_state_utils import restore_game_state
from pokereval.card import Card
from pokereval.hand_evaluator import HandEvaluator

from colorama import init, Fore, Back, Style

def pypokerengine_card_to_pokereval_card(card_str):
    """
    Convert a card string from PyPokerEngine to a Card object from pokereval.card.
    """
    SUIT_MAP = { 'S': 's', 'H': 'h', 'D': 'd', 'C': 'c' }
    rank = card_str[1]
    suit = card_str[0]
    return Card(rank=rank, suit=SUIT_MAP[suit]) 
    
def how_much_to_raise(valid_actions, hand_strengh):
    """Calculate from the hand evaluator how much to raise."""
    # The choice happens between the 0.6 and 1 (hand strength)
    ratio = (hand_strengh - 0.6) / 0.4
    amount2raise = valid_actions[2]['amount']['min'] + ratio * (valid_actions[2]['amount']['max'] - valid_actions[2]['amount']['min'])
    return int(amount2raise)
    


class FishPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
        return action, amount

    def receive_game_start_message(self, game_info):
        print('hey')
        print(game_info)

    def receive_round_start_message(self, round_count, hole_card, seats):
        print("---Round start message received:")
        print("Round count:", round_count)
        print("Hole card:", hole_card)
        print("Seats:", seats)

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


class RLPlayer(BasePokerPlayer):

    def __init__(self):
        pass

    def receive_game_start_message(self, game_info):
        print(game_info)
        player_num = game_info["player_num"]
        max_round = game_info["rule"]["max_round"]
        small_blind_amount = game_info["rule"]["small_blind_amount"]
        ante_amount = game_info["rule"]["ante"]
        blind_structure = game_info["rule"]["blind_structure"]
        
        self.emulator = Emulator()
        self.emulator.set_game_rule(player_num, max_round, small_blind_amount, ante_amount)
        self.emulator.set_blind_structure(blind_structure)
        
        # Register algorithm of each player which used in the simulation.
        for i in range(player_num):
            self.emulator.register_player(game_info["seats"][i]["uuid"], FishPlayer())
            
    def receive_round_start_message(self, round_count, hole_card, seats):
        print("-Round start message received:")
        print("Round count:", round_count)
        print("Hole card:", hole_card)
        print("Seats:", seats)

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    def run_simulation(self, game_state):
        try:
            updated_state, events = self.emulator.run_until_game_finish(game_state)
            return updated_state
        except IndexError:
            print("IndexError occurred in the simulation")
            return None

    def declare_action(self, valid_actions, hole_card, round_state):

        print(f'valid actions: {valid_actions}')

        game_state = restore_game_state(round_state)
        updated_state = self.run_simulation(game_state)

        #if updated_state is None:
         #   print("Simulation failed, choosing a default action")
          #  return "call", valid_actions[1]["amount"]

        print("---- Game state after simulation ----")
        print(hole_card)

        # Simple version
        """ if self.is_good_simulation_result(round_state, hole_card):
            return "call", valid_actions[1]["amount"] #maybe 
        else:
            return "fold", 0 """
            
        # Complex version, with all the choice of actions
        hand_strength = self.is_good_simulation_result(round_state, hole_card)
        if hand_strength < 0.1: # fold
            return "fold", 0
        elif hand_strength < 0.6: # call
            return "call", valid_actions[1]["amount"]
        else : # raise
            return "raise", how_much_to_raise(valid_actions, hand_strength)
        

    def is_good_simulation_result(self, round_state, hole_card):
        my_uuid = self.uuid
        
        # Find my player info in round_state
        for p in round_state["seats"]:
            if p["uuid"] == my_uuid:
                player_info = p
                break
        else:
            raise ValueError("My player info not found in round state")
        
        # Evaluate my hand strength
        print('---------')
        print(f'hole card: {hole_card}')
        community_cards = [pypokerengine_card_to_pokereval_card(card_str) for card_str in round_state["community_card"]]
        print(community_cards)
        hole_cards = [pypokerengine_card_to_pokereval_card(card_str) for card_str in hole_card]
        
        hand_strength = HandEvaluator.evaluate_hand(hole_cards, community_cards)
        print(f'hand strength: {hand_strength}')
        
        # Simple version
        """ if hand_strength >= 0.5:  # replace with your own threshold
            return True, 
        else:
            return False """
            
        # Complex version, with all the choice of actions
        return hand_strength


from pypokerengine.api.game import setup_config, start_poker

config = setup_config(max_round=10, initial_stack=100, small_blind_amount=5)
config.register_player(name="p1", algorithm=RLPlayer())
config.register_player(name="p2", algorithm=FishPlayer())
config.register_player(name="p3", algorithm=FishPlayer())

game_result = start_poker(config, verbose=2)

print(game_result)

