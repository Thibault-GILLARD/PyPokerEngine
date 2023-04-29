from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.game_state_utils import restore_game_state
from pokereval.card import Card
from pokereval.hand_evaluator import HandEvaluator
import random 

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
    
    
NB_SIMULATION = 1000


class HonestPlayer(BasePokerPlayer):
    """ HonestPlayer """
    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state['community_card']
        win_rate = estimate_hole_card_win_rate(
                nb_simulation=NB_SIMULATION,
                nb_player=self.nb_player,
                hole_card=gen_cards(hole_card),
                community_card=gen_cards(community_card)
                )
        if win_rate >= 1.0 / self.nb_player:
            action = valid_actions[1]  # fetch CALL action info
        else:
            action = valid_actions[0]  # fetch FOLD action info
        return action['action'], action['amount']

    def receive_game_start_message(self, game_info):
        seats = game_info['seats']
        # Initialize the player_actions dictionary with the UUIDs of the players in the game
        self.player_actions = {player['uuid']: {'folds': 0, 'calls': 0, 'raises': 0, 'last_action': None} for player in seats}

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
    
class FishPlayer(BasePokerPlayer):
    """ Fish player always calls or folds."""
    
    def declare_action(self, valid_actions, hole_card, round_state, seats):
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
        return action, amount

    def receive_game_start_message(self, game_info):
        print('hey')
        print(game_info)

    def receive_round_start_message(self, round_count, hole_card, seats):
        print("---Round start message received:")
        #print("Round count:", round_count)
        print("Hole card:", hole_card)
        #print("Seats:", seats)

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
    
    

class RLPlayer(BasePokerPlayer):
    """ RLPlayer base on card strength"""

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

    def declare_action(self, valid_actions, hole_card, round_state, seats):

        print(f'valid actions: {valid_actions}')

        game_state = restore_game_state(round_state)
        updated_state = self.run_simulation(game_state)

  
        print(hole_card)
            
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

        community_cards = [pypokerengine_card_to_pokereval_card(card_str) for card_str in round_state["community_card"]]
        print(community_cards)
        hole_cards = [pypokerengine_card_to_pokereval_card(card_str) for card_str in hole_card]
        
        hand_strength = HandEvaluator.evaluate_hand(hole_cards, community_cards)
            
        # Complex version, with all the choice of actions
        return hand_strength
        
import numpy as np
from tensorflow import keras
from pypokerengine.players import BasePokerPlayer

class KerasPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.model = keras.models.load_model('poker_model.h5')
        self.training_data = []
        self.player_actions = {}  # Initialize an empty dictionary here
        self.hole_card = None
        
    # prepare the input 
    #encodage 
    def encode_card(self, card_str):
                 
        rank_str = '23456789TJQKA'
        suit_str = 'CDHS'

        # Extract rank and suit from the card string
        suit_char, rank_char = card_str[0], card_str[1]

        # Find the rank and suit representation
        rank_representation = rank_str.index(rank_char)
        suit_representation = suit_str.index(suit_char)

        # Combine the rank and suit representations
        encoded_card = rank_representation * 4 + suit_representation

        return encoded_card
    
    def my_hand_evaluator(self, round_state, hole_card):
         
        hole_cards = [pypokerengine_card_to_pokereval_card(card_str) for card_str in hole_card]
        community_cards = [pypokerengine_card_to_pokereval_card(card_str) for card_str in round_state["community_card"]]
        # hole_cards = [pypokerengine_card_to_pokereval_card(card_str) for card_str in hole_card]  # Remove this line
        # # Replace with this line
        hand_strength = HandEvaluator.evaluate_hand(hole_cards, community_cards)
    
        return hand_strength
    
    def extract_opponents_behavior(self, round_state, seats):
        opponents_behavior = []
        
        # Example: encode the stack size of each opponent as a percentage of the total chips in the game
        total_chips = sum([seat["stack"] for seat in seats])
        action_mapping = {'fold': 0, 'call': 1, 'raise': 2, None: -1}

        for seat in seats:
            if seat["uuid"] != self.uuid:  # Only consider opponents
                stack_size_percentage = seat["stack"] / total_chips
                opponents_behavior.append(stack_size_percentage)
                
                # Get the last action of this opponent
                last_action = self.player_actions[seat["uuid"]].get("last_action", None)
                opponents_behavior.append(action_mapping[last_action])

        
        return opponents_behavior
    
    def position_relative_to_dealer(self, round_state, seats):
        dealer_btn = round_state["dealer_btn"]
        your_index = None
        dealer_index = None

        for i, seat in enumerate(seats):
            if seat["uuid"] == self.uuid:
                your_index = i
            if i == dealer_btn:
                dealer_index = i

        # Calculate the distance between you and the dealer
        distance = your_index - dealer_index

        # If the distance is negative, add the number of players to get the correct distance
        if distance < 0:
            distance += len(seats)

        return distance
    
    def calculate_pot_odds(self, round_state):
        """
        This function calculates the pot odds for a given round state. Pot odds are the ratio of the current size of the pot
        to the cost of a contemplated call. Pot odds are used by players to determine whether a call is profitable in the 
        long run. The function takes into account the main pot amount and the amount to call, considering the action history 
        of the round.

        :param round_state: A dictionary representing the state of the round, including the pot and action history.
        :return: A float representing the pot odds.
        """

        pot_size = round_state['pot']['main']['amount']
        actions = round_state['action_histories']['preflop']
        if not actions:  # no bets made yet
            return 0
        amount_to_call = actions[-1].get('paid', 0)  # use default value of 0 if key not found
        pot_odds = amount_to_call / (pot_size + amount_to_call)
        return pot_odds
    
    def classify_players(self):
        player_types = {}
        for player_uuid, actions in self.player_actions.items():
            total_actions = sum(actions.values())
            if total_actions == 0:
                player_type = 'unknown'
            else:
                fold_ratio = actions['folds'] / total_actions
                raise_ratio = actions['raises'] / total_actions

                if fold_ratio > 0.7:
                    player_type = 'tight'
                else:
                    player_type = 'loose'

                if raise_ratio > 0.5:
                    player_type += '-aggressive'
                else:
                    player_type += '-passive'

            player_types[player_uuid] = player_type
        return player_types


    def convert_cards_to_input(self, hole_card, community_cards, round_state, seats):
        # Function to convert hole_cards and community_cards to input for your Keras model
        # This depends on how you've trained your model and the required input shape
        # For instance, you might convert the cards to a binary vector representation
        # Here is a simple example where we just flatten the cards into a single list
        # 1. Number of players
        input_vector = []
        num_players = len(seats)
        input_vector.append(num_players)

        # 2. Stack size of each player
        stack_sizes = [seat["stack"] for seat in seats]
        input_vector.extend(stack_sizes)

        # 3. Your hole cards
        print(f'hole_card: {hole_card}')
        hole_cards_vector = [self.encode_card(card_str) for card_str in hole_card]
        input_vector.extend(hole_cards_vector)

        # 4. Community cards (if any)
        # Create a fixed-size list of zeros with a length of 5, representing the maximum possible number of community cards.
        community_cards_vector = [0] * 5
        if round_state["community_card"]:
            encoded_community_cards = [self.encode_card(card_str) for card_str in round_state["community_card"]]
            for i in range(min(len(encoded_community_cards), 5)):
                community_cards_vector[i] = encoded_community_cards[i]
        input_vector.extend(community_cards_vector)


        # 5. Game state (e.g., pre-flop, flop, turn, river)
        street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3, 'showdown': 4}
        game_state = street_map[round_state['street']]
        input_vector.append(game_state)

        # 6. Opponents' behavior (e.g., betting patterns, frequencies of actions like call, fold, or raise)
        # You will need to implement a method to extract or compute these features from the game state
        opponents_behavior = self.extract_opponents_behavior(round_state, seats)
        input_vector.extend(opponents_behavior)

        # 7. Hand strength
        # You will need to implement a method to compute hand strength from your hole cards and community cards
        hand_strength = self.my_hand_evaluator(round_state, hole_card)
        input_vector.append(hand_strength)

        # 8. Position relative to the dealer button
        position_relative_to_button = self.position_relative_to_dealer(round_state, seats)
        input_vector.append(position_relative_to_button)

        # 9. Pot odds
        pot_odds = self.calculate_pot_odds(round_state)
        input_vector.append(pot_odds)

        # 10. Bet sizing history
        bet_sizing_history = []  # Placeholder
        input_vector.extend(bet_sizing_history)

        # 11. Player types (if available, e.g., tight-aggressive, loose-passive, etc.)
        player_types = self.classify_players()
        #player_types_vector = [player_types[player['uuid']] for player in seats]
        player_types_vector = [player_types[player['uuid']] if player_types[player['uuid']] != 'unknown' else -1 for player in seats]
        input_vector.extend(player_types_vector)
        

        # 12. Effective stack sizes
        effective_stack_sizes = []  # Placeholder
        input_vector.extend(effective_stack_sizes)

        # 13. Implied odds
        implied_odds = 0.0  # Placeholder
        input_vector.append(implied_odds)
        

        # 14. Fold equity
        fold_equity = 0.0  # Placeholder
        input_vector.append(fold_equity)

        return input_vector

        

    def predict_action_and_amount(self, hole_card, round_state, seats):
        #community_cards = [pypokerengine_card_to_pokereval_card(card_str) for card_str in round_state["community_card"]]
        #hole_cards = [pypokerengine_card_to_pokereval_card(card_str) for card_str in hole_card]
        community_cards = [card_str for card_str in round_state["community_card"]]
        hole_cards = [card_str for card_str in hole_card]
        print (f'hole_cards: {hole_cards}')
        # Convert cards to input format for the Keras model
        model_input = self.convert_cards_to_input(hole_cards, community_cards, round_state, seats)

        # Predict action and amount using the Keras model
        print(model_input)
        action_pred, amount_pred = self.model.predict(np.array([model_input]))

        return action_pred[0], amount_pred[0]

    def collect_training_data(self, hole_card, round_state, reward):
        # Collect training data during a game
        #community_cards = [pypokerengine_card_to_pokereval_card(card_str) for card_str in round_state["community_card"]]
        #hole_cards = [pypokerengine_card_to_pokereval_card(card_str) for card_str in hole_card]
        community_cards = [card_str for card_str in round_state["community_card"]]
        seats = round_state['seats']  # Add this line to get seats information from round_state
        hole_cards = [card_str for card_str in hole_card]
        model_input = self.convert_cards_to_input(hole_cards, community_cards, round_state, seats)  # Add seats here
        self.training_data.append((model_input, reward))

    def update_model(self, batch_size):
        # Update the model using the collected training data
        # Here, you could use your model's fit function to train on the collected data
        if len(self.training_data) > batch_size:
            batch = random.sample(self.training_data, batch_size)
            X_train, y_train = zip(*batch)
            self.model.fit(np.array(X_train), np.array(y_train), epochs=1, verbose=0) 

    # Implement the required methods for BasePokerPlayer
    def declare_action(self, valid_actions, hole_card, round_state, seats):
        
        action_pred, amount_pred = self.predict_action_and_amount(hole_card, round_state, seats)

        # Define your strategy based on action_pred and amount_pred
        # For example, you could take the action with the highest predicted probability
        action = valid_actions[np.argmax(action_pred)]
        # Ensure that the amount is within the valid range
        if isinstance(action['amount'], int):
            amount_min = action['amount']
            amount_max = action['amount']
        else:
            amount_min = action['amount']['min']
            amount_max = action['amount']['max']
        amount = min(max(amount_pred, amount_min), amount_max)
        
        # Pass round_state and seats to convert_cards_to_input
        community_cards = [pypokerengine_card_to_pokereval_card(card_str) for card_str in round_state["community_card"]]
        #model_input = self.convert_cards_to_input(hole_card, community_cards, round_state, seats)

        return action['action'], amount

    def receive_game_start_message(self, game_info):
        seats = game_info['seats']
        # Initialize the player_actions dictionary with the UUIDs of the players in the game
        self.player_actions = {player['uuid']: {'folds': 0, 'calls': 0, 'raises': 0} for player in seats}
        


    def receive_round_start_message(self, round_count, hole_card, seats):
        self.hole_card = hole_card


    def receive_street_start_message(self, street, round_state):
        pass
    
    def update_player_actions(self, action_histories):

         for street_actions in action_histories.values():
            for action in street_actions:
                player_uuid = action['uuid']
                action_type = action['action']
                if action_type in ['fold', 'call', 'raise']:
                    self.player_actions[player_uuid][action_type.lower()] += 1
                    self.player_actions[player_uuid]["last_action"] = action_type.lower()

    def receive_game_update_message(self, new_action, round_state):
        action_histories = round_state['action_histories']
        self.update_player_actions(action_histories)
        player_uuid = new_action['player_uuid']
        action_type = new_action['action']
        if action_type in ['FOLD', 'CALL', 'RAISE']:
            self.player_actions[player_uuid][action_type.lower()] += 1
            
    def get_reward(self, winners, round_state): 
        # Calculate the reward to be used in the training data
        # Here, you can define your own reward function

        my_uuid = self.uuid
        initial_stack = 0
        final_stack = 0

        for player in round_state['seats']:
            if player['uuid'] == my_uuid:
                initial_stack = player['stack']
                break

        for winner in winners:
            if winner['uuid'] == my_uuid:
                final_stack = winner['stack']
                break

        net_gain_or_loss = final_stack - initial_stack
        return net_gain_or_loss
        
        
        """
        # Get the number of raises by the player in the current round
        player_uuid = round_state["seats"][round_state["next_player"]]["uuid"]
        num_raises = self.player_actions[player_uuid]['raises']
        # Get the number of raises by the opponent in the current round
        opponent_uuid = [player['uuid'] for player in round_state["seats"] if player['uuid'] != player_uuid][0]
        opponent_raises = self.player_actions[opponent_uuid]['raises']
        # Calculate the reward
        if round_state["seats"][round_state["next_player"]]["state"] == "folded":
            reward = -1
        elif round_state["seats"][round_state["next_player"]]["state"] == "called":
            reward = 0
        elif round_state["seats"][round_state["next_player"]]["state"] == "raised":
            reward = 1
        else:
            reward = 0
        return reward
     """
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        # Collect training data for this round
        reward = self.get_reward(winners, round_state)

        self.collect_training_data(self.hole_card, round_state, reward)
        # Update the model using the collected training data
        self.update_model(batch_size=32)
