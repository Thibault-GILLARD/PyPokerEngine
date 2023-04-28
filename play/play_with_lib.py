from mymodule.players import FishPlayer, RLPlayer, KerasPlayer
from pypokerengine.utils.game_state_utils import restore_game_state
from pokereval.card import Card
from pokereval.hand_evaluator import HandEvaluator

from pypokerengine.api.game import setup_config, start_poker

config = setup_config(max_round=5, initial_stack=100, small_blind_amount=5)
config.register_player(name="p1", algorithm=FishPlayer())
config.register_player(name="p2", algorithm=KerasPlayer())
config.register_player(name="p3", algorithm=FishPlayer())

csv_file = open("/Users/thibaultgillard/Documents/Projet_perso/Cleare_poker/PyPokerEngine/save_result/result.csv", "a") 

total_win_p1 = 0
total_win_p2 = 0
total_win_p3 = 0
for i in range(5):
    game_result = start_poker(config, verbose=2)
    
    last_player = 0
    for player_info in game_result["players"]:
        if player_info["stack"] > last_player: 
            last_player = player_info["stack"]
            winner = player_info["name"]
    print (f'winner: {winner}')
    for player_info in game_result["players"]:
        if player_info["name"] == "p1":
            total_win_p1 += player_info["stack"]
        elif player_info["name"] == "p2":
            total_win_p2 += player_info["stack"]
        elif player_info["name"] == "p3":
            total_win_p3 += player_info["stack"]
    print(f'p1: {total_win_p1}, p2: {total_win_p2}, p3: {total_win_p3}')
    print(game_result, winner)
    # set a little time between games
    
if total_win_p2 > total_win_p1 and total_win_p2 > total_win_p3:
    csv_file.write("1\n")    
    print("p2 wins")
else:
    csv_file.write("0\n")
    print("p2 loses")
    
csv_file.close()