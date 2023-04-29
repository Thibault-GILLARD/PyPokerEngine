from mymodule.players import FishPlayer, RLPlayer, KerasPlayer
from pypokerengine.utils.game_state_utils import restore_game_state
from pokereval.card import Card
from pokereval.hand_evaluator import HandEvaluator

from pypokerengine.api.game import setup_config, start_poker

config = setup_config(max_round=35, initial_stack=100, small_blind_amount=5)
config.register_player(name="p1", algorithm=FishPlayer())
config.register_player(name="p2", algorithm=KerasPlayer())
config.register_player(name="p3", algorithm=FishPlayer())

csv_file = open("/Users/thibaultgillard/Documents/Projet_perso/Cleare_poker/PyPokerEngine/save_result/result24.csv", "a") 

total_win_p1 = 0
total_win_p2 = 0
total_win_p3 = 0
for i in range(1):
    csv_file = open("/Users/thibaultgillard/Documents/Projet_perso/Cleare_poker/PyPokerEngine/save_result/result24.csv", "a") 

    game_result = start_poker(config, verbose=2)
    
    
    
    for player_info in game_result["players"]:
        if player_info["name"] == "p1":
            earn_p1 = player_info["stack"]
        elif player_info["name"] == "p2":
            earn_p2 = player_info["stack"]
        elif player_info["name"] == "p3":
            earn_p3 = player_info["stack"]
            
    if earn_p2 > earn_p1 and earn_p2 > earn_p3:
        csv_file.write("1\n")    
        print("p2 wins")
    else:
        csv_file.write("0\n")
        print("p2 loses")   
        
        
    last_player = 0
    for player_info in game_result["players"]:
        if player_info["stack"] > last_player: 
            last_player = player_info["stack"]
            winner = player_info["name"]
    print (f'winner: {winner}')
    print(f'p1: {total_win_p1}, p2: {total_win_p2}, p3: {total_win_p3}')
    print(game_result, winner)
    
    total_win_p1 = 0
    total_win_p2 = 0
    total_win_p3 = 0
    # set a little time between games
    
    csv_file.close()