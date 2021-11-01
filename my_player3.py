# -*- coding: utf-8 -*-
"""

Structre of the file is adapted from initial format that was already provided
by the CSCI670 Staff of FALL2021 at the University of Southern California

"""

from read import readInput
from write import writeOutput
from host import GO
from minmax_player import MinMaxPlayer
from random_player import RandomPlayer
from greedy_player import GreedyPlayer
from GameStats import GameStats


if __name__ == '__main__':
        
    N = 5
    piece_type, previous_board, board = readInput(N)

    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    
    
    minmax_player = MinMaxPlayer(piece_type, N=N, time_limit=9)
    random_player = RandomPlayer()
    greedy_player = GreedyPlayer()
    
    import datetime
    begin_time = datetime.datetime.now()
    
    num_of_random_plays = 2
    sum_of_players = sum([sum(row) for row in go.board])
    num_of_opponents = sum([1 if 3-piece_type == i else 0 for row in go.board for i in row])

    if sum_of_players <= 1:
        GameStats.init_game_step()
    else:
        GameStats.increment_game_step()
    
    print('Current-Step:', GameStats.get_game_step())
    if num_of_opponents <= 1:
        i = j = (N//2)
        if not go.valid_place_check(i, j, piece_type, test_check=True):
            print('Random Play')
            action = random_player.get_input(go, piece_type)
        else:
            print('Optimal Start Play at', i, j)
            action = (i, j)
    #elif GameStats.get_game_step() <= num_of_random_plays:
    #    print('Greedy play')
    #    action = greedy_player.get_input(go, piece_type)
    else:
        offset = 26 if piece_type == 1 else 27
        current_step = GameStats.get_game_step()
        print('Sum of Player:', sum_of_players)
        print('Sum of Opponents:', num_of_opponents)
        max_depth = 6 if num_of_opponents >= 9 else 5 if num_of_opponents >= 8 else 4 if num_of_opponents >= 4 else 3
        #max_depth = 6 if sum_of_players >= 10 else 5 if sum_of_players >= 8 else 4 if sum_of_players >= 5 else 3
        #max_depth = 4
        depth = min(offset - current_step*2, max_depth)
        #depth=4
        print('MinMax Play with Depth', depth)
        action = minmax_player.get_input(go, piece_type, depth=depth)
    print('Time consumed:', datetime.datetime.now() - begin_time)
    
    writeOutput(action)