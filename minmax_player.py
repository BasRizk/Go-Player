# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 21:45:07 2021

@author: Basem Rizk
"""
from read import readInput
from write import writeOutput
from host import GO
from copy import deepcopy

import numpy as np
import random

class RandomPlayer():
    def get_input(self, go, piece_type):
        '''
        Get one input.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''        
        possible_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check = True):
                    possible_placements.append((i,j))

        if not possible_placements:
            return "PASS"
        else:
            return random.choice(possible_placements)

class GreedyPlayer():
    def get_input(self, state, piece_type):
        best_state = None
        score = - np.inf
        opponent_piece_type = 2 if piece_type == 1 else 1            
        for i in range(state.size):
            for j in range(state.size):
                if state.valid_place_check(i, j, piece_type, test_check = True):
                    # Include amount of liberty amount
                    cnt_of_neighbor_opponents =\
                        sum([ 1 if state.board[x][y] == opponent_piece_type else 0 for x, y in state.detect_neighbor(i, j)])
                    if cnt_of_neighbor_opponents > score:
                        score = cnt_of_neighbor_opponents
                        best_state = (i, j)
        return best_state
    
class MinMaxPlayer():
    
    def __init__(self, piece_type):
        self.own_piece_type = piece_type 

    def get_input(self, go_state, piece_type, depth=3):
        '''
        Get one input using Alpha-beta Search

        :param go_state: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''
        v = -np.inf
        a = None
        for action in self._actions_generator(go_state, piece_type):
            v_bar = self.min_val(self._result(go_state, piece_type, action),
                                 piece_type,
                                 -np.inf,
                                 np.inf, depth)
            if v_bar > v:
                v = v_bar
                a = action
        print('minmax choice:', v, a)
        return a
            
    
    def max_val(self, state, piece_type, alpha, beta, depth):
        if state.game_end(piece_type) or depth == 0:
            return self._eval(state)
        depth -= 1  

        v = -np.inf
        cur_piece_type = 2 if piece_type == 1 else 1            
        for action in self._actions_generator(state, piece_type):
            v = max(v, self.min_val(self._result(state, cur_piece_type, action),
                                    cur_piece_type, alpha, beta, depth))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
     
    def min_val(self, state, piece_type, alpha, beta, depth):
        if state.game_end(piece_type) or depth == 0:
            return self._eval(state)
        depth -= 1 
        
        v = np.inf
        cur_piece_type = 2 if piece_type == 1 else 1            
        for action in self._actions_generator(state, piece_type):
            v = min(v, self.max_val(self._result(state, cur_piece_type, action),
                                    cur_piece_type, alpha, beta, depth))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    
    def _result(self, state, piece_type, action):
        new_s = deepcopy(state)
        if action != 'PASS':
            i, j = action
            new_s.place_chess(i, j, piece_type)
            new_s.died_pieces = new_s.remove_died_pieces(3 - piece_type)
        else:
            new_s.previous_board = new_s.board
        return new_s
    
    def _actions_generator(self, state, piece_type):
        def insert_sort(x, arr):
            i = 0
            for i in range(len(arr)):
                if arr[i][1] < x[1]:
                    break
            if i == len(arr) - 1:
                i += 1
            arr.insert(i, x)
            
        possible_moves = []
        opponent_piece_type = 2 if piece_type == 1 else 1            
        for i in range(state.size):
            for j in range(state.size):
                if state.valid_place_check(i, j, piece_type, test_check = True):
                    # insert_sort(((i, j), len(state.ally_dfs(i, j))), possible_moves)
                    cnt_of_neighbor_opponents =\
                        sum([ 1 if state.board[x][y] == opponent_piece_type else 0 for x, y in state.detect_neighbor(i, j)])
                    insert_sort(((i, j), cnt_of_neighbor_opponents), possible_moves)
       
        for move in possible_moves:
            yield move[0]
            
        # for i in range(state.size):
        #     for j in range(state.size):
        #         if state.valid_place_check(i, j, piece_type, test_check = True):
        #             yield (i, j)
        yield 'PASS'

    def _utility(self, state, piece_type):
        winner = state.judge_winner()
        if piece_type == winner:
            return 1
        elif winner == 0:
            return 0
        else: 
            return -1
    
    
    def _eval(self, state):
        # Min count of allies of opponent per cell
        # count_of_opponent_allies = 0
        # for i in range(state.size):
        #     for j in range(state.size):
        #         if state.board[i][j] == self.own_piece_type - 1:
        #             count_of_opponent_allies += len(state.ally_dfs(i, j))
        # # state.visualize_board()
        # # print(count_of_opponent_allies)
        # # print()
        # return -count_of_opponent_allies
                
        cnt_1 = state.score(1)
        cnt_2 = state.score(2)
        side = 1 if self.own_piece_type == 1 else -1
        return side* (cnt_1 - (cnt_2 + state.komi))
        
if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    
    minmax_player = MinMaxPlayer(piece_type)
    random_player = RandomPlayer()
    greedy_player = GreedyPlayer()
    
    import datetime
    begin_time = datetime.datetime.now()
    
    num_of_random_plays = 4
    sum_of_players = sum([sum(row) for row in go.board])
    if sum_of_players == 0:
        print('Random Play')
        action = random_player.get_input(go, piece_type)
    elif sum_of_players < 3*num_of_random_plays:
        print('Greedy play')
        action = greedy_player.get_input(go, piece_type)
    else:   
        print('MinMax Play')
        action = minmax_player.get_input(go, piece_type, depth=3)
    print('Time consumed:', datetime.datetime.now() - begin_time)

    writeOutput(action)
