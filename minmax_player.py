# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 21:45:07 2021

@author: Basem Rizk
"""
from copy import deepcopy

import numpy as np
import datetime


class MinMaxPlayer():
    
    def __init__(self, piece_type, N=5, time_limit=8):
        self.own_piece_type = piece_type 
        self.N = N
        self.board_size = self.N**2
        self.init_mappings()
        self.time_limit = time_limit
        
    def init_expansion_aux(self):
        self.v_dict = {}
        
    def init_mappings(self):
        self.init_expansion_aux()
        self.encoding_dict = {}
    
        self.action_to_listing = {}
        self.listing_to_action = {}
        code = 0
        for i in range(self.N):
            for j in range(self.N):
                self.listing_to_action[code] = (i, j)
                self.action_to_listing[(i, j)] = code
                code += 1
        self.listing_to_action[code] = 'PASS'
        self.action_to_listing['PASS'] = code  
        
        org_oriantation = np.arange(0, self.board_size).reshape((self.N,self.N))
        equi_boards = [
                deepcopy(org_oriantation),
                np.rot90(deepcopy(org_oriantation), k=1, axes=(1,0)),    # rotate 90 right
                np.rot90(deepcopy(org_oriantation), k=1, axes=(0,1)),    # rotate 90 left
                np.rot90(deepcopy(org_oriantation), k=2, axes=(1,0)),    # rotate 180 right
                deepcopy(org_oriantation).T,                             # transpose
                np.rot90(deepcopy(org_oriantation), k=1, axes=(0,1)).T,  # flip over y-axis
                np.rot90(deepcopy(org_oriantation), k=1, axes=(1,0)).T,  # flip over x-axis
                np.rot90(deepcopy(org_oriantation), k=2, axes=(1,0)).T   # flip 180 rotation
            ]
        
        self.sym_dict_forward = [{} for i in range(len(equi_boards))]
            
        for d_i in range(len(equi_boards)):
            for i in range(self.N):
                for j in range(self.N):
                    c = org_oriantation[i][j]
                    self.sym_dict_forward[d_i][c] = equi_boards[d_i][i][j]
    
    def retrieve_v(self, board_state, depth):
        def encode(board_state, depth):
            encoding = self.encoding_dict.get(str(board_state) + str(depth))
            if encoding is not None:
                return encoding
            
            arr2d = board_state
            arr = np.array(arr2d).reshape(len(arr2d)**2)
            
            all_encodings = []
            for mapping in self.sym_dict_forward:
                encoding = ""
                for i in range(len(arr)):
                    encoding += str(arr[mapping[i]])
                all_encodings.append(encoding)
            
            all_encodings = [int(encoding) for encoding in all_encodings]
            encoding = str(all_encodings[np.argmin(all_encodings)]) + str(depth)
            self.encoding_dict[str(board_state)] = encoding
            return encoding
        
        board_state_encoded = encode(board_state, depth)
        v_value = self.v_dict.get(board_state_encoded)
        
        return board_state_encoded, v_value
        
    def record_v(self, board_state_encoded, new_v):
        self.v_dict[board_state_encoded] = new_v
    
    
    def is_time_out(self):
        time_elapsed = (datetime.datetime.now() - self.process_start_time).total_seconds()
        if time_elapsed > self.time_limit and self.best_action is not None:
            print('Exceeded Timelimit')
            return True
        return False
    

    def get_input(self, go_state, piece_type, depth=3):
        self.process_start_time = datetime.datetime.now()
        self.best_action = None
        print('Wanting to reach depth %d' % depth)
        start_depth = min(depth, 3)
        print('Applying Iterative Deepening from depth %d' % start_depth)
        for i in range(start_depth, depth+1):
            print('Trying depth', i)
            self.init_expansion_aux()
            yet_best_action = self.get_input_internal(go_state, piece_type, depth=i)
            time_elapsed = (datetime.datetime.now() - self.process_start_time).total_seconds()
            print('Time elapsed =', time_elapsed)
            if yet_best_action is None:
                print(': Failed')
                break
            print('Depth %d: Succeeded' % i)
            self.best_action = yet_best_action
        return self.best_action
    
    def get_input_internal(self, go_state, piece_type, depth=3):
        '''
        Get one input using Alpha-beta Search

        :param go_state: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''
        self.depth = depth - 1
        self.num_branches = [0 for i in range(depth)]
        self.num_cut_branches = [0 for i in range(depth)]
        depth -= 1
        
        v = -np.inf
        a = None
        
        for action in self._actions_generator(go_state, piece_type):

            resulting_state = self._result(go_state, piece_type, action)
            board_state_encoded, v_bar = self.retrieve_v(resulting_state.board, depth)
            
            if v_bar is None:
                self.num_branches[self.depth-depth] += 1
                v_bar = self.min_val(resulting_state,
                                     piece_type,
                                     -np.inf,
                                     np.inf, depth)
                self.record_v(board_state_encoded, v_bar)
            else:
                self.num_cut_branches[self.depth-depth] += 1
            
            if self.is_time_out():
                return
            
            if v_bar > v:                
                v = v_bar
                a = action
        print('\nPlayed..')
        print('minmax choice:', v, a)
        print('num of branches', self.num_branches)
        print('# of cut sym-branches are', self.num_cut_branches)

        # print('%d branches at depth %d' % (num_branches, d))
        return a
            
    
    def max_val(self, state, piece_type, alpha, beta, depth):
        if state.game_end(piece_type) or depth == 0:
            return self._eval(state)
        depth -= 1  
            
        v = -np.inf
        cur_piece_type = 2 if piece_type == 1 else 1            
        for action in self._actions_generator(state, piece_type):
            
            resulting_state = self._result(state, cur_piece_type, action)
            board_state_encoded, v_bar = self.retrieve_v(resulting_state.board, depth)
            
            if v_bar is None:
                self.num_branches[self.depth-depth] += 1
                v_bar = self.min_val(resulting_state,
                                     cur_piece_type, alpha, beta, depth)
                if v_bar is None:
                    # TIMEOUT
                    return
                v = max(v, v_bar)
                self.record_v(board_state_encoded, v)
            else:
                v = max(v, v_bar)
                self.num_cut_branches[self.depth-depth] += 1
            
            if self.is_time_out():
                return
            
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
            
            resulting_state = self._result(state, cur_piece_type, action)
            board_state_encoded, v_bar = self.retrieve_v(resulting_state.board, depth)
            
            if v_bar is None:
                self.num_branches[self.depth-depth] += 1
                v_bar = self.max_val(resulting_state,
                                        cur_piece_type, alpha, beta, depth)
                if v_bar is None:
                    return
                v = min(v, v_bar)
                self.record_v(board_state_encoded, v)
            else:
                v = min(v, v_bar)
                self.num_cut_branches[self.depth-depth] += 1
            
            if self.is_time_out():
                return
            
            if v <= alpha:
                return v
            beta = min(beta, v)
        # print('v is', v)
        # state.visualize_board()

        return v
    
    def _result(self, state, piece_type, action):
        new_s = deepcopy(state)
        if action != 'PASS':
            i, j = action
            new_s.place_chess(i, j, piece_type)
            new_s.died_pieces = new_s.remove_died_pieces(3 - piece_type)
        else:
            new_s.previous_board = deepcopy(new_s.board)
        return new_s
    
    def _actions_generator(self, state, piece_type):
        def insert_sort(x, arr):
            comparable_index = 1
            i = 0
            while i < len(arr):

                if arr[i][comparable_index] < x[comparable_index]:
                    if comparable_index >= len(x) - 1:
                        break
                    comparable_index += 1
                    continue
                i += 1
                
            if i == len(arr) - 1:
                i += 1
            arr.insert(i, x)
            
        possible_moves = []
        opponent_piece_type = 2 if piece_type == 1 else 1    
        # state.visualize_board()        
        for i in range(state.size):
            for j in range(state.size):
                if state.valid_place_check(i, j, piece_type, test_check = True):
                    cnt_of_neighbor_opponents = len(self.around_dfs(state, i, j, opponent_piece_type))
                    #cnt_of_neighbor_allies = len(state.ally_dfs(i, j))
                    #cnt_of_neighbor_opponents = len(self.detect_neighbor_per_type(state, i, j, opponent_piece_type))
                    cnt_of_neighbor_empty = len(self.detect_neighbor_per_type(state, i, j, 0))
                    # print('Cell: %d, %d, with %d neighbor opponents' % (i, j, cnt_of_neighbor_opponents))
                    insert_sort(
                        ((i, j), cnt_of_neighbor_empty, cnt_of_neighbor_opponents),
                        possible_moves)
        # print()
        
        # print(possible_moves)
        for move in possible_moves:
            yield move[0]
            
        # for i in range(state.size):
        #     for j in range(state.size):
        #         if state.valid_place_check(i, j, piece_type, test_check = True):
        #             yield (i, j)
        yield 'PASS'


    def detect_neighbor_per_type(self, state, i, j, piece_type):
            board = state.board
            neighbors = state.detect_neighbor(i, j)  # Detect neighbors
            group = []
            # Iterate through neighbors
            for piece in neighbors:
                # Add to allies list if having the same color
                if board[piece[0]][piece[1]] == piece_type:
                    group.append(piece)
            return group
        
    def around_dfs(self, state, i, j, piece_type):
        '''
        Using DFS to search for all around of specific type.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: piece_type
        :return: a list containing the all allies row and column (row, column) of position (i, j).
        '''
        stack = [(i, j)]  # stack for DFS serach
        members = []  # record positions during the search
        while stack:
            piece = stack.pop()
            if piece != (i, j):
                members.append(piece)
            neighbor_pieces = self.detect_neighbor_per_type(state, 
                                                       piece[0], piece[1],
                                                       piece_type)
            for piece in neighbor_pieces:
                if piece not in stack and piece not in members:
                    stack.append(piece)
        return members
    
    # def _utility(self, state, piece_type):
    #     winner = state.judge_winner()
    #     if piece_type == winner:
    #         return 1
    #     elif winner == 0:
    #         return 0
    #     else: 
    #         return -1
    
    def _cnt_high_empty_neighbors(self, state, high=2):
        cnt = 0
        for i in range(state.size):
            for j in range(state.size):
                if state.board[i][j] == self.own_piece_type:
                     if len(self.detect_neighbor_per_type(state, i, j, 0)) >= high:
                         cnt += 1
        return cnt
                
    def _eval(self, state):
        cnt_1 = state.score(1)
        cnt_2 = state.score(2)
        side = 1 if self.own_piece_type == 1 else -1
        # board_diff = state.board - state.previous_board
        return side* (cnt_1 - (cnt_2 + state.komi))
        # + (12 - GameStats.get_game_step())
        
    def visualize_board(self, board):
        print('-' * len(board) * 2)
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == 0:
                    print(' ', end=' ')
                elif board[i][j] == 1:
                    print('X', end=' ')
                else:
                    print('O', end=' ')
            print()
        print('-' * len(board) * 2)