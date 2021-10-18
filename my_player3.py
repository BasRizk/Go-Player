# -*- coding: utf-8 -*-

import numpy as np
import random
import json

from copy import deepcopy
from read import readInput
from write import writeOutput
from host import GO

from datetime import datetime
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)


WIN_REWARD = 1
DRAW_REWARD = 0.5
LOSS_REWARD = 0
    
class QValues:

    def __init__(self, N, initial_value, verbose=0):
        self.N = N
        self.board_size = N*N
        self.q_tables = {}
        self.initial_reward = initial_value
        self.init_mappings()
        self.recent_board_state = None
        self.recent_state_q_values = None
        self.index_of_encoding = None
        self.recent_board_encoding = None
        self.verbose = verbose


    def init_mappings(self):
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
        
        board_sample = np.arange(0, self.board_size).reshape((self.N,self.N))
        equi_boards = [
                deepcopy(board_sample),
                np.rot90(deepcopy(board_sample), k=1, axes=(1,0)),    # rotate 90 right
                np.rot90(deepcopy(board_sample), k=1, axes=(0,1)),    # rotate 90 left
                np.rot90(deepcopy(board_sample), k=2, axes=(1,0)),    # rotate 180 right
                deepcopy(board_sample).T,                             # transpose
                np.rot90(deepcopy(board_sample), k=1, axes=(0,1)).T,  # flip over y-axis
                np.rot90(deepcopy(board_sample), k=1, axes=(1,0)).T,  # flip over x-axis
                np.rot90(deepcopy(board_sample), k=2, axes=(1,0)).T   # flip 180 rotation
            ]
        
        self.sym_dict_forward = [{} for i in range(len(equi_boards))]
        self.sym_dict_backward = [{} for i in range(len(equi_boards))]
            
        for d_i in range(len(equi_boards)):
            for i in range(self.N):
                for j in range(self.N):
                    c = board_sample[i][j]
                    x, y = np.asarray(np.where(c == equi_boards[d_i]))
                    x, y = x[0], y[0]
                    self.sym_dict_backward[d_i][c] = board_sample[x][y]
                    self.sym_dict_forward[d_i][c] = equi_boards[d_i][x][y]

    def visualize_q_table(self, board_state, encoding_inc=None, verbose=0):
        def fit_table(table):
            fit_table = np.zeros(len(table) - 1)
            for i in range(len(fit_table)):
                fit_table[i] =\
                    table[self.sym_dict_backward[self.index_of_encoding][i]]
                if self.verbose > 2:
                    print('fit_table at %d moved to %d with value=%.2f' 
                          % (i, self.sym_dict_backward[self.index_of_encoding][i],
                          table[self.sym_dict_backward[self.index_of_encoding][i]]))
            return fit_table
            
        def print_on_board(fit_q_table):
            print('-' * self.N * 6)
            for i in range(self.N):
                for j in range(self.N):
                    if fit_q_table[i][j] == -1:
                        print('     ', end='|')
                    else:
                        print('%5s' % "{:.2f}".format(fit_q_table[i][j]), end='|')
                print()
            print('PASS =', self.recent_state_q_values[-1])
            print('-' * self.N * 6)
        
        self._ensure_recent_state(board_state)
        if encoding_inc is not None and self.index_of_encoding not in encoding_inc:
            return
        
        if self.verbose > 3:
            print('encoding is %d' % self.index_of_encoding)
            print('before transformation')
            print(self.recent_state_q_values)
            print()
            print_on_board(deepcopy(self.recent_state_q_values)[:25].reshape((self.N, self.N)))
        
        fit_q_table = fit_table(self.recent_state_q_values).reshape((self.N, self.N))
        print('after transformation')
        print_on_board(fit_q_table)
        
    
    def _state_q_values(self, board_state):      
        def encode(arr2d):
            arr = np.array(arr2d).reshape(len(arr2d)**2)
            encodings = []
            for mapping in self.sym_dict_forward:
                encoding = ""
                for i in range(len(arr)):
                    encoding += str(arr[mapping[i]])
                encodings.append(encoding)
            
            encodings = [int(encoding) for encoding in encodings]
            index_of_encoding = np.argmin(encoding)
            return index_of_encoding, str(encodings[index_of_encoding])
        
        index_of_encoding, board_state_encoded = encode(board_state)
        relevant_q_values = self.q_tables.get(board_state_encoded)
        if relevant_q_values is None:
            # 25 cells + 1 considering pass option
            relevant_q_values = np.ones(26)*self.initial_reward
            self.q_tables[board_state_encoded] = relevant_q_values
    
        return index_of_encoding, board_state_encoded, np.asarray(relevant_q_values), 
    
    def _ensure_recent_state(self, board_state):
        if board_state != self.recent_board_state:
            self.recent_board_state = deepcopy(board_state)
            self.index_of_encoding, self.recent_board_encoding, self.recent_state_q_values =\
                self._state_q_values(board_state)
    
    def _to_fit_listing(self, action):        
        listing = self.action_to_listing[action]
        if listing != 25:
            # todo check this again
            return self.sym_dict_forward[self.index_of_encoding][listing]
        return listing
    
    def _to_fit_action(self, listing):
        action = self.listing_to_action[listing]
        return self.listing_to_action[self._to_fit_listing(action)]

    def __getitem__(self, selector):
        board_state, action = selector
        self._ensure_recent_state(board_state)
        fit_listing = self._to_fit_listing(action)
        return self.recent_state_q_values[fit_listing]        
    
    def __setitem__(self, selector, value):
        board_state, action = selector
        self._ensure_recent_state(board_state)
        fit_listing = self._to_fit_listing(action)
        self.recent_state_q_values[fit_listing] = value
    
    def get_suboptimal_policy(self, board_state):
        self._ensure_recent_state(board_state)
        other_policy_listings =\
            np.argwhere(
                (self.recent_state_q_values!=self.recent_state_q_values.max()) &\
                    (self.recent_state_q_values!= -1)
                )
        if len(other_policy_listings) == 0:
            return self.get_optimal_policy(board_state)
        selected_policy_listing = random.choice(other_policy_listings)[0]
        
        return self._to_fit_action(selected_policy_listing)
    
    def get_optimal_policy(self, board_state):
        self._ensure_recent_state(board_state)
        optimal_policy_listings =\
            np.argwhere(self.recent_state_q_values==self.recent_state_q_values.max())
        selected_policy_listing = random.choice(optimal_policy_listings)[0]
        return self._to_fit_action(selected_policy_listing)

    def max_q(self, board_state):
        self._ensure_recent_state(board_state)
        return np.max(self.recent_state_q_values)
   
    # def _probalities_of_actions(self, possible_actions, board_state, T):
    #     state_q_values = self._state_q_values(self, board_state)
    #     q_s_a_list = np.zeros(len(possible_actions))
    #     for a in possible_actions:
    #         q_s_a_list = np.exp(
    #             np.e,
    #             state_q_values[self.actions_to_listing[a]]/T
    #             )
    #     return q_s_a_list/q_s_a_list.sum()
    
    def save_into_splits(self, prefix='q_values', split=100000):
        class Encoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()               
                # Let the base class default method raise the TypeError
                return json.JSONEncoder.default(self, obj)
        
        begin_time = datetime.now()
        
        filename = '.'.join([prefix, 'json'])
        with open(filename, 'w') as f: 
            num_of_splits = len(self.q_tables)//split + 1
            splits = [{} for i in range(num_of_splits)]
            count = 0
            for k, v in self.q_tables:
                split_i = count // split
                splits[split_i][k] = v
                count += 1
                
            json.dump(self.q_tables, f, cls=Encoder)    
        
        end_time = datetime.now() - begin_time
        print('Saved %s in' % filename, end_time)
        

    
    def save(self, piece_type, epoch):
        class Encoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()               
                # Let the base class default method raise the TypeError
                return json.JSONEncoder.default(self, obj)
        def ensure_directory(piece_type):
            if not os.path.exists(str(piece_type)):
                os.makedirs(str(piece_type))
        begin_time = datetime.now()
        ensure_directory(piece_type)
        filepath = str(piece_type) + '/' +  '_'.join(['q_values', str(epoch)]) + '.json'
        with open(filepath, 'w') as f:                 
            json.dump(self.q_tables, f, cls=Encoder)    
        end_time = datetime.now() - begin_time
        print('Saved %s in' % filepath, end_time)

    def load(self, piece_type, epoch):
        begin_time = datetime.now()
        filepath = str(piece_type) + '/' +  '_'.join(['q_values', str(epoch)]) + '.json'
        if os.path.isfile(filepath):
            self.q_tables = json.load(open(filepath))
            end_time = datetime.now() - begin_time
            print('Loaded %s in' % filepath, end_time)
            return int(epoch)
        return 0
        
class QLearner:
    
    def __init__(self, N=5, alpha=0.5, gamma=0.5, initial_reward=-0.5,
                 training=False):
        self.N = 5
        self.type = 'qlearner'
        self.alpha = 0.5
        self.gamma = 0.5
        self.q_values = QValues(N, initial_reward)
        self.play_log = []
        self.epsilon = 0.1
    
    def get_input(self, state, piece_type):
        
        self.own_piece_type = piece_type
        
        # if selected move not valid, turn move q-value to lose   
        for i in range(state.size):
            for j in range(state.size):
                if not state.valid_place_check(
                        i, j,
                        self.own_piece_type,
                        test_check = True
                        ):
                    self.q_values[(state.board, (i, j))] = -1
                # else:
                    # print((i, j), end="  ")
        # print()
        
        # TODO Exploration Vs Exploitation
        # Follow epsilon greedy policy
        explore_new_policies = random.random() < self.epsilon
        if explore_new_policies:
            action = self.q_values.get_suboptimal_policy(state.board)
        else:
            action = self.q_values.get_optimal_policy(state.board)
            
        # keep track of movements to learn later from them
        self.play_log.append([deepcopy(state.board), action])
        
        # self.q_values.visualize_q_table(state.board)

        return action
    
    def offline_play(self, go, train, epochs, train_piece_type=0,
              reset=False, save_freq=100, ckpt='q_values',
              epsilon=0.1, verbose=0):
        def reset_game(go):
            go.X_move = True # X chess plays first
            go.died_pieces = [] # Intialize died pieces to be empty
            go.n_move = 0 # Trace the number of moves
            
        from random_player import RandomPlayer
        
        self.q_values.verbose=verbose
        go.verbose = True if verbose == 2 else False
        
        starting_epoch = 0
        if not reset:
            starting_epoch = self.q_values.load(train_piece_type, ckpt)
                
        opponent_player = RandomPlayer()

        play = {
            0: lambda i, x, y: play[1](i, x, y) if i%2 == 0 else play[2](i, x, y),
            1: lambda _, x, y: go.play(x, y, verbose=verbose),
            2: lambda _, x, y: go.play(y, x, verbose=verbose)
            }
        
        winnings_count = recent_winnings_count = 0
        for i in range(starting_epoch, epochs):
            print('Training...  %d / %d with %d winnings' % (i+1, epochs, winnings_count), end='\r')
            if verbose:
                print()
            self.epsilon = (1 - (starting_epoch/epochs))*epsilon
            
            reset_game(go)
            
            winner = play[train_piece_type](i, self, opponent_player)
            winnings_count += 1 if winner == self.own_piece_type else 0
            recent_winnings_count += 1 if winner == self.own_piece_type else 0

            if train:
                self.learn(go, verbose)
                if i > 0 and i % save_freq == 0:
                    self.q_values.save(train_piece_type, i)
                    recent_winnings_count = 0
                    print('\n', 'Winnings recently = %d' % recent_winnings_count)

        if train:
            self.q_values.save(train_piece_type, i)
    
    def learn(self, state, verbose):
        def reward_function(state):
            cnt_1 = state.score(1)
            cnt_2 = state.score(2)
            side = 1 if self.own_piece_type == 1 else -1
            max_score = (self.N**2 + state.komi)
            return (side/max_score)*(cnt_1 - (cnt_2 + state.komi))
        
        if verbose:
            print('Play-log consists of %d states' % len(self.play_log))
            print('Q-values consists of %d tables' % len(self.q_values.q_tables))

        board_state, action = self.play_log.pop()
        if verbose:
            print('Last board state in log:')
            print(board_state)
                    
        # reward = WIN_REWARD if winner == self.own_piece_type else DRAW_REWARD if winner == 0 else LOSS_REWARD
        reward = reward_function(state)
        self.q_values[(board_state, action)] = reward
        
        if verbose:
            print('Reward is %f' % reward)
            
        max_q = self.q_values.max_q(board_state)

        # propagate rewards back from result: update move's q-value per state back to the start of the game
        while len(self.play_log) > 0:
            board_state, action = self.play_log.pop()
                            
            current_q = self.q_values[(board_state, action)]
            self.q_values[(board_state, action)] =\
                (1-self.alpha)*(current_q) + self.alpha*(self.gamma*max_q)
            max_q = self.q_values.max_q(board_state)
    
    def load_q_values(self, piece_type=0, epochs=''):
        self.q_values.load(piece_type, epochs)
        
def visualize_board_encoding(board_encoding, n=5):
    col = 0
    for c in board_encoding:
        c = 'X' if c == '1' else 'O' if c == '2' else  ' '
        print(c, end='|')
        col += 1 
        if col == n:
            print("\n----------")
            col = 0
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t', type=bool, help='learn q-values', default=False)
    parser.add_argument('--epochs', '-i', type=int, help='num of training epochs', default=0)
    parser.add_argument('--piece_type', '-pt', type=int, help='piece_type_to_train', default=0)
    parser.add_argument('--alpha', '-a', type=float, help='alpha - learning rate', default=0.1)
    parser.add_argument('--gamma', '-g', type=float, help='gamma - discount factor', default=0.9)
    parser.add_argument('--initial_reward', '-ir', type=float, help='initial q-value', default=-0.5)
    parser.add_argument('--epsilon', '-e', type=float, help='probability of exploring new_policies', default=0.1)
    parser.add_argument('--save_freq', '-s', type=int, help='freq of saving q_values', default=1)
    parser.add_argument('--ckpt', '-c', type=str, help='ckpt filename', default='q_values')
    parser.add_argument('--reset', '-r', type=bool, help='reset q-values', default=False)
    parser.add_argument('--verbose', '-v', type=int, help='print board', default=0)
    args = parser.parse_args()
    
    N = 5
    go = GO(N)
    q_learner = QLearner(alpha=args.alpha, gamma=args.gamma, initial_reward=args.initial_reward)
    
    if args.epochs > 0:
        if args.train:
            filename = str(args.alpha) + 'alpha' + str(args.gamma) + 'gamma' +\
                    str(args.initial_reward) + 'intial_value'
            if not os.path.exists(filename):
                os.makedirs(filename)
                
        q_learner.offline_play(go, train = args.train,
                        epochs=args.epochs, train_piece_type=args.piece_type,
                        reset=args.reset, save_freq=args.save_freq,
                        ckpt=args.ckpt,
                        epsilon=args.epsilon,
                        verbose=args.verbose)
    else:
        begin_total_time = datetime.now()
        piece_type, previous_board, board = readInput(N)
        go.set_board(piece_type, previous_board, board)
        begin_reading_time = datetime.now()
        q_learner.load_q_values(piece_type=piece_type, epochs=60000)
        # q_values = q_learner.q_values
        action = q_learner.get_input(go, piece_type)
        writeOutput(action)
        print('Total time consumed:', datetime.now() - begin_total_time)
        print('Reading time consumed:', datetime.now() - begin_reading_time)




    