# -*- coding: utf-8 -*-

import numpy as np
import random
import json

from copy import deepcopy
from read import readInput
from write import writeOutput
from host import GO

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


WIN_REWARD = 1
DRAW_REWARD = 0
LOSS_REWARD = -1

horizontal_sym_dict = {
    1:21, 2:16, 3:11, 4:6, 5:1, 
    6:22, 7:17, 8:12, 9:7, 10:2, 
    11:23, 12:18, 13:13, 14:8, 15:3,
    16:24, 17:19, 18:14, 19:9, 20:4,
    21:25, 22:20, 23:15, 24:10, 25:5
    }
 
class QLearner:
    
    def __init__(self, N=5, alpha=0.5, gamma=0.5, initial_reward=-0.5):
        self.N = 5
        self.type = 'qlearner'
        self.alpha = 0.5
        self.gamma = 0.5
        self.q_values = {}
        self.initial_reward = -0.5
        self.play_log = []
        self.init_mapping()
        self.training = False
        
    def init_mapping(self):
        self.action_to_listing = {}
        self.listing_to_action = {}
        code = 0
        for i in range(5):
            for j in range(5):
                self.listing_to_action[code] = (i, j)
                self.action_to_listing[(i, j)] = code
                code += 1
        self.listing_to_action[code] = 'PASS'
        self.action_to_listing['PASS'] = code    
    
    
    def get_input(self, state, piece_type, epsilon=0.1):

        self.own_piece_type = piece_type
        # TODO Exploration Vs Exploitation
        
        # select best q-value @ current-state
        relevant_q_values = self._state_q_values(state.board)

        # if selected move not valid, turn move q-value to lose   
        for i in range(state.size):
            for j in range(state.size):
                if not state.valid_place_check(
                        i, j,
                        self.own_piece_type,
                        test_check = True
                        ):
                    relevant_q_values[self.action_to_listing[(i,j)]] = -1
        
        # Follow epsilon greedy policy
        selected_policy_listing = None
        explore_new_policies = random.random() > epsilon
        if self.training and explore_new_policies:
            other_policy_listings = np.argwhere(
                (relevant_q_values!=relevant_q_values.max()) & (relevant_q_values!= -1)
                )
            if len(other_policy_listings) > 0:
                selected_policy_listing = random.choice(other_policy_listings)[0]
        if selected_policy_listing is None or not self.training or not explore_new_policies :
            optimal_policy_listings = np.argwhere(relevant_q_values==relevant_q_values.max())
            selected_policy_listing = random.choice(optimal_policy_listings)[0]
        
        action = self.listing_to_action[selected_policy_listing]
        # keep track of movements to learn later from them
        self.play_log.append([deepcopy(state.board), action])
        
        return action
    
    # def _probalities_of_actions(self, possible_actions, board_state, T):
    #     state_q_values = self._state_q_values(self, board_state)
    #     q_s_a_list = np.zeros(len(possible_actions))
    #     for a in possible_actions:
    #         q_s_a_list = np.exp(
    #             np.e,
    #             state_q_values[self.actions_to_listing[a]]/T
    #             )
    #     return q_s_a_list/q_s_a_list.sum()
    
    def _state_q_values(self, board_state):
        def encode(arr2d):
            def cost(str25chars):
                cost = 0
                for i, c in zip(range(25, 0, -1), str25chars):
                    cost += (i*int(c)) 
                return cost
                
            arr2d = np.array(arr2d)
            encodings = []
            for arr in [arr2d, np.rot90(arr2d, k=1, axes=(1, 0))]:
                encoding1 = encoding2 = ""
                for row in arr:
                    for c in row:
                        encoding1 = encoding1 + str(c)
                        encoding2 = str(c) + encoding2
                encodings.append(encoding1)
                encodings.append(encoding2)
            costs = [cost(encoding) for encoding in encodings]
            index_of_encoding = np.argmax(costs)
            return index_of_encoding, encodings[index_of_encoding]
                    
        def swap_with_dict(arr, n=25):
            for i in range(n/2):
                j = horizontal_sym_dict[i] 
                arr[i], arr[j] = arr[j], arr[i]
            return arr
        
        def reverse(arr, n=25):
            return arr[:n][::-1]

        fit_q_values = {
            0 : lambda x: x,
            1 : lambda x: reverse(deepcopy(x)),
            2 : lambda x: swap_with_dict(deepcopy(x)),
            3 : lambda x: reverse(swap_with_dict(deepcopy(x)))
            }
            
        index_of_encoding, board_state_encoded = encode(board_state)
        relevant_q_values = self.q_values.get(board_state_encoded)
        if relevant_q_values is not None:
            relevant_q_values = fit_q_values[index_of_encoding](relevant_q_values)
        else:
            # if updated q-value not available, use initial values
            relevant_q_values = np.ones(26)*self.initial_reward
            # considering pass option, illegalize other out of boundaries
            self.q_values[board_state_encoded] = relevant_q_values
        return np.asarray(relevant_q_values)
    
    def train(self, go, epochs, reset=False, save_freq=100, ckpt='q_values', verbose=0):

        from random_player import RandomPlayer
        import os
        
        go.verbose = True if verbose == 2 else False
            
        if not reset and os.path.isfile(''.join([ckpt, 'json'])):
            self.q_values = self.read(filename='q_values')
            
        last_ckpt_term = ckpt.split('_')[-1] 
        try:
            starting_epoch = int(last_ckpt_term)
        except:                
            starting_epoch = 0
            
    
        opponent_player = RandomPlayer()

        
        for i in range(starting_epoch, epochs):
            print('Training...  %d / %d' % (i+1, epochs), end='\r')
    
            go.X_move = True # X chess plays first
            go.died_pieces = [] # Intialize died pieces to be empty
            go.n_move = 0 # Trace the number of moves
            
            if i % 2 == 0:
                winner = go.play(self, opponent_player, verbose=verbose)    
            else:
                winner = go.play(opponent_player, self, verbose=verbose)    
        
            self.learn(winner, verbose)
            if i > 0 and i % save_freq == 0:
                filename = 'q_values_' + str(i)
                self.save_json(self.q_values, filename=filename)

        
        print()
        self.save_json(self.q_values, filename='q_values_' + str(epochs))

    def learn(self, winner, verbose):
        if verbose:
            print('Play-log consists of %d states' % len(self.play_log))
            print('Q-values consists of %d tables' % len(self.q_values))

        board_state, action = self.play_log.pop()
        if verbose:
            print('Last board state in log:')
            print(board_state)
            
        current_state_q_values = self._state_q_values(board_state)
        if verbose:
            print('Before update - corresponding q_values:')
            print(current_state_q_values)
            
        reward = WIN_REWARD if winner == self.own_piece_type else DRAW_REWARD if winner == 0 else LOSS_REWARD
        current_state_q_values[self.action_to_listing[action]] = reward
        
        if verbose:
            print('Reward is %f' % reward)
            print('After update - corresponding q_values:')
            print(current_state_q_values)
            
        max_q = np.max(current_state_q_values)
        # print('Hold max_q value')
        # propagate rewards back from result: update move's q-value per state back to the start of the game
        while len(self.play_log) > 0:
            board_state, action = self.play_log.pop()
            current_state_q_values = self._state_q_values(board_state)
            
            current_q = current_state_q_values[self.action_to_listing[action]]
            current_state_q_values[self.action_to_listing[action]] =\
                            (1-self.alpha)*(current_q) +\
                            self.alpha*(self.gamma*max_q)
                            
            max_q = np.max(current_state_q_values)
            
    def save_json(self, obj, filename='q_values'):
        class Encoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()               
                # Let the base class default method raise the TypeError
                return json.JSONEncoder.default(self, obj)
        
        with open('.'.join([filename, 'json']), 'w') as f:
            json.dump(obj, f, cls=Encoder)        
        print('Saved', filename)
            
    def read(self, filename='q_values'):
        return json.load(open('.'.join([filename, 'json'])))


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
    parser.add_argument('--play', '-p', type=bool, help='play given input file', default=False)
    parser.add_argument('--train', '-t', type=int, help='learn q-values', default=0)
    parser.add_argument('--save_freq', '-s', type=int, help='freq of saving q_values', default=1)
    parser.add_argument('--ckpt', '-c', type=str, help='ckpt filename', default='q_values')
    parser.add_argument('--reset', '-r', type=bool, help='reset q-values', default=False)
    parser.add_argument('--verbose', '-v', type=int, help='print board', default=0)
    args = parser.parse_args()
    
    N = 5
    go = GO(N)
    q_learner = QLearner()
    
    args.train = 5
    if args.train > 0:
        q_learner.training = True
        q_learner.train(go, 
                        epochs=args.train, reset=args.reset, 
                        save_freq=args.save_freq, ckpt=args.ckpt,
                        verbose=args.verbose)
    else:
        import datetime
        begin_total_time = datetime.datetime.now()
        piece_type, previous_board, board = readInput(N)
        go.set_board(piece_type, previous_board, board)
        begin_reading_time = datetime.datetime.now()
        q_values_filename = 'q_values_400000'
        q_learner.q_values = q_learner.read(filename=q_values_filename)
        q_values = q_learner.q_values
        action = q_learner.get_input(go, piece_type)
        writeOutput(action)
        print('Total time consumed:', datetime.datetime.now() - begin_total_time)
        print('Reading time consumed:', datetime.datetime.now() - begin_reading_time)
        print('Q_values file used is', q_values_filename)




    