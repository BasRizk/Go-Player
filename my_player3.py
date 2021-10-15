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
DRAW_REWARD = 0.5
LOSS_REWARD = 0
    
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
    
    def get_input(self, state, piece_type):

        self.own_piece_type = piece_type
        # TODO Exploration Vs Exploitation
        
        # select best q-value @ current-state
        relevant_q_values = self._state_q_values(state.board)

        # TODO include max q value selection in this loop if nessesaery
        # if selected move not valid, turn move q-value to lose   
        for i in range(state.size):
            for j in range(state.size):
                if not state.valid_place_check(
                        i, j,
                        self.own_piece_type,
                        test_check = True
                        ):
                    relevant_q_values[self.action_to_listing[(i,j)]] = -1            
        
        optimal_policy_listing = random.choice(
                np.argwhere(relevant_q_values==relevant_q_values.max())
                )[0]
        action = self.listing_to_action[optimal_policy_listing]
        
        # keep track of movements to learn later from them
        self.play_log.append([deepcopy(state.board), action])
        
        return action
    
    def _state_q_values(self, board_state):
        def encode(arr2d):
            return ''.join([str(i) for row in arr2d for i in row])
        
        board_state_encoded = encode(board_state)
        relevant_q_values = self.q_values.get(board_state_encoded)
        
        if relevant_q_values is None:
            # if updated q-value not available, use initial values
            relevant_q_values = np.ones(26)*self.initial_reward
            # considering pass option, illegalize other out of boundaries
            self.q_values[board_state_encoded] = relevant_q_values
        return np.asarray(relevant_q_values)
    
    def train(self, go, epochs, reset=False, save_freq=100, ckpt='q_values', verbose=False):

        from random_player import RandomPlayer
        import os
        
        go.verbose = verbose
            
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', '-p', type=bool, help='play given input file', default=False)
    parser.add_argument('--train', '-t', type=int, help='learn q-values', default=0)
    parser.add_argument('--save_freq', '-s', type=int, help='freq of saving q_values', default=1000)
    parser.add_argument('--ckpt', '-c', type=str, help='ckpt filename', default='q_values')
    parser.add_argument('--reset', '-r', type=bool, help='reset q-values', default=False)
    parser.add_argument('--verbose', '-v', type=bool, help='print board', default=False)
    args = parser.parse_args()
    
    N = 5
    go = GO(N)
    q_learner = QLearner()

    if args.train > 0:
        q_learner.train(go, 
                        epochs=args.train, reset=args.reset, 
                        save_freq=args.save_freq, ckpt=args.ckpt,
                        verbose=args.verbose)
    else:
        import datetime
        begin_time = datetime.datetime.now()
        piece_type, previous_board, board = readInput(N)
        go.set_board(piece_type, previous_board, board)
        q_learner.q_values = q_learner.read(filename='q_values')
        action = q_learner.get_input(go, piece_type)
        writeOutput(action)
        print('Time consumed:', datetime.datetime.now() - begin_time)

