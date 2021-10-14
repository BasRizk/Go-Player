# -*- coding: utf-8 -*-

import numpy as np
import random
import json

from read import readInput
from write import writeOutput
from host import GO

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class QLearner:
    
    def __init__(self, alpha=0.5, gamma=0.5, initial_reward=-0.5):
        self.type = 'qlearner'
        self.alpha = 0.5
        self.gamma = 0.5
        self.q_values = {}
        self.initial_reward = -0.5
        self.play_log = []
    
    def get_input(self, state, piece_type):
        self.own_piece_type = piece_type
        # TODO Exploration Vs Exploitation        
       
        # TODO game could be over already!?
        # if state.game_end(piece_type, 'PASS'):
        #     print('game end-------------------------')
        #     action='PASS'
        
        # select best q-value @ current-state
        relevant_q_values = self._state_q_values(state.board)

        # if selected move not valid, turn move q-value to lose   
        for i in range(state.size):
            for j in range(state.size):
                if not state.valid_place_check(i, j, self.own_piece_type, test_check = True):
                    relevant_q_values[i][j] = -1            
        
        action = random.choice(np.argwhere(relevant_q_values==relevant_q_values.max()))
            
        if action[0] == state.size and action[1] == state.size:
            return 'PASS'
        # keep track of movements to learn later from them
        self.play_log.append([state.board, action])
        
        return action
    
    def _state_q_values(self, board_state):
        def encode(arr2d):
            return ''.join([str(i) for row in arr2d for i in row])
        
        board_state_encoded = encode(board_state)
        relevant_q_values = self.q_values.get(board_state_encoded)
        
        if relevant_q_values is None:
            # if updated q-value not available, use initial values
            relevant_q_values = np.ones((6, 6))*self.initial_reward
            # considering pass option, illegalize other out of boundaries
            N = 5
            relevant_q_values[N,:N] = relevant_q_values[:N,N] = -1
            self.q_values[board_state_encoded] = relevant_q_values
        return np.asarray(relevant_q_values)
    
    def train(self, go, epochs, reset=False, verbose=False, ckpt=100):

        from random_player import RandomPlayer
        import os
        
        if not reset and os.path.isfile('q_values'):
            self.q_values = self.read(filename='q_values')
    
        opponent_player = RandomPlayer()

        
        for i in range(epochs):
            print("Training...  %d / %d" % (i+1, epochs), end='\r')
    
            # print('Initialize Game -->', end=' ')

            go.X_move = True # X chess plays first
            go.died_pieces = [] # Intialize died pieces to be empty
            go.n_move = 0 # Trace the number of moves
            
            if i % 2 == 0:
                winner = go.play(self, opponent_player, verbose=verbose)    
            else:
                winner = go.play(opponent_player, self, verbose=verbose)    
        
            self.learn(winner)
            if i > 0 and i % ckpt == 0:
                filename = 'q_values_' + str(i)
                self.save_json(self.q_values, filename=filename)

        
        print()
        self.save_json(self.q_values, filename='q_values')

    def learn(self, winner):
        # print('Start Learning -->', end=' ')
        board_state, action = self.play_log.pop()
        # print('Pop Last Action -->', end=' ')
        current_state_q_values = self._state_q_values(board_state)
        # print('Got state Q-Values -->', end=' ')
        reward = 1 if winner == self.own_piece_type else 0 if winner == 0 else -1
        
        if action == 'PASS':
            action = (5, 5)
        current_state_q_values[action[0]][action[1]] = reward
        
        max_q = np.max(current_state_q_values)
        # print('Hold max_q value')
        # propagate rewards back from result: update move's q-value per state back to the start of the game
        while len(self.play_log) > 0:
            board_state, action = self.play_log.pop()
            current_state_q_values = self._state_q_values(board_state)
            current_state_q_values[action[0]][action[1]] =\
                (1-self.alpha)*(current_state_q_values[action[0]][action[1]]) +\
                self.alpha*(self.gamma*max_q)
            max_q = np.max(current_state_q_values)
            
    def save_json(self, obj, filename='q_values'):
        class Encoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()               
                # Let the base class default method raise the TypeError
                return json.JSONEncoder.default(self, obj)
        
        with open(".".join([filename, 'json']), 'w') as f:
            json.dump(obj, f, cls=Encoder)        
        print('Saved', filename)
            
    def read(self, filename='q_values'):
        return json.load(open(".".join([filename, 'json'])))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", "-p", type=bool, help="learn q-values", default=False)
    parser.add_argument("--train", "-t", type=int, help="learn q-values", default=0)
    parser.add_argument("--reset", "-r", type=bool, help="reset q-values", default=False)
    parser.add_argument("--verbose", "-v", type=bool, help="print board", default=False)
    args = parser.parse_args()
    
    N = 5
    go = GO(N)
    q_learner = QLearner()

    args.train =10
    if args.train > 0:
        go.verbose = args.verbose
        q_learner.train(go, epochs=args.train, reset=args.reset, verbose=args.verbose)
    else:
        import datetime
        begin_time = datetime.datetime.now()
        # piece_type, previous_board, board = readInput(N)
        # go.set_board(piece_type, previous_board, board)
        q_learner.q_values = q_learner.read(filename='q_values')
        # action = q_learner.get_input(go, piece_type)
        # writeOutput(action)
        print('Time consumed:', datetime.datetime.now() - begin_time)

