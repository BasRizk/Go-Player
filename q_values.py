# -*- coding: utf-8 -*-

import numpy as np
import random
import json

from copy import deepcopy

from datetime import datetime
import os

BLOCKED_VALUE=-100

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
        self.num_of_first_time_states_per_run = 0

    
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
        self.sym_dict_backward = [{} for i in range(len(equi_boards))]
            
        for d_i in range(len(equi_boards)):
            for i in range(self.N):
                for j in range(self.N):
                    c = org_oriantation[i][j]
                    self.sym_dict_backward[d_i][equi_boards[d_i][i][j]] = c
                    self.sym_dict_forward[d_i][c] = equi_boards[d_i][i][j]
                    

    def visualize_q_table(self, board_state, encoding_inc=None, verbose=0):
        def fit_table(table):
            fit_table = np.zeros(len(table) - 1)
            for i in range(len(fit_table)):
                fit_table[i] =\
                    table[self.sym_dict_backward[self.index_of_encoding][i]]
                if self.verbose >= 4:
                    print('fit_table at %d moved to %d with value=%.2f' 
                          % (i, self.sym_dict_backward[self.index_of_encoding][i],
                          table[self.sym_dict_backward[self.index_of_encoding][i]]))
            return fit_table
            
        def print_on_board(fit_q_table):
            print('-' * self.N * 9)
            for i in range(self.N):
                for j in range(self.N):
                    if fit_q_table[i][j] == BLOCKED_VALUE:
                        print('%8s' % ' ', end='|')
                    else:
                        if len(str(fit_q_table[i][j])) > 7 and -1 < fit_q_table[i][j] < 1:
                            print('%8s' % "{:.1e}".format(fit_q_table[i][j]), end='|')
                        else:
                            print('%8s' % "{:.3f}".format(fit_q_table[i][j]), end='|')

                print()
            print('PASS =', self.recent_state_q_values[-1])
            print('-' * self.N * 9)
        
        self._ensure_recent_state(board_state)
        if encoding_inc is not None and self.index_of_encoding not in encoding_inc:
            return
                    
        if self.verbose >= 3:
            print('encoding is %d' % self.index_of_encoding)
            print('before transformation')
            if self.verbose >= 4:
                print(self.recent_state_q_values)
                print()
            print_on_board(deepcopy(self.recent_state_q_values)[:25].reshape((self.N, self.N)))
        
        fit_q_table = fit_table(self.recent_state_q_values).reshape((self.N, self.N))
        print('after transformation')
        print_on_board(fit_q_table)
        print('Max Element at %s, Min Element exc(%s) at %s' 
              % (self._listing_to_fit_action(np.argmax(self.recent_state_q_values), forward=True),
                 BLOCKED_VALUE,
                 self._listing_to_fit_action(np.argmin(
                     self.recent_state_q_values[
                         self.recent_state_q_values != BLOCKED_VALUE
                         ]), forward=True)))
        print('Max Element is %s, Min Element exc(%s) is %s' 
              % (np.max(self.recent_state_q_values),
                 BLOCKED_VALUE,
                 np.min(
                     self.recent_state_q_values[
                         self.recent_state_q_values != BLOCKED_VALUE
                         ])))
        
    def _state_q_values(self, board_state):      
        def encode(board_state):
            arr2d, piece_type = board_state
            arr = np.array(arr2d).reshape(len(arr2d)**2)
            # TODO check weird behavior here
            # print(arr)
            if piece_type == 2:
                arr = 3 - arr
                arr[arr==3] = 0
                
            all_encodings = []
            for mapping in self.sym_dict_forward:
                encoding = ""
                for i in range(len(arr)):
                    encoding += str(arr[mapping[i]])
                all_encodings.append(encoding)
            
            all_encodings = [int(encoding) for encoding in all_encodings]
            index_of_encoding = np.argmin(all_encodings)
            return index_of_encoding, str(all_encodings[index_of_encoding])
        
        index_of_encoding, board_state_encoded = encode(board_state)
        relevant_q_values = self.q_tables.get(board_state_encoded)
        if relevant_q_values is None:
            if self.verbose >=2 :
                print('First-Time State')
            # 25 cells + 1 considering pass option
            relevant_q_values = np.ones(26)*self.initial_reward
            self.q_tables[board_state_encoded] = relevant_q_values
            self.num_of_first_time_states_per_run += 1
    
        self.index_of_encoding = index_of_encoding
        self.recent_board_encoding = board_state_encoded
        self.recent_state_q_values = np.asarray(relevant_q_values)
    
    def _ensure_recent_state(self, board_state, reset=False):
        if reset or (board_state != self.recent_board_state):
            self.recent_board_state = deepcopy(board_state)
            self._state_q_values(board_state)
    
    def _action_to_fit_listing(self, action, forward=False, verbose=False):        
        listing = self.action_to_listing[action]
        if verbose:
            print('%15s' % 'original listing', listing)
        if listing != 25:
            # todo check this again
            if forward:
                listing = self.sym_dict_forward[self.index_of_encoding][listing]
            else:
                listing = self.sym_dict_backward[self.index_of_encoding][listing]
        if verbose:
            print('%15s' % 'after listing', listing)
        return listing
    
    def _listing_to_fit_action(self, listing, forward=False):
        action = self.listing_to_action[listing]
        # print('%15s' % 'original action', action)
        listing = self._action_to_fit_listing(action, forward=forward, verbose=False)
        return self.listing_to_action[listing]

    def __getitem__(self, selector):
        board_state, action = selector
        self._ensure_recent_state(board_state)
        fit_listing = self._action_to_fit_listing(action)
        return self.recent_state_q_values[fit_listing]        
    
    def __setitem__(self, selector, value):
        board_state, action = selector
        self._ensure_recent_state(board_state)
        fit_listing = self._action_to_fit_listing(action)
        self.recent_state_q_values[fit_listing] = value
        # TODO Check if needed to reflect changed Value
        self._ensure_recent_state(board_state, reset=True)
    
    def get_suboptimal_policy(self, board_state):
        self._ensure_recent_state(board_state)
        other_policy_listings =\
            np.argwhere(
                (self.recent_state_q_values!=self.recent_state_q_values.max()) &\
                    (self.recent_state_q_values!= BLOCKED_VALUE)
                )
        if len(other_policy_listings) == 0:
            return self.get_optimal_policy(board_state)
        selected_policy_listing = random.choice(other_policy_listings)[0]
    
        return self._listing_to_fit_action(selected_policy_listing, forward=True)
    
    def get_optimal_policy(self, board_state):
        self._ensure_recent_state(board_state)
        optimal_policy_listings =\
            np.argwhere(self.recent_state_q_values==self.recent_state_q_values.max())
        # print('optimal_policy_listings')
        # print('max value is', self.recent_state_q_values.max())
        # for p in optimal_policy_listings:
        #     print(p, '->', self._listing_to_fit_action(p[0], forward=True))
        
        # TODO test out random or front selection
        selected_policy_listing = random.choice(optimal_policy_listings)[0]
        # selected_policy_listing = optimal_policy_listings[0][0]
        
        # print('unfit action', self.listing_to_action[selected_policy_listing])
        action = self._listing_to_fit_action(selected_policy_listing, forward=True)
        # print('fit action', action)
        return action
    
    def max_q(self, board_state):
        self._ensure_recent_state(board_state)
        return np.max(self.recent_state_q_values)
    
    def save(self, train_identifier, epoch):
        class Encoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()               
                # Let the base class default method raise the TypeError
                return json.JSONEncoder.default(self, obj)
        def ensure_directory(train_identifier):
            if not os.path.exists(train_identifier):
                os.makedirs(train_identifier)
        begin_time = datetime.now()
        ensure_directory(train_identifier)
        filepath = train_identifier + '/' + '_'.join(['q_values', str(epoch)]) + '.json'
        print('Saving..', end='\r')
        with open(filepath, 'w') as f:                 
            json.dump(self.q_tables, f, cls=Encoder)    
        end_time = datetime.now() - begin_time
        print('Saved %s in' % filepath, end_time)

    def load(self, train_identifier, epoch):
        begin_time = datetime.now()
        filepath = train_identifier + '/' +\
            '_'.join(['q_values', str(epoch)]) + '.json'
        if os.path.isfile(filepath):
            print('Loading..', end='\r')
            self.q_tables = json.load(open(filepath))
            end_time = datetime.now() - begin_time
            print('Loaded %s in' % filepath, end_time)
            return int(epoch)
        else:
            print('filepath %s is NOT found' % filepath)
        return 0