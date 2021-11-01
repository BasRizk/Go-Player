# -*- coding: utf-8 -*-


from copy import deepcopy
from read import readInput
from write import writeOutput
from host import GO

from datetime import datetime
import warnings
import os

import random
import numpy as np

from random_player import RandomPlayer
from greedy_player import GreedyPlayer
from GameStats import GameStats
from q_values import QValues, BLOCKED_VALUE
warnings.simplefilter(action='ignore', category=FutureWarning)

        
class QLearner:
    
    def __init__(self, N=5, alpha=0.5, gamma=0.5, initial_reward=-0.5,
                 epsilon = 0, training=False, 
                 num_of_random_plays = 11,
                 percentage_cutoff = 0.5,
                 verbose=0):
        self.N = N
        self.type = 'qlearner'
        self.alpha = alpha
        self.gamma = gamma
        self.initial_reward = initial_reward
        self.q_values = QValues(N, initial_reward)
        self.play_log = []
        self.epsilon = epsilon
        self.epsilon_update = epsilon
        self.debug_file = None
        self.verbose = verbose
        self.num_of_random_plays = num_of_random_plays
        self.percentage_cutoff = percentage_cutoff
        
    def get_input(self, state, piece_type, train=True):
        if train:
            if self.verbose > 0:
                print('In Training - Using Other plays along (if applicable)')
                
            sum_of_players = sum([sum(row) for row in state.board])
    
            if sum_of_players <= 1:
                GameStats.init_game_step()
            else:
                GameStats.increment_game_step()
            
            random_player = RandomPlayer()
            greedy_player = GreedyPlayer()
            
            if self.verbose > 0:
                print('Step', GameStats.get_game_step())
                
            if sum_of_players <= 1:
                i = j = (self.N//2)
                if not go.valid_place_check(i, j, piece_type, test_check=True):
                    if self.verbose > 0:
                        print('Random Play')
                    action = random_player.get_input(go, piece_type)
                else:
                    if self.verbose > 0:
                        print('Optimal Start Play at', i, j)
                    action = (i, j)
                self.play_log.append([deepcopy(state.board), action])

            elif  GameStats.get_game_step() <= self.num_of_random_plays:
                if self.verbose > 0:
                    print('Greedy play')
                action = greedy_player.get_input(state, piece_type)
                self.play_log.append([deepcopy(state.board), action])
            else:
                if self.verbose > 0:
                    print('Q-Learner play')
                action = q_learner.get_input_internal(state, piece_type)
       
            return action
    
        else:
            return self.get_input_internal(state, piece_type)
            
    
    def get_input_internal(self, state, piece_type):
        self.piece_type = piece_type
        
        # if selected move not valid, turn move q-value to lose   
        for i in range(state.size):
            for j in range(state.size):
                if not state.valid_place_check(
                        i, j,
                        self.piece_type,
                        test_check = True
                        ):
                    self.q_values[((state.board, self.piece_type), (i, j))] = BLOCKED_VALUE
        #         else:
        #             print((i, j), end="  ")
        # print()
        
        # Exploration Vs Exploitation: Follow epsilon greedy policy
        explore_new_policies = random.uniform(0, 1) < self.epsilon_update
        if explore_new_policies:
            # if self.verbose >= 2:
            #     print('Exploring new policies')
            action = self.q_values.get_suboptimal_policy((state.board, self.piece_type))
        else:
            # if self.verbose >= 2:
            #     print('Using local optimal policy')
            action = self.q_values.get_optimal_policy((state.board, self.piece_type))
            
        # keep track of movements to learn later from them
        self.play_log.append([deepcopy(state.board), action])
        
        if self.verbose >= 2:
            self.q_values.visualize_q_table((state.board, self.piece_type))
            if explore_new_policies:
                print('Exploring new polieces')
            else:
                print('Following q-values')
            print('selected action = ', action)

        return action
    
    def load_ckpt(self, train_piece_type, ckpt):
        # if os.path.exists(ckpt):
        #     with open(ckpt, 'r') as f:
        #        piece_type = f.readline() 
        return self.load_q_values(train_piece_type, ckpt)
        
    def set_verbose(self, verbose, go=None):
        self.verbose = verbose
        self.q_values.verbose = verbose
        if go is not None:
            go.verbose = True if verbose > 1 else False
     
    def offline_play(self, go, train, epochs, train_piece_type=0,
              reset=False, save_freq=100, ckpt='999',
              epsilon=0.1, verbose=0):
        def reset_game(go):
            go.X_move = True # X chess plays first
            go.died_pieces = [] # Intialize died pieces to be empty
            go.n_move = 0 # Trace the number of moves
        
        if save_freq > epochs:
            save_freq = epochs
            
        self.set_verbose(verbose, go=go)
        
        self.epsilon = self.epsilon_update = epsilon
        self.piece_type = train_piece_type
        
        train_identifier = self.get_train_identifier()
        if train:
            if not os.path.exists(train_identifier):
                os.makedirs(train_identifier)
            else:
                print('Note that folder', train_identifier, 'already exists!')

            log_file = open(train_identifier + '/' + str(train_piece_type) + '.log', 'a')
            
        starting_epoch = 0
        if not reset:
            starting_epoch = int(self.load_ckpt(train_piece_type, ckpt))
            print('starting epoch following ckpt', ckpt, starting_epoch)
        
        from random_player import RandomPlayer
        opponent_player = RandomPlayer()
        
        play = {
            0: lambda i, x, y: play[1](i, x, y) if i%2 == 0 else play[2](i, x, y),
            1: lambda _, x, y: go.play(x, y, verbose=verbose),
            2: lambda _, x, y: go.play(y, x, verbose=verbose)
            }
        
        cnt_winnings = [0, 0]
        recent_cnt_winnings = [0, 0]
        past_num_of_first_time_states_per_run = 0
        for i in range(starting_epoch, epochs):
            if verbose > 1:
                print('\n\nStarting a game')
            self.epsilon_update = (1 - (starting_epoch/epochs))*self.epsilon

            reset_game(go)
            
            winner = play[train_piece_type](i, self, opponent_player)
            agent_won = 1 if winner == self.piece_type else 0
            cnt_winnings[self.piece_type-1] += agent_won
            recent_cnt_winnings[self.piece_type-1] += agent_won

            statement = 'Played...  %d / %d with %d X %d O winnings, and %d first-time states' %\
                      (i+1, epochs, cnt_winnings[0], cnt_winnings[1],
                       self.q_values.num_of_first_time_states_per_run)
            print(statement, end='\r')
            if train:
                self.learn(go)
                if i+1 < epochs and (i+1) % save_freq == 0:
                    print()
                    log_file.write(statement + '\n')
                    self.save_q_values(train_piece_type, i)
                    recent_first_time_states_pers_run = \
                        self.q_values.num_of_first_time_states_per_run -\
                        past_num_of_first_time_states_per_run
                    statement = '-- Recent # of Winnings = %.2f%% X %.2f%% O, %.2f%% first-time states' % \
                                (recent_cnt_winnings[0]/save_freq*100, recent_cnt_winnings[1]/save_freq*100,
                                 recent_first_time_states_pers_run/(save_freq*12)*100)
                    print(statement)
                    log_file.write(statement + '\n')
                    
                    
                    if recent_cnt_winnings[0]/save_freq > self.percentage_cutoff:
                        self.num_of_random_plays -= 1
                        statement = 'Decreasing num of random plays: %d' % self.num_of_random_plays
                        print(statement)
                        log_file.write(statement + '\n')
                    
                    past_num_of_first_time_states_per_run =\
                        self.q_values.num_of_first_time_states_per_run
                    recent_cnt_winnings = [0, 0]
                log_file.flush()
                    
        if train:
            print()
            self.save_q_values(train_piece_type, i)
            recent_first_time_states_pers_run = \
                        self.q_values.num_of_first_time_states_per_run -\
                        past_num_of_first_time_states_per_run
            statement = '-- Recent # of Winnings = %.2f%% X %.2f%% O, %.2f%% first-time states' % \
                            (recent_cnt_winnings[0]/save_freq*100, recent_cnt_winnings[1]/save_freq*100,
                            recent_first_time_states_pers_run/(save_freq*12)*100)
            print(statement)            
            log_file.write(statement + '\n')            
            log_file.close()
    
    def learn(self, final_state):
        def calc_state_reward(board_state, bias=14.5):
            def score(board, piece_type):
                '''
                Copied from Host File
                '''
                cnt = 0
                for i in range(self.N):
                    for j in range(self.N):
                        if board[i][j] == piece_type:
                            cnt += 1
                return cnt 
            cnt_1 = score(board_state, 1)
            cnt_2 = score(board_state, 2)
            side = 1 if self.piece_type == 1 else -1
            # TODO unhardcode the komi value
            score_diff = (side)*(cnt_1 - (cnt_2 + 2.5)) 
            winner = 'YOU WON' if score_diff > 0 else 'YOU LOST' if score_diff < 0 else 'DRAW'
            # reward = -2 if score_diff < 0 else 1 # (score_diff + bias)
            reward = score_diff
            return reward, winner
        
        def calc_state_action_reward(board_state, next_state):
            diff = np.array(next_state) - np.array(board_state)
            # captured pieces
            return len(diff[diff < 0])
        
        end_reward, winner = calc_state_reward(final_state.board)
        
        if self.verbose > 1:
            print('\nLearning Starting')
            print('Play-log consists of %d states' % len(self.play_log))
            print('Q-values consists of %d tables' % len(self.q_values.q_tables))
            print('Winner is', winner)
            # print('Final Reward is %f' % end_reward)
            print('Final State state:')
            self.visualize_board(final_state.board)
            print('Last board state in log:')
            self.visualize_board(self.play_log[-1][0])
            print('Applied action:', self.play_log[-1][1])
            
        max_q = -np.inf #self.q_values.max_q((board_state, self.piece_type))
        # policy_q = -np.inf # self.q_values[((board_state, self.piece_type), action)] 

        # propagate rewards back from result: update move's q-value per state back to the start of the game
        most_recent_board_state = final_state.board
        while len(self.play_log) > 0:
            board_state, action = self.play_log.pop()
            
            if self.verbose > 1:
                print('Current board state in log:')
                self.visualize_board(board_state)
                print('Applied action:', action)
                print('Corresponding Q-values before update:')
                self.q_values.visualize_q_table((board_state, self.piece_type))
            
            old_q = self.q_values[((board_state, self.piece_type), action)]
            # state_reward, _ = calc_state_reward(board_state)
            
            reward = calc_state_action_reward(board_state, most_recent_board_state)
            most_recent_board_state = board_state
            if max_q == -np.inf:
                # policy_q = end_reward
                # Based on final state of the game
                max_q = self.q_values.max_q((final_state.board, self.piece_type))
            
            new_q = (1-self.alpha)*(old_q) + self.alpha*(reward + self.gamma*max_q)
            # new_q = (1-self.alpha)*(old_q) + self.alpha*(state_reward + self.gamma*policy_q)
            self.q_values[((board_state, self.piece_type), action)] = new_q
                
            if self.verbose > 1:
                print('old_q', old_q, 'at', action, '-> new_q' ,new_q)
                print('reward', reward, 'max_q', max_q)
                # print('state_reward', state_reward, 'max_q', max_q, 'policy_q', policy_q)
            
                print('Corresponding Q-values after update:')
                self.q_values.visualize_q_table((board_state, self.piece_type))
                
            max_q = self.q_values.max_q((board_state, self.piece_type))
            # policy_q = new_q
                
        if self.verbose > 1:
            print('Winner is', winner, )
            print('To recall.. Final Reward is %f' % end_reward)
            print('with game state:')
            self.visualize_board(final_state.board)
            
    
    def get_train_identifier(self):
        return str(self.alpha) + 'alpha' + str(self.gamma) + 'gamma' +\
            str(self.epsilon) + 'eps' + str(self.initial_reward) + 'initial_value'
            
    def save_q_values(self, train_piece_type, epoch):
        self.q_values.save(self.get_train_identifier() + '/' + str(train_piece_type), epoch+1)
        
    def load_q_values(self, piece_type, epochs=''):
        return self.q_values.load(self.get_train_identifier() + '/' + str(piece_type), epochs)
        
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t', type=bool, help='learn q-values', default=True)
    parser.add_argument('--epochs', '-i', type=int, help='num of training epochs', default=5)
    parser.add_argument('--save_freq', '-s', type=int, help='freq of saving q_values', default=1)
    parser.add_argument('--piece_type', '-pt', type=int, help='piece_type_to_train', default=1)
    parser.add_argument('--alpha', '-a', type=float, help='alpha - learning rate', default=0.5)
    parser.add_argument('--gamma', '-g', type=float, help='gamma - discount factor', default=0.9)
    parser.add_argument('--initial_reward', '-ir', type=float, help='initial q-value', default=-30)
    parser.add_argument('--epsilon', '-e', type=float, 
                        help='probability of exploring new policies', default=0)
    parser.add_argument('--num_of_random_plays', '-nrp', type=int,
                        help='plays without q-learner', default=5)
    parser.add_argument('--percentage_cutoff', '-pcut', type=float,
                        help='percentage that when perceived, num_of_random_plays is decreased',
                        default=0.5)
    parser.add_argument('--ckpt', '-c', type=str, help='ckpt epoch', default='999')
    parser.add_argument('--reset', '-r', type=bool, help='reset q-values', default=False)
    parser.add_argument('--verbose', '-v', type=int, help='print board', default=0)
    args = parser.parse_args()
    
    N = 5
    go = GO(N)
    q_learner = QLearner(alpha=args.alpha, gamma=args.gamma,
                         initial_reward=args.initial_reward,
                         num_of_random_plays=args.num_of_random_plays,
                         percentage_cutoff=args.percentage_cutoff,
                         verbose=args.verbose)
    
    
    if args.epochs > 0:
        print('Training or Playing Casual Session')
        q_learner.offline_play(go, train = args.train,
                        epochs=args.epochs, train_piece_type=args.piece_type,
                        reset=args.reset, save_freq=args.save_freq,
                        ckpt=args.ckpt,
                        epsilon=args.epsilon,
                        verbose=args.verbose)
    else:
        
        num_of_random_plays = 4
        sum_of_players = sum([sum(row) for row in go.board])

        begin_total_time = datetime.now()
        piece_type, previous_board, board = readInput(N)
        go.set_board(piece_type, previous_board, board)
        
        random_player = RandomPlayer()
        greedy_player = GreedyPlayer()
    
        if sum_of_players == 0:
            print('Random Play')
            action = random_player.get_input(go, piece_type)
        elif sum_of_players < 3*num_of_random_plays:
            print('Greedy play')
            action = greedy_player.get_input(go, piece_type)
        else: 
            print('QLearner Play')
            begin_reading_time = datetime.now()
            q_learner.load_q_values(piece_type=0, epochs=40000)
            # q_values = q_learner.q_values
            action = q_learner.get_input_internal(go, piece_type)
            writeOutput(action)
            print('Reading time consumed:', datetime.now() - begin_reading_time)
        print('Total time consumed:', datetime.now() - begin_total_time)


    