# -*- coding: utf-8 -*-
import numpy as np
from read import readInput
from write import writeOutput
from host import GO

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
        if best_state is None:
            best_state = 'PASS'
        return best_state
    
    
if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = GreedyPlayer()
    action = player.get_input(go, piece_type)
    writeOutput(action)