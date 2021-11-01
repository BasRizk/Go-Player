# -*- coding: utf-8 -*-

class GameStats:
    def init_game_step():
        with open('game_stats.txt', 'w') as f:
            f.write('1')
    
    def increment_game_step():
        with open('game_stats.txt', 'r+') as f:
            cnt = f.read()
            f.seek(0)
            f.write(str(int(cnt)+1))
            f.truncate() 
            
    def get_game_step():
        with open('game_stats.txt', 'r') as f:
            return int(f.read())
