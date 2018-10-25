import numpy as np


class Sudoku():
    # init_state: 9x9 2d numpy array
    def __init__(self, init_state):
        self.state = init_state

    # action: (x-position, y-position, value)
    def perform(self, action):
        x, y, value = action
        self.state[x][y] = value
        return self.state

    def reward(self, parent, action):
        pass

    def is_terminal(self):
        pass
