import tensorflow as tf
import numpy as np
import random
from collections import deque
from game import SnakeGameAI, Direction, Point

# constants
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
       self.num_games = 0
       self.epsilon = 0 # randomness
       self.gamma = 0 # discount rate

    def get_state(self, game):
        pass

    def remember(self,state, action, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self):
        pass

    def get_action(self,state):
        pass
        

def train():
    pass

if __name__ == '__main__':
    train()