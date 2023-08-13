import tensorflow as tf
import numpy as np
import random
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE

# constants
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
       self.num_games = 0
       self.epsilon = 0 # randomness
       self.gamma = 0 # discount rate
       self.memory=deque(maxlen=MAX_MEMORY) # popleft()
       self.model = None
       self.trainer = None


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # directions
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state=[
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #food loc
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y # food down
            
        ]

        return np.array(state,dtype=int)

    



    def remember(self,state, action, reward, next_state, done):
        self.memory.append((state,action,reward,next_state,done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            rand_sample=random.sample(self.memory,BATCH_SIZE)
        else:
            rand_sample=self.memory


        states,actions,rewards,next_states,dones= zip(*rand_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

        

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state):
        self.epsilon = 80 - self.num_games
        final_move=[0,0,0]
        if random.randint(0,200) < self.epsilon:
            move=random.randint(0,2)
            final_move[move]=1
        else:

            #tf code

            state0=tf.convert_to_tensor(state,dtype=tf.float32)
            state0=tf.expand_dims(state0,0)
            prediction=self.model.predict(state0)
            move=tf.argmax(prediction[0]).numpy()
            final_move[move]=1

        return final_move
        

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # load old state
        old_state = agent.get_state(game)

        # get move
        final_move = agent.get_action(old_state)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new=agent.get_state(game)

        # train short memory
        agent.train_short_memory(old_state,final_move,reward,state_new,done)
        agent.remember(old_state,final_move,reward,state_new,done)

        if done:
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                #agent.model.save()

            print('Game', agent.num_games, 'Score', score, 'Record:', record)

                  




if __name__ == '__main__':
    train()