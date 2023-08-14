import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

class Model:
    def __init__(self,inpu_t,hidden,output):
        self.model = keras.Sequential()
        self.model.add(layers.Dense(hidden,activation='relu',input_shape=(inpu_t,)))
        self.model.add(layers.Dense(hidden,activation='relu'))
        self.model.add(layers.Dense(output,activation='linear'))
    

    
    def save_model(self,file_name='model.h5'):
        self.model.save(file_name)


class Q_Training:

    def __init__(self,model,lr,gamma):
        self.model=model
        self.gamma=gamma
        self.lr=lr
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.loss= keras.losses.MeanSquaredError()
        #self.model.compile(optimizer=self.optimizer,loss=self.loss)

    def train_step(self,state,action,reward,next_state,done):
        
        state=tf.convert_to_tensor(state,dtype=tf.float32)
        action=tf.convert_to_tensor(action,dtype=tf.float32)
        reward=tf.convert_to_tensor(reward,dtype=tf.float32)
        next_state=tf.convert_to_tensor(next_state,dtype=tf.float32)
        done=tf.convert_to_tensor(done,dtype=tf.float32)

        if len(state.shape)==1:
            state=tf.expand_dims(state,0)
            next_state=tf.expand_dims(next_state,0)
            reward=tf.expand_dims(reward,0)
            done=tf.expand_dims(done,0)
            action=tf.expand_dims(action,0)
        
        pred=self.model.model(state)
        target=self.model.model(next_state)

        for ind in range(len(done)):
            Q_new=reward[ind]
            if not done[ind]:
                Q_new=reward[ind]+self.gamma*tf.math.reduce_max(target[ind])
            
            pred[ind][tf.argmax(action[ind]).numpy()]=Q_new

        self.model.compile(optimizer=self.optimizer,loss=self.loss)


