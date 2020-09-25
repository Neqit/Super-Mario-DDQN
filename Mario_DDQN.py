# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 21:42:44 2020

@author: Nichita Vatamaniuc
"""

import time
import random
import numpy as np
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt

import datetime
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from wrappers import wrapper

from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


import warnings
warnings.simplefilter('ignore')

import os
#Use in case training on CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def parameters_defenition():
    parameters = dict()
    parameters['epsilon_decay'] = 0.000111    
    parameters['learning_rate'] = 0.0005    
    parameters['episodes_to_play'] = 10000   
    parameters['memory_size'] = 100000    
    parameters['batch_size'] = 64    
    parameters['burnin'] = 100000    
    parameters['copy'] = 10000    
    parameters['save_each'] = 200000    
    parameters['learn_each'] = 3   
    parameters['train'] = False    
    parameters['environment'] = 'SuperMarioBros-1-1-v0'    
    parameters['render'] = True    
    return parameters


#Limiting GPU memory growth - https://www.tensorflow.org/guide/gpu
def cuda_memgrowth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


#Function for rendering plots in the end of training
def draw_graph(reward, name):
    plt.plot(np.asarray(reward))
    plt.title(name)
    plt.show()


#DDQN agent class
class DDQNagent:
    def __init__(self, parameters, input_dim, actions):
        self.input_dim = input_dim    # Input dimension
        self.actions = actions    # Action Space
        self.learning_rate = parameters['learning_rate']
        self.model_online = self.neural_network_online()
        self.model_target = self.neural_network_target()
        self.memory = deque(maxlen=parameters['memory_size']);
        self.eps = 1
        self.eps_decay = parameters['epsilon_decay']
        self.gamma = 0.90
        self.batch_size = parameters['batch_size']
        self.burnin = parameters['burnin']
        self.copy = parameters['copy']
        self.step = 0    # Number of steps
        self.learn_each = parameters['learn_each']
        self.learn_step = 0    
        self.save_each = parameters['save_each']
        self.flag_reached = 0    # Number of wins
        self.opt = tf.optimizers.Adam(lr=self.learning_rate, )
        self.loss = 0
    

    def neural_network_online(self):
        input_img = Input(shape=(self.input_dim))
        img = Conv2D(filters = 32, kernel_size = [8,8], strides = [4,4])(input_img)
        img = LeakyReLU(0.01)(img)
        img = Conv2D(filters = 64, kernel_size = [4,4], strides = [2,2])(img)
        img = LeakyReLU(0.01)(img)
        img = Conv2D(filters = 64, kernel_size = [3,3], strides = [1,1])(img)
        img = LeakyReLU(0.01)(img)
        img = Flatten()(img)
        img = Dense(512)(img)
        img = LeakyReLU(0.01)(img)
        img = Dense(self.actions)(img)
        
        model = Model(inputs=input_img, outputs=img)
        model.compile(loss = 'mse', optimizer = Adam(self.learning_rate))
        return model
    
    
    def neural_network_target(self):
        if parameters['DDQN']:
            input_img = Input(shape=(self.input_dim))
            img = Conv2D(filters = 32, kernel_size = [8,8], strides = [4,4])(input_img)
            img = LeakyReLU(0.01)(img)
            img = Conv2D(filters = 64, kernel_size = [4,4], strides = [2,2])(img)
            img = LeakyReLU(0.01)(img)
            img = Conv2D(filters = 64, kernel_size = [3,3], strides = [1,1])(img)
            img = LeakyReLU(0.01)(img)
            img = Flatten()(img)
            img = Dense(512)(img)
            img = LeakyReLU(0.01)(img)
            img = Dense(self.actions)(img)
    
            model = Model(inputs=input_img, outputs=img)
            return model
        else:
            return 0
    

    def update_memory(self, experience):
        self.memory.append(experience)
        

    def run(self, state):
       #Epsilon-Greedy policy condition
       if np.random.rand() < self.eps:
           #Random action
           action = np.random.randint(low=0, high=self.actions)
       else:
           #Predicted action
           predict_online = self.model_online((np.expand_dims(state, 0)).astype('float32')/255.)
           action_online = np.argmax(predict_online)
           action = action_online
       self.step += 1
       return action
   

    def copy_model(self):
        self.model_target.set_weights(self.model_online.get_weights())
            

    def save_weights(self):
        online_weights_name = 'online_weights' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.hdf5'
        self.model_online.save_weights(online_weights_name)
        print('Online model was saved as ' + online_weights_name)

        target_weights_name = 'target_weights' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.hdf5'
        self.model_target.save_weights(target_weights_name)
        print('Target model was saved as ' + target_weights_name)
    

    def learn(self):
        #Copy weights
        if self.step % self.copy == 0:
            self.copy_model()
        #Save weights
        if self.step % self.save_each == 0:
            self.save_weights()
        #Burnin
        if self.step < self.burnin:
            return
        #Learn skipping
        if self.learn_step < self.learn_each:
            self.learn_step += 1
            return
        
        #Take some random data from memory
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(np.array, zip(*batch))
        
        #from 0-255 to 0-1
        state = state/255.
        next_state=next_state/255.
        
        #DDQN Algorithm
        dqn_variable = self.model_online.trainable_variables

        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)
            
            #Conveting to tensors
            reward = tf.convert_to_tensor(reward, dtype=tf.float32)
            action = tf.convert_to_tensor(action, dtype=tf.int32)
            done = tf.convert_to_tensor(np.array(done).astype(int), dtype=tf.float32)
            
            #Predicted Q values from Target and Online networks
            target_q = self.model_target(tf.convert_to_tensor(np.stack(next_state), dtype=tf.float32))
            main_q = self.model_online(tf.convert_to_tensor(np.stack(next_state), dtype=tf.float32))
            
            #for main_q gradient will be stoped, we will use argmax function that is not derivable
            main_q = tf.stop_gradient(main_q)
            next_action = tf.argmax(main_q, axis=1)
            
            #With matrix calculations we will find discounted Q-values 
            target_value = tf.reduce_sum(tf.one_hot(next_action, self.actions) * target_q, axis=1)
            target_value = (1-done) * self.gamma * target_value + reward
            main_q = self.model_online(tf.convert_to_tensor(np.stack(state), dtype=tf.float32))
            main_value = tf.reduce_sum(tf.one_hot(action, self.actions) * main_q, axis=1)

            #Calculating of error between predicted and real Q-values
            error = tf.square(main_value - target_value) * 0.5
            error = tf.reduce_mean(error)
        
        #Calculate gradient
        dqn_grads = tape.gradient(error, dqn_variable)
        #Update weights
        self.opt.apply_gradients(zip(dqn_grads, dqn_variable))
        self.loss = float(error)
            
        self.learn_step = 0 


    #Method for testing
    def model_test(self, env):
        done = True
        #Agent will do 5000 actions
        for step in range(5000):
            if done:
                state = env.reset()
            action = np.argmax(self.model_target((np.expand_dims(state, 0)).astype('float32')/255.))
            print(self.model_online((np.expand_dims(state, 0)).astype('float32')/255.))
            state, reward, done, info = env.step(action)
            env.render()


#Main training function
def train_model(parameters):
    #Initialization of environment and agent
    env = gym_super_mario_bros.make(parameters['environment'])
    env = JoypadSpace(env, RIGHT_ONLY)
    env = wrapper(env)

    states = (84, 84, 4)
    actions = env.action_space.n
    
    agent = DDQNagent(parameters, states, actions)

    if parameters['train']:
        #TENSORBOARD
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")    
        log_dir = 'logs/mario/' + current_time + '/10k'    
        summary_writer = tf.summary.create_file_writer(log_dir)    
        summary_writer.set_as_default()
        
        
        maxXpos = 0    # Maximum X position of the Agent
        max_reward = 0    # Maximum reward
        start_time = time.time()    # Start time
        
        #Initialization of varialbes for plots
        graph_reward = np.zeros(parameters['episodes_to_play'])    # Reward
        graph_pos = np.zeros(parameters['episodes_to_play'])    # Pozition
        graph_mean_reward = np.zeros(parameters['episodes_to_play'])    # Mean Reward
    
    
        episodes = parameters['episodes_to_play']    # Number of episodes to train
        rewards = []    # Rewards array
        
        
        start = time.time()    # Time for calculating processed frames per second
        step = 0    # Total steps
        
        #Lerning cycle
        for e in range(episodes):
    
            #Default state of the environment
            state = env.reset()
        
            total_reward = 0    # Reward gained for actual epsiode
            iter = 0
        
            while True:
                #Select an action
                action = agent.run(state)
        
                #Apply action to environment
                next_state, reward, done, info = env.step(action)
    
                #Write new data to memory
                agent.update_memory(experience=(state, next_state, action, reward, done))
    
                #Learn
                agent.learn()
        
                #Sum of rewards for every action
                total_reward += reward
        
                #Change current state to next one
                state = next_state
        
                iter += 1
                
                #Render
                if parameters['render']:
                    env.render()
        
                #Check finish condition
                if done or info['flag_get']:
                    break
        
            #New data for variable that be used for plot
            rewards.append(total_reward / iter)
            
            #Update info
            if maxXpos < info['x_pos']:
                maxXpos = info['x_pos']
            if max_reward < total_reward:
                max_reward = total_reward
            
            if info['flag_get'] == True:
                agent.flag_reached = agent.flag_reached + 1
            
            #Epsilon decay
            if agent.eps >= 0.0:
                agent.eps = agent.eps - agent.eps_decay
            
            #Updtate variables for plots
            graph_reward[e] = total_reward
            graph_pos[e] = info['x_pos']
            graph_mean_reward[e] = np.mean(graph_reward)
            
            #TENSORBOARD
            tf.summary.scalar("Rewards", total_reward, step=e)
            tf.summary.scalar("Position", info['x_pos'], step=e)
            tf.summary.scalar("Mean reward", np.mean(graph_reward), step=e)
            tf.summary.scalar("Flags", agent.flag_reached, step=e)
            tf.summary.scalar("Loss", agent.loss, step=e)
            
            
            #Console information
            print("Episode reward: " + str(total_reward) + ' - Pos: ' + str(info['x_pos']))
            # Print
            if e % 10 == 0:
                end = time.time()
                print('Flags reached: ' + str(agent.flag_reached) + ' - Max reward: ' +str(max_reward))
                print('Episode {e} - '
                      'Frame {f} - '
                      'Frames/sec {fs} - '
                      'Epsilon {eps} - '
                      'Mean Reward {r} - '
                      'Time {t} sec - '
                      'Max pos {pos}'.format(e=e,
                                               f=agent.step,
                                               fs=np.round((agent.step - step) / (time.time() - start)),
                                               eps=np.round(agent.eps, 4),
                                               r=np.mean(rewards[-100:]),
                                               t=round(end - start_time),
                                               pos=maxXpos))
    
    
                start = time.time()    
                step = agent.step    
        
        #After learning draw plots and save weights
        draw_graph(graph_reward,'Rewards')
        draw_graph(graph_pos, 'Position')
        draw_graph(graph_mean_reward, 'Mean reward')
        agent.save_weights()
        env.close() 
        
    else:
        #If train is equal to false, it is possible to load weights and observe result
        print('Weights file path (hdf5): ')
        weights_name = input()
        try:
            agent.model_target.load_weights(weights_name)
            agent.model_test(env)
        except:
            print("Weights with this name or on this path not found")
        env.close()

if __name__ == '__main__':
    cuda_memgrowth()    
    parameters = parameters_defenition()    
    train_model(parameters)