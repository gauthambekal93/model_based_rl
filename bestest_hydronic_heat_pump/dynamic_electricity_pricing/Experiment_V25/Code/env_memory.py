# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:56:11 2025

@author: gauthambekal93
"""

import os
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V25/Code')

import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn

import torch.optim as optim
import torch.distributions as dist

from collections import deque  
import time

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  


class Environment_Memory_Train():
    
    def __init__(self, buffer_size , train_test_ratio ): #was 200   
    
        
        self.train_test_ratio = train_test_ratio
        
        self.states = deque(maxlen = buffer_size) 
        self.actions = deque(maxlen = buffer_size) 
        self.rewards = deque(maxlen = buffer_size) 
        self.next_states = deque(maxlen = buffer_size) 
        
        self.time_diff =  None
        
        self.buffer_size = buffer_size
        
        self.train_index =[]
        self.validation_index = []
        
        
    def remember(self, i_episode, state, action, discrete_action, reward, next_state, env):
    
    
        self.states.append( {  k : v for k, v in zip( env.observations, state) } )  
        
        action_names = env.actions + list( ["oveFan_u"] + ["ovePum_u"] )
        
        action = action.detach().clone().numpy()
        
        temp =  np.where(discrete_action == 0, 0 , 1) 
        
        action = np.concatenate( [ action, temp , temp ], axis = 1  ).reshape(-1)
        
        self.actions.append( {  k:v for k, v in zip( action_names, action) } )
        
        self.next_states.append( {  k:v for k, v in zip( env.observations, next_state) } )
        
        self.rewards.append(  {"reward": reward}  )
      
        
        last_index = len(self.states) - 1
        
        if random.random() <= self.train_test_ratio:  
            self.train_index.append(last_index)
        else:
            self.validation_index.append(last_index)
            
        
    
    def sample_random_states(self, sample_size = 1 ):
    
        random_numbers = torch.randint(0, len(self.train_states), (sample_size,))
            
        return torch.cat(list(self.train_states), dim = 0)[random_numbers] 
        
    
    def memory_size(self):
         return len(self.states) 
     
        
    def is_full(self):
        return self.buffer_size <= len(self.states)  #the greater symbol is just in case there is some anomaly and more data is stored


    def clear_buffer(self):
        
        self.states.clear()
        self.actions.clear()  
        self.rewards.clear()
        self.next_states.clear()
        
        self.train_index =[]
        self.validation_index = []
        



class Environment_Memory_Test():
    
    def __init__(self, buffer_size ): #was 200   
        
        self.states = deque(maxlen = buffer_size) 
        self.actions = deque(maxlen = buffer_size) 
        self.rewards = deque(maxlen = buffer_size) 
        self.next_states = deque(maxlen = buffer_size) 
        
        
        self.time_diff =  None
        
        self.buffer_size = buffer_size
    
        self.test_index = []
        
        
    def remember(self, i_episode, state, action, discrete_action, reward, next_state, env):
    
    
        self.states.append( {  k : v for k, v in zip( env.observations, state) } )
        
        action_names = env.actions + list( ["oveFan_u"] + ["ovePum_u"] )
        
        action = action.detach().clone().numpy()
        
        temp =  np.where(discrete_action == 0, 0 , 1) 
        
        action = np.concatenate( [ action, temp , temp ], axis = 1  ).reshape(-1)
        
        self.actions.append( {  k:v for k, v in zip( action_names, action) } )
        
        self.next_states.append( {  k:v for k, v in zip( env.observations, next_state) } )
        
        self.rewards.append(  {"reward": reward}  )
      
        last_index = len(self.states) - 1
        
        self.test_index.append(last_index)
            
        
    
    def sample_random_states(self, sample_size = 1 ):
    
        random_numbers = torch.randint(0, len(self.train_states), (sample_size,))
            
        return torch.cat(list(self.train_states), dim = 0)[random_numbers] 
        
    
    def memory_size(self):
         return len(self.states) 
     
        
    def is_full(self):
        return self.buffer_size <= len(self.states)  #the greater symbol is just in case there is some anomaly and more data is stored


    def clear_buffer(self):
        
        self.states.clear()
        self.actions.clear()  
        self.rewards.clear()
        self.next_states.clear()
        self.test_index = []        