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


class Agent_Memory:
    def __init__(self, buffer_size = 10000):  
        
        self.states = deque(maxlen=buffer_size) 
        self.actions = deque(maxlen=buffer_size) 
        self.discrete_actions = deque(maxlen=buffer_size) 
        self.rewards = deque(maxlen=buffer_size) 
        self.next_states = deque(maxlen=buffer_size) 
        self.done = deque(maxlen=buffer_size)
        
                                 
        
    def remember(self, state, action, discrete_action, reward, next_state, done ):
        
        self.states.append(  torch.tensor( state ).reshape(1,-1) )
        
        temp =  torch.tensor( np.where(discrete_action == 0, 0 , 1) ) 
        
        action = action.detach().clone() 
        
        self.actions.append( torch.cat( [action, temp, temp ], dim = 1)   )
        
        self.discrete_actions.append( torch.tensor(discrete_action).reshape(1,-1) )
        
        self.rewards.append(torch.tensor(reward).reshape(1,-1))
        
        self.next_states.append(torch.tensor(next_state ).reshape(1,-1) )
        
        self.done.append(torch.tensor(done).reshape(1,-1))
       
        
    def sample_memory(self,  sample_size = 64 , last_element = False ):  
        
        if last_element is False:
           random_numbers = torch.randint(0, torch.cat(list(self.states), dim = 0).shape[0] , (sample_size,))   
        else:
            last_index = self.memory_size() - 1
            
            random_numbers = torch.tensor( [ last_index ] )   
        
        return (
                torch.cat(list(self.states), dim = 0)[random_numbers] , 
                torch.cat(list(self.actions), dim = 0)[random_numbers], 
                torch.cat(list(self.discrete_actions), dim = 0)[random_numbers], 
                torch.cat(list(self.rewards), dim = 0)[random_numbers], 
                torch.cat(list(self.next_states), dim = 0)[random_numbers],    
                torch.cat(list(self.done), dim = 0)[random_numbers]
               )

       
    def memory_size(self):
         return len(self.states)



class Synthetic_Memory:
    def __init__(self, buffer_size = 10000):  
        
        self.states = deque(maxlen=buffer_size) 
        self.actions = deque(maxlen=buffer_size) 
        self.rewards = deque(maxlen=buffer_size) 
        self.next_states = deque(maxlen=buffer_size) 
        self.done = deque(maxlen=buffer_size)
                                 
        
    def remember(self, state, action, reward, next_state, done = 0 ):
        
        self.states.append(  state.reshape(1,-1) )
        self.actions.append( action.reshape(1,-1) )
        self.rewards.append( reward.detach().clone().reshape(1,-1) )
        self.next_states.append( next_state.detach().clone().reshape(1,-1) )
        self.done.append( torch.tensor(done).reshape(1,-1) )
       
        
    def sample_memory(self,  sample_size = 64 ):  
        
        random_numbers = torch.randint(0, torch.cat(list(self.states), dim = 0).shape[0] , (sample_size,))   
        
        return (
                torch.cat(list(self.states), dim = 0)[random_numbers] , 
                torch.cat(list(self.actions), dim = 0)[random_numbers], 
                torch.cat(list(self.rewards), dim = 0)[random_numbers], 
                torch.cat(list(self.next_states), dim = 0)[random_numbers],    
                torch.cat(list(self.done), dim = 0)[random_numbers]
               )

            
    def memory_size(self):
         return len(self.states)
        
        
class Environment_Memory_Train():
    
    def __init__(self, buffer_size , train_test_ratio ): #was 200   
    
        self.name = "Train"
        self.train_test_ratio = train_test_ratio
        
        self.states = deque(maxlen = buffer_size) 
        self.actions = deque(maxlen = buffer_size) 
        self.rewards = deque(maxlen = buffer_size) 
        self.next_states = deque(maxlen = buffer_size) 
        
        self.time_diff =  None
        
        self.buffer_size = buffer_size
        
        self.train_index = deque(maxlen = int(buffer_size * train_test_ratio ) )  
        
        self.validation_index = deque(maxlen = int(buffer_size * (1 - train_test_ratio )) ) 
        
        
    def remember(self, agent_actual_memory):
    
        
        '''
        self.states.append( {  k : v for k, v in zip( env.observations, state) } )  
        
        action_names = env.actions + list( ["oveFan_u"] + ["ovePum_u"] )
        
        action = action.detach().clone().numpy()
        
        temp =  np.where(discrete_action == 0, 0 , 1) 
        
        action = np.concatenate( [ action, temp , temp ], axis = 1  ).reshape(-1) #we concatenate temp twice with same value because fan and pum have same defualt values at any given time.
        
        self.actions.append( {  k:v for k, v in zip( action_names, action) } )
        
        self.next_states.append( {  k:v for k, v in zip( env.observations, next_state) } )
        
        self.rewards.append(  {"reward": reward}  )
        '''
        
        self.states.append( agent_actual_memory.states[-1] )
        
        self.actions.append( agent_actual_memory.actions[-1] )
        
        self.next_states.append( agent_actual_memory.next_states[-1] )
        
        self.rewards.append( agent_actual_memory.rewards[-1] )
        
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
        


'''
class Environment_Memory_Test():
    
    def __init__(self, buffer_size ): #was 200   
        
        self.name = "Test"
        
        self.time_diff =  None
    
        
    def remember(self, states, actions, discrete_actions, env):
        
        self.states = pd.DataFrame(states.numpy(), columns=env.observations)
        
        action_names = env.actions + list( ["oveFan_u"] + ["ovePum_u"] )
        
        actions = actions.detach().clone().numpy()
        
        temp =  np.where(discrete_actions == 0, 0 , 1) 
        
        actions = np.concatenate( [ actions, temp , temp ], axis = 1  )
        
        self.actions = pd.DataFrame(actions, columns = action_names )
        

    
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
'''