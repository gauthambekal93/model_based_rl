# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:14:01 2024

@author: gauthambekal93
"""


import os
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V21/Code')


from boptestGymEnv import BoptestGymEnv, DiscretizedActionWrapper, DiscretizedObservationWrapper, NormalizedObservationWrapper
from boptestGymEnv import BoptestGymEnvRewardClipping, BoptestGymEnvRewardWeightDiscomfort,  BoptestGymEnvRewardClipping
import numpy as np

import random
from collections import deque  

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F



# Seed for random starting times of episodes
seed = 42
random.seed(seed)
# Seed for random exploration and epsilon-greedy schedule
np.random.seed(seed)
from torch.distributions import Categorical


seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  

import torch.distributions as dist
from torch.distributions import Normal


class Agent_Memory:
    def __init__(self, buffer_size = 5000):
        
        self.states = deque(maxlen=buffer_size) 
        self.actions = deque(maxlen=buffer_size) 
        self.rewards = deque(maxlen=buffer_size) 
        self.next_states = deque(maxlen=buffer_size) 
        self.done = deque(maxlen=buffer_size)
                                 
        
    def remember(self, i_episode, state, action, reward, next_state, done ):
        self.states.append(  torch.tensor( state ).reshape(1,-1) )
        self.actions.append(  action.detach().clone()  )
        self.rewards.append(torch.tensor(reward).reshape(1,-1))
        self.next_states.append(torch.tensor(next_state ).reshape(1,-1) )
        self.done.append(torch.tensor(done).reshape(1,-1))
        

    
    def sample_memory(self, sample_size = 200 ):  
        
        
        random_numbers = torch.randint(0, len(self.states), (sample_size,))
        
        return (
                torch.cat(list(self.states), dim = 0)[random_numbers] , 
                torch.cat(list(self.actions), dim = 0)[random_numbers], 
                torch.cat(list(self.rewards), dim = 0)[random_numbers], 
                torch.cat(list(self.next_states), dim = 0)[random_numbers],    
                torch.cat(list(self.done), dim = 0)[random_numbers]
               )

    def memory_size(self):
         return len(self.states)





class Actor(nn.Module):
    
    def __init__(self, s_size, h_size, bin_size, device, no_of_action_types = 4):
        super().__init__()
        
        self.normal_dist = dist.Normal(0, 1)
        
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3  = nn.Linear(h_size, 2)
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
        self.bins = np.linspace(-1, 1, bin_size)
        
        #self.fc3  = [ nn.Linear(h_size, a_size) for _ in range(no_of_action_types) ]
        
        self.device = device
    
    def forward(self, x):
        
        logits = self.leaky_relu(self.fc1(x))
        
        logits = self.leaky_relu(self.fc2(logits))
        
        logits = self.leaky_relu(self.fc3(logits))
        
        mu, log_std = torch.chunk(logits, 2, dim=1)
        
        std = torch.exp(log_std)
        
        return mu, std
        
    
    def get_action_log_probs(self, actions, mu, std):
       
        normal_dist = Normal(mu, std)
        
        action_log_probs = normal_dist.log_prob(actions) + nn.log(1 - self.tanh(actions) )
        
        return action_log_probs
    
    def discretize_action(self, continuous_actions):
        # Map each continuous action value to the closest bin
        discrete_actions = np.digitize(continuous_actions.detach().numpy(), self.bins) - 1
        
        return discrete_actions
    
    
    def select_action(self, state):
    
        if state.ndim ==1:
            state = state. reshape(1,-1)
        
        if not torch.is_tensor(state):    
            state = torch.tensor(state).to("cpu")

        mu, std = self.forward(state)
        
        noise_samples = self.normal_dist.sample( sample_shape = ( mu.shape[0], mu.shape[1] )  ) 
        
        continuous_actions =  self.tanh ( ( mu ) + (noise_samples * std) )
       
        action_log_probs = self.get_action_log_probs(continuous_actions, mu, std) 
       
        discrete_actions = self.discretize_action( continuous_actions)
        
        return continuous_actions, discrete_actions, action_log_probs
   
      
    
   

    
   
class Critic(nn.Module):
    
    def __init__(self, s_size, h_size, device):
        super().__init__()
        self.fc1 = nn.Linear( (s_size + 1 ) , h_size )   #1 is added for a continuous action
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3  = nn.Linear(h_size, 1 )
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.device = device
    
    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
         
        return x  
    
    def get_q_value(self, state, action):
           
        return self.forward(  torch.cat([state, action] , dim = 1) )








        



