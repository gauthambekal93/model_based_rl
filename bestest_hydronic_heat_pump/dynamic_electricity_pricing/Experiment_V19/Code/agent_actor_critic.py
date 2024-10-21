# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:14:01 2024

@author: gauthambekal93
"""


import os
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V19/Code')

#import math
#import json

#import requests
from boptestGymEnv import BoptestGymEnv, DiscretizedActionWrapper, DiscretizedObservationWrapper, NormalizedObservationWrapper
from boptestGymEnv import BoptestGymEnvRewardClipping, BoptestGymEnvRewardWeightDiscomfort,  BoptestGymEnvRewardClipping
import numpy as np

import random
#from IPython.display import clear_output
#from collections import namedtuple
#from itertools import count
from collections import deque  
#import time  

#import numpy as np

#from collections import deque

#import matplotlib.pyplot as plt
# %matplotlib inline

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
#from torch.distributions import Categorical

#from examples.test_and_plot import test_agent, plot_results
# Decide the state-action space of your test case
#import random

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


class Agent_Memory:
    def __init__(self):
        self.states = []
        self.action_log_probs = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.value_preds = []
        self.source = []
        
    def remember(self, state, action, action_log_prob, reward, next_state, value_preds, source):
        self.states.append( torch.tensor(state) )
        self.actions.append(torch.tensor(action))
        self.action_log_probs.append( action_log_prob )
        self.rewards.append(torch.tensor(reward))
        self.next_states.append(torch.tensor(next_state))
        self.value_preds.append(value_preds)
        self.source.append(source)        

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.next_states = []
        self.value_preds = []
        self.source = []
        
    def sample_memory(self, sample_size = 2000 ):  #was 1000

        
        return (
                torch.stack(self.states, dim = 0), 
                torch.stack(self.actions, dim = 0), 
                torch.stack(self.action_log_probs, dim = 0),
                torch.stack(self.rewards, dim = 0).unsqueeze(dim=1), 
                torch.stack(self.next_states, dim = 0),    
                torch.stack(self.value_preds, dim = 0), 
                self.source
               )

    def memory_size(self):
         return len(self.states)



class Actor(nn.Module):
    
    def __init__(self, s_size, a_size, h_size, device, no_of_action_types = 4):
        super().__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3  = [ nn.Linear(h_size, a_size) for _ in range(no_of_action_types) ]
        self.device = device
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = [ F.softmax( fc(x), dim=1).cpu() for fc in self.fc3 ]
        
        return x   
   
    
   
class Critic(nn.Module):
    
    def __init__(self, s_size, h_size, device):
        super().__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3  = nn.Linear(h_size, 1 )
        self.device = device
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return x  



def select_action(x, critic, actor):
    
    x= torch.tensor(x).to("cpu").reshape(1,-1)
    
    action_logits = actor(x)
    
    state_value = critic(x).reshape(-1)
    
    all_actions, all_log_prob = [], []
    
      
    for probs in action_logits:
          
          m = Categorical(probs)
          action = m.sample()
          all_actions.append(np.int32( action.item() ) )
          all_log_prob.append(m.log_prob(action))
      
    
    all_log_prob = torch.cat(all_log_prob, dim=0)
    
    return ( all_actions, all_log_prob, state_value )    
    
    
        
        
def get_losses( action_log_probs, rewards, value_preds, gamma, lam, device, n_envs=1):
    
    T = len(rewards)
    advantages = torch.zeros(T, n_envs, device=device)

    # compute the advantages using GAE
    gae = 0.0
    for t in reversed(range(T - 1)):
        td_error = (
            rewards[t] + gamma  * value_preds[t + 1] - value_preds[t]
        )
        gae = td_error + gamma * lam * gae
        
        advantages[t] = gae

    # calculate the loss of the minibatch for actor and critic
    critic_loss = advantages.pow(2).mean()

   
    actor_loss =  -(advantages.detach() * action_log_probs).mean() #- ent_coef * entropy.mean()  
    
    return ( critic_loss, actor_loss)



