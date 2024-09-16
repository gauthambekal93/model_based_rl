# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:14:01 2024

@author: gauthambekal93
"""


import os
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V16/Code')

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
      
    
    
    #all_log_prob = torch.stack(all_log_prob, dim =0)
    #all_actions = np.stack(all_actions, axis =0)
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



