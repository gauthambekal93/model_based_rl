# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:14:01 2024

@author: gauthambekal93
"""


import os
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V4/Code')

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
from torch.distributions import Categorical

#from examples.test_and_plot import test_agent, plot_results
# Decide the state-action space of your test case
#import random

# Seed for random starting times of episodes
seed = 42
random.seed(seed)
# Seed for random exploration and epsilon-greedy schedule
np.random.seed(seed)


class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size, device, no_of_action_types = 4):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, h_size)
        #self.fc4 = nn.Linear(h_size, a_size)
        self.fc4  = [ nn.Linear(h_size, a_size) for _ in range(no_of_action_types) ]
        self.device = device
        
    def forward(self, x):
        
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #x = self.fc4(x)
        x = [ F.softmax( fc(x), dim=1).cpu() for fc in self.fc4 ]
        #return F.softmax(x, dim=1)
        return x
    '''
    def act(self, state, compress_features = None):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        if compress_features:
            
            state = compress_features.feature_current_state1(state)
            
            state = compress_features.feature_current_state2(state)
            
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    '''
    
    def act(self, state, compress_features = None):
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
       '''
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        #probs = self.forward(state).cpu()
        probs_list = self.forward(state)
        
        all_actions = []
        all_log_prob = []
        
        for probs in probs_list:
            
            m = Categorical(probs)
            action = m.sample()
            all_actions.append(np.int32( action.item() ) )
            all_log_prob.append(m.log_prob(action))
        
        return all_actions, all_log_prob       
    
    
    def calculate_loss(self, rewards, saved_log_probs, max_t, gamma):
        
        returns = deque(maxlen=max_t)
        
        n_steps = len(rewards)
        
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft( gamma*disc_return_t + rewards[t]   )
       
        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

    
        policy_loss = []
        
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append( torch.cat( [-l * disc_return for l in log_prob] ).sum().reshape(1,-1) )
            
        policy_loss = torch.cat(policy_loss).sum()
        

        return policy_loss

    




