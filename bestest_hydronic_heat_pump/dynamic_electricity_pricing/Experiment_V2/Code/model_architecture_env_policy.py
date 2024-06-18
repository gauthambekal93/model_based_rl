# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 08:56:26 2024

@author: gauthambekal93
"""

# -*- coding: utf-8 -*-
"""

Created on Sun May 12 11:42:36 2024

@author: gauthambekal93
"""
import numpy as np
import torch
import random

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.distributions import Categorical
from collections import deque  


#random.seed(seed)
#from unsupervised_pretraining.reinforce.obtain_trajectories import get_trajectories

class Environment_Module(nn.Module):
    
    def __init__(self, obs_dims, n_actions=1, alpha=1, beta=0.2):
        
        input_dims = obs_dims+ n_actions
        
        h_size_1 = int(input_dims * 2)
        
        h_size_2 =  int(input_dims*2) #wasint(input_dims//2)
        
        super(Environment_Module, self).__init__()
        
        self.alpha = alpha
        
        self.beta = beta
        
        #next state prediction
        self.input_layer_next_state = nn.Linear( input_dims , h_size_1)
    
        self.hidden1_next_state =  nn.Linear( h_size_1, h_size_1 )
        
        self.hidden2_next_state =  nn.Linear( h_size_1, h_size_1 )
        
        self.hidden3_next_state =  nn.Linear( h_size_1, h_size_1 )
        
        self.next_state =  nn.Linear( h_size_1, obs_dims )
        
        
        #reward prediction
        self.input_layer_reward = nn.Linear( input_dims  , h_size_2)
        
        self.hidden1_reward =  nn.Linear( h_size_2, h_size_2 )
        
        self.hidden2_reward =  nn.Linear( h_size_2, h_size_2)
        
        self.hidden3_reward =  nn.Linear( h_size_2, h_size_2 )
        
        self.reward = nn.Linear( h_size_2, 1 )
        
        
        
    def forward(self, X):
        
        X = T.tensor(X, dtype=T.float)
        
        #X  = F.leaky_relu( self.hidden3 ( F.leaky_relu( self.hidden2 ( F.leaky_relu( self.hidden1( F.leaky_relu(self.input_layer(X) ) ) ) ) ) ) )
         
        
        pred_next_state = self.next_state(F.leaky_relu(self.hidden3_next_state (F.leaky_relu(self.hidden2_next_state(F.leaky_relu(self.hidden1_next_state (F.leaky_relu(self.input_layer_next_state(X)))))))))
        
        #pred_next_state =  self.next_state( F.leaky_relu( self.hidden_next_state(X) ) )
                
        pred_reward = self.reward(F.leaky_relu(self.hidden3_reward (F.leaky_relu(self.hidden2_reward(F.leaky_relu(self.hidden1_reward(F.leaky_relu( self.input_layer_reward (X)))))))))
        
        #pred_reward =  self.reward( F.leaky_relu( self.hidden_reward(X) ) )
        
        return  pred_next_state, pred_reward
        


    def calc_loss(self, X, Y):
        
        #if not isinstance(batch_dataset, np.ndarray): batch_dataset = np.array(batch_dataset)
    
        #if batch_dataset.ndim ==1: batch_dataset = batch_dataset.reshape(1, -1)       
        
        Y = T.tensor(Y, dtype=T.float)
        
        pred_next_state, pred_reward  = self.forward(X)

        #pred_next_state = pred_next_state.reshape(-1)   
        
        pred_reward = pred_reward.reshape(-1)   
                              
        #forward_loss = nn.MSELoss()
        
        next_state = Y[:,:-1]
        
        reward = Y[:, -1].reshape(-1)
        
        next_state_loss = torch.mean((pred_next_state- next_state)**2) # forward_loss(pred_next_state, next_state) 
        
        reward_loss =  torch.mean((pred_reward- reward)**2) #forward_loss(pred_reward, reward) 
        
        return next_state_loss, reward_loss





class Policy_Module(nn.Module):
    
    def __init__(self, s_size, a_size, h_size, device, no_of_action_types):
        
        super( Policy_Module, self).__init__()
        

        self.fc1 = nn.Linear(s_size, h_size)   #2 is for 2 possible next state  room temperature for the possible actions, 3 is for time, current temp and ambient temp
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, h_size)
        self.fc4  = [ nn.Linear(h_size, a_size) for _ in range(no_of_action_types) ]

        self.device = device
        
        #self.create_trajectories = get_trajectories()
         
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = [ F.softmax( fc(x), dim=1).cpu() for fc in self.fc4 ]

        return x
    
    

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

#y = torch.randn(1, 123)  # Example tensor with shape [1, 123]
#additional_value = torch.tensor([[0]])  # Tensor to concatenate with shape [1, 1]

# Concatenate along the second dimension (columns)
#concatenated_tensor = torch.cat([y, additional_value], dim=1)
#print(concatenated_tensor.shape)  # Should be [1, 124]



