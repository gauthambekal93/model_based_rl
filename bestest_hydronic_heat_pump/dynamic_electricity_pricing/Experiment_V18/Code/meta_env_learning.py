# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:35:00 2024

@author: gauthambekal93
"""


import numpy as np
#import matplotlib.pyplot as plt

import random
#from collections import deque  
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#from torch.distributions import Categorical
from simulation_environments import bestest_hydronic_heat_pump

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  



class Environment_Model(nn.Module):
    
    def __init__(self, input_dims, n_actions, h_size):
        
        super(Environment_Model, self).__init__()
        
        self.n_actions = n_actions
        
        self.lr_inner, self.lr_outer = 0.01, 0.001
        
        self.input_layer1 = nn.Linear( input_dims + n_actions , h_size)
        
        self.hidden_layer1 =  nn.Linear( h_size, h_size )
        
        self.hidden_layer2 =  nn.Linear( h_size, h_size )
        
        self.hidden_next_state =  nn.Linear( h_size, input_dims )
       
        self.hidden_reward =  nn.Linear( h_size, 1 )
        

        
    def forward(self, X):
        
        X = torch.tensor(X, dtype=torch.float)
        
        X = F.leaky_relu(self.hidden_layer2(F.leaky_relu(self.hidden_layer1( F.leaky_relu(self.input_layer1( X )) ) ) ) )
        
        pred_next_state =  self.hidden_next_state( X )
        
        pred_reward = self.hidden_reward ( X )
        
        return  pred_next_state, pred_reward
        




class Meta_Learning:  
    
    def __init__(self):
        
        _, env_attributes  = bestest_hydronic_heat_pump()
        self.model = Environment_Model( int(env_attributes["state_space"]), int(env_attributes["action_bins"]) , int(env_attributes["h_size"]) )     
        self.optimizer = optim.Adam( self.model.parameters(), lr= 0.001 )
    
    def calc_loss(self, X, Y, updated_params = None):
       
       Y = torch.tensor(Y, dtype=torch.float)
       
       if updated_params:
           pred_next_state, pred_reward  = self.model(X, params=updated_params)

       pred_reward = pred_reward.reshape(-1)   
                             
       next_state = Y[:,:-1]
       
       reward = Y[:, -1].reshape(-1)
       
       next_state_loss = torch.mean((pred_next_state- next_state)**2) 
       
       reward_loss =  torch.mean((pred_reward- reward)**2) 
       
       return next_state_loss, reward_loss


    def inner_update(self, batch_data):  #
       
       """Inner loop: perform task-specific parameter update."""
       
       X, Y = batch_data[:, : self.model.input_layer1.in_features ], batch_data[:, self.model.input_layer1.in_features: ]
       #action needs to be converted to one hot encoded
       
       next_state_loss, reward_loss = self.calc_loss(X, Y)
      
       # Compute gradients and update task-specific parameters (theta2)
       grads = torch.autograd.grad(next_state_loss + reward_loss , self.model.parameters(), create_graph=True)
       
       updated_params = { name: param - self.lr_inner * grad  for (name, param), grad in zip(self.model.named_parameters(), grads)  }
       
       return updated_params
    
    

    def outer_update(self, task_batch):
       """Outer loop: meta-update across multiple tasks."""
       
       for k, v in task_batch.items():
           indices  = np.random.permutation(len(v))
           
           train_index , test_index = indices[ : int(0.80* len(indices ) ) ], indices[int(0.80* len(indices ) ) : ]
           
           train_data, test_data = v[train_index], v[test_index]
           
           num_samples, batch_size, total_loss = len(train_data), 50 , 0
            
           for i in range(0, num_samples, batch_size):
               
               train_batch = train_data[i:i + batch_size]
               
               updated_params = self.inner_update(train_batch)
               
               X, Y = test_data[:, : self.model.input_layer1.in_features ], test_data[:, self.model.input_layer1.in_features: ]
               
               next_state_loss, reward_loss = self.calc_loss(X, Y, updated_params)
               
               meta_loss = next_state_loss + reward_loss
             
               total_loss += meta_loss.item()

               # Outer loop: update original model parameters using meta-loss
               self.optimizer.zero_grad()
               
               meta_loss.backward()
               
               self.optimizer.step()
               
           
           


class Env_Memory:
    
    def __init__(self, n_actions):
    
        self.task_data = {}
        self.n_actions = n_actions
        
    def initialize_new_task(self, task_no):

        self.task_data[task_no] = []
        
    def remember(self, task_no, state, action, reward, new_state):  #this needs to be updated since we need all the states running in trajectory
        
        state = torch.tensor(state)
        action = torch.nn.functional.one_hot ( torch.tensor(action, dtype =torch.long), num_classes = self.n_actions ).reshape(-1)
        reward = torch.tensor([reward])
        new_state = torch.tensor(new_state)
        
        self.task_data[task_no] = self.task_data[task_no] + [ torch.cat([state, action, reward ,new_state], dim=0) ]
    
        
      
        
    def clear_memory(self):

        self.task_data = {}
    
    def sample_memory(self, task_no ): 
        
        indices = np.arange(1, task_no + 1)
       
        return { k: torch.stack(self.task_data[k], dim = 0) for k in indices }
        

        
    









