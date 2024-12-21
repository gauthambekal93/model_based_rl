# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:18:33 2024

@author: gauthambekal93
"""
import os
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V23/Code')

import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn

import torch.optim as optim
import torch.distributions as dist

from collections import deque  

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  

#num_tasks = 3

#we will store the complete data instead of just whats just required.
class Environment_Memory():
    
    def __init__(self, buffer_size = 200): #was 5000   
        
        self.train_states = deque(maxlen=buffer_size) 
        self.train_actions = deque(maxlen=buffer_size) 
        self.train_rewards = deque(maxlen=buffer_size) 
        self.train_next_states = deque(maxlen=buffer_size) 
        
        self.test_states = deque(maxlen=buffer_size) 
        self.test_actions = deque(maxlen=buffer_size) 
        self.test_rewards = deque(maxlen=buffer_size) 
        self.test_next_states = deque(maxlen=buffer_size) 
        
        self.buffer_size = buffer_size
        
        self.X_min, self.X_max , self.y_min, self.y_max = None, None, None, None
        
    def remember(self, i_episode, state, action, reward, next_state ):
        

        if random.random() <= 0.8:   #a uniform value between 0 and 1 is sampled, 80 % chance we are having training data. 
            
            self.train_states.append(torch.tensor( state ).reshape(1,-1) )
            self.train_actions.append(action.detach().clone())
            self.train_rewards.append(torch.tensor(reward).reshape(1,1) )  
            self.train_next_states.append(  torch.tensor( next_state ).reshape(1,-1) ) 
            
        else:
            
            self.test_states.append(torch.tensor( state ).reshape(1,-1) )
            self.test_actions.append(action.detach().clone())
            self.test_rewards.append( torch.tensor(reward).reshape(1,1) )   
            self.test_next_states.append( torch.tensor( next_state ).reshape(1,-1) ) 
    
        
    def get_dataset(self):
       
       train_X = torch.cat( [ torch.cat(list(self.train_states), dim=0)         , torch.cat(list(self.train_actions), dim=0) ], dim =1 )
       train_y = torch.cat( [ torch.cat(list(self.train_next_states), dim=0) [:,  [1, -1] ]    , torch.cat(list(self.train_rewards), dim=0) ], dim =1 )
       
       test_X = torch.cat( [ torch.cat(list(self.test_states), dim=0)         , torch.cat(list(self.test_actions), dim=0) ], dim =1 )
       test_y = torch.cat( [ torch.cat(list(self.test_next_states), dim=0)[:,  [1, -1] ]    , torch.cat(list(self.test_rewards), dim=0) ], dim =1 )
       
       
       self.X_min = train_X. min(dim=0, keepdim=True)[0]  
       
       self.X_max = train_X. max(dim=0, keepdim=True)[0]  
       
       self.y_min = train_y. min(dim=0, keepdim=True)[0]  
       
       self.y_max = train_y. max(dim=0, keepdim=True)[0]  
       
       return (train_X -self.X_min)  / (self.X_max - self.X_min), (train_y -self.y_min)  / (self.y_max- self.y_min), (test_X - self.X_min)  / (self.X_max - self.X_min ), (test_y -self.y_min)  / (self.y_max- self.y_min)
    
          
    
    def sample_random_states(self, sample_size = 1 ):
    
        random_numbers = torch.randint(0, len(self.train_states), (sample_size,))
            
        sampled_states = torch.cat(list(self.train_states), dim = 0)[random_numbers] - self.X_min / (self.X_max - self.X_min)
        
        return sampled_states


    def memory_size(self):
         return len(self.train_states) + len(self.test_states)
     
    def is_full(self):
        return self.buffer_size == self.memory_size()

        


class Target_Model():
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        self.weight1 = torch.rand(input_dim, hidden_dim)
        
        self.bias1 = torch.rand( (1, hidden_dim) )
        
        self.weight2 = torch.rand(hidden_dim, hidden_dim)
        
        self.bias2 = torch.rand( (1, hidden_dim) )
        
        self.weight3 = torch.rand(hidden_dim, output_dim )
        
        self.bias3 = torch.rand( (1, output_dim) )
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
    def predict_target(self, X):
        
        logits = self.leaky_relu (  torch.matmul ( X, self.weight1)  + self.bias1 )  
        
        logits = self.leaky_relu (  torch.matmul ( logits, self.weight2 ) + self.bias2 )
        
        logits = self.leaky_relu  ( torch.matmul ( logits, self.weight3 ) + self.bias3 )
        
        return logits
   
    def update_params(self, weights, bias ):
        
        self.weight1 = weights[0].reshape(self.weight1.shape).clone()
        self.bias1 = bias[0].reshape(self.bias1.shape).clone()
        
        self.weight2 = weights[1].reshape(self.weight2.shape).clone()
        self.bias2 = bias[1].reshape(self.bias2.shape).clone()
        
        self.weight3 = weights[2].reshape(self.weight3.shape).clone()
        self.bias3 = bias[2].reshape(self.bias3.shape).clone()
        
        


class Hypernet(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, w1_dim, b1_dim, w2_dim, b2_dim, w3_dim, b3_dim, num_tasks, num_layers  ):
        super().__init__()
        
        self.common1 = nn.Linear(input_dim, hidden_dim)
        
        self.common2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.weight1 = nn.Linear(hidden_dim, w1_dim * 2)
        
        self.bias1 = nn.Linear(hidden_dim, b1_dim * 2)
        
        self.weight2 = nn.Linear(hidden_dim, w2_dim * 2)
        
        self.bias2 = nn.Linear(hidden_dim, b2_dim * 2)
        
        self.weight3 = nn.Linear(hidden_dim, w3_dim * 2)
        
        self.bias3 = nn.Linear(hidden_dim, b3_dim * 2)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
        self.normal_dist = dist.Normal(0, 1)

        self.W_mus, self.W_stds, self.b_mus, self.b_stds = [], [], [] , []
        
        self.num_tasks = num_tasks
        
        self.num_layers = num_layers
        
    def task_conditioned(self, X, layer_no):
       
       logits = self.leaky_relu ( self.common1(X) )
       
       logits = self.leaky_relu ( self.common2(logits) )
       
       if layer_no ==0 :
           
           w_logits = self.leaky_relu (self.weight1 (logits)) 
           
           b_logits = self.leaky_relu (self.bias1 (logits))
           
       if layer_no == 1 :
           
           w_logits = self.leaky_relu (self.weight2 (logits)) 
           
           b_logits = self.leaky_relu (self.bias2 (logits))
           
       if layer_no == 2 :
           
           w_logits = self.leaky_relu (self.weight3 (logits)) 
           
           b_logits = self.leaky_relu (self.bias3 (logits))
        
       
       W_mu, W_std = torch.chunk(w_logits, 2, dim=1)
   
       W_std = torch.exp(W_std)
       
       b_mu, b_std = torch.chunk(b_logits, 2, dim=1)
   
       b_std = torch.exp(b_std)
       
       return W_mu, W_std, b_mu, b_std
   
    
   
    def generate_distribution(self, W_mu, W_std, b_mu, b_std , sample_size):
       
       samples = self.normal_dist.sample(( sample_size, W_mu.shape[1] ))  #sample from unit normal distribution
       
       w_z =  ( W_mu ) + (samples * W_std)
       
       #w_z = torch.mean(w_z, dim =0 ).reshape(1, -1)
       
       samples = self.normal_dist.sample(( sample_size, b_mu.shape[1] ))  #sample from unit normal distribution
       
       b_z = ( b_mu ) + (samples * b_std)
       
       #b_z = torch.mean(b_z, dim =0 ).reshape(1, -1)
       
       return w_z, b_z 
   
    
    
    def generate_weights_bias(self, task_index, sample_size = 1):
    
        
        task_id =  torch.nn.functional.one_hot( torch.tensor(task_index) , num_classes = self.num_tasks)   
        
        weights, bias = [], []
        
        self.W_mus, self.W_stds, self.b_mus, self.b_stds = [], [], [] , []
        
        for i in range(0, self.num_layers):
            
            layer_id = torch.nn.functional.one_hot( torch.tensor(i) , num_classes = self.num_layers)    
            
            X = torch.cat( [ task_id, layer_id ] ).to(dtype=torch.float32)
            
            if X.dim() ==1: 
                X = X.reshape(1,-1)
        
            W_mu, W_std, b_mu, b_std = self.task_conditioned( X, i )        
            
            self.W_mus.append(W_mu)
            
            self.W_stds.append(W_std)
            
            self.b_mus.append(b_mu)
            
            self.b_stds.append(b_std)
            
            W, b = self.generate_distribution(W_mu, W_std, b_mu, b_std, sample_size )
            
            weights.append(W )
            
            bias.append(b)
            
        return weights, bias
           