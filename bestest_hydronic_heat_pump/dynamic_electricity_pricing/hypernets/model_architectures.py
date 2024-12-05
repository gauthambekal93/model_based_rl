# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:18:33 2024

@author: gauthambekal93
"""

import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn

import torch.optim as optim
import torch.distributions as dist

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  

num_tasks = 10

class Target_Model():
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        self.weight1 = torch.rand(input_dim, hidden_dim)
        
        self.bias1 = torch.rand( (1, hidden_dim) )
        
        self.weight2 = torch.rand(hidden_dim, hidden_dim)
        
        self.bias2 = torch.rand( (1, hidden_dim) )
        
        self.weight3 = torch.rand(hidden_dim, 1)
        
        self.bias3 = torch.rand( (1, 1) )
        
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
    
    def __init__(self, input_dim, hidden_dim, w1_dim, b1_dim, w2_dim, b2_dim, w3_dim, b3_dim  ):
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
    
        self.sample_size = 1
        
        self.W_mus, self.W_stds, self.b_mus, self.b_stds = [], [], [] , []
        
        
    def calculate_kl_divergence(self):
         
        kl_div = 0
        
        for i in range(len(self.W_mus)):
            
            prior_mu , prior_std = torch.zeros(self.W_mus[i].shape), torch.ones(self.W_stds[i].shape )
            
            var1 = self.W_stds[i] ** 2
            
            prior_var = prior_std ** 2
            
            kl_div += torch.sum( torch.log(prior_std / self.W_stds[i]) + (var1 + (self.W_mus[i] - prior_mu) ** 2) / (2 * prior_var) - 0.5, dim=1 )
            
            
            prior_mu , prior_std = torch.zeros(self.b_mus[i].shape), torch.ones(self.b_stds[i].shape )
            
            var1 = self.b_stds[i] ** 2
            
            prior_var = prior_std ** 2
            
            kl_div += torch.sum( torch.log(prior_std / self.b_stds[i]) + (var1 + (self.b_mus[i] - prior_mu) ** 2) / (2 * prior_var) - 0.5, dim=1 )
            
            
        return kl_div
     
    
    
    def generate_distribution(self, W_mu, W_std, b_mu, b_std ):
        
        samples = self.normal_dist.sample(( self.sample_size, W_mu.shape[1] ))  #sample from unit normal distribution
        
        w_z =  ( W_mu ) + (samples * W_std)
        
        samples = self.normal_dist.sample(( self.sample_size, b_mu.shape[1] ))  #sample from unit normal distribution
        
        b_z = ( b_mu ) + (samples * b_std)
        
        return w_z, b_z
    
    
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
   
    
   
    def generate_weights_bias(self, task_index, num_layers):
    
        
        task_id =  torch.nn.functional.one_hot( torch.tensor(task_index) , num_classes = num_tasks)   
        
        weights, bias = [], []
        
        self.W_mus, self.W_stds, self.b_mus, self.b_stds = [], [], [] , []
        
        for i in range(0, num_layers):
            
            layer_id = torch.nn.functional.one_hot( torch.tensor(i) , num_classes=num_layers)    
            
            X = torch.cat( [ task_id, layer_id ] ).to(dtype=torch.float32)
            
            if X.dim() ==1: 
                X = X.reshape(1,-1)
        
            W_mu, W_std, b_mu, b_std = self.task_conditioned( X, i )        
            
            self.W_mus.append(W_mu)
            
            self.W_stds.append(W_std)
            
            self.b_mus.append(b_mu)
            
            self.b_stds.append(b_std)
            
            W, b = self.generate_distribution(W_mu, W_std, b_mu, b_std )
            
            weights.append(W )
            
            bias.append(b)
            
        return weights, bias
           