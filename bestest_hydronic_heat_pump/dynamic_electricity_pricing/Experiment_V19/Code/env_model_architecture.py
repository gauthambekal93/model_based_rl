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

import os 
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V19/Code')
import json

with open('all_paths.json', 'r') as openfile:  json_data = json.load(openfile)
exp_path = json_data['experiment_path']

import time
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import random
#from collections import deque  
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist

from simulation_environments import bestest_hydronic_heat_pump
from agent_actor_critic import Actor, Critic

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  
from torch.distributions import Categorical



class Env_Memory:
    
    def __init__(self, n_actions):
    
        #self.task_data = {}
        self.n_actions = n_actions
        
    def initialize_new_task(self):

        #self.task_data[task_no] = []
        self.task_data = []
        
    def remember(self, state, action, reward, new_state):  #this needs to be updated since we need all the states running in trajectory
           
        
        state = torch.tensor(state)
        action = torch.nn.functional.one_hot ( torch.tensor(action, dtype =torch.long), num_classes = self.n_actions ).reshape(-1)
        new_state = torch.tensor(new_state)
        reward = torch.tensor([reward])
        
        #self.task_data[task_no] = self.task_data[task_no] + [ torch.cat([state, action ,new_state, reward], dim=0) ]
        self.task_data = self.task_data + [ torch.cat([state, action ,new_state, reward], dim=0) ]
    
        
    def clear_memory(self):

        #self.task_data = {}
        self.task_data = []
    
    def sample_memory(self ): 
        
        #indices = np.arange(1, task_no + 1)
       
        #return { k: torch.stack(self.task_data[k], dim = 0) for k in indices }
        
        return self.task_data   
        

    def save_to_csv(self):
        
        #tensor_numpy = np.stack(self.task_data[ task_no ] , axis = 0)
        tensor_numpy = np.stack(self.task_data , axis = 0)
        
        df = pd.DataFrame(tensor_numpy)
        
        df.to_csv(exp_path+"/trajectory_data/tensor_data"+".csv", index=False) 
        
        
        
class Bayesian_Env_Model(nn.Module):
    
    #def __init__(self, state_dims, n_actions, h_size     state_dim, n_actions, hidden_dim, output_dim, latent_dim    ):
        
    def __init__(self, input_dim, output_dim, latent_dim , hidden_dim   ):      
        super(Bayesian_Env_Model, self).__init__()
    
        
        self.latent_dim = latent_dim
        
        self.encoder_layer1 = nn.Linear(  input_dim , hidden_dim)
        
        self.encoder_layer2 =nn.Linear(hidden_dim, latent_dim * 2 ) #latent_dim * 2  is because we need mean and standard deviation for every dimension of latent variable
        
        self.decoder_layer1 = nn.Linear( latent_dim, hidden_dim)
        
        self.decoder_layer2 =nn.Linear(hidden_dim, output_dim * 2 ) #input_dim * 2  is because we need mean and standard deviation for every dimension of input variable
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
        self.normal_dist = dist.Normal(0, 1)
        
        self.no_samples = 1000  # was 10000 
        
    def forward_encoder(self, x):    
        logits = self.leaky_relu ( self.encoder_layer1(x) )
        logits = self.leaky_relu( self.encoder_layer2(logits) )
    
        mu, log_std = torch.chunk(logits, 2, dim=1)
        std = torch.exp(log_std)
        
        return mu, std
    
    def forward_decoder(self, z):    
        logits = self.leaky_relu( self.decoder_layer1(z) )
        logits = self.leaky_relu ( self.decoder_layer2(logits) )
        
        mu, log_std = torch.chunk(logits, 2, dim=1)
        std = torch.exp(log_std)
        
        return mu, std

    def sample_latent_variable(self, mu, std, z_per_dist = 1):   #Here batch size can be obtained from directly using number of mu or std. Else can throw error, need to change!!!
        
        #samples = self.normal_dist.sample(( batch_size, self.latent_dim ))  #sample from unit normal distribution
        
        if z_per_dist > 1:
            samples = self.normal_dist.sample(( z_per_dist , mu.shape[1] ))
        else:
            samples = self.normal_dist.sample(( mu.shape[0] , mu.shape[1] ))  #sample from unit normal distribution
        
        z =  ( mu ) + (samples * std)      #reparameterization trick
        
        return z
    
        
    def kl_divergence_gaussians(self, mu1, std1, mu2, std2):
        # Variance is the square of the standard deviation
        var1 = std1 ** 2
        var2 = std2 ** 2
        
        # KL Divergence formula for two Gaussians
        kl_div = torch.log(std2 / std1) + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5
        return kl_div


    def loss_calculation(self, x , y, kl_weight, batch_size):
        
        mu_encoder, std_encoder =  self.forward_encoder(x)
        
        z = self.sample_latent_variable(mu_encoder, std_encoder)
        
        mu_decoder, std_decoder  = self.forward_decoder( z )
        
        output_dist = dist.Normal(mu_decoder, std_decoder )  # std_dev must be positive, so we take the absolute value
    
        # Compute the negative log likelihood for all input data points
        log_likelihood =  output_dist.log_prob(y).mean()  # Sum of log probabilities for all data points
        
        #prior_mu , prior_std = torch.zeros((batch_size, self.latent_dim )), torch.ones((batch_size, self.latent_dim ))
        
        prior_mu , prior_std = torch.zeros(( mu_encoder.shape[0] , mu_encoder.shape[1]  )), torch.ones((mu_encoder.shape[0], mu_encoder.shape[1] ))
        
        kl_div = self.kl_divergence_gaussians( mu_encoder, std_encoder, prior_mu , prior_std ).mean() 
        
        elbo =  log_likelihood - kl_weight  * kl_div
        
        loss = - 1.0 * elbo
        
        return loss
    

        
    


