# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:35:00 2024

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

    def sample_latent_variable(self, mu, std, batch_size):
        
        samples = self.normal_dist.sample(( batch_size, self.latent_dim ))  #sample from unit normal distribution
        
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
        
        z = self.sample_latent_variable(mu_encoder, std_encoder, batch_size)
        
        mu_decoder, std_decoder  = self.forward_decoder( z )
        
        output_dist = dist.Normal(mu_decoder, std_decoder )  # std_dev must be positive, so we take the absolute value
    
        # Compute the negative log likelihood for all input data points
        log_likelihood =  output_dist.log_prob(y).mean()  # Sum of log probabilities for all data points
        
        prior_mu , prior_std = torch.zeros((batch_size, self.latent_dim )), torch.ones((batch_size, self.latent_dim ))
        
        kl_div = self.kl_divergence_gaussians( mu_encoder, std_encoder, prior_mu , prior_std ).mean() 
        
        elbo =  log_likelihood - kl_weight  * kl_div
        
        loss = - 1.0 * elbo
        
        return loss
    
    
    def calculate_metrics(self, batch_data):  #need to fix this function !!!
        
        batch_uncertanity = []
        
        for data in batch_data:
            
            mu_encoder, std_encoder =  self.forward_encoder(data.reshape(1, -1))
            
            z_dist = self.sample_latent_variable(mu_encoder, std_encoder, 1000)
    
            mu_decoder, _  = self.forward_decoder(  z_dist )
            
            uncertanity = mu_decoder.std(dim = 0)
            
            point_values = torch.mean(mu_decoder, dim = 0)
            
            room_tmp, dry_bulb_tmp, reward = point_values[0] , point_values[ 1], point_values[2]
            
            batch_uncertanity.append(uncertanity[0].item())   #here we calculated only for room temperature
            
        return np.mean(batch_uncertanity)
    
    
    


                
def get_data():
    data = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V19/trajectory_data/tensor_data.csv' , index_col= False)
    
    data = data.values
    
    data = torch.tensor(data, dtype = torch.float32 )
    
    return data

def save_model():
    torch.save(env_model.state_dict(), r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V19/Models/bayesian_model.pth')

def load_model():
    env_model.load_state_dict(torch.load( r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V19/Models/bayesian_model.pth' ))
    
def train_model():    
    for epoch in range(1, epochs):
        
        print("Epoch ", epoch)
        
        total_loss = []
        
        #kl_weight = min(1.0, epoch / epochs)
        
        for batch in range( int( len(train_x) / batch_size) ):
            
            index = torch.randint(low = 0, high = len(train_x) , size=(batch_size,))
            
            batch_x, batch_y = train_x[index], train_y[index]
            
            loss = env_model.loss_calculation(batch_x, batch_y, kl_weight, batch_size )
            
            total_loss.append( loss.item() )
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
        
        print("Epoch ", epoch, "Loss ", np.mean( total_loss ) )
        
   
    

def initialize_trajectory():
    env, env_attributes  = bestest_hydronic_heat_pump()
    
    #for actor here we did not consider time step as input
    actor = Actor(10,  int(env_attributes["action_bins"]), int(env_attributes["h_size"]), "cpu", env_attributes["no_of_action_types"]).to("cpu") 
    state = env.reset()[0]
    
    return actor, state


    
def generate_trajectory_data( state,  kl_weight):
    
    action = actor(state)[0].argmax()
    
    action = torch.nn.functional.one_hot ( torch.tensor(action, dtype =torch.long), num_classes = len(action) ).reshape(-1)
    
    x =  torch.cat([state, action], dim= 0 )
    
    mu_encoder, std_encoder =  env_model.forward_encoder(x)
    
    z_dist = env_model.sample_latent_variable(mu_encoder, std_encoder, 1000)

    mu_decoder, _  = env_model.forward_decoder(  z_dist )
    
    uncertanity = mu_decoder.std(dim = 0)
    
    point_values = torch.mean(mu_decoder, dim = 0)
    
    room_tmp, dry_bulb_tmp, reward = point_values[0] , point_values[ 1], point_values[2]
    
    return mu_decoder, uncertanity, room_tmp, dry_bulb_tmp, reward


if __name__ == "__main__":
    
    start_time = time.time()
    
    actor, state = initialize_trajectory() #Actor( s_size=10, a_size = 11, h_size = 100 , device = "cpu", no_of_action_types = 1 )
    
    data = get_data()
    
    X, Y =  data[:, 1: 22] ,   data[:,  [23, 32, 33] ] #data[:, [22, 32, 33] ]  #The indexes needs to be cheked
    
    indices, split_idx = torch.randperm(X.shape[0]),  int( X.shape[0] * 0.80 )
    
    train_x, train_y =  X[  indices[ :split_idx] ], Y[  indices[ :split_idx] ]

    test_x, test_y = X[  indices[split_idx: ] ], Y[  indices[split_idx: ] ]
    
    input_dim , output_dim = X.shape[1], Y.shape[1]
    
    latent_dim, hidden_dim  = input_dim * 5, input_dim * 5
    
    epochs, batch_size = 5000, 100   #was 1000, 100 
    
    kl_weight = 0.01
    
    env_model = Bayesian_Env_Model(input_dim, output_dim, latent_dim , hidden_dim)
    
    optimizer = optim.Adam( env_model.parameters(), lr= 0.0001 )   

    loss_test = env_model.loss_calculation(test_x, test_y, kl_weight, test_x.shape[0] )
    
    uncertanity = env_model.calculate_metrics( test_x  )
    
    print("Before training: Loss {0}, Uncertanity {1}". format(loss_test,  uncertanity) )
    
    train_model()
    
    loss_test = env_model.loss_calculation(test_x, test_y, kl_weight, test_x.shape[0] )
    
    uncertanity = env_model.calculate_metrics( test_x  )
    
    print("After training: Loss {0}, Uncertanity {1}".format( loss_test,  uncertanity) )
    






