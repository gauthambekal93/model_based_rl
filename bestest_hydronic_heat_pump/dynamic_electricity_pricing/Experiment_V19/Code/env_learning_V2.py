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
from env_model_architecture_V2 import Bayesian_Env_Model, Env_Memory

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  
from torch.distributions import Categorical

    
    
class train_environment:
    
    def __init__(self, model_type ):
        
        self.kl_weight = 0.1 #was 0.01, 0.1, 1
        self.model_type = model_type
        
        if model_type == "room_temperature":
            input_dim , output_dim = 22 , 1
            
        if model_type == "dry_bulb_temperature":
            input_dim , output_dim = 11 , 1
        
        if model_type == "rewards":
            input_dim , output_dim = 22 , 1
            
        latent_dim, hidden_dim  =  input_dim * 5, input_dim * 5  #was 3 , 20 
        
        self.epochs, self.batch_size = 200, 100 #3000, 100  #seems like we should have more epochs for further training
        
        self.model = Bayesian_Env_Model(input_dim, output_dim, latent_dim , hidden_dim)
        
        self.optimizer = optim.Adam( self.model.parameters(), lr= 0.0001 )     #wad   lr= 0.00002 for rewards
        
        
    
    def save_model(self):
        torch.save(self.model.state_dict(), r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V19/Models/'+str(self.model_type)+'_'+str(self.epochs)+'.pth')
    
    
    def load_model(self):
        self.model.load_state_dict(torch.load( r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V19/Models/'+str(self.model_type)+'_'+str(self.epochs)+'.pth' ))
        self.create_dataset()
        
    def create_dataset(self):
        
        data = Env_Memory().sample_memory()
        
        data = torch.tensor(data.values, dtype = torch.float32 )
        
        if self.model_type == "room_temperature":
            
            X, Y =  data[:, : 22] ,  data[:, 23:24 ]     
        
        if self.model_type == "dry_bulb_temperature":
            X, Y =  data[:, : 11] ,  data[:, 32:33 ]      # y does not depend on actions
         
        if self.model_type == "rewards":
            X, Y =  data[:, : 22] ,  data[:, 33:34 ]           
        
        #self.x, self.y =  X, Y
        
        indices, split_idx = torch.randperm(X.shape[0]),  int( X.shape[0] * 0.80 )
            
        self.train_x, self.train_y =  X[  indices[ :split_idx] ], Y[  indices[ :split_idx] ]
        
        self.test_x, self.test_y = X[  indices[split_idx: ] ], Y[  indices[split_idx: ] ]      
        
        
    def train_model(self):    
        
        self.create_dataset()
        
        for epoch in range(1, self.epochs):
            
            print("Epoch ", epoch)
            
            total_loss = []
    
            
            for batch in range( int( len(self.train_x) / self.batch_size) ):
                
                index = torch.randint(low = 0, high = len(self.train_x) , size=(self.batch_size,))
                
                batch_x, batch_y = self.train_x[index], self.train_y[index]
                
                #this is typically kept as 1 when training the model, since we sample one z value per input during training.
                loss = self.model.loss_calculation(batch_x, batch_y, self.kl_weight, self.batch_size, no_of_z_samples =1 )
                
                total_loss.append( loss.item() )
                
                self.optimizer.zero_grad()
                
                loss.backward()
                
                self.optimizer.step()
            
            print("Epoch ", epoch, "Loss ", np.mean( total_loss ) )
            
        self.save_model()    
       
        
    def calculate_performance_metrics(self):  
        
        mu_z, std_z = self.model.forward_encoder(self.test_x)
        
        z_samples = self.model.sample_latent_variable(mu_z, std_z, no_of_z_samples = 1000)  #this will be of shape (batch_size, no. of samples, z dim)
        
        mu_y_samples, _  = self.model.forward_decoder(  z_samples )
        
        mu_y = torch.mean(mu_y_samples, dim = 1) 
        
        uncertanity = torch.mean( torch.std(mu_y_samples, dim = 1), dim =0 )  
        
        mse = nn.MSELoss()(mu_y, self.test_y) 
        
        return mse, uncertanity
    

        
    
def load_actor():
    checkpoint = torch.load( r"C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V17/Summary_Results_Models8/actor_model_500_.pkl")
    actor.load_state_dict(checkpoint['model_state_dict'])
    
    

def prediction_and_uncertanity(model, data):
     
        mu_z, std_z = model.model.forward_encoder(data)
        
        #z_sample = model.model.sample_latent_variable(mu_z, std_z, no_of_z_samples = 1) 
        
        #mu_y_sample, _  = model.model.forward_decoder(  z_sample )
        
        z_samples = model.model.sample_latent_variable(mu_z, std_z, no_of_z_samples = 1000)  #this will be of shape (batch_size, no. of samples, z dim)
        
        mu_y_samples, _  = model.model.forward_decoder(  z_samples )
        
        uncertanity = torch.mean( torch.std(mu_y_samples, dim = 1), dim =0 ) 
        
        mu_y_sample = torch.mean(mu_y_samples, dim = 1)
        
        return mu_y_sample, uncertanity


def run_trajectories(model_room_temp, model_dry_bulb_temp, model_rewards, state, action):
        
        room_temp, room_temp_uncertanity  = prediction_and_uncertanity(model_room_temp,  torch.cat([state, action], dim= 1 ) )
        
        dry_bulb_last_timestep, dry_bulb_last_timestep_uncertanity = prediction_and_uncertanity( model_dry_bulb_temp, state )
        
        reward, reward_uncertanity = prediction_and_uncertanity(model_rewards,  torch.cat([state, action], dim= 1 ) )
        
        time_step  = state[:, 0:1]  + 0.002976179999999995
        
        dry_bulb_tmp = state[:, 3:]
        
        next_state = torch.cat( [ time_step, room_temp , dry_bulb_tmp,  dry_bulb_last_timestep ], dim = 1)  #next_state, future_dry_temp in crrent time, dry_bulb_temp
        
        return next_state, reward,  torch.cat( [ room_temp_uncertanity, dry_bulb_last_timestep_uncertanity, reward_uncertanity ] , dim = 0)
    
    
def get_initial_uncertanity_range(model):

    model.create_dataset()
        
    mu_z, std_z = model.model.forward_encoder(model.train_x)
    
    z_samples = model.model.sample_latent_variable(mu_z, std_z, no_of_z_samples = 1000)  #this will be of shape (batch_size, no. of samples, z dim)
    
    mu_y_samples, _  = model.model.forward_decoder(  z_samples )
    
    uncertanities = torch.std(mu_y_samples, dim = 1)
    
    return torch.min( uncertanities, dim=0 )[0] ,  torch.max( uncertanities, dim=0 )[0]

 

if __name__ == "__main__":
    
    model_room_temp = train_environment(model_type = "room_temperature")
    
    #mse_room_temp, uncertanity_room_temp = model_room_temp.calculate_performance_metrics()
    
    #print("MSE and Uncertanity Room Temperature: ", mse_room_temp, uncertanity_room_temp)
    
    #model_room_temp.train_model()
    model_room_temp.load_model()
    
    mse_room_temp, uncertanity_room_temp = model_room_temp.calculate_performance_metrics()
    
    print("MSE and Uncertanity Room Temperature: ", mse_room_temp, uncertanity_room_temp)
    
    
    
    model_dry_bulb_temp = train_environment(model_type = "dry_bulb_temperature")
    
    #model_dry_bulb_temp.train_model()
    model_dry_bulb_temp.load_model()
    
    mse_dry_bulb_temp = model_dry_bulb_temp.calculate_performance_metrics()
    
    print("MSE Dry Bulb Temperature: ", mse_dry_bulb_temp )
    
    
    
    model_rewards = train_environment(model_type = "rewards")
    
    #model_rewards.train_model()
    model_rewards.load_model()
    
    mse_rewards = model_rewards.calculate_performance_metrics()
    
    print("MSE Rewards: ", mse_rewards )   #MSE REWARDS HAS COMPARITIVELY HIGH LOSS! MIGHT NEED MORE DATA OR MORE EPOCHS TO TRAIN
    
    
    
    env, env_attributes  = bestest_hydronic_heat_pump()
    
    actor = Actor(11,  int(env_attributes["action_bins"]), int(env_attributes["h_size"]), "cpu", env_attributes["no_of_action_types"]).to("cpu") 
    
    load_actor()
    
    len_of_trajectories =  300  #was 1000, 300
    
    state = torch.tensor( env.reset()[0],  dtype = torch.float32).reshape(1, -1)
    
    action = actor(state)[0].clone().detach()
    
    action = torch.nn.functional.one_hot ( action.argmax(), num_classes = action.shape[1] ).reshape(1,-1)
    
    uncertanity = []
    
    for t in range(1, len_of_trajectories):
         
         state, reward, tmp  =  run_trajectories( model_room_temp, model_dry_bulb_temp, model_rewards, state, action)
         
         action = actor(state)[0].clone().detach()
         
         action = torch.nn.functional.one_hot ( action.argmax(), num_classes = action.shape[1] ).reshape(1,-1)
         
         uncertanity.append( tmp )
         
    uncertanity = torch.stack(uncertanity, dim = 0) 
   


#total_sum = torch.sum( torch.exp( -1.0* uncertanity[:, 2] ), dim =0)
#probs = torch.exp( -1.0* uncertanity[:, 2] ) / total_sum





'''
def generate_random_number(low, high):
    return low + torch.rand(1).item() * (high - low)


tmp = 0
for i in range(0, 100):
    
    rand_no = generate_random_number( uncertanity[:, 2 ].min(),  uncertanity[:, 2 ].max() )
    #print(i,"  ", rand_no)
    if rand_no > 0.2:
       tmp += 1

'''


