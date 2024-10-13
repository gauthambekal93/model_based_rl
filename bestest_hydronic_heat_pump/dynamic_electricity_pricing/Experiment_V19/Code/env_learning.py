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
from env_model_architecture import Bayesian_Env_Model

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  
from torch.distributions import Categorical

    
    
class train_environment:
    
    def __init__(self, model_type ):
        
        self.kl_weight = 0.01
        self.model_type = model_type
        
        data = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V19/trajectory_data/tensor_data.csv' , index_col= False)
        
        data = torch.tensor(data.values, dtype = torch.float32 )
        
        if model_type == "room_temperature":
            X, Y =  data[:, : 22] ,  data[:, 23:24 ]     
        
        if model_type == "dry_bulb_temperature":
            X, Y =  data[:, : 11] ,  data[:, 32:33 ]      # y does not depend on actions
        
        if model_type == "rewards":
            X, Y =  data[:, : 22] ,  data[:, 33:34 ]   
            
        self.x, self.y =  X, Y
        
        indices, split_idx = torch.randperm(X.shape[0]),  int( X.shape[0] * 0.80 )
        
        self.train_x, self.train_y =  X[  indices[ :split_idx] ], Y[  indices[ :split_idx] ]
    
        self.test_x, self.test_y = X[  indices[split_idx: ] ], Y[  indices[split_idx: ] ]      
        
        input_dim , output_dim = X.shape[1], Y.shape[1]
        
        latent_dim, hidden_dim  = input_dim * 5, input_dim * 5
        
        self.epochs, self.batch_size = 2000, 100  
        
        self.model = Bayesian_Env_Model(input_dim, output_dim, latent_dim , hidden_dim)
        
        self.optimizer = optim.Adam( self.model.parameters(), lr= 0.0001 )   
       
        
    
    def save_model(self):
        torch.save(self.model.state_dict(), r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V19/Models/'+str(self.model_type)+'_'+str(self.epochs)+'.pth')
    
    
    def load_model(self):
        self.model.load_state_dict(torch.load( r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V19/Models/'+str(self.model_type)+'_'+str(self.epochs)+'.pth' ))
        
    
        
    def train_model(self):    
        for epoch in range(1, self.epochs):
            
            print("Epoch ", epoch)
            
            total_loss = []
    
            
            for batch in range( int( len(self.train_x) / self.batch_size) ):
                
                index = torch.randint(low = 0, high = len(self.train_x) , size=(self.batch_size,))
                
                batch_x, batch_y = self.train_x[index], self.train_y[index]
                
                loss = self.model.loss_calculation(batch_x, batch_y, self.kl_weight, self.batch_size )
                
                total_loss.append( loss.item() )
                
                self.optimizer.zero_grad()
                
                loss.backward()
                
                self.optimizer.step()
            
            print("Epoch ", epoch, "Loss ", np.mean( total_loss ) )
            
        self.save_model()    
       
    def calculate_performance_metrics(self):  
        
        mu_z, std_z = self.model.forward_encoder(self.test_x)
        
        z_dist = self.model.sample_latent_variable(mu_z, std_z)
        
        mu_y, std_y  = self.model.forward_decoder(  z_dist )
        
        mse_loss = nn.MSELoss()
        
        return mse_loss(mu_y, self.test_y) 
       
        
    
def load_actor():
    checkpoint = torch.load( r"C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V17/Summary_Results_Models8/actor_model_500_.pkl")
    actor.load_state_dict(checkpoint['model_state_dict'])
    
def common_compute(model, model_input, no_of_trajectories):
    
    mu_z, std_z =  model.model.forward_encoder( model_input )
    
    z_samples_one_model = model.model.sample_latent_variable(mu_z, std_z, no_of_trajectories)  
    
    mu_y, std_y  = model.model.forward_decoder(  z_samples_one_model )
    
    return mu_y, std_y



def initialize_trajectories():
   
    #state = env.reset()[0]
    
    #state = torch.tensor(state, dtype = torch.float32).reshape(1, -1)
    
    state = model_dry_bulb_temp.test_x[0].reshape(1,-1)
    
    action = actor(state)[0].clone().detach()
    
    #action = torch.nn.functional.one_hot ( action.argmax(), num_classes = action.shape[1] ).reshape(1,-1)
    
    room_temp, _ = common_compute(model_room_temp,  torch.cat([state, action], dim= 1 ), no_of_trajectories)
    
    dry_bulb_last_timestep, _ = common_compute(model_dry_bulb_temp, state, no_of_trajectories)
    
    rewards, _ = common_compute(model_rewards,  torch.cat([state, action], dim= 1 ), no_of_trajectories)
    
    #env_input =  torch.cat([state, action], dim= 1 )
    
    #mu_z, std_z =  env_model.forward_encoder( env_input )
    
    #z_samples_one_model = env_model.sample_latent_variable(mu_z, std_z, no_of_trajectories)  
    
    #mu_y, std_y  = env_model.forward_decoder(  z_samples_one_model )
    
    time_step  = state[:, 0].repeat(room_temp.shape[0] , 1) + 0.002976179999999995
    
    dry_bulb_tmp = state[:, 3:].repeat(room_temp.shape[0] , 1)
    
    #room_temp = mu_y[:,0].reshape(-1,1)
    
    #dry_bulb_tmp, dry_bulb_last_timestep = state[:, 3:].repeat(mu_y.shape[0] , 1), mu_y[:, 1].reshape(-1,1)
    
    next_state = torch.cat( [  time_step ,  room_temp  ,dry_bulb_tmp,  dry_bulb_last_timestep ] , dim = 1) 
    
    return  next_state


    
def run_trajectories( state):
    
    action = actor(state)[0].clone().detach()   #action values are changing very little for different states
    
    #action = torch.nn.functional.one_hot ( action.argmax( dim = 1), num_classes = action.shape[1] )
    
    room_temp, _ = common_compute(model_room_temp,  torch.cat([state, action], dim= 1 ), no_of_trajectories)
    
    dry_bulb_last_timestep, _ = common_compute(model_dry_bulb_temp, state, no_of_trajectories)
    
    rewards, _ = common_compute(model_rewards,  torch.cat([state, action], dim= 1 ), no_of_trajectories)
    
    
    #env_inputs =  torch.cat([state, action], dim= 1 )
   
    #mu_z , std_z =  env_model.forward_encoder( env_inputs )
   
    #z_sample_per_model = env_model.sample_latent_variable( mu_z, std_z )  
   
    #mu_y, std_y  = env_model.forward_decoder(  z_sample_per_model )
   

    time_step = state[:, 0:1] + 0.002976179999999995
    
    dry_bulb_tmp = state[:, 3:] 
    
    #room_temp = mu_y[:,0].reshape(-1,1)
    
    #dry_bulb_tmp, dry_bulb_last_timestep = state[:, 3:] , mu_y[:, 1 ].reshape(-1,1)
    
    next_state = torch.cat( [ time_step, room_temp , dry_bulb_tmp,  dry_bulb_last_timestep ], dim = 1)  #next_state, future_dry_temp in crrent time, dry_bulb_temp
    
    uncertanity =  torch.cat( [ room_temp.std(dim = 0).detach().clone(), dry_bulb_last_timestep.std(dim = 0).detach().clone(), rewards.std(dim = 0).detach().clone()], dim = 0)
    
    return next_state, uncertanity, room_temp
    


def start_simulation():
    
    state = initialize_trajectories()
    
    trajectory_uncertanity = []
    
    for t in range(2, len_of_trajectories):
        
        state, uncertanity, room_temp = run_trajectories( state )   #here we are sampling only one z per distribution but we have multiple such z distributions
        
        #print("Time step ", t, "Room Temp: ", room_temp, "Uncertanity: ",uncertanity[0] )
        print("Time step ", t, "Uncertanity: ",uncertanity )
        
        trajectory_uncertanity.append(uncertanity)
        
    return trajectory_uncertanity


if __name__ == "__main__":
    
    model_room_temp = train_environment(model_type = "room_temperature")
    
    #model_room_temp.train_model()
    model_room_temp.load_model()
    
    mse_room_temp = model_room_temp.calculate_performance_metrics()
    
    print("MSE Room Temperature: ", mse_room_temp)
    
    model_dry_bulb_temp = train_environment(model_type = "dry_bulb_temperature")
    
    model_dry_bulb_temp.load_model()
    #model_dry_bulb_temp.train_model()
    
    mse_dry_bulb_temp = model_dry_bulb_temp.calculate_performance_metrics()
    
    print("MSE Dry Bulb Temperature: ", mse_dry_bulb_temp )
    
    model_rewards = train_environment(model_type = "rewards")
    
    model_rewards.load_model()
    
    #model_rewards.train_model()
    
    mse_rewards = model_rewards.calculate_performance_metrics()
    
    print("MSE Rewards: ", mse_rewards )   #MSE REWARDS HAS COMPARITIVELY HIGH LOSS! MIGHT NEED MORE DATA OR MORE EPOCHS TO TRAIN
    
    env, env_attributes  = bestest_hydronic_heat_pump()
    
    actor = Actor(11,  int(env_attributes["action_bins"]), int(env_attributes["h_size"]), "cpu", env_attributes["no_of_action_types"]).to("cpu") 
    
    load_actor()
    
    no_of_trajectories, len_of_trajectories = 1000, 300  #was 1000, 300
    
    all_uncertanity = torch.stack(  start_simulation(),  dim = 0)

    #initialize_trajectories() 
   





