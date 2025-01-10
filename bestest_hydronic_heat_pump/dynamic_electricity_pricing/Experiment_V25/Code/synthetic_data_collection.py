# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 19:11:59 2025

@author: gauthambekal93
"""
import os
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V25/Code')

import numpy as np
#import pandas as pd
import random

import torch
import torch.nn as nn

import torch.optim as optim
import time 
#import torch.distributions as dist

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  



def collect_from_learnt_env(state_samples, action_samples, reward_samples, next_state_samples, done_samples, actor, realT_zon_model, dry_bulb_model, reward_model, task_index):
       
      state_samples_duplication = state_samples * 9
      
      actions, _, _ = actor.select_action(state_samples_duplication)  #9 actions to be sampled by the agent per, state
      
      actions = actions.detach().clone()
      
      #------------------------------------------------------------------------
      X = torch.cat( [state_samples, actions] , dim = 1)
      
      weights, bias = realT_zon_model.hypernet.generate_weights_bias(task_index )   #weights and bias generated initially is nearly 20 times larger than other code
     
      realT_zon_model.target_model.update_params(weights, bias)
      
      realT_zon_predictions = realT_zon_model.target_model.predict_target(X)   #even before any gradient update this produced hugre predictions avg (825)
           
      #------------------------------------------------------------------------
      
      X = torch.cat( [state_samples, actions] , dim = 1)
      
      weights, bias = dry_bulb_model.hypernet.generate_weights_bias(task_index )   #weights and bias generated initially is nearly 20 times larger than other code
     
      dry_bulb_model.target_model.update_params(weights, bias)
      
      dry_bulb_predictions = dry_bulb_model.target_model.predict_target(X)   #even before any gradient update this produced hugre predictions avg (825)
      
      #------------------------------------------------------------------------
      
      time_delta = 0 #needs to add time difference between two states
      
      next_state_predictions = state_samples[:, 0] + time_delta , realT_zon_predictions, dry_bulb_predictions
      
      #-----------------------------------------------------------------------
      X = torch.cat( [state_samples, actions] , dim = 1)
      
      weights, bias = reward_model.hypernet.generate_weights_bias(task_index )   #weights and bias generated initially is nearly 20 times larger than other code
     
      reward_model.target_model.update_params(weights, bias)
      
      reward_predictions = reward_model.target_model.predict_target(X)   #even before any gradient update this produced hugre predictions avg (825)
      
      reward_predictions = reward_model.inverse_scale(reward_predictions)
      
      #------------------------------------------------------------------------
      
      #------------------------------------------------------------------------
      done_predictions = 0 
      
      #------------------------------------------------------------------------
      
      synthetic_data = torch.cat( [ state_samples_duplication, actions, next_state_predictions, reward_predictions, done_predictions ], dim = 1 )
      
      actual_data = torch.cat( [ state_samples, action_samples, reward_samples, next_state_samples, done_samples], dim = 1 )
      
      
      #start = time.time()
      
      #env_sample_size, hypernet_sample_size = 10, 1000   #instead of 1 we can have N number of sates used and thus speed up data aquasition
      
      #states =  env_memory.sample_random_states( sample_size = env_sample_size )
      
      current_time = env_memory.time_diff + states[:, 0:1 ]
      
      #actions, discrete_actions, _ = actor.select_action(states)
      
      #actions = actions.detach().clone()
      
      input_data = torch.cat( [  states , actions  ]  , dim = 1)
      
      input_data = (input_data - env_memory.X_min ) / (env_memory.X_max - env_memory.X_min)   # scaling the input
      
      weights, bias = hypernet.generate_weights_bias(task_index , sample_size = hypernet_sample_size )  
      
      predictions = []
      
      for i in range(hypernet_sample_size):
          
          target_model.update_params( [ weights[0][i], weights[1][i], weights[2][i] ] , [ bias[0][i], bias[1][i], bias[2][i] ] )
          
          sample_predictions = target_model.predict_target(input_data)
          
          predictions.append(sample_predictions)
      
      predictions = torch.stack( predictions , dim = 0)  
      
      uncertanities  = torch.mean( torch.std ( predictions , dim = 0) , dim =1).detach().clone()   
      
      final_predictions = torch.mean(predictions , dim = 0 ).detach().clone()   
      
      final_predictions = final_predictions * (env_memory.y_max - env_memory.y_min)  +  env_memory.y_min #inverse scaling the output
      
      next_state_preds = torch.cat( [ current_time , final_predictions[:, 0:1], states[:, 3:], final_predictions[:, 1:2] ], dim =1 ) 
      
      rewards = final_predictions[:, 2: ]
      
      #print("time take: ",time.time() - start)
      
      return states , actions, rewards, next_state_preds , uncertanities
  
     



        