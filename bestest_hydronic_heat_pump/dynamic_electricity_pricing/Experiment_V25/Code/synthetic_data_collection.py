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


def create_synthetic_data( actor, realT_zon_model, dry_bulb_model, reward_model, agent_actual_memory, agent_synthetic_memory, env_attributes, env_model_attributes, env ):
      
      if agent_synthetic_memory.memory_size() == 0:
          
          state_samples, _, _, _, _, _ =  agent_actual_memory.sample_memory(  sample_size = agent_actual_memory.memory_size() )
      
      else:
          
          state_samples, _, _, _, _, _ =  agent_actual_memory.sample_memory(  last_element = True )
          
      
      state_samples = state_samples.repeat( env_attributes["multiplicative_factor"], 1 )
      
      #should check the below line for sanity purpose
      actions, discrete_actions, _ = actor.select_action( state_samples[ :,   env_attributes["state_mask"] ]  )  #9 actions to be sampled by the agent per, state
      
      actions = actions.detach().clone()
      
      temp =  np.where(discrete_actions == 0, 0 , 1) 
      
      actions = torch.cat( [ actions, torch.tensor(temp) , torch.tensor(temp) ], axis = 1  )
      
      #input_column_index = [ i for i, obs in enumerate( env.observations) if obs in realT_zon_model.input_columns ]
      
      
      """ Predict the Zone operative temperature"""
      test_X =  torch.cat([ state_samples[:, realT_zon_model.input_state_index ], actions ] , dim = 1 )
       
      weights, bias = realT_zon_model.hypernet.generate_weights_bias(env_model_attributes["task_index"] )   #weights and bias generated initially is nearly 20 times larger than other code
     
      realT_zon_model.target_model.update_params(weights, bias)
      
      predicted_realT_zon = realT_zon_model.target_model.predict_target(test_X)   
      
      
      
      """ Predict the Dry bulb temperature"""
      #test_X = dry_bulb_model.get_dataset(env_memory_test)
      
      #input_column_index = [ i for i, obs in enumerate( env.observations) if obs in dry_bulb_model.input_columns ]
      
      test_X =   state_samples[:, dry_bulb_model.input_state_index ]
      
      weights, bias = dry_bulb_model.hypernet.generate_weights_bias(env_model_attributes["task_index"] )   #weights and bias generated initially is nearly 20 times larger than other code
     
      dry_bulb_model.target_model.update_params(weights, bias)
      
      predicted_dry_bulb = dry_bulb_model.target_model.predict_target(test_X)  

          
      #we can use the env.observations to obtain the ways columns are arranged and create next state
        
      #test_X = reward_model.get_dataset(env_memory_test)
      
      
     
      
      time_diff = agent_actual_memory.states[1][0,0] - agent_actual_memory.states[0][0,0] 
      
      time_data = state_samples[:, [0]] + time_diff
      
      next_state_predictions = torch.empty(( len(state_samples),  0 ))
      
      dry_bulb_indices = []
      
      for observation in env.observations:
          
          if "time" in observation:
              next_state_predictions  = torch.cat ( [ next_state_predictions, time_data ] , dim = 1)

              
          if "reaTZon" in observation:
              next_state_predictions = torch.cat ( [ next_state_predictions, predicted_realT_zon ] , dim = 1)


          if ("TDryBul" in observation ) and len( dry_bulb_indices ) == 0:
              
              dry_bulb_indices = [i for i , obs in enumerate( env.observations) if "TDryBul" in obs]
              
              dry_bulb = state_samples[:, dry_bulb_indices]
              
              dry_bulb = torch.cat([dry_bulb, predicted_dry_bulb] , dim =1)
              
              dry_bulb =  dry_bulb[: , 1:]
              
              next_state_predictions = torch.cat ( [ next_state_predictions, dry_bulb ] , dim = 1)
              
              
          if "reaTSetCoo" in observation:
              tmp = torch.empty((next_state_predictions.shape[0], 1))
              
              next_state_predictions = torch.cat ( [ next_state_predictions, tmp ] , dim = 1)
          
            
          if "reaTSetHea_y" in observation:
              tmp = torch.empty((next_state_predictions.shape[0], 1))
              
              next_state_predictions = torch.cat ( [ next_state_predictions, tmp ] , dim = 1)
              

      """ Predict the Reward """
     #input_column_index = [ i for i, obs in enumerate( env.observations) if obs in reward_model.input_columns ]
       
      test_X = torch.cat([ state_samples[:, reward_model.input_state_index], actions ] , dim = 1 )
       
      weights, bias = reward_model.hypernet.generate_weights_bias(env_model_attributes["task_index"] )   #weights and bias generated initially is nearly 20 times larger than other code
      
      reward_model.target_model.update_params(weights, bias)
       
      predicted_rewards = reward_model.target_model.predict_target(test_X)  
       
      predicted_rewards = reward_model.inverse_scale(predicted_rewards)

 
      start = time.time()
      for state_sample, action, next_state, reward in zip( state_samples, actions, next_state_predictions, predicted_rewards) :
          agent_synthetic_memory.remember (state_sample, action, next_state, reward )
      
      print(time.time() - start)
      
  
      
  
     



        