# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:22:30 2024

@author: gauthambekal93
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 18:31:44 2024

@author: gauthambekal93
"""

import os
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V24/Code')

import numpy as np
#import pandas as pd
import random

import torch
import torch.nn as nn

import torch.optim as optim
#import torch.distributions as dist

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  

import time 

from env_model import State_Model, Reward_Model, Environment_Memory

mse_loss = nn.MSELoss()


def initialize_env_model(env_model_attributes):     
     
     state_model = State_Model(env_model_attributes)
    
     reward_model = Reward_Model(env_model_attributes)
     
     env_memory = Environment_Memory(env_model_attributes["buffer_size"] ,  env_model_attributes["train_test_ratio"])
     
     env_memory_2 = Environment_Memory(env_model_attributes["buffer_size_2"] ,  env_model_attributes["train_test_ratio_2"])
     
     return state_model, reward_model, env_memory , env_memory_2
 


def calculate_train_loss( epoch, y_pred, y, task_index, hypernet, hypernet_old):
    
    beta, regularizer = 0.01, 0.0   
    
    for previous_task_index in range(0, task_index): 
        
        weights, bias = hypernet.generate_weights_bias( previous_task_index)
        
        weights_old, bias_old = hypernet_old.generate_weights_bias( previous_task_index)
        
        for layer_no in range(len( weights )):
                
            regularizer = regularizer  + mse_loss( hypernet.W_mus[layer_no] , hypernet_old.W_mus[layer_no] ) + mse_loss( hypernet.b_mus[layer_no], hypernet_old.b_mus[layer_no] )
    
    mse = mse_loss(y_pred, y ) 
     
    #mse = torch.mean( ( y_pred - y )**2 / (torch.abs(y) + 1e-4) )
    
    loss = mse  + beta * regularizer    
    
    if epoch %10 == 0:
        print("Epoch ",epoch, "MSE ", mse.item(), "Regularizer ", beta * regularizer,"Train Loss ", loss.item()   )
        print("------------------------------------------------------------------------------------")
          
    return loss, hypernet
    
       
   
def test_model(test_X, test_y, hypernet, target_model, task_index, sample_size = 1000):
    
    weights, bias = hypernet.generate_weights_bias(task_index , sample_size)
    
    final_predictions = []
    
    for i in range(sample_size):
        target_model.update_params( [ weights[0][i], weights[1][i], weights[2][i] ] , [ bias[0][i], bias[1][i], bias[2][i] ] )
        
        sample_predictions = target_model.predict_target(test_X)   #shape: 42 x 3
        
        final_predictions.append(sample_predictions)
    

    
    final_predictions = torch.stack( final_predictions , dim = 0)
    
    test_loss = mse_loss( torch.mean(final_predictions, dim = 0), test_y ) 
    
    uncertanity  = torch.mean( torch.std ( final_predictions , dim = 0) )
    
    print("Hypernet Test Loss: ", test_loss.item(), "Uncertanity: ", uncertanity.item() )
    
    print("------------------------------------------------------------------------------------")
    
    return  test_loss.item(), torch.mean(final_predictions, dim = 0)
    
  
def data_processing(env_memory):
    
    train_dataset = torch.cat( [ torch.cat(list(env_memory.train_X), dim=0), torch.cat(list(env_memory.train_y), dim=0) ], dim =1 )
    
    test_dataset = torch.cat( [ torch.cat(list(env_memory.test_X), dim=0), torch.cat(list(env_memory.test_y), dim=0) ], dim =1 )
    
    col_min = train_dataset. min(dim=0, keepdim=True)[0]  
    
    col_max = train_dataset. max(dim=0, keepdim=True)[0]  
   
    train_dataset = (train_dataset - col_min) / (col_max - col_min)
    
    test_dataset = (test_dataset - col_min) / (col_max - col_min)
     
    return train_dataset[:, : -3], train_dataset[:, -3: ], test_dataset[:, : -3], test_dataset[:, -3: ]  #was -3 now -1



def calculate_max_grad(hypernet):
    tmp = []    
    for name, param in hypernet.named_parameters():
        if param.grad is not None:
            tmp.append(param.grad.norm().item())
           # print(f"Gradient norm for {name}: {param.grad.norm().item()}")
    if tmp:
       print("Max gradient ", np.mean(tmp) )





def train_hypernet(model, env_memory, exp_path,  env_model_attributes ):
    
    task_index, epochs, batch_size = env_model_attributes["task_index"], env_model_attributes["epochs"] , env_model_attributes["batch_size"]
    
    hypernet_old =  env_model_attributes["hypernet_old"]
    
    train_X, train_y, test_X, test_y = model.get_dataset(env_memory)
     
    for epoch in range(epochs):
                
        indices  = torch.randperm(len(train_X) )
        
        
        for batch_no in range( 0, int(len(train_X) ), batch_size ):
            
            index = indices [batch_no : batch_no + batch_size]
            
            batch_X , batch_y = train_X[index], train_y[index]
            
            weights, bias = model.hypernet.generate_weights_bias(task_index )   #weights and bias generated initially is nearly 20 times larger than other code
           
            model.target_model.update_params(weights, bias)
            
            predictions = model.target_model.predict_target(batch_X)   #even before any gradient update this produced hugre predictions avg (825)
                
            loss, hypernet = calculate_train_loss(epoch, predictions, batch_y, task_index, model.hypernet, model.hypernet_old )
            
            model.hypernet_optimizer.zero_grad()
            
            torch.nn.utils.clip_grad_norm_(model.hypernet.parameters(), max_norm=1.0)
            
            loss.backward()   
            
            model.hypernet_optimizer.step()        
         
        if epoch %50 ==0:
             test_loss, test_predictions = test_model(test_X, test_y, model.hypernet, model.target_model, task_index )
             
             if test_loss <= 0.0070:
                 
                 checkpoint = { 'model_state_dict': model.hypernet.state_dict(),  'optimizer_state_dict': model.hypernet_optimizer.state_dict() }
        
                 torch.save(checkpoint, exp_path + '/Models/'+'hypernet_'+model.name+"_"+str(epoch)+'_.pkl')
                 
                 break
    
        if epoch % 999 ==0  :
        
             checkpoint = { 'model_state_dict': model.hypernet.state_dict(),  'optimizer_state_dict': model.hypernet_optimizer.state_dict() }
    
             torch.save(checkpoint, exp_path + '/Models/'+'hypernet_'+model.name+"_"+str(epoch)+'_.pkl')
         
            
    

def collect_from_learnt_env(env_memory, actor, hypernet, target_model, task_index):
      
      #start = time.time()
      
      env_sample_size, hypernet_sample_size = 10, 1000   #instead of 1 we can have N number of sates used and thus speed up data aquasition
      
      states =  env_memory.sample_random_states( sample_size = env_sample_size )
      
      current_time = env_memory.time_diff + states[:, 0:1 ]
      
      actions, discrete_actions, _ = actor.select_action(states)
      
      actions = actions.detach().clone()
      
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
  
     

def diagnosis_hypernet(state_model, reward_model, env_memory, exp_path, env_model_attributes ):
    
    task_index = env_model_attributes["task_index"]
    
    _, _, test_X, test_y = state_model.get_dataset(env_memory)
    
    test_loss, test_predictions  = test_model(test_X, test_y, state_model.hypernet, state_model.target_model, task_index )
    
    _, _, test_X, test_y = reward_model.get_dataset(env_memory)
    
    test_loss, test_predictions  = test_model(test_X, test_y, reward_model.hypernet, reward_model.target_model, task_index )

        
        
        