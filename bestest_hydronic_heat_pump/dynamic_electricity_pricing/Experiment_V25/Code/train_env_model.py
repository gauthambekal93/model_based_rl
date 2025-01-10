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
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V25/Code')

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

from env_model import ReaTZon_Model, TDryBul_Model, Reward_Model

from env_memory import Environment_Memory_Train, Environment_Memory_Test
mse_loss = nn.MSELoss()


def initialize_env_model(env_model_attributes):     
     
     realT_zon_model = ReaTZon_Model(env_model_attributes)
    
     dry_bulb_model = TDryBul_Model(env_model_attributes)
     
     reward_model = Reward_Model(env_model_attributes)
     
     env_memory_train = Environment_Memory_Train(env_model_attributes["buffer_size"] ,  env_model_attributes["train_test_ratio"])
     
     env_memory_test = Environment_Memory_Test(env_model_attributes["buffer_size_2"] )
     
     return realT_zon_model, dry_bulb_model, reward_model, env_memory_train , env_memory_test
 


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
    
       
   
def validate_model(test_X, test_y, hypernet, target_model, task_index, sample_size = 1000):
    
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
    
    train_X, train_y, validation_X, validation_y = model.get_dataset(env_memory)
     
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
             test_loss, test_predictions = validate_model(validation_X, validation_y, model.hypernet, model.target_model, task_index )
             
             if test_loss <= 0.0070:
                 
                 checkpoint = { 'model_state_dict': model.hypernet.state_dict(),  'optimizer_state_dict': model.hypernet_optimizer.state_dict() }
        
                 torch.save(checkpoint, exp_path + '/Models/'+'hypernet_'+model.name+"_"+str(epoch)+'_.pkl')
                 
                 break
    
        if epoch % 999 ==0  :
        
             checkpoint = { 'model_state_dict': model.hypernet.state_dict(),  'optimizer_state_dict': model.hypernet_optimizer.state_dict() }
    
             torch.save(checkpoint, exp_path + '/Models/'+'hypernet_'+model.name+"_"+str(epoch)+'_.pkl')
         
            
    
def diagnosis_hypernet(realT_zon_model, dry_bulb_model, reward_model, env_memory, exp_path, env_model_attributes ):
    
    task_index = env_model_attributes["task_index"]
    
    
    _, _, test_X, test_y = realT_zon_model.get_dataset(env_memory)
    
    test_loss, test_predictions  = validate_model(test_X, test_y, realT_zon_model.hypernet, realT_zon_model.target_model, task_index )
    
    
    _, _, test_X, test_y = dry_bulb_model.get_dataset(env_memory)
    
    test_loss, test_predictions  = validate_model(test_X, test_y, dry_bulb_model.hypernet, dry_bulb_model.target_model, task_index )
    
    
    _, _, test_X, test_y = reward_model.get_dataset(env_memory)
    
    test_loss, test_predictions  = validate_model(test_X, test_y, reward_model.hypernet, reward_model.target_model, task_index )
        
        