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
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V23/Code')

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

from env_model import Target_Model, Hypernet, Environment_Memory

mse_loss = nn.MSELoss()


def initialize_env_model(env_attributes):     
     
     t_input_dim = int( env_attributes['state_space'] ) + int(env_attributes["no_of_action_types"] ) 
     
     t_hidden_dim = int(env_attributes["env_h_size"])
     
     t_output_dim = 3 #was 1   #this is because we will predict room_temp, dry_bulb_temp, reward
     
     num_layers = 3
     
     num_tasks = 2
     
     target_model = Target_Model( t_input_dim, t_hidden_dim, t_output_dim )
     
     h_input_dim = num_tasks + num_layers   #we need to definetask_id and layer_id as input to the hypernet model
     
     w1_dim = target_model.weight1.shape[0] * target_model.weight1.shape[1] 
     
     b1_dim = target_model.weight1.shape[1]
     
     w2_dim = target_model.weight2.shape[0] * target_model.weight2.shape[1] 
     b2_dim = target_model.weight2.shape[1] 
     
     w3_dim = target_model.weight3.shape[0] * target_model.weight3.shape[1] 
     b3_dim = target_model.weight3.shape[1] 
     

     h_hidden_dim = max(w1_dim, w2_dim, w3_dim)
     
     hypernet = Hypernet(  h_input_dim, h_hidden_dim, w1_dim, b1_dim, w2_dim, b2_dim, w3_dim, b3_dim, num_tasks, num_layers ) 
      
     hypernet_optimizer = optim.Adam( hypernet.parameters(), lr = 0.0001 )  #was 0.001
      
     env_memory = Environment_Memory()
     
     return target_model, hypernet, hypernet_optimizer, env_memory 
 


def calculate_train_loss( y_pred, y, task_index, hypernet, hypernet_old):
    
    beta, regularizer = 0.01, 0.0   
    
    for previous_task_index in range(0, task_index): 
        
        weights, bias = hypernet.generate_weights_bias( previous_task_index)
        
        weights_old, bias_old = hypernet_old.generate_weights_bias( previous_task_index)
        
        for layer_no in range(len( weights )):
                
            regularizer = regularizer  + mse_loss( hypernet.W_mus[layer_no] , hypernet_old.W_mus[layer_no] ) + mse_loss( hypernet.b_mus[layer_no], hypernet_old.b_mus[layer_no] )
    
    mse = mse_loss(y_pred, y )
     
    loss = mse  + beta * regularizer    
    
    print("MSE ", mse.item(), "Regularizer ", beta * regularizer,"Train Loss ", loss.item()   )
    print("------------------------------------------------------------------------------------")
          
    return loss, hypernet
    
       
   
def test_model(test_X, test_y, hypernet, target_model, task_index, sample_size = 1000):
    
    weights, bias = hypernet.generate_weights_bias(task_index , sample_size)
    
    final_predictions = []
    
    for i in range(sample_size):
        target_model.update_params( [ weights[0][i], weights[1][i], weights[2][i] ] , [ bias[0][i], bias[1][i], bias[2][i] ] )
        
        sample_predictions = target_model.predict_target(test_X)   #shape: 42 x 3
        
        final_predictions.append(sample_predictions)
    

    #final_predictions = torch.cat( final_predictions , dim = 1)
    
    #test_loss = mse_loss( torch.mean(final_predictions, dim = 1).reshape(-1,1) , test_y ) 
    
    #uncertanity  = torch.mean( torch.std ( final_predictions , dim = 1) )
    
    final_predictions = torch.stack( final_predictions , dim = 0)
    
    test_loss = mse_loss( torch.mean(final_predictions, dim = 0), test_y ) 
    
    uncertanity  = torch.mean( torch.std ( final_predictions , dim = 0) )
    
    print("Hypernet Test Loss: ", test_loss.item(), "Uncertanity: ", uncertanity.item() )
    print("------------------------------------------------------------------------------------")
  
    
  
def data_processing(env_memory):
    
    train_dataset = torch.cat( [ torch.cat(list(env_memory.train_X), dim=0), torch.cat(list(env_memory.train_y), dim=0) ], dim =1 )
    
    test_dataset = torch.cat( [ torch.cat(list(env_memory.test_X), dim=0), torch.cat(list(env_memory.test_y), dim=0) ], dim =1 )
    
    col_min = train_dataset. min(dim=0, keepdim=True)[0]  
    
    col_max = train_dataset. max(dim=0, keepdim=True)[0]  
   
    train_dataset = (train_dataset - col_min) / (col_max - col_min)
    
    test_dataset = (test_dataset - col_min) / (col_max - col_min)
     
    #return train_dataset[:, : -3], train_dataset[:, -1: ], test_dataset[:, : -3], test_dataset[:, -1: ]  #was -3 now -1
    return train_dataset[:, : -3], train_dataset[:, -3: ], test_dataset[:, : -3], test_dataset[:, -3: ]  #was -3 now -1

'''
def data_processing2():
    
    weights = np.random.uniform(-10, 10, 13)
    
    bias = np.random.uniform(-5, 5, 1)

    # Generate input data
    X = np.random.uniform(-1, 1, (1000, 13 ))
    
    noise = np.random.normal(0, 0.1, (1000, 1))
    
    y = np.dot(X, weights.reshape(-1, 1)) + bias + noise
    
    X, y = torch.tensor( X , dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    split = int(0.80 * len(X))

    train_X, train_y, test_X, test_y = X[ : split], y[: split], X[split: ], y[split:] 
   
    min_y, max_y = train_y.min(), train_y.max() 
   
    train_y = (train_y - min_y) / (max_y - min_y)
    
    test_y = (test_y - min_y) / (max_y - min_y)
    
    return train_X, train_y, test_X, test_y
'''
    
def calculate_max_grad(hypernet):
    tmp = []    
    for name, param in hypernet.named_parameters():
        if param.grad is not None:
            tmp.append(param.grad.norm().item())
           # print(f"Gradient norm for {name}: {param.grad.norm().item()}")
    if tmp:
       print("Max gradient ", np.mean(tmp) )




def train_hypernet(target_model, hypernet, hypernet_optimizer, env_memory, task_index, epochs, hypernet_old = None):
    
    #hypernet.load_state_dict(torch.load("hypernet_model.pth"))
    
    if hypernet_old:
        
        hypernet_old.load_state_dict(torch.load("hypernet_model.pth"))
        
        for param in hypernet_old.parameters():
            param.requires_grad = False
    


    batch_size = 20 #was 32
    
    train_X, train_y, test_X, test_y = env_memory.get_dataset()  #data_processing(env_memory) #data_processing2() 
    
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        
        indices  = torch.randperm(len(train_X) )
        
        
        for batch_no in range( 0, int(len(train_X) ), batch_size ):
            
            index = indices [batch_no : batch_no + batch_size]
            
            batch_X , batch_y = train_X[index], train_y[index]
            
            weights, bias = hypernet.generate_weights_bias(task_index )   #weights and bias generated initially is nearly 20 times larger than other code
           
            target_model.update_params(weights, bias)
            
            predictions = target_model.predict_target(batch_X)   #even before any gradient update this produced hugre predictions avg (825)
                
            loss, hypernet = calculate_train_loss(predictions, batch_y, task_index, hypernet, hypernet_old )
            
            #calculate_max_grad(hypernet)
           
            
            hypernet_optimizer.zero_grad()
            
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_norm=1.0)
            
            loss.backward()   
            
            hypernet_optimizer.step()        
         
        if epoch %10 ==0:
             test_model(test_X, test_y, hypernet, target_model, task_index )
             
    torch.save( hypernet.state_dict(), "hypernet_model.pth")    
    

def collect_from_learnt_env(env_memory, actor, hypernet, target_model, task_index):
      
      env_sample_size, hypernet_sample_size = 1, 1000   #instead of 1 we can have N number of sates used and thus speed up data aquasition
      
      random_state =  env_memory.sample_random_states( sample_size = env_sample_size )
      
      action, discrete_action, _ = actor.select_action(random_state)
      
      input_data = torch.cat( [ torch.tensor( random_state ).reshape(1,-1) , action.detach().clone()  ]  , dim = 1)
      
      weights, bias = hypernet.generate_weights_bias(task_index , sample_size = hypernet_sample_size )  
      
      predictions = []
      
      for i in range(hypernet_sample_size):
          
          target_model.update_params( [ weights[0][i], weights[1][i], weights[2][i] ] , [ bias[0][i], bias[1][i], bias[2][i] ] )
          
          sample_predictions = target_model.predict_target(input_data)
          
          predictions.append(sample_predictions)
      
      predictions = torch.stack( predictions , dim = 0)  
      
      uncertanity  = torch.mean( torch.std ( predictions , dim = 0) )   
      
      final_predictions = torch.mean(predictions , dim =1 ).reshape(-1,1)
      
      next_state_pred = torch.cat( [ random_state[0], final_predictions[0], random_state[2:], final_predictions[1]], dim =1 )
      
      reward = final_predictions[3]
      
      return random_state , action, reward, next_state_pred , uncertanity

     

#for param_group in hypernet_optimizer.param_groups:
#    print(param_group["params"])
    #print(f"Learning rate updated to {param_group['lr']}")

#for param_group in hypernet_optimizer.param_groups:
#    param_group['lr'] = 0.00002
        
        

        
        
        