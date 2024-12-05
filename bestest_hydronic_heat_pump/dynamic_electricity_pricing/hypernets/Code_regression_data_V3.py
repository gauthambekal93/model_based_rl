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

from model_architectures import Target_Model, Hypernet

num_tasks = 10 #10 #was 2

num_samples_per_task = 1000  # Number of samples in each dataset

input_dim , output_dim = 3, 1  # Number of input dimensions, # Number of output dimensions

hidden_dim =  input_dim * 2

epochs = 1000 #100000

num_layers = 3

task_index = 0    


mse_loss = nn.MSELoss()

# Generate datasets
datasets = []
for task in range(num_tasks):
    # Generate random weights and bias for the unique solution
    weights = np.random.uniform(-10, 10, input_dim)
    bias = np.random.uniform(-5, 5, output_dim)

    # Generate input data
    X = np.random.uniform(-1, 1, (num_samples_per_task, input_dim))

    # Generate outputs (y = Xw + b + noise)
    noise = np.random.normal(0, 0.1, (num_samples_per_task, output_dim))
    y = np.dot(X, weights.reshape(-1, 1)) + bias + noise

    # Store the dataset
    dataset = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(input_dim)])
    dataset["y"] = y
    datasets.append(dataset)




def target_model_predictions(X, weights, bias ):
    
    leaky_relu = nn.LeakyReLU(negative_slope=0.0005)  #was 0.001
    
    for W, b in zip(weights, bias):
        
        X = leaky_relu (torch.matmul(X, W) + b )
   
    return X



    

def create_target_model():
    
    target_model = Target_Model(input_dim, hidden_dim, output_dim)
    
    return target_model


def create_hypernet(target_model):
    
    input_dim = num_tasks + num_layers
    
    w1_dim = target_model.weight1.shape[0] * target_model.weight1.shape[1] 
    
    b1_dim = target_model.weight1.shape[1]
    
    w2_dim = target_model.weight2.shape[0] * target_model.weight2.shape[1] 
    b2_dim = target_model.weight2.shape[1] 
    
    w3_dim = target_model.weight3.shape[0] * target_model.weight3.shape[1] 
    b3_dim = target_model.weight3.shape[1] 
    

    hidden_dim = max(w1_dim, w2_dim, w3_dim)
    
    hypernet = Hypernet(  input_dim, hidden_dim, w1_dim, b1_dim, w2_dim, b2_dim, w3_dim, b3_dim ) 
     
    model_optimizer = optim.Adam( hypernet.parameters(), lr= 0.001 )  #was 0.01
     
    return hypernet, model_optimizer


def create_data(task_index):
    
    X, y = torch.tensor( datasets[task_index].values[:, :-1], dtype=torch.float32), torch.tensor(datasets[task_index].values[:, -1:], dtype=torch.float32)

    split = int(0.80 * len(X))

    train_X, train_y, test_X, test_y = X[ : split], y[: split], X[split: ], y[split:] 
   
    min_y, max_y = train_y.min(), train_y.max() 
   
    train_y = (train_y - min_y) / (max_y - min_y)
    
    test_y = (test_y - min_y) / (max_y - min_y)
    
    return train_X, train_y, test_X, test_y

   
def test_model(test_X, test_y, model, task_index):
    
    weights, bias = model.generate_weights_bias(task_index , num_layers)
   
    target_model.update_params(weights, bias)
    
    predictions = target_model.predict_target(test_X)
        
    test_loss = mse_loss(predictions, test_y ) 
    
    print("Test Loss ", test_loss.item())



def calculate_loss(y_pred, y, task_index, hypernet, hypernet_old):
    
    beta, regularizer = 0.01, 0.0   # was 0.01  
    
    for previous_task_index in range(0, task_index): 
        
        weights, bias = hypernet.generate_weights_bias( previous_task_index, num_layers)
        
        weights_old, bias_old = hypernet_old.generate_weights_bias( previous_task_index, num_layers)
        
        for layer_no in range(len( weights )):
                
            regularizer = regularizer  + mse_loss( hypernet.W_mus[layer_no] , hypernet_old.W_mus[layer_no] ) + mse_loss( hypernet.b_mus[layer_no], hypernet_old.b_mus[layer_no] )
    
    kl_divergence = hypernet.calculate_kl_divergence()
        
    loss = mse_loss(y_pred, y )  + kl_divergence + beta * regularizer    
    
    print("mse ", loss.item(),"KL Divergence ", kl_divergence.item(), "Regularizer ", beta * regularizer   )
    
    return loss, hypernet
    
       
    
    
    
target_model =  create_target_model()

hypernet, model_optimizer = create_hypernet(target_model)

hypernet_old = None

batch_size = 20

datasets = [ create_data(task_index) for task_index in range(num_tasks) ]
       
    
for task_index in range(num_tasks):
       
    train_X, train_y, test_X, test_y = datasets[task_index]
    
    if task_index > 0:
        
        hypernet_old, _ = create_hypernet(target_model)
        
        hypernet_old.load_state_dict(torch.load("hypernet_model.pth"))
        
        for param in hypernet_old.parameters():
            param.requires_grad = False
    
    for epoch in range(1, epochs): 
        
        print("Epoch: ",epoch)
              
        index  = torch.randperm(len(train_X) )[:batch_size]
        
        batch_X , batch_y =  train_X[index], train_y[index]
        
        weights, bias = hypernet.generate_weights_bias(task_index , num_layers)
       
        target_model.update_params(weights, bias)
        
        predictions = target_model.predict_target(batch_X)
            
        loss, hypernet = calculate_loss(predictions, batch_y, task_index, hypernet, hypernet_old  )
        
        test_model(test_X, test_y, hypernet, task_index)
        
        model_optimizer.zero_grad()
     
        loss.backward()
        
        model_optimizer.step()        
        
    torch.save( hypernet.state_dict(), "hypernet_model.pth")    
        



    
for task_index, dataset in enumerate(datasets):
    train_X, train_y, test_X, test_y = dataset
    print( task_index , test_model(test_X, test_y, hypernet, task_index))
    
    
    
  
        
        
        
        
        
        
        