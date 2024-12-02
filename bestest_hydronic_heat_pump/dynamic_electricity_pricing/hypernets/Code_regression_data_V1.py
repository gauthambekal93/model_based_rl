# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:36:01 2024

@author: gauthambekal93
"""

import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn

import torch.optim as optim

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  



# Parameters
num_tasks = 10
num_samples_per_task = 1000  # Number of samples in each dataset
input_dim = 3  # Number of input dimensions
output_dim = 1  # Number of output dimensions

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

# Save or display datasets
#for i, dataset in enumerate(datasets):
#    dataset.to_csv(f"linear_task_{i+1}.csv", index=False)
#    print(f"Task {i+1} dataset saved as 'linear_task_{i+1}.csv'")




class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
    def forward(self, x):
       logits = self.leaky_relu ( self.fc1(x) )
       logits = self.leaky_relu( self.fc2(logits) )
       outputs =self.leaky_relu( self.fc3(logits) )
       
       return outputs
   
epochs = 10000

mse_loss = nn.MSELoss()
    


def create_model():
    model = MLP(input_dim, hidden_dim = input_dim*2)
     
    model_optimizer = optim.Adam( model.parameters(), lr= 0.001 )
    
    return model, model_optimizer

def create_data(task_index):
    
    X, y = torch.tensor( datasets[task_index].values[:, :-1], dtype=torch.float32), torch.tensor(datasets[task_index].values[:, -1:], dtype=torch.float32)

    split = int(0.80 * len(X))

    train_X, train_y, test_X, test_y = X[ : split], y[: split], X[split: ], y[split:] 
   
    return train_X, train_y, test_X, test_y

   

def train_model(train_X, train_y, model, model_optimizer):   
   
   index  = torch.arange(0, 100)

   for epoch in range(1, epochs):  
      
      index = index[torch.randperm(index.size(0))]
      
      batch_X , batch_y =  train_X[index], train_y[index]
      
      output = model(batch_X )
      
      loss = mse_loss(output, batch_y ) 
      
      model_optimizer.zero_grad()
      
      loss.backward()
      
      model_optimizer.step()
     
   #return model, model_optimizer


def test_model(test_X, test_y, model):
    
    test_output = model(test_X )

    test_loss = mse_loss(test_output, test_y )    
    
    print("Test loss ", test_loss)  
     
    
    
model, model_optimizer = create_model()
    
task_index = 0    

train_X, train_y, test_X, test_y =   create_data(task_index)  

print("Test loss on 0 before training 0: ")

test_model(test_X, test_y, model)

train_model(train_X, train_y, model, model_optimizer)    #this line is acting weird!!!

print("Test loss on 0 after training 0: ")

test_model(test_X, test_y, model)

#-------------------------------------

task_index = 1  

train_X, train_y, test_X, test_y =   create_data(task_index)  

print("Test loss on 1 after training 0 before training 1: ")

test_model(test_X, test_y, model)

train_model(train_X, train_y, model, model_optimizer)    #this line is acting weird!!!

print("Test loss 1 : ")

test_model(test_X, test_y, model)

print("Test loss on 1 after training 0 and 1: ")

test_model(test_X, test_y, model)    

#-------------------------------------

    
model, model_optimizer = create_model()
    
task_index = 0    

train_X, train_y, test_X, test_y =   create_data(task_index)  

print("Test loss on 0 after training 0 and 1: ")

test_model(test_X, test_y, model)

