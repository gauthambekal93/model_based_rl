# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:38:10 2024

@author: gauthambekal93
"""

import os
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V25/Code')

import numpy as np
import torch
import random

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  

 
import torch.optim as optim
import torch.nn as nn

import json


from simulation_environments import bestest_hydronic

from agent_model import Actor, Critic, Agent_Memory

#from save_results import save_models, save_train_results, save_test_results

#import time
import warnings
#import csv
import copy
import re
import pickle

# Filter out the specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message="WARN: env.step_period to get variables from other wrappers is deprecated*")
warnings.filterwarnings("ignore", category=UserWarning, message="WARN: env.get_kpis to get variables from other wrappers is deprecated*")


#env   = bestest_hydronic()

#n_training_episodes = int( env_attributes['n_training_episodes'] )

#max_t = int(env_attributes['max_t'])

#gamma = float(env_attributes['gamma'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#print("Device ", device)


with open('all_paths.json', 'r') as openfile:  json_data = json.load(openfile)

exp_path = json_data['experiment_path']
metrics_path = json_data['metrics_path']
rl_data_path = json_data['rl_data_path']


plot_scores_train_extrinsic = {}

train_start_episode = 2
rho = 0.99
alpha = 0.15
no_of_updates = 1

mse_loss = nn.MSELoss()


def initialize_agent(env_attributes):     
     
     
     actor = Actor(int(env_attributes["state_space"]),  int(env_attributes["agent_h_size"]),  int(env_attributes["action_bins"]), device, int(env_attributes["no_of_action_types"] ) ).to(device) #was  int(env_attributes["action_space"])
    
     actor_optimizer = optim.Adam( actor.parameters(), lr= float(env_attributes["actor_lr"]) )
     
     
     critic_1 = Critic(int(env_attributes["state_space"]),  int(env_attributes["agent_h_size"]), device, int(env_attributes["no_of_action_types"] )).to(device) #was  
     
     critic_optimizer_1 = optim.Adam( critic_1.parameters(), lr =  float(env_attributes["critic_lr"]) )
     
     
     critic_2 = Critic(int(env_attributes["state_space"]), int(env_attributes["agent_h_size"]), device, int(env_attributes["no_of_action_types"] )).to(device) #was  int(env_attributes["action_space"])
     
     critic_optimizer_2 = optim.Adam( critic_2.parameters(), lr =  float(env_attributes["critic_lr"]) )
     
     #the entire architecture and weights of model is copied and anychanges to critic_1 weights would not affect critic_target_1
     critic_target_1 = copy.deepcopy(critic_1)
     
     #this is to ensure that we are not by chance updating the weights of target network
     for param in critic_target_1.parameters():
         param.requires_grad = False
         
     critic_target_2 = copy.deepcopy(critic_2)
     
     for param in critic_target_2.parameters():
        param.requires_grad = False
     
     agent_memory = Agent_Memory(env_attributes["buffer_size"])
     
     return actor, actor_optimizer, critic_1 , critic_optimizer_1, critic_2 , critic_optimizer_2, critic_target_1, critic_target_2, agent_memory

    

def load_models(actor, actor_optimizer, critic_1, critic_optimizer_1, critic_2, critic_optimizer_2, env_attributes):     
     
     model_files = [f for f in os.listdir(exp_path + '/Models/' ) if f.startswith('actor_model_') and f.endswith('.pkl')]
     
     last_loaded_epoch = max( [int(re.search(r'actor_model_(\d+)', f).group(1)) for f in model_files] )
     
     checkpoint = torch.load(exp_path + '/Models/'+'actor_model_'+str(last_loaded_epoch )+'_.pkl')
    
     actor.load_state_dict(checkpoint['model_state_dict'])
     
     actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


     checkpoint = torch.load(exp_path + '/Models/'+'critic_model_1_'+str(last_loaded_epoch)+'_.pkl')
     
     critic_1.load_state_dict(checkpoint['model_state_dict'])
     
     critic_optimizer_1.load_state_dict(checkpoint['optimizer_state_dict'])
     
     
     checkpoint = torch.load(exp_path + '/Models/'+'critic_model_2_'+str(last_loaded_epoch)+'_.pkl')
     
     critic_2.load_state_dict(checkpoint['model_state_dict'])
     
     critic_optimizer_2.load_state_dict(checkpoint['optimizer_state_dict'])
     
     
     #the entire architecture and weights of model is copied and anychanges to critic_1 weights would not affect critic_target_1
     critic_target_1 = copy.deepcopy(critic_1)
     
     #this is to ensure that we are not by chance updating the weights of target network
     for param in critic_target_1.parameters():
         param.requires_grad = False
         
     critic_target_2 = copy.deepcopy(critic_2)
     
     for param in critic_target_2.parameters():
        param.requires_grad = False
     
     with open(exp_path + '/Models/'+"agent_buffer_data.pkl", "rb") as f:
        agent_memory = pickle.load(f)
     
     return actor, actor_optimizer, critic_1 , critic_optimizer_1, critic_2 , critic_optimizer_2, critic_target_1, critic_target_2, agent_memory, last_loaded_epoch  
    
    
def update_hyperparameters(actor_optimizer, critic_optimizer_1, critic_optimizer_2, env_attributes):
    
    for param_group in actor_optimizer.param_groups:
        param_group['lr'] = env_attributes["actor_lr"]
        
    for param_group in critic_optimizer_1.param_groups:
        param_group['lr'] = env_attributes["critic_lr"]

    for param_group in critic_optimizer_2.param_groups:
        param_group['lr'] = env_attributes["critic_lr"]
        

    
    
def collect_from_actual_env( env, action):
    
    next_state, reward, done, _, res  = env.step(action )
    
    if done:
        done = 1
    else:
        done = 0
        
    return next_state, reward, done
    


def compute_target( actor, critic_target_1, critic_target_2,  reward_samples, next_state_samples, done_samples, gamma ):
    
    with torch.no_grad():
        next_action, _, action_log_probs = actor.select_action(next_state_samples)
    
    q_val_target_1 = critic_target_1.get_q_value(next_state_samples, next_action)
    
    q_val_target_2 = critic_target_2.get_q_value(next_state_samples, next_action)
    
    q_val_next = torch.min( q_val_target_1 , q_val_target_2) 
    
    q_val_target = reward_samples +  gamma*( 1 - done_samples )* ( q_val_next  - alpha * action_log_probs   ) 
    
    return q_val_target



def train_critic( critic_1, critic_2, critic_optimizer_1, critic_optimizer_2, state_samples, action_samples , q_val_target ):
    
        
    critic_loss_1 = mse_loss (  critic_1.get_q_value(state_samples, action_samples),  q_val_target )
    
    critic_loss_2 = mse_loss (  critic_2.get_q_value(state_samples, action_samples),  q_val_target )
    
    
    critic_optimizer_1.zero_grad()
    
    critic_optimizer_2.zero_grad()
    
    
    critic_loss_1.backward()
    
    critic_loss_2.backward()
    
    # Optional: Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(critic_1.parameters(), max_norm=1.0)
    
    torch.nn.utils.clip_grad_norm_(critic_2.parameters(), max_norm=1.0)
   
    
    critic_optimizer_1.step()
    
    critic_optimizer_2.step()
    
    return critic_loss_1.item(), critic_loss_2.item()


def train_actor( actor, critic_1, critic_2, actor_optimizer, state_samples):
    
    
    action, _, action_log_probs = actor.select_action(state_samples)
    
    # Freeze critic parameters
    for param in critic_1.parameters():
        param.requires_grad = False
        
    for param in critic_2.parameters():
        param.requires_grad = False
    
    
    q_val_1 = critic_1.get_q_value( state_samples, action )  
    
    q_val_2 = critic_2.get_q_value( state_samples, action )  
    
    q_val = torch.min( q_val_1, q_val_2  )
    
    actor_loss = - torch.mean( ( q_val - alpha* (action_log_probs ) ) ) #we need maximize actor_loss 
    
    
    actor_optimizer.zero_grad()
    
    actor_loss.backward()
    
    actor_optimizer.step()
    
    
    # Unfreeze critic parameters
    for param in critic_1.parameters():
       param.requires_grad = True
    
    for param in critic_2.parameters():
       param.requires_grad = True
     
    return actor_loss.item()    
    
    

def update_target_critic(critic_target_1, critic_target_2, critic_1, critic_2):
    
    for target_param, param in zip(critic_target_1.parameters(), critic_1.parameters()):
        target_param.data.copy_(rho * target_param.data + (1 - rho) * param.data)


    for target_param, param in zip(critic_target_2.parameters(), critic_2.parameters()):
        target_param.data.copy_(rho * target_param.data + (1 - rho) * param.data)

 
      














