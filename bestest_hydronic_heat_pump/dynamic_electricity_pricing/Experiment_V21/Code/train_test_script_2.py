# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:38:10 2024

@author: gauthambekal93
"""

import os
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V21/Code')

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


from simulation_environments import bestest_hydronic_heat_pump, max_episode_length, start_time_tests,episode_length_test, warmup_period_test

from updated_plot import test_agent, plot_results
#from memory_module import Memory
from agent_actor_critic import Actor, Critic,  Agent_Memory

import datetime
import time


import matplotlib.pyplot as plt

import warnings
import csv
import copy

# Filter out the specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message="WARN: env.step_period to get variables from other wrappers is deprecated*")
warnings.filterwarnings("ignore", category=UserWarning, message="WARN: env.get_kpis to get variables from other wrappers is deprecated*")

 

year = 2024

env, env_attributes  = bestest_hydronic_heat_pump()

n_training_episodes = int( env_attributes['n_training_episodes'] )

max_t = int(env_attributes['max_t'])

gamma = float(env_attributes['gamma'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device ", device)


with open('all_paths.json', 'r') as openfile:  json_data = json.load(openfile)

exp_path = json_data['experiment_path']
metrics_path = json_data['metrics_path']
rl_data_path = json_data['rl_data_path']


plot_scores_train_extrinsic = {}

plot_scores_test_extrinsic_jan17 = {}
plot_scores_test_extrinsic_apr19 = {}
plot_scores_test_extrinsic_nov15 = {}
plot_scores_test_extrinsic_dec08 = {}


train_start_episode = 1
rho = 0.99
alpha = 0.15
no_of_updates = 2

mse_loss = nn.MSELoss()

def initialize_agent():     
     
     
     actor = Actor(int(env_attributes["state_space"]),  int(env_attributes["h_size"]),  int(env_attributes["action_bins"]), device).to(device) #was  int(env_attributes["action_space"])
     
     actor_optimizer = optim.Adam( actor.parameters(), lr= float(env_attributes["actor_lr"]) )
     
     
     critic_1 = Critic(int(env_attributes["state_space"]),  int(env_attributes["h_size"]), device).to(device) #was  
     
     critic_optimizer_1 = optim.Adam( critic_1.parameters(), lr =  float(env_attributes["critic_lr"]) )
     
     
     critic_2 = Critic(int(env_attributes["state_space"]), int(env_attributes["h_size"]), device).to(device) #was  int(env_attributes["action_space"])
     
     critic_optimizer_2 = optim.Adam( critic_2.parameters(), lr =  float(env_attributes["critic_lr"]) )
     
     #the entire architecture and weights of model is copied and anychanges to critic_1 weights would not affect critic_target_1
     critic_target_1 = copy.deepcopy(critic_1)
     
     #this is to ensure that we are not by chance updating the weights of target network
     for param in critic_target_1.parameters():
         param.requires_grad = False
         
     critic_target_2 = copy.deepcopy(critic_2)
     
     for param in critic_target_2.parameters():
        param.requires_grad = False
     
     return actor, actor_optimizer, critic_1 , critic_optimizer_1, critic_2 , critic_optimizer_2, critic_target_1, critic_target_2

    
    
    
def collect_from_actual_env( action):
    
    next_state, reward, done, _, res  = env.step(action )
    
    if done:
        done = 1
    else:
        done = 0
        
    return next_state, reward, done
    


def compute_target( reward_samples, next_state_samples, done_samples ):
    
    with torch.no_grad():
        next_continuous_action, _, action_log_probs = actor.select_action(next_state_samples)
    
    q_val_target_1 = critic_target_1.get_q_value(next_state_samples, next_continuous_action)
    
    q_val_target_2 = critic_target_2.get_q_value(next_state_samples, next_continuous_action)
    
    q_val_next = torch.min( q_val_target_1 , q_val_target_2) 
    
    q_val_target = reward_samples +  gamma*( 1 - done_samples )* ( q_val_next  - alpha * action_log_probs   ) #need to replace with log probs !!!
    
    return q_val_target



def train_critic( state_samples, action_samples , q_val_target ):
    
    critic_loss_1 = mse_loss (  critic_1.get_q_value(state_samples, action_samples),  q_val_target )
    
    critic_loss_2 = mse_loss (  critic_2.get_q_value(state_samples, action_samples),  q_val_target )
    
    
    critic_optimizer_1.zero_grad()
    
    critic_optimizer_2.zero_grad()
    
    
    critic_loss_1.backward()
    
    critic_loss_2.backward()
    
    
    critic_optimizer_1.step()
    
    critic_optimizer_2.step()
    
    


def train_actor(state_samples):
    
    action, _, action_log_probs = actor.select_action(state_samples)
    
    q_val_1 = critic_1.get_q_value( state_samples, action )  
    
    q_val_2 = critic_2.get_q_value( state_samples, action )  
    
    q_val = torch.min( q_val_1, q_val_2  )
    
    actor_loss = - 1.0* ( q_val - alpha* (action_log_probs ) )  #we need maximize actor_loss 
    
    
    actor_optimizer.zero_grad()
    
    actor_loss.backward()
    
    actor_optimizer.step()
    
    
    

def update_target_critic():
    #we need to define them global since they are being assigned new values instide the function and are defined globally.
    
    global critic_target_1, critic_target_2   
    
    critic_target_1 = rho * critic_target_1 + (1 - rho) * critic_1
    
    critic_target_2 = rho * critic_target_2 + (1 - rho) * critic_2
    
    for target_param, param in zip(critic_target_1.parameters(), critic_1.parameters()):
        target_param.data.copy_(rho * target_param.data + (1 - rho) * param.data)


    for target_param, param in zip(critic_target_2.parameters(), critic_2.parameters()):
        target_param.data.copy_(rho * target_param.data + (1 - rho) * param.data)

    


actor, actor_optimizer, critic_1 , critic_optimizer_1, critic_2 , critic_optimizer_2, critic_target_1, critic_target_2 = initialize_agent()

memory = Agent_Memory()


with open(metrics_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Type','episode', 'time_steps', 'Length', 'Date', 'actor_loss', 'critic_loss','baseline_loss','cost_tot', 'emis_tot','ener_tot','idis_tot','pdih_tot','pele_tot','pgas_tot','tdis_tot','extrinsic_reward'])
    file.close()
 

    
train_time, test_jan17_time, test_apr19_time, test_nov15_time, test_dec08_time = 0, 0, 0, 0, 0


   
for i_episode in range(1, n_training_episodes+1): 
        
        done, time_step = 0, 1
        
        state = env.reset()[0]
        
        while done==0:
            
            print("Episode: ", i_episode, time_step)
            
            episode_rewards, episode_actor_loss, episode_critic_loss = [], [], []
            
            continuous_action, discrete_action, _ = actor.select_action(state)
                
            next_state, reward, done = collect_from_actual_env( discrete_action )
            
            memory.remember( i_episode, state, continuous_action, reward, next_state, done)
            
            state = next_state.copy()
            
            time_step += 1
            
        
        if i_episode > train_start_episode : 
                
                for update in range(1, no_of_updates):
                    
                    state_samples, action_samples, reward_samples, next_state_samples, done_samples =  memory.sample_memory()
                    
                    q_val_target = compute_target( reward_samples, next_state_samples, done_samples )
                    
                    train_critic(state_samples, action_samples , q_val_target)
                    
                    train_actor(state_samples)
                    
                    update_target_critic()
                    
                
                
               

























