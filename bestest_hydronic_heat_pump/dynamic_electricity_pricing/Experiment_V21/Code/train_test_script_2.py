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


from simulation_environments import bestest_hydronic_heat_pump

from agent_actor_critic import Actor, Critic,  Agent_Memory

from save_results import save_models, save_train_results, save_test_results

import time
import warnings
import csv
import copy

# Filter out the specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message="WARN: env.step_period to get variables from other wrappers is deprecated*")
warnings.filterwarnings("ignore", category=UserWarning, message="WARN: env.get_kpis to get variables from other wrappers is deprecated*")

 

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

train_start_episode = 2
rho = 0.99
alpha = 0.15
no_of_updates = 1

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
        next_action, _, action_log_probs = actor.select_action(next_state_samples)
    
    q_val_target_1 = critic_target_1.get_q_value(next_state_samples, next_action)
    
    q_val_target_2 = critic_target_2.get_q_value(next_state_samples, next_action)
    
    q_val_next = torch.min( q_val_target_1 , q_val_target_2) 
    
    q_val_target = reward_samples +  gamma*( 1 - done_samples )* ( q_val_next  - alpha * action_log_probs   ) 
    
    return q_val_target



def train_critic( state_samples, action_samples , q_val_target ):
    
        
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


def train_actor(state_samples):
    
    
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
    
    

def update_target_critic():
    #we need to define them global since they are being assigned new values instide the function and are defined globally.
    
    global critic_target_1, critic_target_2   
    
    
    for target_param, param in zip(critic_target_1.parameters(), critic_1.parameters()):
        target_param.data.copy_(rho * target_param.data + (1 - rho) * param.data)


    for target_param, param in zip(critic_target_2.parameters(), critic_2.parameters()):
        target_param.data.copy_(rho * target_param.data + (1 - rho) * param.data)

    


actor, actor_optimizer, critic_1 , critic_optimizer_1, critic_2 , critic_optimizer_2, critic_target_1, critic_target_2 = initialize_agent()

memory = Agent_Memory()


with open(metrics_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Type','episode', 'time_steps', 'Length', 'Date', 'actor_loss', 'critic_1_loss','critic_2_loss','cost_tot', 'emis_tot','ener_tot','idis_tot','pdih_tot','pele_tot','pgas_tot','tdis_tot','extrinsic_reward'])
    file.close()
 

    
train_time = 0
   
for i_episode in range(1, n_training_episodes+1): 
        
        done, time_step = 0, 1
        
        state = env.reset()[0]
        
        episode_rewards, episode_actor_loss, episode_critic_1_loss, episode_critic_2_loss = [], [], [], []
        
        while done==0:
            
            print("Episode: ", i_episode, time_step)
            
            start_time = time.time()
            
            action, discrete_action, _ = actor.select_action(state)
                
            next_state, reward, done = collect_from_actual_env( discrete_action )
            
            memory.remember( i_episode, state, action, reward, next_state, done)
            
            plot_scores_train_extrinsic[i_episode] = plot_scores_train_extrinsic.get(i_episode, 0) + reward
            
            state = next_state.copy()
            
            time_step +=1
        
            if i_episode >= train_start_episode : 
                    
                    for update in range(no_of_updates):
                        
                        state_samples, action_samples, reward_samples, next_state_samples, done_samples =  memory.sample_memory()
                        
                        q_val_target = compute_target( reward_samples, next_state_samples, done_samples )
                        
                        l1, l2 = train_critic(state_samples, action_samples , q_val_target)
                        
                        a1 = train_actor(state_samples) 
                        
                        update_target_critic()
                        
                        episode_critic_1_loss.append(l1)
                        
                        episode_critic_2_loss.append(l2)
                        
                        episode_actor_loss.append(a1)
                        
                        episode_rewards.append(reward)
        
        train_time = train_time + ( time.time() - start_time)                
                        
     
        if i_episode >= train_start_episode : 
            save_train_results(i_episode, metrics_path, env , exp_path, train_time, episode_rewards, plot_scores_train_extrinsic, episode_actor_loss, episode_critic_1_loss, episode_critic_2_loss)
        
        if i_episode % 5 == 0:
            save_test_results(i_episode, metrics_path, env, exp_path, actor)
            
        if i_episode % 5 == 0:
               save_models(i_episode, exp_path, actor,actor_optimizer,critic_1, critic_optimizer_1 , critic_2 , critic_optimizer_2)    
        


        
      













