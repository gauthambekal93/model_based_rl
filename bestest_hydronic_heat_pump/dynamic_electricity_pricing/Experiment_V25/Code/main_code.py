# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:38:10 2024

@author: gauthambekal93
"""

import os
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V25/Code')

import numpy as np
import torch
import pandas as pd
import random

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  

 
#import torch.optim as optim
#import torch.nn as nn
import json
import time
import warnings
import csv

# Filter out the specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message="WARN: env.step_period to get variables from other wrappers is deprecated*")
warnings.filterwarnings("ignore", category=UserWarning, message="WARN: env.get_kpis to get variables from other wrappers is deprecated*")

from simulation_environments import bestest_hydronic_heat_pump

from save_results import save_models, save_train_results, save_test_results

from train_agent import initialize_agent, load_models , update_hyperparameters, collect_from_actual_env, compute_target, train_critic, train_actor, update_target_critic

from train_env_model import initialize_env_model, train_hypernet,  diagnosis_hypernet

from synthetic_data_collection import collect_from_learnt_env

env, env_attributes, env_model_attributes  = bestest_hydronic_heat_pump()

n_training_episodes = int( env_attributes['n_training_episodes'] )

max_t = int(env_attributes['max_t'])

gamma = float(env_attributes['gamma'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device ", device)


with open('all_paths.json', 'r') as openfile:  json_data = json.load(openfile)

exp_path = json_data['experiment_path']
metrics_path = json_data['metrics_path']
rl_data_path = json_data['rl_data_path']




train_agent_start_episode = 2
rho = 0.99
alpha = 0.15
no_of_updates = 1

last_loaded_epoch = 0

actor, actor_optimizer, critic_1 , critic_optimizer_1, critic_2 , critic_optimizer_2, critic_target_1, critic_target_2, agent_memory = initialize_agent(env_attributes)

#actor, actor_optimizer, critic_1 , critic_optimizer_1, critic_2 , critic_optimizer_2, critic_target_1, critic_target_2, agent_memory, last_loaded_epoch = load_models(actor, actor_optimizer, critic_1, critic_optimizer_1, critic_2, critic_optimizer_2, env_attributes)

#update_hyperparameters(actor_optimizer, critic_optimizer_1, critic_optimizer_2, env_attributes)


realT_zon_model, dry_bulb_model, reward_model, env_memory_train , env_memory_test  = initialize_env_model(env_model_attributes)

if last_loaded_epoch !=0:  
    
    temp = pd.read_csv(metrics_path)
    temp = temp.loc[temp['episode']<=last_loaded_epoch ]
    temp.to_csv(metrics_path, index = False)
    
    filtered_df = temp.loc[temp["Type"] == "Train", ["episode", "extrinsic_reward"]]
    plot_scores_train_extrinsic = filtered_df.set_index("episode")["extrinsic_reward"].to_dict()
    
    train_time = temp.loc[ temp["Type"] == "Train"]['time_steps'].iloc[-1]
else:        
    
    with open(metrics_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Type','episode', 'time_steps', 'Length', 'Date', 'actor_loss', 'critic_1_loss','critic_2_loss','cost_tot', 'emis_tot','ener_tot','idis_tot','pdih_tot','pele_tot','pgas_tot','tdis_tot','extrinsic_reward'])
        file.close()
    
    plot_scores_train_extrinsic = {}
    
    train_time = 0
     
    



completed_initial_env_train = False  

for i_episode in range(last_loaded_epoch + 1, n_training_episodes+1): 
        
        start_time = time.time()
    
        done, time_step = 0, 1
        
        state, _ = env.reset()
        
        episode_rewards, episode_actor_loss, episode_critic_1_loss, episode_critic_2_loss = [], [], [], []
        
        while done==0:
            
            if time_step % 100 == 0:  print("Episode: ", i_episode, time_step)
            
            action, discrete_action, _ = actor.select_action(state [ env_attributes["filter_mask"] ] )
            
            next_state, reward, done = collect_from_actual_env( env, discrete_action )  
        
            agent_memory.remember( i_episode, state, action, reward, next_state, done)
        
            env_memory_train.remember(i_episode, state, action, discrete_action, reward, next_state, env) 
            
            state = next_state.copy()
            
            time_step +=1
            
            plot_scores_train_extrinsic[i_episode] = plot_scores_train_extrinsic.get(i_episode, 0) + reward
            
            episode_rewards.append(reward)
            
            
            if ( env_memory_train.is_full() ) and ( not completed_initial_env_train) : 
                
                train_hypernet(realT_zon_model, env_memory_train, exp_path, env_model_attributes) 
                
                train_hypernet(dry_bulb_model, env_memory_train, exp_path, env_model_attributes) 
                
                train_hypernet(reward_model, env_memory_train, exp_path, env_model_attributes) 
                
                completed_initial_env_train = True
                
                
            if completed_initial_env_train:
                
                env_memory_test.remember(i_episode, state, action, discrete_action, reward, next_state, env) 
                
                if env_memory_test.is_full():
                    
                    diagnosis_hypernet(realT_zon_model, dry_bulb_model, reward_model, env_memory_train, exp_path,  env_model_attributes )
                    
                    env_memory_test.clear_buffer()
        
            
            if  (agent_memory.memory_size() >= max_t )  : 
                    
                    for update in range(no_of_updates):
                        
                        state_samples, action_samples, reward_samples, next_state_samples, done_samples, _ =  agent_memory.sample_memory( env_attributes["filter_mask"], sample_size = env_attributes["sample_size"])
                        
                        if completed_initial_env_train:
                            
                           state_samples, action_samples, reward_samples, next_state_samples, done_samples = collect_from_learnt_env(state_samples, action_samples, reward_samples, next_state_samples, done_samples, actor, realT_zon_model, dry_bulb_model, reward_model, task_index = 0)
                        
                        batch_size =  env_attributes["sample_size"]
                        
                        for i in range(0, state_samples.size(0), batch_size):
                            
                            batch_states = state_samples[i:i + batch_size]
                            
                            batch_actions = action_samples[i:i + batch_size]
                            
                            batch_rewards = reward_samples[i:i + batch_size]
                            
                            batch_next_states = next_state_samples[i:i + batch_size]
                            
                            batch_done = done_samples[i:i + batch_size]
                            
                            
                            q_val_target = compute_target( actor, critic_target_1, critic_target_2, batch_rewards, batch_next_states, batch_done, gamma )
                            
                            l1, l2 = train_critic( critic_1, critic_2, critic_optimizer_1, critic_optimizer_2, batch_states, batch_actions , q_val_target )
                            
                            a1 = train_actor(actor, critic_1, critic_2, actor_optimizer, batch_states) 
                            
                            update_target_critic(critic_target_1, critic_target_2, critic_1, critic_2)
                            
                            episode_critic_1_loss.append(l1)
                            
                            episode_critic_2_loss.append(l2)
                            
                            episode_actor_loss.append(a1)
                            
        
        
        train_time = train_time + ( time.time() - start_time)                
                        
        env_type = "bestest_hydronic_heat_pump"
          
        
        save_train_results(i_episode, metrics_path, env , exp_path, env_attributes["points"], train_time, episode_rewards, plot_scores_train_extrinsic, episode_actor_loss, episode_critic_1_loss, episode_critic_2_loss, env_type)
           
        save_test_results(i_episode, metrics_path, env, exp_path, env_attributes["points"], actor, env_type) 
        
        if  i_episode % 10 == 0 : #was 5
               save_models(i_episode, exp_path, actor,actor_optimizer,critic_1, critic_optimizer_1 , critic_2 , critic_optimizer_2, agent_memory, env_type)    
        








