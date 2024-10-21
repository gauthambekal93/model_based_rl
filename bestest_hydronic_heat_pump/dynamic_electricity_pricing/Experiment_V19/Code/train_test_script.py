# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:27:03 2024

@author: gauthambekal93
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 27 08:40:14 2024

@author: gauthambekal93
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:36:23 2024

@author: gauthambekal93
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:51:02 2024

@author: gauthambekal93
"""
import os
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V19/Code')

import numpy as np
import torch
import random

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  

 
import torch 
import torch.optim as optim
import json


from simulation_environments import bestest_hydronic_heat_pump, max_episode_length, start_time_tests,episode_length_test, warmup_period_test

from updated_plot import test_agent, plot_results
#from memory_module import Memory
from agent_actor_critic import Actor, Critic, get_losses, select_action, Agent_Memory

from env_learning_V2 import train_environment, run_trajectories, get_initial_uncertanity_range

from env_model_architecture_V2 import Env_Memory

#from sklearn.preprocessing import MinMaxScaler
import datetime
import time

#import torch.optim.lr_scheduler as lr_scheduler

import matplotlib.pyplot as plt

import warnings
import csv

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



plot_scores_train_extrinsic = {}

plot_scores_test_extrinsic_jan17 = {}
plot_scores_test_extrinsic_apr19 = {}
plot_scores_test_extrinsic_nov15 = {}
plot_scores_test_extrinsic_dec08 = {}



def initialize_agent():
    
     agent_update_step = 20
     
     actor = Actor(int(env_attributes["state_space"]),  int(env_attributes["action_bins"]), int(env_attributes["h_size"]), device, env_attributes["no_of_action_types"]).to(device) #was  int(env_attributes["action_space"])
     
     critic = Critic(int(env_attributes["state_space"]), int(env_attributes["h_size"]), device).to(device) #was  int(env_attributes["action_space"])
     
     actor_optimizer = optim.Adam( actor.parameters(), lr= float(env_attributes["actor_lr"]) )
     
     critic_optimizer = optim.Adam( critic.parameters(), lr =  float(env_attributes["critic_lr"]) )
     
     return actor, critic, actor_optimizer, critic_optimizer, agent_update_step
        
 
def initialize_env_models():
    
    env_update_step = 1 #50 

    model_room_temp = train_environment(model_type = "room_temperature")
    
    model_dry_bulb_temp = train_environment(model_type = "dry_bulb_temperature")
    
    model_rewards = train_environment(model_type = "rewards")
    
    return  model_room_temp, model_dry_bulb_temp, model_rewards, env_update_step
   


def initialize_memory_buffers():
    
    memory = Agent_Memory()
    
    env_memory = Env_Memory( int(env_attributes["action_bins"]) )  
    
    return memory, env_memory



def update_policy():
    
        state, actions, action_log_probs, rewards, next_state, value_preds = memory.sample_memory()
        
        critic_loss, actor_loss = get_losses(action_log_probs, rewards, value_preds, gamma, lam = 0.95 , device ="cpu")
        
        critic_optimizer.zero_grad()
        
        critic_loss.backward()

        critic_optimizer.step()
        
        actor_optimizer.zero_grad()
        
        actor_loss.backward()
        
        actor_optimizer.step()
       
        return critic_loss, actor_loss
    
    
    
def update_surrogate_env():
        
        model_room_temp.train_model()
        
        model_dry_bulb_temp.train_model()

        model_rewards.train_model()
        
 
    
def collect_from_actual_env(state, action, action_log_prob, state_value):
    
    next_state, reward, done, _, res  = env.step(np.array(action) )
    
    memory.remember(state, action, action_log_prob, reward, next_state, state_value)
                           
    env_memory.remember( state, action, next_state, reward)                 

    return next_state, reward, done
    


def collect_from_surrogate_env(state, action):
    
    state = torch.tensor(state).reshape(1, -1)
    
    action = torch.nn.functional.one_hot ( torch.tensor(action[0],  dtype=torch.int64), 
                                          num_classes = actor.fc3[0].out_features ).reshape(1,-1)
    
    next_state, reward, uncertanity = run_trajectories( model_room_temp, model_dry_bulb_temp, model_rewards, state, action )
    
    return next_state, reward, uncertanity



def update_uncertanity_range( uncertanity ):
    
    if uncertanity is None:
      min_uncertanity[0], max_uncertanity[0] =  get_initial_uncertanity_range(model_room_temp)
      min_uncertanity[1], max_uncertanity[1] =  get_initial_uncertanity_range(model_dry_bulb_temp)
      min_uncertanity[2], max_uncertanity[2] =  get_initial_uncertanity_range(model_rewards)
    
    else:
          if uncertanity[0] < min_uncertanity[0]:    min_uncertanity[0] = uncertanity[0]
          if uncertanity[1] < min_uncertanity[1]:    min_uncertanity[1] = uncertanity[1]
          if uncertanity[2] < min_uncertanity[2]:    min_uncertanity[2] = uncertanity[2]
          
          if uncertanity[0] > max_uncertanity[0]:    max_uncertanity[0] = uncertanity[0]
          if uncertanity[1] > max_uncertanity[1]:    max_uncertanity[1] = uncertanity[1]
          if uncertanity[2] > max_uncertanity[2]:    max_uncertanity[2] = uncertanity[2]
          


def use_surrogate_data():
    
    def generate_random_number(low, high):
        return low + torch.rand(1).item() * (high - low)
    
    rand_nums = torch.cat( [ generate_random_number( low, high ) for low, high in zip(min_uncertanity, max_uncertanity) ] , dim =0)
        
    return (rand_nums > uncertanity).all()   #all the uncertanities should be true
      
    
    
    




actor, critic, actor_optimizer, critic_optimizer, agent_update_step = initialize_agent()

model_room_temp, model_dry_bulb_temp, model_rewards, env_update_step = initialize_env_models()

memory, env_memory = initialize_memory_buffers()


with open(exp_path+'/Results/complete_metrics_2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Type','episode', 'time_steps', 'Length', 'Date', 'actor_loss', 'critic_loss','cost_tot', 'emis_tot','ener_tot','idis_tot','pdih_tot','pele_tot','pgas_tot','tdis_tot','extrinsic_reward'])
    file.close()
 
    
train_time, test_jan17_time, test_apr19_time, test_nov15_time, test_dec08_time = 0, 0, 0, 0, 0

#we simply initialize them to prevent syntax error
min_uncertanity, max_uncertanity, uncertanity = [None, None, None], [None, None, None], None
    
for i_episode in range(1, n_training_episodes+1): 
        
        with open(exp_path+'/Results/complete_metrics_2.csv', 'a', newline='') as file:
            
            writer = csv.writer(file)
            
            print("Episode: ", i_episode)
            
            state = env.reset()[0]
            
            episode_rewards, episode_actor_loss, episode_critic_loss = [], [], []
            
            start_time = time.time()
        
            for t in range(max_t): 
                
                print("Time step ", t)
                
                action, action_log_prob, state_value = select_action(state, critic, actor)
                
                if i_episode==1:
                    next_state, reward, done = collect_from_actual_env( state, action, action_log_prob, state_value )
                
                else:
                    
                    next_state, reward, uncertanity = collect_from_surrogate_env( state, action )
                    
                    update_uncertanity_range  ( uncertanity)
                    
                    if  use_surrogate_data():
                        
                        memory.remember(state, action, action_log_prob, reward, next_state, state_value)
                    else:
                        next_state, reward, done = collect_from_actual_env( state, action, action_log_prob, state_value )

                
                if ( (t % agent_update_step == 0 ) or ( t == (max_t - 1)) ) and ( t !=0 ) :
                    
                    critic_loss, actor_loss = update_policy()
                    
                    episode_actor_loss.append(actor_loss.item())
                    
                    episode_critic_loss.append(critic_loss.item())
    
                    memory.clear_memory()
                
                
                episode_rewards.append(reward)
                
                plot_scores_train_extrinsic[i_episode] = plot_scores_train_extrinsic.get(i_episode, 0) + reward  #only used for plotting purpose
                
                state = next_state.copy()
                
            train_time = train_time + ( time.time() - start_time)
            
            env_memory.save_to_csv()
            
            if i_episode == env_update_step:
               update_surrogate_env() 
               
            if i_episode ==1:   
               update_uncertanity_range(uncertanity)
               
            
            
            if i_episode % 10 ==0:
                print("save policy model....")
                
                checkpoint = { 'model_state_dict': actor.state_dict(),  'optimizer_state_dict': actor_optimizer.state_dict() }
                
                torch.save(checkpoint, exp_path + '/Models/'+'actor_model_'+str(i_episode)+'_.pkl')
                
            
                checkpoint = { 'model_state_dict': critic.state_dict(),  'optimizer_state_dict': critic_optimizer.state_dict() }
                
                torch.save(checkpoint, exp_path + '/Models/'+'critic_model_'+str(i_episode)+'_.pkl')
       
        
            if i_episode % 1 == 0:
                 print("=========Train=========")
                  
                 day_of_year = plot_results(env, episode_rewards, points=['reaTZon_y','reaTSetHea_y','reaTSetCoo_y','oveHeaPumY_u',
                                         'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y'],
                                 log_dir=os.getcwd(), model_name='last_model', save_to_file=False, testcase ='bestest_hydronic_heat_pump', i_episode=i_episode)
                 
                 
                 
                 print("Actor Loss ", np.mean(episode_actor_loss) , "Critic Loss ", np.mean(episode_critic_loss))
                 print("KPIs \n ",  env.get_kpis() )
                 train_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(day_of_year) - 1)
        
                 kpis = env.get_kpis()
                 
                 tmp = ['Train', i_episode, train_time, max_episode_length/24/3600,  train_date.strftime("%B %d, %Y") , np.mean(episode_actor_loss), np.mean(episode_critic_loss) ] +[kpis['cost_tot'] , kpis['emis_tot'], kpis['ener_tot'], kpis['idis_tot'], kpis['pdih_tot'],kpis['pele_tot'],kpis['pgas_tot'],kpis['tdis_tot'] ] + [list(plot_scores_train_extrinsic.values())[-1] ]
                 writer.writerow(tmp )
                 
                 
            if i_episode % 25 == 0:  #need to make it 10   #was 10
                 print("=========Test=========")
                 
                 start = time.time()
                 observations, actions, extrinsic_rewards_test, kpis, day_of_year, results = test_agent(env, actor, critic, start_time_tests[0], episode_length_test, warmup_period_test, log_dir=os.getcwd(), model_name='last_model', save_to_file=False, plot=True, 
                                                                   points=['reaTZon_y','reaTSetHea_y','reaTSetCoo_y','oveHeaPumY_u',
                                                                            'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y'], 
                                                                   testcase='bestest_hydronic_heat_pump', i_episode=i_episode)
                 
                 test_jan17_time = test_jan17_time + (time.time() - start)
                 
                 plot_scores_test_extrinsic_jan17[i_episode] = sum(extrinsic_rewards_test)
                 
                 test_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(day_of_year) - 1)
                 
                 kpis = env.get_kpis()
                 
                 tmp = ['Test', i_episode, test_jan17_time, max_episode_length/24/3600,  test_date.strftime("%B %d, %Y") , "NA", "NA" ] +[kpis['cost_tot'] , kpis['emis_tot'], kpis['ener_tot'], kpis['idis_tot'], kpis['pdih_tot'],kpis['pele_tot'],kpis['pgas_tot'],kpis['tdis_tot'] ] + [list(plot_scores_test_extrinsic_jan17.values())[-1] ] 
                    
                 writer.writerow(tmp )
                 
                 #------------------------------------------------------------#
                 
                 start = time.time()
                 observations, actions, extrinsic_rewards_test, kpis, day_of_year, results = test_agent(env, actor, critic,  start_time_tests[1], episode_length_test, warmup_period_test, log_dir=os.getcwd(), model_name='last_model', save_to_file=False, plot=True, 
                                                                   points=['reaTZon_y','reaTSetHea_y','reaTSetCoo_y','oveHeaPumY_u',
                                                                            'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y'], 
                                                                   testcase='bestest_hydronic_heat_pump', i_episode=i_episode)
                 
                 test_apr19_time = test_apr19_time + (time.time() - start)
                 
                 plot_scores_test_extrinsic_apr19[i_episode] = sum(extrinsic_rewards_test)
                 
                 test_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(day_of_year) - 1)
                 
                 kpis = env.get_kpis()
    
                 tmp = ['Test', i_episode, test_apr19_time, max_episode_length/24/3600,  test_date.strftime("%B %d, %Y") , "NA", "NA" ] +[kpis['cost_tot'] , kpis['emis_tot'], kpis['ener_tot'], kpis['idis_tot'], kpis['pdih_tot'],kpis['pele_tot'],kpis['pgas_tot'],kpis['tdis_tot'] ] + [list(plot_scores_test_extrinsic_apr19.values())[-1] ] 
                   
                 writer.writerow(tmp )
                 
                 #------------------------------------------------------------#
                 
                 start = time.time()
                 observations, actions, extrinsic_rewards_test, kpis, day_of_year, results = test_agent(env, actor, critic,  start_time_tests[2], episode_length_test, warmup_period_test, log_dir=os.getcwd(), model_name='last_model', save_to_file=False, plot=True, 
                                                                   points=['reaTZon_y','reaTSetHea_y','reaTSetCoo_y','oveHeaPumY_u',
                                                                            'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y'], 
                                                                   testcase='bestest_hydronic_heat_pump', i_episode=i_episode)
                 test_nov15_time = test_nov15_time + (time.time() - start)
                 
                 plot_scores_test_extrinsic_nov15[i_episode] = sum(extrinsic_rewards_test)
                 
                 test_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(day_of_year) - 1)
                 
                 kpis = env.get_kpis()
    
                 tmp = ['Test', i_episode, test_nov15_time, max_episode_length/24/3600,  test_date.strftime("%B %d, %Y") , "NA", "NA" ] +[kpis['cost_tot'] , kpis['emis_tot'], kpis['ener_tot'], kpis['idis_tot'], kpis['pdih_tot'],kpis['pele_tot'],kpis['pgas_tot'],kpis['tdis_tot'] ] + [list(plot_scores_test_extrinsic_nov15.values())[-1] ] 
                   
                 writer.writerow(tmp )
                    
                 #------------------------------------------------------------#
                 start = time.time()   
                 observations, actions, extrinsic_rewards_test, kpis, day_of_year, results = test_agent(env, actor, critic,  start_time_tests[3], episode_length_test, warmup_period_test, log_dir=os.getcwd(), model_name='last_model', save_to_file=False, plot=True, 
                                                                  points=['reaTZon_y','reaTSetHea_y','reaTSetCoo_y','oveHeaPumY_u',
                                                                           'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y'], 
                                                                  testcase='bestest_hydronic_heat_pump', i_episode=i_episode)
                 test_dec08_time = test_dec08_time + (time.time() - start)
                 
                 plot_scores_test_extrinsic_dec08[i_episode] = sum(extrinsic_rewards_test)
                 
                 test_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(day_of_year) - 1)
                 
                 kpis = env.get_kpis()
   
                 tmp = ['Test', i_episode, test_dec08_time, max_episode_length/24/3600,  test_date.strftime("%B %d, %Y") , "NA", "NA" ] +[kpis['cost_tot'] , kpis['emis_tot'], kpis['ener_tot'], kpis['idis_tot'], kpis['pdih_tot'],kpis['pele_tot'],kpis['pgas_tot'],kpis['tdis_tot'] ] + [list(plot_scores_test_extrinsic_dec08.values())[-1] ] 
                  
                 writer.writerow(tmp )
                          
                    

                    
            if i_episode % 5 == 0: 
                 
                 
                 plt.title( 'train_extrinsic_boptest_hydronic_heat_pump')
                 plt.xlabel('Episodes')
                 plt.ylabel('Extrinsic Rewards')
                 plt.plot( list(plot_scores_train_extrinsic.keys()), list(plot_scores_train_extrinsic.values()) )
                 plt.tight_layout()
                 plt.savefig(exp_path+ '/Results/train_extrinsic_boptest_hydronic_heat_pump.png')
                 plt.close() 
                 
                 
                 
                 if plot_scores_test_extrinsic_jan17:
                     plt.title('test_jan17_boptest_hydronic_heat_pump')
                     plt.xlabel('Episodes')
                     plt.ylabel('Test overall Rewards')
                     plt.plot( list(plot_scores_test_extrinsic_jan17.keys()), list(plot_scores_test_extrinsic_jan17.values()) )
                     plt.tight_layout()
                     plt.savefig(exp_path+'/Results/test_jan17_boptest_hydronic_heat_pump.png')
                     plt.close()    
                 
                 if plot_scores_test_extrinsic_apr19:
                     plt.title('test_apr19_boptest_hydronic_heat_pump')
                     plt.xlabel('Episodes')
                     plt.ylabel('Test overall Rewards')
                     plt.plot( list(plot_scores_test_extrinsic_apr19.keys()), list(plot_scores_test_extrinsic_apr19.values()) )
                     plt.tight_layout()
                     plt.savefig( exp_path + '/Results/test_apr19_boptest_hydronic_heat_pump.png')
                     plt.close() 
                 
                 if plot_scores_test_extrinsic_nov15:
                        plt.title('test_nov15_boptest_hydronic_heat_pump')
                        plt.xlabel('Episodes')
                        plt.ylabel('Test overall Rewards')
                        plt.plot( list(plot_scores_test_extrinsic_nov15.keys()), list(plot_scores_test_extrinsic_nov15.values()) )
                        plt.tight_layout()
                        plt.savefig( exp_path + '/Results/test_nov15_boptest_hydronic_heat_pump.png')
                        plt.close() 
                   
                 if plot_scores_test_extrinsic_dec08:
                       plt.title('test_dec08_boptest_hydronic_heat_pump')
                       plt.xlabel('Episodes')
                       plt.ylabel('Test overall Rewards')
                       plt.plot( list(plot_scores_test_extrinsic_dec08.keys()), list(plot_scores_test_extrinsic_dec08.values()) )
                       plt.tight_layout()
                       plt.savefig( exp_path + '/Results/test_dec08_boptest_hydronic_heat_pump.png')
                       plt.close() 
              
                
                
            file.close()
            
    
