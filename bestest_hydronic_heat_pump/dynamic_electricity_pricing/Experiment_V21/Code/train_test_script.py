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
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V21/Code')


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
from agent_actor_critic import Actor, Critic, Baseline, get_losses, select_action, Agent_Memory

#from env_learning_V2 import train_environment, run_trajectories, get_initial_uncertanity_range

#from env_model_architecture_V2 import Env_Memory

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
metrics_path = json_data['metrics_path']
rl_data_path = json_data['rl_data_path']


plot_scores_train_extrinsic = {}

plot_scores_test_extrinsic_jan17 = {}
plot_scores_test_extrinsic_apr19 = {}
plot_scores_test_extrinsic_nov15 = {}
plot_scores_test_extrinsic_dec08 = {}



def initialize_agent():
    
     train_start_time = 1000 #was 100
     
     actor = Actor(int(env_attributes["state_space"]),  int(env_attributes["action_bins"]), int(env_attributes["h_size"]), device, env_attributes["no_of_action_types"]).to(device) #was  int(env_attributes["action_space"])
     
     critic_1 = Critic(int(env_attributes["state_space"]), int(env_attributes["action_bins"]), int(env_attributes["h_size"]), device).to(device) #was  int(env_attributes["action_space"])
     
     critic_2 = Critic(int(env_attributes["state_space"]), int(env_attributes["action_bins"]), int(env_attributes["h_size"]), device).to(device) #was  int(env_attributes["action_space"])
     
     critic_target1 = critic_1.copy()
     
     critic_target2 = critic_2.copy()
     
     actor_optimizer = optim.Adam( actor.parameters(), lr= float(env_attributes["actor_lr"]) )
     
     critic_optimizer_1 = optim.Adam( critic_1.parameters(), lr =  float(env_attributes["critic_lr"]) )
     
     critic_optimizer_2 = optim.Adam( critic_2.parameters(), lr =  float(env_attributes["critic_lr"]) )
     
     return actor, critic_1 , critic_2 , critic_target1, critic_target2, actor_optimizer, critic_optimizer_1, critic_optimizer_2, train_start_time




def initialize_memory_buffers():
    
    memory = Agent_Memory()
    
    return memory


    
    
def collect_from_actual_env( i_episode, state, action):
    
    next_state, reward, done, _, res  = env.step(np.array(action) )

    return next_state, reward, done
    


def compute_target():
    
    next_action, action_log_probs = select_action(next_state_samples, actor)
    
    q_val_next = torch.min( torch.stack( [ critic_target1(next_state_samples, next_action), critic_target2(next_state_samples, next_action) ], dim=1), dim = 1)
    
    q_val_target = reward +  gamma*( 1 - done )* ( q_val_next  - alpha * action_log_probs   ) #need to replace with log probs !!!
    
    return q_val_target



def train_critic( state_samples, action_samples , q_val_target ):
    
    critic_loss_1 =  (critic_1(state_samples, action_samples) -  q_val_target )**2 
    
    critic_loss_2 =  (critic_2(state_samples, action_samples) -  q_val_target )**2
    
    


def train_actor():
    
    next_action, action_log_probs = select_action(next_state_samples, actor)
    
    q_val_1 = critic_1( state_samples, next_action )  
    
    q_val_2 = critic_2( state_samples, next_action )  
    
    q_val = torch.min( torch.stack([q_val_1, q_val_2 ] ), dim =1 )
    
    actor_loss = ( q_val - alpha* (action_log_probs ) )
    
    

def update_target_critic():
    
    critic_target1 = rho * critic_target1 + (1 - rho) * critic_1
    
    critic_target2 = rho * critic_target2 + (1 - rho) * critic_2

    


actor, critic_1 , critic_2 , critic_target1, critic_target1, actor_optimizer, critic_optimizer_1, critic_optimizer_2, train_start_time = initialize_agent()


memory = initialize_memory_buffers()


with open(metrics_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Type','episode', 'time_steps', 'Length', 'Date', 'actor_loss', 'critic_loss','baseline_loss','cost_tot', 'emis_tot','ener_tot','idis_tot','pdih_tot','pele_tot','pgas_tot','tdis_tot','extrinsic_reward'])
    file.close()
 

    
train_time, test_jan17_time, test_apr19_time, test_nov15_time, test_dec08_time = 0, 0, 0, 0, 0


    
for i_episode in range(1, n_training_episodes+1): 
        
        print("Episode: ", i_episode)
        
        state = env.reset()[0]
        
        episode_rewards, episode_actor_loss, episode_critic_loss = [], [], []
        
        start_time = time.time()
        
        action = select_action(state, actor)
            
        next_state, reward, done = collect_from_actual_env(  i_episode, state, action )
        
        memory.remember( i_episode, state, action, reward, next_state, done)
            
        if  t > train_start_time : 
                
                for update in no_of_updates:
                    
                    state_samples, action_samples, reward_samples, next_state_samples, done_samples =  memory.sample_memory()
                    
                    q_val_target = compute_target()
                    
                    train_critic(state_samples, action_samples , q_val_target)
                    
                    train_actor()
                    
                    update_target_critic()
                    
                    
                    #need to replace with log probs !!!
                    actor_loss = critic_1(state_samples, select_action(state_samples, actor) -  alpha*select_action(state_samples, actor) )
                    
                    critic_loss, actor_loss, baseline_loss = update_policy()
                    
                    episode_actor_loss.append(actor_loss.item())
                    
                    episode_critic_loss.append(critic_loss.item())

             
            episode_rewards.append(reward)
            
            plot_scores_train_extrinsic[i_episode] = plot_scores_train_extrinsic.get(i_episode, 0) + reward  #only used for plotting purpose
            
            state = next_state.copy()
                
                
        train_time = train_time + ( time.time() - start_time)
        
          
            
        if i_episode % 10 ==0:
            print("save policy model....")
            
            checkpoint = { 'model_state_dict': actor.state_dict(),  'optimizer_state_dict': actor_optimizer.state_dict() }
            
            torch.save(checkpoint, exp_path + '/Models/'+'actor_model_'+str(i_episode)+'_.pkl')
            
        
            checkpoint = { 'model_state_dict': critic.state_dict(),  'optimizer_state_dict': critic_optimizer.state_dict() }
            
            torch.save(checkpoint, exp_path + '/Models/'+'critic_model_'+str(i_episode)+'_.pkl')
        
        
        with open(metrics_path, 'a', newline='') as file:
               
                writer = csv.writer(file)   
            
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
                     
                     
                if i_episode % 10 == 0:  #need to make it 10   #was 25
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
                          
                file.close()

                    
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
          
                
                
            
            
    
