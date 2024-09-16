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
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V17/Code')

import numpy as np
import torch
import random

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  


#from collections import deque  
import torch 
import torch.optim as optim
import json


from simulation_environments import bestest_hydronic_heat_pump, max_episode_length, start_time_tests,episode_length_test, warmup_period_test

from updated_plot import test_agent, plot_results
from memory_module import Memory
from agent_actor_critic import Actor, Critic, get_losses, select_action


from sklearn.preprocessing import MinMaxScaler
import datetime
import time

import torch.optim.lr_scheduler as lr_scheduler

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

memory = Memory()


#env_model = Environment_Module(int(env_attributes["state_space"]), n_actions=  int(env_attributes["action_space"]) , alpha=1, beta=0.5) #beta was 0.2

#env_model_optimizer = optim.Adam(env_model.parameters(), lr=float(0.1 ) ) #icm_parameters["lr"]  #was 0.0001


    

#scheduler_gamma = 0.8
#scheduler = lr_scheduler.StepLR(env_model_optimizer, step_size=10, gamma= scheduler_gamma)

#we need to put a logic on when to load the model


actor = Actor(int(env_attributes["state_space"]),  int(env_attributes["action_bins"]), int(env_attributes["h_size"]), device, env_attributes["no_of_action_types"]).to(device) #was  int(env_attributes["action_space"])

critic = Critic(int(env_attributes["state_space"]), int(env_attributes["h_size"]), device).to(device) #was  int(env_attributes["action_space"])

actor_optimizer = optim.Adam( actor.parameters(), lr= float(env_attributes["actor_lr"]) )

critic_optimizer = optim.Adam( critic.parameters(), lr =  float(env_attributes["critic_lr"]) )

agent_update_step = 20 #was 100
#checkpoint = torch.load(r'model_based_rl/model_based_rl_v1/Results/Result11/Models_Nov_Jan_April/policy_model_15_.pkl')
#policy.load_state_dict(checkpoint['model_state_dict'])


#policy_scheduler_gamma = 0.1
#policy_scheduler = lr_scheduler.StepLR(policy_optimizer, step_size=10, gamma= policy_scheduler_gamma)


plot_scores_train_extrinsic = {}
#plot_scores_train_intrinsic = {}
#plot_scores_train_overall = {}

plot_scores_test_extrinsic_jan17 = {}
plot_scores_test_extrinsic_apr19 = {}
plot_scores_test_extrinsic_nov15 = {}
plot_scores_test_extrinsic_dec08 = {}



actual_env_train_step = 10#was 5 #was 5, and policy was not converging well

loss_thresh = 0.008 #was 0.008, tried 0.001



def update_policy( only_use_actual_env ):
    
    if only_use_actual_env:
        
        states, actions, action_log_probs, rewards, new_states, value_preds = memory.sample_memory()
        
        critic_loss, actor_loss = get_losses(action_log_probs, rewards, value_preds, gamma, lam = 0.95 , device ="cpu")
        
        critic_optimizer.zero_grad()
        
        critic_loss.backward()

        critic_optimizer.step()
        
        actor_optimizer.zero_grad()
        
        actor_loss.backward()
        
        actor_optimizer.step()
       
        return critic_loss, actor_loss
    

#exploration_episodes = 1  #was 1

with open(exp_path+'/Results/complete_metrics.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Type','episode', 'time_steps', 'Length', 'Date', 'actor_loss', 'critic_loss','cost_tot', 'emis_tot','ener_tot','idis_tot','pdih_tot','pele_tot','pgas_tot','tdis_tot','extrinsic_reward'])
    file.close()
 
train_time, test_jan17_time, test_apr19_time, test_nov15_time, test_dec08_time = 0, 0, 0, 0, 0
    
for i_episode in range(0, n_training_episodes+1): 
        
        with open(exp_path+'/Results/complete_metrics.csv', 'a', newline='') as file:
            
            writer = csv.writer(file)
            
            print("Episode: ", i_episode)
            
            state = env.reset()[0]
            
            episode_rewards, episode_actor_loss, episode_critic_loss = [], [], []
            
            start_time = time.time()
            
            for t in range(max_t): 
                
                print("Time step ", t)
                
                action, action_log_prob, state_value = select_action(state, critic, actor)
                
                new_state, reward, done, _, res  = env.step(np.array(action))
                
                memory.remember(state, action, action_log_prob, reward, new_state, state_value)
                
                if ( (t % agent_update_step == 0 ) or (t == (max_t - 1)) ) and ( t !=0 ) :
                    
                    critic_loss, actor_loss = update_policy(only_use_actual_env = True)
                    
                    episode_actor_loss.append(actor_loss.item())
                    
                    episode_critic_loss.append(critic_loss.item())
    
                    memory.clear_memory()
                
                episode_rewards.append(reward)
                
                plot_scores_train_extrinsic[i_episode] = plot_scores_train_extrinsic.get(i_episode, 0) + reward  #only used for plotting purpose
                
            train_time = train_time + ( time.time() - start_time)
            
            
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
            
    
