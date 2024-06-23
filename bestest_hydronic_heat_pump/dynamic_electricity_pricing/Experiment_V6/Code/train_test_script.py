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
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V6/Code')

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
#from simulation_environments import icm_parameters
#from policy_gradient_reinforce import Policy
from updated_plot import test_agent, plot_results
from memory_module import Memory
from model_architecture_env_policy import Environment_Module, Policy_Module


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



env_model = Environment_Module(int(env_attributes["state_space"]), n_actions=  int(env_attributes['no_of_action_types']) , alpha=1, beta=0.5) #beta was 0.2

env_model_optimizer = optim.Adam(env_model.parameters(), lr=float(0.0001 ) ) #was 0.001



checkpoint = torch.load(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V2/Summary_Result_Models/env_model_300_.pkl')  
env_model.load_state_dict(checkpoint['model_state_dict'])
env_model_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    

scheduler_gamma = 0.8
scheduler = lr_scheduler.StepLR(env_model_optimizer, step_size=10, gamma= scheduler_gamma)

#we need to put a logic on when to load the model


policy = Policy_Module(int(env_attributes["state_space"]),  int(env_attributes["action_bins"]), int(env_attributes["h_size"]), device, env_attributes["no_of_action_types"]).to(device) #was  int(env_attributes["action_space"])

policy_optimizer = optim.Adam(policy.parameters(), lr=float(env_attributes["lr"]))

#checkpoint = torch.load(r'model_based_rl/model_based_rl_v1/Results/Result11/Models_Nov_Jan_April/policy_model_15_.pkl')
#policy.load_state_dict(checkpoint['model_state_dict'])


policy_scheduler_gamma = 0.1
policy_scheduler = lr_scheduler.StepLR(policy_optimizer, step_size=10, gamma= policy_scheduler_gamma)


plot_scores_train_extrinsic = {}
plot_scores_train_intrinsic = {}
plot_scores_train_overall = {}

plot_scores_test_extrinsic_jan17 = {}
plot_scores_test_extrinsic_apr19 = {}

'''
class MinMaxScaler:
    
    def fit_transform(self, data):
        self.min = {}
        self.max = {}
        
        for i in range(data.shape[1]):
            
            if i==0: self.min[i] , self.max[i] = 0, 604800
            
            if i==1: self.min[i] , self.max[i] = 280, 310
            
            if i>=2 and i<=10 : self.min[i] , self.max[i] = 265, 303
            
            if i>=11 and i<=19: self.min[i] , self.max[i] = 0, 862
            
            if i==0: self.min[i] , self.max[i] = 0, 604800
            
            if i==0: self.min[i] , self.max[i] = 0, 604800
            
            if i==0: self.min[i] , self.max[i] = 0, 604800
            
            
    def transform(self, data):
        pass

    def inverse_transform(self, data):
        pass
'''    
    


def standerdize_features():
    
    states, actions,  rewards,  new_states = memory.sample_memory()
    
    X_scaler = MinMaxScaler()
    
    Y_scaler = MinMaxScaler()
    
    #split = 1.00 #0.80
    
    X = np.concatenate((states, actions ), axis=1)
    
    Y = np.concatenate((  new_states,  rewards.reshape(-1, 1)  ), axis=1)
    
    #split_index = int(len( X ) * split)
                      
    X_train = X_scaler.fit_transform( X  ) 
    
    #X_test = X_scaler.transform( X [split_index  : ] )
    
    Y_train = Y_scaler.fit_transform(Y )
    
    #Y_test = Y_scaler.transform( Y[ split_index : ] )
    
    return X_train, Y_train,  X_scaler, Y_scaler


def test_env(X_test, Y_test):
    
    next_state_loss_test, reward_loss_test = env_model.calc_loss( X_test, Y_test )
    return next_state_loss_test.item(),reward_loss_test.item()  


 
    
def train_env(X_train, Y_train, X_scaler, Y_scaler):
       
    epochs = 50 #was 1001 
    
    batch_size = 100 #was 10

    for epoch in range(epochs):
        
        if epoch%50==0:
            print("Epoch ", epoch, "X_train shape ",X_train.shape, "Y_train shape ",Y_train.shape)
        
        train_loss_next_state = []
        
        train_loss_reward = []
        
        for i in range(0, len(X_train), batch_size):
            
            next_state_loss_train, reward_loss_train = env_model.calc_loss( X_train[i: i+ batch_size],  Y_train[i: i+ batch_size] )
            
            env_loss_train = next_state_loss_train + reward_loss_train
            
            train_loss_next_state.append(next_state_loss_train.item())
            
            train_loss_reward.append(reward_loss_train.item())
            
            env_model_optimizer.zero_grad()
            
            env_loss_train.backward()
            
            env_model_optimizer.step()
        
        
        if epoch%50==0:
            print("Train Loss ",np.mean( train_loss_next_state)," ",np.mean(train_loss_reward) )
            #next_state_loss_test , reward_loss_test = test_env(X_test, Y_test)
            #print(" Test Loss ",  next_state_loss_test," ", reward_loss_test  )
            print( "\n Learning rate ",env_model_optimizer.param_groups[0]["lr"] )
        
    
        if env_model_optimizer.param_groups[0]["lr"] * scheduler_gamma >=0.001 :  
            scheduler.step()
        
         #shuffle dataset
        indices = np.random.permutation(len(X_train))
        
        X_train = X_train[indices]
        
        Y_train = Y_train[indices]
        
    #return env_model, X_scaler, Y_scaler



actual_env_train_step = 20  #was 10

#loss_thresh = 0.008 #was 0.008, tried 0.001

def combined_env( only_use_actual_env, update_env, update_policy, add_to_memory ):
    
    if memory.memory_size() >0 :  #we can only statderdize if data is present in memory
        
        X_train, Y_train, X_scaler, Y_scaler = standerdize_features()
    
        if update_env:  
            
            indices  = np.arange(650, len(X_train)-1)  #small set of initial values for training
            
            indices = np.random.permutation(indices)
            
            X_train = X_train[indices]
            
            Y_train = Y_train[indices]
            
            train_env(X_train, Y_train, X_scaler, Y_scaler)  
            
    
   
    policy_update_loop = 1 if update_env else 1  #was 5 if update_env else 1

    for _ in range(policy_update_loop):
        
        rewards = []
        
        log_probs = []
        
        state = env.reset()[0]
        
        count = 0
        
        for t in range(672): 
            
            action, log_prob = policy.act(state )
            
            log_probs.append(log_prob)
            
            action = np.array( action)  
        
            if ( t % actual_env_train_step == 0) or (only_use_actual_env):
                 
                  new_state, reward, done, _, res  = env.step(action)
                  
                  rewards.append(reward)
                  
                  if add_to_memory:
                      memory.remember(state, action, reward, new_state)
                
                  state = new_state.copy()
            
            else:
        
                  new_state, reward = env_model.forward( X_scaler.transform( np.concatenate((state,action)).reshape(1, - 1) )  )
                
                  Y_pred = Y_scaler.inverse_transform( np.concatenate( (np.array( new_state.detach()) ,  np.array(reward.detach()) ) , axis = 1) )
                
                  new_state, reward = Y_pred[:,:-1], Y_pred[:,-1:]
                
                  state =  new_state.copy().reshape(-1)  
                
                  rewards.append(reward[0,0])
                  
                  if rewards[-1]>0:
                      count+=1
                  print("Time ",t , "Reward ", rewards[-1])
        
         
            #if done: break   #we have commented this part
            
        if not only_use_actual_env:
            print("No. of Anomaly Rewards: ",count )
            
            with open(exp_path+'/Results/anomaly_rewards.csv', 'a', newline='') as file2:
                writer2 = csv.writer(file2)
                writer2.writerow([count])
            
        if update_policy:
            
            policy_loss = policy.calculate_loss( np.array(rewards) , log_probs, max_t, gamma) 
           
            policy_optimizer.zero_grad()
            
            policy_loss.backward()
            
            policy_optimizer.step() 
            
        
        else:
            
            policy_loss = policy.calculate_loss( np.array(rewards) , log_probs, max_t, gamma) 
            
            return rewards, log_probs, policy_loss
        
        #print("Env Train time ", time.time() - start)

  
  
time_taken = 0

exploration_episodes = 1  #was 1

with open(exp_path+'/Results/complete_metrics.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Type','episode', 'time_steps', 'Length', 'Date', 'Loss', 'cost_tot', 'emis_tot','ener_tot','idis_tot','pdih_tot','pele_tot','pgas_tot','tdis_tot','extrinsic_reward'])
    
    for i_episode in range(1, n_training_episodes+1): 
        
            print("Episode: ", i_episode)
            
            if i_episode <= exploration_episodes:
                
               _, _, _ = combined_env( only_use_actual_env = True, update_env = False, update_policy= False, add_to_memory = True)
               
               #standerdize_features()
               
               continue
           
            else:  
                start_time = time.time()
                
                if i_episode <=10:
                    
                    combined_env( only_use_actual_env = False, update_env = False, update_policy= True, add_to_memory = True)
                else:
                    combined_env( only_use_actual_env = False, update_env = True, update_policy= True, add_to_memory = True)
                    
                
                #combined_env( only_use_actual_env = True, update_env = False, update_policy= True, add_to_memory = False)
                
                time_taken = time_taken + ( time.time() - start_time )
                
                #writer.writerow([i_episode, time_taken])
                
                #print("Time taken: ", time.time() - start_time )
                
                rewards, log_probs, policy_loss = combined_env( only_use_actual_env= True, update_env = False, update_policy =False, add_to_memory = False )
    
                plot_scores_train_extrinsic[i_episode] = sum(rewards)
                    
                policy_loss = policy.calculate_loss( np.array(rewards) , log_probs, max_t, gamma) #log_probs is a tensor
                
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()
                    

            if i_episode % 10 ==0:
                print("save policy model....")
                model_path = exp_path + '/Models/'+'policy_model_'+str(i_episode)+'_.pkl'
        
                # Create a dictionary to store all necessary components
                checkpoint = {
                    'model_state_dict': policy.state_dict(),  # Model's state dictionary
                    'optimizer_state_dict': policy_optimizer.state_dict(),  # Optimizer's state dictionary (if using optimizer)
                    # Add any additional information you want to save
                }
                
                # Save the dictionary containing all components
                torch.save(checkpoint, model_path)
                
                
                print("save env model....")
                model_path = exp_path + '/Models/'+'env_model_'+str(i_episode)+'_.pkl'
        
                # Create a dictionary to store all necessary components
                checkpoint = {
                    'model_state_dict': env_model.state_dict(),  # Model's state dictionary
                    'optimizer_state_dict': env_model_optimizer.state_dict(),  # Optimizer's state dictionary (if using optimizer)
                    # Add any additional information you want to save
                }
                
                # Save the dictionary containing all components
                torch.save(checkpoint, model_path)
                
        
            if i_episode % 1 == 0:
                 print("=========Train=========")
                 
                 day_of_year = plot_results(env, rewards, points=['reaTZon_y','reaTSetHea_y','reaTSetCoo_y','oveHeaPumY_u',
                                         'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y'],
                                 log_dir=os.getcwd(), model_name='last_model', save_to_file=False, testcase ='bestest_hydronic_heat_pump', i_episode=i_episode)
                 
                 #print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
                 print("Policy Loss ", policy_loss.item())
                 print("KPIs \n ",  env.get_kpis() )
                 date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(day_of_year) - 1)
        
                 kpis = env.get_kpis()
                 data = {'Episode': i_episode,
                         "Time Taken":time_taken,
                         'Length':max_episode_length/24/3600,
                          'Date': date.strftime("%B %d"),
                         'Loss': policy_loss.item(),
                         'KPIs':  kpis,
                         "Intrinsic Rewards": sum(rewards),
                         "Extrinsic Reward": list(plot_scores_train_extrinsic.values())[-1] #sum(extrinsic_rewards)
                         }
                 
                 with open(exp_path + '/Results/Train_KPIs.json', "a") as json_file:
                     json_file.write(json.dumps(data, indent=4) )
                 
                 tmp = ['Train', i_episode, time_taken, max_episode_length/24/3600,  date.strftime("%B %d") , policy_loss.item()] +[kpis['cost_tot'] , kpis['emis_tot'], kpis['ener_tot'], kpis['idis_tot'], kpis['pdih_tot'],kpis['pele_tot'],kpis['pgas_tot'],kpis['tdis_tot'] ] + [list(plot_scores_train_extrinsic.values())[-1] ]
                 writer.writerow(tmp )
                 
                 
            if i_episode % 1 == 0:  #need to make it 10   #was 10
                 print("=========Test=========")
                 
                 
                 observations, actions, extrinsic_rewards_test, kpis, day_of_year, results = test_agent(env, policy, start_time_tests[0], episode_length_test, warmup_period_test, log_dir=os.getcwd(), model_name='last_model', save_to_file=False, plot=True, 
                                                                   points=['reaTZon_y','reaTSetHea_y','reaTSetCoo_y','oveHeaPumY_u',
                                                                            'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y'], 
                                                                   testcase='bestest_hydronic_heat_pump', i_episode=i_episode)
                 
                 plot_scores_test_extrinsic_jan17[i_episode] = sum(extrinsic_rewards_test)
                 
                 date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(day_of_year) - 1)
                 
                 kpis = env.get_kpis()
                 data = {'Episode': i_episode,
                         "Time Taken":time_taken,
                         'Date':date.strftime("%B %d"),
                         'Length':episode_length_test/24/3600,
                         'Loss': policy_loss.item(),
                         'KPIs':  env.get_kpis(),
                         "Extrinsic Reward":list(plot_scores_test_extrinsic_jan17.values())[-1]
                         }
                 
                 with open(exp_path + '/Results/Test_KPIs.json', "a") as json_file:
                     json_file.write(json.dumps(data, indent=4) )
                 
                 tmp = ['Test', i_episode, time_taken, max_episode_length/24/3600,  date.strftime("%B %d") , policy_loss.item()] +[kpis['cost_tot'] , kpis['emis_tot'], kpis['ener_tot'], kpis['idis_tot'], kpis['pdih_tot'],kpis['pele_tot'],kpis['pgas_tot'],kpis['tdis_tot'] ] + [list(plot_scores_test_extrinsic_jan17.values())[-1] ] 
                    
                 writer.writerow(tmp )
                 #for t, (test_reward, res) in enumerate(zip(extrinsic_rewards_test, results)):
                 #    writer.writerow(  ['Test_Jan17', str(i_episode), str(t), str(test_reward) ]+list(res.values()) )   
                 
                 observations, actions, extrinsic_rewards_test, kpis, day_of_year, results = test_agent(env, policy, start_time_tests[1], episode_length_test, warmup_period_test, log_dir=os.getcwd(), model_name='last_model', save_to_file=False, plot=True, 
                                                                   points=['reaTZon_y','reaTSetHea_y','reaTSetCoo_y','oveHeaPumY_u',
                                                                            'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y'], 
                                                                   testcase='bestest_hydronic_heat_pump', i_episode=i_episode)
                 
                 plot_scores_test_extrinsic_apr19[i_episode] = sum(extrinsic_rewards_test)
                 
                 kpis = env.get_kpis()
                 date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(day_of_year) - 1)
                 data = {'Episode': i_episode,
                         "Time Taken":time_taken,
                         'Date':date.strftime("%B %d"),
                         'Length':episode_length_test/24/3600,
                         'Loss': policy_loss.item(),
                         'KPIs':  env.get_kpis(),
                         "Extrinsic Reward":list(plot_scores_test_extrinsic_apr19.values())[-1]  
                         }
                 
                 with open(exp_path + '/Results/Test_KPIs.json', "a") as json_file:
                     json_file.write(json.dumps(data, indent=4) )
            
                 tmp = ['Test', i_episode, time_taken, max_episode_length/24/3600,  date.strftime("%B %d") , policy_loss.item()] +[kpis['cost_tot'] , kpis['emis_tot'], kpis['ener_tot'], kpis['idis_tot'], kpis['pdih_tot'],kpis['pele_tot'],kpis['pgas_tot'],kpis['tdis_tot'] ] + [list(plot_scores_test_extrinsic_apr19.values())[-1] ] 
                   
                 writer.writerow(tmp )
                    
                 
                    
            if i_episode % 5 == 0: 
                 
                 
                 plt.title( 'train_extrinsic_boptest_hydronic_heat_pump')
                 plt.xlabel('Episodes')
                 plt.ylabel('Extrinsic Rewards')
                 plt.plot( list(plot_scores_train_extrinsic.keys()), list(plot_scores_train_extrinsic.values()) )
                 plt.tight_layout()
                 plt.savefig(exp_path+ '/Results/train_extrinsic_boptest_hydronic_heat_pump.png')
                 plt.close() 
                 
                 
                 '''
                 if plot_scores_train_overall:
                     plt.title( 'train_overall_boptest_hydronic_heat_pump')
                     plt.xlabel('Episodes')
                     plt.ylabel('Overall Rewards')
                     plt.plot( list(plot_scores_train_overall.keys()), list(plot_scores_train_overall.values()) )
                     plt.tight_layout()
                     plt.savefig(exp_path + '/Results/train_overall_boptest_hydronic_heat_pump.png')
                     plt.close()    
                     
               
                 if plot_scores_train_intrinsic:
                     plt.title('train_intrinsic_boptest_hydronic_heat_pump')
                     plt.xlabel('Episodes')
                     plt.ylabel('Intrinsic Rewards')
                     plt.plot( list(plot_scores_train_intrinsic.keys()), list(plot_scores_train_intrinsic.values()) )
                     plt.tight_layout()
                     plt.savefig(exp_path+ '/Results/train_intrinsic_boptest_hydronic_heat_pump.png')
                     plt.close() 
                 
                 '''
                 
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
             
        
