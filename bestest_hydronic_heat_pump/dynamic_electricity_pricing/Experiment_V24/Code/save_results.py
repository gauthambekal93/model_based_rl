# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:48:10 2024

@author: gauthambekal93
"""

import os
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V24/Code')

import torch
import csv
import os
import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle 

from updated_plot import test_agent, plot_results
from simulation_environments import max_episode_length, start_time_tests,episode_length_test, warmup_period_test


year = 2024

test_jan17_time, test_apr19_time, test_nov15_time, test_dec08_time = 0, 0, 0, 0

#plot_scores_test_extrinsic_jan17 = {}
#plot_scores_test_extrinsic_apr19 = {}
#plot_scores_test_extrinsic_nov15 = {}
#plot_scores_test_extrinsic_dec08 = {}



def save_models(i_episode, exp_path, actor,actor_optimizer,critic_1, critic_optimizer_1 , critic_2 , critic_optimizer_2, agent_memory, env_type):
    
    print("save policy model....")
    
    checkpoint = { 'model_state_dict': actor.state_dict(),  'optimizer_state_dict': actor_optimizer.state_dict() }
    
    torch.save(checkpoint, exp_path + '/Models/'+'actor_model_'+str(i_episode)+'_.pkl')
    

    checkpoint = { 'model_state_dict': critic_1.state_dict(),  'optimizer_state_dict': critic_optimizer_1.state_dict() }
    
    torch.save(checkpoint, exp_path + '/Models/'+'critic_model_1_'+str(i_episode)+'_.pkl')       
    
    
    checkpoint = { 'model_state_dict': critic_2.state_dict(),  'optimizer_state_dict': critic_optimizer_2.state_dict() }
    
    torch.save(checkpoint, exp_path + '/Models/'+'critic_model_2_'+str(i_episode)+'_.pkl')    
    
    
    with open(exp_path + '/Models/'+"agent_buffer_data.pkl" , "wb") as f:
        pickle.dump(agent_memory, f)

    
    
def save_train_results(i_episode, metrics_path, env , exp_path, points, train_time, episode_rewards, plot_scores_train_extrinsic, episode_actor_loss, episode_critic_1_loss, episode_critic_2_loss, env_type):
    
    with open(metrics_path, 'a', newline='') as file:
            
        writer = csv.writer(file)   
        
        if i_episode % 1 == 0:
             print("=========Train=========")
              
             day_of_year = plot_results(env, episode_rewards, points = points,
                             log_dir=os.getcwd(), save_to_file=True, testcase = env_type, i_episode=i_episode)
             
             
             
             print("Actor Loss ", np.mean(episode_actor_loss) , "Critic 1 Loss ", np.mean(episode_critic_1_loss),"Critic 2 Loss ", np.mean(episode_critic_2_loss))
             print("KPIs \n ",  env.get_kpis() )
             train_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(day_of_year) - 1)
    
             kpis = env.get_kpis()
             
             tmp = ['Train', i_episode, train_time, max_episode_length/24/3600,  train_date.strftime("%B %d, %Y") , np.mean(episode_actor_loss), np.mean(episode_critic_1_loss), np.mean(episode_critic_2_loss) ] +[kpis['cost_tot'] , kpis['emis_tot'], kpis['ener_tot'], kpis['idis_tot'], kpis['pdih_tot'],kpis['pele_tot'],kpis['pgas_tot'],kpis['tdis_tot'] ] + [list(plot_scores_train_extrinsic.values())[-1] ]
             writer.writerow(tmp )
             
             
             plt.title( 'train_extrinsic_'+env_type)
             plt.xlabel('Episodes')
             plt.ylabel('Extrinsic Rewards')
             plt.plot( list(plot_scores_train_extrinsic.keys()), list(plot_scores_train_extrinsic.values()) )
             plt.tight_layout()
             plt.savefig(exp_path+ '/Results/train_extrinsic_'+ env_type + '.png')
             plt.close() 

        file.close()



def save_test_results(i_episode, metrics_path, env, exp_path, points, actor, env_type):
    
    #global test_jan17_time, test_apr19_time #test_nov15_time, test_dec08_time
    
    #global plot_scores_test_extrinsic_jan17, plot_scores_test_extrinsic_apr19, plot_scores_test_extrinsic_nov15, plot_scores_test_extrinsic_dec08
    
    df = pd.read_csv(metrics_path)
    
    df["Date"] = pd.to_datetime(df['Date'], format='mixed')
    
    
    filtered_df = df.loc[ df['Date'] == '2024-01-16', ["episode", "extrinsic_reward"]]
    
    plot_scores_test_extrinsic_jan17 = filtered_df.set_index("episode")["extrinsic_reward"].to_dict()
    
    temp = df.loc[ df['Date'] == '2024-01-16']['time_steps']
    
    if len(temp) ==0:
        test_jan17_time = 0
    else:
        test_jan17_time = temp.iloc[-1]
        
    
    filtered_df = df.loc[ df['Date'] == '2024-04-17', ["episode", "extrinsic_reward"]]
    
    plot_scores_test_extrinsic_apr19 = filtered_df.set_index("episode")["extrinsic_reward"].to_dict()
    
    temp = df.loc[ df['Date'] == '2024-04-17']['time_steps']
    
    if len(temp) ==0:
        test_apr19_time = 0
    else:
        test_apr19_time = temp .iloc[-1]
        
    
    
    with open(metrics_path, 'a', newline='') as file:
            
            writer = csv.writer(file)   
         
            print("=========Test=========")
            
            start = time.time()
            observations, actions, extrinsic_rewards_test, kpis, day_of_year, results = test_agent(env, actor, start_time_tests[0], episode_length_test, warmup_period_test, log_dir=os.getcwd(), save_to_file=True, plot=True, 
                                                              points = points, 
                                                              testcase = env_type, i_episode=i_episode)
            
            test_jan17_time = test_jan17_time + (time.time() - start)
            
            plot_scores_test_extrinsic_jan17[i_episode] = sum(extrinsic_rewards_test)
            
            test_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(day_of_year) - 1)
            
            kpis = env.get_kpis()
            
            tmp = ['Test', i_episode, test_jan17_time, max_episode_length/24/3600,  test_date.strftime("%B %d, %Y") , "NA", "NA", "NA" ] +[kpis['cost_tot'] , kpis['emis_tot'], kpis['ener_tot'], kpis['idis_tot'], kpis['pdih_tot'],kpis['pele_tot'],kpis['pgas_tot'],kpis['tdis_tot'] ] + [list(plot_scores_test_extrinsic_jan17.values())[-1] ] 
               
            writer.writerow(tmp )
            
            plt.title('test_jan17_' +env_type )
            plt.xlabel('Episodes')
            plt.ylabel('Test overall Rewards')
            plt.plot( list(plot_scores_test_extrinsic_jan17.keys()), list(plot_scores_test_extrinsic_jan17.values()) )
            plt.tight_layout()
            plt.savefig(exp_path+'/Results/test_jan17_' + env_type+ '.png')
            plt.close()  
            
            
            #------------------------------------------------------------#
            
            start = time.time()
            observations, actions, extrinsic_rewards_test, kpis, day_of_year, results = test_agent(env, actor,  start_time_tests[1], episode_length_test, warmup_period_test, log_dir=os.getcwd(), save_to_file=True, plot=True, 
                                                              points=points, 
                                                              testcase = env_type, i_episode=i_episode)
            
            test_apr19_time = test_apr19_time + (time.time() - start)
            
            plot_scores_test_extrinsic_apr19[i_episode] = sum(extrinsic_rewards_test)
            
            test_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(day_of_year) - 1)
            
            kpis = env.get_kpis()
   
            tmp = ['Test', i_episode, test_apr19_time, max_episode_length/24/3600,  test_date.strftime("%B %d, %Y") , "NA", "NA", "NA" ] +[kpis['cost_tot'] , kpis['emis_tot'], kpis['ener_tot'], kpis['idis_tot'], kpis['pdih_tot'],kpis['pele_tot'],kpis['pgas_tot'],kpis['tdis_tot'] ] + [list(plot_scores_test_extrinsic_apr19.values())[-1] ] 
              
            writer.writerow(tmp )
            
            plt.title('test_apr19_'+ env_type)
            plt.xlabel('Episodes')
            plt.ylabel('Test overall Rewards')
            plt.plot( list(plot_scores_test_extrinsic_apr19.keys()), list(plot_scores_test_extrinsic_apr19.values()) )
            plt.tight_layout()
            plt.savefig( exp_path + '/Results/test_apr19_' + env_type + '.png')
            plt.close() 
            
            #------------------------------------------------------------#
            
            '''
            start = time.time()
            observations, actions, extrinsic_rewards_test, kpis, day_of_year, results = test_agent(env, actor,  start_time_tests[2], episode_length_test, warmup_period_test, log_dir=os.getcwd(), save_to_file=True, plot=True, 
                                                              points=points, 
                                                              testcase = env_type, i_episode=i_episode)
            test_nov15_time = test_nov15_time + (time.time() - start)
            
            plot_scores_test_extrinsic_nov15[i_episode] = sum(extrinsic_rewards_test)
            
            test_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(day_of_year) - 1)
            
            kpis = env.get_kpis()
   
            tmp = ['Test', i_episode, test_nov15_time, max_episode_length/24/3600,  test_date.strftime("%B %d, %Y") , "NA", "NA", "NA" ] +[kpis['cost_tot'] , kpis['emis_tot'], kpis['ener_tot'], kpis['idis_tot'], kpis['pdih_tot'],kpis['pele_tot'],kpis['pgas_tot'],kpis['tdis_tot'] ] + [list(plot_scores_test_extrinsic_nov15.values())[-1] ] 
              
            writer.writerow(tmp )
            
            plt.title('test_nov15_'+ env_type)
            plt.xlabel('Episodes')
            plt.ylabel('Test overall Rewards')
            plt.plot( list(plot_scores_test_extrinsic_nov15.keys()), list(plot_scores_test_extrinsic_nov15.values()) )
            plt.tight_layout()
            plt.savefig( exp_path + '/Results/test_nov15_' + env_type + '.png')
            plt.close() 
            
            #------------------------------------------------------------#
            start = time.time()   
            observations, actions, extrinsic_rewards_test, kpis, day_of_year, results = test_agent(env, actor,  start_time_tests[3], episode_length_test, warmup_period_test, log_dir=os.getcwd(), save_to_file=True, plot=True, 
                                                             points=points, 
                                                             testcase = env_type, i_episode=i_episode)
            test_dec08_time = test_dec08_time + (time.time() - start)
            
            plot_scores_test_extrinsic_dec08[i_episode] = sum(extrinsic_rewards_test)
            
            test_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(day_of_year) - 1)
            
            kpis = env.get_kpis()
  
            tmp = ['Test', i_episode, test_dec08_time, max_episode_length/24/3600,  test_date.strftime("%B %d, %Y") , "NA", "NA", "NA" ] +[kpis['cost_tot'] , kpis['emis_tot'], kpis['ener_tot'], kpis['idis_tot'], kpis['pdih_tot'],kpis['pele_tot'],kpis['pgas_tot'],kpis['tdis_tot'] ] + [list(plot_scores_test_extrinsic_dec08.values())[-1] ] 
             
            writer.writerow(tmp )
            
            plt.title('test_dec08_'+ env_type)
            plt.xlabel('Episodes')
            plt.ylabel('Test overall Rewards')
            plt.plot( list(plot_scores_test_extrinsic_dec08.keys()), list(plot_scores_test_extrinsic_dec08.values()) )
            plt.tight_layout()
            plt.savefig( exp_path + '/Results/test_dec08_'+ env_type +'.png')
            plt.close() 

            '''
            
            file.close()
