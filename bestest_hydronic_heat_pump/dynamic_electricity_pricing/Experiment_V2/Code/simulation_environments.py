# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:50:32 2024

@author: gauthambekal93
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:36:01 2024

@author: gauthambekal93
"""

import os

#os.chdir('C:/Users/gauthambekal93/Research/rl_collaboration_project/project1-boptest-gym-master')
os.chdir(r'C:/Users/gauthambekal93/Research/rl_collaboration_project/project1-boptest-gym-master/model_based_rl/model_based_rl_v1/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V2/Code')

import numpy as np
import torch
import random

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  

#import math
#import json

#import requests
from boptestGymEnv import BoptestGymEnv, DiscretizedActionWrapper, DiscretizedObservationWrapper, NormalizedObservationWrapper
from boptestGymEnv import BoptestGymEnvRewardClipping, BoptestGymEnvRewardWeightDiscomfort

#import random
#from IPython.display import clear_output
#from collections import namedtuple
#from itertools import count
#from collections import deque  
#import time  

#import numpy as np

#from collections import deque

#import matplotlib.pyplot as plt
# %matplotlib inline

# PyTorch
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#from torch.distributions import Categorical

#from examples.test_and_plot import test_agent, plot_results
# Decide the state-action space of your test case
#import random

# Seed for random starting times of episodes
#seed = 123456
#random.seed(seed)
# Seed for random exploration and epsilon-greedy schedule
#np.random.seed(seed)

from collections import OrderedDict

#with open('Progressive_neural_nets/all_paths.json', 'r') as openfile:
#    paths = json.load(openfile)

url = 'http://127.0.0.1:5000' 
operational_data = 'http://127.0.0.1:5000/initialize'


warmup_period = 24*3600
episode_length_test = 14*24*3600 #was 7*24*3600
warmup_period_test  = 7*24*3600
start_time_tests    = [(23-7)*24*3600, (115-7)*24*3600]  # (Jan 16 to Jan 30) and (April 18 to May 02) 

max_episode_length = 14*24*3600 #was 7*24*3600
random_start_time =True#False #True
step_period = 900
render_episodes = False # False

predictive_period =  24*3600  #2700 #was 24*3600 #None
regressive_period = 6*3600  #None #2700 #was 6*3600 #None

icm_parameters =  { "lr": str(0.001) }#0.0001


def twozone_commercial_hydronic():
   
    actions = ["hydronicSystem_oveMDayZ_u","hydronicSystem_oveMNigZ_u","hydronicSystem_oveMpumCon_u","hydronicSystem_oveTHea_u"]
    #no_of_action_types = len(actions)
    points=  ["hydronicSystem_oveMDayZ_u", "hydronicSystem_oveMNigZ_u", "thermostatDayZon_oveTsetZon_u", "thermostatNigZon_oveTsetZon_u", "dayZon_reaTRooAir_y", "nigZon_reaTRooAir_y", "dayZon_reaTavgFloHea_y",               "dayZon_reaCO2RooAir_y", "weatherStation_reaWeaTDryBul_y", "weatherStation_reaWeaHDirNor_y", "hydronicSystem_oveMpumCon_u", "hydronicSystem_oveTHea_u"]
    
    observations = {
          'time':(0,604800),
          
          'dayZon_reaTRooAir_y': (273,320), #[K] [min=None, max=None]: Air temperature of zone Bth
          'nigZon_reaTRooAir_y': (273,320),#[K] [min=None, max=None]: Air temperature of zone Liv
          
      #    'dayZon_reaTavgFloHea_y':(273,320),
       #   'nigZon_reaTavgFloHea_y':(273,320),
          
          'dayZon_reaCO2RooAir_y':(400,1000), #[ppm] [min=None, max=None]: Zone air CO2 concentration
          'nigZon_reaCO2RooAir_y':(400,1000), #[ppm] [min=None, max=None]: Zone air CO2 concentration
          
           'Occupancy[Day]':(0,10),
           'Occupancy[Night]':(0,10),
           
           'weatherStation_reaWeaTDryBul_y':(273,320),
           
           'dayZon_reaMFloHea_y':(0, 10),  #the range i put here is arbitary
           'nigZon_reaMFloHea_y':(0, 10),
           
           'dayZon_reaTsupFloHea_y':(273,330),
           'nigZon_reaTsupFloHea_y':(273,330)
        }



    excluding_periods = []
    for start_time_test in start_time_tests:
         excluding_periods.append((start_time_test, start_time_test+episode_length_test))
     # Summer period (from June 21st till September 22nd). 
     # Excluded since no heating during this period (nothing to learn).
    excluding_periods.append((173*24*3600, 266*24*3600))  # June 23 to Sept 26 its hot so we dont need to learn anything since heating system is off


    n_bins_act = 2 #was 1


    env = BoptestGymEnv(      
                url                  = url,
                actions              = actions,
                observations         = observations,
                #predictive_period    = predictive_period,            #was commented
                #regressive_period    = regressive_period,             #was commented
                random_start_time    = random_start_time,   #was True
                excluding_periods    = excluding_periods,    #was commented
                max_episode_length   = max_episode_length, #was  7*24*3600, 
                warmup_period        = warmup_period,
                step_period          = step_period,
                #render_episodes      =  render_episodes
                )  #was False    
        
        
    env = NormalizedObservationWrapper(env)
    env = DiscretizedActionWrapper(env, n_bins_act)  #action space same as original code
        

    #env = create_train_environment() 
    
    obs = env.reset()[0] 
        
    obs_dim = obs.shape[0]

    env_attributes = {
                      'state_space': str( obs_dim),
                      "actions":actions,
                      "action_space":str(env.action_space.n),
                      "action_bins": n_bins_act + 1,
                      "points":points,
                      "h_size":str(100),
                      "n_training_episodes": str(1000),
                      "n_evaluation_episodes": (10),
                      "max_t": str(1000),   
                      "gamma": str(0.90),
                      "lr": str(0.001)

                      }
    return  env,  env_attributes 





def bestest_hydronic_heat_pump():
    

    
    #actions = ['oveFan_u', 'oveHeaPumY_u','ovePum_u','oveTSet_u']
    actions = ['oveHeaPumY_u']
    #no_of_action_types = len(actions)
    points = ["reaTZon_y","reaTSetHea_y","reaTSetCoo_y","oveHeaPumY_u", "weaSta_reaWeaTDryBul_y", "weaSta_reaWeaHDirNor_y"]
    

    #o'nly time and realTZon_y was used
    
    observations          = OrderedDict([('time',(0,604800)),
                             ('reaTZon_y',(280.,310.)),
                             ('TDryBul',(265,303)),
                             ('HDirNor',(0,862)),
                             ('InternalGainsRad[1]',(0,219)),
                             ('PriceElectricPowerHighlyDynamic',(-0.4,0.4)),
                             ('LowerSetp[1]',(280.,310.)),
                             ('UpperSetp[1]',(280.,310.))])
           
    excluding_periods = []
    for start_time_test in start_time_tests:
         excluding_periods.append((start_time_test, start_time_test+episode_length_test))
     # Summer period (from June 21st till September 22nd). 
     # Excluded since no heating during this period (nothing to learn).
    excluding_periods.append((173*24*3600, 266*24*3600))  # June 23 to Sept 26 its hot so we dont need to learn anything since heating system is off
    
    n_bins_act = 10 #was 1
    

    env = BoptestGymEnv(      
                url                  = url,
                actions              = actions,
                observations         = observations,
                predictive_period    = predictive_period,            #was commented
                regressive_period    = regressive_period,             #was commented
                scenario              = {'electricity_price':'highly_dynamic'},
                random_start_time    = random_start_time,   #was True
                excluding_periods    = excluding_periods,    #was commented
                max_episode_length   = max_episode_length, #was  7*24*3600, 
                warmup_period        = warmup_period,
                step_period          = step_period,
                #render_episodes      =  render_episodes
                )  #was False    
    
        
    env = NormalizedObservationWrapper(env)
    env = DiscretizedActionWrapper(env, n_bins_act)  #action space same as original code
        

    obs = env.reset()[0] 
        
    obs_dim = obs.shape[0]

    env_attributes = {
                       "state_space": str( obs_dim),
                       "actions": actions,
                       "action_space": str(env.action_space.n),
                       "action_bins": str(n_bins_act + 1),
                       "points":points,
                       "h_size":str(int((obs_dim//2))),  #was 50, was 5, was 100, was 100
                       "n_training_episodes": str(400),
                      # "n_evaluation_episodes": str(10),
                       "max_t": str(2000), #was 2000   
                       "gamma": str(0.90),
                        "lr": str( 0.001),  #was 0.0001  #was 0.01  was 0.001, 0.0001, was 0.01 #was0.01  #was 0.001,  was 0.001
                        "no_of_action_types":len(actions)
                      }
    
    return  env,  env_attributes 





def bestest_hydronic():
        
    #actions = ['ovePum_u','oveTSetSup_u']  
    actions = ["ovePum_u","oveTSetCoo_u","oveTSetHea_u", "oveTSetSup_u"] #was only 'oveTSetSup_u'  
    #no_of_action_types = len(actions)
    
    points = ["reaTRoo_y","reaCO2RooAir_y","oveTSetHea_u","oveTSetCoo_u","oveTSetSup_u","ovePum_u","oveTSetSup_activate","weaSta_reaWeaTDryBul_y","weaSta_reaWeaHDirNor_y"]
    
    #'reaTRoo_y': (280, 285) THIS OBSERVATION LEADS TO A STRAIGHT LINE AFTER A FEW EPISODES
    observations = { 'time':(0,604800),
                     'reaCO2RooAir_y':(0,1000),   #(-100000,100000)  #was 0, 100
                     'reaTRoo_y':(280,310),  #(-1000,1000)
                     'weaSta_reaWeaTDryBul_y':(275, 330),
                     'weaSta_reaWeaTWetBul_y':(275,330)
                     #'TDryBul':(265,303),   #'Outside drybulb temperature measurement
                     #'HDirNor':(0,862),      #Direct normal radiation  'measurement
                     #'InternalGainsRad[1]':(0,219),    #forecasting variable
                     #'PriceElectricPowerHighlyDynamic':(-0.4,0.4),  #forecasting variable
                     #'LowerSetp[1]':(280.,310.),
                     #'UpperSetp[1]':(280.,310.)  
                     }
    
    
    
    
    excluding_periods = []
    for start_time_test in start_time_tests:
         excluding_periods.append((start_time_test, start_time_test+episode_length_test))
     # Summer period (from June 21st till September 22nd). 
     # Excluded since no heating during this period (nothing to learn).
    excluding_periods.append((79*24*3600, 355*24*3600))  # June 23 to Sept 26 its hot so we dont need to learn anything since heating system is off
    
    
    n_bins_act = 1
    
    #testcase ='bestest_hydronic'
    

    env = BoptestGymEnv(      
                url                  = url,
                actions              = actions,
                observations         = observations,
                predictive_period    = predictive_period,            #was commented
                regressive_period    = regressive_period,             #was commented
                random_start_time    = random_start_time,   #was True
                excluding_periods     = excluding_periods,    #was commented
                max_episode_length   = max_episode_length, #was  7*24*3600, 
                warmup_period        = warmup_period,
                step_period          = step_period
               # render_episodes      =  render_episodes,
                )  #was True    
        
        
    env = NormalizedObservationWrapper(env)
    env = DiscretizedActionWrapper(env, n_bins_act)  #action space same as original code
        
    

    obs = env.reset()[0] 
        
    obs_dim = obs.shape[0]

    env_attributes = {
                      "state_space": str( obs_dim),
                      "actions": actions,
                      "action_space": str(env.action_space.n),
                      "action_bins":  str(n_bins_act + 1),
                      "points":points,
                      "h_size":str(50),
                      "n_training_episodes": str(300),
                      "n_evaluation_episodes":str(10),
                      "max_t": "1000",   #we need to change it according to the length of our trajectory
                      "gamma": str(0.90),
                      "lr":str(0.001)
                      }
   

    
    return  env,  env_attributes 
    
    
    
    
if __name__ == "__main__":    
    env, env_attributes  = bestest_hydronic_heat_pump()
    

    


