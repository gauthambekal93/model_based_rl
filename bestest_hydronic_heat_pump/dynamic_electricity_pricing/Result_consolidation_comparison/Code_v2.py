# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 20:53:24 2024

@author: gauthambekal93
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Result_consolidation_comparison')


metrics_exp1 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V4/Summary_Result_Models/complete_metrics_exp4.csv')

test_data= metrics_exp1.loc[(metrics_exp1['Type']=='Test') & (metrics_exp1['Date'] =='January 16')][['time_steps','extrinsic_reward']].values

# Separate the data into X and Y components
x1 = test_data[:, 0]
y1 = test_data[:, 1]



#-----------------------------------------------
metrics_exp2 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V8/Summary_Results/complete_metrics_exp8.csv')

test_data= metrics_exp2.loc[(metrics_exp2['Type']=='Test') &(metrics_exp2['Date'] =='January 16') ][['time_steps','extrinsic_reward']].values

x2 = test_data[:, 0]
y2 = test_data[:, 1]



# Create a scatter plot
plt.plot(x1, y1, color='blue',  alpha = 0.7, marker='s', label='RL based Pretraining')
plt.plot(x2, y2, color='green',  alpha = 0.7, marker='x', label='Environment Model based Pretraining')
# Add title and labels to the plot
plt.title('Bestest_hydronic_heat_pump\n Test data for Jan 16')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.legend()
# Show the plot
plt.show()








#------------------------------------------------------


metrics_exp4 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V4/Summary_Result_Models/complete_metrics_exp4.csv')

test_data= metrics_exp4.loc[(metrics_exp4['Type']=='Test') & (metrics_exp4['Date'] =='January 16')][['time_steps','extrinsic_reward']].values

# Separate the data into X and Y components
x1 = test_data[:, 0]
y1 = test_data[:, 1]



#-----------------------------------------------
metrics_exp8 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V8/Summary_Results/complete_metrics_exp8.csv')

test_data= metrics_exp8.loc[(metrics_exp8['Type']=='Test') &(metrics_exp8['Date'] =='January 16') ][['time_steps','extrinsic_reward']].values

x2 = test_data[:, 0]
y2 = test_data[:, 1]


#----------------------------------
metrics_exp7 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V7/Summary_Results/complete_metrics_exp7.csv')

test_data= metrics_exp7.loc[(metrics_exp7['Type']=='Test') &(metrics_exp7['Date'] =='January 16')][['time_steps','extrinsic_reward']].values

x3 = test_data[:, 0]
y3 = test_data[:, 1]

# Create a scatter plot
plt.plot(x1, y1, color='red',  alpha = 0.7, marker='o', label='No Transfer Learning')
plt.plot(x2, y2, color='blue',  alpha = 0.7, marker='s', label='RL based Transfer Learning')
plt.plot(x3, y3, color='green',  alpha = 0.7, marker='x', label='Environment Model based Transfer Learning')

# Add title and labels to the plot
plt.title('Bestest_hydronic_heat_pump\n Test data for Jan 16')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.legend()
# Show the plot
plt.show()








#-------------------------------------

metrics_exp14 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V14/Summary_Result_Models/complete_metrics_exp14.csv')

test_data= metrics_exp14.loc[(metrics_exp14['Type']=='Test') &(metrics_exp14['Date'] =='November 13')][['time_steps','extrinsic_reward']].values

x2 = test_data[:, 0]
y2 = test_data[:, 1]

#-----------------------------------
metrics_exp15 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V15/Summary_Result_Models/complete_metrics_exp15.csv')

test_data= metrics_exp15.loc[(metrics_exp15['Type']=='Test') &(metrics_exp15['Date'] =='November 13')][['time_steps','extrinsic_reward']].values

x3 = test_data[:, 0]
y3 = test_data[:, 1]

# Create a scatter plot
# Create a scatter plot
plt.plot(x1, y1, color='red',  alpha = 0.7, marker='o', label='No Transfer Learning')
plt.plot(x2, y2, color='blue',  alpha = 0.7, marker='s', label='RL based Transfer Learning')
plt.plot(x3, y3, color='green',  alpha = 0.7, marker='x', label='Environment Model based Transfer Learning')


# Add title and labels to the plot
plt.title('Bestest_hydronic \n Test data for November 13')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
# Show the plot
plt.show()



#----------------------------------------------------

metrics_exp13 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V13/Summary_Result_Models/complete_metrics_exp13.csv')

test_data= metrics_exp13.loc[(metrics_exp13['Type']=='Test') &(metrics_exp13['Date'] =='December 06')][['time_steps','extrinsic_reward']].values

x1 = test_data[:, 0]
y1 = test_data[:, 1]


#-------------------------------------

metrics_exp14 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V14/Summary_Result_Models/complete_metrics_exp14.csv')

test_data= metrics_exp14.loc[(metrics_exp14['Type']=='Test') &(metrics_exp14['Date'] =='December 06')][['time_steps','extrinsic_reward']].values

x2 = test_data[:, 0]
y2 = test_data[:, 1]

#-----------------------------------
metrics_exp15 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V15/Summary_Result_Models/complete_metrics_exp15.csv')

test_data= metrics_exp15.loc[(metrics_exp15['Type']=='Test') &(metrics_exp15['Date'] =='December 06')][['time_steps','extrinsic_reward']].values

x3 = test_data[:, 0]
y3 = test_data[:, 1]

# Create a scatter plot
# Create a scatter plot
plt.plot(x1, y1, color='red',  alpha = 0.7, marker='o',  label='No Transfer Learning')
plt.plot(x2, y2, color='blue',  alpha = 0.7, marker='s',  label='RL based Transfer Learning')
plt.plot(x3, y3, color='green',  alpha = 0.7, marker='x',  label='Environment Model based Transfer Learning')


# Add title and labels to the plot
plt.title('Bestest_hydronic \n Test data for December 06')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
# Show the plot
plt.show()



