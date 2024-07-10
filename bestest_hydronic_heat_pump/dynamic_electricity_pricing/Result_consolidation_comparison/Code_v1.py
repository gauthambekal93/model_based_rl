# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:13:34 2024

@author: gauthambekal93
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Result_consolidation_comparison')

#------------------------------------------------------
metrics_exp4 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V4/Summary_Result_Models/complete_metrics_exp4.csv')

train_data= metrics_exp4.loc[metrics_exp4['Type']=='Train'][['time_steps','extrinsic_reward']].values

# Separate the data into X and Y components
x = train_data[:, 0]
y = train_data[:, 1]

# Create a scatter plot
plt.scatter(x, y, color='red', marker='o')

# Add title and labels to the plot
plt.title('Bestest_hydronic_heat_pump \n Train data with No transfer learning for March 01')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()

#------------------------------------------------------------
metrics_exp8 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V8/Summary_Results/complete_metrics_exp8.csv')

train_data= metrics_exp8.loc[metrics_exp8['Type']=='Train'][['time_steps','extrinsic_reward']].values

x = train_data[:, 0]
y = train_data[:, 1]

# Create a scatter plot
plt.scatter(x, y, color='blue', marker='o')

# Add title and labels to the plot
plt.title('Bestest_hydronic_heat_pump\n Train data with RL Agent Model based transfer learning for March 01')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()


#----------------------------------
metrics_exp7 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V7/Summary_Results/complete_metrics_exp7.csv')

train_data= metrics_exp7.loc[metrics_exp7['Type']=='Train'][['time_steps','extrinsic_reward']].values

x = train_data[:, 0]
y = train_data[:, 1]

# Create a scatter plot
plt.scatter(x, y, color='green', marker='o')

# Add title and labels to the plot
plt.title('Bestest_hydronic_heat_pump\n Train data with Environment Model based transfer learning for March 01')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()



#----------------------------------------April Test-----------------------------------------------


metrics_exp4 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V4/Summary_Result_Models/complete_metrics_exp4.csv')

train_data= metrics_exp4.loc[(metrics_exp4['Type']=='Test') & (metrics_exp4['Date'] =='April 17')][['time_steps','extrinsic_reward']].values

# Separate the data into X and Y components
x = train_data[:, 0]
y = train_data[:, 1]

# Create a scatter plot
plt.scatter(x, y, color='red', marker='o')

# Add title and labels to the plot
plt.title('Bestest_hydronic_heat_pump\n Test data with No transfer learning for April 17')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()

#-----------------------------------------------------

metrics_exp8 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V8/Summary_Results/complete_metrics_exp8.csv')

train_data= metrics_exp8.loc[(metrics_exp8['Type']=='Test') &(metrics_exp8['Date'] =='April 17') ][['time_steps','extrinsic_reward']].values

x = train_data[:, 0]
y = train_data[:, 1]

# Create a scatter plot
plt.scatter(x, y, color='blue', marker='o')

# Add title and labels to the plot
plt.title('Bestest_hydronic_heat_pump\n Test data with RL Agent Model based transfer learning for April 17')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()


#----------------------------------
metrics_exp7 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V7/Summary_Results/complete_metrics_exp7.csv')

train_data= metrics_exp7.loc[(metrics_exp7['Type']=='Test') &(metrics_exp7['Date'] =='April 17')][['time_steps','extrinsic_reward']].values

x = train_data[:, 0]
y = train_data[:, 1]

# Create a scatter plot
plt.scatter(x, y, color='green', marker='o')

# Add title and labels to the plot
plt.title('Bestest_hydronic_heat_pump\n Test data with Environment Model based transfer learning for April 17')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()

#--------------------------------Jan Test---------------------------


metrics_exp4 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V4/Summary_Result_Models/complete_metrics_exp4.csv')

train_data= metrics_exp4.loc[(metrics_exp4['Type']=='Test') & (metrics_exp4['Date'] =='January 16')][['time_steps','extrinsic_reward']].values

# Separate the data into X and Y components
x = train_data[:, 0]
y = train_data[:, 1]

# Create a scatter plot
plt.scatter(x, y, color='red', marker='o')

# Add title and labels to the plot
plt.title('Bestest_hydronic_heat_pump\n Test data with No transfer learning for Jan 16')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()

#-----------------------------------------------
metrics_exp8 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V8/Summary_Results/complete_metrics_exp8.csv')

train_data= metrics_exp8.loc[(metrics_exp8['Type']=='Test') &(metrics_exp8['Date'] =='January 16') ][['time_steps','extrinsic_reward']].values

x = train_data[:, 0]
y = train_data[:, 1]

# Create a scatter plot
plt.scatter(x, y, color='blue', marker='o')

# Add title and labels to the plot
plt.title('Bestest_hydronic_heat_pump\n Test data with RL Agent Model based transfer learning for Jan 16')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()

#----------------------------------
metrics_exp7 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V7/Summary_Results/complete_metrics_exp7.csv')

train_data= metrics_exp7.loc[(metrics_exp7['Type']=='Test') &(metrics_exp7['Date'] =='January 16')][['time_steps','extrinsic_reward']].values

x = train_data[:, 0]
y = train_data[:, 1]

# Create a scatter plot
plt.scatter(x, y, color='green', marker='o')

# Add title and labels to the plot
plt.title('Bestest_hydronic_heat_pump\n Test data with Environment model based transfer learning for Jan 16')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()



#---------------------------------PRETRAINED ON bestest_hydronic_HEAT_PUMP AND FINETUNED ON bestest_hydronic------------------------------

#-----------------------Train for December 14------------------------------------------------
metrics_exp13 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V13/Summary_Result_Models/complete_metrics_exp13.csv')

train_data= metrics_exp13.loc[metrics_exp13['Type']=='Train'][['time_steps','extrinsic_reward']].values

# Separate the data into X and Y components
x1 = train_data[:, 0]
y1 = train_data[:, 1]

# Create a scatter plot
plt.scatter(x1, y1, color='red')

# Add title and labels to the plot
plt.title('Bestest_hydronic \n Train data with No transfer learning for December 14')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()

#------------------------------------
metrics_exp14 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V14/Summary_Result_Models/complete_metrics_exp14.csv')

train_data= metrics_exp14.loc[metrics_exp14['Type']=='Train'][['time_steps','extrinsic_reward']].values

# Separate the data into X and Y components
x2 = train_data[:, 0]
y2 = train_data[:, 1]

# Create a scatter plot
plt.scatter(x2, y2, color='blue')

# Add title and labels to the plot
plt.title('Bestest_hydronic \n Train data with RL Agent Model based transfer learning for December 14')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()


#-----------------------------------
metrics_exp15 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V15/Summary_Result_Models/complete_metrics_exp15.csv')

train_data= metrics_exp15.loc[metrics_exp15['Type']=='Train'][['time_steps','extrinsic_reward']].values

# Separate the data into X and Y components
x3 = train_data[:, 0]
y3 = train_data[:, 1]

# Create a scatter plot
plt.scatter(x3, y3, color='green')

# Add title and labels to the plot
plt.title('Bestest_hydronic \n Train data with Environment model for December 14')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()

#-----------------------------------------------TEST DATASET Nov 13-------------------

metrics_exp13 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V13/Summary_Result_Models/complete_metrics_exp13.csv')

test_data= metrics_exp13.loc[(metrics_exp13['Type']=='Test') &(metrics_exp13['Date'] =='November 13')][['time_steps','extrinsic_reward']].values

x1 = test_data[:, 0]
y1 = test_data[:, 1]

# Create a scatter plot
plt.scatter(x1, y1, color='red')

# Add title and labels to the plot
plt.title('Bestest_hydronic \n Test data with No transfer learning for November 13')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()


#-------------------------------------

metrics_exp14 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V14/Summary_Result_Models/complete_metrics_exp14.csv')

test_data= metrics_exp14.loc[(metrics_exp14['Type']=='Test') &(metrics_exp14['Date'] =='November 13')][['time_steps','extrinsic_reward']].values

x2 = test_data[:, 0]
y2 = test_data[:, 1]

# Create a scatter plot
plt.scatter(x2, y2, color='blue')

# Add title and labels to the plot
plt.title('Bestest_hydronic \n Test data with RL Agent Model based transfer learning for November 13')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()

#-----------------------------------
metrics_exp15 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V15/Summary_Result_Models/complete_metrics_exp15.csv')

test_data= metrics_exp15.loc[(metrics_exp15['Type']=='Test') &(metrics_exp15['Date'] =='November 13')][['time_steps','extrinsic_reward']].values

x3 = test_data[:, 0]
y3 = test_data[:, 1]

# Create a scatter plot

plt.scatter(x3, y3, color='green')

# Add title and labels to the plot
plt.title('Bestest_hydronic \n Test data with Environment model based transfer learning for November 13')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()



