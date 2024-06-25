# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:13:34 2024

@author: gauthambekal93
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Result_consolidation_comparison')

metrics_exp4 = pd.read_csv(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V4/Summary_Result_Models/complete_metrics_exp4.csv')

train_data= metrics_exp4.loc[metrics_exp4['Type']=='Train'][['time_steps','extrinsic_reward']].values

# Separate the data into X and Y components
x = train_data[:, 0]
y = train_data[:, 1]

# Create a scatter plot
plt.scatter(x, y, color='blue', marker='o')

# Add title and labels to the plot
plt.title('Train data with No transfer learning')
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
plt.scatter(x, y, color='blue', marker='o')

# Add title and labels to the plot
plt.title('Train data with transfer learning')
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
plt.scatter(x, y, color='blue', marker='o')

# Add title and labels to the plot
plt.title('April Test data with No transfer learning')
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
plt.scatter(x, y, color='blue', marker='o')

# Add title and labels to the plot
plt.title('April Test data with transfer learning')
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
plt.scatter(x, y, color='blue', marker='o')

# Add title and labels to the plot
plt.title('January Test data with No transfer learning')
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
plt.scatter(x, y, color='blue', marker='o')

# Add title and labels to the plot
plt.title('January Test data with transfer learning')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()
