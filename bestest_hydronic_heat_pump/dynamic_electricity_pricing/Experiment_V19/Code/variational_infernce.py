# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:27:02 2024

@author: gauthambekal93
"""


import torch
import numpy as np
import torch.nn as nn
import torch.distributions as dist
import torch.optim as optim
import random

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  

class Bayesian_Learning(nn.Module):
    
    def __init__(self, input_dim, output_dim, latent_dim, hidden_dim):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder_layer1 = nn.Linear(  input_dim , hidden_dim)
        
        self.encoder_layer2 =nn.Linear(hidden_dim, latent_dim * 2 ) #latent_dim * 2  is because we need mean and standard deviation for every dimension of latent variable
        
        self.decoder_layer1 = nn.Linear( latent_dim, hidden_dim)
        
        self.decoder_layer2 =nn.Linear(hidden_dim, output_dim * 2 ) #input_dim * 2  is because we need mean and standard deviation for every dimension of input variable
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
        self.normal_dist = dist.Normal(0, 1)
        
        self.no_samples = 10000 
        
    def forward_encoder(self, x):    
        logits = self.leaky_relu ( self.encoder_layer1(x) )
        logits = self.leaky_relu( self.encoder_layer2(logits) )
    
        mu, log_std = torch.chunk(logits, 2, dim=1)
        std = torch.exp(log_std)
        
        return mu, std
    
    def forward_decoder(self, z):    
        logits = self.leaky_relu( self.decoder_layer1(z) )
        logits = self.leaky_relu ( self.decoder_layer2(logits) )
        
        mu, log_std = torch.chunk(logits, 2, dim=1)
        std = torch.exp(log_std)
        
        return mu, std

    def sample_latent_variable(self, mu, std, batch_size):
        
        samples = self.normal_dist.sample(( batch_size, self.latent_dim ))  #sample from unit normal distribution
        
        z =  ( mu ) + (samples * std)      #reparameterization trick
        
        return z
    
        
    def kl_divergence_gaussians(self, mu1, std1, mu2, std2):
        # Variance is the square of the standard deviation
        var1 = std1 ** 2
        var2 = std2 ** 2
        
        # KL Divergence formula for two Gaussians
        kl_div = torch.log(std2 / std1) + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5
        return kl_div

    def loss_calculation(self, x , y, kl_weight, batch_size):
        
        mu_encoder, std_encoder =  self.forward_encoder(x)
        
        z = self.sample_latent_variable(mu_encoder, std_encoder, batch_size)
        
        mu_decoder, std_decoder  = self.forward_decoder( z )
        
        output_dist = dist.Normal(mu_decoder, std_decoder )  # std_dev must be positive, so we take the absolute value
    
        # Compute the negative log likelihood for all input data points
        log_likelihood =  output_dist.log_prob(y).mean()  # Sum of log probabilities for all data points
        
        prior_mu , prior_std = torch.zeros((batch_size, self.latent_dim )), torch.ones((batch_size, self.latent_dim ))
        
        kl_div = self.kl_divergence_gaussians( mu_encoder, std_encoder, prior_mu , prior_std ).mean() 
        
        elbo =  log_likelihood - kl_weight  * kl_div
        
        loss = - 1.0 * elbo
        
        #print("log_likelihood ",log_likelihood, "Kl_div ", kl_div, "loss ", loss)
        
        return loss
    
    def obtain_uncertanity(self, x, y, kl_weight, batch_size):
        
        mu_encoder, std_encoder =  self.forward_encoder(x)
        
        z_dist = torch.stack( [ self.sample_latent_variable(mu_encoder, std_encoder, batch_size) for _ in range(self.no_samples )], dim = 0)
        
        tmp =[]
        
        for i in range( z_dist.shape[0]):
           
            mu_decoder, _  = self.forward_decoder(  z_dist[i]  )
            
            tmp.append(mu_decoder.reshape(-1))
            
        y_dist = torch.stack(tmp, dim = 0 ).transpose(0, 1)
        
        uncertanity = y_dist.std(dim=1) 
        
        return y_dist, uncertanity
        


import matplotlib.pyplot as plt

# Number of data points
num_points = 10000 #10000 #10000  

# Generate x values from 0 to 2*pi
x_values = torch.linspace(0, 2 * torch.pi, num_points)

# True amplitude and frequency for the sine wave
true_amplitude = 1
true_frequency = 1

# Generate sine wave without noise
y_values = true_amplitude * torch.sin(true_frequency * x_values)

# Add Gaussian noise to the sine wave
noise_std = 0.1  # Standard deviation of the noise
noise = torch.normal(0, noise_std, size=(num_points,))

# Create noisy y values
y_values_noisy = y_values + noise

# Plot the sine wave with noise
plt.figure(figsize=(10, 6))
plt.plot(x_values.numpy(), y_values.numpy(), label="True Sine Wave")
plt.scatter(x_values.numpy(), y_values_noisy.numpy(), color='red', s=1, label="Noisy Data", alpha=0.3)
plt.legend()
plt.title("Sine Wave with 10,000 Noisy Data Points (PyTorch)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

x_values = x_values.reshape(-1,1)
y_values_noisy = y_values_noisy.reshape(-1,1)

x_values = torch.cat( [ x_values , torch.zeros(x_values.shape[0], 1 ) ] , dim = 1)  

input_dim, output_dim = x_values.shape[1],  y_values_noisy.shape[1]

latent_dim, hidden_dim  = 5, 5

epochs, batch_size = 10, 100  #was 2000, 100

model = Bayesian_Learning(input_dim, output_dim, latent_dim , hidden_dim)

optimizer = optim.Adam( model.parameters(), lr= 0.0001 ) #lr= 0.001

indices, split_idx = torch.randperm(x_values.shape[0]),  int( x_values.shape[0] * 0.80 )

train_x, train_y =  x_values[  indices[ :split_idx] ], y_values_noisy[  indices[ :split_idx] ]

test_x, test_y = x_values[  indices[split_idx: ] ], y_values_noisy[  indices[split_idx: ] ]

updated_test_x = torch.stack( [  test_x[:,0], torch.ones( test_x.shape[0] ) ] , dim = 1 )

kl_weight = 0.01

def save_model():
    torch.save(model.state_dict(), r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V19/Models/bayesian_model.pth')

def load_model():
    model.load_state_dict(torch.load( r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V19/Models/bayesian_model.pth' ))
    
def train_model():    
    for epoch in range(1, epochs):
        
        print("Epoch ", epoch)
        total_loss = []
        
        #kl_weight = min(1.0, epoch / epochs)
        
        for batch in range( int( len(train_x) / batch_size) ):  #was x_values which is incorrect
            
            index = torch.randint(low = 0, high = len(train_x) , size=(batch_size,))
            
            batch_x, batch_y = train_x[index], train_y[index]
            
            loss = model.loss_calculation(batch_x, batch_y, kl_weight, batch_size )
            
            total_loss.append( loss.item() )
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
        
        print("Epoch ", epoch, "Loss ", np.mean( total_loss ) )
        
#load_model()
train_model()

#save_model()
 

    
loss_test = model.loss_calculation(test_x, test_y, kl_weight, len(test_x ))

y_dist_test, uncertanity_test = model.obtain_uncertanity( test_x, test_y, kl_weight, len(test_x ) )

print("original features \nLoss: {0}  uncertanity: {1}  ".format( loss_test, uncertanity_test.mean() ) )


#loss_updated_test = model.loss_calculation(updated_test_x, test_y, kl_weight, len(updated_test_x) )


#y_dist_updated_test, uncertanity_updated_test = model.obtain_uncertanity( updated_test_x, test_y, kl_weight, len(updated_test_x ) )

#print( "Updated features \nLoss: {0}  uncertanity: {1}".format (loss_updated_test, uncertanity_updated_test.mean() )  )




