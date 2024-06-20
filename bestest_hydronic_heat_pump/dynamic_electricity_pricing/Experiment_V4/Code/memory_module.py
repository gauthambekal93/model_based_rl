import numpy as np
import torch
import random

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []
    #    self.log_probs = []
    
    def remember(self, state, action, reward, new_state):
        self.actions.append(action)
        self.rewards.append(reward)
        self.states.append(state)
        self.new_states.append(new_state)
       # self.log_probs.append(log_p)


    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []
        #self.log_probs = []
        
    def sample_memory(self, sample_size = 2000 ):  #was 1000
        
        if self.memory_size()< sample_size:
            indices  = np.random.permutation(self.memory_size())
        else:
            indices  = np.random.permutation(sample_size)
        
        #indices  = np.random.permutation(self.memory_size())
            
        return ( 
            np.array(self.states)[indices], 
            np.array(self.actions)[indices],  
            np.array(self.rewards)[indices],
            np.array(self.new_states)[indices], 
           # np.array(self.log_probs)[indices]
            )

    def memory_size(self):
         return len(self.states)

