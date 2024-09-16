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
        self.action_log_probs = []
        self.actions = []
        self.rewards = []
        self.new_states = []
        self.value_preds = []
    
    def remember(self, state, action, action_log_prob, reward, new_state, value_preds):
        self.states.append( torch.tensor(state) )
        self.actions.append(torch.tensor(action))
        self.action_log_probs.append( action_log_prob )
        self.rewards.append(torch.tensor(reward))
        self.new_states.append(torch.tensor(new_state))
        self.value_preds.append(value_preds)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.new_states = []
        self.value_preds = []
        
    def sample_memory(self, sample_size = 2000 ):  #was 1000
        '''
        if self.memory_size()< sample_size:
            indices  = np.random.permutation(self.memory_size())
        else:
            indices  = np.random.permutation(sample_size)
    
        '''
        
        return (
                torch.stack(self.states, dim = 0), 
                torch.stack(self.actions, dim = 0), 
                torch.stack(self.action_log_probs, dim = 0),
                torch.stack(self.rewards, dim = 0).unsqueeze(dim=1), 
                torch.stack(self.new_states, dim = 0),    
                torch.stack(self.value_preds, dim = 0) 
               )
        '''
        return ( 
            np.array(self.states)[indices], 
            np.array(self.actions)[indices], 
            np.array(self.action_log_probs)[indices],
            np.array(self.rewards)[indices],
            np.array(self.new_states)[indices],
            np.array(self.value_preds)[indices]
            )
        '''
    def memory_size(self):
         return len(self.states)

