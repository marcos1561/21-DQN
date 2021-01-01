import torch
import random

class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0


    def save_memory(self, current_state, action, reward, new_state, done):
        current_state_t = torch.Tensor(current_state)
        action_t = torch.Tensor([action])
        reward_t = torch.Tensor([reward])
        new_state_t = torch.Tensor(new_state)
        done_t = torch.Tensor([done]) 

        transition_state = [current_state_t, action_t, reward_t, new_state_t, done_t]
        
        if len(self.memory) < self.capacity:
            self.memory.append(transition_state)
        else:
            self.memory[self.position] = transition_state
            self.position = (self.position+1) % self.capacity

    
    def sample(self, batchsize=10):
        minibatch = random.sample(self.memory, batchsize)

        current_state_batch = torch.stack(tuple(trans[0] for trans in minibatch))
        action_batch = torch.stack(tuple(trans[1] for trans in minibatch))
        reward_bach = torch.stack(tuple(trans[2] for trans in minibatch))
        new_state_bach = torch.stack(tuple(trans[3] for trans in minibatch))
        done_bach = torch.stack(tuple(trans[4] for trans in minibatch))

        return current_state_batch, action_batch, reward_bach, new_state_bach, done_bach
