# Constants
from collections import deque
import random
import numpy as np

# Global Variables
MAX_MEMORY = 100_000

class GeneralAgent():
    def __init__(self) -> None:
        self.memory = deque(maxlen=MAX_MEMORY)
        self.eps = None

    def remember(self, observation, action, reward, observation_, done):
        # pop left if max memory is reached
        self.memory.append((observation, action, reward, observation_, done))

    def train_long_memory(self):
        if len(self.memory) > self.bs:
            batch = random.sample(self.memory, self.bs) # list of BATCH_SIZE tuples
        else:
            batch = self.memory

        observations, actions, rewards, observations_, dones = zip(*batch)
        self.train_step(np.array(observations), np.array(actions), np.array(rewards), np.array(observations_), np.array(dones))

    def train_short_memory(self, observation, action, reward, observation_, done):            
        self.train_step(observation, action, reward, observation_, done)

    
