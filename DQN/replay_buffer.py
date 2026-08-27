from collections import deque
from typing import Deque, List
import random
import numpy as np
from numpy.typing import NDArray
from collections.abc import Iterable

class ReplayBuffer:
    """
    Fixed-size buffer to store and sample transitions for experience replay.
    """
    def __init__(self, capacity: int):
        """ Initialize the replay buffer with some maximum capacity. """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.
        Returns:
            states                 : (B, *state_shape) float32
            actions                : (B, 1) long
            rewards                : (B,) float32
            next_states            : (B, *state_shape) float32
            dones                  : (B,) float32  (1.0 if done else 0.0)   
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)