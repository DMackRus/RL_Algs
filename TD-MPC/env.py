import gymnasium as gym
import numpy as np

# Environment wrapper that accepts actions in range [-1, +1] and scales them to the environment's action space.
class envWrapper:
    def __init__(self, env):
        self.env = env
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        # Scale the action from [-1, +1] to the environment's action space
        scaled_action = self.scale_action(action)
        return self.env.step(scaled_action)

    def scale_action(self, action):
        # Scale the action from [-1, +1] to the environment's action space
        low = self.env.action_space.low
        high = self.env.action_space.high
        scaled_action = low + (action + 1.0) * 0.5 * (high - low)
        return np.clip(scaled_action, low, high)