import gymnasium as gym
from PIL import Image
import os

# Initialise the environment
env = gym.make("InvertedPendulum-v5", render_mode="human")
env = gym.make("Walker2d-v5", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset()

for _ in range(1000):
    # Randomly sample actions from the action space for testing purposes
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    next_obs, reward, terminated, truncated, info = env.step(action)
    print(f"next_obs shape: {next_obs.shape}, reward: {reward}, terminated: {terminated}, truncated: {truncated}")


    observation = next_obs

    # If the episode has ended then we can reset to start a new episode
    # if terminated:
    #     observation, info = env.reset()

env.close()

