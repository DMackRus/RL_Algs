import gymnasium as gym
from env import envWrapper
import numpy as np

def main():
    print("Starting test script")

    env = gym.make("Walker2d-v5", render_mode="human")
    wrapped_env = envWrapper(env)

    obs = wrapped_env.reset()
    done = False
    while not done:
        action = wrapped_env.action_space.sample()  # Sample a random action in the range [-1, +1]
        print(f"Sampled action: {action}")
        obs, reward, done, info = wrapped_env.step(action)
        print(f"Observation: {obs}, Reward: {reward}, Done: {done}, Info: {info}")

    env.close()


if __name__ == "__main__":
    main()