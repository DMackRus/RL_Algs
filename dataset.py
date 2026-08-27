import gymnasium as gym
from PIL import Image
import os

# Initialise the environment
env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
# env = gym.make("Acrobot-v1", render_mode="rgb_array")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)

episodes = []
episode = []

for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    rgb_image = env.render()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    next_obs, reward, terminated, truncated, info = env.step(action)
    print(f"next_obs shape: {next_obs.shape}, reward: {reward}, terminated: {terminated}, truncated: {truncated}")


    episode.append({
        "obs": rgb_image,
        "action": action,
        "reward": reward,
        "done": terminated or truncated,
    })

    observation = next_obs

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        episodes.append(episode)
        episode = []
        observation, info = env.reset()

episodes.append(episode)
episode = []


env.close()

# Save the episodes to a file
os.makedirs("data", exist_ok=True)
with open("data/episodes.pkl", "wb") as f:
    import pickle
    pickle.dump(episodes, f)