import os

import torch as T
import yaml
import cv2
import numpy as np
from env import make_env
from tdmpc import TDMPC
from replay_buffer import Episode

def save_video(frames, path, fps=15):
    """Write a list of RGB uint8 (H, W, 3) frames to an mp4."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        print(f"could not open video writer for {path}")
        return
    for f in frames:
        writer.write(cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"saved {path} ({len(frames)} frames @ {fps} fps)")


def main():
    print("Evaluation begins")

if __name__ == "__main__":

    test_folder = "configs/fixed_versus_adaptive/fixed_dt"
    yaml_name = "config.yaml"
    model_name = "checkpoint_step30000.pt"
    video_folder_name = "eval_videos"

    config_filepath = test_folder + "/" + yaml_name

    with open(config_filepath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    env = make_env(config)

    IMAGE_OBSERVATIONS = config["image_observations"]
    if IMAGE_OBSERVATIONS:
        state_dim = (3 * config["frame_stack"], 64, 64)  # (C, H, W)
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    config["action_dim"] = action_dim
    config["state_dim"] = state_dim

    #TODO - This should not be hardcoded
    config["episode_length"] = 500


    agent = TDMPC(config)
    model_path = test_folder + "/model_checkpoints/" + model_name
    agent.load(model_path)

    num_eval_episodes = config["eval_episodes"]

    VIDEO_DIR = test_folder + "/" + video_folder_name
    os.makedirs(VIDEO_DIR, exist_ok=True)

    episode_returns = []

    for episode_idx in range(num_eval_episodes):

        obs = env.reset()
        episode = Episode(config, obs)
        frames = [env.render(mode="rgb_array")]
        step = 0
        while not episode.done:
            step += 1
            action = agent.plan(obs, eval_mode = True, step=step, t0=episode.first)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode += (obs, action, reward, done)
            frames.append(env.render(mode="rgb_array"))
        episode_returns.append(episode.cumulative_reward)

        print(f"Episode {episode_idx} finished with total reward: {episode.cumulative_reward}")
        save_video(frames, os.path.join(VIDEO_DIR, f"episode_{episode_idx}.mp4"))

    print(f"Average return over {num_eval_episodes} episodes: {np.mean(episode_returns)}")