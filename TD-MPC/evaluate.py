import os

import torch as T
import yaml
import cv2
import numpy as np
from env import make_env
from tdmpc import TDMPC
from tdmpc_adaptive import TDMPCAdaptive
from lewm import LeWM
from goals import goal_observation
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

    # test_folder = "configs/fixed_versus_adaptive/fixed_dt"
    test_folder = "configs/lewm"      # LeWorldModel (goal-conditioned, reward-free)
    # test_folder = "configs/default"
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

    agent = None
    if config["training_algorithm"] == "tdmpc":
        agent = TDMPC(config)
    elif config["training_algorithm"] == "tdmpc_adaptive":
        agent = TDMPCAdaptive(config)
    elif config["training_algorithm"] == "lewm":
        agent = LeWM(config)
    else:
        raise ValueError(f"Unknown training algorithm: {config['training_algorithm']}")
    IS_LEWM = config["training_algorithm"] == "lewm"
    model_path = test_folder + "/model_checkpoints/" + model_name
    agent.load(model_path)

    num_eval_episodes = config["eval_episodes"]

    VIDEO_DIR = test_folder + "/" + video_folder_name
    os.makedirs(VIDEO_DIR, exist_ok=True)

    episode_returns = []
    goal_best_dists, goal_final_dists, goal_successes = [], [], []
    goal_thr = float(config.get("goal_success_dist", 0.1))

    for episode_idx in range(num_eval_episodes):

        obs = env.reset()
        episode = Episode(config, obs)

        # LeWM is reward-free: sample a task-specific goal state and score the
        # rollout by how close we get to it (see goals.py).
        goal = None
        if IS_LEWM:
            goal = goal_observation(env)
            agent.set_goal(goal)

        frames = [env.render(mode="rgb_array")]
        step = 0
        best_dist = float("inf")
        dist = float("inf")
        while not episode.done:
            step += 1
            action = agent.plan(obs, eval_mode = True, step=step, t0=episode.first)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode += (obs, action, reward, done)
            frames.append(env.render(mode="rgb_array"))
            if IS_LEWM:
                dist = float(np.linalg.norm(np.asarray(obs, np.float32) - goal))
                best_dist = min(best_dist, dist)
        episode_returns.append(episode.cumulative_reward)

        if IS_LEWM:
            goal_best_dists.append(best_dist)
            goal_final_dists.append(dist)
            goal_successes.append(float(best_dist < goal_thr))
            print(f"Episode {episode_idx}: goal dist best {best_dist:.3f}, "
                  f"final {dist:.3f}, success {best_dist < goal_thr} "
                  f"(env reward {episode.cumulative_reward:.2f})")
        else:
            print(f"Episode {episode_idx} finished with total reward: {episode.cumulative_reward}")
        save_video(frames, os.path.join(VIDEO_DIR, f"episode_{episode_idx}.mp4"))

    if IS_LEWM:
        print(f"Over {num_eval_episodes} episodes: "
              f"best goal dist {np.mean(goal_best_dists):.3f}, "
              f"final goal dist {np.mean(goal_final_dists):.3f}, "
              f"success rate {np.mean(goal_successes):.2f}")
    print(f"Average return over {num_eval_episodes} episodes: {np.mean(episode_returns)}")