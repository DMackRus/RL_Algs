import os

# Headless GPU rendering. GLFW (dm_control's default) needs an X/GLX context,
# which fails inside the container. Must be set before dm_control is imported.
os.environ.setdefault("MUJOCO_GL", "egl")

import yaml
import cv2
import numpy as np
from env import make_env
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


if __name__ == "__main__":

    test_folder = "configs/reacher_hard"
    yaml_name = "config.yaml"
    model_name = "checkpoint_step30000.pt"
    video_folder_name = "eval_videos"

    config_filepath = test_folder + "/" + yaml_name

    with open(config_filepath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    env = make_env(config)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    config["action_dim"] = action_dim
    config["state_dim"] = state_dim
    config["episode_length"] = env.ep_len

    agent = LeWM(config)
    agent.load(test_folder + "/model_checkpoints/" + model_name)

    num_eval_episodes = config["eval_episodes"]

    VIDEO_DIR = test_folder + "/" + video_folder_name
    os.makedirs(VIDEO_DIR, exist_ok=True)

    episode_returns = []
    goal_best_dists, goal_final_dists, goal_successes = [], [], []
    goal_thr = float(config.get("goal_success_dist", 0.1))

    for episode_idx in range(num_eval_episodes):

        obs = env.reset()
        episode = Episode(config, obs)

        # LeWM is reward-free: build a task-specific goal state and score the
        # rollout by how close we get to it (see goals.py).
        goal = goal_observation(env)
        agent.set_goal(goal)

        frames = [env.render(mode="rgb_array")]
        step = 0
        best_dist = float("inf")
        dist = float("inf")
        while not episode.done:
            step += 1
            action = agent.plan(obs, eval_mode=True, step=step, t0=episode.first)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode += (obs, action, reward, done)
            frames.append(env.render(mode="rgb_array"))
            dist = float(np.linalg.norm(np.asarray(obs, np.float32) - goal))
            best_dist = min(best_dist, dist)
        episode_returns.append(episode.cumulative_reward)

        goal_best_dists.append(best_dist)
        goal_final_dists.append(dist)
        goal_successes.append(float(best_dist < goal_thr))
        print(f"Episode {episode_idx}: goal dist best {best_dist:.3f}, "
              f"final {dist:.3f}, success {best_dist < goal_thr} "
              f"(env reward {episode.cumulative_reward:.2f})")
        save_video(frames, os.path.join(VIDEO_DIR, f"episode_{episode_idx}.mp4"))

    print(f"Over {num_eval_episodes} episodes: "
          f"best goal dist {np.mean(goal_best_dists):.3f}, "
          f"final goal dist {np.mean(goal_final_dists):.3f}, "
          f"success rate {np.mean(goal_successes):.2f}")
    print(f"Average env return over {num_eval_episodes} episodes: {np.mean(episode_returns)}")
