import torch as T
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
import yaml
from collections import defaultdict

from replay_buffer import ReplayBuffer, Episode
from utils import symlog, symexp, make_frame_stacker, process_image
from env import make_env
from tdmpc import TDMPC

class Logger:
    """Collects per-iteration training / evaluation rewards and update metrics.

    One "iteration" is one pass of the outer loop in ``testing_run``: collect a
    single episode, then run a batch of model updates. Evaluation is only run
    every few iterations (``eval_policy_every``); on the iterations in between
    we repeat the most recent eval score so the eval arrays line up 1:1 with the
    training arrays and can be plotted against a shared x-axis without any
    bookkeeping.

    Outputs (both written to ``work_dir``):
      - ``log.txt``            human-readable, one row per iteration
      - ``training_stats.npz`` all history as numpy arrays for offline plotting
    """

    def __init__(self, work_dir, cfg):
        self.work_dir = work_dir
        self.cfg = cfg
        os.makedirs(work_dir, exist_ok=True)
        self.log_file = os.path.join(work_dir, "log.txt")

        # Per-iteration history (all lists stay the same length).
        self.iterations = []
        self.steps = []
        self.elapsed = []
        self.train_rewards = []
        self.train_lengths = []
        self.eval_rewards = []
        self.eval_lengths = []
        self.metrics = defaultdict(list)   # update-metric name -> per-iteration value

        # Most recent evaluation result, carried forward between eval runs.
        self._last_eval_reward = float("nan")
        self._last_eval_length = float("nan")

        self.start_time = time.time()

    def set_eval(self, reward, length):
        """Record a fresh evaluation result. Call on eval iterations only; the
        value is then repeated on every following iteration until the next call."""
        self._last_eval_reward = float(reward)
        self._last_eval_length = float(length)

    def log_iteration(self, iteration, step, episode, train_metrics):
        """Store everything for one outer training-loop iteration."""
        n = len(self.iterations)

        self.iterations.append(int(iteration))
        self.steps.append(int(step))
        self.elapsed.append(time.time() - self.start_time)
        self.train_rewards.append(float(episode.cumulative_reward))
        self.train_lengths.append(int(len(episode)))

        # Repeat the latest eval score so eval arrays align with training arrays.
        self.eval_rewards.append(self._last_eval_reward)
        self.eval_lengths.append(self._last_eval_length)

        # Update metrics. ``train_metrics`` is empty during the seed phase, and a
        # metric first seen mid-run is back-filled with NaN for earlier rows.
        for key in set(self.metrics) | set(train_metrics):
            col = self.metrics[key]
            col.extend([float("nan")] * (n - len(col)))
            col.append(float(train_metrics.get(key, float("nan"))))

        self._write_log()

    def _write_log(self):
        metric_keys = sorted(self.metrics)
        cols = ["iter", "step", "elapsed_s", "train_reward", "train_length",
                "eval_reward", "eval_length"] + metric_keys
        lines = ["\t".join(cols)]
        for i in range(len(self.iterations)):
            row = [self.iterations[i], self.steps[i], f"{self.elapsed[i]:.1f}",
                   f"{self.train_rewards[i]:.2f}", self.train_lengths[i],
                   f"{self.eval_rewards[i]:.2f}", f"{self.eval_lengths[i]:.1f}"]
            row += [f"{self.metrics[k][i]:.4f}" for k in metric_keys]
            lines.append("\t".join(str(x) for x in row))
        with open(self.log_file, "w") as f:
            f.write("\n".join(lines) + "\n")

    def save(self, filename="training_stats.npz"):
        """Dump all collected history to a ``.npz`` for offline plotting."""
        arrays = dict(
            iterations=np.array(self.iterations),
            steps=np.array(self.steps),
            elapsed_seconds=np.array(self.elapsed),
            training_episode_rewards=np.array(self.train_rewards),
            training_episode_lengths=np.array(self.train_lengths),
            evaluation_episode_rewards=np.array(self.eval_rewards),
            evaluation_episode_lengths=np.array(self.eval_lengths),
        )
        for key, values in self.metrics.items():
            arrays[f"metric_{key}"] = np.array(values)
        np.savez(os.path.join(self.work_dir, filename), **arrays)


@T.no_grad()
def evaluate(env, agent, num_episodes, step):
    """Run ``num_episodes`` noise-free planning rollouts and return the mean
    cumulative reward and mean episode length."""
    rewards, lengths = [], []
    for _ in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0.0, 0
        while not done:
            action = agent.plan(obs, eval_mode=True, step=step, t0=(t == 0))
            obs, reward, done, _ = env.step(action.cpu().numpy())
            ep_reward += reward
            t += 1
        rewards.append(ep_reward)
        lengths.append(t)
    return float(np.mean(rewards)), float(np.mean(lengths))

def testing_run(config_filepath):

    # Load the config yaml file
    folder_path = os.path.dirname(config_filepath)
    with open(config_filepath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    IMAGE_OBSERVATIONS = config["image_observations"]
    FRAME_STACK = config.get("frame_stack", 1)
    SAMPLING_NOISE_STD = config["sampling_noise"]
    EVAL_EVERY = config["eval_policy_every"]          # eval once every N iterations
    EVAL_EPISODES = config.get("eval_episodes", 5)    # rollouts averaged per eval

    # Create the environment (benchmark / task / action_repeat set in the config).
    env = make_env(config)

    if IMAGE_OBSERVATIONS:
        state_dim = (3 * config["frame_stack"], 64, 64)  # (C, H, W)
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    config["action_dim"] = action_dim
    config["state_dim"] = state_dim
    config["episode_length"] = env.ep_len

    print(f"Env control timestep: {env.control_timestep()}")
    print(f"Env episode length: {config['episode_length']}")

    # Make a TDMPC object
    tdmpc = TDMPC(config)

    # Instantiate the replay buffer (prioritized experience replay by default)
    replay_buffer = ReplayBuffer(
        config
    )

    logger = Logger(folder_path, config)

    print("Training starts...")
    episode_idx = 0
    for step in range(0, config["train_steps"]+config["episode_length"], config["episode_length"]):

        obs = env.reset()
        episode = Episode(config, obs)
        while not episode.done:
            action = tdmpc.plan(obs, step=step, t0=episode.first)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode += (obs, action, reward, done)
        assert len(episode) == config["episode_length"]
        replay_buffer += episode

        print(f"Step {step}: Collected episode with reward {episode.cumulative_reward:.2f} and length {len(episode)}")

        # Update the models
        train_metrics = {}
        if step >= config["seed_steps"]:
            num_updates = config["seed_steps"] if step == config["seed_steps"] else config["episode_length"]
            for i in range(num_updates):
                train_metrics.update(tdmpc.update(replay_buffer, step+i))

        # Evaluate the current policy periodically (noise-free planning). The
        # score is carried forward by the logger onto the intervening iterations.
        if step >= config["seed_steps"] and episode_idx % EVAL_EVERY == 0:
            eval_reward, eval_length = evaluate(env, tdmpc, EVAL_EPISODES, step)
            print(f"Step {step}: eval reward over {EVAL_EPISODES} episodes: {eval_reward:.2f}")
            logger.set_eval(eval_reward, eval_length)

            os.makedirs(f"{folder_path}/model_checkpoints", exist_ok=True)
            tdmpc.save(f"{folder_path}/model_checkpoints/checkpoint_step{step}.pt")

        # Record this iteration: training reward/length, carried-forward eval
        # reward/length, and the latest update metrics (NaN during seed phase).
        logger.log_iteration(episode_idx, step, episode, train_metrics)
        logger.save()

        episode_idx += 1

    logger.save()
    # env.close()

if __name__ == "__main__":

    # Testing adaptive timestep size
    testing_run("configs/fixed_versus_adaptive/adaptive_dt/config.yaml")
    # Testing non adaptive timestep size
    testing_run("configs/fixed_versus_adaptive/fixed_dt/config.yaml")

    #Just a single testing run
    # testing_run("configs/default/default.yaml")

    # # test_name = "testing_horizons"
    # test_name = "testing_time_lambdas"

    # # Load the testing config yaml file
    # testing_config_path = f"configs/{test_name}.yaml"
    # with open(testing_config_path, "r") as f:
    #     testing_config = yaml.load(f, Loader=yaml.FullLoader)

    # config_filepath = f"configs/{testing_config['default_config']}/{testing_config['default_config']}.yaml"

    # # Load the default settings from theconfig file
    # with open(config_filepath, "r") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)

    # # Loop over the second keys in testing config file (e.g., different horizons)
    # for key, value in testing_config.items():
    #     if key == "default_config":
    #         continue  # Skip the default config key

    #     # Second key will always be a list, so loop over the list
    #     for item in value:
    #         print(f"Testing with {key}: {item}")

    #         # Update the config with the current testing parameter
    #         config[f"{key}"] = item

    #         #Save the update config to a new folder and file for data storage
    #         new_folder_path = f"configs/{test_name}/{key}_{item}"

    #         if not os.path.exists(new_folder_path):
    #             os.makedirs(new_folder_path)

    #         # Save the updated config to file
    #         temp_config_path = f"{new_folder_path}/{key}_{item}.yaml"
    #         with open(temp_config_path, "w") as f:
    #             yaml.dump(config, f)

    #         # Run the test with the new config file
    #         testing_run(temp_config_path)