import torch as T
import torch.nn as nn

from models import RepresentationModel, LatentDynamics, RewardPredictor, ValuePredictor, PolicyModel
from replay_buffer import ReplayBuffer, Episode
from planner import PredictiveSampler, MPPISampler
from utils import symlog, symexp, make_frame_stacker, process_image
from env import make_env

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
import yaml

from tdmpc import TDMPC

def testing_run(config_filepath):

    # Load the config yaml file
    folder_path = os.path.dirname(config_filepath)
    with open(config_filepath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    IMAGE_OBSERVATIONS = config["image_observations"]
    FRAME_STACK = config.get("frame_stack", 1)
    SAMPLING_NOISE_STD = config["sampling_noise"]
    EVAL_EVERY = config["eval_policy_every"]

    # Create the dm_control environment (task / action_repeat set in the config).
    env = make_env(config)

    if IMAGE_OBSERVATIONS:
        state_dim = (3 * config["frame_stack"], 64, 64)  # (C, H, W)
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    config["action_dim"] = action_dim
    config["state_dim"] = state_dim
    # TODO - hardcoded - needs updating based on env
    config["episode_length"] = 500

    # Make a TDMPC object
    tdmpc = TDMPC(config)

    # Instantiate the replay buffer (prioritized experience replay by default)
    replay_buffer = ReplayBuffer(
        config
    )

    training_rewards, training_episode_length = [], []
    evaluation_rewards, evaluation_episode_length = [], []

    global_epoch = 0
    step = 0
    print("Training starts...")
    episode_idx, start_time = 0, time.time()
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

        # Save training episode metrics
        episode_idx += 1


        # Evaluate the models periodically
        # if round_idx % EVAL_EVERY == 0:
        #     # Evaluate the model
        #     episode = collect_play_data(env, tdmpc, replay_buffer,
        #                     config=config, num_episodes=5, noise_std=0.0, train = False, fixed_episode_length=EPISODE_LENGTH)
        #     print(f"Round {round_idx}: Average reward over 5 evaluation episodes: {eval_score:.2f}")
        #     evaluation_rewards.append(eval_score)
        #     evaluation_episode_length.append(avg_episode_length)

        #     if os.path.exists(f"{folder_path}/model_checkpoints") == False:
        #         os.makedirs(f"{folder_path}/model_checkpoints")
        #     T.save({
        #         "representation_model": representation_model.state_dict(),
        #         "latent_dynamics": latent_dynamics.state_dict(),
        #         "reward_predictor": reward_predictor.state_dict(),
        #         "value_predictor": value_predictor.state_dict(),
        #         "policy_model": policy_model.state_dict(),
        #     }, f"{folder_path}/model_checkpoints/checkpoint_round{round_idx}.pt")

        #     global_epoch += 1

    # Save training data
    np.savez(
        f"{folder_path}/training_stats.npz",
        training_episode_rewards=np.array(training_rewards),
        training_episode_lengths=np.array(training_episode_length),
        evaluation_episode_rewards=np.array(evaluation_rewards),
        evaluation_episode_lengths=np.array(evaluation_episode_length)
    )
    env.close()

if __name__ == "__main__":

    #Just a single testing run
    testing_run("configs/default/default.yaml")

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