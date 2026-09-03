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

DEVICE = T.device(
    "cuda" if T.cuda.is_available() else "cpu"
)
print(f"Using device: {DEVICE}")


def collect_play_data(env, tdmpc, replay_buffer,
                       config=None, noise_std=0.3, train = True, fixed_episode_length=None):
    """
    Roll out `num_episodes` episodes and (if train) push transitions to the buffer.

    random_actions:
        Ignore the planner and sample actions uniformly from the env action space.
        Used to seed the buffer with diverse, full-amplitude dynamics data before
        any model has been trained.

    Returns (mean_episode_reward, mean_episode_length).
    """

    if config is None:
        print("No config provided, using default values for horizon and sampling noise.")
        return [], []

    obs = env.reset()
    episode = Episode(config, obs)

    reward = 0
    episode_length = 0
    total_reward = 0
    done = False
    t0 = True
    time_step = 0

    with T.no_grad():

        # NOTE: reuse the reset above that seeded `episode`; a second env.reset()
        # here would desync episode.obs[0] from the rollout trajectory.
        state = obs

        if config["image_observations"]:
            # state = process_image(env.render())
            stack_reset, stack_push = make_frame_stacker(config["frame_stack"])
            state = stack_reset(process_image(env.render()))
        
        while not done:

            # Handle if the state is a numpy array or a torch tensor
            if isinstance(state, np.ndarray):
                state_tensor = T.from_numpy(state).unsqueeze(0).float().to(DEVICE)
            else:
                state_tensor = state.unsqueeze(0).float().to(DEVICE)

            action = tdmpc.plan(state_tensor, eval_mode=False, step=time_step, t0=False)
            t0 = False

            # Add noise to the action for exploration
            # if train:
            #     noise = np.random.normal(0, noise_std, size=action.shape)
            #     action = np.clip(action + noise, -1, 1)

            next_state, reward, done, _ = env.step(action.squeeze(0).cpu().numpy())
            # print(f"Predicted reward: {predicted['reward'].item():.4f}, Actual reward: {reward:.4f}")
            episode_length += 1

            done = terminated or truncated
            if fixed_episode_length is not None and episode_length >= fixed_episode_length:
                done = True

            total_reward += reward
            if config["image_observations"]:
                next_state = stack_push(process_image(env.render()))
            episode += (next_state, action, reward, done)
            state = next_state
            time_step += 1

    return episode

def get_exploration_std(round_idx, num_rounds, std_start=1.0, std_end=0.05, decay_fraction=0.5):
    """
    Linearly anneal std from std_start to std_end over the first
    `decay_fraction` of training, then hold at std_end.
    """
    decay_rounds = num_rounds * decay_fraction
    progress = min(round_idx / decay_rounds, 1.0)
    return std_start + progress * (std_end - std_start)

def testing_run(config_filepath):

    # Load the config yaml file
    folder_path = os.path.dirname(config_filepath)
    with open(config_filepath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    latent_dim = config["latent_dim"]
    hidden_dim = config["hidden_dim"]
    INITIAL_NUM_EPISODES = config.get("seed_episodes", 10)
    EPISODE_LENGTH = config["max_episode_length"]
    IMAGE_OBSERVATIONS = config["image_observations"]
    FRAME_STACK = config.get("frame_stack", 1)
    SAMPLING_NOISE_STD = config["sampling_noise"]
    HORIZON = config["horizon"]
    NUM_TRAINING_ROUNDS = config["training_rounds"]
    STEPS_PER_ROUND = config["training_steps_per_round"]
    NEW_EPISODES_PER_ROUND = config["num_episodes_per_round"]
    EVAL_EVERY = config["eval_policy_every"]
    rho = config["gamma"]
    time_lambda = config["time_lambda"]

    # Create the dm_control environment (task / action_repeat set in the config).
    print(config["seed"])
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

        # Collect new episode data
        # noise_std = get_exploration_std(round_idx, NUM_TRAINING_ROUNDS, std_start=0.2, std_end=0.05, decay_fraction=0.5)
        # episode = collect_play_data(env, tdmpc, replay_buffer, config=config,
        #                             noise_std=noise_std, train = True, fixed_episode_length=EPISODE_LENGTH)
        # assert len(episode) == config["episode_length"]
        # replay_buffer += episode
        # step += config["episode_length"]
        # Collect trajectory

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

                # print(f"Round {round_idx}, Step {step+i}: Total loss: {total_loss:.4f}, Reward loss: {reward_loss:.4f}, Value loss: {value_loss:.4f}, Consistency loss: {consistency_loss:.4f}, Actor loss: {actor_loss:.4f}, Mean z norm: {np.mean(z_norms[1:]):.4f}, Mean Q: {mean_q:.4f}, Mean target: {mean_target:.4f}")


        # Log training episode
        # train_metrics["total_loss"] = total_loss
        # train_metrics["reward_loss"] = reward_loss
        # train_metrics["value_loss"] = value_loss
        # train_metrics["consistency_loss"] = consistency_loss
        # train_metrics["actor_loss"] = actor_loss if actor_loss is not None else np.nan
        # train_metrics["z_std"] = np.mean(z_norms[1:]) if len(z_norms) > 1 else np.nan
        # train_metrics["q"] = mean_q
        # train_metrics["target"] = mean_target

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

def update_pi(policy_model, policy_optimizer, value_predictor_1, value_predictor_2, time_lambda, zs, z_masks=None):
    """
    Update policy using a sequence of latent states.

    z_masks: optional list (same length as zs) of (B,) 0/1 tensors -- latents
    that come from rolling the dynamics past an episode end are masked out.
    """

    actor_loss = 0
    policy_optimizer.zero_grad(set_to_none=True)

    # Turn off gradients for value predictors during policy update
    for param in value_predictor_1.parameters():
        param.requires_grad = False
    for param in value_predictor_2.parameters():
        param.requires_grad = False

    for t, z in enumerate(zs):

        action = policy_model(z)
        # noise = (T.randn_like(action) * 0.2).clamp(-0.5, 0.5)
        # action = (action + noise).clamp(-1.0, 1.0)  # match your action bounds

        q = T.min(value_predictor_1(z, action), value_predictor_2(z, action)).squeeze(-1)

        if z_masks is not None:
            m = z_masks[t]
            actor_loss += -(time_lambda ** t) * (m * q).sum() / m.sum().clamp(min=1.0)
        else:
            actor_loss += -(time_lambda ** t) * q.mean()

    actor_loss.backward()
    T.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=10.0)
    policy_optimizer.step()

    # Turn gradients back on for value predictors
    for param in value_predictor_1.parameters():
        param.requires_grad = True
    for param in value_predictor_2.parameters():
        param.requires_grad = True

    return actor_loss.item()


# def update(replay_buffer, representation_model, offline_representation_model, latent_dynamics,
#            reward_predictor, value_predictor, value_predictor2, policy_model,
#            offline_value_predictor, offline_value_predictor2,
#            optimizer, policy_optimizer, epoch, config=None,
#            policy_delay=2, target_noise_std=0.2, target_noise_clip=0.5):
#     """
#     Single update step for the latent dynamics model, reward predictor, value predictor, and policy model.

#     Returns:
#         total_loss: Total loss for thje update step
#         reward_loss: Loss for the reward predictor
#         value_loss: Loss for the value predictor
#         consistency_loss: Loss for the latent dynamics model
#         actor_loss: Loss for the policy model
#         z_norms: List of norms of the latent states at each timestep, for monitoring purposes
#     """

#     if config is None:
#         print("No config provided in update function")
#         return 0, 0, 0, 0, 0, [], 0.0, 0.0

#     c1, c2, c3 = config["reward_loss_weight"], config["value_loss_weight"], config["consistency_loss_weight"]
#     rho = config["gamma"]
#     time_lambda = config["time_lambda"]
#     H = config["horizon"]


#     # `weights` are the PER importance-sampling corrections (all ones when the
#     # buffer is not prioritized). `masks` is (B, H): 0 for horizon steps that
#     # fall past an episode end (spliced-in next-episode data).
#     states, actions, rewards, next_states, dones, terminateds, weights, masks = replay_buffer.sample()
#     # print(f"states shape: {states.shape}")

#     # DrQ image augmentation -- pixel observations only, gradient update only.
#     # Independent random shift per (batch element, timestep); states[:, 0] and
#     # every next_states[:, t] are separate encoder inputs so this is correct.
#     aug_pad = config.get("aug_pad", 0)
#     if config["image_observations"] and aug_pad > 0:
#         B, Hs, C, Hh, Ww = next_states.shape
#         s0 = random_shift(states[:, 0], aug_pad)
#         next_states = random_shift(
#             next_states.reshape(B * Hs, C, Hh, Ww), aug_pad
#         ).reshape(B, Hs, C, Hh, Ww)
#     else:
#         s0 = states[:, 0]

#     z = representation_model(s0)
#     zs = [z.detach()]
#     z_norms = [z.detach().norm(dim=-1).mean().item()]

#     total_loss = 0
#     reward_loss = 0
#     value_loss = 0
#     consistency_loss = 0
#     td_error_accum = T.zeros(states.shape[0], device=DEVICE)
#     z_masks = [T.ones(states.shape[0], device=DEVICE)]   # zs[0] = enc(s0), always valid

#     # running trackers for logging
#     q_sum = 0.0
#     target_sum = 0.0
#     valid_sum = 0.0

#     for t in range(H):
#         reward_pred = reward_predictor(z, actions[:, t]).squeeze(-1)
#         value_pred = value_predictor(z, actions[:, t]).squeeze(-1)
#         value_pred2 = value_predictor2(z, actions[:, t]).squeeze(-1)
#         z = latent_dynamics(z, actions[:, t])

#         with T.no_grad():
#             # latent_state_encoded_next = offline_representation_model(next_states[:, t])
#             latent_state_encoded_next = representation_model(next_states[:, t])

#             # target policy smoothing: noisy, clipped next action
#             next_action = policy_model(latent_state_encoded_next)

#             # TODO - Should I use this noise adding code?
#             # noise = (T.randn_like(next_action) * target_noise_std).clamp(
#             #     -target_noise_clip, target_noise_clip
#             # )
#             # next_action = (next_action + noise).clamp(-1.0, 1.0)  # match your action bounds

#             target_q1 = offline_value_predictor(latent_state_encoded_next, next_action).squeeze(-1)
#             target_q2 = offline_value_predictor2(latent_state_encoded_next, next_action).squeeze(-1)
#             target_q = T.min(target_q1, target_q2)

#             # target_q = symexp(T.min(target_q1, target_q2))

#             not_terminal = (~terminateds[:, t]).float()
#             td_target = rewards[:, t] + rho * target_q * not_terminal  # Mask out the target for terminal states
            

#             # print(f"Number of dones at step {t}: {dones[:, t].sum().item()}/{dones.shape[0]}")

#         z_norms.append(z.detach().std(dim=0).mean().item())
#         zs.append(z.detach())
#         z_masks.append(masks[:, t])

#         # m: 1 for real horizon steps, 0 for steps past an episode end.
#         m = masks[:, t]
#         wm = weights * m
#         denom = m.sum().clamp(min=1.0)

#         # reward + consistency: model-learning losses, discounted over the
#         # horizon because later predictions compound model error.
#         reward_loss += (time_lambda ** t) * (wm * (reward_pred - rewards[:, t]) ** 2).sum() / denom
#         consistency_loss += (time_lambda ** t) * (wm * ((z - latent_state_encoded_next) ** 2).mean(-1)).sum() / denom

#         value_loss += (wm * (value_pred  - td_target) ** 2).sum() / denom
#         value_loss += (wm * (value_pred2 - td_target) ** 2).sum() / denom

#         # Accumulate |TD error| per sample (masked) to re-prioritize windows.
#         td_error_accum += m * (value_pred - td_target).detach().abs()

#         q_sum += (value_pred.detach() * m).sum().item()
#         target_sum += (td_target * m).sum().item()
#         valid_sum += m.sum().item()

#     total_loss = (c1 * reward_loss) + (c2 * value_loss) + (c3 * consistency_loss)
#     optimizer.zero_grad()
#     total_loss.backward()
#     T.nn.utils.clip_grad_norm_(
#         list(representation_model.parameters()) +
#         list(latent_dynamics.parameters()) +
#         list(reward_predictor.parameters()) +
#         list(value_predictor.parameters()) +
#         list(value_predictor2.parameters()),
#         max_norm=10.0
#     )
#     optimizer.step()

#     # Update the priorities of the sampled windows using their mean |TD error|
#     # over the valid (unmasked) horizon steps.
#     replay_buffer.update_priorities(td_error_accum / masks.sum(dim=1).clamp(min=1.0))

#     # delayed actor update
#     actor_loss = None
#     actor_loss = update_pi(policy_model, policy_optimizer, value_predictor, value_predictor2, time_lambda, zs, z_masks)
#     if epoch % policy_delay == 0:
        
#         update_target_network(epoch, value_predictor, offline_value_predictor, tau=0.01)
#         update_target_network(epoch, value_predictor2, offline_value_predictor2, tau=0.01)
#         # update_target_network(epoch, representation_model, offline_representation_model, tau=0.005)
            

#     mean_q = q_sum / max(valid_sum, 1.0)
#     mean_target = target_sum / max(valid_sum, 1.0)

#     return (total_loss.item(), reward_loss.item(), value_loss.item(),
#             consistency_loss.item(), actor_loss, z_norms, mean_q, mean_target)


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