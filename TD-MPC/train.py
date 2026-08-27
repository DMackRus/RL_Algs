import torch as T
import torch.nn as nn

from models import RepresentationModel, LatentDynamics, RewardPredictor, ValuePredictor, PolicyModel
from replay_buffer import ReplayBuffer
from planner import PredictiveSampler

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
import yaml

DEVICE = T.device(
    "cuda" if T.cuda.is_available() else "cpu"
)
print(f"Using device: {DEVICE}")

def symlog(x):
    # return x
    return T.sign(x) * T.log(1 + T.abs(x))

def symexp(x):
    return T.sign(x) * (T.exp(T.abs(x)) - 1)

def process_image(image):
    # Resize to 64x64
    image = cv2.resize(image, (64, 64))

    # Convert to float in [-1, 1]
    image = (image.astype("float32") / 255.0 - 0.5) / 0.5

    # HWC -> CHW
    image = T.from_numpy(image).permute(2, 0, 1)

    return image

def collect_play_data(env, replay_buffer, representation_model, dynamics_model, reward_model, value_model, policy_model,
                       config=None, num_episodes=10, noise_std=0.3, train = True, fixed_episode_length=None):

    if config is None:
        print("No config provided, using default values for horizon and sampling noise.")
        return [], []

    representation_model.eval()
    dynamics_model.eval()
    reward_model.eval()
    value_model.eval()
    policy_model.eval()

    rewards = []
    episode_lengths = []

    with T.no_grad():
        for episode in range(num_episodes):

            # Make a predictive sampler for this episode
            predictive_sampler = PredictiveSampler(
                config=config,
                representation_model=representation_model,
                dynamics_model=dynamics_model,
                reward_model=reward_model,
                value_model=value_model,
                policy=policy_model,
                action_dim=env.action_space.shape[0],
                device=DEVICE,
            )

            state, info = env.reset()
            total_reward = 0
            done = False
            episode_length = 0

            if config["image_observations"]:
                state = process_image(env.render())
            
            while not done:

                # Handle if the state is a numpy array or a torch tensor
                if isinstance(state, np.ndarray):
                    state_tensor = T.from_numpy(state).unsqueeze(0).float().to(DEVICE)
                else:
                    state_tensor = state.unsqueeze(0).float().to(DEVICE)

                action, _ = predictive_sampler.plan(state_tensor)
                action = action.squeeze(0).cpu().numpy()

                if train:
                    # Add noise to the action for exploration
                    noise = np.random.normal(0, noise_std, size=action.shape)
                    action = np.clip(action + noise, -1, 1)

                next_state, reward, terminated, truncated, info = env.step(action)
                # print(f"Predicted reward: {predicted['reward'].item():.4f}, Actual reward: {reward:.4f}")
                episode_length += 1

                if fixed_episode_length is not None and episode_length >= fixed_episode_length:
                    done = True
                elif terminated or truncated:
                    done = True

                total_reward += reward
                if config["image_observations"]:
                    next_state = process_image(env.render())
                if train:
                    replay_buffer.add(state, action, reward, next_state, done, terminated)
                state = next_state
            rewards.append(total_reward)
            episode_lengths.append(episode_length)

    representation_model.train()
    dynamics_model.train()
    reward_model.train()
    value_model.train()
    policy_model.train()

    return np.mean(rewards), np.mean(episode_lengths)

def update_target_network(epoch, value_predictor, offline_value_predictor, tau=0.005):
    """
    Update the offline value predictor using a slow-moving average of the online value predictor.
    """
    with T.no_grad():
        for target_param, param in zip(offline_value_predictor.parameters(), value_predictor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def sample_fixed_eval_batch(replay_buffer, start_index=None):
    """
    Sample a batch from the replay buffer.
    """
    if start_index is not None:
        states, actions, rewards, next_states, dones = replay_buffer.sample(start_index)
    else:
        states, actions, rewards, next_states, dones = replay_buffer.sample()
        
    return {
        "states": states.clone(),
        "actions": actions.clone(),
        "rewards": rewards.clone(),
        "next_states": next_states.clone(),
        "dones": dones.clone(),
    }

def get_exploration_std(round_idx, num_rounds, std_start=1.0, std_end=0.05, decay_fraction=0.5):
    """
    Linearly anneal std from std_start to std_end over the first
    `decay_fraction` of training, then hold at std_end.
    """
    decay_rounds = num_rounds * decay_fraction
    progress = min(round_idx / decay_rounds, 1.0)
    return std_start + progress * (std_end - std_start)


# TODO: Add ability to do reward rollouts without noise, to see if the model is actually improving, separate from exploration noise.
@T.no_grad()
def evaluate_fixed_batch(eval_batch, representation_model, offline_representation_model,
                          latent_dynamics, reward_predictor, value_predictor,
                          policy_model, offline_value_predictor,
                          rho=0.99, time_lambda=0.5, H=5):
    """
    Computes the same losses as update(), on the same fixed batch every time,
    with no backward pass and no parameter/target updates. Lets you see
    whether the model is actually improving, separate from per-epoch
    training-batch sampling noise.
    """
    representation_model.eval()
    latent_dynamics.eval()
    reward_predictor.eval()
    value_predictor.eval()
    policy_model.eval()

    states = eval_batch["states"]
    actions = eval_batch["actions"]
    rewards = eval_batch["rewards"]
    next_states = eval_batch["next_states"]

    z = representation_model(states[:, 0])

    reward_loss = 0.0
    value_loss = 0.0
    consistency_loss = 0.0

    for t in range(H - 1):
        reward_pred = reward_predictor(z, actions[:, t]).squeeze(-1)
        value_pred = value_predictor(z, actions[:, t]).squeeze(-1)
        z = latent_dynamics(z, actions[:, t])

        latent_state_encoded_next = offline_representation_model(next_states[:, t])
        td_target = rewards[:, t] + rho * offline_value_predictor(
            latent_state_encoded_next, policy_model(latent_state_encoded_next)
        ).squeeze(-1)

        reward_loss += (time_lambda ** t) * nn.MSELoss()(reward_pred, symlog(rewards[:, t])).item()
        value_loss += (time_lambda ** t) * nn.MSELoss()(value_pred, symlog(td_target)).item()
        consistency_loss += (time_lambda ** t) * nn.MSELoss()(z, latent_state_encoded_next).item()

    representation_model.train()
    latent_dynamics.train()
    reward_predictor.train()
    value_predictor.train()
    policy_model.train()

    return reward_loss, value_loss, consistency_loss

def testing_run(config_filepath):

    # Load the config yaml file
    folder_path = os.path.dirname(config_filepath)
    with open(config_filepath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    latent_dim = config["latent_dim"]
    hidden_dim = config["hidden_dim"]
    INITIAL_NUM_EPISODES = 10
    EPISODE_LENGTH = config["max_episode_length"]
    IMAGE_OBSERVATIONS = config["image_observations"]
    SAMPLING_NOISE_STD = config["sampling_noise"]
    HORIZON = config["horizon"]
    NUM_TRAINING_ROUNDS = config["training_rounds"]
    STEPS_PER_ROUND = config["training_steps_per_round"]
    NEW_EPISODES_PER_ROUND = config["num_episodes_per_round"]
    EVAL_EVERY = config["eval_policy_every"]
    rho = config["gamma"]
    time_lambda = config["time_lambda"]

    # Create the gymnasium environment
    env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    # env = gym.make("Walker2d-v5", render_mode="rgb_array")
    # env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    # env = gym.make("dm_control/acrobot-swingup-v0", render_mode="rgb_array") # Doesnt work?

    if IMAGE_OBSERVATIONS:
        state_dim = (3, 64, 64)  # (C, H, W)
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Create the models
    representation_model = RepresentationModel(latent_dim, state_dim, hidden_dim, image_state=IMAGE_OBSERVATIONS).to(DEVICE)
    latent_dynamics = LatentDynamics(latent_dim, action_dim, hidden_dim).to(DEVICE)
    reward_predictor = RewardPredictor(latent_dim, action_dim,hidden_dim).to(DEVICE)
    value_predictor = ValuePredictor(latent_dim, action_dim, hidden_dim).to(DEVICE)
    offline_value_predictor = ValuePredictor(latent_dim, action_dim, hidden_dim).to(DEVICE)
    offline_representation_model = RepresentationModel(latent_dim, state_dim, hidden_dim, image_state=IMAGE_OBSERVATIONS).to(DEVICE)
    policy_model = PolicyModel(latent_dim, action_dim, hidden_dim).to(DEVICE)
    value_predictor2 = ValuePredictor(latent_dim, action_dim, hidden_dim).to(DEVICE)
    offline_value_predictor2 = ValuePredictor(latent_dim, action_dim, hidden_dim).to(DEVICE)

    # Instantiate the replay buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim, image_observations = IMAGE_OBSERVATIONS, horizon=HORIZON, batch_size=config["batch_size"], device=DEVICE)    

    optimizer = T.optim.Adam(
        list(representation_model.parameters()) +
        list(latent_dynamics.parameters()) +
        list(reward_predictor.parameters()) +
        list(value_predictor.parameters()) +
        list(value_predictor2.parameters()) +
        list(policy_model.parameters()),
        lr=float(config["learning_rate"]))

    policy_optimizer = T.optim.Adam(policy_model.parameters(), lr=float(config["learning_rate_actor"]))

    offline_value_predictor.load_state_dict(value_predictor.state_dict())
    offline_representation_model.load_state_dict(representation_model.state_dict()) # TODO - DO we definitely need an offline representation model?
    offline_value_predictor2.load_state_dict(value_predictor2.state_dict())

    # Collect initial data if not already collected
    # if os.path.exists("initial_data.pkl"):
    #     print("Loading initial data from file...")
    #     replay_buffer.load("initial_data.pkl")
    # else:
    #     print("Collecting initial data...")
    avg_reward, avg_episode_length = collect_play_data(env, replay_buffer, representation_model, latent_dynamics, 
                                                        reward_predictor, value_predictor, policy_model, config=config,
                                                        num_episodes=INITIAL_NUM_EPISODES, noise_std=0.3, 
                                                        train = True, fixed_episode_length=EPISODE_LENGTH)
    print(f"Average reward from initial data collection: {avg_reward}")
        # replay_buffer.save("initial_data.pkl")

    training_rewards, training_episode_length = [], []
    evaluation_rewards, evaluation_episode_length = [], []

    global_epoch = 0
    print("Training starts...")
    for round_idx in range(NUM_TRAINING_ROUNDS):
        avg_reward = 0
        if round_idx > 0:

            noise_std = get_exploration_std(round_idx, NUM_TRAINING_ROUNDS, std_start=1.0, std_end=0.05, decay_fraction=0.5)
            avg_reward, avg_episode_length = collect_play_data(env, replay_buffer, representation_model, latent_dynamics, 
                                                            reward_predictor, value_predictor, policy_model, config=config,
                                                            num_episodes=NEW_EPISODES_PER_ROUND, noise_std=noise_std, 
                                                            train = True, fixed_episode_length=EPISODE_LENGTH)

            print(f"Round {round_idx}: Average reward over {NEW_EPISODES_PER_ROUND} new episodes: {avg_reward:.2f}, Average episode length: {avg_episode_length:.2f}")
            print(f" Replay buffer size: {len(replay_buffer)}")

            training_rewards.append(avg_reward)
            training_episode_length.append(avg_episode_length)

        # How many gradient steps to take per round
        for epoch in range(STEPS_PER_ROUND):

            total_loss, reward_loss, value_loss, consistency_loss, actor_loss, z_norms = update(
                replay_buffer, representation_model, offline_representation_model,
                latent_dynamics, reward_predictor, value_predictor, value_predictor2, policy_model,
                offline_value_predictor, offline_value_predictor2, optimizer, policy_optimizer, global_epoch, config=config,
            )
            print(f"Round {round_idx} Epoch {epoch}: Total Loss: {total_loss:.4f}, "
                  f"Reward: {reward_loss:.4f}, Value: {value_loss:.4f}, "
                  f"Actor: {actor_loss:.4f}, Consistency: {consistency_loss:.4f}, "
                  f"Z Norm: {np.mean(z_norms):.4f}")

            global_epoch += 1

        if round_idx % EVAL_EVERY == 0:
            # Evaluate the model
            eval_score, avg_episode_length = collect_play_data(env, replay_buffer, representation_model, latent_dynamics, reward_predictor, value_predictor, policy_model,
                            config=config, num_episodes=5, noise_std=0.0, train = False, fixed_episode_length=EPISODE_LENGTH)
            print(f"Round {round_idx}: Average reward over 5 evaluation episodes: {eval_score:.2f}")
            evaluation_rewards.append(eval_score)
            evaluation_episode_length.append(avg_episode_length)

            if os.path.exists(f"{folder_path}/model_checkpoints") == False:
                os.makedirs(f"{folder_path}/model_checkpoints")
            T.save({
                "representation_model": representation_model.state_dict(),
                "latent_dynamics": latent_dynamics.state_dict(),
                "reward_predictor": reward_predictor.state_dict(),
                "value_predictor": value_predictor.state_dict(),
                "policy_model": policy_model.state_dict(),
            }, f"{folder_path}/model_checkpoints/checkpoint_round{round_idx}.pt")


    # Save training data
    np.savez(
        f"{folder_path}/training_stats.npz",
        training_episode_rewards=np.array(training_rewards),
        training_episode_lengths=np.array(training_episode_length),
        evaluation_episode_rewards=np.array(evaluation_rewards),
        evaluation_episode_lengths=np.array(evaluation_episode_length)
    )
    env.close()

def update_pi(policy_model, policy_optimizer, value_predictor_1, value_predictor_2, time_lambda, zs):
    """
    Update policy using a sequence of latent states.
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

        q = T.min(value_predictor_1(z, action), value_predictor_2(z, action))

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

def update(replay_buffer, representation_model, offline_representation_model, latent_dynamics,
           reward_predictor, value_predictor, value_predictor2, policy_model,
           offline_value_predictor, offline_value_predictor2,
           optimizer, policy_optimizer, epoch, config=None,
           policy_delay=2, target_noise_std=0.2, target_noise_clip=0.5):

    if config is None:
        print("No config provided, using default values for horizon and sampling noise.")
        return 0, 0, 0, 0, 0, []

    c1, c2, c3 = config["reward_loss_weight"], config["value_loss_weight"], config["consistency_loss_weight"]
    rho = config["gamma"]
    time_lambda = config["time_lambda"]
    H = config["horizon"]

    states, actions, rewards, next_states, dones, terminateds, weights = replay_buffer.sample()
    # print(f"states shape: {states.shape}")

    z = representation_model(states[:, 0])
    zs = [z.detach()]
    z_norms = [z.detach().norm(dim=-1).mean().item()]

    total_loss = 0
    reward_loss = 0
    value_loss = 0
    consistency_loss = 0
    td_error_accum = T.zeros(states.shape[0], device=DEVICE)

    for t in range(H):
        reward_pred = reward_predictor(z, actions[:, t]).squeeze(-1)
        value_pred = value_predictor(z, actions[:, t]).squeeze(-1)
        value_pred2 = value_predictor2(z, actions[:, t]).squeeze(-1)
        z = latent_dynamics(z, actions[:, t])

        with T.no_grad():
            # latent_state_encoded_next = offline_representation_model(next_states[:, t])
            latent_state_encoded_next = representation_model(next_states[:, t])

            # target policy smoothing: noisy, clipped next action
            next_action = policy_model(latent_state_encoded_next)

            # TODO - Should I use this noise adding code?
            # noise = (T.randn_like(next_action) * target_noise_std).clamp(
            #     -target_noise_clip, target_noise_clip
            # )
            # next_action = (next_action + noise).clamp(-1.0, 1.0)  # match your action bounds

            target_q1 = offline_value_predictor(latent_state_encoded_next, next_action).squeeze(-1)
            target_q2 = offline_value_predictor2(latent_state_encoded_next, next_action).squeeze(-1)
            target_q = T.min(target_q1, target_q2)

            # target_q = symexp(T.min(target_q1, target_q2))

            not_terminal = (~terminateds[:, t]).float()
            td_target = rewards[:, t] + rho * target_q * not_terminal  # Mask out the target for terminal states
            

            # print(f"Number of dones at step {t}: {dones[:, t].sum().item()}/{dones.shape[0]}")

        z_norms.append(z.detach().norm(dim=-1).mean().item())
        zs.append(z.detach())

        # Loss terms with bit masking for alive states (i.e. to prevent boundary resets affecting the loss)
        # reward_loss += (time_lambda ** t) * (alive * (reward_pred - symlog(rewards[:, t])) ** 2).mean()
        # value_loss  += (time_lambda ** t) * (alive * (value_pred  - symlog(td_target)) ** 2).mean()
        # value_loss += (time_lambda ** t) * (alive * (value_pred2 - symlog(td_target)) ** 2).mean()
        # consistency_loss += (time_lambda ** t) * (alive * ((z - latent_state_encoded_next) ** 2).mean(-1)).mean()

        # td_error_accum += (value_pred - symlog(td_target)).detach().abs()

        reward_loss += (time_lambda ** t) * ((reward_pred - rewards[:, t]) ** 2).mean()
        value_loss  += (time_lambda ** t) * ((value_pred  - td_target) ** 2).mean()
        value_loss += (time_lambda ** t) * ((value_pred2 - td_target) ** 2).mean()
        consistency_loss += (time_lambda ** t) * ((z - latent_state_encoded_next) ** 2).mean(-1).mean()

        # td_error_accum += (value_pred - td_target).detach().abs()

    total_loss = (c1 * reward_loss) + (c2 * value_loss) + (c3 * consistency_loss)
    optimizer.zero_grad()
    total_loss.backward()
    T.nn.utils.clip_grad_norm_(
        list(representation_model.parameters()) +
        list(latent_dynamics.parameters()) +
        list(reward_predictor.parameters()) +
        list(value_predictor.parameters()) +
        list(value_predictor2.parameters()),
        max_norm=10.0
    )
    optimizer.step()

    # Update the priorities of the replay buffer based on the TD error
    # replay_buffer.update_priorities(td_error_accum / H)

    # delayed actor update
    actor_loss = None
    actor_loss = update_pi(policy_model, policy_optimizer, value_predictor, value_predictor2, time_lambda, zs)
    if epoch % policy_delay == 0:
        
        update_target_network(epoch, value_predictor, offline_value_predictor, tau=0.005)
        update_target_network(epoch, value_predictor2, offline_value_predictor2, tau=0.005)
        # update_target_network(epoch, representation_model, offline_representation_model, tau=0.005)
            

    return (total_loss.item(), reward_loss.item(), value_loss.item(),
            consistency_loss.item(), actor_loss, z_norms)


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