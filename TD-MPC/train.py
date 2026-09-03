import torch as T
import torch.nn as nn

from models import RepresentationModel, LatentDynamics, RewardPredictor, ValuePredictor, PolicyModel
from replay_buffer import ReplayBuffer
from planner import PredictiveSampler, MPPISampler
from utils import symlog, symexp, make_frame_stacker, process_image, random_shift, update_target_network

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
import yaml

from dm_control import suite

DEVICE = T.device(
    "cuda" if T.cuda.is_available() else "cpu"
)
print(f"Using device: {DEVICE}")


def collect_play_data(env, replay_buffer, representation_model, dynamics_model, reward_model, value_model, policy_model,
                       config=None, num_episodes=10, noise_std=0.3, train = True, fixed_episode_length=None,
                       random_actions=False):
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

    representation_model.eval()
    dynamics_model.eval()
    reward_model.eval()
    value_model.eval()
    policy_model.eval()

    rewards = []
    episode_lengths = []

    with T.no_grad():
        for episode in range(num_episodes):

            # Fresh planner per episode (skipped when taking uniform random actions)
            predictive_sampler = None
            if not random_actions:
                predictive_sampler = MPPISampler(
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
                # state = process_image(env.render())
                stack_reset, stack_push = make_frame_stacker(config["frame_stack"])
                state = stack_reset(process_image(env.render()))
            
            while not done:

                if random_actions:
                    action = env.action_space.sample()
                else:
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
                    # next_state = process_image(env.render())
                    next_state = stack_push(process_image(env.render()))
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


def sample_fixed_eval_batch(replay_buffer, start_index=None):
    """
    Sample a batch from the replay buffer.
    """
    if start_index is not None:
        states, actions, rewards, next_states, dones, *_ = replay_buffer.sample(start_index)
    else:
        states, actions, rewards, next_states, dones, *_ = replay_buffer.sample()
        
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
    # Gradient steps per round are tied to how many env steps were collected that
    # round (update-to-data ratio). Set to null/omit to fall back to the fixed
    # training_steps_per_round.
    UPDATES_PER_ENV_STEP = config.get("updates_per_env_step", 1.0)
    rho = config["gamma"]
    time_lambda = config["time_lambda"]

    # Create the gymnasium environment
    # env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    env = gym.make("Walker2d-v5", render_mode="rgb_array", reset_noise_scale=0.01)
    # env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    # env = gym.make("dm_control/acrobot-swingup-v0", render_mode="rgb_array") # Doesnt work?

    if IMAGE_OBSERVATIONS:
        state_dim = (3 * config["frame_stack"], 64, 64)  # (C, H, W)
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Create the models
    representation_model = RepresentationModel(latent_dim, state_dim, hidden_dim, image_state=IMAGE_OBSERVATIONS, frame_stack=FRAME_STACK).to(DEVICE)
    latent_dynamics = LatentDynamics(latent_dim, action_dim, hidden_dim).to(DEVICE)
    reward_predictor = RewardPredictor(latent_dim, action_dim,hidden_dim).to(DEVICE)
    value_predictor = ValuePredictor(latent_dim, action_dim, hidden_dim).to(DEVICE)
    offline_value_predictor = ValuePredictor(latent_dim, action_dim, hidden_dim).to(DEVICE)
    offline_representation_model = RepresentationModel(latent_dim, state_dim, hidden_dim, image_state=IMAGE_OBSERVATIONS, frame_stack=FRAME_STACK).to(DEVICE)
    policy_model = PolicyModel(latent_dim, action_dim, hidden_dim).to(DEVICE)
    value_predictor2 = ValuePredictor(latent_dim, action_dim, hidden_dim).to(DEVICE)
    offline_value_predictor2 = ValuePredictor(latent_dim, action_dim, hidden_dim).to(DEVICE)

    # Zero-init ONLY the reward head's final layer (reward predictions start at
    # 0). The Q heads are deliberately left with their default init: with gamma
    # near 1 the value target has to climb from ~r to ~r/(1-gamma) over many
    # bootstrap steps, and a zeroed Q head makes that warmup far too slow.
    reward_predictor.net[-1].weight.data.fill_(0)
    reward_predictor.net[-1].bias.data.fill_(0)

    # Zero initialise the weights of the Q models and reward model
    # for model in [value_predictor, offline_value_predictor, value_predictor2, offline_value_predictor2, reward_predictor]:
    #     for layer in model.modules():
    #         if isinstance(layer, nn.Linear):
    #             nn.init.zeros_(layer.weight)
    #             if layer.bias is not None:
    #                 nn.init.zeros_(layer.bias)

    # Instantiate the replay buffer (prioritized experience replay by default)
    replay_buffer = ReplayBuffer(
        state_dim, action_dim,
        image_observations=IMAGE_OBSERVATIONS,
        capacity=config.get("replay_capacity", 100000),
        horizon=HORIZON,
        batch_size=config["batch_size"],
        device=DEVICE,
        prioritized=config.get("prioritized_replay", True),
        alpha=config.get("per_alpha", 0.6),
        beta_start=config.get("per_beta_start", 0.4),
        beta_frames=config.get("per_beta_frames", 200000),
    )

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

    # Seed the buffer with uniform random actions (full action amplitude) so the
    # dynamics / reward models see diverse transitions before any planning.
    avg_reward, avg_episode_length = collect_play_data(env, replay_buffer, representation_model, latent_dynamics,
                                                        reward_predictor, value_predictor, policy_model, config=config,
                                                        num_episodes=INITIAL_NUM_EPISODES, noise_std=0.3,
                                                        train = True, fixed_episode_length=EPISODE_LENGTH,
                                                        random_actions=True)
    seed_env_steps = INITIAL_NUM_EPISODES * avg_episode_length
    print(f"Seeded replay buffer: {len(replay_buffer)} random-action transitions "
          f"(avg reward {avg_reward:.2f}, avg episode length {avg_episode_length:.1f})")
        # replay_buffer.save("initial_data.pkl")

    training_rewards, training_episode_length = [], []
    evaluation_rewards, evaluation_episode_length = [], []

    global_epoch = 0
    print("Training starts...")
    for round_idx in range(NUM_TRAINING_ROUNDS):
        avg_reward = 0
        if round_idx == 0:
            # Round 0 trains on the random-action seed data only.
            env_steps_this_round = seed_env_steps
        else:

            noise_std = get_exploration_std(round_idx, NUM_TRAINING_ROUNDS, std_start=0.2, std_end=0.05, decay_fraction=0.5)
            avg_reward, avg_episode_length = collect_play_data(env, replay_buffer, representation_model, latent_dynamics,
                                                            reward_predictor, value_predictor, policy_model, config=config,
                                                            num_episodes=NEW_EPISODES_PER_ROUND, noise_std=noise_std,
                                                            train = True, fixed_episode_length=EPISODE_LENGTH)
            env_steps_this_round = NEW_EPISODES_PER_ROUND * avg_episode_length

            print(f"Round {round_idx}: Average reward over {NEW_EPISODES_PER_ROUND} new episodes: {avg_reward:.2f}, Average episode length: {avg_episode_length:.2f}")
            print(f" Replay buffer size: {len(replay_buffer)}")

            training_rewards.append(avg_reward)
            training_episode_length.append(avg_episode_length)

        # Gradient steps this round track env steps collected (update-to-data ratio).
        if UPDATES_PER_ENV_STEP is None:
            num_updates = STEPS_PER_ROUND
        else:
            num_updates = max(1, int(round(UPDATES_PER_ENV_STEP * env_steps_this_round)))

        round_log = {"total": [], "reward": [], "value": [], "consistency": [],
                     "actor": [], "z_std": [], "q": [], "target": []}
        for epoch in range(num_updates):

            (total_loss, reward_loss, value_loss, consistency_loss,
             actor_loss, z_norms, mean_q, mean_target) = update(
                replay_buffer, representation_model, offline_representation_model,
                latent_dynamics, reward_predictor, value_predictor, value_predictor2, policy_model,
                offline_value_predictor, offline_value_predictor2, optimizer, policy_optimizer, global_epoch, config=config,
            )
            round_log["total"].append(total_loss)
            round_log["reward"].append(reward_loss)
            round_log["value"].append(value_loss)
            round_log["consistency"].append(consistency_loss)
            round_log["actor"].append(actor_loss if actor_loss is not None else np.nan)
            # z_norms[0] is a latent norm; z_norms[1:] are per-dim batch stds
            round_log["z_std"].append(np.mean(z_norms[1:]) if len(z_norms) > 1 else np.nan)
            round_log["q"].append(mean_q)
            round_log["target"].append(mean_target)

            global_epoch += 1

        print(f"  [train {round_idx}] updates={num_updates} | "
              f"total {np.mean(round_log['total']):.3f} | "
              f"reward {np.mean(round_log['reward']):.4f} | "
              f"value {np.mean(round_log['value']):.3f} | "
              f"consist {np.mean(round_log['consistency']):.4f} | "
              f"actor {np.nanmean(round_log['actor']):.3f} | "
              f"z_std {np.nanmean(round_log['z_std']):.4f} | "
              f"Q {np.mean(round_log['q']):.2f} vs tgt {np.mean(round_log['target']):.2f}")

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

def update(replay_buffer, representation_model, offline_representation_model, latent_dynamics,
           reward_predictor, value_predictor, value_predictor2, policy_model,
           offline_value_predictor, offline_value_predictor2,
           optimizer, policy_optimizer, epoch, config=None,
           policy_delay=2, target_noise_std=0.2, target_noise_clip=0.5):
    """
    Single update step for the latent dynamics model, reward predictor, value predictor, and policy model.

    Returns:
        total_loss: Total loss for thje update step
        reward_loss: Loss for the reward predictor
        value_loss: Loss for the value predictor
        consistency_loss: Loss for the latent dynamics model
        actor_loss: Loss for the policy model
        z_norms: List of norms of the latent states at each timestep, for monitoring purposes
    """

    if config is None:
        print("No config provided in update function")
        return 0, 0, 0, 0, 0, [], 0.0, 0.0

    c1, c2, c3 = config["reward_loss_weight"], config["value_loss_weight"], config["consistency_loss_weight"]
    rho = config["gamma"]
    time_lambda = config["time_lambda"]
    H = config["horizon"]
    # Extra weight on horizon steps that end in termination -- otherwise the
    # rare "the episode ended here" signal is drowned out and the critic never
    # learns that falling over is bad.
    terminal_boost = config.get("terminal_loss_weight", 9.0)

    # `weights` are the PER importance-sampling corrections (all ones when the
    # buffer is not prioritized). `masks` is (B, H): 0 for horizon steps that
    # fall past an episode end (spliced-in next-episode data).
    states, actions, rewards, next_states, dones, terminateds, weights, masks = replay_buffer.sample()
    # print(f"states shape: {states.shape}")

    # DrQ image augmentation -- pixel observations only, gradient update only.
    # Independent random shift per (batch element, timestep); states[:, 0] and
    # every next_states[:, t] are separate encoder inputs so this is correct.
    aug_pad = config.get("aug_pad", 0)
    if config["image_observations"] and aug_pad > 0:
        B, Hs, C, Hh, Ww = next_states.shape
        s0 = random_shift(states[:, 0], aug_pad)
        next_states = random_shift(
            next_states.reshape(B * Hs, C, Hh, Ww), aug_pad
        ).reshape(B, Hs, C, Hh, Ww)
    else:
        s0 = states[:, 0]

    z = representation_model(s0)
    zs = [z.detach()]
    z_norms = [z.detach().norm(dim=-1).mean().item()]

    total_loss = 0
    reward_loss = 0
    value_loss = 0
    consistency_loss = 0
    td_error_accum = T.zeros(states.shape[0], device=DEVICE)
    z_masks = [T.ones(states.shape[0], device=DEVICE)]   # zs[0] = enc(s0), always valid

    # running trackers for logging
    q_sum = 0.0
    target_sum = 0.0
    valid_sum = 0.0

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

        z_norms.append(z.detach().std(dim=0).mean().item())
        zs.append(z.detach())
        z_masks.append(masks[:, t])

        # m: 1 for real horizon steps, 0 for steps past an episode end.
        m = masks[:, t]
        wm = weights * m
        denom = m.sum().clamp(min=1.0)

        # reward + consistency: model-learning losses, discounted over the
        # horizon because later predictions compound model error.
        reward_loss += (time_lambda ** t) * (wm * (reward_pred - rewards[:, t]) ** 2).sum() / denom
        consistency_loss += (time_lambda ** t) * (wm * ((z - latent_state_encoded_next) ** 2).mean(-1)).sum() / denom

        # value: accuracy matters equally at every horizon step for planning, so
        # it is NOT horizon-discounted; terminal steps are up-weighted so the
        # "falling ends the episode" signal survives.
        v_w = 1.0 + terminal_boost * terminateds[:, t].float()
        value_loss += (wm * v_w * (value_pred  - td_target) ** 2).sum() / denom
        value_loss += (wm * v_w * (value_pred2 - td_target) ** 2).sum() / denom

        # Accumulate |TD error| per sample (masked) to re-prioritize windows.
        td_error_accum += m * (value_pred - td_target).detach().abs()

        q_sum += (value_pred.detach() * m).sum().item()
        target_sum += (td_target * m).sum().item()
        valid_sum += m.sum().item()

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

    # Update the priorities of the sampled windows using their mean |TD error|
    # over the valid (unmasked) horizon steps.
    replay_buffer.update_priorities(td_error_accum / masks.sum(dim=1).clamp(min=1.0))

    # delayed actor update
    actor_loss = None
    actor_loss = update_pi(policy_model, policy_optimizer, value_predictor, value_predictor2, time_lambda, zs, z_masks)
    if epoch % policy_delay == 0:
        
        update_target_network(epoch, value_predictor, offline_value_predictor, tau=0.005)
        update_target_network(epoch, value_predictor2, offline_value_predictor2, tau=0.005)
        # update_target_network(epoch, representation_model, offline_representation_model, tau=0.005)
            

    mean_q = q_sum / max(valid_sum, 1.0)
    mean_target = target_sum / max(valid_sum, 1.0)

    return (total_loss.item(), reward_loss.item(), value_loss.item(),
            consistency_loss.item(), actor_loss, z_norms, mean_q, mean_target)


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