import torch as T
import torch.nn as nn
import torch.optim as optim

import math
import numpy as np
import gymnasium as gym
import os 

from replay_buffer import ReplayBuffer
from utils import compute_td_targets, selection_action_epislon_greedy, soft_update, sample_replay_buffer_for_training, DQNTrainingConfig

DEVICE = T.device("cuda" if T.cuda.is_available() else "cpu")

EPSILON_DECAY = 0.995

# Deep Q-Network (DQN) implementation

# Q-value Q(s, a) - How good is the action a in state s. The higher the Q-value, the better the action.


# TODO - Add seeding

class EpsilonScheduler:
    """Exponential decay schedule."""

    def __init__(self, eps_start: float, eps_end: float, decay: int) -> None:
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay = decay
        self.time_step = 0  # step counter

    def step(self) -> float:
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -self.time_step / self.decay
        )
        self.time_step += 1
        return eps

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = T.relu(self.fc1(x))
        x = T.relu(self.fc2(x))
        return self.fc3(x)



@T.no_grad()
def evaluate_policy(
    q_net: nn.Module,
    training_config: DQNTrainingConfig,
    device: T.device,
    env: gym.Env,
) -> float:
    """
    Run a few evaluation episodes with greedy actions (ε=0).
    Returns mean return.
    """
    total_reward = 0.0

    for _ in range(training_config.eval_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        for _ in range(training_config.max_steps_per_episode):
            s = T.as_tensor(state, dtype=T.float32, device=device).unsqueeze(0)
            q = q_net(s)
            action = int(T.argmax(q, dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        total_reward += episode_reward

    env.close()
    average_reward = total_reward / training_config.eval_episodes
    return average_reward
    
def dqn_loss(
    online_q_net: nn.Module,
    states: T.Tensor,
    actions: T.Tensor,
    targets: T.Tensor,
    criterion: nn.Module,
) -> T.Tensor:
    """Compute Smooth L1 (Huber) loss on TD errors.

    Shapes:
        states:   (B, n_obs)
        actions:  (B, 1) int64
        targets:  (B,)
        returns:  scalar loss
    """
    q_values = online_q_net(states)  # (B, n_actions)
    q_sa = q_values.gather(1, actions).squeeze(1)  # (B,)
    return criterion(q_sa, targets)


def optimize_dqn_step(
    training_config: DQNTrainingConfig,
    online_Q: nn.Module,
    target_Q: nn.Module,
    replay_buffer: ReplayBuffer,
    optimizer: T.optim.Optimizer,
):
    """
    One gradient step from a minibatch samples from the replay buffer.
    """

    if len(replay_buffer) < training_config.batch_size:
        return None  # Not enough samples to perform a training step

    states, actions, rewards, non_final_next_states, non_final_mask, dones = (
        sample_replay_buffer_for_training(replay_buffer, training_config, DEVICE)
    )

    # print(f"states.shape: {states.shape}")
    # print(f"actions.shape: {actions.shape}")
    # print(f"rewards.shape: {rewards.shape}")
    # print(f"non_final_next_states.shape: {non_final_next_states.shape}")
    # print(f"non_final_mask.shape: {non_final_mask.shape}")
    # print(f"dones.shape: {dones.shape}")

    targets = compute_td_targets(
        target_q_net=target_Q,
        non_final_next_states=non_final_next_states,
        non_final_mask=non_final_mask,
        rewards=rewards,
        dones=dones,
        gamma=training_config.gamma,
        device=DEVICE
    )

    loss = dqn_loss(
        online_q_net=online_Q,
        states=states,
        actions=actions,
        targets=targets,
        criterion=nn.SmoothL1Loss(),
    )

    optimizer.zero_grad()
    loss.backward()

    # Clip gradients to prevent exploding gradients
    if training_config.grad_clip_norm is not None:
        nn.utils.clip_grad_norm_(online_Q.parameters(), max_norm=training_config.grad_clip_norm)
    optimizer.step()

    soft_update(target_Q, online_Q, training_config.tau)

    return loss.item()  # Return the loss value for logging purposes

def main():
    print("Starting Gymnasium/DQN/main.py")

    # env = gym.make("LunarLander-v3", render_mode="human")
    env = gym.make("LunarLander-v3", continuous=False, render_mode="rgb_array")

    training_config = DQNTrainingConfig()

    online_Q = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n).to(DEVICE)
    target_Q = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n).to(DEVICE)
    target_Q.load_state_dict(online_Q.state_dict())

    replay_buffer = ReplayBuffer(capacity=training_config.replay_buffer_capacity)

    optimizer = optim.AdamW(
        online_Q.parameters(), lr=training_config.learning_rate, amsgrad=True
    )

    observation, info = env.reset(seed=42)

    epsilon_scheduler = EpsilonScheduler(
        eps_start=training_config.eps_start,
        eps_end=training_config.eps_end,
        decay=training_config.eps_decay
    )
    
    total_steps = 0
    episode_rewards = []
    episode_lengths = []

    for episode in range(training_config.total_episodes):

        episode_reward = 0
        state, info = env.reset(seed=42)
        done = False
        episode_counter = 0

        while not done:

            # Select action using epsilon-greedy policy
            eps = epsilon_scheduler.step()
            action = selection_action_epislon_greedy(online_Q, state, env, eps, device=DEVICE)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Add transition to replay buffer
            replay_buffer.push(state, action, next_state, reward, done)

            loss_value = optimize_dqn_step(
                training_config=training_config,
                online_Q=online_Q,
                target_Q=target_Q,
                replay_buffer=replay_buffer,
                optimizer=optimizer,
            )

            # print(f"Episode {episode}, Step {episode_counter}, Loss: {loss_value}, Epsilon: {eps:.4f}")

            episode_reward += reward
            state = next_state
            total_steps += 1

            episode_counter += 1
            if done:
                break

            if episode_counter >= training_config.max_steps_per_episode:
                print(f"Episode {episode} reached max steps ({training_config.max_steps_per_episode}).")
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_counter)

        if (episode + 1) % 50 == 0 or episode == 0:
            avg_r = np.mean(episode_rewards[-25:])
            print(
                f"[Episode {episode + 1:4d}]  len={episode_counter + 1:4d}  reward={episode_reward:8.2f}  avg25={avg_r:8.2f}"
            )

        if episode % training_config.eval_episodes == 0:
            eval_rewards = evaluate_policy(
                q_net=online_Q,
                training_config=training_config,
                device=DEVICE,
                env=gym.make("LunarLander-v3", continuous=False, render_mode="rgb_array")
            )
            print(f"Episode {episode+1}/{training_config.total_episodes}, Eval mean reward: {eval_rewards:.2f}")

            # Save model checkpoint
            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"dqn_checkpoint_episode_{episode+1}.pth")
            T.save(online_Q.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")

        # Save episode rewards and lengths to a file for later analysis
        np.savez(
            "training_stats.npz",
            episode_rewards=np.array(episode_rewards),
            episode_lengths=np.array(episode_lengths)
        )

    env.close()



if __name__ == "__main__":
    main()