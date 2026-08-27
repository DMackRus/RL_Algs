import torch as T
import torch.nn as nn
import numpy as np
from typing import Tuple
import random
from dataclasses import dataclass

from replay_buffer import ReplayBuffer

@dataclass
class DQNTrainingConfig:
    seed: int = 42
    total_episodes: int = 500
    replay_buffer_capacity: int = 100_000
    batch_size: int = 128
    gamma: float = 0.99
    learning_rate: float = 1e-4
    tau: float = 5e-3
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: int = 10000
    grad_clip_norm: float | None = 10.0  # set None to disable
    eval_episodes: int = 5
    max_steps_per_episode: int = 1000

def selection_action_epislon_greedy(online_q_net, state, env, epsilon, device):
    """
    Select an action using epsilon-greedy policy.
    """
    if random.random() < epsilon:
        # Explore: select a random action
        return env.action_space.sample()
    else:
        # Exploit: select the action with max Q-value
        state_tensor = T.as_tensor(state, dtype=T.float32, device=device).unsqueeze(0)
        with T.no_grad():
            q_values = online_q_net(state_tensor)
        return q_values.argmax().item()


def compute_td_targets(
    target_q_net: nn.Module,
    non_final_next_states: T.Tensor,
    non_final_mask: T.Tensor,
    rewards: T.Tensor,
    dones: T.Tensor,
    gamma: float,
    device: T.device
) -> T.Tensor:
    """
    Compute one-step TD targets (no gradients through the target network).
    """

    B = rewards.shape[0]
    next_state_values = T.zeros(B, device=device)

    if non_final_next_states.numel() > 0:
        # What is difference between .no_grad and .inference mode?
        with T.no_grad():
            max_next_q = target_q_net(non_final_next_states).max(dim=1).values
        next_state_values[non_final_mask] = max_next_q

    targets = rewards + (gamma * next_state_values * (1 - dones))
    return targets #(B,)

# Soft / Polyak update of target network parameters
@T.no_grad()
def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

def sample_replay_buffer_for_training(
    replay_buffer: ReplayBuffer,
    training_config: DQNTrainingConfig,
    device: T.device
) -> tuple[
    T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor
]:
    """
    Sample a batch from `replay_buffer` and convert to tensors for DQN training.

    Returns:
        states_t                 : (B, *state_shape) float32
        actions_t                : (B, 1) long
        rewards_t                : (B,) float32
        non_final_next_states_t  : (N, *state_shape) float32  [N <= B]
        non_final_mask           : (B,) bool
        dones_t                  : (B,) float32  (1.0 if done else 0.0)
    """

    states, actions, next_states, rewards, dones = replay_buffer.sample(training_config.batch_size)

    states_t = T.as_tensor(np.stack(states), dtype=T.float32, device=device)
    actions_t = T.as_tensor(actions, dtype=T.long, device=device).unsqueeze(1)
    rewards_t = T.as_tensor(rewards, dtype=T.float32, device=device)
    dones_t = T.as_tensor(dones, dtype=T.float32, device=device)

    # Mask and pack only non-final next states
    non_final_mask = T.as_tensor(
        [ns is not None for ns in next_states], dtype=T.bool, device=device
    )

    if non_final_mask.any().item():
        non_final_next_states_t = T.as_tensor(
            np.stack([ns for ns in next_states if ns is not None]),
            dtype=T.float32,
            device=device,
        )
    else:
        # Safe placeholder (will not be indexed when mask is all False)
        state_dim = states_t.shape[1:]
        non_final_next_states_t = T.empty(
            (0, *state_dim), dtype=T.float32, device=device
        )

    return (
        states_t,
        actions_t,
        rewards_t,
        non_final_next_states_t,
        non_final_mask,
        dones_t,
    )