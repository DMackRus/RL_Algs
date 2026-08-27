import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn as nn

device = T.device("cuda" if T.cuda.is_available() else "cpu")

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

def plot_training(training_file="training_stats.npz", window=25):
    """Plot episode rewards and lengths with rolling averages."""

    if not os.path.exists(training_file):
        print(f"No training file found: {training_file}")
        return

    data = np.load(training_file)

    rewards = data["episode_rewards"]
    lengths = data["episode_lengths"]

    def rolling_average(x, window):
        if len(x) < window:
            return np.asarray(x)

        return np.convolve(
            x,
            np.ones(window) / window,
            mode="valid"
        )

    reward_avg = rolling_average(rewards, window)
    length_avg = rolling_average(lengths, window)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Rewards
    ax[0].plot(rewards, alpha=0.35, label="Episode reward")
    ax[0].plot(
        np.arange(window - 1, len(rewards)),
        reward_avg,
        linewidth=2,
        label=f"{window}-episode average",
    )
    ax[0].set_ylabel("Reward")
    ax[0].legend()
    ax[0].grid(True)

    # Episode lengths
    ax[1].plot(lengths, alpha=0.35, label="Episode length")
    ax[1].plot(
        np.arange(window - 1, len(lengths)),
        length_avg,
        linewidth=2,
        label=f"{window}-episode average",
    )
    ax[1].set_ylabel("Length")
    ax[1].set_xlabel("Episode")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig("training_plot.png")


def evaluate(
    checkpoint="checkpoints/dqn_checkpoint_episode_491.pth",
    episodes=5,
):
    env = gym.make(
        "LunarLander-v3",
        continuous=False,
        render_mode="human",
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = DQN(state_dim, action_dim).to(device)

    checkpoint_data = T.load(checkpoint, map_location=device)

    # Works whether you saved just the state_dict or a dict containing it
    if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state_dict"])
    else:
        model.load_state_dict(checkpoint_data)

    model.eval()

    rewards = []

    for episode in range(episodes):

        obs, info = env.reset(seed=episode)

        done = False
        total_reward = 0

        while not done:

            state = T.tensor(
                obs,
                dtype=T.float32,
                device=device,
            ).unsqueeze(0)

            with T.no_grad():
                q_values = model(state)

            action = q_values.argmax(dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

        print(
            f"Episode {episode + 1:2d}: reward = {total_reward:.1f}"
        )

    env.close()

    print("-" * 40)
    print(f"Mean reward : {np.mean(rewards):.2f}")
    print(f"Std reward  : {np.std(rewards):.2f}")


def main():

    # Set False if you only want evaluation
    SHOW_TRAINING = True

    if SHOW_TRAINING:
        plot_training("training_stats.npz", window=25)

    # evaluate(
    #     checkpoint="checkpoints/dqn_checkpoint_episode_491.pth",
    #     episodes=5,
    # )


if __name__ == "__main__":
    main()