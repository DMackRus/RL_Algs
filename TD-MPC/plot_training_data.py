import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn as nn

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

def main():

    plot_training("training_stats.npz", window=25)

if __name__ == "__main__":
    main()