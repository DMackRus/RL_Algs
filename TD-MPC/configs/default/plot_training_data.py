import os

import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn as nn


def rolling_average(x, window):
    if len(x) < window:
        return np.asarray(x)

    return np.convolve(
        x,
        np.ones(window) / window,
        mode="valid"
    )


def _smoothed(x, y, window):
    """Rolling average of y (and the matching x) with NaN rows dropped first."""
    x, y = np.asarray(x), np.asarray(y)
    ok = ~np.isnan(y)
    x, y = x[ok], y[ok]
    if len(y) < window:
        return x, y
    y_avg = rolling_average(y, window)
    return x[window - 1:], y_avg


def plot_training(training_file="default/training_stats.npz", window=25):
    """Plot episode rewards and lengths with rolling averages."""

    if not os.path.exists(training_file):
        print(f"No training file found: {training_file}")
        return

    data = np.load(training_file)

    rewards = data["training_episode_rewards"]
    lengths = data["training_episode_lengths"]

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


def plot_losses(training_file="default/training_stats.npz", window=25):
    """Plot the TOLD update losses (total / consistency / reward / value) in a
    2x2 grid, raw plus a rolling average. Losses are NaN during the seed phase."""

    if not os.path.exists(training_file):
        print(f"No training file found: {training_file}")
        return

    data = np.load(training_file)

    # x-axis: environment steps if logged, else the iteration index.
    if "steps" in data.files:
        x = data["steps"]
        xlabel = "Environment step"
    else:
        x = np.arange(len(data["training_episode_rewards"]))
        xlabel = "Iteration"

    panels = [
        ("metric_total_loss", "Total loss"),
        ("metric_consistency_loss", "Consistency loss"),
        ("metric_reward_loss", "Reward loss"),
        ("metric_value_loss", "Value loss"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for ax, (key, title) in zip(axes.flat, panels):
        ax.set_title(title)
        ax.set_ylabel("Loss")
        ax.grid(True)

        if key not in data.files:
            ax.text(0.5, 0.5, f"{key} not in stats file",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        y = data[key]
        ax.plot(x, y, alpha=0.35, label="per iteration")

        x_avg, y_avg = _smoothed(x, y, window)
        if len(y_avg):
            ax.plot(x_avg, y_avg, linewidth=2, label=f"{window}-iter average")

        ax.legend()

    for ax in axes[-1]:
        ax.set_xlabel(xlabel)

    plt.tight_layout()
    plt.savefig("training_losses.png")


def main():

    plot_training("training_stats.npz", window=25)
    plot_losses("training_stats.npz", window=25)


if __name__ == "__main__":
    main()
