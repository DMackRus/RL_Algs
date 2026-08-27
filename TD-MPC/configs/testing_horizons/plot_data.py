import os
import numpy as np
import matplotlib.pyplot as plt


def rolling_average(x, window):
    """Compute rolling average."""
    if len(x) < window:
        return np.asarray(x)

    return np.convolve(
        x,
        np.ones(window) / window,
        mode="valid"
    )


def load_training_data():
    """
    Load all training_stats.npz files inside subfolders.

    Expected structure:
    
    configs/
        testing_horizons/
            horizon_1/
                training_stats.npz
            horizon_2/
                training_stats.npz
            horizon_3/
                training_stats.npz
    """

    experiments = {}

    # Loop through folders in current directory
    for folder in sorted(os.listdir(".")):
        folder_path = os.path.join(".", folder)

        if not os.path.isdir(folder_path):
            continue

        training_file = os.path.join(folder_path, "training_stats.npz")

        if not os.path.exists(training_file):
            print(f"Skipping {folder}: no training_stats.npz found")
            continue

        data = np.load(training_file)

        experiments[folder] = {
            "rewards": data["training_episode_rewards"],
            "lengths": data["training_episode_lengths"],
            "eval_rewards": data["evaluation_episode_rewards"],
            "eval_lengths": data["evaluation_episode_lengths"]
        }

        print(f"Loaded {folder}")

    return experiments


def plot_training_comparison(
        window=25
):
    experiments = load_training_data()

    if len(experiments) == 0:
        print("No experiments found.")
        return

    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    for name, data in experiments.items():

        rewards = data["rewards"]
        lengths = data["lengths"]

        # Repeat these values 5 times per results as we only generated evaluation data every 5 episodes
        eval_rewards = np.repeat(data["eval_rewards"], 5)
        eval_lengths = np.repeat(data["eval_lengths"], 5)

        reward_avg = rolling_average(rewards, window)
        length_avg = rolling_average(lengths, window)

        # Reward plot
        ax[0].plot(
            rewards,
            alpha=0.15
        )

        ax[0].plot(
            np.arange(window - 1, len(rewards)),
            reward_avg,
            linewidth=2,
            label=name
        )

        # Length plot
        ax[1].plot(
            lengths,
            alpha=0.15
        )

        ax[1].plot(
            np.arange(window - 1, len(lengths)),
            length_avg,
            linewidth=2,
            label=name
        )

        ax[2].plot(
            eval_rewards,
            label=name
        )

        ax[3].plot(
            eval_lengths,
            label=name
        )


    ax[0].set_ylabel("Reward")
    ax[0].set_title("Training Reward Comparison")
    ax[0].legend()
    ax[0].grid(True)


    ax[1].set_ylabel("Episode Length")
    ax[1].set_xlabel("Episode")
    ax[1].set_title("Episode Length Comparison")
    ax[1].legend()
    ax[1].grid(True)

    ax[2].set_ylabel("Evaluation Reward")
    ax[2].set_xlabel("Episode")
    ax[2].legend()
    ax[2].grid(True)

    ax[3].set_ylabel("Evaluation Length")
    ax[3].set_xlabel("Episode")
    ax[3].legend()
    ax[3].grid(True)

    plt.tight_layout()

    plt.savefig(
        "testing_horizons_comparison.png",
        dpi=300
    )

    plt.show()


def main():
    plot_training_comparison(
        window=25
    )


if __name__ == "__main__":
    main()