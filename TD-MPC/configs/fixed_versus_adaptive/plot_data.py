import os
import numpy as np
import matplotlib.pyplot as plt


def rolling_average(steps, values, window):
    """Compute a rolling average of values, returning the matching x-steps."""
    if len(values) < window:
        return steps, np.asarray(values)

    avg = np.convolve(values, np.ones(window) / window, mode="valid")
    return steps[window - 1:], avg


def load_training_data():
    """
    Load all training_stats.npz files inside subfolders.

    Expected structure:

    configs/
        fixed_versus_adaptive/
            fixed_dt/
                training_stats.npz
            adaptive_dt/
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
            "steps": data["steps"],
            "rewards": data["training_episode_rewards"],
            "total_loss": data["metric_total_loss"],
        }

        print(f"Loaded {folder}")

    return experiments


def plot_fixed_versus_adaptive(window=25):
    experiments = load_training_data()

    if len(experiments) == 0:
        print("No experiments found.")
        return

    # --- Training reward vs steps ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for name, data in experiments.items():
        steps = data["steps"]
        rewards = data["rewards"]
        avg_steps, reward_avg = rolling_average(steps, rewards, window)

        line, = ax1.plot(avg_steps, reward_avg, linewidth=2, label=name)
        ax1.plot(steps, rewards, alpha=0.15, color=line.get_color())

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Training Reward")
    ax1.set_title("Training Reward: Fixed vs Adaptive dt")
    ax1.legend()
    ax1.grid(True)

    fig1.tight_layout()
    fig1.savefig("fixed_versus_adaptive_reward.png", dpi=300)

    # --- Total loss vs steps ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for name, data in experiments.items():
        steps = data["steps"]
        total_loss = data["total_loss"]
        avg_steps, loss_avg = rolling_average(steps, total_loss, window)

        line, = ax2.plot(avg_steps, loss_avg, linewidth=2, label=name)
        ax2.plot(steps, total_loss, alpha=0.15, color=line.get_color())

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Total Loss")
    ax2.set_title("Total Loss: Fixed vs Adaptive dt")
    ax2.legend()
    ax2.grid(True)

    fig2.tight_layout()
    fig2.savefig("fixed_versus_adaptive_total_loss.png", dpi=300)

    plt.show()


def main():
    plot_fixed_versus_adaptive(window=25)


if __name__ == "__main__":
    main()
