import os
import numpy as np
import matplotlib.pyplot as plt


def rolling_average(steps, values, window):
    """Compute a rolling average of values, returning the matching x-steps."""
    if len(values) < window:
        return steps, np.asarray(values)

    avg = np.convolve(values, np.ones(window) / window, mode="valid")
    return steps[window - 1:], avg


# npz key -> (nice axis label, plot title)
LOSS_METRICS = {
    "metric_consistency_loss": ("Consistency Loss", "Consistency Loss"),
    "metric_reward_loss": ("Reward Loss", "Reward Loss"),
    "metric_value_loss": ("Value Loss", "Value Loss"),
    "metric_total_loss": ("Total Loss", "Total Loss"),
}

# Diagnostic metrics plotted separately from the losses above (not present for
# every training run yet, e.g. z_std is only logged by tdmpc_adaptive.py so
# far, not tdmpc.py -> loaded/plotted only when present).
OTHER_METRICS = {
    "metric_z_std": ("Latent Std (z)", "Latent Std"),
}


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

        record = {
            "steps": data["steps"],
            "rewards": data["training_episode_rewards"],
        }
        for key in list(LOSS_METRICS) + list(OTHER_METRICS):
            if key in data.files:
                record[key] = data[key]
            else:
                print(f"  {folder}: missing {key}")

        experiments[folder] = record

        print(f"Loaded {folder}")

    return experiments


def _plot_series(ax, steps, values, window, label):
    """Plot a raw (faint) + rolling-average (bold) series, skipping NaNs."""
    steps = np.asarray(steps, dtype=float)
    values = np.asarray(values, dtype=float)

    mask = np.isfinite(values)
    if not mask.any():
        return
    steps, values = steps[mask], values[mask]

    avg_steps, avg = rolling_average(steps, values, window)
    line, = ax.plot(avg_steps, avg, linewidth=2, label=label)
    ax.plot(steps, values, alpha=0.15, color=line.get_color())


def plot_fixed_versus_adaptive(window=25):
    experiments = load_training_data()

    if len(experiments) == 0:
        print("No experiments found.")
        return

    # --- Training reward vs steps ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for name, data in experiments.items():
        _plot_series(ax1, data["steps"], data["rewards"], window, name)

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Training Reward")
    ax1.set_title("Training Reward: Fixed vs Adaptive dt")
    ax1.legend()
    ax1.grid(True)

    fig1.tight_layout()
    fig1.savefig("fixed_versus_adaptive_reward.png", dpi=300)

    # --- Losses (consistency / reward / value / total) vs steps ---
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    for ax, (key, (ylabel, title)) in zip(axes.flat, LOSS_METRICS.items()):
        for name, data in experiments.items():
            if key not in data:
                continue
            _plot_series(ax, data["steps"], data[key], window, name)

        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}: Fixed vs Adaptive dt")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("Step")

    fig2.tight_layout()
    fig2.savefig("fixed_versus_adaptive_losses.png", dpi=300)

    # --- Other diagnostics (e.g. latent std) vs steps, only when present ---
    available_other = [key for key in OTHER_METRICS
                        if any(key in data for data in experiments.values())]

    if available_other:
        fig3, axes3 = plt.subplots(len(available_other), 1,
                                    figsize=(10, 5 * len(available_other)),
                                    squeeze=False)

        for ax, key in zip(axes3[:, 0], available_other):
            ylabel, title = OTHER_METRICS[key]
            for name, data in experiments.items():
                if key not in data:
                    print(f"  {name}: skipping {key} plot, not logged for this run")
                    continue
                _plot_series(ax, data["steps"], data[key], window, name)

            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title}: Fixed vs Adaptive dt")
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig3.tight_layout()
        fig3.savefig("fixed_versus_adaptive_diagnostics.png", dpi=300)

    plt.show()


def main():
    plot_fixed_versus_adaptive(window=25)


if __name__ == "__main__":
    main()
