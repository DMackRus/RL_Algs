import glob
import os
import re
import warnings
import numpy as np
import matplotlib.pyplot as plt


def rolling_average(values, window):
    """NaN-aware trailing rolling average along the last axis.

    Returns an array the same length as ``values``; early entries average over
    however many finite values are available so far, and NaNs are ignored
    (an entry stays NaN only if its whole window is NaN).
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    kernel = np.ones(window)

    def conv(x):
        return np.convolve(x, kernel, mode="full")[:x.shape[-1]]

    sums = np.apply_along_axis(conv, -1, filled)
    counts = np.apply_along_axis(conv, -1, finite.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


# npz key -> (nice axis label, plot title)
REWARD_METRICS = {
    "training_episode_rewards": ("Training Reward", "Training Reward"),
    "evaluation_episode_rewards": ("Evaluation Reward", "Evaluation Reward"),
}

LOSS_METRICS = {
    "metric_consistency_loss": ("Consistency Loss", "Consistency Loss"),
    "metric_reward_loss": ("Reward Loss", "Reward Loss"),
    "metric_value_loss": ("Value Loss", "Value Loss"),
    "metric_total_loss": ("Total Loss", "Total Loss"),
}

# Diagnostic metrics plotted separately from the losses above (loaded/plotted
# only when present).
OTHER_METRICS = {
    "metric_z_std": ("Latent Std (z)", "Latent Std"),
}

ALL_KEYS = list(REWARD_METRICS) + list(LOSS_METRICS) + list(OTHER_METRICS)


def _seed_files(folder_path):
    """Per-seed stats files (training_stats_<seed>.npz), sorted by seed. Falls
    back to the legacy single training_stats.npz if no per-seed files exist."""
    files = glob.glob(os.path.join(folder_path, "training_stats_*.npz"))
    files = [f for f in files if re.search(r"training_stats_(\d+)\.npz$", f)]
    files.sort(key=lambda f: int(re.search(r"_(\d+)\.npz$", f).group(1)))
    if not files:
        legacy = os.path.join(folder_path, "training_stats.npz")
        if os.path.exists(legacy):
            files = [legacy]
    return files


def load_training_data():
    """
    Load every seed's training_stats_<seed>.npz inside each subfolder and stack
    them into (num_seeds, num_iterations) arrays.

    Expected structure:

    configs/
        fixed_versus_adaptive/
            fixed_dt/
                training_stats_0.npz
                training_stats_1.npz
                ...
            adaptive_dt/
                training_stats_0.npz
                ...

    Seeds that ran for different numbers of iterations are truncated to the
    shortest one so they share a common step axis.
    """

    experiments = {}

    for folder in sorted(os.listdir(".")):
        folder_path = os.path.join(".", folder)

        if not os.path.isdir(folder_path):
            continue

        files = _seed_files(folder_path)
        if not files:
            print(f"Skipping {folder}: no training_stats files found")
            continue

        runs = [np.load(f) for f in files]
        n = min(len(r["steps"]) for r in runs)

        record = {"steps": runs[0]["steps"][:n], "num_seeds": len(runs)}
        for key in ALL_KEYS:
            present = [r for r in runs if key in r.files]
            if len(present) < len(runs):
                print(f"  {folder}: {key} missing in {len(runs) - len(present)}/{len(runs)} seeds")
            if present:
                record[key] = np.stack([r[key][:n].astype(float) for r in present])

        experiments[folder] = record
        print(f"Loaded {folder}: {len(runs)} seed(s), {n} iterations")

    return experiments


def _plot_series(ax, steps, values, window, label, log=False):
    """Plot the across-seed mean (of per-seed rolling averages) as a bold line,
    with a lighter +-1 std band across seeds, plus the across-seed mean of the
    raw (unsmoothed) values as a faint line."""
    steps = np.asarray(steps, dtype=float)
    smoothed = rolling_average(values, window)          # (num_seeds, N)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns (seed phase)
        mean = np.nanmean(smoothed, axis=0)
        std = np.nanstd(smoothed, axis=0)
        raw_mean = np.nanmean(values, axis=0)

    mask = np.isfinite(mean)
    if not mask.any():
        return
    raw_mask = np.isfinite(raw_mean)
    raw_steps, raw_mean = steps[raw_mask], raw_mean[raw_mask]
    steps, mean, std = steps[mask], mean[mask], std[mask]

    lower, upper = mean - std, mean + std
    if log:
        # Keep the band drawable on a log axis.
        lower = np.maximum(lower, mean * 1e-2)

    n_seeds = values.shape[0]
    line, = ax.plot(steps, mean, linewidth=2, label=f"{label} (n={n_seeds})")
    ax.fill_between(steps, lower, upper, alpha=0.2, color=line.get_color(), linewidth=0)
    # ax.plot(raw_steps, raw_mean, alpha=0.4, linewidth=1, color=line.get_color())


def plot_fixed_versus_adaptive(window=5):
    experiments = load_training_data()

    if len(experiments) == 0:
        print("No experiments found.")
        return

    # --- Training / evaluation reward vs steps ---
    fig1, axes1 = plt.subplots(1, len(REWARD_METRICS), figsize=(16, 6), squeeze=False)

    for ax, (key, (ylabel, title)) in zip(axes1[0], REWARD_METRICS.items()):
        for name, data in experiments.items():
            if key not in data:
                continue
            _plot_series(ax, data["steps"], data[key], window, name)

        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}: Fixed vs Adaptive dt (mean ± std over seeds)")
        ax.legend()
        ax.grid(True)

    fig1.tight_layout()
    fig1.savefig("fixed_versus_adaptive_reward_seeds.png", dpi=300)

    # --- Losses (consistency / reward / value / total) vs steps ---
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    for ax, (key, (ylabel, title)) in zip(axes.flat, LOSS_METRICS.items()):
        for name, data in experiments.items():
            if key not in data:
                continue
            _plot_series(ax, data["steps"], data[key], window, name, log=True)

        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}: Fixed vs Adaptive dt")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("Step")

    fig2.tight_layout()
    fig2.savefig("fixed_versus_adaptive_losses_seeds.png", dpi=300)

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
        fig3.savefig("fixed_versus_adaptive_diagnostics_seeds.png", dpi=300)

    plt.show()


def main():
    plot_fixed_versus_adaptive(window=5)


if __name__ == "__main__":
    main()
