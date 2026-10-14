"""
Plot LeWM training runs. Every subfolder of this configs/ directory that holds
training_stats_<seed>.npz files (written by train.py) is one experiment and
gets one line per panel: the across-seed mean of a rolling average, with a 95%
confidence band over seeds.

Figures (saved next to this script):
  - lewm_performance.png  goal distance / success rate / env reward
  - lewm_losses.png       prediction / SIGReg / total loss and grad norm
"""

import glob
import os
import re
import warnings
import numpy as np
import matplotlib.pyplot as plt

CONFIGS_DIR = os.path.dirname(os.path.abspath(__file__))


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


# npz key -> (axis label, plot title)
PERFORMANCE_METRICS = {
    "evaluation_goal_dist_best": ("Distance", "Eval: Best Goal Distance (lower is better)"),
    "evaluation_goal_dist_final": ("Distance", "Eval: Final Goal Distance (lower is better)"),
    "evaluation_goal_success_rate": ("Success Rate", "Eval: Goal Success Rate"),
    "evaluation_episode_rewards": ("Env Reward", "Eval: Env Reward (not optimised)"),
    "training_episode_rewards": ("Env Reward", "Training: Env Reward (not optimised)"),
}

# npz key -> (axis label, plot title, log y-axis)
LOSS_METRICS = {
    "metric_pred_loss": ("Loss", "Prediction Loss  ||f(z_t, a_t) - h(o_t+1)||²", True),
    "metric_sigreg_loss": ("Loss", "SIGReg Loss", True),
    "metric_total_loss": ("Loss", "Total Loss  (pred + λ·SIGReg)", True),
    "metric_grad_norm": ("Norm", "Gradient Norm (pre-clip)", False),
}

ALL_KEYS = list(PERFORMANCE_METRICS) + list(LOSS_METRICS)


def _seed_files(folder_path):
    """Per-seed stats files (training_stats_<seed>.npz), sorted by seed."""
    files = glob.glob(os.path.join(folder_path, "training_stats_*.npz"))
    files = [f for f in files if re.search(r"training_stats_(\d+)\.npz$", f)]
    files.sort(key=lambda f: int(re.search(r"_(\d+)\.npz$", f).group(1)))
    return files


def load_training_data():
    """
    Load every seed's training_stats_<seed>.npz inside each experiment folder
    and stack them into (num_seeds, num_iterations) arrays.

    Expected structure:

    configs/
        reacher_hard/
            config.yaml
            training_stats_0.npz
            training_stats_1.npz
            ...
        <another_experiment>/
            ...

    Seeds that ran for different numbers of iterations are truncated to the
    shortest one so they share a common step axis.
    """

    experiments = {}

    for folder in sorted(os.listdir(CONFIGS_DIR)):
        folder_path = os.path.join(CONFIGS_DIR, folder)

        if not os.path.isdir(folder_path):
            continue

        files = _seed_files(folder_path)
        if not files:
            print(f"Skipping {folder}: no training_stats_<seed>.npz files found")
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


# Two-sided 95% Student-t critical values, indexed by degrees of freedom - 1.
_T_CRIT_95 = np.array([
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
])


def _ci95_halfwidth(values):
    """Half-width of the 95% confidence interval of the across-seed mean
    (t * sample std / sqrt(n)) at each step, using however many seeds are
    finite there. Zero where fewer than two seeds are available."""
    n = np.isfinite(values).sum(axis=0)
    df = np.maximum(n - 1, 1)
    t = np.where(df <= len(_T_CRIT_95), _T_CRIT_95[np.minimum(df, len(_T_CRIT_95)) - 1], 1.96)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # columns with < 2 finite seeds
        sem = np.nanstd(values, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return np.where(n > 1, t * sem, 0.0)


def _plot_series(ax, steps, values, window, label, log=False):
    """Plot the across-seed mean (of per-seed rolling averages) as a bold line,
    with a lighter 95% confidence band of that mean across seeds."""
    steps = np.asarray(steps, dtype=float)
    smoothed = rolling_average(values, window)          # (num_seeds, N)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns (seed phase)
        mean = np.nanmean(smoothed, axis=0)
    ci = _ci95_halfwidth(smoothed)

    mask = np.isfinite(mean)
    if not mask.any():
        return
    steps, mean, ci = steps[mask], mean[mask], ci[mask]

    lower, upper = mean - ci, mean + ci
    if log:
        # Keep the band drawable on a log axis.
        lower = np.maximum(lower, mean * 1e-2)

    n_seeds = values.shape[0]
    line, = ax.plot(steps, mean, linewidth=2, label=f"{label} (n={n_seeds})")
    ax.fill_between(steps, lower, upper, alpha=0.2, color=line.get_color(), linewidth=0)


def plot_lewm(window=5):
    experiments = load_training_data()

    if len(experiments) == 0:
        print("No experiments found.")
        return

    # --- Performance: goal-reaching metrics + env reward vs steps ---
    fig1, axes1 = plt.subplots(2, 3, figsize=(20, 10), sharex=True)

    for ax, (key, (ylabel, title)) in zip(axes1.flat, PERFORMANCE_METRICS.items()):
        for name, data in experiments.items():
            if key not in data:
                continue
            _plot_series(ax, data["steps"], data[key], window, name)

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    for ax in axes1.flat[len(PERFORMANCE_METRICS):]:
        ax.set_visible(False)
    for ax in axes1[-1]:
        ax.set_xlabel("Step")
    # sharex hides the x tick labels on the top row; keep them where the
    # bottom-row panel below is hidden.
    for col in range(len(PERFORMANCE_METRICS) - axes1.shape[1], axes1.shape[1]):
        axes1[0, col].set_xlabel("Step")
        axes1[0, col].xaxis.set_tick_params(labelbottom=True)

    fig1.suptitle(f"LeWM performance (mean ± 95% CI over seeds, {window}-iteration rolling average)")
    fig1.tight_layout()
    fig1.savefig(os.path.join(CONFIGS_DIR, "lewm_performance.png"), dpi=300)

    # --- Losses vs steps ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    for ax, (key, (ylabel, title, log)) in zip(axes2.flat, LOSS_METRICS.items()):
        for name, data in experiments.items():
            if key not in data:
                continue
            _plot_series(ax, data["steps"], data[key], window, name, log=log)

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if log:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    for ax in axes2[-1]:
        ax.set_xlabel("Step")

    fig2.suptitle(f"LeWM training losses (mean ± 95% CI over seeds, {window}-iteration rolling average)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(CONFIGS_DIR, "lewm_losses.png"), dpi=300)

    plt.show()


def main():
    plot_lewm(window=5)


if __name__ == "__main__":
    main()
