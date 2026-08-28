"""
Visualisation sanity-checks for the pixel-observation pipeline.

  visualise_trajectory()  -- load a gym env, roll a short random trajectory,
                             and show the raw render frames next to the
                             64x64 frames the encoder actually sees.

  visualise_pixel_shift() -- take one 64x64 frame and show several independent
                             random_shift() augmentations of it (the DrQ-style
                             augmentation applied during the gradient update).

Both save a PNG next to this file and also try to pop up a window.
"""

import os

import gymnasium as gym
import numpy as np
import torch as T
import matplotlib.pyplot as plt

from utils import process_image, random_shift

HERE = os.path.dirname(os.path.abspath(__file__))

# Env used for the pixel pipeline. Must be created with render_mode="rgb_array"
# so env.render() returns an (H, W, 3) uint8 frame.
ENV_ID = "LunarLanderContinuous-v3"
# ENV_ID = "Walker2d-v5"


def _chw_uint8_to_hwc(img: T.Tensor) -> np.ndarray:
    """(3, 64, 64) uint8 torch tensor -> (64, 64, 3) uint8 numpy, for imshow."""
    return img.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)


def visualise_trajectory(env_id: str = ENV_ID, num_steps: int = 8, seed: int = 0):
    """
    Roll a short trajectory with random actions and display, for each step:
      top row    -- the full-resolution frame from env.render()
      bottom row -- the same frame after process_image() (resize to 64x64)
    """
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)

    raw_frames = []       # list of (H, W, 3) uint8
    resized_frames = []    # list of (64, 64, 3) uint8

    raw_frames.append(env.render())
    resized_frames.append(_chw_uint8_to_hwc(process_image(env.render())))

    for _ in range(num_steps):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)

        raw_frames.append(env.render())
        resized_frames.append(_chw_uint8_to_hwc(process_image(env.render())))

        if terminated or truncated:
            env.reset(seed=seed + 1)

    env.close()

    n = len(raw_frames)
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.4))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        axes[0, i].imshow(raw_frames[i])
        axes[0, i].set_title(f"t={i}\nraw {raw_frames[i].shape[1]}x{raw_frames[i].shape[0]}",
                             fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(resized_frames[i], interpolation="nearest")
        axes[1, i].set_title("64x64 (encoder input)", fontsize=8)
        axes[1, i].axis("off")

    fig.suptitle(f"{env_id}: random trajectory, raw render vs. 64x64 training frame")
    fig.tight_layout()

    out = os.path.join(HERE, "test_trajectory.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")
    try:
        plt.show()
    except Exception as e:
        print(f"(plt.show skipped: {e})")
    plt.close(fig)


def visualise_pixel_shift(env_id: str = ENV_ID, pad: int = 4, num_aug: int = 7,
                          frame_stack: int = 3, seed: int = 0):
    """
    Grab one 64x64 frame from the env, build a frame stack from it, and show
    `num_aug` independent random_shift() augmentations.

    random_shift expects (B, C, H, W) and applies ONE random integer shift in
    [0, 2*pad] pixels per batch element, shared across all channels. We stack
    `num_aug` copies of the same frame into the batch dimension so every row is
    a different shift of the identical input -- exactly what the encoder sees
    across a training batch.
    """
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    # take a couple of steps so the frame isn't the degenerate initial pose
    for _ in range(5):
        env.step(env.action_space.sample())
    frame = process_image(env.render())          # (3, 64, 64) uint8
    env.close()

    # (frame_stack copies of the same rgb frame) -> (3 * frame_stack, 64, 64)
    stacked = T.cat([frame] * frame_stack, dim=0).float()

    # batch of identical stacks, one per requested augmentation
    batch = stacked.unsqueeze(0).repeat(num_aug, 1, 1, 1)   # (num_aug, 3k, 64, 64)

    T.manual_seed(seed)
    shifted = random_shift(batch, pad=pad)                  # (num_aug, 3k, 64, 64) float

    def first_rgb(x):
        # first frame of the stack, (3, 64, 64) -> (64, 64, 3) uint8 for imshow
        return x[:3].clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    cols = num_aug + 1
    fig, axes = plt.subplots(1, cols, figsize=(2.0 * cols, 2.6))

    axes[0].imshow(first_rgb(stacked), interpolation="nearest")
    axes[0].set_title("original\n64x64", fontsize=8)
    axes[0].axis("off")

    for i in range(num_aug):
        axes[i + 1].imshow(first_rgb(shifted[i]), interpolation="nearest")
        axes[i + 1].set_title(f"shift #{i + 1}", fontsize=8)
        axes[i + 1].axis("off")

    fig.suptitle(f"random_shift(pad={pad}) -- DrQ augmentation, "
                 f"integer shift in [0, {2 * pad}] px per sample")
    fig.tight_layout()

    out = os.path.join(HERE, "test_pixel_shift.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")
    try:
        plt.show()
    except Exception as e:
        print(f"(plt.show skipped: {e})")
    plt.close(fig)


# Claude function to test overfitting to a single batch. Moved here from train.py script to avoid clutter.
def overfit_batch_test(config_filepath, num_steps=3000, log_every=50, num_collect_episodes=8):
    """
    Diagnostic: can the world model (encoder + latent dynamics + reward/value
    heads) *overfit a single fixed batch* of trajectories?

    Trains only the world-model losses -- reward, consistency, and value against
    a STATIONARY Monte-Carlo target (no bootstrap, no target networks, no policy,
    no augmentation) -- on one frozen batch for `num_steps` gradient steps.

    Reading the result:
      * reward loss and consistency loss should fall by 1-2+ orders of magnitude
        and keep dropping. That means the pipeline is correct and the encoder
        input carries the needed information -> your real problem is RL / data
        (exploration, tiny all-crash buffer, too many grad steps per round).
      * If they plateau high, the model cannot fit even data it sees every step
        -> the encoder input lacks the signal (image resolution / observation),
        or there is a wiring bug. More training episodes will NOT help.
    """
    folder_path = os.path.dirname(config_filepath)
    with open(config_filepath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    latent_dim = config["latent_dim"]
    hidden_dim = config["hidden_dim"]
    IMAGE = config["image_observations"]
    FRAME_STACK = config.get("frame_stack", 1)
    H = config["horizon"]
    rho = config["gamma"]
    time_lambda = config["time_lambda"]

    env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    if IMAGE:
        state_dim = (3 * FRAME_STACK, 64, 64)
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    representation_model = RepresentationModel(latent_dim, state_dim, hidden_dim,
                                              image_state=IMAGE, frame_stack=FRAME_STACK).to(DEVICE)
    latent_dynamics = LatentDynamics(latent_dim, action_dim, hidden_dim).to(DEVICE)
    reward_predictor = RewardPredictor(latent_dim, action_dim, hidden_dim).to(DEVICE)
    value_predictor = ValuePredictor(latent_dim, action_dim, hidden_dim).to(DEVICE)
    policy_model = PolicyModel(latent_dim, action_dim, hidden_dim).to(DEVICE)  # only used to drive data collection

    replay_buffer = ReplayBuffer(state_dim, action_dim, image_observations=IMAGE, horizon=H,
                                 batch_size=config["batch_size"], device=DEVICE)

    # --- collect a little data (untrained planner + noise -> varied trajectories) ---
    print(f"Collecting {num_collect_episodes} episodes for the overfit test...")
    collect_play_data(env, replay_buffer, representation_model, latent_dynamics, reward_predictor,
                      value_predictor, policy_model, config=config, num_episodes=num_collect_episodes,
                      noise_std=0.5, train=True, fixed_episode_length=config["max_episode_length"])
    print(f"Buffer size: {len(replay_buffer)}")

    # --- freeze ONE batch ---
    states, actions, rewards, next_states, dones, terminateds, weights = replay_buffer.sample()
    states = states.clone()
    actions = actions.clone()
    rewards = rewards.clone()
    next_states = next_states.clone()

    # stationary Monte-Carlo return target over the horizon window (no bootstrap)
    with T.no_grad():
        mc_target = T.zeros_like(rewards)
        running = T.zeros(rewards.shape[0], device=DEVICE)
        for t in reversed(range(H)):
            running = rewards[:, t] + rho * running
            mc_target[:, t] = running

    opt = T.optim.Adam(
        list(representation_model.parameters()) +
        list(latent_dynamics.parameters()) +
        list(reward_predictor.parameters()) +
        list(value_predictor.parameters()),
        lr=float(config["learning_rate"]))

    for m in (representation_model, latent_dynamics, reward_predictor, value_predictor):
        m.train()

    hist = {"step": [], "reward": [], "value": [], "consistency": [], "z_std": []}

    for step in range(num_steps):
        z = representation_model(states[:, 0])
        reward_loss = 0.0
        value_loss = 0.0
        consistency_loss = 0.0
        for t in range(H):
            reward_pred = reward_predictor(z, actions[:, t]).squeeze(-1)
            value_pred = value_predictor(z, actions[:, t]).squeeze(-1)
            z = latent_dynamics(z, actions[:, t])
            with T.no_grad():
                z_next = representation_model(next_states[:, t])
            reward_loss += (time_lambda ** t) * T.nn.functional.smooth_l1_loss(reward_pred, rewards[:, t])
            value_loss += (time_lambda ** t) * T.nn.functional.smooth_l1_loss(value_pred, mc_target[:, t])
            consistency_loss += (time_lambda ** t) * ((z - z_next) ** 2).mean(-1).mean()

        loss = reward_loss + value_loss + consistency_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0:
            with T.no_grad():
                z_std = representation_model(states[:, 0]).std(dim=0).mean().item()
            hist["step"].append(step)
            hist["reward"].append(reward_loss.item())
            hist["value"].append(value_loss.item())
            hist["consistency"].append(consistency_loss.item())
            hist["z_std"].append(z_std)
            print(f"step {step:5d} | reward {reward_loss.item():9.4f} | "
                  f"value {value_loss.item():10.4f} | consistency {consistency_loss.item():9.5f} | "
                  f"z.std {z_std:.3f}")

    r0, rN = hist["reward"][0], hist["reward"][-1]
    c0, cN = hist["consistency"][0], hist["consistency"][-1]
    print(f"\nreward loss:      {r0:.4f} -> {rN:.4f}  ({r0 / max(rN, 1e-9):.1f}x reduction)")
    print(f"consistency loss: {c0:.5f} -> {cN:.5f}  ({c0 / max(cN, 1e-9):.1f}x reduction)")
    print("PASS (pipeline OK, info present) if both drop >~20x and are still trending down.")
    print("FAIL (info/wiring problem) if they plateau after an initial dip.")

    try:
        plt.switch_backend("Agg")
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for a, key, title in zip(ax, ("reward", "value", "consistency"),
                                 ("reward loss", "value loss (fixed MC target)", "consistency loss")):
            a.plot(hist["step"], hist[key])
            a.set_title(title)
            a.set_xlabel("gradient step")
            a.set_yscale("log")
        fig.suptitle(f"Overfit-one-batch test ({'image' if IMAGE else 'state'} obs)")
        fig.tight_layout()
        out_path = f"{folder_path}/overfit_test.png"
        fig.savefig(out_path, dpi=120)
        print(f"Saved loss curves to {out_path}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    env.close()


if __name__ == "__main__":
    visualise_trajectory()
    visualise_pixel_shift()
