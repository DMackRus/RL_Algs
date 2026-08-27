import os
import pickle
import random
import cv2
import matplotlib.pyplot as plt

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

from models import RSSM, RewardModel, ContinueModel
from encoder_decoder import Encoder, Decoder
from utils import load_config, get_base_directory, initialize_weights, create_normal_dist

from replay_buffer import ReplayBuffer

from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if T.cuda.is_available() else "cpu"

IMAGE_OBSERVATIONS = True

def process_image(image):
    # Resize to 64x64
    image = cv2.resize(image, (64, 64))

    # Convert to float in [-1, 1]
    image = (image.astype("float32") / 255.0 - 0.5) / 0.5

    # HWC -> CHW
    image = T.from_numpy(image).permute(2, 0, 1)

    return image


def collect_episodes(env, num_episodes, episode_length, replay_buffer):
    """
    Collect episodes using a random policy and store them in the replay buffer.

    Parameters
    ----------
    env : gym.Env
        The environment to collect episodes from.
    num_episodes : int
        The number of episodes to collect.

    """

    for episode in range(num_episodes):

        state, info = env.reset()

        if IMAGE_OBSERVATIONS:
            state = process_image(env.render())

        done = False
        total_reward = 0

        for step in range(episode_length):

            action = env.action_space.sample()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if IMAGE_OBSERVATIONS:
                next_state = process_image(env.render())

            replay_buffer.add(
                state,
                action,
                reward,
                next_state,
                done,
            )

            state = next_state
            total_reward += reward

            if done:
                break

def _denormalize_for_display(image_tensor):
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 0.5) + 0.5
    return np.clip(image, 0.0, 1.0)


@T.no_grad()
def save_world_model_visualization(
    encoder,
    rssm,
    decoder,
    replay_buffer,
    device,
    epoch,
    horizon=8,
):
    encoder.eval()
    rssm.eval()
    decoder.eval()
 
    states, actions, rewards, next_states, dones = replay_buffer.sample()

    #Drop the batch dimension
    states = states[0]
    actions = actions[0]

    embeds = encoder(
        states
    )
 
    if actions.ndim == 1:
        actions = actions.unsqueeze(-1)
 
    # -------------------------------------------------
    # Encode all observations once
    # -------------------------------------------------
 
    # embeds = encoder(states)
 
    # if isinstance(embeds, tuple):
    #     embeds = embeds[0]
 
    # embeds = embeds.flatten(1)
 
    # -------------------------------------------------
    # Teacher-forced rollout
    # -------------------------------------------------
 
    deter = rssm.recurrent_model.input_init(1)
    posterior = rssm.transition_model.input_init(1)
 
    # None at index 0 marks "nothing to display" for that column
    teacher_recons = [None]
    imagined = [None]
    ground_truth = [img.cpu() for img in states]
 
    for t in range(1, horizon + 1):
 
        if t == 1:
            # posterior from first observation, using raw init deter
            # (this deter is never trained on -> t=0 has no meaningful
            # decode, which is why we don't decode/display it at all)
            _, posterior = rssm.represntation_model(
                embeds[0].unsqueeze(0),
                deter,
            )
 
        action = actions[t - 1].unsqueeze(0)
 
        deter = rssm.recurrent_model(
            posterior,
            action,
            deter,
        )
 
        _, prior = rssm.transition_model(deter)
 
        # ----------------------------
        # Imagination uses PRIOR
        # ----------------------------
 
        imagined.append(
            decoder(prior, deter).mean.squeeze(0).cpu()
        )
 
        # Teacher forcing:
        # overwrite latent with posterior
        _, posterior = rssm.represntation_model(
            embeds[t].unsqueeze(0),
            deter,
        )
 
        teacher_recons.append(
            decoder(posterior, deter).mean.squeeze(0).cpu()
        )
    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
 
    cols = horizon + 1
 
    fig, axes = plt.subplots(
        3,
        cols,
        figsize=(2.3 * cols, 7),
    )
 
    row_titles = [
        "Ground Truth",
        "Teacher Forced",
        "Imagined",
    ]
 
    rows = [
        ground_truth,
        teacher_recons,
        imagined,
    ]
 
    for r in range(3):
 
        axes[r, 0].set_ylabel(row_titles[r], fontsize=12)
 
        for c in range(cols):
 
            ax = axes[r, c]
 
            frame = rows[r][c]
 
            if frame is not None:
                ax.imshow(_denormalize_for_display(frame))
 
            # keep the axis box (so the blank cell is visibly part of
            # the grid) but strip ticks instead of calling axis("off"),
            # which would also remove the row label on column 0
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
 
    plt.tight_layout()
 
    os.makedirs("reconstructions", exist_ok=True)
 
    plt.savefig(
        f"reconstructions/world_model_epoch_{epoch:03d}.png"
    )
 
    plt.close()
 
    encoder.train()
    rssm.train()
    decoder.train()


def training_iteration(
    encoder,
    rssm,
    decoder,
    reward_model,
    continue_model,
    optimizer,
    replay_buffer,
    config,
    num_epochs=100,
    batch_size=256,
    horizon=16,
    kl_scale=0.01,
):

    free_nats = config.parameters.dreamer.free_nats

    for epoch in range(num_epochs):

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        # Collect a batch of sequences from the dataset
        states, actions, rewards, next_states, dones = replay_buffer.sample()
        # States: [B, T, C, H, W], Actions: [B, T-1, action_dim]
        B = states.shape[0]
        horizon = states.shape[1]

        # ----------------------------------------
        # Encode whole sequence once
        # ----------------------------------------

        embeds = encoder(
            states.reshape(B * horizon, *states.shape[2:])
        )

        embeds = embeds.flatten(1)
        embeds = embeds.view(B, horizon, -1)

        # ----------------------------------------
        # Initial RSSM state
        # ----------------------------------------

        deter = rssm.recurrent_model.input_init(B)
        posterior = rssm.transition_model.input_init(B)

        priors = []
        posteriors = []

        prior_dists = []
        posterior_dists = []

        deters = []

        # ----------------------------------------
        # RSSM rollout
        # ----------------------------------------

        posterior_dist, posterior = rssm.represntation_model(
                embeds[:, 0],
                deter,
            )

        for t in range(1, horizon):

            action = actions[:, t-1]

            if action.ndim == 1:
                action = action.unsqueeze(-1)

            deter = rssm.recurrent_model(
                posterior,
                action,
                deter,
            )

            prior_dist, prior = rssm.transition_model(deter)

            posterior_dist, posterior = rssm.represntation_model(
                embeds[:, t],
                deter,
            )

            priors.append(prior)
            posteriors.append(posterior)

            prior_dists.append(prior_dist)
            posterior_dists.append(posterior_dist)

            deters.append(deter)

        priors = T.stack(priors, dim=1)
        posteriors = T.stack(posteriors, dim=1)
        deters = T.stack(deters, dim=1)

        # ----------------------------------------
        # Decode all states at once
        # ----------------------------------------

        B2, T2 = posteriors.shape[:2]

        recon_dist = decoder(
            posteriors.reshape(B2 * T2, -1),
            deters.reshape(B2 * T2, -1),
        )

        # recon_loss = -recon_dist.log_prob(
        #     states[:, 1:].reshape(B2 * T2, *states.shape[2:])
        # ).mean()

        n_pixels = states.shape[2] * states.shape[3] * states.shape[4]  # C * H * W

        recon_loss = -recon_dist.log_prob(
            states[:, 1:].reshape(B2 * T2, *states.shape[2:])
        ).mean() / n_pixels
        # recon_loss = -recon_dist.log_prob(
        #     states[:, 1:].reshape(B2 * T2, *states.shape[2:])
        # ).mean()

        # recon_loss = -recon_dist.log_prob(...).mean() / math.prod(self.observation_shape)

        # ----------------------------------------
        # KL loss
        # ----------------------------------------

        kl = []

        for post_dist, prior_dist in zip(
            posterior_dists,
            prior_dists,
        ):

            k = T.distributions.kl.kl_divergence(
                post_dist,
                prior_dist,
            )

            kl.append(k)

        kl = T.stack(kl, dim=1)

        # kl_loss = kl.mean()

        # kl_loss = T.maximum(
        #     kl_loss,
        #     T.tensor(
        #         free_nats,
        #         device=device,
        #     ),
        # )
        kl_loss = T.maximum(
            kl,
            T.tensor(
                free_nats,
                device=DEVICE,
            ),
        )

        kl_loss = kl_loss.mean()

        loss = recon_loss + kl_scale * kl_loss

        optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(
            list(encoder.parameters())
            + list(rssm.parameters())
            + list(decoder.parameters()),
            config.parameters.dreamer.clip_grad,
            norm_type=config.parameters.dreamer.grad_norm_type,
        )

        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss:.4f} | Recon: {total_recon:.4f} | KL: {total_kl:.4f}")


def main():

    print("Testing the Dreamer model components...")

    # Load the configuration
    config_file = "lander.yml"
    config = load_config(config_file)


    # env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    env = gym.make("Walker2d-v5", render_mode="rgb_array")

    action_dim = env.action_space.shape[0]

    # Models
    encoder = Encoder(observation_shape=(3, 64, 64), config=config).to(DEVICE)
    rssm = RSSM(action_size=action_dim, config=config).to(DEVICE)
    decoder = Decoder(observation_shape=(3, 64, 64), config=config).to(DEVICE)
    reward_model = RewardModel(config=config).to(DEVICE)
    continue_model = ContinueModel(config=config).to(DEVICE)

    encoder.train()
    rssm.train()
    decoder.train()
    reward_model.train()
    continue_model.train()

    optimizer = T.optim.Adam(
        list(encoder.parameters())
        + list(rssm.parameters())
        + list(decoder.parameters())
        + list(reward_model.parameters())
        + list(continue_model.parameters()),
        lr=1e-3,
    )

    # Replay buffer
    replay_buffer = ReplayBuffer(
        state_dim=(3, 64, 64),
        action_dim=action_dim,
        image_observations=IMAGE_OBSERVATIONS,
        capacity=100000,
        horizon=16,
        batch_size=256,
        device=DEVICE,
    )

    NUM_TRAINING_ROUNDS = 50
    VISUALIZE_EVERY = 5

    for i in range(NUM_TRAINING_ROUNDS):
        print(f"\nTraining round {i+1}/{NUM_TRAINING_ROUNDS}...")

        # Collect play data 
        collect_episodes(
            env=env,
            num_episodes=5,
            episode_length=1000,
            replay_buffer=replay_buffer,
        )

        training_iteration(encoder, rssm, decoder, reward_model, continue_model, optimizer, replay_buffer, config,
                             num_epochs=10, batch_size=256, horizon=16, kl_scale=0.01)


        # Every so many iterations - evaluate the model by visualizing reconstructions and imagined trajectories

        if i % VISUALIZE_EVERY == 0:
            save_world_model_visualization(encoder, rssm, decoder, replay_buffer, DEVICE, i, horizon=8)


if __name__ == "__main__":
    main()
