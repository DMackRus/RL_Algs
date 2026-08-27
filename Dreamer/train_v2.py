"""
Full Dreamer training script.

Assumes your models.py now contains: RSSM, RewardModel, ContinueModel, Actor, Critic
and encoder_decoder.py contains: Encoder, Decoder

NOTE on naming: your RSSM used `represntation_model` (typo) in the draft you
shared. This script calls it as `representation_model` (correct spelling).
If your models.py still has the typo, either rename the attribute in models.py
or do a find/replace of `representation_model` -> `represntation_model` below.

DynamicInfos and compute_lambda_values are defined inline here rather than
assumed to live in utils.py -- move them there if you prefer, just update
the import.
"""

import os
from types import SimpleNamespace

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import gymnasium as gym

from models import RSSM, RewardModel, ContinueModel, Actor, Critic
from encoder_decoder import Encoder, Decoder
from utils import load_config
from replay_buffer import ReplayBuffer

DEVICE = "cuda" if T.cuda.is_available() else "cpu"
IMAGE_OBSERVATIONS = True
# 0.5ln(2pi) = 0.9189385332
RECONSTRUCTION_LOSS_OFFSET = 0.9189385332


# =====================================================================
# Small utilities
# =====================================================================

class DynamicInfos:
    """Accumulates per-timestep tensors during a rollout, then stacks them."""

    def __init__(self, device):
        self.device = device
        self.data = {}

    def append(self, **kwargs):
        for k, v in kwargs.items():
            self.data.setdefault(k, []).append(v)

    def get_stacked(self, dim=1):
        stacked = {k: T.stack(v, dim=dim) for k, v in self.data.items()}
        self.data = {}
        return SimpleNamespace(**stacked)


def compute_lambda_values(rewards, values, continues, horizon_length, device, lambda_):
    """
    TD(lambda) returns for imagined rollouts.

    rewards, values, continues: [B, H], all aligned so index t is the
    prediction *after* taking the t-th imagined action.
    Returns lambda_values: [B, H-1]
    """
    rewards = rewards[:, :-1]
    continues = continues[:, :-1]
    next_values = values[:, 1:]

    last = next_values[:, -1]
    inputs = rewards + continues * next_values * (1 - lambda_)

    outputs = []
    for t in reversed(range(horizon_length - 1)):
        last = inputs[:, t] + continues[:, t] * lambda_ * last
        outputs.append(last)

    outputs = list(reversed(outputs))
    return T.stack(outputs, dim=1)


def process_image(image):
    image = cv2.resize(image, (64, 64))
    image = (image.astype("float32") / 255.0 - 0.5) / 0.5
    image = T.from_numpy(image).permute(2, 0, 1)
    return image


def _denormalize_for_display(image_tensor):
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 0.5) + 0.5
    return np.clip(image, 0.0, 1.0)


# =====================================================================
# Dreamer agent
# =====================================================================

class Dreamer:
    def __init__(self, action_size, config, device):
        self.device = device
        self.action_size = action_size
        self.config = config.parameters.dreamer

        self.encoder = Encoder(observation_shape=(3, 64, 64), config=config).to(device)
        self.decoder = Decoder(observation_shape=(3, 64, 64), config=config).to(device)
        self.rssm = RSSM(action_size=action_size, config=config).to(device)
        self.reward_predictor = RewardModel(config=config).to(device)
        self.continue_predictor = ContinueModel(config=config).to(device)
        self.actor = Actor(
            discrete_action_bool=False, action_size=action_size, config=config
        ).to(device)
        self.critic = Critic(config=config).to(device)

        self.model_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.rssm.parameters())
            + list(self.reward_predictor.parameters())
            + list(self.continue_predictor.parameters())
        )

        self.model_optimizer = T.optim.Adam(
            self.model_params, lr=self.config.model_learning_rate
        )
        self.actor_optimizer = T.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.critic_optimizer = T.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_learning_rate
        )

        self.continue_criterion = nn.BCELoss()

        self.dynamic_learning_infos = DynamicInfos(device)
        self.behavior_learning_infos = DynamicInfos(device)

        self.num_total_episode = 0

    # -----------------------------------------------------------------
    # Environment interaction (real env, uses the learned actor)
    # -----------------------------------------------------------------

    @T.no_grad()
    def environment_interaction(self, env, num_episodes, replay_buffer,
                                 episode_length=1000, train=True,
                                 explore_noise=0.3):
        scores = []
        episode_lengths = []

        for _ in range(num_episodes):
            state, info = env.reset()
            if IMAGE_OBSERVATIONS:
                state = process_image(env.render())

            deterministic = self.rssm.recurrent_model.input_init(1)
            posterior = self.rssm.transition_model.input_init(1)
            action = T.zeros(1, self.action_size, device=self.device)

            score = 0.0
            for step in range(episode_length):
                embed = self.encoder(state.unsqueeze(0).to(self.device))
                embed = embed.flatten(1)

                deterministic = self.rssm.recurrent_model(posterior, action, deterministic)
                _, posterior = self.rssm.representation_model(embed, deterministic)

                action = self.actor(posterior, deterministic)
                if train:
                    action = action + explore_noise * T.randn_like(action)
                    action = action.clamp(-1.0, 1.0)

                action_np = action.squeeze(0).cpu().numpy()
                next_state, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated

                if IMAGE_OBSERVATIONS:
                    next_state_img = process_image(env.render())
                else:
                    next_state_img = next_state

                if train:
                    # store `terminated` separately so bootstrapping / continue
                    # targets are never contaminated by truncation
                    replay_buffer.add(state, action_np, reward, next_state_img, terminated)

                score += reward
                state = next_state_img

                if done:
                    break
                
            episode_lengths.append(step + 1)

            scores.append(score)
            if train:
                self.num_total_episode += 1

        # print(f"Environment interaction: {num_episodes} episodes | "
        #       f"avg score: {np.mean(scores):.2f} | "
        #       f"avg length: {np.mean(episode_lengths):.2f}")

        return float(np.mean(scores)) if scores else 0.0

    # Just learn the encoder decoder.
    def reconstruction_learning(self, states, actions):
        # states: [B, T, C, H, W]  actions: [B, T-1, A]
        B, horizon = states.shape[0], states.shape[1]

        embeds = self.encoder(states.reshape(B * horizon, *states.shape[2:]))
        embeds = embeds.flatten(1).view(B, horizon, -1)

        deterministic = self.rssm.recurrent_model.input_init(B)
        posterior = self.rssm.transition_model.input_init(B)

        _, posterior = self.rssm.representation_model(embeds[:, 0], deterministic)

        for t in range(1, horizon):
            action = actions[:, t - 1]
            if action.ndim == 1:
                action = action.unsqueeze(-1)

            deterministic = self.rssm.recurrent_model(posterior, action, deterministic)
            # prior_dist, prior = self.rssm.transition_model(deterministic)
            posterior_dist, posterior = self.rssm.representation_model(
                embeds[:, t], deterministic
            )

            self.dynamic_learning_infos.append(
                # priors=prior,
                # prior_dist_means=prior_dist.mean,
                # prior_dist_stds=prior_dist.scale,
                posteriors=posterior,
                posterior_dist_means=posterior_dist.mean,
                posterior_dist_stds=posterior_dist.scale,
                deterministics=deterministic,
            )

        infos = self.dynamic_learning_infos.get_stacked()
        self._model_reconstruction_update(states, infos)
        return infos.posteriors.detach(), infos.deterministics.detach()

    def _model_reconstruction_update(self, states, info):
        recon_dist = self.decoder(info.posteriors, info.deterministics)
        n_pixels = states.shape[2] * states.shape[3] * states.shape[4]
        recon_log_prob = recon_dist.log_prob(states[:, 1:]) / n_pixels

        model_loss = (
            - recon_log_prob.mean()
        )

        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model_params, self.config.clip_grad, norm_type=self.config.grad_norm_type
        )
        self.model_optimizer.step()

        print(f"Model loss: {model_loss.item():.4f} | "
              f"Recon loss: {-recon_log_prob.mean().item():.4f}")

        return {
            "model_loss": model_loss.item()
        }

    # -----------------------------------------------------------------
    # World model training on a batch of real sequences
    # -----------------------------------------------------------------

    def dynamic_learning(self, states, actions, rewards, terminateds):
        # states: [B, T, C, H, W]  actions: [B, T-1, A]
        B, horizon = states.shape[0], states.shape[1]

        embeds = self.encoder(states.reshape(B * horizon, *states.shape[2:]))
        embeds = embeds.flatten(1).view(B, horizon, -1)

        deterministic = self.rssm.recurrent_model.input_init(B)
        posterior = self.rssm.transition_model.input_init(B)

        _, posterior = self.rssm.representation_model(embeds[:, 0], deterministic)

        for t in range(1, horizon):
            action = actions[:, t - 1]
            if action.ndim == 1:
                action = action.unsqueeze(-1)

            deterministic = self.rssm.recurrent_model(posterior, action, deterministic)
            prior_dist, prior = self.rssm.transition_model(deterministic)
            posterior_dist, posterior = self.rssm.representation_model(
                embeds[:, t], deterministic
            )

            self.dynamic_learning_infos.append(
                priors=prior,
                prior_dist_means=prior_dist.mean,
                prior_dist_stds=prior_dist.scale,
                posteriors=posterior,
                posterior_dist_means=posterior_dist.mean,
                posterior_dist_stds=posterior_dist.scale,
                deterministics=deterministic,
            )

        infos = self.dynamic_learning_infos.get_stacked()
        self._model_update(states, rewards, terminateds, infos)
        return infos.posteriors.detach(), infos.deterministics.detach()

    def _model_update(self, states, rewards, terminateds, info):
        recon_dist = self.decoder(info.posteriors, info.deterministics)
        n_pixels = states.shape[2] * states.shape[3] * states.shape[4]
        recon_log_prob = recon_dist.log_prob(states[:, 1:]) / n_pixels

        reward_dist = self.reward_predictor(info.posteriors, info.deterministics)
        # reward_log_prob = reward_dist.log_prob(rewards[:, 1:])
        reward_log_prob = reward_dist.log_prob(rewards[:, 1:].unsqueeze(-1))

        continue_dist = self.continue_predictor(info.posteriors, info.deterministics)
        # target is "did the episode actually end (terminated)", not truncated
        continue_target = 1.0 - terminateds[:, 1:].float()
        continue_loss = self.continue_criterion(continue_dist.probs.squeeze(-1), continue_target)

        prior_dist = T.distributions.Normal(info.prior_dist_means, info.prior_dist_stds)
        posterior_dist = T.distributions.Normal(
            info.posterior_dist_means, info.posterior_dist_stds
        )
        kl = T.distributions.kl.kl_divergence(posterior_dist, prior_dist).sum(-1)
        kl = T.maximum(kl, T.tensor(self.config.free_nats, device=self.device))
        kl_loss = kl.mean()

        model_loss = (
            self.config.kl_divergence_scale * kl_loss
            - recon_log_prob.mean()
            - reward_log_prob.mean()
            + continue_loss
        )

        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model_params, self.config.clip_grad, norm_type=self.config.grad_norm_type
        )
        self.model_optimizer.step()

        # print(f"Model loss: {model_loss.item():.4f} | "
        #       f"Recon loss: {-recon_log_prob.mean().item():.4f} | "
        #       f"Reward loss: {-reward_log_prob.mean().item():.4f} | "
        #       f"Continue loss: {continue_loss.item():.4f} | "
        #       f"KL loss: {kl_loss.item():.4f}")

        # print(f"Model loss: {model_loss.item():.4f} | "
        #       f"Recon loss: {-recon_log_prob.mean().item():.4f} | "
        #       f"Reward loss: {-reward_log_prob.mean().item():.4f} | "
        #       f"KL loss: {kl_loss.item():.4f}")

        return {
            "model_loss": model_loss.item(),
            "recon_loss": -recon_log_prob.mean().item(),
            "reward_loss": -reward_log_prob.mean().item(),
            "continue_loss": continue_loss.item(),
            "kl_loss": kl_loss.item(),
        }

    # -----------------------------------------------------------------
    # Actor-critic training entirely inside imagined rollouts
    # -----------------------------------------------------------------

    def behavior_learning(self, posteriors, deterministics):
        state = posteriors.reshape(-1, self.config.stochastic_size)
        deterministic = deterministics.reshape(-1, self.config.deterministic_size)

        for t in range(self.config.horizon_length):
            action = self.actor(state, deterministic)
            deterministic = self.rssm.recurrent_model(state, action, deterministic)
            _, state = self.rssm.transition_model(deterministic)
            self.behavior_learning_infos.append(priors=state, deterministics=deterministic)

        return self._agent_update(self.behavior_learning_infos.get_stacked())

    def _agent_update(self, info):
        predicted_rewards = self.reward_predictor(info.priors, info.deterministics).mean
        values = self.critic(info.priors, info.deterministics).mean
        continues = self.continue_predictor(info.priors, info.deterministics).mean

        lambda_values = compute_lambda_values(
            predicted_rewards,
            values,
            continues,
            self.config.horizon_length,
            self.device,
            self.config.lambda_,
        )

        actor_loss = -lambda_values.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.clip_grad, norm_type=self.config.grad_norm_type
        )
        self.actor_optimizer.step()

        value_dist = self.critic(
            info.priors.detach()[:, :-1], info.deterministics.detach()[:, :-1]
        )
        critic_loss = -value_dist.log_prob(lambda_values.detach()).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.clip_grad, norm_type=self.config.grad_norm_type
        )
        self.critic_optimizer.step()

        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

    # Save reconstructed images and imagined rollouts for performance visualization
    @T.no_grad()
    def save_world_model_visualization(self, replay_buffer, epoch, horizon=8):
        self.encoder.eval(); self.rssm.eval(); self.decoder.eval()

        states, actions, rewards, next_states, terminateds = replay_buffer.sample(start_index=0)
        states, actions = states[0], actions[0]
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)

        embeds = self.encoder(states.to(self.device))

        deterministic = self.rssm.recurrent_model.input_init(1)
        posterior = self.rssm.transition_model.input_init(1)

        teacher_recons, imagined = [None], [None]
        ground_truth = [img.cpu() for img in states]

        for t in range(1, horizon + 1):
            if t == 1:
                # Encode the first frame to get the initial posterior and deterministic state
                # _, posterior = self.rssm.representation_model(
                #     embeds[0].unsqueeze(0), deterministic
                # )
                posterior_dist, _ = self.rssm.representation_model(
                    embeds[0].unsqueeze(0), deterministic
                )

                # print(
                #     f"t={t}: "
                #     f"mean std = {posterior_dist.scale.mean():.3f}, "
                #     f"max std = {posterior_dist.scale.max():.3f}"
                # )

                posterior = posterior_dist.mean

            action = actions[t - 1].unsqueeze(0).to(self.device)
            deterministic = self.rssm.recurrent_model(posterior, action, deterministic)


            # _, prior = self.rssm.transition_model(deterministic)
            prior_dist, _ = self.rssm.transition_model(deterministic)
            prior = prior_dist.mean

            imagined.append(self.decoder(prior, deterministic).mean.squeeze(0).cpu())

            # Teacher-forced reconstruction of the next frame using the actual embedding
            _, posterior = self.rssm.representation_model(
                embeds[t].unsqueeze(0), deterministic
            )
            teacher_recons.append(self.decoder(posterior, deterministic).mean.squeeze(0).cpu())

        cols = horizon + 1
        fig, axes = plt.subplots(3, cols, figsize=(2.3 * cols, 7))
        rows = [ground_truth, teacher_recons, imagined]
        row_titles = ["Ground Truth", "Teacher Forced", "Imagined"]

        for r in range(3):
            axes[r, 0].set_ylabel(row_titles[r], fontsize=12)
            for c in range(cols):
                ax = axes[r, c]
                frame = rows[r][c]
                if frame is not None:
                    ax.imshow(_denormalize_for_display(frame))
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

        plt.tight_layout()
        os.makedirs("reconstructions", exist_ok=True)
        plt.savefig(f"reconstructions/world_model_epoch_{epoch:03d}.png")
        plt.close()

        self.encoder.train(); self.rssm.train(); self.decoder.train()


# =====================================================================
# Main training loop
# =====================================================================

def main():
    config_file = "lander.yml"
    config = load_config(config_file)

    # env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    # env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    env = gym.make("Walker2d-v5", render_mode="rgb_array", terminate_when_unhealthy=False)
    action_size = env.action_space.shape[0]

    agent = Dreamer(action_size=action_size, config=config, device=DEVICE)

    replay_buffer = ReplayBuffer(
        state_dim=(3, 64, 64),
        action_dim=action_size,
        image_observations=IMAGE_OBSERVATIONS,
        capacity=100000,
        horizon=config.parameters.dreamer.batch_length,
        batch_size=config.parameters.dreamer.batch_size,
        device=DEVICE,
    )

    dreamer_cfg = config.parameters.dreamer

    # seed the buffer with a few random/noisy episodes before any training
    print("Seeding replay buffer...")
    agent.environment_interaction(
        env, dreamer_cfg.seed_episodes, replay_buffer,
        train=True, explore_noise=1.0,
    )

    NUM_TRAIN_ITERATIONS = dreamer_cfg.train_iterations
    VISUALIZE_EVERY = 5

    # Purely encoder-decoder reconstruction learning
    # for iteration in range(NUM_TRAIN_ITERATIONS):
    #     for _ in range(dreamer_cfg.collect_interval):
    #         states, actions, rewards, next_states, terminateds = replay_buffer.sample()
    #         # print(f"States shape: {states.shape}, actions shape: {actions.shape}")

    #         agent.reconstruction_learning(states, actions)

    #     if iteration % VISUALIZE_EVERY == 0:
    #         agent.save_world_model_visualization(replay_buffer, iteration, horizon=8)

    for iteration in range(NUM_TRAIN_ITERATIONS):
        for _ in range(dreamer_cfg.collect_interval):
            states, actions, rewards, next_states, terminateds = replay_buffer.sample()

            posteriors, deterministics = agent.dynamic_learning(
                states, actions, rewards, terminateds
            )
            agent.behavior_learning(posteriors, deterministics)

        train_score = agent.environment_interaction(
            env, dreamer_cfg.num_interaction_episodes, replay_buffer,
            train=True, explore_noise=0.3,
        )
        eval_score = agent.environment_interaction(
            env, dreamer_cfg.num_evaluate, replay_buffer,
            train=False,
        )

        print(f"Iter {iteration+1}/{NUM_TRAIN_ITERATIONS} | "
              f"train score: {train_score:.1f} | eval score: {eval_score:.1f}")

        if iteration % VISUALIZE_EVERY == 0:
            agent.save_world_model_visualization(replay_buffer, iteration, horizon=8)


if __name__ == "__main__":
    main()