import numpy as np
import torch as T
from models import RepresentationModel, LatentDynamics, RewardPredictor, ValuePredictor, PolicyModel


class MPPISampler:

    def __init__(
        self,
        config,
        representation_model,
        dynamics_model,
        reward_model,
        value_model,
        policy,
        action_dim,
        device,
    ):

        self.config = config
        self.device = device

        # TODO: hardcoded for now
        self.lower_action_bound = -1.0
        self.upper_action_bound = 1.0

        self.representation_model = representation_model
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.value_model = value_model
        self.policy = policy

        self.num_trajectories = config["MPPI"]["num_candidates"]
        self.horizon = config["horizon"]
        self.noise_std = config["MPPI"]["noise_std"]
        self.temperature = config["MPPI"]["temperature"]
        self.action_dim = action_dim
        self.num_opt_iterations = config["MPPI"]["num_opt_iterations"]
        self.gamma = config["gamma"]

        self.nominal_actions = T.zeros(
            self.horizon,
            self.action_dim,
            device=device,
        )


    # def plan(self, x0):
    #     """
    #     Plan a trajectory using predictive sampling, from an initial state x0.
    #     x0:
    #         (1, obs_dim)
    #     """

    #     self.nominal_actions = self.nominal_actions + self.noise_std * T.randn(self.num_trajectories, self.horizon, self.action_dim, device=self.device)

    #     for i in range(self.num_opt_iterations):

    #         # Costs and weights for eeach trajectory sample
    #         returns = T.zeros(self.num_trajectories, device=self.device)
    #         weights = T.zeros(self.num_trajectories, device=self.device)
    #         with T.no_grad():
    #             # Encode the initial observation into latent space
    #             z = self.representation_model(x0)

    #             # Repeat latent state for all trajectories
    #             z = z.repeat(self.num_trajectories, 1)

    #             for t in range(self.horizon):
    #                 action = self.nominal_actions[:, t]

    #                 reward = self.reward_model(z, action)
    #                 reward = reward.squeeze(-1)

    #                 returns += reward

    #                 z = self.dynamics_model(z, action)

    #             # ------------------------------------------------
    #             # Terminal value estimate
    #             # ------------------------------------------------

    #             final_action = self.policy(z)
    #             value = self.value_model(z, final_action).squeeze(-1)
    #             returns += (self.gamma ** self.horizon) * value   # S_k

    #             # Compute best cost for softmax normalization
    #             best_return = T.max(returns)

    #             weights = T.exp((returns - best_return) / self.temperature)

    #             # Normalized weights
    #             weights /= T.sum(weights)

    #             weights = weights.unsqueeze(-1).unsqueeze(-1)  # Shape: (num_trajectories, 1, 1)

    #             noise = self.noise_std * T.randn(self.num_trajectories, self.horizon, self.action_dim, device=self.device)
    #             # print(f"Noise shape: {noise.shape}, Weights shape: {weights.shape}, Nominal actions shape: {self.nominal_actions.shape}")
    #             # self.nominal_actions = self.nominal_actions + (weights * noise)

    #             self.nominal_actions += T.sum(weights * noise, dim=0)
    #             # print(f"Nominal actions shape: {self.nominal_actions.shape}, Weights shape: {weights.shape}, Noise shape: {noise.shape}")

    #     best_idx = T.argmax(returns)
    #     best_sequence = self.nominal_actions[best_idx]

    #     #Warm start next MPC iteration
    #     copy_nominal_actions = self.nominal_actions.clone()
    #     self.nominal_actions[:-1] = copy_nominal_actions[:-1]

    #     return best_sequence[0], returns[0]  # Return the first action of the best trajectory

    def plan(self, x0):
        """
        MPPI planning.

        x0 : (1, obs_dim)
        """

        with T.no_grad():

            # Encode initial observation
            z0 = self.representation_model(x0)
            for _ in range(self.num_opt_iterations):

                # -------------------------------------------------
                # Sample noisy action sequences
                # -------------------------------------------------

                noise = self.noise_std * T.randn(
                    self.num_trajectories,
                    self.horizon,
                    self.action_dim,
                    device=self.device,
                )

                sampled_actions = (
                    self.nominal_actions.unsqueeze(0)
                    + noise
                )

                sampled_actions = sampled_actions.clamp(
                    self.lower_action_bound,
                    self.upper_action_bound,
                )

                # -------------------------------------------------
                # Rollout trajectories
                # -------------------------------------------------

                z = z0.repeat(self.num_trajectories, 1)

                returns = T.zeros(self.num_trajectories, device=self.device)

                discount = 1.0

                for t in range(self.horizon):

                    action = sampled_actions[:, t]

                    reward = self.reward_model(z, action).squeeze(-1)

                    returns += discount * reward

                    z = self.dynamics_model(z, action)

                    discount *= self.gamma

                # -------------------------------------------------
                # Terminal value
                # -------------------------------------------------
                terminal_action = self.policy(z)
                terminal_value = self.value_model(z, terminal_action).squeeze(-1)
                returns += discount * terminal_value

                # -------------------------------------------------
                # MPPI weights
                # -------------------------------------------------

                beta = returns.max()
                weights = T.exp((returns - beta) / self.temperature)
                weights /= weights.sum()

                # -------------------------------------------------
                # Update nominal sequence
                # -------------------------------------------------

                self.nominal_actions += (weights[:, None, None] * noise).sum(dim=0)

                self.nominal_actions.clamp_(
                    self.lower_action_bound,
                    self.upper_action_bound,
                )

            # -------------------------------------------------
            # First action to execute
            # -------------------------------------------------

            action = self.nominal_actions[0].clone()

            # -------------------------------------------------
            # Warm start
            # -------------------------------------------------
            self.nominal_actions[:-1] = self.nominal_actions[1:].clone()
            self.nominal_actions[-1] = self.policy(z[:1])

            return action, returns.max()

class PredictiveSampler:

    def __init__(
        self,
        config,
        representation_model,
        dynamics_model,
        reward_model,
        value_model,
        policy,
        action_dim,
        device,
    ):

        self.config = config

        # TODO: hardcoded for now
        self.lower_action_bound = -1.0
        self.upper_action_bound = 1.0

        self.representation_model = representation_model
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.value_model = value_model
        self.policy = policy

        self.action_dim = action_dim
        self.horizon = config["horizon"]
        self.num_samples = config["predictive-sampling"]["num_candidates"]
        self.noise_std = config["predictive-sampling"]["noise_std"]
        self.gamma = config["gamma"]
        self.device = device

        # Uninitialised action sequence
        self.nominal_actions = T.zeros(
            self.horizon,
            action_dim,
            device=device,
        )


    def plan(self, x0):

        """
        Plan a trajectory using predictive sampling, from an initial state x0.
        x0:
            (1, obs_dim)

        returns:
            first action of best trajectory
        """

        with T.no_grad():

            # Encode the initial observation into latent space
            z = self.representation_model(x0)

            # ------------------------------------------------
            # Generate nominal trajectory using policy
            # ------------------------------------------------

            # z = z.clone()
            # nominal = []

            # for t in range(self.horizon):

            #     action = self.policy(z)

            #     nominal.append(action.squeeze(0))

            #     z = self.dynamics(
            #         z,
            #         action
            #     )


            # nominal = T.stack(nominal)


            # ------------------------------------------------
            # Sample noisy action sequences
            #
            # Shape:
            # (num_samples, horizon, action_dim)
            # ------------------------------------------------

            actions = (
                self.nominal_actions.unsqueeze(0) + self.noise_std * T.randn(self.num_samples, self.horizon, self.action_dim, device=self.device)
            )
            actions = actions.clamp(self.lower_action_bound, self.upper_action_bound)

            # ------------------------------------------------
            # Repeat latent state for all trajectories
            #
            # z:
            # (num_samples, latent_dim)
            # ------------------------------------------------

            z = z.repeat(self.num_samples, 1)

            returns = T.zeros(self.num_samples, device=self.device)


            # ------------------------------------------------
            # Roll out all trajectories in parallel
            # ------------------------------------------------

            for t in range(self.horizon):
                action = actions[:, t]

                reward = self.gamma ** t * self.reward_model(z, action)

                reward = reward.squeeze(-1)

                returns += reward

                z = self.dynamics_model(z, action)

            # ------------------------------------------------
            # Terminal value estimate
            # ------------------------------------------------

            final_action = self.policy(z)
            value = self.value_model(z, final_action).squeeze(-1)
            returns += (self.gamma ** self.horizon) * value
            
            # ------------------------------------------------
            # Pick best trajectory
            # ------------------------------------------------

            best_idx = T.argmax(returns)
            best_sequence = actions[best_idx]


            # ------------------------------------------------
            # Warm start next MPC iteration
            # ------------------------------------------------

            self.nominal_actions[:-1] = (
                best_sequence[1:]
            )

            # self.nominal_actions[-1] = self.policy(
            #     z0
            # ).squeeze(0)


            return best_sequence[0], returns[best_idx]
