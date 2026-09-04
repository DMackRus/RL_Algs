import numpy as np
import torch as T

class LatentPlanner:
    """
    Shared machinery for sampling-based planners that optimise an action
    sequence by rolling candidates out through the TOLD latent model
    (see MPPISampler, CEMPlanner).
    """

    def __init__(
        self,
        config,
        model,
    ):

        self.config = config
        self.device = config["device"]
        self.model = model

        # TODO: hardcoded for now
        self.lower_action_bound = -1.0
        self.upper_action_bound = 1.0

        self.horizon = config["horizon"]
        self.action_dim = config["action_dim"]
        self.gamma = config["gamma"]

        # Std used when sampling actions from the learned policy (policy-seeded
        # "mixture" trajectories and the terminal value bootstrap).
        self.min_std = config["min_std"]

        # Hardcoded for now, planner always only advance by a single timestep.
        k = 1
        self.dt = k * float(config["dt_base"]) if self.model.cond_dt else None

        # Warm-started nominal / mean action sequence, carried across time steps.
        self.nominal_actions = T.zeros(
            self.horizon,
            self.action_dim,
            device=self.device,
        )

    def reset(self):
        """Clear the warm-started plan. Called at the start of an episode (t0)."""
        self.nominal_actions = T.zeros(
            self.horizon,
            self.action_dim,
            device=self.device,
        )

    def _policy_trajectories(self, z0, num_pi_trajs):
        """
        Roll out `num_pi_trajs` action sequences from the learned policy.

        z0 : (1, latent_dim)

        returns:
            (horizon, num_pi_trajs, action_dim), or None when num_pi_trajs == 0
        """

        if num_pi_trajs <= 0:
            return None

        pi_actions = T.empty(
            self.horizon,
            num_pi_trajs,
            self.action_dim,
            device=self.device,
        )
        z = z0.repeat(num_pi_trajs, 1)

        

        for t in range(self.horizon):
            pi_actions[t] = self.model.pi(z, self.min_std)
            
            z, _ = self.model.next(z, pi_actions[t], self.dt)

        return pi_actions

    def _estimate_value(self, z, actions):
        """
        Estimate the discounted return of each candidate action sequence.

        z       : (num_traj, latent_dim)
        actions : (horizon, num_traj, action_dim)

        returns : (num_traj, 1)
        """

        G, discount = 0.0, 1.0

        for t in range(self.horizon):
            z, reward = self.model.next(z, actions[t], self.dt)
            G += discount * reward
            discount *= self.gamma

        # Bootstrap with the terminal value under the learned policy
        terminal_action = self.model.pi(z, self.min_std)
        G += discount * T.min(*self.model.Q(z, terminal_action))

        return G

    def _warm_start(self, plan, z0):
        """
        Store `plan` shifted forward one step as the nominal sequence for the
        next call, seeding the freed final step from the learned policy.
        """
        shifted = T.zeros_like(plan)
        shifted[:-1] = plan[1:].clone()
        shifted[-1] = self.model.pi(z0, self.min_std).squeeze(0)
        self.nominal_actions = shifted

class MPPISampler(LatentPlanner):

    def __init__(
        self,
        config,
        model,
    ):

        super().__init__(config, model)

        self.num_samples = config["MPPI"]["num_candidates"]
        self.noise_std = config["MPPI"]["noise_std"]
        self.temperature = config["MPPI"]["temperature"]
        self.num_opt_iterations = config["MPPI"]["num_opt_iterations"]
        # Fraction of candidate trajectories seeded from the learned policy.
        self.mixture_coef = config["MPPI"].get("mixture_coef", 0.0)

    def plan(self, x0, t0=False):
        """
        MPPI planning in latent space.

        x0 : (1, obs_dim)
        t0 : True on the first step of an episode -> reset the warm-started plan

        returns:
            first action of the optimised plan, and its estimated value
        """

        with T.no_grad():

            if t0:
                self.reset()

            z0 = self.model.h(x0)

            # -------------------------------------------------
            # Policy-seeded (mixture) trajectories
            # -------------------------------------------------

            num_pi_trajs = int(self.mixture_coef * self.num_samples)
            pi_actions = self._policy_trajectories(z0, num_pi_trajs)

            z = z0.repeat(self.num_samples + num_pi_trajs, 1)

            for _ in range(self.num_opt_iterations):

                # -------------------------------------------------
                # Sample noisy action sequences around the nominal plan
                # -------------------------------------------------

                noise = self.noise_std * T.randn(
                    self.horizon,
                    self.num_samples,
                    self.action_dim,
                    device=self.device,
                )
                actions = (self.nominal_actions.unsqueeze(1) + noise).clamp(
                    self.lower_action_bound,
                    self.upper_action_bound,
                )

                if pi_actions is not None:
                    actions = T.cat([actions, pi_actions], dim=1)

                # -------------------------------------------------
                # Evaluate candidates
                # -------------------------------------------------

                returns = self._estimate_value(z, actions).squeeze(1).nan_to_num_(0)

                # -------------------------------------------------
                # MPPI weights + nominal update
                # -------------------------------------------------

                beta = returns.max()
                weights = T.exp((returns - beta) / self.temperature)
                weights /= weights.sum() + 1e-9

                # Weighted average of the candidate sequences (equivalent to
                # nominal += sum_k w_k * noise_k, since sum_k w_k == 1).
                self.nominal_actions = (weights.view(1, -1, 1) * actions).sum(dim=1).clamp(
                    self.lower_action_bound,
                    self.upper_action_bound,
                )

            # -------------------------------------------------
            # Output + warm start
            # -------------------------------------------------

            action = self.nominal_actions[0].clone()
            value = returns.max()

            self._warm_start(self.nominal_actions, z0)

            return action, value

class CEMPlanner(LatentPlanner):

    def __init__(
        self,
        config,
        model,
    ):
        super().__init__(config, model)

        self.num_samples = config["CEM"]["num_candidates"]
        self.num_elites = config["CEM"]["num_elites"]
        # Floor / ceiling on the per-step sampling std of the action distribution.
        self.min_sample_std = config["CEM"]["noise_std"]
        self.max_sample_std = config["CEM"].get("max_noise_std", 2.0)
        self.temperature = config["CEM"]["temperature"]
        self.momentum = config["CEM"]["momentum"]
        self.num_opt_iterations = config["CEM"]["num_opt_iterations"]
        # Fraction of candidate trajectories seeded from the learned policy.
        self.mixture_coef = config["CEM"].get("mixture_coef", 0.0)

    def plan(self, x0, t0=False):
        """
        CEM planning in latent space.

        x0 : (1, obs_dim)
        t0 : True on the first step of an episode -> reset the warm-started plan

        returns:
            first action of the sampled action sequence, and its estimated value
        """

        with T.no_grad():

            if t0:
                self.reset()

            z0 = self.model.h(x0)

            # -------------------------------------------------
            # Policy-seeded (mixture) trajectories
            # -------------------------------------------------

            num_pi_trajs = int(self.mixture_coef * self.num_samples)
            pi_actions = self._policy_trajectories(z0, num_pi_trajs)

            z = z0.repeat(self.num_samples + num_pi_trajs, 1)

            # -------------------------------------------------
            # Initialise the CEM distribution (mean warm started from last plan)
            # -------------------------------------------------

            mean = self.nominal_actions.clone()
            std = self.max_sample_std * T.ones(
                self.horizon,
                self.action_dim,
                device=self.device,
            )

            for _ in range(self.num_opt_iterations):

                # Sample candidate action sequences
                actions = T.clamp(
                    mean.unsqueeze(1)
                    + std.unsqueeze(1)
                    * T.randn(
                        self.horizon,
                        self.num_samples,
                        self.action_dim,
                        device=self.device,
                    ),
                    self.lower_action_bound,
                    self.upper_action_bound,
                )

                if pi_actions is not None:
                    actions = T.cat([actions, pi_actions], dim=1)

                # Evaluate candidates
                value = self._estimate_value(z, actions).nan_to_num_(0)

                # Select elites
                elite_idxs = T.topk(
                    value.squeeze(1),
                    self.num_elites,
                    dim=0,
                ).indices
                elite_value = value[elite_idxs]
                elite_actions = actions[:, elite_idxs]

                # Score-weighted update of the distribution
                max_value = elite_value.max(0)[0]
                score = T.exp(self.temperature * (elite_value - max_value))
                score /= score.sum(0)

                _mean = T.sum(
                    score.unsqueeze(0) * elite_actions, dim=1
                ) / (score.sum(0) + 1e-9)
                _std = T.sqrt(
                    T.sum(
                        score.unsqueeze(0)
                        * (elite_actions - _mean.unsqueeze(1)) ** 2,
                        dim=1,
                    )
                    / (score.sum(0) + 1e-9)
                )
                _std = _std.clamp_(self.min_sample_std, self.max_sample_std)

                mean = self.momentum * mean + (1 - self.momentum) * _mean
                std = _std

            # -------------------------------------------------
            # Sample an action sequence from the elite distribution
            # -------------------------------------------------

            score = score.squeeze(1).cpu().numpy()
            plan = elite_actions[
                :, np.random.choice(np.arange(score.shape[0]), p=score)
            ]

            # -------------------------------------------------
            # Output + warm start
            # -------------------------------------------------

            action = plan[0].clone()
            value = elite_value.max()

            self._warm_start(mean, z0)

            return action, value


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
