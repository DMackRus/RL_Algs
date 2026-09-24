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

        # Hierarchical planner goal-matching (see _estimate_value /
        # CEMPlannerHierarchical). goal_reward_coef weights the explicit pull
        # toward the sub-goal latent; goal_match_tau is the latent-distance
        # scale of the match weight that blends the coarse plan's value-to-go
        # against the learned terminal Q.
        self.goal_reward_coef = float(config.get("goal_reward_coef", 1.0))
        self.goal_match_tau = float(config.get("goal_match_tau", 1.0))

        if config["training_algorithm"] == "tdmpc":
            self.adaptive_dt = False
        else:
            self.adaptive_dt = True

        # Hardcoded for now, planner always only advance by a single timestep.
        # k = 1
        # self.dt = k * float(config["dt_base"]) if self.model.cond_dt else None

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

    def _policy_trajectories(self, z0, num_pi_trajs, k=1, step_k=None):
        """
        Roll out `num_pi_trajs` action sequences from the learned policy.

        z0 : (1, latent_dim)
        k  : scalar macro-step size for the rollout (dt = k * dt_base when the
             model is dt-conditioned)
        step_k : optional per-horizon-step sequence of macro-step sizes. When
             given it overrides `k` (used by CEMPlannerMultistep).

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
        dt_base = float(self.config["dt_base"]) if self.adaptive_dt else None

        for t in range(self.horizon):
            pi_actions[t] = self.model.pi(z, self.min_std)

            if self.adaptive_dt:
                kt = step_k[t] if step_k is not None else k
                z, _ = self.model.next(z, pi_actions[t], kt * dt_base)
            else:
                z, _ = self.model.next(z, pi_actions[t])

        return pi_actions

    def _estimate_value(self, z, actions, num_timesteps=1, goal_z=None):
        """
        Estimate the discounted return of each candidate action sequence.

        z       : (num_traj, latent_dim)
        actions : (horizon, num_traj, action_dim)
        num_timesteps : macro-step size. Either a scalar k held for the whole
                        horizon (each model step advances k * dt_base seconds;
                        1, i.e. a single base step, when dt is fixed), or a
                        per-horizon-step sequence of k's (CEMPlannerMultistep).
                        Rewards are discounted by gamma**k per model step.
        goal_z  : (1, latent_dim) or None. Sub-goal latent handed down from the
                  coarser stage of the hierarchical planner.

        Terminal bootstrap:
          - goal_z is None (coarsest stage): learned terminal value, min-Q at
            the endpoint under the policy action.
          - goal_z given (finer stages): a terminal cost that pulls the endpoint
            latent onto the coarse waypoint,
            goal_reward_coef * exp(-||z_T - goal_z|| / goal_match_tau).
            No learned-value term here.

        returns : (num_traj, 1)
        """

        G, discount = 0.0, 1.0
        # Debug metrics
        reward_sum, end_value = 0.0, 0.0

        if isinstance(num_timesteps, (list, tuple)):
            k_seq = [int(k) for k in num_timesteps]
        else:
            k_seq = [int(num_timesteps)] * self.horizon
        dt_base = float(self.config["dt_base"]) if self.adaptive_dt else None

        for t in range(self.horizon):
            k = k_seq[t]
            if self.adaptive_dt:
                z, reward = self.model.next(z, actions[t], k * dt_base)
            else:
                z, reward = self.model.next(z, actions[t])
            G += discount * reward
            reward_sum += discount * reward
            discount *= self.gamma ** k

        if goal_z is not None:
            # Finer stages: terminal cost matching the endpoint latent to the
            # waypoint the coarser plan converged to.
            match = T.exp(-T.norm(z - goal_z, dim=1, keepdim=True) / self.goal_match_tau)
            G += discount * self.goal_reward_coef * match
            end_value = discount * self.goal_reward_coef * match
        else:
            #TODO - hardcoded for now
            # V_term = T.min(*self.model.V(z))
            # G += discount * V_term
            # end_value = discount * V_term
            terminal_action = self.model.pi(z, self.min_std)
            q_term = T.min(*self.model.Q(z, terminal_action))
            G += discount * q_term
            end_value = discount * q_term

            # if self.adaptive_dt:
            #     # Coarsest stage: learned terminal value estimate.
            #     V_term = T.min(*self.model.V(z))
            #     G += discount * V_term
            #     end_value = discount * V_term
            # else:
            #     # Coarsest stage: learned terminal value estimate.
            #     terminal_action = self.model.pi(z, self.min_std)
            #     q_term = T.min(*self.model.Q(z, terminal_action))
            #     G += discount * q_term
            #     end_value = discount * q_term

        return G, end_value, reward_sum

    def _reference_rollout(self, z0, action_seq, k):
        """
        Noise-free rollout of a single action sequence.

        z0         : (1, latent_dim)
        action_seq : (horizon, action_dim)
        k          : macro-step size for this rollout

        returns:
            latents : (horizon + 1, latent_dim) - latent at base-times
                      0, k, 2k, ..., k * horizon
        """
        dt = k * float(self.config["dt_base"])
        z = z0.clone()
        latents = [z]
        for t in range(action_seq.shape[0]):
            z, _ = self.model.next(z, action_seq[t].unsqueeze(0), dt)
            latents.append(z)
        return T.cat(latents, dim=0)

    @staticmethod
    def _select_goal(latents, k_prev, k_next, horizon):
        """
        Pick the sub-goal latent for a refined stage: the node of the coarser
        rollout closest to (but not past) the real-time endpoint the refined
        plan can actually reach.

        The refined plan spans k_next * horizon base steps; the coarser rollout
        has nodes at base-times 0, k_prev, 2*k_prev, ... so we take node
        floor(k_next * horizon / k_prev).

        returns : goal_z (1, latent_dim)
        """
        j = (k_next * horizon) // k_prev
        j = int(min(j, latents.shape[0] - 1))
        return latents[j:j + 1]

    # def _warm_start(self, plan, z0):
    #     shifted = T.zeros_like(plan)
    #     shifted[:-1] = plan[1:].clone()

    #     # HARDCODED - test always cloning last control
    #     # shifted[-1] = plan[-1].clone()

    #     if self.adaptive_dt:
    #         # No actor in the V-only model — repeat the last action instead of
    #         # asking a policy network for one.
    #         shifted[-1] = plan[-1].clone()
    #     else:
    #         shifted[-1] = self.model.pi(z0, self.min_std).squeeze(0)
    #     self.nominal_actions = shifted

    def _warm_start(self, plan, z0):
        """
        Store `plan` shifted forward one step as the nominal sequence for the
        next call, seeding the freed final step from the learned policy.
        """
        shifted = T.zeros_like(plan)
        shifted[:-1] = plan[1:].clone()
        shifted[-1] = self.model.pi(z0, self.min_std).squeeze(0)
        self.nominal_actions = shifted

class CEMPlannerHierarchical(LatentPlanner):
    """
    Coarse-to-fine CEM planner.

    A single plan() call runs several CEM iterations whose macro-step size k is
    driven by `k_schedule` (non-increasing, e.g. (4, 4, 2, 2, 1, 1) -> two
    iterations each at dt = 4*dt_base, 2*dt_base, dt_base). Early iterations take
    big steps and see far into the future; later iterations refine the near term.

    Every stage optimises the *same* fixed-length nominal action sequence
    `mean` (horizon, action_dim) - it is never reshaped. What is "passed down"
    the chain from a coarse stage to the next finer one is:

      1. the warm-started mean / std (carried directly), and
      2. a sub-goal latent: the coarse plan is rolled out noise-free and the
         node closest to the finer plan's real-time reach gives a goal latent.

    Terminal bootstrap per stage (see _estimate_value):
      - coarsest stage (no goal yet): the learned terminal value (min-Q).
      - every finer stage: a terminal cost only, pulling the endpoint latent
        onto the sub-goal, goal_reward_coef * exp(-||z_T - goal_z|| / tau).
    """

    def __init__(
        self,
        config,
        model,
    ):
        super().__init__(config, model)

        cem = config["CEM"]
        self.num_samples = cem["num_candidates"]
        self.num_elites = cem["num_elites"]
        # Floor / ceiling on the per-step sampling std of the action distribution.
        self.min_sample_std = cem["noise_std"]
        self.max_sample_std = cem.get("max_noise_std", 2.0)
        self.temperature = cem["temperature"]
        self.momentum = cem["momentum"]
        # Fraction of candidate trajectories seeded from the learned policy.
        self.mixture_coef = cem.get("mixture_coef", 0.0)

        # Coarse-to-fine macro-step schedule: one entry per CEM iteration, each
        # value k meaning "roll candidates out with dt = k * dt_base". Must be
        # non-increasing. Overrides CEM.num_opt_iterations.
        self.k_schedule = [int(k) for k in cem.get("k_schedule", (4, 4, 2, 2, 1, 1))]
        assert all(
            a >= b for a, b in zip(self.k_schedule, self.k_schedule[1:])
        ), f"k_schedule must be non-increasing, got {self.k_schedule}"
        self.num_opt_iterations = len(self.k_schedule)

        # Reset std back to the ceiling whenever k drops, to re-open exploration
        # at the finer (more sensitive) dynamics.
        self.reset_std_on_refine = cem.get("reset_std_on_refine", True)

    def plan(self, x0, t0=False):
        """
        Coarse-to-fine CEM planning in latent space.

        x0 : (1, obs_dim)
        t0 : True on the first step of an episode -> reset the warm-started plan

        returns:
            first action of the sampled action sequence, and its estimated value
        """

        with T.no_grad():

            if t0:
                self.reset()

            z0 = self.model.h(x0)

            # Nominal / distribution carried across every stage (fixed length).
            mean = self.nominal_actions.clone()
            std = self.max_sample_std * T.ones(
                self.horizon,
                self.action_dim,
                device=self.device,
            )

            goal_z = None
            elite_actions = elite_value = score = None

            for i, k in enumerate(self.k_schedule):

                # ---------------------------------------------------------
                # Resolution change: derive the sub-goal from the plan the
                # previous (coarser) stage converged to, and re-open std.
                # ---------------------------------------------------------
                if i > 0 and k != self.k_schedule[i - 1]:
                    k_prev = self.k_schedule[i - 1]
                    ref_latents = self._reference_rollout(z0, mean, k_prev)
                    goal_z = self._select_goal(ref_latents, k_prev, k, self.horizon)
                    if self.reset_std_on_refine:
                        std = self.max_sample_std * T.ones_like(std)

                # ---------------------------------------------------------
                # Policy-seeded (mixture) trajectories. _policy_trajectories
                # rolls out at dt_base, so only mix them in at the k == 1
                # stages where that matches the candidate rollout.
                # ---------------------------------------------------------
                pi_actions = None
                if self.mixture_coef > 0 and k == 1:
                    n_pi = int(self.mixture_coef * self.num_samples)
                    pi_actions = self._policy_trajectories(z0, n_pi)

                n_pi = 0 if pi_actions is None else pi_actions.shape[1]
                z = z0.repeat(self.num_samples + n_pi, 1)

                # ---------------------------------------------------------
                # Sample candidate action sequences around the nominal
                # ---------------------------------------------------------
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

                # ---------------------------------------------------------
                # Evaluate candidates (goal_z is None at the coarsest stage
                # -> falls back to the learned terminal value bootstrap)
                # ---------------------------------------------------------
                value, reward, value_end = self._estimate_value(z, actions, k, goal_z)

                #Debug metrics
                # print(f"total value: {value.mean():.3f} +- {value.std():.3f}")
                # print(f"total reward: {reward.mean():.3f} +- {reward.std():.3f}")
                # print(f"end value: {value_end.mean():.3f} +- {value_end.std():.3f}")      

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
            # Sample an action sequence from the final elite distribution
            # -------------------------------------------------

            score = score.squeeze(1).cpu().numpy()
            plan = elite_actions[
                :, np.random.choice(np.arange(score.shape[0]), p=score)
            ]

            action = plan[0].clone()
            value = elite_value.max()

            self._warm_start(mean, z0)

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
                value, reward, value_end = self._estimate_value(z, actions)

                #Debug metrics
                # print(f"total value: {value.mean():.3f} +- {value.std():.3f}")
                # print(f"total reward: {reward.mean():.3f} +- {reward.std():.3f}")
                # print(f"end value: {value_end.mean():.3f} +- {value_end.std():.3f}")

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

class CEMPlannerMultistep(LatentPlanner):
    """
    CEM planner with a per-horizon-step macro-step schedule (fine -> coarse).

    Like the plain CEMPlanner: a fixed number of CEM iterations, learned value
    function as the terminal bootstrap. The only difference is the rollout -
    horizon step t advances the world model by `step_k_schedule[t]` base env
    steps (dt = step_k_schedule[t] * dt_base) rather than a single fixed step.

    The schedule is non-decreasing, e.g. (1, 1, 2, 2, 4, 4): small steps near
    the start of the plan, where the action actually executed needs resolution,
    and big steps toward the end for cheap long-range lookahead. A horizon-H
    plan then reaches sum(step_k_schedule[:H]) * dt_base seconds ahead while
    only rolling the model H times.

    Every CEM iteration uses the same schedule - contrast CEMPlannerHierarchical,
    which sweeps a single (whole-rollout) k from coarse to fine across its
    iterations.
    """

    def __init__(
        self,
        config,
        model,
    ):
        super().__init__(config, model)

        assert self.adaptive_dt, (
            "CEMPlannerMultistep needs a dt-conditioned model "
            "(training_algorithm != 'tdmpc')"
        )

        cem = config["CEM"]
        self.num_samples = cem["num_candidates"]
        self.num_elites = cem["num_elites"]
        # Floor / ceiling on the per-step sampling std of the action distribution.
        self.min_sample_std = cem["noise_std"]
        self.max_sample_std = cem.get("max_noise_std", 2.0)
        self.temperature = cem["temperature"]
        self.momentum = cem["momentum"]
        self.num_opt_iterations = cem.get("num_opt_iterations", 6)
        # Fraction of candidate trajectories seeded from the learned policy.
        self.mixture_coef = cem.get("mixture_coef", 0.0)

        # One macro-step k per horizon step, non-decreasing. Clipped (if longer)
        # or padded with its last value (if shorter) to exactly `horizon` entries.
        sched = [int(k) for k in cem.get("step_k_schedule", (1, 1, 2, 2, 4, 4))]
        if len(sched) >= self.horizon:
            sched = sched[: self.horizon]
        else:
            sched = sched + [sched[-1]] * (self.horizon - len(sched))
        self.step_k = sched
        assert all(
            a <= b for a, b in zip(self.step_k, self.step_k[1:])
        ), f"step_k_schedule must be non-decreasing, got {self.step_k}"

    def plan(self, x0, t0=False):
        """
        CEM planning in latent space with a fine -> coarse per-step schedule.

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
            # Policy-seeded (mixture) trajectories - rolled with the same
            # per-step schedule as the candidates.
            # -------------------------------------------------

            num_pi_trajs = int(self.mixture_coef * self.num_samples)
            pi_actions = self._policy_trajectories(
                z0, num_pi_trajs, step_k=self.step_k
            )
            n_pi = 0 if pi_actions is None else pi_actions.shape[1]

            z = z0.repeat(self.num_samples + n_pi, 1)

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

                # Evaluate candidates under the fine -> coarse schedule
                value, _, _ = self._estimate_value(z, actions, self.step_k)

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

            action = plan[0].clone()
            value = elite_value.max()

            self._warm_start(mean, z0)

            return action, value

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

                returns, _, _ = self._estimate_value(z, actions)
                returns = returns.squeeze(1).nan_to_num_(0)

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


class PolicyPlanner(LatentPlanner):
    """
    Degenerate "planner" that ignores the dynamics / reward models and just
    queries the learned policy at the current state. Useful as a baseline for
    testing how good pi is on its own, with the same plan() interface as the
    sampling planners.
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        # Sampling std for the policy. 0 -> deterministic (mean action).
        self.policy_std = float(config.get("policy_std", 0.0))

    def plan(self, x0, t0=False):
        """
        x0 : (1, obs_dim)
        t0 : unused (kept for interface parity)

        returns:
            action taken by the policy, and its estimated value (min over the
            two Q heads)
        """
        with T.no_grad():
            z = self.model.h(x0)
            action = self.model.pi(z, self.policy_std)
            value = T.min(*self.model.Q(z, action))
            return action.squeeze(0), value.squeeze(0)


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
