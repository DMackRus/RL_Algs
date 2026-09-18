"""
LeWorldModel (LeWM), state-observation variant.

Reference: Maes, Le Lidec, Scieur, LeCun, Balestriero, "LeWorldModel: Stable
End-to-End Joint-Embedding Predictive Architecture from Pixels" (arXiv 2603.19312),
code https://github.com/lucas-maes/le-wm.

LeWM is a *reward-free* JEPA world model:

  - an encoder  h : obs -> z            (here: an MLP + BatchNorm projector,
                                          replacing the paper's ViT + BN head)
  - a predictor f : (z_t, a_t) -> z_hat_{t+1}   (here: a plain MLP; the paper's
                                          autoregressive transformer collapses
                                          to a Markov step for state inputs)

trained end-to-end - *no* target encoder, *no* EMA, *no* stop-gradient - on

      L = ||f(z_t, a_t) - h(o_{t+1})||^2   +   lambda * SIGReg({z})

where SIGReg (Sketched Isotropic Gaussian Regularizer, verbatim from the
reference) is the single anti-collapse term: it pushes the batch of embeddings
towards an isotropic Gaussian via random 1-D projections and the Epps-Pulley
normality statistic.

There is no reward head, no value function and no policy. At evaluation the
world model stays in the loop for goal-conditioned planning: CEM over action
sequences, minimising the latent distance between the rolled-out endpoint and a
goal embedding z_g = h(o_g). Goal observations come from ``goals.py``.

Training data is collected with *pure exploratory actions* (uniform or a
Brownian random walk) - ``plan()`` only does goal-conditioned planning in
``eval_mode``.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from goals import goal_observation


# ---------------------------------------------------------------------------
# SIGReg - Sketched Isotropic Gaussian Regularizer.
# Reproduced from le-wm/module.py (single-GPU version), unchanged.
# proj: (T, B, D) -> scalar. Averages the Epps-Pulley statistic over T random-
# projection ensembles; drives {z} towards N(0, I).
# ---------------------------------------------------------------------------

class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


# ---------------------------------------------------------------------------
# Model: encoder (h) + latent predictor (f). State observations only.
# ---------------------------------------------------------------------------

class LeWMModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        state_dim = int(cfg["state_dim"])
        latent = int(cfg["latent_dim"])
        act_dim = int(cfg["action_dim"])
        enc_hidden = int(cfg.get("lewm_enc_dim", cfg.get("enc_dim", 256)))
        pred_hidden = int(cfg.get("lewm_mlp_dim", cfg.get("mlp_dim", 512)))

        self._trunk = nn.Sequential(
            nn.Linear(state_dim, enc_hidden), nn.ELU(),
            nn.Linear(enc_hidden, enc_hidden), nn.ELU(),
        )
        # Projector head with BatchNorm (as in the reference's projection head).
        self._proj = nn.Sequential(
            nn.Linear(enc_hidden, latent),
            nn.BatchNorm1d(latent),
        )
        self._predictor = nn.Sequential(
            nn.Linear(latent + act_dim, pred_hidden), nn.ELU(),
            nn.Linear(pred_hidden, pred_hidden), nn.ELU(),
            nn.Linear(pred_hidden, latent),
        )

    def h(self, obs):
        """Encode an observation (..., state_dim) -> latent (..., latent_dim)."""
        lead, d = obs.shape[:-1], obs.shape[-1]
        x = self._trunk(obs.reshape(-1, d))
        z = self._proj(x)
        return z.reshape(*lead, z.shape[-1])

    def next(self, z, a):
        """Predict the next latent, z_hat_{t+1} = f(z_t, a_t)."""
        return self._predictor(torch.cat([z, a], dim=-1))


# ---------------------------------------------------------------------------
# Goal-conditioned CEM planner. Cost = latent distance to the goal embedding.
# No reward / value / policy terms (contrast planners.LatentPlanner).
# ---------------------------------------------------------------------------

class GoalCEMPlanner:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.device = cfg["device"]
        self.horizon = int(cfg["horizon"])
        self.action_dim = int(cfg["action_dim"])

        cem = cfg["CEM"]
        self.num_samples = int(cem["num_candidates"])
        self.num_elites = int(cem["num_elites"])
        self.min_std = float(cem["noise_std"])
        self.max_std = float(cem.get("max_noise_std", 2.0))
        self.temperature = float(cem["temperature"])
        self.momentum = float(cem["momentum"])
        self.iters = int(cem.get("num_opt_iterations", 6))

        # "terminal": match only the rolled-out endpoint to the goal (paper).
        # "dense":    sum the latent distance over every horizon step.
        self.dense = str(cfg.get("goal_cost", "terminal")).lower() == "dense"

        self.goal_z = None
        self.nominal_actions = torch.zeros(self.horizon, self.action_dim, device=self.device)

    def set_goal(self, goal_z):
        self.goal_z = goal_z

    def reset(self):
        self.nominal_actions = torch.zeros(self.horizon, self.action_dim, device=self.device)

    @torch.no_grad()
    def plan(self, x0, t0=False):
        assert self.goal_z is not None, "call set_goal() before planning"
        if t0:
            self.reset()

        z0 = self.model.h(x0)                       # (1, latent)
        mean = self.nominal_actions.clone()
        std = self.max_std * torch.ones(self.horizon, self.action_dim, device=self.device)

        elite_cost = None
        for _ in range(self.iters):
            actions = torch.clamp(
                mean.unsqueeze(1)
                + std.unsqueeze(1)
                * torch.randn(self.horizon, self.num_samples, self.action_dim, device=self.device),
                -1.0, 1.0,
            )

            z = z0.repeat(self.num_samples, 1)
            cost = torch.zeros(self.num_samples, device=self.device)
            for t in range(self.horizon):
                z = self.model.next(z, actions[t])
                if self.dense:
                    cost = cost + ((z - self.goal_z) ** 2).sum(-1)
            if not self.dense:
                cost = ((z - self.goal_z) ** 2).sum(-1)

            elite_idx = torch.topk(-cost, self.num_elites, dim=0).indices
            elite_actions = actions[:, elite_idx]           # (H, E, A)
            elite_cost = cost[elite_idx]                    # (E,)

            # Lower cost -> higher weight (mirror of the value-based CEM update).
            w = torch.softmax(-self.temperature * (elite_cost - elite_cost.min()), dim=0)
            _mean = (w.view(1, -1, 1) * elite_actions).sum(1)
            _std = torch.sqrt(
                (w.view(1, -1, 1) * (elite_actions - _mean.unsqueeze(1)) ** 2).sum(1)
            ).clamp(self.min_std, self.max_std)

            mean = self.momentum * mean + (1 - self.momentum) * _mean
            std = _std

        # Warm-start: shift forward one step, zero-pad the freed final step.
        self.nominal_actions = torch.cat(
            [mean[1:], torch.zeros(1, self.action_dim, device=self.device)], dim=0
        )
        return mean[0].clone(), -elite_cost.min()


# ---------------------------------------------------------------------------
# Agent. Same surface as TDMPC (plan / update / save / load) so train.py can
# drive it, plus set_goal() for the goal-conditioned evaluation.
# ---------------------------------------------------------------------------

class LeWM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda")
        self.horizon = int(cfg["horizon"])
        self.action_dim = int(cfg["action_dim"])

        self.model = LeWMModel(cfg).cuda()
        self.sigreg = SIGReg(
            knots=int(cfg.get("sigreg_knots", 17)),
            num_proj=int(cfg.get("sigreg_num_proj", 1024)),
        ).cuda()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=float(cfg["lr"]))

        self.lambd = float(cfg.get("sigreg_weight", 0.1))
        self.grad_clip_norm = float(cfg.get("grad_clip_norm", 10.0))
        self.prioritized = bool(cfg.get("lewm_prioritized", False))

        self.planner = GoalCEMPlanner(cfg, self.model)

        # Exploratory data collection.
        self.exploration = str(cfg.get("exploration", "uniform")).lower()
        self.expl_std = float(cfg.get("expl_std", 0.3))
        self._walk = np.zeros(self.action_dim, dtype=np.float32)

        self.model.eval()

    # -- goal handling -----------------------------------------------------
    @torch.no_grad()
    def set_goal(self, goal_obs):
        g = torch.as_tensor(goal_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.model.eval()
        self.goal_z = self.model.h(g)
        self.planner.set_goal(self.goal_z)

    # -- checkpoint ------------------------------------------------------
    def state_dict(self):
        return {"model": self.model.state_dict()}

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        self.model.load_state_dict(torch.load(fp)["model"])

    # -- acting ---------------------------------------------------------
    @torch.no_grad()
    def plan(self, obs, eval_mode=False, step=None, t0=True):

        # Seed steps - perform random actions to fill replay buffer initially.
        if step < self.cfg["seed_steps"] and not eval_mode:
            return torch.empty(self.cfg["action_dim"], dtype=torch.float32, device=self.device).uniform_(-1, 1)

        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.planner.plan(obs, t0=t0)
        return action

    # -- learning -----------------------------------------------------
    def update(self, replay_buffer, step):
        obs, next_obses, action, _reward, idxs, weights, _ = replay_buffer.sample()
        H, B = self.horizon, obs.shape[0]

        self.model.train()

        # Observation sequence o_0..o_H  -> embeddings z_0..z_H (joint grad).
        obs_seq = torch.cat([obs.unsqueeze(0), next_obses[:H]], dim=0)   # (H+1, B, D)
        z_all = self.model.h(obs_seq)                                    # (H+1, B, latent)

        # Predicted next latents z_hat_{t+1} = f(z_t, a_t), t = 0..H-1.
        z_pred = torch.stack(
            [self.model.next(z_all[t], action[t]) for t in range(H)], dim=0
        )                                                               # (H, B, latent)

        # No stop-gradient on the target: z_all[1:] keeps its graph.
        pred_loss = F.mse_loss(z_pred, z_all[1:])
        sigreg_loss = self.sigreg(z_all)
        loss = pred_loss + self.lambd * sigreg_loss

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_norm, error_if_nonfinite=False
        )
        self.optim.step()

        if self.prioritized:
            with torch.no_grad():
                per = (z_pred[0] - z_all[1]).pow(2).mean(dim=1, keepdim=True)
            replay_buffer.update_priorities(idxs, per.detach())

        self.model.eval()
        return {
            "pred_loss": float(pred_loss.item()),
            "sigreg_loss": float(sigreg_loss.item()),
            "total_loss": float(loss.item()),
            "grad_norm": float(grad_norm),
        }
