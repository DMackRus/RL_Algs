"""
LeWM adaptive: a temporal hierarchy of LeWorldModels.

A stack of L levels, ordered top (coarsest) -> bottom (finest). Level l steps
its latent forward by k_l base env steps at a time (its "stride") and plans
H_l of those steps ahead. Every level is a full LeWM of its own:

  - state encoder   h_l : obs -> z_l                (MLP + BatchNorm projector)
  - action encoder  g_l : (a_t .. a_{t+k_l-1}) -> e_l   (chunk of k_l raw actions)
  - predictor       f_l : (z_l, e_l) -> z_hat_l at t + k_l

      L_l = ||f_l(h_l(o_t), g_l(a_{t:t+k_l})) - h_l(o_{t+k_l})||^2 + lambda * SIGReg({z_l})

Because each level has its own latent space, an upper level's predicted state
cannot be used as a goal for the level below directly. Each lower level l > 0
therefore also owns a translator T_l : z_{l-1} -> z_l, trained on the same
observations with stop-gradient on both sides (it never shapes the world
models themselves).

Planning (every env step):
  1. Top level: CEM towards the encoded goal observation.
  2. Roll out the top level's mean plan -> predicted waypoints at times
     k_0, 2 k_0, ..., H_0 k_0.
  3. Next level: CEM where the cost is the distance between its predicted
     latents and the translated upper waypoints at the time steps where the
     two levels line up (t * k_l a multiple of k_{l-1}).
  4. Repeat down to the bottom level, whose first action is executed.

CEM samples one action per level step, held constant over that step's k_l
base steps, so each level searches an H_l x action_dim space regardless of k_l.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm import SIGReg


def _level_specs(cfg):
    """Parse cfg["levels"] (top -> bottom) into a list of dicts with stride,
    horizon and the merged CEM settings for that level."""
    specs = []
    for lv in cfg["levels"]:
        cem = dict(cfg["CEM"])
        cem.update(lv.get("CEM", {}))
        specs.append(dict(stride=int(lv["stride"]), horizon=int(lv["horizon"]), CEM=cem))
    assert len(specs) >= 1, "cfg['levels'] needs at least one level"

    # Every lower level's horizon must end exactly on an upper-level waypoint,
    # otherwise it would have no subgoal at its terminal step.
    for up, lo in zip(specs[:-1], specs[1:]):
        span = lo["stride"] * lo["horizon"]
        assert span % up["stride"] == 0, (
            f"lower level span {span} (stride {lo['stride']} x horizon {lo['horizon']}) "
            f"must be a multiple of the upper stride {up['stride']}")
        assert span // up["stride"] <= up["horizon"], (
            f"lower level span {span} reaches past the upper level's horizon "
            f"({up['stride']} x {up['horizon']})")
    return specs


def required_window(cfg):
    """Longest base-step window any level trains on; the replay buffer must keep
    this many steps of margin at the end of each episode."""
    return max(s["stride"] * s["horizon"] for s in _level_specs(cfg))


# ---------------------------------------------------------------------------
# One level of the hierarchy: encoder (h), action encoder (g), predictor (f).
# ---------------------------------------------------------------------------

class LevelModel(nn.Module):
    def __init__(self, cfg, stride):
        super().__init__()
        self.stride = stride
        state_dim = int(cfg["state_dim"])
        latent = int(cfg["latent_dim"])
        act_dim = int(cfg["action_dim"])
        act_latent = int(cfg.get("action_latent_dim", 32))
        enc_hidden = int(cfg.get("enc_dim", 256))
        pred_hidden = int(cfg.get("mlp_dim", 512))

        self._trunk = nn.Sequential(
            nn.Linear(state_dim, enc_hidden), nn.ELU(),
            nn.Linear(enc_hidden, enc_hidden), nn.ELU(),
        )
        self._proj = nn.Sequential(
            nn.Linear(enc_hidden, latent),
            nn.BatchNorm1d(latent),
        )
        self._act_enc = nn.Sequential(
            nn.Linear(stride * act_dim, enc_hidden), nn.ELU(),
            nn.Linear(enc_hidden, act_latent),
        )
        self._predictor = nn.Sequential(
            nn.Linear(latent + act_latent, pred_hidden), nn.ELU(),
            nn.Linear(pred_hidden, pred_hidden), nn.ELU(),
            nn.Linear(pred_hidden, latent),
        )

    def h(self, obs):
        """Encode an observation (..., state_dim) -> latent (..., latent_dim)."""
        lead, d = obs.shape[:-1], obs.shape[-1]
        z = self._proj(self._trunk(obs.reshape(-1, d)))
        return z.reshape(*lead, z.shape[-1])

    def g(self, chunk):
        """Encode an action chunk (..., stride, action_dim) -> (..., action_latent_dim)."""
        return self._act_enc(chunk.flatten(-2))

    def next(self, z, e):
        """Predict the latent one level step (stride base steps) ahead."""
        return self._predictor(torch.cat([z, e], dim=-1))

    def rollout(self, z0, actions):
        """Roll out held actions. z0 (N, latent), actions (H, N, A), one action
        per level step held for `stride` base steps. Returns (H, N, latent)."""
        z, zs = z0, []
        for a in actions:
            chunk = a.unsqueeze(-2).expand(*a.shape[:-1], self.stride, a.shape[-1])
            z = self.next(z, self.g(chunk))
            zs.append(z)
        return torch.stack(zs, dim=0)


def _translator(cfg):
    latent = int(cfg["latent_dim"])
    hidden = int(cfg.get("mlp_dim", 512))
    return nn.Sequential(
        nn.Linear(latent, hidden), nn.ELU(),
        nn.Linear(hidden, hidden), nn.ELU(),
        nn.Linear(hidden, latent),
    )


# ---------------------------------------------------------------------------
# Hierarchical goal-conditioned CEM planner.
# ---------------------------------------------------------------------------

class HierarchicalCEMPlanner:
    def __init__(self, cfg, levels, translators, specs):
        self.cfg = cfg
        self.levels = levels
        self.translators = translators      # translators[l-1]: level l-1 -> level l
        self.specs = specs
        self.device = cfg["device"]
        self.action_dim = int(cfg["action_dim"])

        # Top level: "terminal" matches only the endpoint to the goal, "dense"
        # sums the distance over every step.
        self.dense = str(cfg.get("goal_cost", "terminal")).lower() == "dense"
        # Lower levels: "aligned" matches every step that lines up with an upper
        # waypoint, "terminal" only the last one.
        self.subgoal_cost = str(cfg.get("subgoal_cost", "aligned")).lower()

        # For each lower level, pairs (t, j): its step t (0-indexed) lands on
        # the same base time as upper waypoint j (0-indexed).
        self.aligned = [None]
        for up, lo in zip(specs[:-1], specs[1:]):
            pairs = [(t, (t + 1) * lo["stride"] // up["stride"] - 1)
                     for t in range(lo["horizon"])
                     if ((t + 1) * lo["stride"]) % up["stride"] == 0]
            if self.subgoal_cost == "terminal":
                pairs = pairs[-1:]
            self.aligned.append(pairs)

        self.goal_obs = None
        self.reset()

    def set_goal(self, goal_obs):
        """goal_obs (1, state_dim). Only the top level needs it encoded."""
        self.goal_obs = goal_obs

    def reset(self):
        self.nominal = [torch.zeros(s["horizon"], self.action_dim, device=self.device)
                        for s in self.specs]
        self.since_shift = [0] * len(self.specs)

    def _cem(self, level, spec, nominal, z0, cost_fn):
        cem = spec["CEM"]
        H, N = spec["horizon"], int(cem["num_candidates"])
        num_elites = int(cem["num_elites"])
        min_std, max_std = float(cem["noise_std"]), float(cem.get("max_noise_std", 2.0))
        temperature, momentum = float(cem["temperature"]), float(cem["momentum"])

        mean = nominal.clone()
        std = max_std * torch.ones_like(mean)
        z0 = z0.expand(N, -1)
        elite_cost = None
        for _ in range(int(cem.get("num_opt_iterations", 6))):
            actions = torch.clamp(
                mean.unsqueeze(1)
                + std.unsqueeze(1) * torch.randn(H, N, self.action_dim, device=self.device),
                -1.0, 1.0,
            )
            cost = cost_fn(level.rollout(z0, actions))

            elite_idx = torch.topk(-cost, num_elites, dim=0).indices
            elite_actions = actions[:, elite_idx]
            elite_cost = cost[elite_idx]

            w = torch.softmax(-temperature * (elite_cost - elite_cost.min()), dim=0)
            _mean = (w.view(1, -1, 1) * elite_actions).sum(1)
            _std = torch.sqrt(
                (w.view(1, -1, 1) * (elite_actions - _mean.unsqueeze(1)) ** 2).sum(1)
            ).clamp(min_std, max_std)
            mean = momentum * mean + (1 - momentum) * _mean
            std = _std
        return mean, elite_cost.min()

    @torch.no_grad()
    def plan(self, x0, t0=False):
        assert self.goal_obs is not None, "call set_goal() before planning"
        if t0:
            self.reset()

        waypoints, costs = None, []
        for l, (level, spec) in enumerate(zip(self.levels, self.specs)):
            z0 = level.h(x0)                                     # (1, latent)

            if l == 0:
                goal_z = level.h(self.goal_obs)
                if self.dense:
                    cost_fn = lambda zs, g=goal_z: ((zs - g) ** 2).sum(-1).sum(0)
                else:
                    cost_fn = lambda zs, g=goal_z: ((zs[-1] - g) ** 2).sum(-1)
            else:
                ts, js = zip(*self.aligned[l])
                targets = self.translators[l - 1](waypoints[list(js)])   # (P, latent)
                cost_fn = lambda zs, ts=list(ts), g=targets: \
                    ((zs[ts] - g.unsqueeze(1)) ** 2).sum(-1).sum(0)

            mean, cost = self._cem(level, spec, self.nominal[l], z0, cost_fn)
            costs.append(float(cost))
            action = mean[0].clone()            # kept from the bottom level

            # Waypoints for the level below: this level's mean plan, rolled out.
            waypoints = level.rollout(z0, mean.unsqueeze(1)).squeeze(1)  # (H, latent)

            # Warm start. One env step has elapsed; a level step spans `stride`
            # env steps, so only shift the plan once a full level step has passed.
            self.since_shift[l] += 1
            if self.since_shift[l] >= spec["stride"]:
                self.since_shift[l] = 0
                mean = torch.cat([mean[1:], torch.zeros(1, self.action_dim, device=self.device)], dim=0)
            self.nominal[l] = mean

        return action, costs


# ---------------------------------------------------------------------------
# Agent: same interface as lewm.LeWM (set_goal / plan / update / save / load).
# ---------------------------------------------------------------------------

class LeWMAdaptive:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        self.action_dim = int(cfg["action_dim"])
        self.specs = _level_specs(cfg)

        self.levels = nn.ModuleList(
            [LevelModel(cfg, s["stride"]) for s in self.specs]).to(self.device)
        self.translators = nn.ModuleList(
            [_translator(cfg) for _ in self.specs[1:]]).to(self.device)
        self.sigreg = SIGReg(
            knots=int(cfg.get("sigreg_knots", 17)),
            num_proj=int(cfg.get("sigreg_num_proj", 1024)),
        ).to(self.device)
        self.optim = torch.optim.Adam(
            list(self.levels.parameters()) + list(self.translators.parameters()),
            lr=float(cfg["lr"]))

        self.lambd = float(cfg.get("sigreg_weight", 0.1))
        self.grad_clip_norm = float(cfg.get("grad_clip_norm", 10.0))

        self.planner = HierarchicalCEMPlanner(cfg, self.levels, self.translators, self.specs)
        self._eval()

    def _eval(self):
        self.levels.eval()
        self.translators.eval()

    # -- goal handling -----------------------------------------------------
    @torch.no_grad()
    def set_goal(self, goal_obs):
        g = torch.as_tensor(goal_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.planner.set_goal(g)

    # -- checkpoint ------------------------------------------------------
    def state_dict(self):
        return {"levels": self.levels.state_dict(), "translators": self.translators.state_dict()}

    def save(self, fp):
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        d = torch.load(fp, map_location=self.device)
        self.levels.load_state_dict(d["levels"])
        self.translators.load_state_dict(d["translators"])

    # -- acting ---------------------------------------------------------
    @torch.no_grad()
    def plan(self, obs, eval_mode=False, step=None, t0=True):
        if step < self.cfg["seed_steps"] and not eval_mode:
            return torch.empty(self.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)

        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.planner.plan(obs, t0=t0)
        return action

    # -- learning -----------------------------------------------------
    def update(self, replay_buffer, step):
        self.levels.train()
        self.translators.train()

        metrics, loss = {}, 0.0
        for l, (level, spec) in enumerate(zip(self.levels, self.specs)):
            # Each level gets its own batch at its own stride.
            obs_seq, chunks = replay_buffer.sample_strided(spec["horizon"], spec["stride"])
            z_all = level.h(obs_seq)                                 # (H+1, B, latent)
            z_pred = level.next(z_all[:-1], level.g(chunks))         # (H, B, latent)

            pred_loss = F.mse_loss(z_pred, z_all[1:])
            sigreg_loss = self.sigreg(z_all)
            loss = loss + pred_loss + self.lambd * sigreg_loss
            metrics[f"L{l}_pred_loss"] = float(pred_loss.item())
            metrics[f"L{l}_sigreg_loss"] = float(sigreg_loss.item())

            # Translator from the level above, on the same observations. Both
            # embeddings are detached so it never shapes either world model.
            if l > 0:
                with torch.no_grad():
                    z_up = self.levels[l - 1].h(obs_seq)
                trans_loss = F.mse_loss(self.translators[l - 1](z_up), z_all.detach())
                loss = loss + trans_loss
                metrics[f"L{l}_trans_loss"] = float(trans_loss.item())

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.levels.parameters()) + list(self.translators.parameters()),
            self.grad_clip_norm, error_if_nonfinite=False)
        self.optim.step()

        self._eval()
        metrics["total_loss"] = float(loss.item())
        metrics["grad_norm"] = float(grad_norm)
        return metrics
