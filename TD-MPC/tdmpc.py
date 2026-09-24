import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from planners import PredictiveSampler, MPPISampler, CEMPlanner, CEMPlannerHierarchical, CEMPlannerMultistep, PolicyPlanner

import utils
from utils import set_requires_grad, enc, mlp, q, v, orthogonal_init, NormalizeImg, Flatten, TruncatedNormal

class TOLD(nn.Module):
	"""Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._encoder = enc(cfg)
		self._dynamics = mlp(cfg["latent_dim"]+cfg["action_dim"], cfg["mlp_dim"], cfg["latent_dim"])
		self._reward = mlp(cfg["latent_dim"]+cfg["action_dim"], cfg["mlp_dim"], 1)
		self._pi = mlp(cfg["latent_dim"], cfg["mlp_dim"], cfg["action_dim"])
		self._Q1, self._Q2 = q(cfg), q(cfg)
		# self._V1, self._V2 = v(cfg), v(cfg)
		self.apply(utils.orthogonal_init)
		for m in [self._reward, self._Q1, self._Q2]:
			m[-1].weight.data.fill_(0)
			m[-1].bias.data.fill_(0)

	def track_q_grad(self, enable=True):
		"""Utility function. Enables/disables gradient tracking of Q-networks."""
		for m in [self._Q1, self._Q2]:
			set_requires_grad(m, enable)

	def h(self, obs):
		"""Encodes an observation into its latent representation (h)."""
		return self._encoder(obs)

	def next(self, z, a):
		"""Predicts next latent state (d) and reward (R)."""
		x = torch.cat([z, a], dim=-1)
		return self._dynamics(x), self._reward(x)

	def pi(self, z, std=0):
		"""Samples an action from the learned policy (pi)."""
		mu = torch.tanh(self._pi(z))
		if std > 0:
			std = torch.ones_like(mu) * std
			return TruncatedNormal(mu, std).sample(clip=0.3)
		return mu

	def Q(self, z, a):
		"""Predict state-action value (Q)."""
		x = torch.cat([z, a], dim=-1)
		return self._Q1(x), self._Q2(x)

	def V(self, z):
		"""Predict state value (V)."""
		return self._V1(z), self._V2(z)

class TDMPC():
	"""Implementation of TD-MPC learning + inference."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cuda')
		self.std = utils.linear_schedule(cfg["std_schedule"], 0)
		self.model = TOLD(cfg).cuda()
		self.model_target = deepcopy(self.model)
		self.optim = torch.optim.Adam(self.model.parameters(), lr=float(self.cfg["lr"]))
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=float(self.cfg["lr"]))
		self.aug = utils.RandomShiftsAug(cfg)
		planners = {
			"CEM": CEMPlanner,
			"CEM_hierarchical": CEMPlannerHierarchical,
			"CEM_multistep": CEMPlannerMultistep,
			"MPPI": MPPISampler,
			"policy": PolicyPlanner,
		}
		self.planner = planners[cfg["planner"]](self.cfg, self.model)
		self.model.eval()
		self.model_target.eval()

	def state_dict(self):
		"""Retrieve state dict of TOLD model, including slow-moving target network."""
		return {'model': self.model.state_dict(),
				'model_target': self.model_target.state_dict()}

	def save(self, fp):
		"""Save state dict of TOLD model to filepath."""
		torch.save(self.state_dict(), fp)
	
	def load(self, fp):
		"""Load a saved state dict from filepath into current agent."""
		d = torch.load(fp)
		self.model.load_state_dict(d['model'])
		self.model_target.load_state_dict(d['model_target'])

	@torch.no_grad()
	def plan(self, obs, eval_mode=False, step=None, t0=True):
		"""
		Plan next action using TD-MPC inference.
		obs: raw input observation.
		eval_mode: uniform sampling and action noise is disabled during evaluation.
		step: current time step. determines e.g. planning horizon.
		t0: whether current step is the first step of an episode.
		"""
		# Seed steps - perform random actions to fill replay buffer initially.
		if step < self.cfg["seed_steps"] and not eval_mode:
			return torch.empty(self.cfg["action_dim"], dtype=torch.float32, device=self.device).uniform_(-1, 1)

		obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
		action, _ = self.planner.plan(obs, t0=t0)
		return action

	def update_pi(self, zs):
		"""Update policy using a sequence of latent states."""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)

		# Loss is a weighted sum of Q-values
		pi_loss = 0
		for t,z in enumerate(zs):
			a = self.model.pi(z, self.cfg["min_std"])
			Q = torch.min(*self.model.Q(z, a))
			pi_loss += -Q.mean() * (self.cfg["rho"] ** t)

		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg["grad_clip_norm"], error_if_nonfinite=False)
		self.pi_optim.step()
		self.model.track_q_grad(True)
		return pi_loss.item()

	@torch.no_grad()
	def _td_target(self, next_obs, reward):
		"""Compute the TD-target from a reward and the observation at the next step."""
		next_z = self.model.h(next_obs)
		td_target = reward + self.cfg["discount"] * \
			torch.min(*self.model_target.Q(next_z, self.model.pi(next_z, self.cfg["min_std"])))
		return td_target

	# @torch.no_grad()
	# def _td_target(self, next_obs, reward):
	# 	"""Compute the TD-target from a (k-step) reward and the observation k base
	# 	steps later. reward is the discounted return over the macro-step, so the
	# 	bootstrap term is discounted by gamma**k. k may be a scalar or a per-branch
	# 	(batch,) tensor of strides."""

	# 	next_z = self.model.h(next_obs)
	# 	# discount = self.cfg["discount"] ** k
	# 	discount = self.cfg["discount"]
	# 	if torch.is_tensor(discount):
	# 		discount = discount.to(reward.device, reward.dtype).view(-1, 1)
	# 	td_target = reward + discount * torch.min(*self.model_target.V(next_z))
	# 	return td_target

	def update(self, replay_buffer, step):
		"""Main update function. Corresponds to one iteration of the TOLD model learning."""
		obs, next_obses, action, reward, idxs, weights, _ = replay_buffer.sample()
		self.optim.zero_grad(set_to_none=True)
		self.std = utils.linear_schedule(self.cfg["std_schedule"], step)
		self.model.train()

		# Representation
		z = self.model.h(self.aug(obs))
		zs = [z.detach()]

		consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
		for t in range(self.cfg["horizon"]):

			# Predictions
			Q1, Q2 = self.model.Q(z, action[t])
			# V1, V2 = self.model.V(z)
			z, reward_pred = self.model.next(z, action[t])
			with torch.no_grad():
				next_obs = self.aug(next_obses[t])
				next_z = self.model_target.h(next_obs)
				td_target = self._td_target(next_obs, reward[t])
			zs.append(z.detach())

			# Losses
			rho = (self.cfg["rho"] ** t)
			consistency_loss += rho * torch.mean(utils.mse(z, next_z), dim=1, keepdim=True)
			reward_loss += rho * utils.mse(reward_pred, reward[t])
			value_loss += rho * (utils.mse(Q1, td_target) + utils.mse(Q2, td_target))
			priority_loss += rho * (utils.l1(Q1, td_target) + utils.l1(Q2, td_target))
			# value_loss += rho * (utils.mse(V1, td_target) + utils.mse(V2, td_target))
			# priority_loss += rho * (utils.l1(V1, td_target) + utils.l1(V2, td_target))

		# Optimize model
		total_loss = self.cfg["consistency_loss_weight"] * consistency_loss.clamp(max=1e4) + \
					 self.cfg["reward_loss_weight"] * reward_loss.clamp(max=1e4) + \
					 self.cfg["value_loss_weight"] * value_loss.clamp(max=1e4)
		weighted_loss = (total_loss.squeeze(1) * weights).mean()
		weighted_loss.register_hook(lambda grad: grad * (1/self.cfg["horizon"]))
		weighted_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip_norm"], error_if_nonfinite=False)
		self.optim.step()
		replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

		# Update policy + target network
		pi_loss = self.update_pi(zs)
		# pi_loss = 0.0
		if step % self.cfg["update_freq"] == 0:
			utils.ema(self.model, self.model_target, self.cfg["tau"])

		z_std = torch.std(z, dim=0)

		self.model.eval()
		return {'consistency_loss': float(consistency_loss.mean().item()),
				'reward_loss': float(reward_loss.mean().item()),
				'value_loss': float(value_loss.mean().item()),
				'pi_loss': pi_loss,
				'total_loss': float(total_loss.mean().item()),
				'weighted_loss': float(weighted_loss.mean().item()),
				'grad_norm': float(grad_norm),
				'z_std': float(z_std.mean().item())}