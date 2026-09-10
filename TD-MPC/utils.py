from collections import deque

import cv2
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import re
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

__REDUCE__ = lambda b: 'mean' if b else 'none'

class TruncatedNormal(pyd.Normal):
	"""Utility class implementing the truncated normal distribution."""
	def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
		super().__init__(loc, scale, validate_args=False)
		self.low = low
		self.high = high
		self.eps = eps

	def _clamp(self, x):
		clamped_x = T.clamp(x, self.low + self.eps, self.high - self.eps)
		x = x - x.detach() + clamped_x.detach()
		return x

	def sample(self, clip=None, sample_shape=T.Size()):
		shape = self._extended_shape(sample_shape)
		eps = _standard_normal(shape,
							   dtype=self.loc.dtype,
							   device=self.loc.device)
		eps *= self.scale
		if clip is not None:
			eps = T.clamp(eps, -clip, clip)
		x = self.loc + eps
		return self._clamp(x)

class NormalizeImg(nn.Module):
	"""Normalizes pixel observations to [0,1) range."""
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.)

class Flatten(nn.Module):
	"""Flattens its input to a (batched) vector."""
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x.view(x.size(0), -1)

def enc(cfg):
	"""Returns a TOLD encoder."""
	if cfg["image_observations"]:
		C = int(3*cfg["frame_stack"])
		layers = [NormalizeImg(),
				  nn.Conv2d(C, cfg["num_channels"], 7, stride=2), nn.ReLU(),
				  nn.Conv2d(cfg["num_channels"], cfg["num_channels"], 5, stride=2), nn.ReLU(),
				  nn.Conv2d(cfg["num_channels"], cfg["num_channels"], 3, stride=2), nn.ReLU(),
				  nn.Conv2d(cfg["num_channels"], cfg["num_channels"], 3, stride=2), nn.ReLU()]
		out_shape = _get_out_shape((C, cfg["img_size"], cfg["img_size"]), layers)
		layers.extend([Flatten(), nn.Linear(np.prod(out_shape), cfg["latent_dim"])])
	else:
		layers = [nn.Linear(cfg["state_dim"], cfg["enc_dim"]), nn.ELU(),
				  nn.Linear(cfg["enc_dim"], cfg["latent_dim"])]
	return nn.Sequential(*layers)


def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
	"""Returns an MLP."""
	if isinstance(mlp_dim, int):
		mlp_dim = [mlp_dim, mlp_dim]
	return nn.Sequential(
		nn.Linear(in_dim, mlp_dim[0]), act_fn,
		nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
		nn.Linear(mlp_dim[1], out_dim))

def q(cfg, act_fn=nn.ELU()):
	"""Returns a Q-function that uses Layer Normalization."""
	return nn.Sequential(nn.Linear(cfg["latent_dim"]+cfg["action_dim"], cfg["mlp_dim"]), nn.LayerNorm(cfg["mlp_dim"]), nn.Tanh(),
						 nn.Linear(cfg["mlp_dim"], cfg["mlp_dim"]), nn.ELU(),
						 nn.Linear(cfg["mlp_dim"], 1))

def set_requires_grad(net, value):
	"""Enable/disable gradients for a given (sub)network."""
	for param in net.parameters():
		param.requires_grad_(value)

def orthogonal_init(m):
	"""Orthogonal layer initialization."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if m.bias is not None:
			nn.init.zeros_(m.bias)
	elif isinstance(m, nn.Conv2d):
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data, gain)
		if m.bias is not None:
			nn.init.zeros_(m.bias)

def linear_schedule(schdl, step):
	"""
	Outputs values following a linear decay schedule.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	try:
		return float(schdl)
	except ValueError:
		match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
		if match:
			init, final, duration = [float(g) for g in match.groups()]
			mix = np.clip(step / duration, 0.0, 1.0)
			return (1.0 - mix) * init + mix * final
	raise NotImplementedError(schdl)

def l1(pred, target, reduce=False):
	"""Computes the L1-loss between predictions and targets."""
	return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))

def mse(pred, target, reduce=False):
	"""Computes the MSE loss between predictions and targets."""
	return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


def _get_out_shape(in_shape, layers):
	"""Utility function. Returns the output shape of a network for a given input shape."""
	x = T.randn(*in_shape).unsqueeze(0)
	return (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x).squeeze(0).shape

def symlog(x):
    # return x
    return T.sign(x) * T.log(1 + T.abs(x))

def symexp(x):
    return T.sign(x) * (T.exp(T.abs(x)) - 1)

def make_frame_stacker(k):
    frames = deque(maxlen=k)
    def reset(frame):          # frame: (3, 64, 64)
        for _ in range(k):
            frames.append(frame)
        return T.cat(list(frames), dim=0)   # (3k, 64, 64)
    def push(frame):
        frames.append(frame)
        return T.cat(list(frames), dim=0)
    return reset, push

def ema(m, m_target, tau):
	"""Update slow-moving average of online network (target network) at rate tau."""
	with T.no_grad():
		for p, p_target in zip(m.parameters(), m_target.parameters()):
			p_target.data.lerp_(p.data, tau)

def process_image(image):
    # Resize to 64x64. Keep as uint8 [0, 255] -- normalization to [-1, 1] happens
    # inside RepresentationModel.forward so the replay buffer can store uint8.
    image = cv2.resize(image, (64, 64))

    # HWC -> CHW, still uint8
    image = T.from_numpy(image).permute(2, 0, 1).contiguous()

    return image

class RandomShiftsAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, cfg):
		super().__init__()
		self.pad = int(cfg.img_size/21) if cfg["image_observations"] == True else None

	def forward(self, x):
		if not self.pad:
			return x
		n, c, h, w = x.size()
		assert h == w
		padding = tuple([self.pad] * 4)
		x = F.pad(x, padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = T.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = T.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = T.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)