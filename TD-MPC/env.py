"""
Environment construction for TD-MPC.

Two backends, both handed to the rest of the codebase through the same small
classic-gym interface: an ``observation_space`` / ``action_space`` with a
``.shape``, ``reset() -> obs``, ``step(a) -> (obs, reward, done, info)``, plus
``ep_len``, ``control_timestep()`` and ``render(mode="rgb_array")``.

  - dm_control suite   ->  cfg["benchmark"] == "dmc"        (default)
  - Meta-World          ->  cfg["benchmark"] == "metaworld"

No gymnasium dependency here: the spaces are a local ``Box`` (the code only ever
reads ``.shape``), and the Meta-World backend consumes whatever the ``metaworld``
package returns. There are no standalone gym tasks.
"""

from collections import deque, defaultdict

import numpy as np
import dm_env
from dm_control import suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import specs

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Box:
	"""Minimal stand-in for ``gym.spaces.Box``. The codebase only reads
	``.shape`` (and occasionally ``.low`` / ``.high`` / ``.dtype``)."""

	def __init__(self, low, high, shape, dtype=np.float32):
		shape = tuple(int(x) for x in shape)
		self.low = low if np.ndim(low) else np.full(shape, low, dtype=dtype)
		self.high = high if np.ndim(high) else np.full(shape, high, dtype=dtype)
		self.shape = shape
		self.dtype = dtype


# ---------------------------------------------------------------------------
# dm_control wrappers (operate on the dm_env TimeStep API)
# ---------------------------------------------------------------------------

class ActionRepeatWrapper(dm_env.Environment):
	def __init__(self, env, num_repeats):
		self._env = env
		self._num_repeats = num_repeats

	def step(self, action):
		reward = 0.0
		discount = 1.0
		for i in range(self._num_repeats):
			time_step = self._env.step(action)
			reward += (time_step.reward or 0.0) * discount
			discount *= time_step.discount
			if time_step.last():
				break

		return time_step._replace(reward=reward, discount=discount)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._env.action_spec()

	def reset(self):
		return self._env.reset()

	def __getattr__(self, name):
		return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
	def __init__(self, env, num_frames, pixels_key='pixels'):
		self._env = env
		self._num_frames = num_frames
		self._frames = deque([], maxlen=num_frames)
		self._pixels_key = pixels_key

		wrapped_obs_spec = env.observation_spec()
		assert pixels_key in wrapped_obs_spec

		pixels_shape = wrapped_obs_spec[pixels_key].shape
		if len(pixels_shape) == 4:
			pixels_shape = pixels_shape[1:]
		self._obs_spec = specs.BoundedArray(shape=np.concatenate(
			[[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
											dtype=np.uint8,
											minimum=0,
											maximum=255,
											name='observation')

	def _transform_observation(self, time_step):
		assert len(self._frames) == self._num_frames
		obs = np.concatenate(list(self._frames), axis=0)
		return time_step._replace(observation=obs)

	def _extract_pixels(self, time_step):
		pixels = time_step.observation[self._pixels_key]
		if len(pixels.shape) == 4:
			pixels = pixels[0]
		return pixels.transpose(2, 0, 1).copy()

	def reset(self):
		time_step = self._env.reset()
		pixels = self._extract_pixels(time_step)
		for _ in range(self._num_frames):
			self._frames.append(pixels)
		return self._transform_observation(time_step)

	def step(self, action):
		time_step = self._env.step(action)
		pixels = self._extract_pixels(time_step)
		self._frames.append(pixels)
		return self._transform_observation(time_step)

	def observation_spec(self):
		return self._obs_spec

	def action_spec(self):
		return self._env.action_spec()

	def __getattr__(self, name):
		return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
	def __init__(self, env, dtype):
		self._env = env
		wrapped_action_spec = env.action_spec()
		self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
											   dtype,
											   wrapped_action_spec.minimum,
											   wrapped_action_spec.maximum,
											   'action')

	def step(self, action):
		action = action.astype(self._env.action_spec().dtype)
		return self._env.step(action)

	def observation_spec(self):
		return self._env.observation_spec()

	def action_spec(self):
		return self._action_spec

	def reset(self):
		return self._env.reset()

	def __getattr__(self, name):
		return getattr(self._env, name)


# ---------------------------------------------------------------------------
# Classic-gym adapters (one per backend)
# ---------------------------------------------------------------------------

class DMCEnv:
	"""dm_control env -> classic-gym interface used by the rest of the code."""

	def __init__(self, env, domain, task, action_repeat, modality):
		self.env = env
		self.domain = domain
		self.task = task
		self.modality = modality
		self.ep_len = 1000 // action_repeat
		self.t = 0

		if modality == 'pixels':
			self.observation_space = Box(0, 255, env.observation_spec().shape, np.uint8)
		else:
			dim = sum(int(np.prod(v.shape)) for v in env.observation_spec().values())
			self.observation_space = Box(-np.inf, np.inf, (dim,), np.float32)

		aspec = env.action_spec()
		self.action_space = Box(aspec.minimum, aspec.maximum, aspec.shape, aspec.dtype)

	@property
	def unwrapped(self):
		return self.env

	def control_timestep(self):
		return self.env.control_timestep()

	def _obs(self, observation):
		if self.modality == 'pixels':
			return observation
		return np.concatenate([v.flatten() for v in observation.values()])

	def reset(self):
		self.t = 0
		return self._obs(self.env.reset().observation)

	def step(self, action):
		self.t += 1
		ts = self.env.step(action)
		done = ts.last() or self.t == self.ep_len
		return self._obs(ts.observation), ts.reward, done, defaultdict(float)

	def render(self, mode='rgb_array', width=384, height=384, camera_id=0):
		camera_id = dict(quadruped=2).get(self.domain, camera_id)
		return self.env.physics.render(height, width, camera_id)


class MetaWorldEnv:
	"""Meta-World (gymnasium API) -> the same classic-gym interface as DMCEnv."""

	def __init__(self, env, action_repeat, max_episode_steps):
		self.env = env
		self._action_repeat = action_repeat
		self.ep_len = max_episode_steps // action_repeat
		self.t = 0

		o, a = env.observation_space, env.action_space
		self.observation_space = Box(o.low, o.high, o.shape, np.float32)
		self.action_space = Box(a.low, a.high, a.shape, np.float32)

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def control_timestep(self):
		inner = self.env.unwrapped
		return float(inner.model.opt.timestep) * int(inner.frame_skip)

	def reset(self):
		self.t = 0
		obs, _ = self.env.reset()
		return np.asarray(obs, dtype=np.float32)

	def step(self, action):
		self.t += 1
		total_reward, terminated, truncated, info = 0.0, False, False, {}
		for _ in range(self._action_repeat):
			obs, reward, terminated, truncated, info = self.env.step(action)
			total_reward += float(reward)
			if terminated or truncated:
				break
		done = terminated or truncated or self.t == self.ep_len
		return np.asarray(obs, dtype=np.float32), total_reward, done, defaultdict(float, info)

	def render(self, mode='rgb_array', width=384, height=384, camera_id=0):
		return self.env.render()


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _make_dmc(cfg):
	"""dm_control suite. Adapted from https://github.com/facebookresearch/drqv2"""
	domain, task = cfg["task"].replace('-', '_').split('_', 1)
	domain = dict(cup='ball_in_cup').get(domain, domain)
	assert (domain, task) in suite.ALL_TASKS, f"unknown dm_control task: {domain} {task}"

	env = suite.load(domain,
					 task,
					 task_kwargs={'random': cfg["seed"]},
					 visualize_reward=False)
	# Physical seconds advanced per agent action (control step * action repeat).
	# Used as the base unit for Delta t conditioning.
	cfg["dt_base"] = float(env.control_timestep()) * cfg["action_repeat"]

	env = ActionDTypeWrapper(env, np.float32)
	env = ActionRepeatWrapper(env, cfg["action_repeat"])
	env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

	if cfg["image_observations"]:
		camera_id = dict(quadruped=2).get(domain, 0)
		render_kwargs = dict(height=84, width=84, camera_id=camera_id)
		env = pixels.Wrapper(env, pixels_only=True, render_kwargs=render_kwargs)
		env = FrameStackWrapper(env, cfg.get('frame_stack', 1), cfg["modality"])

	return DMCEnv(env, domain, task, cfg["action_repeat"], cfg["modality"])


def _make_metaworld(cfg):
	"""Meta-World single-task (Farama fork). ``pip install metaworld``.

	UNTESTED against this repo's dependency set - verify once Meta-World is
	installed. ``cfg["task"]`` is the full env name (e.g. "reach-v2");
	``cfg["metaworld_task_idx"]`` picks the goal among the 50 train tasks.
	"""
	try:
		import metaworld
	except ImportError as e:
		raise ImportError(
			"Meta-World is not installed. Install the Farama fork "
			"(`pip install metaworld`, needs mujoco), then set "
			"benchmark: metaworld in the config."
		) from e

	assert not cfg["image_observations"], "metaworld backend is state-only"

	name = cfg["task"]
	mt1 = metaworld.MT1(name, seed=cfg["seed"])
	env = mt1.train_classes[name](render_mode="rgb_array")
	tasks = mt1.train_tasks
	env.set_task(tasks[cfg.get("metaworld_task_idx", 0) % len(tasks)])

	inner = env.unwrapped
	cfg["dt_base"] = (
		float(inner.model.opt.timestep) * int(inner.frame_skip) * cfg["action_repeat"]
	)

	return MetaWorldEnv(
		env,
		action_repeat=cfg["action_repeat"],
		max_episode_steps=int(getattr(inner, "max_path_length", 500)),
	)


_BACKENDS = {
	"dmc": _make_dmc,
	"metaworld": _make_metaworld,
}


def make_env(cfg):
	"""Build the environment for a TD-MPC experiment.

	Dispatches on ``cfg["benchmark"]`` ("dmc" by default). Sets ``cfg["dt_base"]``
	and the ``obs_shape`` / ``action_shape`` / ``action_dim`` convenience keys.
	"""
	benchmark = cfg.get("benchmark", "dmc")
	if benchmark not in _BACKENDS:
		raise ValueError(
			f"unknown benchmark {benchmark!r}; expected one of {sorted(_BACKENDS)}"
		)

	env = _BACKENDS[benchmark](cfg)

	cfg["obs_shape"] = tuple(int(x) for x in env.observation_space.shape)
	cfg["action_shape"] = tuple(int(x) for x in env.action_space.shape)
	cfg["action_dim"] = env.action_space.shape[0]

	return env
