import torch
import numpy as np


class Episode(object):
    """Storage object for a single episode (state observations)."""
    def __init__(self, cfg, init_obs):
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        L = cfg["episode_length"]

        self.obs = torch.empty((L+1, *init_obs.shape), dtype=torch.float32, device=self.device)
        self.obs[0] = torch.tensor(init_obs, dtype=torch.float32, device=self.device)
        self.action = torch.empty((L, cfg["action_dim"]), dtype=torch.float32, device=self.device)
        self.reward = torch.empty((L,), dtype=torch.float32, device=self.device)
        self.cumulative_reward = 0
        self.done = False
        self._idx = 0

    def __len__(self):
        return self._idx

    @property
    def first(self):
        return len(self) == 0

    def __add__(self, transition):
        self.add(*transition)
        return self

    def add(self, obs, action, reward, done):
        self.obs[self._idx+1] = torch.tensor(obs, dtype=self.obs.dtype, device=self.obs.device)
        self.action[self._idx] = action
        self.reward[self._idx] = reward
        self.cumulative_reward += reward
        self.done = done
        self._idx += 1


class ReplayBuffer():
    """
    Storage and sampling for LeWM. Kept in GPU memory (state observations only).

    Sampling is uniform by default. With cfg["prioritized_replay"] the agent
    feeds back per-sample priorities (next-latent prediction error) through
    update_priorities(), and starts are drawn proportional to priority**per_alpha.

    Rewards are stored (for logging / future use) but LeWM never trains on them.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg["device"]
        self.capacity = cfg["replay_capacity"]
        self.episode_length = cfg["episode_length"]
        self.horizon = cfg.get("horizon")
        # Steps reserved at each episode end so a sampled window stays inside one
        # episode. lewm_adaptive sets buffer_window to its longest level span.
        self.window = cfg.get("buffer_window", self.horizon)
        self.batch_size = cfg["batch_size"]
        self.per_alpha = cfg.get("per_alpha", 0.6) if cfg.get("prioritized_replay", False) else 0.0
        self.per_beta = cfg.get("per_beta", 0.4)

        state_dim = cfg["state_dim"]
        obs_shape = tuple(state_dim) if hasattr(state_dim, "__iter__") else (state_dim,)

        self._obs = torch.empty((self.capacity, *obs_shape), dtype=torch.float32, device=self.device)
        self._action = torch.empty((self.capacity, cfg["action_dim"]), dtype=torch.float32, device=self.device)
        self._reward = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)
        self._priorities = torch.ones((self.capacity,), dtype=torch.float32, device=self.device)
        self._eps = 1e-6
        self._full = False
        self.idx = 0

    def __add__(self, episode: Episode):
        self.add(episode)
        return self

    def add(self, episode: Episode):
        L = self.episode_length
        self._obs[self.idx:self.idx+L] = episode.obs[:-1]
        self._action[self.idx:self.idx+L] = episode.action
        self._reward[self.idx:self.idx+L] = episode.reward
        if self._full:
            max_priority = self._priorities.max().item()
        else:
            max_priority = 1. if self.idx == 0 else self._priorities[:self.idx].max().item()
        # Zero the priority of start states too close to the episode end, so a
        # sampled window o_t .. o_{t+window} always stays inside one stored episode.
        assert self.window < L, "training window longer than an episode"
        new_priorities = torch.full((L,), max_priority, device=self.device)
        new_priorities[torch.arange(L, device=self.device) >= L - self.window] = 0
        self._priorities[self.idx:self.idx+L] = new_priorities
        self.idx = (self.idx + L) % self.capacity
        self._full = self._full or self.idx == 0

    def update_priorities(self, idxs, priorities):
        self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self._eps

    def sample(self):
        """Returns
            obs      (B, D)       o_t
            next_obs (H, B, D)    o_{t+1} .. o_{t+H}
            action   (H, B, A)    a_t .. a_{t+H-1}
            idxs     (B,)         buffer indices of o_t (for update_priorities)
            weights  (B,)         importance-sampling weights (all 1 when uniform)
        """
        priorities = self._priorities if self._full else self._priorities[:self.idx]
        # x**0 == 1 would revive the masked-out tail starts, so mask explicitly.
        probs = torch.where(priorities > 0, priorities ** self.per_alpha, torch.zeros_like(priorities))
        probs /= probs.sum()
        total = len(probs)
        idxs = torch.from_numpy(np.random.choice(total, self.batch_size, p=probs.cpu().numpy(),
                                                 replace=not self._full)).to(self.device)
        weights = (total * probs[idxs]) ** (-self.per_beta)
        weights /= weights.max()

        H = self.horizon
        obs = self._obs[idxs]
        next_obs = torch.stack([self._obs[idxs + t + 1] for t in range(H)], dim=0)
        action = torch.stack([self._action[idxs + t] for t in range(H)], dim=0)
        return obs, next_obs, action, idxs, weights

    def sample_strided(self, horizon, stride):
        """Uniform sample of H steps at a stride of k base steps (lewm_adaptive).
        Requires horizon * stride <= buffer_window. Returns
            obs     (H+1, B, D)       o_t, o_{t+k}, .., o_{t+Hk}
            chunks  (H, B, k, A)      chunks[i] = a_{t+ik} .. a_{t+ik+k-1}
        """
        assert horizon * stride <= self.window, "strided window longer than buffer_window"
        priorities = self._priorities if self._full else self._priorities[:self.idx]
        idxs = torch.multinomial((priorities > 0).float(), self.batch_size, replacement=True)

        steps = torch.arange(horizon + 1, device=self.device) * stride
        obs = self._obs[idxs.unsqueeze(0) + steps.unsqueeze(1)]
        offs = torch.arange(horizon * stride, device=self.device).view(horizon, 1, stride)
        chunks = self._action[idxs.view(1, -1, 1) + offs]
        return obs, chunks
