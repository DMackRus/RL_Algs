import torch 
import numpy as np

class Episode(object):
    """Storage object for a single episode."""
    def __init__(self, cfg, init_obs):
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        if cfg["image_observations"]:
            dtype = torch.uint8
            # TODO - this is wrong - fix
            obs_shape = (3, *state_dim[-2:])
        else:
            dtype = torch.float32
            state_dim = cfg["state_dim"]
            obs_shape = tuple(state_dim) if hasattr(state_dim, "__iter__") else (state_dim,)

        self.obs = torch.empty((cfg["episode_length"]+1, *init_obs.shape), dtype=dtype, device=self.device)
        self.obs[0] = torch.tensor(init_obs, dtype=dtype, device=self.device)
        self.action = torch.empty((cfg["episode_length"], cfg["action_dim"]), dtype=torch.float32, device=self.device)
        self.reward = torch.empty((cfg["episode_length"],), dtype=torch.float32, device=self.device)
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
    Storage and sampling functionality for training TD-MPC / TOLD.
    The replay buffer is stored in GPU memory when training from state.
    Uses prioritized experience replay by default.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg["device"]
        self.capacity = cfg["replay_capacity"]
        self.image_observations = cfg["image_observations"]
        if cfg["image_observations"]:
            dtype = torch.uint8
            # TODO - this is wrong - fix
            obs_shape = (3, *state_dim[-2:])
        else:
            dtype = torch.float32
            state_dim = cfg["state_dim"]
            obs_shape = tuple(state_dim) if hasattr(state_dim, "__iter__") else (state_dim,)

        # TODO - Episode length hardcoded - change urgently
        self.episode_length = cfg["episode_length"]

        # Variable macro-step ("Delta t") sampling. When set, sample() draws a
        # per-(step, batch) stride k and returns k-step transitions; add() must
        # reserve a matching training-window margin at each episode end.
        self.adaptive = cfg["training_algorithm"] == "tdmpc_adaptive"
        self.k_max = int(cfg.get("k_max", 4))
        self.k_min = int(cfg.get("k_min", 1))
        self.k_sampling = cfg.get("k_sampling", "batch")

        self._obs = torch.empty((self.capacity+1, *obs_shape), dtype=dtype, device=self.device)
        self._last_obs = torch.empty((self.capacity//self.episode_length, *obs_shape), dtype=dtype, device=self.device)
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
        if self.image_observations:
            self._obs[self.idx:self.idx+self.episode_length] = episode.obs[:-1, -3:]
        else:
            self._obs[self.idx:self.idx+self.episode_length] = episode.obs[:-1]

        # self._obs[self.idx:self.idx+self.episode_length] = episode.obs[:-1] if self.cfg.modality == 'state' else episode.obs[:-1, -3:]
        self._last_obs[self.idx//self.episode_length] = episode.obs[-1]
        self._action[self.idx:self.idx+self.episode_length] = episode.action
        self._reward[self.idx:self.idx+self.episode_length] = episode.reward
        if self._full:
            max_priority = self._priorities.max().to(self.device).item()
        else:
            max_priority = 1. if self.idx == 0 else self._priorities[:self.idx].max().to(self.device).item()
        # Zero the priority of start states too close to the episode end to form a
        # full training window. With variable Delta t a macro-step spans up to
        # k_max base steps, so the window can be up to (horizon+1)*k_max long.
        # This MUST match the read pattern in sample() (same self.adaptive gate).
        if self.adaptive:
            margin = (self.cfg["horizon"] + 1) * self.k_max
        else:
            margin = self.cfg["horizon"]
        assert margin < self.cfg["episode_length"], "training window longer than an episode"
        mask = torch.arange(self.cfg["episode_length"]) >= self.cfg["episode_length"] - margin
        new_priorities = torch.full((self.cfg["episode_length"],), max_priority, device=self.device)
        new_priorities[mask] = 0
        self._priorities[self.idx:self.idx+self.cfg["episode_length"]] = new_priorities
        self.idx = (self.idx + self.cfg["episode_length"]) % self.capacity
        self._full = self._full or self.idx == 0

    def update_priorities(self, idxs, priorities):
        self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self._eps

    def _get_obs(self, arr, idxs):
        if not self.image_observations:
            return arr[idxs]
        obs = torch.empty((self.cfg["batch_size"], 3*self.cfg["frame_stack"], *arr.shape[-2:]), dtype=arr.dtype, device=torch.device('cuda'))
        obs[:, -3:] = arr[idxs].cuda()
        _idxs = idxs.clone()
        mask = torch.ones_like(_idxs, dtype=torch.bool)
        for i in range(1, self.cfg["frame_stack"]):
            mask[_idxs % self.cfg["episode_length"] == 0] = False
            _idxs[mask] -= 1
            obs[:, -(i+1)*3:-i*3] = arr[_idxs].cuda()
        return obs.float()

    def sample(self):
        probs = (self._priorities if self._full else self._priorities[:self.idx]) ** self.cfg["per_alpha"]
        probs /= probs.sum()
        total = len(probs)
        idxs = torch.from_numpy(np.random.choice(total, self.cfg["batch_size"], p=probs.cpu().numpy(), replace=not self._full)).to(self.device)
        weights = (total * probs[idxs]) ** (-self.cfg["per_beta"])
        weights /= weights.max()

        B, H = self.cfg["batch_size"], self.cfg["horizon"]
        obs = self._get_obs(self._obs, idxs)
        next_obs_shape = self._last_obs.shape[1:] if not self.image_observations else (3*self.cfg["frame_stack"], *self._last_obs.shape[-2:])
        next_obs = torch.empty((H+1, B, *next_obs_shape), dtype=obs.dtype, device=obs.device)
        action = torch.empty((H+1, B, *self._action.shape[1:]), dtype=torch.float32, device=self.device)
        reward = torch.empty((H+1, B), dtype=torch.float32, device=self.device)

        if not self.adaptive:
            # Original single-step transitions.
            k = 1
            for t in range(H+1):
                _idxs = idxs + t
                next_obs[t] = self._get_obs(self._obs, _idxs+1)
                action[t] = self._action[_idxs]
                reward[t] = self._reward[_idxs]
            mask = (_idxs+1) % self.cfg["episode_length"] == 0
            next_obs[-1, mask] = self._last_obs[_idxs[mask]//self.cfg["episode_length"]].cuda().float()
        else:
            # Variable macro-step: model step t of branch b spans k[t, b] base env
            # steps, i.e. dt = k[t, b] * dt_base seconds. Over that window:
            #   reward[t]  = discounted return   sum_{i<k} gamma**i * r[base+i]
            #   action[t]  = mean of the k base actions applied in the window
            #   next_obs[t]= obs at base + k[t]  (the window's end)
            # The priority mask in add() reserves (horizon+1)*k_max base steps at
            # each episode end (same self.adaptive gate), so every window stays
            # inside one episode and no terminal fixup is needed.
            #
            # k_sampling controls how much k is allowed to vary:
            #   "batch"    - one stride drawn per sample() call (original behaviour)
            #   "sequence" - one stride per branch, constant along the branch
            #   "step"     - an independent stride for every (t, b)
            k_max, k_min, mode = self.k_max, self.k_min, self.k_sampling
            if mode == "batch":
                k = torch.full((H+1, B), int(np.random.randint(k_min, k_max + 1)),
                               dtype=torch.long, device=self.device)
            elif mode == "sequence":
                k = torch.randint(k_min, k_max + 1, (1, B), device=self.device) \
                        .expand(H+1, B).contiguous()
            elif mode == "step":
                k = torch.randint(k_min, k_max + 1, (H+1, B), device=self.device)
            else:
                raise ValueError(f"unknown k_sampling '{mode}'")

            gamma = self.cfg["discount"]
            A = self._action.shape[1]
            # base[t] = idxs + (number of base steps consumed by steps 0..t-1)
            cum = torch.zeros((H+1, B), dtype=torch.long, device=self.device)
            cum[1:] = torch.cumsum(k[:-1], dim=0)
            for t in range(H+1):
                base = idxs + cum[t]
                next_obs[t] = self._get_obs(self._obs, base + k[t])
                r = torch.zeros(B, dtype=torch.float32, device=self.device)
                a_acc = torch.zeros(B, A, dtype=torch.float32, device=self.device)
                # d = torch.ones(B, dtype=torch.float32, device=self.device)
                for i in range(k_max):
                    active = (i < k[t]).float()                 # (B,) 1 while i < k
                    r = r + (active * self._reward[base + i])
                    a_acc = a_acc + active.unsqueeze(1) * self._action[base + i]
                    # d = d * gamma
                reward[t] = r
                action[t] = a_acc / k[t].float().unsqueeze(1)   # mean over the window

        return obs, next_obs, action, reward.unsqueeze(2), idxs, weights, k

    # def sample(self):
    #     probs = (self._priorities if self._full else self._priorities[:self.idx]) ** self.cfg["per_alpha"]
    #     probs /= probs.sum()
    #     total = len(probs)
    #     idxs = torch.from_numpy(np.random.choice(total, self.cfg["batch_size"], p=probs.cpu().numpy(), replace=not self._full)).to(self.device)
    #     weights = (total * probs[idxs]) ** (-self.cfg["per_beta"])
    #     weights /= weights.max()

    #     B, H = self.cfg["batch_size"], self.cfg["horizon"]
    #     obs = self._get_obs(self._obs, idxs)
    #     next_obs_shape = self._last_obs.shape[1:] if not self.image_observations else (3*self.cfg["frame_stack"], *self._last_obs.shape[-2:])
    #     next_obs = torch.empty((H+1, B, *next_obs_shape), dtype=obs.dtype, device=obs.device)
    #     action = torch.empty((H+1, B, *self._action.shape[1:]), dtype=torch.float32, device=self.device)
    #     reward = torch.empty((H+1, B), dtype=torch.float32, device=self.device)

    #     if not self.adaptive:
    #         # Original single-step transitions.
    #         k = 1
    #         for t in range(H+1):
    #             _idxs = idxs + t
    #             next_obs[t] = self._get_obs(self._obs, _idxs+1)
    #             action[t] = self._action[_idxs]
    #             reward[t] = self._reward[_idxs]
    #         mask = (_idxs+1) % self.cfg["episode_length"] == 0
    #         next_obs[-1, mask] = self._last_obs[_idxs[mask]//self.cfg["episode_length"]].cuda().float()
    #     else:
    #         # Variable macro-step: model step t of branch b spans k[t, b] base env
    #         # steps, i.e. dt = k[t, b] * dt_base seconds. Over that window:
    #         #   reward[t]  = discounted return   sum_{i<k} gamma**i * r[base+i]
    #         #   action[t]  = mean of the k base actions applied in the window
    #         #   next_obs[t]= obs at base + k[t]  (the window's end)
    #         # The priority mask in add() reserves (horizon+1)*k_max base steps at
    #         # each episode end (same self.adaptive gate), so every window stays
    #         # inside one episode and no terminal fixup is needed.
    #         #
    #         # k_sampling controls how much k is allowed to vary:
    #         #   "batch"    - one stride drawn per sample() call (original behaviour)
    #         #   "sequence" - one stride per branch, constant along the branch
    #         #   "step"     - an independent stride for every (t, b)
    #         k_max, k_min, mode = self.k_max, self.k_min, self.k_sampling
    #         if mode == "batch":
    #             k = torch.full((H+1, B), int(np.random.randint(k_min, k_max + 1)),
    #                            dtype=torch.long, device=self.device)
    #         elif mode == "sequence":
    #             k = torch.randint(k_min, k_max + 1, (1, B), device=self.device) \
    #                     .expand(H+1, B).contiguous()
    #         elif mode == "step":
    #             k = torch.randint(k_min, k_max + 1, (H+1, B), device=self.device)
    #         else:
    #             raise ValueError(f"unknown k_sampling '{mode}'")

    #         gamma = self.cfg["discount"]
    #         A = self._action.shape[1]
    #         # base[t] = idxs + (number of base steps consumed by steps 0..t-1)
    #         cum = torch.zeros((H+1, B), dtype=torch.long, device=self.device)
    #         cum[1:] = torch.cumsum(k[:-1], dim=0)
    #         for t in range(H+1):
    #             base = idxs + cum[t]
    #             next_obs[t] = self._get_obs(self._obs, base + k[t])
    #             r = torch.zeros(B, dtype=torch.float32, device=self.device)
    #             a_acc = torch.zeros(B, A, dtype=torch.float32, device=self.device)
    #             d = torch.ones(B, dtype=torch.float32, device=self.device)
    #             for i in range(k_max):
    #                 active = (i < k[t]).float()                 # (B,) 1 while i < k
    #                 r = r + active * d * self._reward[base + i]
    #                 a_acc = a_acc + active.unsqueeze(1) * self._action[base + i]
    #                 d = d * gamma
    #             reward[t] = r
    #             action[t] = a_acc / k[t].float().unsqueeze(1)   # mean over the window

    #     if not action.is_cuda:
    #         action, reward, idxs, weights = \
    #             action.cuda(), reward.cuda(), idxs.cuda(), weights.cuda()

    #     return obs, next_obs, action, reward.unsqueeze(2), idxs, weights, k
