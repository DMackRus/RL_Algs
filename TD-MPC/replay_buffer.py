import numpy as np
import torch as T
import pickle


# class SumTree:
#     """
#     Standard binary sum-tree for O(log n) proportional sampling and updates.
#     Leaves correspond to data indices [0, capacity). Internal nodes store
#     the sum of their children, so tree[1] (the root) holds the total priority.
#     """
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.tree = np.zeros(2 * capacity, dtype=np.float64)

#     def update(self, data_idx, priority):
#         tree_idx = data_idx + self.capacity
#         delta = priority - self.tree[tree_idx]
#         self.tree[tree_idx] = priority
#         tree_idx //= 2
#         while tree_idx >= 1:
#             self.tree[tree_idx] += delta
#             tree_idx //= 2

#     def total(self):
#         return self.tree[1]

#     def get(self, s):
#         """
#         Given a value s in [0, total()), walk down the tree and return the
#         data index whose priority "bucket" contains s, along with its priority.
#         """
#         idx = 1
#         while idx < self.capacity:
#             left = 2 * idx
#             if s <= self.tree[left]:
#                 idx = left
#             else:
#                 s -= self.tree[left]
#                 idx = left + 1
#         data_idx = idx - self.capacity
#         return data_idx, self.tree[idx]

#     def priority(self, data_idx):
#         return self.tree[data_idx + self.capacity]


# class ReplayBuffer:
#     def __init__(
#         self,
#         state_dim,
#         action_dim,
#         image_observations=False,
#         capacity=100000,
#         horizon=16,
#         batch_size=256,
#         device="cpu",
#         prioritized=True,
#         alpha=0.6,          # how much prioritization is used (0 = uniform)
#         beta_start=0.4,     # importance-sampling correction, annealed to 1.0
#         beta_frames=200000, # how many add() calls until beta reaches 1.0
#         priority_eps=1e-6,  # avoid zero-priority transitions
#     ):
#         self.capacity = capacity
#         self.horizon = horizon
#         self.batch_size = batch_size
#         self.device = device

#         if image_observations:
#             # uint8 [0, 255] -- 4x less memory than float32. RepresentationModel
#             # normalizes to [-1, 1] on the way in.
#             self.states = np.zeros((capacity, *state_dim), dtype=np.uint8)
#             self.next_states = np.zeros((capacity, *state_dim), dtype=np.uint8)
#         else:
#             self.states = np.zeros((capacity, state_dim), dtype=np.float32)
#             self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
#         self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
#         self.rewards = np.zeros(capacity, dtype=np.float32)
#         self.dones = np.zeros(capacity, dtype=np.bool_)
#         self.terminated = np.zeros(capacity, dtype=np.bool_)

#         self.ptr = 0
#         self.size = 0

#         # ---- PER state ----
#         self.prioritized = prioritized
#         self.alpha = alpha
#         self.beta_start = beta_start
#         self.beta_frames = beta_frames
#         self.priority_eps = priority_eps
#         self.frame = 0  # counts add() calls, drives beta annealing

#         self.max_priority = 1.0  # new transitions get this until they're TD-scored
#         if self.prioritized:
#             self.tree = SumTree(capacity)

#         # indices returned by the most recent sample() call, needed by
#         # update_priorities() afterwards
#         self._last_sampled_starts = None

#     def _current_beta(self):
#         progress = min(self.frame / self.beta_frames, 1.0)
#         return self.beta_start + progress * (1.0 - self.beta_start)

#     def add(self, state, action, reward, next_state, done, terminated):
#         self.states[self.ptr] = state
#         self.actions[self.ptr] = action
#         self.rewards[self.ptr] = reward
#         self.next_states[self.ptr] = next_state
#         self.dones[self.ptr] = done
#         self.terminated[self.ptr] = terminated  # for clarity, same as dones

#         if self.prioritized:
#             # The transition just written at self.ptr completes the window
#             # that STARTS at (self.ptr - horizon + 1). That start index is
#             # now a valid, fully-populated window for the first time, so
#             # give it a priority. Everything before self.size >= horizon
#             # doesn't have a valid completed window yet.
#             completed_start = self.ptr - self.horizon + 1
#             if completed_start >= 0 and self.size + 1 >= self.horizon:
#                 # new transitions default to max priority so they get
#                 # sampled at least once before their TD error is known
#                 self.tree.update(completed_start % self.capacity, self.max_priority)
#             elif completed_start < 0 and self.size + 1 >= self.horizon:
#                 # wrapped case: completed start index wraps to the end of the buffer
#                 wrapped_start = completed_start % self.capacity
#                 self.tree.update(wrapped_start, self.max_priority)

#             # The window that used to start near the old ptr may now be
#             # invalidated because it wraps across the new write head.
#             # We don't need to zero it explicitly -- sample() already
#             # rejects any window with start < ptr < end -- but if you want
#             # tighter tree mass (fewer rejected draws) you could additionally
#             # zero self.tree.update(self.ptr, 0.0) here. Left out for simplicity.

#         self.ptr = (self.ptr + 1) % self.capacity
#         self.size = min(self.size + 1, self.capacity)
#         self.frame += 1

#     def sample(self, start_index=None):
#         assert self.size > self.horizon

#         states, actions, rewards, next_states, dones, terminateds = [], [], [], [], [], []
#         starts = []
#         weights = []

#         wrapped = self.size == self.capacity  # buffer has wrapped at least once
#         beta = self._current_beta()

#         use_priority = self.prioritized and start_index is None and self.tree.total() > 0

#         attempts = 0
#         max_attempts = self.batch_size * 50  # safety valve against pathological rejection loops

#         while len(states) < self.batch_size:
#             attempts += 1
#             if start_index is not None:
#                 start = start_index
#             elif use_priority:
#                 s = np.random.uniform(0, self.tree.total())
#                 start, _ = self.tree.get(s)
#                 if start > self.size - self.horizon:
#                     # stale / not-yet-valid leaf (can happen right after a
#                     # capacity wrap before priorities catch up) -- fall back
#                     if attempts > max_attempts:
#                         start = np.random.randint(0, self.size - self.horizon)
#                     else:
#                         continue
#             else:
#                 start = np.random.randint(0, self.size - self.horizon)

#             end = start + self.horizon

#             # Exclude windows that cross the current write pointer --
#             # those splice together temporally unrelated transitions
#             if wrapped and start < self.ptr < end:
#                 continue

#             # Don't allow trajectories that cross episode boundaries
#             if np.any(self.dones[start:start + self.horizon - 1]):
#                 continue

#             states.append(self.states[start:end])
#             actions.append(self.actions[start:end])
#             rewards.append(self.rewards[start:end])
#             next_states.append(self.next_states[start:end])
#             dones.append(self.dones[start:end])
#             terminateds.append(self.terminated[start:end])
#             starts.append(start)

#             if use_priority:
#                 priority = self.tree.priority(start)
#                 prob = priority / self.tree.total()
#                 # importance-sampling weight, normalized per-batch below
#                 weight = (self.size * prob) ** (-beta)
#                 weights.append(weight)

#         states = T.tensor(np.array(states), device=self.device)
#         actions = T.tensor(np.array(actions), device=self.device)
#         rewards = T.tensor(np.array(rewards), device=self.device)
#         next_states = T.tensor(np.array(next_states), device=self.device)
#         dones = T.tensor(np.array(dones), device=self.device)
#         terminateds = T.tensor(np.array(terminateds), device=self.device)

#         if use_priority:
#             weights = np.array(weights, dtype=np.float32)
#             weights /= weights.max()  # normalize so max weight is 1 (stabilizes LR)
#             weights = T.tensor(weights, device=self.device)
#         else:
#             weights = T.ones(self.batch_size, device=self.device)

#         self._last_sampled_starts = starts

#         return states, actions, rewards, next_states, dones, terminateds, weights

#     def update_priorities(self, td_errors, starts=None):
#         """
#         Call this after computing per-sample TD errors for the batch most
#         recently returned by sample(). td_errors should be a 1D array/tensor
#         of length batch_size, one scalar magnitude per sample (e.g. mean
#         absolute TD error across the horizon for that window).
#         """
#         if not self.prioritized:
#             return

#         if starts is None:
#             starts = self._last_sampled_starts
#         assert starts is not None, "No sampled starts available -- call sample() first"

#         if T.is_tensor(td_errors):
#             td_errors = td_errors.detach().cpu().numpy()
#         td_errors = np.abs(td_errors)

#         priorities = (td_errors + self.priority_eps) ** self.alpha

#         for start, priority in zip(starts, priorities):
#             self.tree.update(start, priority)
#             self.max_priority = max(self.max_priority, priority)

#     def clear(self):
#         self.ptr = 0
#         self.size = 0
#         self.frame = 0
#         self.max_priority = 1.0
#         self.states.fill(0)
#         self.actions.fill(0)
#         self.rewards.fill(0)
#         self.next_states.fill(0)
#         self.dones.fill(0)
#         self.terminated.fill(0)
#         if self.prioritized:
#             self.tree = SumTree(self.capacity)

#     def save(self, path):
#         with open(path, "wb") as f:
#             pickle.dump(
#                 {
#                     "states": self.states,
#                     "actions": self.actions,
#                     "rewards": self.rewards,
#                     "next_states": self.next_states,
#                     "dones": self.dones,
#                     "terminated": self.terminated,
#                     "ptr": self.ptr,
#                     "size": self.size,
#                 },
#                 f,
#             )

#     def load(self, path):
#         with open(path, "rb") as f:
#             data = pickle.load(f)
#         self.states = data["states"]
#         self.actions = data["actions"]
#         self.rewards = data["rewards"]
#         self.next_states = data["next_states"]
#         self.dones = data["dones"]
#         self.terminated = data["terminated"]
#         self.ptr = data["ptr"]
#         self.size = data["size"]

#         # Priorities aren't saved/restored -- rebuild the tree by giving
#         # every valid window max priority so everything gets sampled and
#         # re-scored at least once early in the resumed run.
#         if self.prioritized:
#             self.tree = SumTree(self.capacity)
#             wrapped = self.size == self.capacity
#             valid_range = self.size - self.horizon if not wrapped else self.capacity
#             for start in range(max(0, valid_range)):
#                 if wrapped and start < self.ptr < start + self.horizon:
#                     continue
#                 if np.any(self.dones[start:start + self.horizon - 1]):
#                     continue
#                 self.tree.update(start, self.max_priority)

#     def __len__(self):
#         return self.size


# Old replay buffer code - without a prioritised experience replay buffer

import numpy as np
import torch as T
import pickle

class ReplayBuffer:
    def __init__(
        self,
        state_dim,
        action_dim,
        image_observations=False,
        capacity=100000,
        horizon=16,
        batch_size=256,
        device="cpu",
    ):
        self.capacity = capacity
        self.horizon = horizon
        self.batch_size = batch_size
        self.device = device
        self.image_observations = image_observations

        if image_observations:
            # Pixel observations are stored as uint8 [0, 255] -- 4x less memory
            # than float32. RepresentationModel.forward() normalizes to [-1, 1]
            # (and random_shift() casts to float), so nothing downstream needs
            # a pre-scaled copy. state_dim is (C, H, W).
            self.states = np.zeros((capacity, *state_dim), dtype=np.uint8)
            self.next_states = np.zeros((capacity, *state_dim), dtype=np.uint8)
        else:
            self.states = np.zeros((capacity, state_dim), dtype=np.float32)
            self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.terminated = np.zeros(capacity, dtype=np.bool_)

        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done, terminated):
        if self.image_observations:
            # state / next_state arrive as uint8 CHW torch tensors (or arrays)
            # straight from process_image(); keep them uint8 in the buffer.
            state = np.asarray(state, dtype=np.uint8)
            next_state = np.asarray(next_state, dtype=np.uint8)

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.terminated[self.ptr] = terminated

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, start_index=None):
        assert self.size > self.horizon

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        terminated = []
        wrapped = self.size == self.capacity  # buffer has wrapped at least once

        while len(states) < self.batch_size:

            if start_index is not None:
                start = start_index
            else:
                start = np.random.randint(0, self.size - self.horizon)

            end = start + self.horizon

            # Exclude windows that cross the current write pointer —
            # those splice together temporally unrelated transitions
            if wrapped and start < self.ptr < end:
                continue

            # Don't allow trajectories that cross episode boundaries
            if np.any(self.dones[start:start+self.horizon-1]):
                continue

            states.append(self.states[start:end])
            actions.append(self.actions[start:end])
            rewards.append(self.rewards[start:end])
            next_states.append(self.next_states[start:end])
            dones.append(self.dones[start:end])
            terminated.append(self.terminated[start:end])

        states = T.tensor(np.array(states), device=self.device)
        actions = T.tensor(np.array(actions), device=self.device)
        rewards = T.tensor(np.array(rewards), device=self.device)
        next_states = T.tensor(np.array(next_states), device=self.device)
        dones = T.tensor(np.array(dones), device=self.device)
        terminated = T.tensor(np.array(terminated), device=self.device)

        return states, actions, rewards, next_states, dones, terminated

    def clear(self):
        self.ptr = 0
        self.size = 0
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.next_states.fill(0)
        self.dones.fill(0)
        self.terminated.fill(0)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "states": self.states,
                    "actions": self.actions,
                    "rewards": self.rewards,
                    "next_states": self.next_states,
                    "dones": self.dones,
                    "terminated": self.terminated,
                    "ptr": self.ptr,
                    "size": self.size,
                },
                f,
            )

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.states = data["states"]
            self.actions = data["actions"]
            self.rewards = data["rewards"]
            self.next_states = data["next_states"]
            self.dones = data["dones"]
            self.terminated = data["terminated"]
            self.ptr = data["ptr"]
            self.size = data["size"]

    def __len__(self):
        return self.size