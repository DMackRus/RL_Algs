import numpy as np
import torch as T
import pickle


class SumTree:
    """
    Binary sum-tree for O(log n) proportional sampling and priority updates.

    Leaves map to data (window start) indices [0, capacity). Internal nodes
    hold the sum of their children, so tree[1] is the total priority mass.
    The leaf count is rounded up to a power of two so the tree is perfect and
    the traversal depth is uniform regardless of `capacity`.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree_capacity = 1
        while self.tree_capacity < capacity:
            self.tree_capacity *= 2
        self.tree = np.zeros(2 * self.tree_capacity, dtype=np.float64)

    def update(self, data_idx, priority):
        idx = data_idx + self.tree_capacity
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        idx //= 2
        while idx >= 1:
            self.tree[idx] += delta
            idx //= 2

    def total(self):
        return self.tree[1]

    def get(self, s):
        """
        Given s in [0, total()), walk down the tree and return the data index
        whose cumulative priority bucket contains s, plus that leaf's priority.
        """
        idx = 1
        while idx < self.tree_capacity:
            left = 2 * idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        return idx - self.tree_capacity, self.tree[idx]

    def priority(self, data_idx):
        return self.tree[data_idx + self.tree_capacity]


class ReplayBuffer:
    """
    Sequence replay buffer for TD-MPC style training. sample() returns batches
    of length-`horizon` windows. When `prioritized` is True, windows are drawn
    proportionally to their last-seen TD error (Schaul et al., 2016), and
    sample() also returns per-sample importance-sampling weights.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        image_observations=False,
        capacity=100000,
        horizon=16,
        batch_size=256,
        device="cpu",
        prioritized=True,
        alpha=0.6,          # how strongly to prioritize (0 = uniform sampling)
        beta_start=0.4,     # importance-sampling correction, annealed to 1.0
        beta_frames=200000, # add() calls over which beta reaches 1.0
        priority_eps=1e-6,  # keeps every window sampleable
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

        # ---- prioritized experience replay state ----
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.priority_eps = priority_eps
        self.frame = 0            # counts add() calls, drives beta annealing
        self.max_priority = 1.0   # new windows get this until they're TD-scored
        self.tree = SumTree(capacity) if prioritized else None

        # window start indices returned by the most recent sample() call,
        # consumed by update_priorities() afterwards
        self._last_sampled_starts = None

    # ------------------------------------------------------------------ #
    #  helpers                                                            #
    # ------------------------------------------------------------------ #
    def _current_beta(self):
        progress = min(self.frame / self.beta_frames, 1.0)
        return self.beta_start + progress * (1.0 - self.beta_start)

    def _window_is_valid(self, start):
        """
        True when `start` begins a usable length-`horizon` window: it stays
        inside the physical buffer and doesn't straddle the write head (which
        would splice unrelated transitions).

        Windows that cross an episode boundary ARE allowed now -- sample()
        returns a per-step mask that zeros out every step after the first
        `done`, so the terminal transition is kept (at full weight, wherever it
        lands in the window) while the next episode's spliced-in steps are
        ignored by the loss.
        """
        if start < 0 or start > self.size - self.horizon:
            return False
        if start + self.horizon > self.capacity:
            return False
        if self.size == self.capacity and start < self.ptr < start + self.horizon:
            return False
        return True

    # ------------------------------------------------------------------ #
    #  writing                                                            #
    # ------------------------------------------------------------------ #
    def add(self, state, action, reward, next_state, done, terminated):
        if self.image_observations:
            # state / next_state arrive as uint8 CHW tensors (or arrays)
            # straight from process_image(); keep them uint8 in the buffer.
            state = np.asarray(state, dtype=np.uint8)
            next_state = np.asarray(next_state, dtype=np.uint8)

        if self.prioritized:
            # the window that used to start at self.ptr is about to be partly
            # overwritten -- drop its priority so it can't be drawn any more
            self.tree.update(self.ptr, 0.0)

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.terminated[self.ptr] = terminated

        if self.prioritized:
            # writing at self.ptr completes the window that starts horizon-1
            # slots back -- it is fully populated for the first time now, so
            # seed it with max priority to guarantee it gets sampled (and
            # TD-scored) at least once. Windows that contain a terminal are
            # fine (sample() masks the spliced tail); only skip windows that
            # would wrap the physical buffer end.
            completed_start = self.ptr - self.horizon + 1
            if completed_start >= 0 and self.size + 1 >= self.horizon:
                self.tree.update(completed_start, self.max_priority)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.frame += 1

    # ------------------------------------------------------------------ #
    #  sampling                                                           #
    # ------------------------------------------------------------------ #
    def sample(self, start_index=None):
        assert self.size > self.horizon

        states, actions, rewards, next_states, dones, terminateds = [], [], [], [], [], []
        starts, weights, masks = [], [], []

        beta = self._current_beta()
        use_priority = (
            self.prioritized and start_index is None and self.tree.total() > 0
        )

        attempts = 0
        max_attempts = self.batch_size * 100  # safety valve against pathological rejection

        while len(states) < self.batch_size:
            attempts += 1
            fallback = attempts > max_attempts

            if start_index is not None:
                start = start_index
            elif use_priority and not fallback:
                s = np.random.uniform(0, self.tree.total())
                start, _ = self.tree.get(s)
                if not self._window_is_valid(start):
                    continue
            else:
                start = np.random.randint(0, self.size - self.horizon)
                if not self._window_is_valid(start):
                    continue

            end = start + self.horizon

            # per-step mask: 1 up to and including the first episode end in the
            # window, 0 afterwards (those steps belong to the next episode)
            window_dones = self.dones[start:end]
            mask = np.ones(self.horizon, dtype=np.float32)
            if window_dones.any():
                first_done = int(np.argmax(window_dones))
                mask[first_done + 1:] = 0.0

            states.append(self.states[start:end])
            actions.append(self.actions[start:end])
            rewards.append(self.rewards[start:end])
            next_states.append(self.next_states[start:end])
            dones.append(self.dones[start:end])
            terminateds.append(self.terminated[start:end])
            masks.append(mask)
            starts.append(start)

            if use_priority:
                prob = self.tree.priority(start) / self.tree.total()
                # importance-sampling weight, normalized per-batch below
                weights.append((self.size * prob) ** (-beta))

        states = T.tensor(np.array(states), device=self.device)
        actions = T.tensor(np.array(actions), device=self.device)
        rewards = T.tensor(np.array(rewards), device=self.device)
        next_states = T.tensor(np.array(next_states), device=self.device)
        dones = T.tensor(np.array(dones), device=self.device)
        terminateds = T.tensor(np.array(terminateds), device=self.device)
        masks = T.tensor(np.array(masks), device=self.device)

        if use_priority:
            weights = np.asarray(weights, dtype=np.float32)
            # a zero-priority window could sneak in through the uniform
            # fallback and give a non-finite weight -- neutralize it
            weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
            weights /= weights.max()  # max weight -> 1, only scales the update down
            weights = T.tensor(weights, device=self.device)
        else:
            weights = T.ones(self.batch_size, device=self.device)

        self._last_sampled_starts = starts

        return states, actions, rewards, next_states, dones, terminateds, weights, masks

    def update_priorities(self, td_errors, starts=None):
        """
        Call after computing per-sample TD errors for the batch most recently
        returned by sample(). `td_errors` is a 1D array/tensor of length
        batch_size (one magnitude per window, e.g. mean absolute TD error over
        the horizon).
        """
        if not self.prioritized:
            return

        if starts is None:
            starts = self._last_sampled_starts
        assert starts is not None, "call sample() before update_priorities()"

        if T.is_tensor(td_errors):
            td_errors = td_errors.detach().cpu().numpy()
        td_errors = np.abs(np.asarray(td_errors, dtype=np.float64)).reshape(-1)

        priorities = (td_errors + self.priority_eps) ** self.alpha
        for start, priority in zip(starts, priorities):
            priority = float(priority)
            self.tree.update(int(start), priority)
            self.max_priority = max(self.max_priority, priority)

    # ------------------------------------------------------------------ #
    #  bookkeeping                                                        #
    # ------------------------------------------------------------------ #
    def clear(self):
        self.ptr = 0
        self.size = 0
        self.frame = 0
        self.max_priority = 1.0
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.next_states.fill(0)
        self.dones.fill(0)
        self.terminated.fill(0)
        self._last_sampled_starts = None
        if self.prioritized:
            self.tree = SumTree(self.capacity)

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

        # Priorities aren't persisted -- rebuild the tree by giving every valid
        # window max priority so everything gets sampled and re-scored early in
        # the resumed run.
        if self.prioritized:
            self.tree = SumTree(self.capacity)
            self.max_priority = 1.0
            for start in range(max(0, self.size - self.horizon + 1)):
                if self._window_is_valid(start):
                    self.tree.update(start, self.max_priority)

    def __len__(self):
        return self.size
