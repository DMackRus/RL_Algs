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

        if image_observations:
            self.states = np.zeros((capacity, *state_dim), dtype=np.float32)
            self.next_states = np.zeros((capacity, *state_dim), dtype=np.float32)
        else:
            self.states = np.zeros((capacity, state_dim), dtype=np.float32)
            self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, start_index=None):
        assert self.size > self.horizon

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
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

        states = T.tensor(np.array(states), device=self.device)
        actions = T.tensor(np.array(actions), device=self.device)
        rewards = T.tensor(np.array(rewards), device=self.device)
        next_states = T.tensor(np.array(next_states), device=self.device)
        dones = T.tensor(np.array(dones), device=self.device)

        return states, actions, rewards, next_states, dones

    def clear(self):
        self.ptr = 0
        self.size = 0
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.next_states.fill(0)
        self.dones.fill(0)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "states": self.states,
                    "actions": self.actions,
                    "rewards": self.rewards,
                    "next_states": self.next_states,
                    "dones": self.dones,
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
            self.ptr = data["ptr"]
            self.size = data["size"]

    def __len__(self):
        return self.size