import random
from collections import deque
import numpy as np



class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity=60000, alpha=0.6, beta=0.4, beta_increment_per_sampling=1e-5, alpha_decrement_per_sampling=0):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # store priorities
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.alpha_decrement_per_sampling = alpha_decrement_per_sampling
        self.eps = 1e-5
        self.max_priority = 1.0  # to initialize new transitions' priorities

    def push(self, state, action, reward, next_state, done):
        """Add a new transition with max priority (so it gets sampled at least once)."""
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size=64):
        if len(self.buffer) == 0:
            return None

        # Calculate probabilities
        # p_i = priorities[i]^alpha / sum(priorities^alpha)
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices according to probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Importance-sampling weights
        # w_i = (1 / (N * P(i)))^beta / max(w_i)
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        # Normalize weights by max
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        # can be removed if does not work well try 
        self.alpha = max(0.3, self.alpha - self.alpha_decrement_per_sampling)

        # Fetch transitions
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            s, a, r, n_s, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(n_s)
            dones.append(d)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            weights
        )

    def update_priorities(self, batch_indices, batch_priorities):
        # priorities are absolute TD errors, e.g. |r + gamma * maxQ(next) - Q(curr)|
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio + self.eps
            self.max_priority = max(self.max_priority, self.priorities[idx])

    def __len__(self):
        return len(self.buffer)
