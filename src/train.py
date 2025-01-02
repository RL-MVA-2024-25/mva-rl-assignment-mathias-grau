import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population


class QNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DeeperQNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super(DeeperQNetwork, self).__init__()
        
        # Shared feature trunk
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x):
        # Common feature extraction
        return self.net(x)

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

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super(DuelingQNetwork, self).__init__()

        # Feature extraction 
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Value stream 
        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage stream 
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        # Common feature extraction
        features = self.feature(x)

        # Value and Advantage
        values = self.value_stream(features)              # shape: (batch_size, 1)
        advantages = self.advantage_stream(features)      # shape: (batch_size, action_dim)

        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
        return q_values


class BiggerDuelingQNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super(BiggerDuelingQNetwork, self).__init__()
        
        # Shared feature trunk
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        # Common feature extraction
        features = self.feature(x)
        
        # Value and Advantage
        value = self.value_stream(features)               # shape: (batch_size, 1)
        advantage = self.advantage_stream(features)       # shape: (batch_size, action_dim)

        # Combine to get Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class BBiggerDuelingQNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super(BBiggerDuelingQNetwork, self).__init__()
        
        # Shared feature trunk
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        # Common feature extraction
        features = self.feature(x)
        
        # Value and Advantage
        value = self.value_stream(features)               # shape: (batch_size, 1)
        advantage = self.advantage_stream(features)       # shape: (batch_size, action_dim)

        # Combine to get Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class BBBiggerDuelingQNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super(BBBiggerDuelingQNetwork, self).__init__()
        
        # Shared feature trunk
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        # Common feature extraction
        features = self.feature(x)
        
        # Value and Advantage
        value = self.value_stream(features)               # shape: (batch_size, 1)
        advantage = self.advantage_stream(features)       # shape: (batch_size, action_dim)

        # Combine to get Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class BBBBiggerDuelingQNetwork(nn.Module): # le meilleur pour l'instant 
    def __init__(self, state_dim=6, action_dim=4):
        super(BBBBiggerDuelingQNetwork, self).__init__()
        
        # Shared feature trunk
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
        )
        
        # Value stream256
        self.value_stream = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        # Common feature extraction
        features = self.feature(x)
        
        # Value and Advantage
        value = self.value_stream(features)               # shape: (batch_size, 1)
        advantage = self.advantage_stream(features)       # shape: (batch_size, action_dim)

        # Combine to get Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class BBBBBiggerDuelingQNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super(BBBBBiggerDuelingQNetwork, self).__init__()
        
        # Shared feature trunk
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            # nn.Linear(1024, 1024),
            # nn.SiLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 256), 
            nn.SiLU(),
            nn.Linear(256, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        # Common feature extraction
        features = self.feature(x)
        
        # Value and Advantage
        value = self.value_stream(features)               # shape: (batch_size, 1)
        advantage = self.advantage_stream(features)       # shape: (batch_size, action_dim)

        # Combine to get Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values



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


class ProjectAgent:
    def __init__(self, use_dueling=True, use_per=True, state_dim=6, n_actions=4):
        """
        DQN Agent that optionally uses:
          - Dueling Q-Network architecture
          - Prioritized Experience Replay
        """
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.gamma = 0.85 # 0.85 well
        self.save_path = "project_agent.pt"

        # Replay buffer
        self.use_per = use_per
        if self.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=60000)
        else:
            self.replay_buffer = ReplayBuffer(capacity=60000)

        # DQN parameters
        self.lr = 1e-3 # 1e-3 well
        self.batch_size = 1024 # 64 well
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9965  # =>  step_size = 160 et si 0.997 et num_episodes = 1000 => step_size = 200 et num_episodes = 1500
        self.target_update_freq = 1000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.update_count = 0

        # Q-networks: either standard or dueling
        if use_dueling:
            self.q_network = BBBBBiggerDuelingQNetwork(self.state_dim, self.n_actions).to(self.device)
            self.target_network = BBBBBiggerDuelingQNetwork(self.state_dim, self.n_actions).to(self.device)
        else:
            print("Using standard DQN architecture.")
            self.q_network = QNetwork(self.state_dim, self.n_actions).to(self.device)
            self.target_network = QNetwork(self.state_dim, self.n_actions).to(self.device)
                
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(
                                self.q_network.parameters(),
                                lr=self.lr,            
                                betas=(0.5, 0.999),    # beta 1 is changed inspiration from PyTroch GAN implementation which is also very sensitive
                            )
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=350, gamma=0.5)

    def act(self, observation, use_random=False):
        """Pick an action given the current observation (epsilon-greedy if use_random)."""
        # Epsilon-greedy
        if use_random and random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        # Forward pass
        state_t = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_t)
        return q_values.argmax(dim=1).item()

    def train_step(self):
        """Perform one training step of DQN."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        if self.use_per:
            sample = self.replay_buffer.sample(self.batch_size)
            if sample is None:
                return
            states, actions, rewards, next_states, dones, indices, weights = sample
            weights_t = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            indices, weights_t = None, 1.0

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q
        q_values = self.q_network(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN: compute max_next_q
        with torch.no_grad():
            # Select action using the main Q-network
            next_actions = self.q_network(next_states_t).argmax(1)
            # Evaluate Q-value of the selected action using the target network
            max_next_q = self.target_network(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # Compute target
            target = rewards_t + self.gamma * max_next_q * (1 - dones_t)


        # Weighted MSE loss (if PER is used)
        loss = (q_values - target) ** 2
        if self.use_per:
            loss = loss * weights_t.squeeze()  # element-wise multiply
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update priorities in PER
        if self.use_per and indices is not None:
            td_errors = (q_values - target).detach().cpu().numpy()
            new_priorities = np.abs(td_errors)
            self.replay_buffer.update_priorities(indices, new_priorities)

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def update_epsilon(self):
        """Decay epsilon after each episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def step_scheduler(self):
        """
        Call this function once per episode (or once every X episodes)
        to step the learning rate scheduler.
        """
        self.scheduler.step()

    def save(self, path):
        """Save the DQN model."""
        torch.save(self.q_network.state_dict(), path)
        print(f"DQN model saved to {path}")

    def load(self, path = None):
        """Load the DQN model."""
        if path is None:
            path = self.save_path
        self.q_network.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"DQN model loaded from {path}")

def main():
    USE_DUELING = True
    USE_PER = True

    env = TimeLimit(HIVPatient(domain_randomization=True), max_episode_steps=200)
    agent = ProjectAgent(
        use_dueling=USE_DUELING,
        use_per=USE_PER
    )

    num_episodes = 1500
    reward_history = []

    BEST_VALIDATION_SCORE = 0.0

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0

        actions = {0 : 0, 1 : 0, 2 : 0, 3 : 0}

        for _ in range(200):
            # print int of each state
            action = agent.act(state, use_random=True)
            actions[action] += 1
            next_state, reward, done, truncated, _info = env.step(action)

            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            episode_reward += reward
            if done or truncated:
                break
        agent.update_epsilon()
        agent.step_scheduler()
        print(f"Episode {episode:4d}, Reward: {int(episode_reward):11d}, Epsilon: {agent.epsilon:.2f}, LR: {agent.scheduler.get_last_lr()[0]:.5f}, Alpha: {agent.replay_buffer.alpha:.2f}, Beta: {agent.replay_buffer.beta:.2f}, Actions: {actions}")
        reward_history.append(episode_reward)

        if episode_reward > 3.5e10:
            validation_score = evaluate_HIV(agent = agent, nb_episode=3)
            validation_score_dr = evaluate_HIV_population(agent = agent, nb_episode=5)
            if (validation_score+validation_score_dr)/2 > BEST_VALIDATION_SCORE:
                BEST_VALIDATION_SCORE = (validation_score+validation_score_dr)/2
                print(f"New model with validation score: {BEST_VALIDATION_SCORE:.2f}")
                agent.save("best_" + agent.save_path)
            print(f"Validation score: {validation_score:.2f}, Validation score DR: {validation_score_dr:.2f}")
            if validation_score > 4e10 and validation_score_dr > 2.5e10:
                agent.save("best_top_score_" + agent.save_path)
                break

    agent.save(agent.save_path)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards per Episode")
    plt.legend()
    plt.savefig("images/episode_rewards.png", dpi=300, bbox_inches="tight")
    plt.close()
    # moving average
    plt.figure(figsize=(10, 5))
    plt.plot(np.convolve(reward_history, np.ones(10)/10, mode='valid'), label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards per Episode (Moving Average)")
    plt.legend()
    plt.savefig(f"images/episode_rewards_ma_{USE_DUELING}_{USE_PER}_{agent.lr}_{agent.batch_size}_{agent.gamma}.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
