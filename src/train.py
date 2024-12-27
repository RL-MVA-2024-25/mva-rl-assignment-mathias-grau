import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from gymnasium.wrappers import TimeLimit
from collections import deque
import random

from env_hiv import HIVPatient

class QNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
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


class ProjectAgent:
    def __init__(self):
        # Hyperparameters
        self.gamma = 0.99
        self.lr = 1e-3
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = 1000

        # Device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Q-networks
        self.q_network = QNetwork().to(self.device)
        self.target_network = QNetwork().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer()

        self.update_count = 0
        self.save_path = "project_agent.pt"

    def act(self, observation, use_random=False):
        # Epsilon-greedy
        if use_random and random.random() < self.epsilon:
            return np.random.randint(0, 4)

        state_t = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_t)
        return q_values.argmax(dim=1).item()

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self):
        self.q_network.load_state_dict(torch.load(self.save_path, map_location=self.device))
        print(f"Model loaded from {self.save_path}")

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_network(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states_t).max(1)[0]

        target = rewards_t + self.gamma * next_q_values * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())


def main():
    env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
    agent = ProjectAgent()

    num_episodes = 200
    reward_history = []
    epsilon_history = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0

        for _ in range(200):
            action = agent.act(state, use_random=True)
            next_state, reward, done, truncated, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)

            agent.train_step()
            state = next_state
            episode_reward += reward

            if done or truncated:
                break

        reward_history.append(episode_reward)
        epsilon_history.append(agent.epsilon)
        agent.update_epsilon()

        if episode % 10 == 0:
            print(f"Episode {episode:4d}, Reward: {episode_reward:10.2f}, Epsilon: {agent.epsilon:.2f}")

    # Save the trained model
    agent.save("project_agent.pt")

    # --- Plot the results and save the figures ---

    # 1) Raw Rewards
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards per Episode")
    plt.legend()
    # Save fig to disk
    plt.savefig("episode_rewards.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 2) Smoothed Rewards (rolling average)
    window_size = 10
    rolling_avgs = []
    for i in range(len(reward_history)):
        if i < window_size:
            avg = np.mean(reward_history[:i+1])
        else:
            avg = np.mean(reward_history[i-window_size+1:i+1])
        rolling_avgs.append(avg)

    plt.figure(figsize=(10, 5))
    plt.plot(rolling_avgs, color='orange', label=f"Moving Average (window={window_size})")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.title("Smoothed Rewards (Moving Average)")
    plt.legend()
    plt.savefig("episode_rewards_smoothed.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 3) Epsilon Decay
    plt.figure(figsize=(10, 5))
    plt.plot(epsilon_history, label="Epsilon")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay")
    plt.legend()
    plt.savefig("epsilon_decay.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
