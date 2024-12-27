import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import TimeLimit
from collections import deque
import random

from env_hiv import HIVPatient


# 1) Define a simple neural network for Q(s,a).
class QNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super(QNetwork, self).__init__()
        # A small MLP with two hidden layers
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# 2) Implement a simple replay buffer
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


# 3) Our Agent that follows the required interface
class ProjectAgent:
    def __init__(self):
        # Hyperparameters
        self.gamma = 0.99
        self.lr = 1e-3
        self.batch_size = 64
        self.epsilon = 1.0  # Start fully random
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.device = torch.device("cpu")

        # Define Q-network
        self.q_network = QNetwork().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # We'll store the path we want to save/load from
        self.save_path = "project_agent.pt"

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer()

    def act(self, observation, use_random=False):
        """
        Decide action from the current observation. 
        If use_random=True or with probability epsilon, pick random action.
        Otherwise pick action = argmax Q(s,a).
        """
        if (use_random and random.random() < self.epsilon):
            # Random action
            return np.random.randint(0, 4)

        # Convert observation to torch tensor
        state_t = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        # Compute Q-values
        q_values = self.q_network(state_t)
        # Pick the action with highest Q
        action = q_values.argmax(dim=1).item()
        return action

    def save(self, path):
        """
        Save model parameters to disk.
        """
        torch.save(self.q_network.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self):
        """
        Load model parameters from self.save_path (harcoded).
        This must exactly match how you saved the model.
        """
        self.q_network.load_state_dict(torch.load(self.save_path, map_location=self.device))
        print(f"Model loaded from {self.save_path}")

    def update_epsilon(self):
        """
        Gradually decay epsilon after each episode.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_step(self):
        """
        One batch update of the Q-network from the replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_values = self.q_network(states_t)
        # Gather the Q-value corresponding to each action
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Next Q-values (no target network in this minimal example)
        with torch.no_grad():
            next_q_values = self.q_network(next_states_t).max(1)[0]

        # Bellman target
        target = rewards_t + self.gamma * next_q_values * (1 - dones_t)

        # Compute loss (MSE or Smooth L1)
        loss = nn.MSELoss()(q_values, target)

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# 4) The main training routine
def main():
    # Create environment with time limit 200 steps
    env = TimeLimit(
        env=HIVPatient(domain_randomization=False),
        max_episode_steps=200
    )
    agent = ProjectAgent()

    # Training hyperparameters
    num_episodes = 2000
    max_steps_per_episode = 200  # Already enforced by the TimeLimit

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0

        for t in range(max_steps_per_episode):
            # 1. Agent acts
            action = agent.act(state, use_random=True)  # pass use_random=True for epsilon-greedy
            # 2. Environment step
            next_state, reward, done, truncated, info = env.step(action)
            # 3. Store in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            # 4. Agent training step
            agent.train_step()

            state = next_state
            episode_reward += reward
            if done or truncated:
                break

        # Decay epsilon
        agent.update_epsilon()

        # Print progress occasionally
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.2f}")

    # Save the trained model
    agent.save("project_agent.pt")


# Entry point: you can comment/uncomment it if you want to run training directly
if __name__ == "__main__":
    main()
