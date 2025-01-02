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
from dqns import QNetwork, BiggerDuelingQNetwork, BBBBBiggerDuelingQNetwork
from replay_buffers import ReplayBuffer, PrioritizedReplayBuffer

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
    plt.savefig(f"images/episode_rewards_ma.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
