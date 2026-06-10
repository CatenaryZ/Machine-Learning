import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# =========================
# Q-Network
# =========================
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# DQN Agent
# =========================
class DQNAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=5000,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000
    ):
        self.q_net = QNetwork(obs_dim, act_dim)
        self.target_net = QNetwork(obs_dim, act_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.replay_buffer = deque(maxlen=buffer_size)
        self.step_count = 0
        self.act_dim = act_dim

    def select_action(self, obs):
        self.step_count += 1
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * np.exp(-1.0 * self.step_count / self.epsilon_decay)

        if random.random() < self.epsilon:
            return random.randint(0, self.act_dim - 1)
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                q_values = self.q_net(obs_t)
            return q_values.argmax().item()

    def store(self, transition):
        self.replay_buffer.append(transition)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.q_net(obs).gather(1, actions).squeeze()
        with torch.no_grad():
            max_next_q = self.target_net(next_obs).max(1)[0]
            target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


# =========================
# Training Loop
# =========================
def train_dqn(episodes=2000):
    env = gym.make("MountainCar-v0")
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.n
    )

    reward_history = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # =========================
            # Reward Shaping
            # =========================
            shaped_reward = reward + 0.1 * next_obs[0]  # position shaping

            agent.store((obs, action, shaped_reward, next_obs, done))
            agent.update()

            obs = next_obs
            ep_reward += reward  # 原始 reward 用于评估

        reward_history.append(ep_reward)

        if (ep + 1) % 50 == 0:
            print(
                f"Episode {ep+1:4d} | "
                f"Avg Reward (50): {np.mean(reward_history[-50:]):.2f}"
            )

    env.close()

    # =========================
    # Visualization
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(reward_history, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("DQN on MountainCar-v0 (with Reward Shaping)")
    plt.legend()
    plt.grid()

    plt.savefig("dqn_mountaincar_reward.png")
    plt.show()


if __name__ == "__main__":
    train_dqn()
