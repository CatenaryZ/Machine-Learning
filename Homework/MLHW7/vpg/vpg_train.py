import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class VPGAgent:
    def __init__(self, state_dim=2, action_dim=3, hidden_dim=128, 
                 lr=0.001, gamma=0.99, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        self.log_probs.append(log_prob)
        return action.item()
    
    def update(self):
        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear buffers
        self.log_probs = []
        self.rewards = []
        
        return policy_loss.item()

def train_vpg(episodes=2000, max_steps=200):
    env = gym.make('MountainCar-v0')
    agent = VPGAgent()
    
    episode_rewards = []
    episode_lengths = []
    moving_avg_rewards = deque(maxlen=100)
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        log_probs = []
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Modify reward to encourage reaching the goal
            position = next_state[0]
            velocity = next_state[1]
            
            # Reward shaping: extra reward for being close to goal
            if position >= 0.5:  # Goal reached
                reward += 100
            elif position > 0.3:  # Close to goal
                reward += 10
            elif position > 0:  # Right side of valley
                reward += 1
            
            agent.rewards.append(reward)
            total_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        # Update policy
        loss = agent.update()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)
        moving_avg_rewards.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(list(moving_avg_rewards))
            print(f"Episode {episode}, Reward: {total_reward:.2f}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}, "
                  f"Length: {step + 1}")
    
    env.close()
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
    plt.plot(moving_avg)
    plt.title('Moving Average (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    plt.savefig('vpg_training.png')
    plt.show()
    
    return agent

if __name__ == "__main__":
    train_vpg()