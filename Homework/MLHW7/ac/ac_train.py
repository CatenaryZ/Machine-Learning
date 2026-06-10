import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import random

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)
    
    def get_action(self, state, device='cpu'):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.forward(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ACAgent:
    def __init__(self, state_dim, action_dim, actor_lr=0.001, critic_lr=0.001, gamma=0.99):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        
    def select_action(self, state):
        action, log_prob = self.actor.get_action(state, self.device)
        value = self.critic(torch.FloatTensor(state).unsqueeze(0).to(self.device))
        return action, log_prob, value
    
    def update(self, log_probs, values, rewards, next_state, done):
        # Calculate returns and advantages
        returns = []
        advantages = []
        R = 0 if done else self.critic(torch.FloatTensor(next_state).unsqueeze(0).to(self.device)).item()
        
        # Calculate returns from the end
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, device=self.device).unsqueeze(1)
        values = torch.cat(values)
        
        # Calculate advantages
        advantages = returns - values
        
        # Calculate actor loss
        actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
        
        # Calculate critic loss
        critic_loss = advantages.pow(2).mean()
        
        # Update networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()

def train_ac(env_name="MountainCar-v0", num_episodes=1000, render_freq=100):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = ACAgent(state_dim, action_dim)
    
    episode_rewards = []
    moving_avg_rewards = []
    moving_window = deque(maxlen=100)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        log_probs = []
        values = []
        rewards = []
        
        while not (done or truncated):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Optional: Modify reward
            if not done:
                reward = reward + 0.1 * abs(next_state[1])  # Reward for velocity
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            total_reward += reward
            
            # Update every step
            if len(rewards) > 0:
                actor_loss, critic_loss = agent.update(log_probs, values, rewards, next_state, done or truncated)
                log_probs = []
                values = []
                rewards = []
            
            state = next_state
        
        # Record rewards
        episode_rewards.append(total_reward)
        moving_window.append(total_reward)
        moving_avg = np.mean(moving_window)
        moving_avg_rewards.append(moving_avg)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"Reward: {total_reward:.2f}, "
                  f"Moving Avg: {moving_avg:.2f}")
        
        # Render occasionally
        if (episode + 1) % render_freq == 0:
            test_ac_agent(agent, env_name, render=True)
    
    # Save the models
    torch.save(agent.actor.state_dict(), 'ac_actor_mountaincar.pth')
    torch.save(agent.critic.state_dict(), 'ac_critic_mountaincar.pth')
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.6)
    plt.plot(moving_avg_rewards, linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Actor-Critic Training Rewards')
    plt.legend(['Episode Reward', 'Moving Avg (100)'])
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ac_training.png')
    plt.show()
    
    env.close()
    return agent

def test_ac_agent(agent, env_name="MountainCar-v0", num_episodes=5, render=True):
    env = gym.make(env_name, render_mode="human" if render else None)
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action, _, _ = agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        
        total_rewards.append(total_reward)
        print(f"Test Episode {episode+1}: Reward = {total_reward}")
    
    env.close()
    print(f"Average test reward: {np.mean(total_rewards):.2f}")
    return total_rewards

if __name__ == "__main__":
    agent = train_ac(num_episodes=500, render_freq=50)