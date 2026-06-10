import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import random

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor头 - 策略网络
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic头 - 价值网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化参数
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        
    def forward(self, x):
        shared_features = self.shared(x)
        action_logits = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_logits, state_value
    
    def get_action(self, state):
        action_logits, state_value = self.forward(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return action, action_dist.log_prob(action), state_value.squeeze(), action_dist.entropy()

class PPOMemory:
    """PPO经验回放缓冲区"""
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
        self.batch_size = batch_size
        
    def store(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
    def get_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.states), np.array(self.actions), np.array(self.log_probs), \
               np.array(self.values), np.array(self.rewards), np.array(self.dones), batches

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, ppo_epochs=10, batch_size=64, entropy_coef=0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        
        self.memory = PPOMemory(batch_size)
        
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value, entropy = self.model.get_action(state_tensor)
        
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, next_value, rewards, values, dones):
        """计算广义优势估计(GAE)"""
        gae = 0
        returns = []
        advantages = []
        
        # 反向计算
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[step + 1]
                next_values = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_values * next_non_terminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            
        return np.array(advantages), np.array(returns)
    
    def update(self):
        """PPO更新步骤"""
        # 获取经验数据
        states, actions, old_log_probs, values, rewards, dones, batches = self.memory.get_batches()
        
        # 计算GAE和回报
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(states[-1]).unsqueeze(0).to(self.device)
            _, next_value = self.model(next_state_tensor)
            next_value = next_value.squeeze().item()
        
        advantages, returns = self.compute_gae(next_value, rewards, values, dones)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # 归一化优势
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO多次更新
        for _ in range(self.ppo_epochs):
            for batch in batches:
                # 获取批量数据
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_log_probs = old_log_probs[batch]
                batch_returns = returns[batch]
                batch_advantages = advantages[batch]
                
                # 前向传播
                action_logits, state_values = self.model(batch_states)
                action_probs = torch.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                
                # 计算新log probs和熵
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 计算代理损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 计算critic损失
                critic_loss = 0.5 * (state_values.squeeze() - batch_returns).pow(2).mean()
                
                # 总损失
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optimizer.step()
        
        # 清空内存
        self.memory.clear()
        
        return actor_loss.item(), critic_loss.item()

def train_ppo(env_name="MountainCar-v0", num_episodes=1000, max_timesteps=500, update_freq=10, render_freq=50):
    # 创建环境
    env = gym.make(env_name, max_episode_steps=max_timesteps)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(state_dim, action_dim, lr=3e-4, gamma=0.99, 
                     gae_lambda=0.95, clip_epsilon=0.2, ppo_epochs=4,
                     batch_size=32, entropy_coef=0.01)
    
    episode_rewards = []
    moving_avg_rewards = []
    moving_window = deque(maxlen=100)
    success_rate = deque(maxlen=100)
    
    timestep = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        success = False
        
        # 自定义奖励参数
        last_position = state[0]
        swing_count = 0
        last_action = None
        
        while not (done or truncated):
            # 选择动作
            action, log_prob, value = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, truncated, _ = env.step(action)
            
            # 自定义奖励函数
            position = next_state[0]
            velocity = next_state[1]
            
            # 1. 基础奖励
            base_reward = -1 if position < 0.5 else 0
            
            # 2. 位置奖励
            position_reward = 100 * position if position > last_position else 0
            
            # 3. 摆动奖励（鼓励反直觉策略）
            if last_action is not None:
                if action != last_action:  # 动作变化
                    swing_count += 1
                    swing_reward = min(swing_count * 0.5, 10)
                else:
                    swing_reward = -0.1  # 惩罚连续相同动作
            else:
                swing_reward = 0
            
            # 4. 速度奖励
            velocity_reward = 50 * abs(velocity)
            
            # 5. 进展奖励
            progress = position - last_position
            progress_reward = 200 * progress if progress > 0 else 0
            
            # 组合奖励
            total_reward = (base_reward + position_reward + swing_reward + 
                          velocity_reward + progress_reward)
            
            # 存储经验
            agent.memory.store(state, action, log_prob, value, total_reward, done)
            
            episode_reward += total_reward
            timestep += 1
            
            # 定期更新
            if timestep % update_freq == 0:
                actor_loss, critic_loss = agent.update()
            
            # 更新状态
            state = next_state
            last_position = position
            last_action = action
            
            # 检查是否成功
            if position >= 0.5:
                success = True
                done = True
        
        # 如果回合结束，进行一次更新
        if len(agent.memory.states) > 0:
            actor_loss, critic_loss = agent.update()
        
        # 记录结果
        episode_rewards.append(episode_reward)
        moving_window.append(episode_reward)
        moving_avg = np.mean(moving_window)
        moving_avg_rewards.append(moving_avg)
        
        success_rate.append(1 if success else 0)
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_success = np.mean(success_rate) * 100
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"Reward: {episode_reward:.2f}, "
                  f"Avg Reward: {moving_avg:.2f}, "
                  f"Success Rate: {avg_success:.1f}%")
        
        # 定期测试
        if (episode + 1) % render_freq == 0:
            test_ppo_agent(agent, env_name, num_episodes=3)
    
    # 保存模型
    torch.save({
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }, 'ppo_mountaincar_optimized.pth')
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards, alpha=0.6, label='Episode Reward')
    plt.plot(moving_avg_rewards, linewidth=2, label='Moving Avg (100)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('PPO Training Rewards')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    success_rates = []
    for i in range(len(episode_rewards)):
        start = max(0, i-99)
        success_rates.append(np.mean(list(success_rate)[start:i+1]) * 100)
    plt.plot(success_rates)
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate Over Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ppo_training_optimized.png')
    plt.show()
    
    env.close()
    return agent

def test_ppo_agent(agent, env_name="MountainCar-v0", num_episodes=5, render=True):
    env = gym.make(env_name, render_mode="human" if render else None)
    total_rewards = []
    success_count = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        truncated = False
        
        positions = []
        actions = []
        velocities = []
        
        while not (done or truncated):
            action, _, _ = agent.select_action(state)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            positions.append(state[0])
            actions.append(action)
            velocities.append(state[1])
        
        total_rewards.append(total_reward)
        
        # 检查是否成功
        if state[0] >= 0.5:
            success_count += 1
            print(f"Test Episode {episode+1}: SUCCESS in {steps} steps!")
            
            # 分析成功策略
            print("  Strategy analysis:")
            for i in range(min(20, len(actions))):
                action_names = ["Left", "No-op", "Right"]
                print(f"    Step {i+1}: {action_names[actions[i]]}, Position: {positions[i]:.3f}, Velocity: {velocities[i]:.3f}")
        else:
            print(f"Test Episode {episode+1}: Failed in {steps} steps, final position: {state[0]:.3f}")
        
        # 统计动作分布
        left_actions = sum(1 for a in actions if a == 0)
        right_actions = sum(1 for a in actions if a == 2)
        noop_actions = sum(1 for a in actions if a == 1)
        
        print(f"  Actions: Left={left_actions}, Right={right_actions}, No-op={noop_actions}")
        
        # 绘制轨迹
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(positions)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Position')
        ax1.set_title(f'Position Trajectory (Final: {positions[-1]:.3f})')
        ax1.axhline(y=0.5, color='r', linestyle='--', label='Goal')
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(velocities)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Velocity')
        ax2.set_title('Velocity Profile')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    env.close()
    
    success_rate = success_count / num_episodes * 100
    print(f"\nTest Results: Success Rate = {success_rate:.1f}%, Average Reward = {np.mean(total_rewards):.2f}")
    
    return total_rewards

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 启用CUDA优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # 训练
    agent = train_ppo(num_episodes=500, max_timesteps=400, update_freq=5, render_freq=50)