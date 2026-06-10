# train_mountaincar_ppo.py
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

# 使用您提供的 PPOAgent 类
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, epsilon=0.2, K=3):
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.K = K  # PPO更新轮数
        self.memory = []
    
    def get_action(self, state, training=True):
        state_t = torch.FloatTensor(state)
        logits = self.actor(state_t)
        probs = torch.softmax(logits, dim=-1)
        
        if training:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item(), probs.detach()
        else:
            return probs.argmax().item()
    
    def store_transition(self, state, action, log_prob, reward, next_state, done):
        self.memory.append((state, action, log_prob, reward, next_state, done))
    
    def compute_advantages(self, states, rewards, next_states, dones):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            # 确保维度匹配
            next_values = next_values.squeeze()
            # 对于终止状态，下一个状态的价值设为0
            mask = dones.bool()
            next_values[mask] = 0.0
            
            # 计算TD误差作为优势估计
            td_targets = rewards + self.gamma * next_values
            advantages = td_targets - values.squeeze()
            
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            return advantages, td_targets
    
    def update(self):
        if len(self.memory) < 32:  # 小批量更新
            return 0, 0
        
        # 准备数据
        states, actions, old_log_probs, rewards, next_states, dones = zip(*self.memory)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        old_log_probs = torch.FloatTensor(np.array(old_log_probs))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        
        # 计算优势和TD目标
        advantages, td_targets = self.compute_advantages(states, rewards, next_states, dones)
        
        # 多轮PPO更新
        actor_losses = []
        critic_losses = []
        
        for _ in range(self.K):
            # 计算新策略的概率
            logits = self.actor(states)
            new_probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(new_probs)
            new_log_probs = dist.log_prob(actions)
            
            # 计算概率比
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 计算裁剪PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 计算Critic损失
            values = self.critic(states).squeeze()
            critic_loss = nn.MSELoss()(values, td_targets)
            
            # 总损失
            loss = actor_loss + 0.5 * critic_loss
            
            # 更新
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer.step()
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
        
        # 清空记忆
        self.memory = []
        
        return np.mean(actor_losses), np.mean(critic_losses)

# 奖励重塑函数，帮助算法更快学习
def shaped_reward(state, next_state, done, base_reward):
    """为Mountain Car设计的奖励重塑函数"""
    position = next_state[0]
    
    # 基础奖励是每步-1，但我们可以调整
    reward = base_reward
    
    # 如果到达目标，给大量正奖励
    if done and position >= 0.5:
        reward += 100
    
    # 额外奖励：位置越高越好
    # 原始位置范围是[-1.2, 0.6]，我们归一化到[0, 1]
    normalized_position = (position + 1.2) / 1.8
    reward += normalized_position * 0.5
    
    # 额外奖励：速度越高越好（但要注意方向）
    velocity = next_state[1]
    reward += abs(velocity) * 0.1
    
    # 特别奖励：当位置在右侧且速度为正值时
    if position > 0 and velocity > 0:
        reward += 0.5
    
    return reward

# 训练函数
def train_agent(env, agent, episodes=1000, max_steps=200):
    rewards_history = []
    steps_history = []
    success_history = []
    actor_losses = []
    critic_losses = []
    
    # 用于计算移动平均的队列
    reward_window = deque(maxlen=100)
    success_window = deque(maxlen=100)
    
    print("开始训练PPO Agent...")
    start_time = time.time()
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        for step in range(max_steps):
            # 获取动作（训练模式）
            action, log_prob, _ = agent.get_action(state, training=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 使用奖励重塑
            shaped_reward_value = shaped_reward(state, next_state, done, reward)
            
            # 存储转移
            agent.store_transition(state, action, log_prob, shaped_reward_value, next_state, done)
            
            total_reward += shaped_reward_value
            state = next_state
            
            # 定期更新（每收集32个样本）
            if len(agent.memory) >= 32:
                actor_loss, critic_loss = agent.update()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            if done:
                break
        
        # 回合结束时，如果还有未更新的样本，强制更新
        if len(agent.memory) > 0:
            actor_loss, critic_loss = agent.update()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
        
        # 记录历史
        rewards_history.append(total_reward)
        steps_history.append(step + 1)
        
        # 记录是否成功
        success = 1 if terminated else 0
        success_history.append(success)
        success_window.append(success)
        reward_window.append(total_reward)
        
        # 定期输出训练进度
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(list(reward_window))
            avg_success = np.mean(list(success_window))
            avg_steps = np.mean(steps_history[-50:]) if len(steps_history) >= 50 else np.mean(steps_history)
            
            print(f"Episode {episode + 1}/{episodes}:")
            print(f"  Reward: {total_reward:.2f}, Avg Reward (last 100): {avg_reward:.2f}")
            print(f"  Steps: {step + 1}, Success: {success}, Success Rate (last 100): {avg_success:.2f}")
            print(f"  Avg Steps (last 50): {avg_steps:.1f}")
            
            if len(actor_losses) > 0:
                avg_actor_loss = np.mean(actor_losses[-50:]) if len(actor_losses) >= 50 else np.mean(actor_losses)
                avg_critic_loss = np.mean(critic_losses[-50:]) if len(critic_losses) >= 50 else np.mean(critic_losses)
                print(f"  Actor Loss: {avg_actor_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}")
            print("-" * 50)
    
    end_time = time.time()
    print(f"训练完成！总耗时: {end_time - start_time:.2f}秒")
    
    return rewards_history, steps_history, success_history, actor_losses, critic_losses

# 测试训练好的智能体
def test_agent(env, agent, num_episodes=10, render=False):
    print(f"\n测试智能体 ({num_episodes}个回合)...")
    
    success_count = 0
    total_steps = []
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < 200:
            if render:
                env.render()
            
            # 获取动作（测试模式，使用贪婪策略）
            action = agent.get_action(state, training=False)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            steps += 1
            
            if done:
                if terminated:  # 成功到达目标
                    success_count += 1
                break
        
        total_steps.append(steps)
        total_rewards.append(episode_reward)
        
        print(f"回合 {episode + 1}: 奖励={episode_reward:.2f}, 步数={steps}, "
              f"成功={'是' if terminated else '否'}")
    
    print(f"\n测试结果:")
    print(f"成功率: {success_count}/{num_episodes} = {100*success_count/num_episodes:.1f}%")
    print(f"平均步数: {np.mean(total_steps):.1f}")
    print(f"平均奖励: {np.mean(total_rewards):.2f}")
    
    return success_count / num_episodes

# 可视化训练结果
def plot_training_results(rewards_history, steps_history, success_history, 
                         actor_losses=None, critic_losses=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 奖励曲线
    axes[0, 0].plot(rewards_history)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 移动平均奖励
    window_size = 50
    if len(rewards_history) >= window_size:
        moving_avg = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
        axes[0, 1].plot(moving_avg)
        axes[0, 1].set_title(f'Moving Average Reward (window={window_size})')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 步数曲线
    axes[0, 2].plot(steps_history)
    axes[0, 2].axhline(y=200, color='r', linestyle='--', alpha=0.5, label='Max Steps')
    axes[0, 2].set_title('Episode Steps')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Steps')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 成功率
    success_rate = []
    window = 50
    for i in range(len(success_history)):
        start = max(0, i - window + 1)
        rate = np.mean(success_history[start:i+1])
        success_rate.append(rate)
    
    axes[1, 0].plot(success_rate)
    axes[1, 0].set_title('Success Rate')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 演员损失
    if actor_losses and len(actor_losses) > 0:
        axes[1, 1].plot(actor_losses)
        axes[1, 1].set_title('Actor Loss')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 评论家损失
    if critic_losses and len(critic_losses) > 0:
        axes[1, 2].plot(critic_losses)
        axes[1, 2].set_title('Critic Loss')
        axes[1, 2].set_xlabel('Update Step')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ppo_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()

# 主函数
def main():
    # 创建环境
    env = gym.make('MountainCar-v0')
    
    # 获取状态和动作空间维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"环境: MountainCar-v0")
    print(f"状态空间维度: {state_dim}")
    print(f"动作空间维度: {action_dim}")
    print(f"动作含义: 0=向左加速, 1=不加速, 2=向右加速")
    
    # 创建PPO智能体
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=0.0003,      # 学习率
        gamma=0.99,     # 折扣因子
        epsilon=0.2,    # PPO裁剪参数
        K=3            # PPO更新轮数
    )
    
    # 训练参数
    episodes = 1000     # 训练回合数
    max_steps = 200     # 每回合最大步数
    
    # 训练智能体
    rewards_history, steps_history, success_history, actor_losses, critic_losses = train_agent(
        env, agent, episodes=episodes, max_steps=max_steps
    )
    
    # 绘制训练结果
    plot_training_results(rewards_history, steps_history, success_history, 
                         actor_losses, critic_losses)
    
    # 测试智能体
    test_env = gym.make('MountainCar-v0', render_mode='human')
    success_rate = test_agent(test_env, agent, num_episodes=10, render=True)
    test_env.close()
    
    # 保存模型
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }, 'ppo_mountaincar_model.pth')
    
    print("\n模型已保存为 'ppo_mountaincar_model.pth'")
    
    # 显示最终统计
    print("\n训练统计:")
    print(f"总训练回合数: {episodes}")
    print(f"最终100回合平均奖励: {np.mean(rewards_history[-100:]):.2f}")
    print(f"最终100回合成功率: {np.mean(success_history[-100:]):.2%}")
    print(f"最终100回合平均步数: {np.mean(steps_history[-100:]):.1f}")
    
    env.close()

if __name__ == "__main__":
    main()