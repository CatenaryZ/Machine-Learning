## Lecture 8: Reinforcement Learning

> - MDP、价值函数与 Bellman 方程
> - 动态规划与 value iteration
> - Q-learning 与 DQN
> - 策略梯度、REINFORCE 与 Actor-Critic

### 什么是强化学习

- 强化学习研究的是：智能体（agent）如何通过与环境交互来最大化累计回报。

### 马尔可夫决策过程（MDP）

- 一个 MDP 包括：
  - 状态空间（state space）。
  - 动作空间（action space）。
  - 转移函数（transition function）。
  - 奖励函数（reward function）。
  - 折扣因子（discount factor）。

#### 智能体的目标
- 目标是在长期意义下获得尽可能大的总回报。

### 状态价值函数与动作价值函数

- 状态价值函数：

$$
V^\pi(s)=\mathbb E_\pi\left[\sum_{t=0}^{\infty}\gamma^t r_t \mid s_0=s\right].
$$

- 动作价值函数：

$$
Q^\pi(s,a)=\mathbb E_\pi\left[\sum_{t=0}^{\infty}\gamma^t r_t \mid s_0=s, a_0=a\right].
$$

### Bellman 方程

- Bellman 方程给出了价值函数的递归关系，是强化学习的核心数学工具。

#### Bellman 最优方程
- 当考虑最优策略时，可以得到 Bellman optimality equations。

#### 对 Q 值的推导
- 讲义随后逐步推导了 Q-value 对应的 Bellman 方程，并给出了完整推导。

### 解法概览

- 在有模型的情况下，可以用动态规划。
- 在无模型的情况下，可以用基于价值的方法或基于策略的方法。

### 探索与利用

- 强化学习中始终存在一个核心张力：
  - exploration：尝试新动作，收集信息。
  - exploitation：利用当前已知最优动作。

---

### Section 1: Dynamic Programming

### 动态规划方法

- 动态规划假设环境模型已知。
- 它利用 Bellman 递推关系反复更新价值函数和策略。

#### Policy Evaluation
- 给定一个策略，计算该策略下的价值函数。

#### Policy Improvement
- 在已有价值函数基础上改进策略。

#### Policy Iteration
- 交替执行 policy evaluation 与 policy improvement。

#### Value Iteration
- 直接使用 Bellman optimality update 迭代逼近最优价值函数。

#### Q-value Dynamic Programming
- 动态规划也可以直接写在动作价值函数上。

### 动态规划的数学核心

- 讲义通过一个具体小例子展示了 value iteration 的计算过程。
- 包括：
  - 问题设定。
  - 数学形式化。
  - 第一次迭代。
  - 第二次迭代。
  - 收敛趋势。
  - 最优解。

---

### Section 2: Q-learning

### Q-learning 的迭代更新

- Q-learning 用如下更新式迭代逼近最优 Q 函数：

$$
Q(s,a) \leftarrow Q(s,a)
+ \alpha \Bigl[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Bigr].
$$

### 更新公式的理解

- 讲义把这个公式拆成多个部分来理解：
  - 当前估计值。
  - 目标值。
  - 学习率控制下的修正幅度。

### 探索-利用困境

- 如果永远选择当前最优动作，就可能错过更好的策略。
- 如果总在探索，就无法稳定积累收益。

### Epsilon-Greedy

- 一个常见方案是 $\varepsilon$-greedy：
  - 以较大概率选择当前最好动作。
  - 以较小概率随机探索。

### 什么是 DQN

- 表格式 Q-learning 在状态空间很大时不再可行。
- Deep Q-Network（DQN）用神经网络近似 $Q(s,a)$。

### Q-learning 表格法的局限

- 状态太多时，表格无法存储或泛化。

### DQN 的核心思路

- 用网络把状态映射到各动作的 Q 值。
- 这样模型能够在相似状态之间共享表示。

### DQN 的困难与改进

- 直接训练 DQN 往往不稳定。
- 讲义指出了 DQN 为稳定训练而采用的关键创新。

---

### Section 3: Policy-based Method

### 基于策略的方法

- 这类方法不再先学价值函数再导出策略，而是直接优化参数化策略

$$
\pi_\theta(a\mid s).
$$

### 数学框架

- 目标是最大化策略诱导下的期望回报。

### Monte Carlo 估计

- 一种直接方法是用采样轨迹估计回报。

#### Step 1: 计算回报
- 先对每条轨迹中的每个时间步计算 return。

#### Step 2: 对数概率梯度
- 利用

$$
\nabla_\theta \log \pi_\theta(a\mid s)
$$

来构造梯度估计。

#### Step 3: 梯度累积
- 沿整条轨迹累积贡献，得到 REINFORCE 算法。

### 完整的 REINFORCE

- 讲义给出了完整算法流程。

### Policy Gradient Theorem

- 接着，讲义系统推导了 policy gradient theorem。
- 推导过程包括：
  - 从 Bellman 方程开始。
  - 对价值函数求梯度。
  - 展开 $\nabla_\theta Q^{\pi_\theta}(s,a)$。
  - 做递归展开。
  - 引入 discounted state distribution。
  - 重排求和。
  - 使用 log-derivative trick。
  - 最后得到 trajectory-based form。

#### 结论
- 该定理把目标函数梯度写成关于策略对数梯度和回报/价值的期望形式，从而把理论推导和实际算法连接起来。

### 关键理解

- 这一结果的重要性在于：它给出了可以直接优化策略参数的可计算梯度表达式。

### 从定理到算法

- 有了 policy gradient theorem，就可以构造一系列实际可训练的策略优化算法。

### Advantage Function

- advantage function 用来度量某个动作相对于基准水平究竟“好多少”。
- 它常用于降低梯度估计的方差。

### Critic 的类型

- actor-critic 方法在策略学习之外，再引入一个 critic 来估计价值。

### 典型算法

- A2C：Advantage Actor-Critic。
- PPO：Proximal Policy Optimization。
- SAC：Soft Actor-Critic。

### 小结

- 强化学习从 MDP 出发，用 Bellman 方程刻画价值递推。
- 在已知模型时，可用动态规划。
- 在未知模型时，可以用 Q-learning / DQN 等 value-based 方法，也可以用 REINFORCE、A2C、PPO、SAC 等 policy-based / actor-critic 方法。
