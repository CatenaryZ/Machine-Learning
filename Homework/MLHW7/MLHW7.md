# 机器学习的数学原理 HW7

**蔡天卓 数31 2023011246**

## 1 Problem: Gridworld Navigation

Consider a simple \(3\times 3\) Gridworld where an agent navigates to reach a goal state. The states are labeled \(s_{1}\) through \(s_{9}\) as shown below:

| | | |
| ---- | ---- | ---- |
| \(s_{1}\) | \(s_{2}\) | \(s_{3}\) |
| \(s_{4}\) | \(s_{5}\) | \(s_{6}\) |
| \(s_{7}\) | \(s_{8}\) | \(s_{9}\) |


The agent can take four possible actions: \(\mathcal{A}=\{\text{Up},\text{Down},\text{Left},\text{Right}\}\). State \(s_{9}\) is a terminal goal state with reward +10. All other transitions yield a reward of \(-1\). If the agent attempts to move outside the grid, it remains in the current state and receives the transition reward.

### Parameters

* Discount factor: \(\gamma=0.9\)

* The environment is deterministic

* The terminal state \(s_{9}\) has value \(V(s_{9})=0\) (no further rewards after termination)

#### Part 1: Q-value Calculation

Assume the agent follows a uniform random policy: \(\pi(a|s)=0.25\) for all \(a\in\mathcal{A}\) in all non-terminal states.

1. Compute \(Q(s_{5},\text{Right})\) using the Bellman expectation equation: \[Q^{\pi}(s,a)=\sum_{s^{\prime}}P(s^{\prime}|s,a)\left[R(s,a,s^{\prime})+\gamma V^{\pi}(s^{\prime})\right]\] where \(V^{\pi}(s^{\prime})\) is given by the following current value function estimates (after some iterations of policy evaluation): 

| | | |
| ---- | ---- | ---- |
| $V(s_{1})=2.1$ | $V(s_{2})=3.5$ | $V(s_{3})=5.0$ |
| $V(s_{4})=1.8$ | $V(s_{5})=4.2$ | $V(s_{6})=6.8$ |
| $V(s_{7})=0.5$ | $V(s_{8})=2.9$ | $V(s_{9})=0$ |

2. Compute \(Q(s_{2},\text{Down})\) using the same value function.

#### Part 2: State Value Calculation

Now consider a different policy where:

* In \(s_{5}\): \(\pi(\text{Right}|s_{5})=0.7\), \(\pi(\text{Down}|s_{5})=0.3\), \(\pi(\text{Up}|s_{5})=0\), \(\pi(\text{Left}|s_{5})=0\)

* In \(s_{8}\): \(\pi(\text{Right}|s_{8})=0.6\), \(\pi(\text{Up}|s_{8})=0.4\), \(\pi(\text{Down}|s_{8})=0\), \(\pi(\text{Left}|s_{8})=0\)

* Assume deterministic transitions with rewards as specified earlier

The Q-values for these states under some policy are given as:

\begin{tabular}{l|c} State-Action pair & Q-value \\ \hline \(Q(s_{5},\text{Right})\) & 8.2 \\ \(Q(s_{5},\text{Down})\) & 6.5 \\ \(Q(s_{8},\text{Right})\) & 9.8 \\ \(Q(s_{8},\text{Up})\) & 7.3 \\

1. Compute \(V^{\pi}(s_{5})\) using the policy-weighted average: \[V^{\pi}(s)=\sum_{a\in\mathcal{A}}\pi(a|s)Q^{\pi}(s,a)\]

2. Compute \(V^{\pi}(s_{8})\) using the same method.

#### Part 3: Optimal Values

Assuming the agent finds the optimal policy \(\pi^{*}\):

1. Write the Bellman optimality equation for \(Q^{*}(s,a)\).

2. If \(Q^{*}(s_{6},\text{Right})=9.0\) and all other actions from \(s_{6}\) have lower Q-values, what is \(V^{*}(s_{6})\)?

3. If \(V^{*}(s_{2})=8.4\) and taking "Right" from \(s_{2}\) leads deterministically to \(s_{3}\) with \(V^{*}(s_{3})=9.0\), what is \(Q^{*}(s_{2},\text{Right})\)? (Use the reward of \(-1\) for this transition)

#### Solution of Part 1

**1.**  
根据贝尔曼期望方程：  
$$
Q^{\pi}(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^{\pi}(s') \right]
$$  

环境是确定性的，从 \(s_5\) 执行动作"Right"必然转移到 \(s_6\)，且奖励 \(R = -1\)。已知 \(\gamma = 0.9\)，\(V(s_6) = 6.8\)，代入得：  
$$
Q(s_5, Right) = 1 \times \left[ (-1) + 0.9 \times 6.8 \right] = -1 + 6.12 = 5.12
$$

**2.**  
从 \(s_2\) 执行动作"Down"必然转移到 \(s_5\)，奖励 \(R = -1\)，已知 \(V(s_5) = 4.2\)：  
$$
Q(s_2, Down) = 1 \times \left[ (-1) + 0.9 \times 4.2 \right] = -1 + 3.78 = 2.78
$$

#### Solution of Part 2

**1.**  
$$
V^{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) Q^{\pi}(s, a)
$$

在 \(s_5\) 的策略为：\(\pi(Right \mid s_5) = 0.7\)，\(\pi(Down \mid s_5) = 0.3\)，其他动作概率为 0。已知 \(Q(s_5, Right) = 8.2\)，\(Q(s_5, Down) = 6.5\)：  
$$
V^{\pi}(s_5) = 0.7 \times 8.2 + 0.3 \times 6.5 = 5.74 + 1.95 = 7.69
$$

**2.**  
在 \(s_8\) 的策略为：\(\pi(Right \mid s_8) = 0.6\)，\(\pi(Up \mid s_8) = 0.4\)，其他动作概率为 0。已知 \(Q(s_8, Right) = 9.8\)，\(Q(s_8, Up) = 7.3\)：  
\[
V^{\pi}(s_8) = 0.6 \times 9.8 + 0.4 \times 7.3 = 5.88 + 2.92 = 8.80
\]

#### Solution of Part 3

**1.**  
\[
Q^{*}(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \max_{a'} Q^{*}(s', a') \right]
\]

**2.**  
最优状态价值函数：  
\[
V^{*}(s) = \max_{a} Q^{*}(s, a)
\] 

已知 \(Q^{*}(s_6, Right) = 9.0\)，且其他动作的 Q 值均小于 9.0，因此：  
\[
V^{*}(s_6) = 9.0
\]

**3.**  
从 \(s_2\) 执行动作Right必然转移到 \(s_3\)，奖励 \(R = -1\)，已知 \(V^{*}(s_3) = 9.0\)：  
\[
Q^{*}(s_2, Right) = 1 \times \left[ (-1) + 0.9 \times 9.0 \right] = -1 + 8.1 = 7.1
\]


## 2 Montain car

The environment is described in https://gymnasium.farama.org/environments/classic_control/mountain_car/. Find the optimal policy by using neural network and following algorithm

1. Deep Q learning

2. Vanilla policy gradient.

3. Actor-Critic.

4. PPO.


## 1 问题：网格世界导航

考虑一个简单的 \(3\times 3\) 网格世界，其中智能体导航以到达目标状态。状态标记为 \(s_{1}\) 到 \(s_{9}\)，如下所示：

\begin{tabular}{|c|c|c|} \hline \(s_{1}\) & \(s_{2}\) & \(s_{3}\) \\ \hline \(s_{4}\) & \(s_{5}\) & \(s_{6}\) \\ \hline \(s_{7}\) & \(s_{8}\) & \(s_{9}\) \\ \hline \end{tabular}

智能体可以采取四种可能的动作：\(\mathcal{A}=\{\text{上},\text{下},\text{左},\text{右}\}\)。状态 \(s_{9}\) 是终止目标状态，奖励为 +10。所有其他转移产生的奖励为 \(-1\)。如果智能体试图移动到网格外，它将保持在当前状态并接收转移奖励。

### 参数

* *折扣因子：\(\gamma=0.9\)

* *环境是确定性的

* *终止状态 \(s_{9}\) 的价值为 \(V(s_{9})=0\)（终止后无进一步奖励）

#### 第一部分：Q值计算

假设智能体遵循均匀随机策略：在所有非终止状态中，对于所有 \(a\in\mathcal{A}\)，\(\pi(a|s)=0.25\)。

1. 使用贝尔曼期望方程计算 \(Q(s_{5},\text{右})\)： \[Q^{\pi}(s,a)=\sum_{s^{\prime}}P(s^{\prime}|s,a)\left[R(s,a,s^{\prime})+\gamma V^{\pi}(s^{\prime})\right]\] 其中 \(V^{\pi}(s^{\prime})\) 由以下当前价值函数估计给出（经过一些策略评估迭代后）： \[\begin{array}[]{|l|l|l|}\hline V(s_{1})=2.1&V(s_{2})=3.5&V(s_{3})=5.0\\ \hline V(s_{4})=1.8&V(s_{5})=4.2&V(s_{6})=6.8\\ \hline V(s_{7})=0.5&V(s_{8})=2.9&V(s_{9})=0\\ \hline\end{array}\]

2. 使用相同的价值函数计算 \(Q(s_{2},\text{下})\)。

#### 第二部分：状态价值计算

现在考虑一个不同的策略，其中：

* *在 \(s_{5}\)：\(\pi(\text{右}|s_{5})=0.7\)，\(\pi(\text{下}|s_{5})=0.3\)，\(\pi(\text{上}|s_{5})=0\)，\(\pi(\text{左}|s_{5})=0\)

* *在 \(s_{8}\)：\(\pi(\text{右}|s_{8})=0.6\)，\(\pi(\text{上}|s_{8})=0.4\)，\(\pi(\text{下}|s_{8})=0\)，\(\pi(\text{左}|s_{8})=0\)

* *假设转移是确定性的，奖励如前所述

这些状态在某个策略下的Q值如下：

\begin{tabular}{l|c} 状态-动作对 & Q值 \\ \hline \(Q(s_{5},\text{右})\) & 8.2 \\ \(Q(s_{5},\text{下})\) & 6.5 \\ \(Q(s_{8},\text{右})\) & 9.8 \\ \(Q(s_{8},\text{上})\) & 7.3 \\

1. 使用策略加权平均计算 \(V^{\pi}(s_{5})\)： \[V^{\pi}(s)=\sum_{a\in\mathcal{A}}\pi(a|s)Q^{\pi}(s,a)\]

2. 使用相同方法计算 \(V^{\pi}(s_{8})\)。


第三部分：最优值

假设智能体找到最优策略 \(\pi^{*}\)：

1. 写出 \(Q^{*}(s,a)\) 的贝尔曼最优性方程。

2. 如果 \(Q^{*}(s_{6},\text{右})=9.0\) 并且从 \(s_{6}\) 出发的所有其他动作的Q值都较低，那么 \(V^{*}(s_{6})\) 是多少？

3. 如果 \(V^{*}(s_{2})=8.4\) 并且从 \(s_{2}\) 采取“右”动作确定性地导致 \(s_{3}\)，且 \(V^{*}(s_{3})=9.0\)，那么 \(Q^{*}(s_{2},\text{右})\) 是多少？（对于此转移，使用奖励 \(-1\)）

## 2 山地车

环境描述在 https://gymnasium.farama.org/environments/classic_control/mountain_car/

通过使用神经网络和以下算法找到最优策略：

1. Deep Q learning

2. Vanilla policy gradient.

3. Actor-Critic.

4. PPO.
