以下是对《Lecture 8: Reinforcement learning》PPT的**逐页中文翻译**，格式与样式符合您的要求：

---

### 第 1 页

#### Lecture 8: 强化学习

谢丹  
清华大学数学系  
2025年12月10日

---

### 第 2 页

#### 什么是强化学习？

**定义**  
强化学习是一种机器学习范式，**智能体**通过与**环境**交互来学习决策，以最大化累积**奖励**。

- **智能体**
  - 奖励 \( R_i \)
  - 动作 \( A_i \)
  - \( R_{e1} \)  
  - 环境

---

### 第 3 页

#### 马尔可夫决策过程

MDP由五元组 \((S, A, P, R, \gamma)\) 定义：

**核心组成部分**

- \(S\)：状态集合（状态空间）
- \(A\)：动作集合（动作空间）
- \(P(s'|s, a)\)：状态转移概率函数
- \(R(s, a, s')\)：奖励函数
- \(\gamma\)：折扣因子（\(0 \leq \gamma \leq 1\)）

**马尔可夫性质**  
未来仅取决于当前状态与动作：

\[P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1}|s_t, a_t)\]

---

### 第 4 页

#### 网格世界：状态空间

状态空间 \( S \)

- 4×4 网格世界
- 状态表示智能体位置
- \( S = \{(x, y) | x \in \{0, 1, 2, 3\}, y \in \{0, 1, 2, 3\}\} \)
- \( |S| = 16 \) 个状态

\[\begin{array}{c|c}
(0,3)(1,3)(2,3) & 终点 \\
\hline
(0,2)(1,2)(2,2)(3,2) & \\
(0,1)(1,1)(2,1)(3,1) & \\
\end{array}\]

起点 \( 1,0)(2,0)(3,0)\)

---

### 第 5 页

#### 网格世界：动作空间

动作空间 \( A \)

- 四种可能移动
- \( A = \{上, 右, 下, 左\} \)
- \( A = \{0, 1, 2, 3\} \)（数值表示）

上（0）

左（3）  
智能体  
右（1）

下（2）

---

### 第 6 页

#### 转移函数

**转移概率 \( P(s'|s, a) \)**  
确定性转移，含边界处理：

- **上**：  
  \[  P((x, y-1)|(x, y), 上) = 1 \text{ 若 } y > 0\]
- **右**：  
  \[  P((x+1, y)|(x, y), 右) = 1 \text{ 若 } x < 3\]
- **下**：  
  \[  P((x, y+1)|(x, y), 下) = 1 \text{ 若 } y < 3\]
- **左**：  
  \[  P((x-1, y)|(x, y), 左) = 1 \text{ 若 } x > 0\]

**边界情况**：若移动受阻，智能体保持原状态。

**终止状态**  
状态 \((3,3)\) 为终止状态：  
\[P((3,3)|(3,3), a) = 1 \text{ 对所有 } a \in A\]

---

### 第 7 页

#### 奖励函数

奖励函数 \( R(s, a, s') \)

- **目标奖励**：  
  \[  R(s, a, s') = +10 \text{ 若 } s' = (3,3)\]
- **步数惩罚**：  
  \[  R(s, a, s') = -1 \text{ 其他情况}\]

**目的**

- 正奖励鼓励到达目标
- 负奖励鼓励高效（更短路径）
- 智能体学习平衡探索与利用

**注意**  
奖励是稀疏的：仅在到达特定状态时非零

---

### 第 8 页

#### 折扣因子与目标

**折扣因子**
- \(\gamma = 0.9\)
- 更重视即时奖励
- 确保无限和收敛

---

### 第 9 页

#### 智能体的目标

智能体的目标是最大化期望折扣回报：

\[G_t = \mathbb{E} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \right]\]

寻找最优策略 \(\pi^* : S \to A\)，使得从任意状态出发的 \(G_t\) 最大。注意 \(\pi\) 是随机选择。

---

### 第 10 页

#### 状态价值函数与动作价值函数

**状态价值函数 \( V_{\pi}(s) \)**  
从状态 \(s\) 开始，遵循策略 \(\pi\) 的期望回报：

\[V_{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t = s]\]

**动作价值函数 \( Q_{\pi}(s, a) \)**  
从状态 \(s\) 开始，执行动作 \(a\)，然后遵循策略 \(\pi\) 的期望回报：

\[Q_{\pi}(s, a) = \mathbb{E}_{\pi}[G_t | S_t = s, A_t = a]\]

**关系**

\[V_{\pi}(s) = \sum_{a \in A} \pi(a|s) Q_{\pi}(s, a)\]

---

### 第 11 页

#### 贝尔曼方程

**贝尔曼期望方程**

\[V^{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) \left[ r + \gamma V^{\pi}(s') \right]\]

\[Q^{\pi}(s,a) = \sum_{s',r} p(s',r|s,a) \left[ r + \gamma \sum_{a'} \pi(a'|s') Q_{\pi}(s',a') \right]\]

---

### 第 12 页

#### 贝尔曼最优方程

**最优状态价值函数**：

\[V_{*}(s) = \max_{a} \sum_{s',r} P(s',r|s,a) \left[ r + \gamma V_{*}(s') \right]\]

**最优动作价值函数**：

\[Q_{*}(s,a) = \sum_{s',r} P(s',r|s,a) \left[ r + \gamma \max_{a'} Q_{*}(s',a') \right]\]

> p.s.$$\begin{aligned}V_{*}(s) - \max_{a} Q_{*}(s,a) =& \max_{a} \sum_{s',r} P(s',r|s,a) \left[ r + \gamma V_{*}(s') \right] \\ &- \max_{a} \sum_{s',r} P(s',r|s,a) \left[ r + \gamma \max_{a'} Q_{*}(s',a') \right]\\ =& \max_{a} \sum_{s',r} P(s',r|s,a) \gamma \left[V_{*}(s') - \max_{a'} Q_{*}(s',a')\right],\end{aligned}$$ 依次类推, 可以递归到终止条件的s'. 而在终止条件处 $V_{\pi^*}(s) = \sum_{a \in A} \pi^*(a|s) Q_{\pi^*}(s, a) = Q_{*}(s, a)$, 从而 $V_{*}(s) = \max\limits_{a} Q_{*}(s,a)$

**最优策略**：

\[\pi_{*}(s) = \arg \max_{a} Q_{*}(s,a) = \arg \max_{a} \sum_{s',r} P(s',r|s,a) \left[ r + \gamma V_{*}(s') \right]\]

---

### 第 13 页

#### Q 值贝尔曼方程推导

- 起点：\( q_\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a] \)
- 分解回报：\( G_t = R_{t+1} + \gamma G_{t+1} \)
- 取期望：

\[q_\pi(s, a) = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} | S_t = s, A_t = a] \\
= \mathbb{E}[R_{t+1} | s, a] + \gamma \mathbb{E}_\pi[G_{t+1} | s, a]\]

---

### 第 14 页

#### 完整推导

**步骤1：即时奖励**

\[\mathbb{E}[R_{t+1} \mid s, a] = \sum_{s', r} r \cdot p(s', r|s, a)\]


> 我们想计算在给定 \( s, a \) 的条件下，即时奖励 \( R_{t+1} \) 的期望值。环境模型 \( p(s'， r|s, a) \) 给出了在 \( s, a \) 下，转移到 \( s' \) 并获得奖励 \( r \) 的联合概率。要得到奖励的期望，我们需要对所有可能的 \( (s'， r) \) 结果进行加权平均，权重就是其概率 \( p \)，而值就是奖励 \( r \) 本身。这个求和已经涵盖了环境转移的所有可能性。

**步骤2：未来回报**

\[\mathbb{E}_{\pi}[G_{t+1} \mid s, a] = \sum_{s'} p(s'|s, a) \cdot \mathbb{E}_{\pi}[G_{t+1} \mid S_{t+1} = s']\]
\[= \sum_{s'} p(s'|s, a) \cdot v_{\pi}(s')\]

**步骤3：替换 \( v_{\pi}(s') \)**

\[v_{\pi}(s') = \sum_{a'} \pi(a'|s') q_{\pi}(s', a')\]

---

### 第 15 页

#### 最终贝尔曼期望方程

**Q 值的贝尔曼期望方程**

\[q_\pi(s,a) = \sum_{s',r} p(s',r|s,a) \left[ r + \gamma \sum_{a'} \pi(a'|s') q_\pi(s',a') \right]\]

**解释**：
\((s,a)\) 的 Q 值 =

- 期望即时奖励 \(r\)
- 加上折扣后的下一状态动作对 Q 值的期望
- 下一状态的动作遵循策略 \(\pi\)

---

### 第 16 页

#### 求解方法

**动态规划**
- 值迭代
- 策略迭代
- 需要模型知识（\(P, R\)）

**强化学习**
- Q学习（离策略）
- SARSA（同策略）
- 无模型学习

---

### 第 17 页

#### 探索与利用

**基本权衡**

- **探索**：尝试新动作以发现更好奖励
- **利用**：选择已知好动作以最大化奖励

**常用策略**
- \(\epsilon\)-贪心：以概率 \(\epsilon\) 随机探索
- Softmax：基于概率分布选择动作
- 上置信界：平衡估计值与不确定性

---

### 第 18 页

#### 第一节：动态规划

---

### 第 19 页

#### 动态规划方法

- **完美模型**：已知 \( P(s' \mid s, a) \) 和 \( R(s, a, s') \)
- **马尔可夫性质**：未来状态仅取决于当前状态与动作

---

### 第 20 页

#### 策略评估

**问题**：给定 \(\pi\)，求 \(v_{\pi}\)

**迭代更新**：

\[v_{k+1}(s) = \sum_{a} \pi(a|s) \sum_{s',r} p(s',r|s,a) \left[ r + \gamma v_k(s') \right]\]

**原地更新**：

\[v(s) \leftarrow \sum_{a} \pi(a|s) \sum_{s',r} p(s',r|s,a) \left[ r + \gamma v(s') \right]\]

---

### 第 21 页

#### 策略改进

**策略改进定理**：若 \(\pi\) 与 \(\pi'\) 满足：

\[q_{\pi}(s, \pi'(s)) \geq v_{\pi}(s) \quad \forall s \in \mathcal{S}\]

则：

\[v_{\pi'}(s) \geq v_{\pi}(s) \quad \forall s \in \mathcal{S}\]

**贪心改进**：

\[\pi'(s) = \arg \max_{a} q_{\pi}(s, a) = \arg \max_{a} \sum_{s', r} p(s', r|s, a) \left[ r + \gamma v_{\pi}(s') \right]\]

---

### 第 22 页

#### 策略迭代

**算法**：

1. **策略评估**：\( V_{\pi_k} \)
2. **策略改进**：  
   \[    \pi_{k+1}(s) = \arg\max_{a} \sum_{s',r} p(s',r|s,a) \left[ r + \gamma v_{\pi_k}(s') \right]\]

**重复直至**  
   \[    \pi_{k+1} = \pi_k\]

**单调改进**：

\[v_{\pi_{k+1}}(s) \geq v_{\pi_k}(s) \quad \forall s \in S\]

**收敛性**：对有限 MDP，有限次迭代收敛至 \(\pi_*\)

---

### 第 23 页

#### 值迭代

直接计算最优值：

\[ v_{k+1}(s) = \max_a \sum_{s',r} p(s',r|s,a) [r + \gamma v_k(s')] \]

作为贝尔曼最优算子：

\[ v_{k+1} = T^* v_k \]

其中 \( T^* \) 是贝尔曼最优算子

**收敛性**：

\[\lim_{k \to \infty} v_k = v_*\]
且
\[\|v_k - v_*\|_{\infty} \leq \gamma^k \|v_0 - v_*\|_{\infty}\]

提取最优策略：

\[\pi_*(s) = \arg \max_a \sum_{s',r} p(s',r|s,a) [r + \gamma v_*(s')]\]

---

### 第 24 页

#### Q 值动态规划

**Q 值策略评估**：

\[q_{k+1}(s, a) = \sum_{s', r} p(s', r|s, a) \left[ r + \gamma \sum_{a'} \pi(a'|s') q_k(s', a') \right]\]

**Q 值值迭代**：

\[q_{k+1}(s, a) = \sum_{s', r} p(s', r|s, a) \left[ r + \gamma \max_{a'} q_k(s', a') \right]\]

---

### 第 25 页

#### 强化学习中动态规划的数学核心

1. **MDP** = \((S, \mathcal{A}, P, \mathcal{R}, \gamma)\)
2. **贝尔曼方程**：
   \[   v_{\pi}(s) = \mathbb{E}_{\pi}[r + \gamma v_{\pi}(s')]\]
   \[   v_{*}(s) = \max_{a} \mathbb{E}[r + \gamma v_{*}(s')]\]
3. **DP 算法**：
   - 策略评估：\(v_{k+1} = T^{\pi} v_k\)
   - 策略迭代：评估 → 改进
   - 值迭代：\(v_{k+1} = T^{*} v_k\)
4. **保证**：
   - 压缩映射：\(\|Tv - Tv'\|_{\infty} \leq \gamma \|v - v'\|_{\infty}\)
   - 收敛至最优：\(\lim_{k \to \infty} v_k = v_{*}\)

---

### 第 26 页

#### 问题描述：冰冻酸奶店

**业务挑战**：管理每日库存以最大化利润。

**参数**：
- **状态**：库存水平 \( s \in \{0,1,2,3,4,5\} \)
- **动作**：订购数量 \( a \in \{0,1,2,3,4,5\} \)
- **最大容量**：5 桶
- **需求**：\( d \sim \text{Uniform}\{0,1,2,3\} \)

**成本与价格**：
- 售价：$20/桶
- 订购成本：$12/桶
- 持有成本：$2/桶
- 缺货惩罚：$5/桶

**目标**：找到最优订购策略 \(\pi^*(s)\)

---

### 第 27 页

#### 数学建模

**转移动态**：

\[s' = \max(0, \min(5, s + a - d))\]

**奖励函数**：

\[r(s, a, d) = 20 \cdot \min(s+a, d) - 12a - 2 \cdot \max(0, s + a - d) - 5 \cdot \max(0, d - s - a)\]

**需求分布**：

\[P(d = 0) = 0.1, \, P(d = 1) = 0.3, \, P(d = 2) = 0.4, \, P(d = 3) = 0.2\]

**MDP 参数**：

\[\gamma = 0.9, \quad \theta = 0.001\]

---

### 第 28 页

#### 动态规划：值迭代

**贝尔曼最优方程**：

\[V_{k+1}(s) = \max_a \sum_{d=0}^3 P(d) \left[ r(s, a, d) + 0.9 V_k(s') \right]\]
其中 \( s' = \max(0, \min(5, s + a - d)) \)

**算法步骤**：
1. 初始化 \( V_0(s) = 0 \) 对所有 \( s\)
2. 重复直至收敛：
   \[   V_{k+1}(s) = \max_a \mathbb{E}_{d\sim P} \left[ r(s, a, d) + 0.9 V_k(s') \right]\]
3. 提取最优策略：
   \[   \pi^*(s) = \arg \max_a \mathbb{E}_{d\sim P} \left[ r(s, a, d) + 0.9 V_k(s') \right]\]

---

### 第 29 页

#### 手动计算：状态 s=2，第一次迭代

**计算 \( V_1(2) \)**：

| 动作 a | 值    | 分解    |
|---|---|---|
| \( a = 0 \)   | 28.0    | \( 0.1 \times (-4) + 0.3 \times 18 + 0.4 \times 40 + 0.2 \times 35 \) |
| \( a = 1 \)   | 14.4    | \( 0.1 \times (-18) + 0.3 \times 4 + 0.4 \times 26 + 0.2 \times 23 \) |
| \( a = 2 \)   | 0.4    | \( 0.1 \times (-32) + 0.3 \times (-10) + 0.4 \times 12 + 0.2 \times 9 \) |
| \( a \geq 3 \)   | < 0    | 负数（超出容量或持有成本高） |

**最佳动作**：\( a = 0 \)，值 28.0  
\[V_1(2) = 28.0, \quad \pi_1(2) = 0\]

---

### 第 30 页

#### 第一次迭代结果

| 状态 s | 最优 a | \( V_1(s) \) |
|---|---|---|
| 0    | 3    | 16.8   |
| 1    | 2    | 24.6   |
| 2    | 0    | 28.0   |
| 3    | 0    | 24.6   |
| 4    | 0    | 16.8   |
| 5    | 0    | 0.0   |

**初步洞察**：
- 库存低时订购（状态 0,1）
- 库存高时不订购（状态 2-5）
- 价值在中等级别库存时达到峰值

---

### 第 31 页

#### 第二次迭代：状态 s=2

**使用上一轮的 \( V_1 \)**：

\[V_2(2) = \max_{a} \mathbb{E}[r + 0.9V_1(s')]\]

\[a = 0 : 0.1(-4 + 0.9 \times 28.0) + 0.3(18 + 0.9 \times 24.6) + 0.4(40 + 0.9 \times 16.8) + 0.2(35 + 0.9 \times 16.8)\]
\[= 2.12 + 12.04 + 22.05 + 10.02 = 46.23\]

\[a = 1 : 0.1(-18 + 0.9 \times 24.6) + 0.3(4 + 0.9 \times 28.0) + 0.4(26 + 0.9 \times 24.6) + 0.2(23 + 0.9 \times 16.8)\]
\[= 0.41 + 8.76 + 19.26 + 7.62 = 36.05\]

**更新**：\( V_2(2) = 46.23 \)，仍为 \( a = 0 \)

---

### 第 32 页

#### 收敛模式

**值迭代进展**

| 迭代 | \( V(0) \) | \( V(2) \) | \( V(5) \) |
|---|---|---|---|
| 0    | 0.00   | 0.00   | 0.00   |
| 1    | 16.80  | 28.00  | 0.00   |
| 2    | 46.23  | 46.23  | 16.80  |
| 5    | 102.35  | 112.48  | 92.35  |
| 10    | 132.18  | 144.21  | 128.15  |
| 20    | 137.89  | 149.92  | 137.89  |
| 45    | 138.42  | 150.28  | 138.42  |

**45 次迭代后收敛（\(\Delta < 0.001\)）**

---

### 第 33 页

#### 最优解

**最终最优值与策略**

| 状态 \( s \) | \( V^* (s) \) | \( \pi^* (s) \) |
|---|---|---|
| 0    | 138.42    | 3    |
| 1    | 146.35    | 2    |
| 2    | 150.28    | 1    |
| 3    | 150.28    | 0    |
| 4    | 146.35    | 0    |
| 5    | 138.42    | 0    |

**最优策略解释**：
\[ \pi^* (s) = \max(0, 3 - s) \]
“订购至 3 桶”策略

---

### 第 34 页

#### 第二节：Q学习

---

### 第 35 页

#### Q学习：迭代更新

在 Q学习中，通过迭代更新优化 Q表  
\( (t = 0, 1, 2, \ldots) \)。

在下方方程中：
- \( Q_t(s_t, a_t) \) 是当前 Q值
- \( Q_{t+1}(s_t, a_t) \) 是更新后的 Q值

---

### 第 36 页

#### Q学习更新方程

**核心更新规则**

\[ Q_{t+1}(s_t, a_t) = Q_t(s_t, a_t) + \alpha \left( r_t + \gamma \max_a Q_t(s_{t+1}, a) - Q_t(s_t, a_t) \right) \]

其中 \(\alpha\) 是学习率。当前状态为 \(s_t\)，动作为 \(a_t\)，奖励为 \(r_t\)，下一状态为 \(s_{t+1}\)。\(Q_t\) 是当前 Q表。

---

### 第 37 页

#### 理解方程：第一步

假设你在状态 \( s_t \) 执行动作 \( a_t \)，结果如下：

- 获得奖励 \( r_t \)
- 状态变为 \( s_{t+1} \)

| 状态 \( s_t \) | 动作 \( a_t \)    | 状态 \( s_{t+1} \) |
|---|---|---|
|    | 奖励：\( r_t \)    |    |

---

### 第 38 页

#### 理解方程：第二步

在 \( s_{t+1} \) 的最优下一动作为：

\[a_{t+1} = \arg\max_a Q(s_{t+1}, a)\]

通过执行此最优动作，获得期望未来折扣奖励：

\[r_t + \gamma \max_a Q(s_{t+1}, a)\]

其中：
- \( r_t \)：即时奖励
- \(\gamma \max_a Q(s_{t+1}, a)\)：最大折扣未来奖励

---

### 第 39 页

#### 理解方程：第三步

1. 计算 **TD 目标**（最优期望值）：
   \[   Target = r_t + \gamma \max_a Q_t(s_{t+1}, a)\]
2. 与 **当前估计** 比较：
   \[   TD Error = \left( r_t + \gamma \max_a Q_t(s_{t+1}, a) \right) - Q_t(s_t, a_t)\]
3. 使用学习率 \(\alpha\) 更新当前值：
   \[   Q_{t+1}(s_t, a_t) = Q_t(s_t, a_t) + \alpha \times TD Error\]

---

### 第 40 页

#### Q学习更新：总结

\[ Q_{t+1}(s_t, a_t) = \underbrace{Q_t(s_t, a_t)}_{当前值} + \alpha \left( \underbrace{r_t + \gamma \max_a Q_t(s_{t+1}, a)}_{目标} - \underbrace{Q_t(s_t, a_t)}_{当前估计} \right) \]

**关键组成部分**：
- \(\alpha\)：学习率（更新幅度）
- \(\gamma\)：折扣因子（未来奖励的重要性）
- \(\max_a Q_t(s_{t+1}, a)\)：最佳估计未来值

---

### 第 41 页

#### 探索与利用的困境

**强化学习中的基本挑战**

- **利用**：选择已知奖励最高的动作
- **探索**：尝试新动作以发现可能更好的奖励

**为何困难？**  
过度利用 → 可能错过更好选项  
过度探索 → 永不收敛至最优策略

---

### 第 42 页

#### Epsilon贪心策略：解决方案

简单有效的策略
- \( \epsilon \)：小概率（如 0.1 或 0.01）
- 以概率 \( 1 - \epsilon \)：选择**贪心**动作（利用）
- 以概率 \( \epsilon \)：选择**随机**动作（探索）

**算法**  
**输入**：当前状态 \( s \)，探索率 \( \epsilon \)，Q表 \( Q \)

1. 生成随机数 \( r \sim U(0, 1) \)
2. 若 \( r < \epsilon \)：
   - 均匀选择随机动作 \( a \)
3. 否则：
   - 选择贪心动 \( a = \arg\max_{a'} Q(s, a') \)

---

### 第 43 页

#### 什么是深度 Q网络？

**核心概念**  
深度 Q网络是一种深度强化学习方法，结合：

- **Q学习**原理
- **神经网络**（深度学习）

**主要动机**  
该思想深受先前讨论的普通 Q学习方法启发。

**为何从 Q表转向神经网络？**

- 更好处理高维状态空间
- 更高效表示复杂环境
- 能在相似状态间泛化

---

### 第 44 页

#### Q学习与表格的局限性

**Q表的问题**

- 在 Q表中离散化状态**计算效率低**
- 对复杂环境**不够智能**
- 需为**每个可能状态动作对**存储单独值
- **无法泛化**从已见状态到未见状态

**维度灾难**

- 状态多时 Q表变得**巨大**
- 内存需求随状态维度**指数增长**
- 对实际问题学习变得**不切实际**

---

### 第 45 页

#### DQN 解决方案：从表到网络

**核心思想**  
DQN 使用以下替代 Q表：

- **Q函数** \( Q(s, a; \theta) \)
- 实现为**神经网络**（Q网络）
- 参数化为权重 \(\theta\)

**网络结构**

- **输入**：状态表示
- **输出**：每个动作的 Q值
- **隐藏层**：学习复杂特征
- 可处理**连续**状态空间

---

### 第 46 页

#### DQN 工作原理：基本方法

**与 Q学习的相似性**

- 类似 Q学习，优化 Q网络以匹配实际经验
- 目标：学习最优 Q值 \( Q^* (s, a) \)
- 使用相同的 TD学习原理

**Q网络的损失函数**

\[L(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]\]

**其中**：
- \(\theta\)：当前网络参数
- \(\theta^-\)：目标网络参数
- \(\mathcal{D}\)：经验回放缓冲区

---

### 第 47 页

#### 挑战：训练稳定性

**为何深度 Q学习困难？**

- 神经网络是**非线性**函数逼近器
- 顺序观测**相关**
- Q值估计可能**不稳定**
- 目标值在训练中**变化**

**关键问题**

1. 如何提高训练稳定性？
2. 哪些技巧能帮助 DQN 有效学习？
3. 实践中如何实现？

---

### 第 48 页

#### DQN 的关键创新

**三项关键技术**  
DQN 引入多项创新以稳定训练：

1. **经验回放**
   - 将经验存储在缓冲区
   - 训练时随机抽样批次
   - 打破连续样本间的相关性

2. **目标网络**
   - 用于目标 Q值的独立网络
   - 定期更新
   - 稳定学习目标

3. **梯度裁剪**
   - 限制梯度更新大小
   - 防止梯度爆炸

---

### 第 49 页

#### 第三节：基于策略的方法

---

### 第 50 页

#### 基于策略的方法：核心概念

**基本思想**  
在基于策略的方法中，我们直接优化策略 \(\pi(a|s)\) 本身，无需先学习价值函数。

- 将策略参数化为 \(\pi_\theta(a|s)\)，参数为 \(\theta\)
- 目标是找到最优参数：
  \[\theta^* = \arg \max_{\theta} J(\theta)\]
  其中 \(J(\theta)\) 是策略 \(\pi_\theta\) 下的期望回报

**为何参数化？**

- **泛化**：能处理未见状态
- **紧凑表示**：避免为每个状态单独存储策略
- **梯度优化**：允许使用梯度上升方法

---

### 第 51 页

#### 数学框架

**目标**  
最大化期望回报：

\[J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]\]

**策略梯度定理**

\[\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta (a_t | s_t) \cdot G_t \right]\]

其中 \( G_t = \sum_{k=t}^T \gamma^{k-t} r_k, \tau \) 是由 \(\pi_\theta\) 生成的轨迹。

---

### 第 52 页

#### 蒙特卡洛估计

**从回合样本**  
对于回合 \(\tau = (s_0, a_0, r_0, \ldots, s_T)\)：

\[g(\tau) = \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\]

**批次平均**  
对于 \(N\) 个回合：

\[g = \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T_i} \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)}) \cdot G_t^{(i)}\]

---

### 第 53 页

#### 步骤1：计算回报

**从时间 \( t \) 开始的折扣回报**

\[G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{T-t} r_T\]

**递归计算**

\[G_T = r_T\]
\[G_t = r_t + \gamma G_{t+1} \quad \text{对 } t = T-1, \ldots, 0\]

**示例**  
对于 \(\gamma = 0.9\)，\(T = 3\)：

\[G_1 = r_1 + 0.9 r_2 + 0.9^2 r_3\]

---

### 第 54 页

#### 步骤2：对数概率梯度

**离散动作（Softmax）**

\[\pi_\theta(a|s) = \frac{e^{\theta^\top} \phi(s,a)}{\sum_{a'} e^{\theta^\top} \phi(s,a')}\]
\[\nabla_\theta \log \pi_\theta(a|s) = \phi(s,a) - \sum_{a'} \pi_\theta(a'|s) \phi(s,a')\]

**连续动作（高斯分布）**

\[\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma^2)\]
\[\nabla_\theta \log \pi_\theta(a|s) = \frac{a - \mu_\theta(s)}{\sigma^2} \nabla_\theta \mu_\theta(s)\]

---

### 第 55 页

#### 步骤3：梯度累积

**算法**

\[ g \leftarrow 0 \]
对 \( t = 0 \) 到 \( T \)：
\[ g \leftarrow g + \nabla_\theta \log \pi_\theta (a_t | s_t) \cdot G_t \]

**参数更新**

\[\theta \leftarrow \theta + \alpha \cdot g\]

其中 \(\alpha\) 是学习率

---

### 第 56 页

#### 完整 REINFORCE 算法

1. 初始化策略参数 \(\theta\)
2. 对迭代 \(k = 1, 2, \ldots\)：
   - 使用 \(\pi_\theta\) 收集 \(N\) 个回合
   - 计算梯度估计 \(g\)
   - 更新 \(\theta \leftarrow \theta + \alpha g\)
3. 直至收敛

---

### 第 57 页

#### 策略梯度定理：证明

**状态分布形式**

\[J(\theta) = \mathbb{E}_{s_0 \sim p(s_0)} [V^{\pi_\theta}(s_0)] = \sum_s p(s_0 = s) V^{\pi_\theta}(s)\]

**状态价值函数**

\[V^{\pi_\theta}(s) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \bigg| s_0 = s \right]\]

---

### 第 58 页

#### 步骤1：贝尔曼方程

**价值函数**

\[V^{\pi_\theta}(s) = \sum_a \pi_\theta(a|s) Q^{\pi_\theta}(s,a)\]

**动作价值函数**

\[Q^{\pi_\theta}(s,a) = r(s,a) + \gamma \sum_{s'} p(s'|s,a) V^{\pi_\theta}(s')\]

---

### 第 59 页

#### 步骤2：价值函数的梯度

对 \(\theta\) 求梯度：

\[\nabla_\theta V^{\pi_\theta}(s) = \nabla_\theta \left[ \sum_a \pi_\theta(a|s) Q^{\pi_\theta}(s, a) \right]\]

应用乘积法则：

\[\nabla_\theta V^{\pi_\theta}(s) = \sum_a \underbrace{\nabla_\theta \pi_\theta(a|s) Q^{\pi_\theta}(s, a)}_{策略梯度} + \underbrace{\pi_\theta(a|s) \nabla_\theta Q^{\pi_\theta}(s, a)}_{价值梯度}\]

---

### 第 60 页

#### 步骤3：展开 \(\nabla_\theta Q^{\pi_\theta}(s, a)\)

由贝尔曼方程：

\[Q^{\pi_\theta}(s, a) = r(s, a) + \gamma \sum_{s'} p(s'|s, a) V^{\pi_\theta}(s')\]

由于 \(r(s, a)\) 和 \(p(s'|s, a)\) 与 \(\theta\) 无关：

\[\nabla_\theta Q^{\pi_\theta}(s, a) = \gamma \sum_{s'} p(s'|s, a) \nabla_\theta V^{\pi_\theta}(s')\]

---

### 第 61 页

#### 步骤4：递归形式

代回：

\[\nabla_\theta V^{\pi_\theta}(s) = \sum_a \nabla_\theta \pi_\theta(a|s) Q^{\pi_\theta}(s,a) + \gamma \sum_a \pi_\theta(a|s) \sum_{s'} p(s'|s,a) \nabla_\theta V^{\pi_\theta}(s')\]

定义转移概率：

\[P(s \to s',1,\pi_\theta) = \sum_a \pi_\theta(a|s) p(s'|s,a)\]
\[\nabla_\theta V^{\pi_\theta}(s) = \sum_a \nabla_\theta \pi_\theta(a|s) Q^{\pi_\theta}(s,a) + \gamma \sum_{s'} P(s \to s',1,\pi_\theta) \nabla_\theta V^{\pi_\theta}(s')\]

---

### 第 62 页

#### 步骤5：展开递归

这是递归方程，无限展开：

\[\nabla_\theta V^{\pi_\theta}(s) = \sum_{x \in S} \sum_{k=0}^\infty \gamma^k P(s \to x, k, \pi_\theta) \sum_a \nabla_\theta \pi_\theta (a|x) Q^{\pi_\theta}(x, a)\]

其中 \(P(s \to x, k, \pi_\theta)\) 是在 \(\pi_\theta\) 下从 \(s\) 出发 \(k\) 步到达状态 \(x\) 的概率。

---

### 第 63 页

#### 步骤6：折扣状态分布

定义折扣状态访问分布：

\[d^{\pi_\theta}(s) = \sum_{s_0} p(s_0) \sum_{k=0}^\infty \gamma^k P(s_0 \to s, k, \pi_\theta)\]

图示：
- \( \gamma \)  
- \( s_1 \)  
- \( \gamma^2 \)  
- \( \gamma \)  
- \( s_2 \)

---

### 第 64 页

#### 步骤7：目标函数的梯度

回忆 \( J(\theta) = \sum_{s} p(s_0 = s) V^{\pi_\theta}(s) \)

\[\nabla_\theta J(\theta) = \sum_{s} p(s_0 = s) \nabla_\theta V^{\pi_\theta}(s)\]

代入步骤5：

\[\nabla_\theta J(\theta) = \sum_{s} p(s_0 = s) \sum_{x} \sum_{k=0}^\infty \gamma^k P(s \to x, k, \pi_\theta) \sum_a \nabla_\theta \pi_\theta (a|x) Q^{\pi_\theta}(x, a|\theta)\]

---

### 第 65 页

#### 步骤8：重排求和顺序

改变求和顺序：

\[\nabla_\theta J(\theta) = \sum_x \left[ \sum_s p(s_0 = s) \sum_{k=0}^\infty \gamma^k P(s \to x, k, \pi_\theta) \right] \times \sum_a \nabla_\theta \pi_\theta (a|x) Q^{\pi_\theta} (x, a)\]

括号内项为 \( d^{\pi_\theta} (x) \)：

\[\nabla_\theta J(\theta) = \sum_x d^{\pi_\theta} (x) \sum_a \nabla_\theta \pi_\theta (a|x) Q^{\pi_\theta} (x, a)\]

---

### 第 66 页

#### 步骤9：对数导数技巧

使用恒等式：

\[\nabla_{\theta} \pi_{\theta}(a|s) = \pi_{\theta}(a|s) \nabla_{\theta} \log \pi_{\theta}(a|s)\]

代入：

\[\nabla_{\theta} J(\theta) = \sum_{s} d^{\pi_{\theta}}(s) \sum_{a} \pi_{\theta}(a|s) \nabla_{\theta} \log \pi_{\theta}(a|s) Q^{\pi_{\theta}}(s, a)\]

改写为期望：

\[\nabla_{\theta} J(\theta) = \mathbb{E}_{s \sim d^{\pi_{\theta}}, a \sim \pi_{\theta}} [\nabla_{\theta} \log \pi_{\theta}(a|s) Q^{\pi_{\theta}}(s, a)]\]

---

### 第 67 页

#### 步骤10：基于轨迹的形式

使用完整轨迹的替代表达：

\[\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^\infty \gamma^t Q^{\pi_\theta}(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t|s_t) \right]\]

或使用回报 \( G_t = \sum_{k=t}^\infty \gamma^{k-t} r(s_k, a_k) \)：

\[\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta(a_t|s_t) G_t \right]\]

---

### 第 68 页

#### 策略梯度定理

**定理1（状态动作形式）**

\[\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta (a|s) Q^{\pi_\theta}(s, a) \right]\]

**定理2（轨迹形式）**

\[\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta (a_t | s_t) G_t \right]\]

---

### 第 69 页

#### 关键洞察

**无需模型**
- 仅需采样
- 无需环境动态
- 无模型方法

**信用分配**
- 每个动作按其贡献调整
- 以 Q值加权
- 时序信用分配

**对数导数技巧**
- 转换概率比
- 允许蒙特卡洛估计
- 简单梯度计算

**兼容近似**
- 可使用 \( Q_w(s, a) \) 近似
- 满足特定条件
- 不引入偏差

---

### 第 70 页

#### 为何重要

**理论基础**
- 为 REINFORCE 算法提供依据
- 为演员-评论家方法奠基
- 连接策略与价值方法

**实际意义**
- 支持直接策略优化
- 处理连续动作空间
- 是现代强化学习算法的基础

---

### 第 71 页

#### 从定理到算法

1. 采样轨迹：\( \tau \sim \pi_\theta \)
2. 计算回报：  
   \[   G_t = \sum_{k=t}^T \gamma^{k-t} r_k\]
3. 估计梯度：  
   \[   \hat{g} = \sum_{t} \nabla_\theta \log \pi_\theta (a_t | s_t) G_t\]
4. 更新参数：  
   \[   \theta \leftarrow \theta + \alpha \hat{g}\]

这就是 REINFORCE 算法！

---

### 第 72 页

#### 扩展：优势函数

用优势 \( A^{\pi_\theta}(s, a) \) 替代 \( Q^{\pi_\theta}(s, a) \)：

\[A^{\pi_\theta}(s, a) = Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s)\]
\[\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) A^{\pi_\theta}(s, a)]\]

更低方差，相同期望！

---

### 第 73 页

#### 总结

- 从第一性原理推导策略梯度定理
- 展示策略梯度与价值函数间的联系
- 建立策略优化的理论基础
- 展示实用算法推导
- 为现代强化学习方法提供基础

**策略梯度：从理论到实践**

---

### 第 74 页

#### 优势函数与演员-评论家方法

用学习到的估计替换 \( Q^\pi(s, a) \)：

\[\nabla_\theta J(\theta) \approx \mathbb{E} \left[ \nabla_\theta \log \pi(a|s; \theta) \cdot \hat{Q}(s, a; \phi) \right]\]
其中 \(\hat{Q}(s, a; \phi)\) 是评论家的估计。

**定义**  
优势函数衡量动作相对于平均有多好：

\[A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)\]

**带优势的演员-评论家**  
更新规则变为：

\[\nabla_\theta J(\theta) \approx \mathbb{E} \left[ \nabla_\theta \log \pi(a|s; \theta) \cdot A^\pi(s, a) \right]\]

---

### 第 75 页

#### 评论家类型

**价值型评论家**
- 学习 \( V(s) \)：状态价值函数
- 优势：\( A(s, a) = Q(s, a) - V(s) \)
- 对离散动作：\( Q(s, a) = r + \gamma V(s') \)
- 简单有效

**Q值型评论家**
- 学习 \( Q(s, a) \)：动作价值函数
- 直接优势：\( A(s, a) = Q(s, a) - V(s) \)
- 其中 \( V(s) = \mathbb{E}_{a \sim \pi} [Q(s, a)] \)
- 更具表达力但更复杂

---

### 第 76 页

#### A2C：优势演员-评论家

**关键特性**

- **同步**版本（无异步性）
- 使用 n步回报进行优势估计
- 多个并行环境
- 比单步演员-评论家更稳定

**优势估计**

\[A(s_t, a_t) = \sum_{i=0}^{k-1} \gamma^i r_{t+i} + \gamma^k V(s_{t+k}) - V(s_t)\]
其中 \( k \) 是步数。

**更新规则**

\[\theta \leftarrow \theta + \alpha_\theta \nabla_\theta \log \pi(a|s; \theta) A(s, a)\]

---

### 第 77 页

#### PPO：近端策略优化

**动机**  
防止策略更新过大导致性能崩溃。

**裁剪目标**

\[J^{CLIP}(\theta) = \mathbb{E} \left[ \min \left( r_t(\theta) \hat{A}_t, clip(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]\]
其中 \( r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_\theta_{old}(a_t|s_t)} \) 是概率比。

- **裁剪**：防止大更新
- **多轮次**：重用经验
- **GAE**：广义优势估计
- 在许多任务中达到先进水平

---

### 第 78 页

#### SAC：软演员-评论家

**关键思想**  
最大熵强化学习：鼓励探索。

\[\pi^* = \arg \max_{\pi} \mathbb{E}_{\pi} \left[ \sum_{t} r(s_t, a_t) + \alpha H(\pi(\cdot|s_t)) \right]\]
其中 \( H \) 是熵，\(\alpha\) 是温度。

**组成部分**

- **随机演员**：\(\pi(a|s)\)
- **两个 Q网络**：\(Q_{\theta_1}\), \(Q_{\theta_2}\)
- **价值网络**：\(V_{\psi}\)
- **目标网络**：稳定学习

**优点**

- 连续控制任务先进
- 自动温度调整
- 样本效率高
- 对超参数鲁棒

---