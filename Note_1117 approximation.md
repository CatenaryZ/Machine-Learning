好的，这是您提供的 PDF 讲座文稿 `lecture_6.2_approximation.pdf` 的逐页翻译。

---

===== Page 1 =====

# 第六讲第二部分：推断第二部分

谢丹
清华大学数学系

2025年11月12日

===== Page 2 =====

第二部分：EM算法

===== Page 3 =====

# 问题设置

我们常常需要引入潜变量来处理复杂的概率问题。观测变量的边缘概率评估至关重要。

- 观测数据：\( X = \{x_1, x_2, ..., x_N\} \)
- 潜变量：\( Z = \{z_1, z_2, ..., z_N\} \)
- 模型参数：\(\theta\)

我们想要最大化边缘似然（证据）：

\[\log p_\theta(X) = \log \int p_\theta(X, Z) dZ\]

---

## 挑战
对于复杂模型，该积分通常是难以处理的！

===== Page 4 =====

# 变分推断方法

- **引入变分分布 \( q_\phi(\mathbf{Z}) \)**
- **近似真实后验 \( p_\theta(\mathbf{Z}|\mathbf{X}) \)**
- **找到使 \( q_\phi(\mathbf{Z}) \) 接近真实后验的 \(\phi\)**

===== Page 5 =====

# 推导ELBO - 步骤 1

从证据开始：

\[\log p_\theta(\mathbf{X}) = \log \int p_\theta(\mathbf{X}, \mathbf{Z}) d\mathbf{Z}\]

引入变分分布：

\[\log p_\theta(\mathbf{X}) = \log \int q_\phi(\mathbf{Z}) \frac{p_\theta(\mathbf{X}, \mathbf{Z})}{q_\phi(\mathbf{Z})} d\mathbf{Z}\]

===== Page 6 =====

# 推导ELBO - 步骤 2

应用 Jensen 不等式（因为 \(\log\) 是凹函数）：

\[\log \int q_\phi(\mathbf{Z}) \frac{p_\theta(\mathbf{X}, \mathbf{Z})}{q_\phi(\mathbf{Z})} d\mathbf{Z} \geq \int q_\phi(\mathbf{Z}) \log \frac{p_\theta(\mathbf{X}, \mathbf{Z})}{q_\phi(\mathbf{Z})} d\mathbf{Z}\]

这给出了证据下界（ELBO）：

\[\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi(\mathbf{Z})} \left[ \log \frac{p_\theta(\mathbf{X}, \mathbf{Z})}{q_\phi(\mathbf{Z})} \right]\]

该式对任何分布 \(q_\phi(\mathbf{Z})\) 都成立，该分布通常由神经网络定义。

===== Page 7 =====

ELBO的替代形式

形式 1

\[ \mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi(\mathbf{Z})} [\log p_\theta(\mathbf{X}, \mathbf{Z})] - \mathbb{E}_{q_\phi(\mathbf{Z})} [\log q_\phi(\mathbf{Z})] \]

形式 2

\[ \mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi(\mathbf{Z})} [\log p_\theta(\mathbf{X}|\mathbf{Z})] - D_{KL}(q_\phi(\mathbf{Z})||p_\theta(\mathbf{Z})) \]

其中 \( D_{KL} \) 是 Kullback-Leibler 散度。

===== Page 8 =====

# 解释

- \( \mathbb{E}_{q_\phi} (\mathbf{Z}) [\log p_\theta (\mathbf{X} | \mathbf{Z})] \)：**重构项**
  - 衡量从潜变量重构数据的能力
- \( D_{KL}(q_\phi (\mathbf{Z}) || p_\theta (\mathbf{Z})) \)：**正则化项**
  - 保持近似后验接近先验
  - 防止过拟合

===== Page 9 =====

# 优化

我们最大化 ELBO：

\[\theta^*, \phi^* = \arg \max_{\theta, \phi} \mathcal{L}(\theta, \phi)\]

这为我们提供了：
- 好的模型参数 \(\theta\)
- 好的变分近似 \(q_\phi(\mathbf{Z})\)

因为

\[\log p_\theta(\mathbf{X}) = \mathcal{L}(\theta, \phi) + D_{KL}(q_\phi(\mathbf{Z})||p_\theta(\mathbf{Z}|\mathbf{X}))\]

最大化 ELBO 即最小化 \(D_{KL}(q_\phi(\mathbf{Z})||p_\theta(\mathbf{Z}|\mathbf{X}))\)

===== Page 10 =====

# EM策略：一个下界

让我们证明
\[\log p_\theta(\mathbf{X}) = \mathcal{L}(\theta, \phi) + D_{KL}(q_\phi(\mathbf{Z})||p_\theta(\mathbf{Z}|\mathbf{X}))\]

证明：
\[\mathcal{L}(\theta, \phi) + D_{KL}(q_\phi(\mathbf{Z})||p_\theta(\mathbf{Z}|\mathbf{X})) = \int q_\phi(\mathbf{Z})\left[\log\frac{p_\theta(\mathbf{X}, \mathbf{Z})}{q_\phi(\mathbf{Z})}\right] d\mathbf{Z} + \int q_\phi(\mathbf{Z})\log\frac{q_\phi(\mathbf{Z})}{p_\theta(\mathbf{Z}|\mathbf{X})} d\mathbf{Z}\]

使用等式 \( p_\theta(\mathbf{X}, \mathbf{Z}) = p_\theta(\mathbf{Z}|\mathbf{X})p_\theta(\mathbf{X}) \)，我们得到这个重要的恒等式。

备注

1. 注意该和式与分布 \( q_\phi(\mathbf{Z}) \) 无关。
2. \( D_{KL}(q_\phi(\mathbf{Z})||p_\theta(\mathbf{Z}|\mathbf{X})) \geq 0 \)，且 ELBO 给出了下界。

===== Page 11 =====

EM的两个步骤 I

EM 是一种迭代算法，在以下两步之间交替进行：

E步：固定 \(\theta\)，关于 \(q\) 最大化 \(\mathcal{L}\)
（利用 KL 散度的性质）固定 \(\theta^{old}\)。最优的 \(q\) 是后验分布：

\[q^{opt}(\mathbf{Z}) = p(\mathbf{Z}|\mathbf{X}, \theta^{old})\]

我们计算 **Q函数**：

\[Q(\theta, \theta^{old}) = \mathbb{E}_{p(\mathbf{Z}|\mathbf{X}, \theta^{old})}[\log p(\mathbf{X}, \mathbf{Z}|\theta)] + H(q^{opt})\]

这“填补”了对数似然中缺失的数据 \(\mathbf{Z}\)。

M步：固定 \(q\)，关于 \(\theta\) 最大化 \(\mathcal{L}\)

===== Page 12 =====

EM的两个步骤 II

固定 \( q \)。找到最大化 Q 函数的新参数：

\[\theta^{new} = \arg \max_{\theta} Q(\theta, \theta^{old})\]

这通常是一个更容易的优化问题。

===== Page 13 =====

EM算法的可视化

对数似然
ln p(x | θ(n+2))
ln p(x | θ(n+1))
ln p(x | θ(n))
θ(n) --- θ(n+1) --- θ(n+2)
Q(θ | θ(n+1))
Q(θ | θ(n))
ln p(x)

===== Page 14 =====

为什么它有效？保证

定理（对数似然的单调递增）
EM 算法从不降低对数似然：

\[\log p(\mathbf{X}|\theta^{new}) \geq \log p(\mathbf{X}|\theta^{old})\]

证明。

\[\log p(\mathbf{X}|\theta) = L(q, \theta) + D_{KL}(q(\mathbf{Z}) \parallel p(\mathbf{Z}|\mathbf{X}, \theta)) \geq L(q, \theta) \quad (\text{因为 KL 散度 } \geq 0)\]

在 E 步中，我们设 \( q = p(\mathbf{Z}|\mathbf{X}, \theta^{old}) \)，使得 \( KL = 0 \)，所以

\[\log p(\mathbf{X}|\theta^{old}) = L(q, \theta^{old}).\]
在 M 步中，

\[L(q, \theta^{new}) \geq L(q, \theta^{old}).\]
因为 \( KL \geq 0 \)，

\[\log p(\mathbf{X}|\theta^{new}) \geq L(q, \theta^{new}).\]
因此，

\[\log p(\mathbf{X}|\theta^{new}) \geq \log p(\mathbf{X}|\theta^{old}).\]

===== Page 15 =====

# 总结与应用

- **目的**：为含有潜变量的模型寻找 MLE/MAP 估计。
- **核心思想**：迭代地最大化对数似然的下界（ELBO）。
- **步骤**：
  1. **E步**：估算潜变量（计算期望）。
  2. **M步**：使用“完整”数据更新参数。
- **保证**：单调增加对数似然。

## 常见应用

- 高斯混合模型
- 隐马尔可夫模型
- 主题模型
- 软分配聚类
- 缺失数据插补

===== Page 16 =====

通用伪代码

输入/输出

► 输入：观测数据 \( X \)，潜变量 \( Z \)，参数 \( \theta \)

► 输出：收敛后的参数 \( \theta_{final} \)

1. 初始化 \( \theta_{old} \)，阈值 \( \epsilon \)，最大迭代次数 max_iters
2. for 迭代次数 = 1 到 max_iters do
3. E步：计算 \( Q(\theta|\theta_{old}) = E_{P(Z|X,\theta_{old})}[\log P(X,Z|\theta)] \)
4. M步：\( \theta_{new} = \arg\max_{\theta} Q(\theta|\theta_{old}) \)
5. if \( |\log P(X|\theta_{new}) - \log P(X|\theta_{old})| < \epsilon \) then
6. break
7. else
8. \( \theta_{old} \leftarrow \theta_{new} \)
9. end if
10. end for
11. return \( \theta_{final} = \theta_{new} \)

===== Page 17 =====

# 高斯混合模型 示例问题设置

- **观测**：数据点 \( X \)
- **潜变量**：分量分配 \( Z \)
- **参数**：
  \[  \theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^{K}\]

## E步：责任值
对于每个点 \( i \) 和分量 \( k \)：

\[\gamma_{ik} = \frac{\pi_k \cdot \mathcal{N}(X_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \cdot \mathcal{N}(X_i | \mu_j, \Sigma_j)}\]

## M步：参数更新

- \( N_k = \sum_{i=1}^{N} \gamma_{ik} \)
- \( \pi_k^{new} = \frac{N_k}{N} \)
- \( \mu_k^{new} = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} X_i \)

===== Page 18 =====

\( E \) 步是 \( Q(\theta | \theta_{old}) = E_{P(Z|X, \theta_{old})} [\log P(X, Z|\theta)] \)，其中

===== Page 19 =====

第三部分：变分方法

===== Page 20 =====

什么是变分方法？

- 用于近似复杂概率分布的数学框架
- 将难处理的问题转化为可处理的优化问题
- 广泛应用于贝叶斯推断和深度学习
- 核心思想：找到一个更简单的分布来近似真实后验

===== Page 21 =====

关键概念

关键方程

\[\log p_\theta(\mathbf{X}) = \mathcal{L}(\theta, q_\phi) + D_{KL}(q_\phi(\mathbf{Z})||p_\theta(\mathbf{Z}|\mathbf{X}))\]

证据下界

\[\mathcal{L}(q) = \mathbb{E}_{q(\mathbf{z})}[\log p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_{q(\mathbf{z})}[\log q(\mathbf{z})]\]

Kullback-Leibler 散度

\[KL(q||p) = \int q(\mathbf{z})\log \frac{q(\mathbf{z})}{p(\mathbf{z}|\mathbf{x})} d\mathbf{z}\]

===== Page 22 =====

# 变分推断框架

- **目标**：用更简单的分布 \( q(z) \) 近似后验 \( p(z|x) \)
- **目标函数**：最小化 \( KL(q(z)\|p(z|x))\)
- **方法**：最大化 ELBO（等价于最小化 KL 散度）
- **族**：选择变分族 \( q(z; \lambda) \) 及其参数 \(\lambda\)，并在 \(\lambda\) 空间上最大化 ELBO。

在 EM 算法中，我们找到使证据 \( p(X|\theta) \) 最大化的参数。变分方法也为证据和归一化常数 \( Z \)（其中 \( X \) 不出现）提供了近似。

===== Page 23 =====

示例：伊辛模型的平均场

核心思想
用来自可处理族的更简单分布 \( q(s) \) 来近似复杂概率分布 \( p(s) \)

精确分布
玻尔兹曼分布：

\[p(s) = \frac{1}{Z} e^{\beta J \sum_{(ij)} s_i s_j + \beta h \sum_i s_i}\]

变分分布
平均场假设：

\[q(s) = \prod_{i=1}^{N} q_i(s_i)\]

关键见解
平均场理论 = 使用因子化 \( q \) 的变分推断的特例

===== Page 24 =====

# 数学基础

变分分布参数化
对于二值自旋 \( s_i = \pm 1 \)：

\[q_i(s_i) = \frac{1 + m_i s_i}{2}, \quad \mathbb{E}_q[s_i] = m_i\]

---

### 证据下界

最大化：

\[\mathcal{L}[q] = \mathbb{E}_q[\log p(s)] - \mathbb{E}_q[\log q(s)]\]

- **能量项**：\(\mathbb{E}_q[\log p(s)]\)
- **熵项**：\(-\mathbb{E}_q[\log q(s)]\)

===== Page 25 =====

# ELBO 推导 I

## 能量项

\[\mathbb{E}_q[\log p(\mathbf{s})] = \beta J \sum_{\langle ij \rangle} \mathbb{E}_q[s_i s_j] + \beta h \sum_i \mathbb{E}_q[s_i] - \log Z\]

\[= \beta J \sum_{\langle ij \rangle} m_i m_j + \beta h \sum_i m_i - \log Z\]

## 熵项

\[\mathbb{E}_q[\log q(\mathbf{s})] = \sum_i \mathbb{E}_{q_i} [\log q_i (s_i)]\]

\[= \sum_i \left[ \frac{1 + m_i}{2} \log \left( \frac{1 + m_i}{2} \right) + \frac{1 - m_i}{2} \log \left( \frac{1 - m_i}{2} \right) \right]\]

===== Page 26 =====

ELBO 推导 II

完整 ELBO

\[L[{m_i}] = \beta J \sum_{\langle ij \rangle} m_i m_j + \beta h \sum_i m_i - \sum_i S(m_i) - \log Z\]

其中 \( S(m_i) \) 是二值熵。

===== Page 27 =====

# 优化与自洽性

## 坐标上升
对 ELBO 求导：

\[\frac{\partial \mathcal{L}}{\partial m_i} = 2\beta J \sum_{j \in n.n.(i)} m_j + \beta h - \frac{1}{2} \log \left( \frac{1 + m_i}{1 - m_i} \right)\]

## 自洽方程
设 \(\partial \mathcal{L}/\partial m_i = 0\)：

\[\frac{1}{2} \log \left( \frac{1 + m_i}{1 - m_i} \right) = 2\beta J \sum_{j \in n.n.(i)} m_j + \beta h\]

使用 $arctanh(x) = \frac{1}{2} \log \left( \frac{1+x}{1-x} \right)$：

\[m_i = \tanh \left( 2\beta J \sum_{j \in n.n.(i)} m_j + \beta h \right)\]

===== Page 28 =====

# 算法实现

## 坐标上升变分推断

1. 初始化 \( m_i \sim \text{Uniform}(-0.1, 0.1) \)
2. **for** 迭代次数 = 1 到 max_iters **do**
3. **for** 每个格点 \( i \) **do**
4. \(\text{neighbor\_sum} \leftarrow \sum_{j \in \text{neighbors}(i)} m_j\)
5. \(m_{i}^{\text{new}} \leftarrow \tanh(2\beta J \cdot \text{neighbor\_sum} + \beta h_i)\)
6. **end for**
7. **if** \(\max_i |m_{i}^{\text{new}} - m_i| < \text{容差 then break}\)
8. **end if**
9. \(m \gets m_{i}^{\text{new}}\)

10. **end for**

---

## 直接 ELBO 优化

或者，使用梯度方法直接最大化 ELBO：

\[\nabla_{m_i} \mathcal{L} = 2\beta J \sum_{j \in n.n.(i)} m_j + \beta h - \text{arctanh}(m_i)\]

===== Page 29 =====

第四部分：采样方法

===== Page 30 =====

# 为什么需要采样？

## 主要动机

- 近似复杂积分：

\[\mathbb{E}[f(x)] = \int f(x)p(x)dx \approx \frac{1}{N} \sum_{i=1}^{N} f(x_i)\]

- 在复杂概率模型中执行推断
- 为模拟生成合成数据

## 示例

蒙特卡洛积分：与其解析地求解困难积分，我们使用来自分布的样本对其进行经验近似。

## 基本挑战

如何从复杂的高维概率分布中高效地生成样本？

===== Page 31 =====

采样方法分类

基本方法
逆变换采样    拒绝采样
重要性采样
MCMC 方法
Metropolis-Hastings Gibbs 采样    哈密顿蒙特卡洛
高级方法
序贯蒙特卡洛

===== Page 32 =====

# 逆变换采样

核心思想
使用均匀随机变量和逆 CDF 生成样本。

## 算法
1. 生成 \( u \sim \text{Uniform}(0, 1) \)
2. 计算 \( x = F^{-1}(u) \)
3. 返回 \( x \) 作为样本

## 示例
指数分布
\[p(x) = \lambda e^{-\lambda x}\]
CDF：
\[F(x) = 1 - e^{-\lambda x}\]
逆 CDF：
\[F^{-1}(u) = -\frac{\ln(1-u)}{\lambda}\]

## 定理（概率积分变换）
如果 \( U \sim \text{Uniform}(0, 1) \)，则
\[X = F^{-1}(U)\] 具有分布 \( F \)。

## 优缺点
- **优点**：精确采样，实现简单
- **缺点**：需要解析的逆 CDF，在高维情况下效率低

===== Page 33 =====

# 拒绝采样 I

## 概念
从提议分布 \( q(x) \) 采样，并根据目标分布 \( p(x) \) 接受/拒绝。

### 算法
1. 找到 \( M \) 使得
   \[   p(x) \leq Mq(x)\]
2. 采样 \( x \sim q(x) \)
3. 采样 \( u \sim \text{Uniform}(0, 1) \)
4. 如果 \( u < \frac{p(x)}{Mq(x)} \) 则接受

### 接受率与局限性

---

## 密度

- **拒绝区域**
  - \( Mq(x) \)
- **接受区域**

===== Page 34 =====

# 拒绝采样 II

接受率 = \(\frac{1}{M}\)

- 效率取决于 \(q(x)\) 与 \(p(x)\) 的匹配程度
- 维度灾难：接受率指数下降
- 在高维中难以找到好的 \(M\)

===== Page 35 =====

# 重要性采样 I

核心思想
对来自提议分布的样本进行加权，而不是生成精确样本。

## 算法
1. 采样 \( x_i \sim q(x) \)，其中 \( i = 1, \ldots, N \)

2. 计算权重：
   \[    w_i = \frac{p(x_i)}{q(x_i)}\]

3. 归一化：
   \[    \tilde{w}_i = \frac{w_i}{\sum_j w_j}\]

## 期望估计

\[\mathbb{E}_{p(x)}[f(x)] \approx \sum_{i=1}^N \tilde{w}_i f(x_i)\]

---

### 有效样本量

\[ESS = \frac{1}{\sum_{i=1}^N \tilde{w}_i^2}\]

衡量我们拥有多少“有用”样本。

## 权重退化
在高维中，少数样本主导权重，使得估计不可靠。

===== Page 36 =====

# 重要性采样 II

## 优点与局限性

- **优点**：总是有效，提供无偏估计
- **缺点**：权重退化，对提议分布选择敏感

===== Page 37 =====

# 马尔可夫链蒙特卡洛 基础

核心思想
构建一个马尔可夫链，其平稳分布是目标分布 \( p(x) \)。

## 关键性质

- **细致平衡**：
  \[  p(x)T(x \to x') = p(x')T(x' \to x)\]

- **遍历性**：链收敛到平稳分布

- **预热期**：丢弃初始样本

## 收敛保证

在温和条件下，无论初始状态如何，链都会收敛到目标分布。

---

### 示例

**马尔可夫性质**
\[p(x_{t+1}|x_t, x_{t-1}, \ldots, x_1) = p(x_{t+1}|x_t)\]

===== Page 38 =====

Metropolis-Hastings 算法

MCMC 的主力
最通用和广泛使用的 MCMC 方法。

算法
对于 \( t = 0, 1, 2, \ldots \)：

1. 采样 \( x^* \sim q(x^*|x_t) \)
2. 计算接受概率：

\[\alpha = \min \left( 1, \frac{p(x^*)q(x_t|x^*)}{p(x_t)q(x^*|x_t)} \right)\]

3. 采样 \( u \sim \text{Uniform}(0, 1) \)
4. 如果 \( u < \alpha \)，则接受：
   \[   x_{t+1} = x^*\]
   否则拒绝：
   \[   x_{t+1} = x_t\]

提议分布变体

- **随机游走 MH**：
   \[   q(x^*|x) = \mathcal{N}(x, \sigma^2)\]
- **独立 MH**：
   \[   q(x^*|x) = q(x^*)\]

示例
对称提议分布 当
\[q(x^*|x) = q(x|x^*) \]（对称）时，
接受概率简化为：

\[\alpha = \min \left( 1, \frac{p(x^*)}{p(x_t)} \right)\]

===== Page 39 =====

# Gibbs 采样 I

## Metropolis-Hastings 的特例
一次从一个变量的全条件分布中采样。

### 算法
对于目标 \( p(x_1, x_2, \ldots, x_D) \)：

1. 初始化 \( x_1^{(0)}, \ldots, x_D^{(0)} \)

2. 对于 \( t = 1, 2, \ldots \)：

- 采样 \( x_1^{(t)} \sim p(x_1|x_2^{(t-1)}, \ldots, x_D^{(t-1)}) \)
- 采样 \( x_2^{(t)} \sim p(x_2|x_1^{(t)}, x_3^{(t-1)}, \ldots) \)
- 采样 \( x_D^{(t)} \sim p(x_D|x_1^{(t)}, \ldots, x_{D-1}^{(t)}) \)

### 示例
条件依赖性 每个变量在其马尔可夫毯给定的条件下被采样。

---

### 优点与局限性

===== Page 40 =====

# Gibbs 采样 II

- **优点**：无需调整参数，接受率 = 1
- **缺点**：需要从条件分布采样，可能混合缓慢

===== Page 41 =====

# MCMC 的理论基础

## 目标
从复杂的目标分布 \(\pi(x)\) 中采样。

- \(\pi(x)\) 通常是高维的且仅知其归一化常数：
  \[  \pi(x) \propto P(x).\]

- 直接采样（例如逆变换）是不可能的。

- 解决方案：构建一个马尔可夫链，其**平稳分布**为 \(\pi(x)\)。

===== Page 42 =====

# 马尔可夫链与转移核

## 马尔可夫性质
未来状态仅依赖于当前状态。

\[P(X_{t+1} = x'|X_t = x, X_{t-1}, \ldots) = P(X_{t+1} = x'|X_t = x)\]

## 转移核
描述从 \(x\) 转移到 \(x'\) 的概率。

- [ ] 离散：\(T(x \to x')\) 或 \(P(x'|x)\)
- [x] 连续：\(T(x, x')\)

===== Page 43 =====

# 平稳分布

## 定义（平稳分布）

一个分布 \(\pi(x)\) 对于具有转移核 \(T\) 的马尔可夫链是**平稳的**，如果：

\[\pi(x') = \sum_{x} \pi(x) T(x \to x')\]

（连续情况将求和替换为积分）。

## 解释

一旦链达到分布 \(\pi\)，它就保持在该分布。流入每个状态的概率质量等于流出的质量。

这是**全局平衡条件**。

===== Page 44 =====

# 从全局平衡到细致平衡

**全局平衡**通常难以直接检查和强制执行。

## 定义（细致平衡条件）

一个马尔可夫链关于 \(\pi\) 满足**细致平衡**，如果对于所有 \(x, x'\)：

\[\pi(x) \cdot T(x \to x') = \pi(x') \cdot T(x' \to x)\]

===== Page 45 =====

为什么细致平衡至关重要

定理
如果一个马尔可夫链对于一个分布 \(\pi\) 满足细致平衡条件，那么 \(\pi\) 是该链的一个平稳分布。

证明。
从细致平衡开始：\(\pi(x)T(x \to x') = \pi(x')T(x' \to x)\)。
现在，对所有 \(x\) 求和：

\[\sum_{x} \pi(x)T(x \to x') = \sum_{x} \pi(x')T(x' \to x)\]

右边简化为：

\[\pi(x') \sum_{x} T(x' \to x) = \pi(x') \cdot 1 = \pi(x')\]

因此，\(\sum_{x} \pi(x)T(x \to x') = \pi(x')\)，这就是全局平衡条件。

===== Page 46 =====

Metropolis-Hastings：一个细致平衡机器

我们如何构建一个对于任何 \(\pi\) 都满足细致平衡的链？

算法

1. 从当前状态 \(x\)，使用**提议分布** \(q(x'|x)\) 提出一个新状态 \(x'\)。

2. 计算**接受概率**：

\[A(x,x') = \min\left(1, \frac{\pi(x') \cdot q(x|x')}{\pi(x) \cdot q(x'|x)}\right)\]

3. 以概率 \(A(x,x')\)，接受该移动并将下一个状态设为 \(x'\)。否则，拒绝并停留在 \(x\)。

转移核是：
\[T(x \to x') = q(x'|x) \cdot A(x,x')\]

===== Page 47 =====

验证 M-H 的细致平衡
我们需要检查：

\[\pi(x) \cdot T(x \to x') = \pi(x') \cdot T(x' \to x)\]

证明。

\[\pi(x) \cdot T(x \to x') = \pi(x) \cdot q(x'|x) \cdot A(x, x')\]
\[\pi(x') \cdot T(x' \to x) = \pi(x') \cdot q(x|x') \cdot A(x', x)\]

不失一般性，假设 \(\pi(x')q(x|x') > \pi(x)q(x'|x)\)。

- 那么 \(A(x, x') = 1\)
- 且 \(A(x', x) = \frac{\pi(x)q(x'|x)}{\pi(x')q(x|x')}\)

代入：

\[LHS = \pi(x)q(x'|x) \cdot 1\]
\[RHS = \pi(x')q(x|x') \cdot \frac{\pi(x)q(x'|x)}{\pi(x')q(x|x')} = \pi(x)q(x'|x)\]

===== Page 48 =====

# 随机游走问题

## Metropolis-Hastings 的局限性

- **随机地**提出新状态
- 在高维中拒绝率高
- 参数空间**探索缓慢**
- 对于相关分布效率低

===== Page 49 =====

# 哈密顿蒙特卡洛：一个物理类比

## 物理系统

- **位置** \( q \)：参数
- **势能** \( U(q) \)：
  \(- \log \pi(q)\)
- **动量** \( p \)：辅助变量
- **动能** \( K(p) \)：\( p \) 的二次型

===== Page 50 =====

# 哈密顿力学

定义（哈密顿量）
系统的总能量：

\[ H(q,p) = U(q) + K(p) \]

## 哈密顿方程

\[\frac{dq}{dt} = +\frac{\partial H}{\partial p} = \frac{\partial K}{\partial p}\] (注：通常写作 \( \frac{\partial H}{\partial p} \)，且 \( K(p) \) 通常为 \( \frac{p^T M^{-1} p}{2} \)，故 \( \frac{\partial K}{\partial p} = M^{-1} p \)，这里符号似乎有误，但按原文翻译)
\[\frac{dp}{dt} = -\frac{\partial H}{\partial q} = -\frac{\partial U}{\partial q}\] (注：这里原文是 --，应为 -)

## 定理（守恒性）

**哈密顿量守恒：**
\[\frac{dH}{dt} = 0\]

===== Page 51 =====

# 与概率的联系

## 玻尔兹曼分布

\[\pi(q,p) \propto \exp(-H(q,p)) = \exp(-U(q)) \cdot \exp(-K(p))\]

---

## 巧妙的选择

- **势能**：\( U(q) = -\log \pi(q) \)
- **动能**：\( K(p) = \frac{1}{2} p^\top M^{-1} p \)
- **动量**：\( p \sim \mathcal{N}(0,M) \)

---

## 关键见解

\( q \) 的边缘分布正是我们的目标分布 \(\pi(q)\)！

\[\pi(q) = \int \pi(q,p) dp \propto \exp(-U(q))\]

===== Page 52 =====

# Leapfrog 积分器

## 为什么使用 Leapfrog？

- **时间可逆**
- **体积守恒**
- **辛格式**（近似守恒哈密顿量）

---

## 一个 Leapfrog 步

\[p \leftarrow p - \frac{\epsilon}{2} \frac{\partial U}{\partial q}\] (动能项更新通常在位置更新之后？原文顺序如此，按原文翻译)
\[q \leftarrow q + \epsilon \frac{\partial K}{\partial p}\] (通常 \( \frac{\partial K}{\partial p} = M^{-1} p \))
\[p \leftarrow p - \frac{\epsilon}{2} \frac{\partial U}{\partial q}\]

===== Page 53 =====

# 完整 HMC 算法

1. **采样动量**：\( p \sim \mathcal{N}(0, M) \)

2. **模拟动力学**（L 个 leapfrog 步）：

   for \( i = 1 \) 到 \( L \) do
    \[    p \leftarrow p - \frac{\epsilon}{2} \nabla U(q)\]
    \[    q \leftarrow q + \epsilon M^{-1} p\] (这里使用了 \( \frac{\partial K}{\partial p} = M^{-1} p \))
    \[    p \leftarrow p - \frac{\epsilon}{2} \nabla U(q)\]

   end for

3. **Metropolis 接受步骤：**

   \[   \alpha = \min(1, \exp(H(q,p) - H(q^*,p^*)))\]

   高接受率
   由于哈密顿量守恒，\( \alpha \approx 1! \)

===== Page 54 =====

# 性质与优点

## 理论性质
- **时间可逆**
- **体积守恒**
- **哈密顿量守恒**
- **遍历性**（在温和条件下）

## 实际优点
- **远距离提议**
- **高接受率**
- **避免随机游走**
- **在高维中高效**

===== Page 55 =====

# 关键参数

## 步长 \(\epsilon\)

- **太大**：积分效果差，接受率低
- **太小**：探索缓慢，浪费计算
- **最优**：在保持高接受率的前提下尽可能大

## 轨迹长度 \(L\)

- **太小**：随机游走行为
- **太大**：浪费计算（循环）
- **挑战**：固定的 \(L\) 通常不是最优的

===== Page 56 =====

No-U-Turn 采样器

自动化轨迹长度

构建轨迹直到它开始回转（“U-turn”）

自动确定最优的 L

无需手动调整！

===== Page 57 =====

# 实际考虑

## 梯度要求

- 需要梯度 \( \nabla U(q) = -\nabla \log \pi(q) \)
- 自动微分使其可行
- 无梯度 → 使用随机游走 Metropolis

## 质量矩阵 \( M \)

- 可以适应目标分布的几何形状
- 对角 \( M \) 用于轴对齐缩放
- 完整的 \( M \) 用于相关参数

===== Page 58 =====

# Langevin 动力学：物理起源

势场中的布朗运动
描述具有摩擦和随机碰撞的粒子运动：

\[m \frac{d^2 q}{dt^2} = -\nabla U(q) - \gamma \frac{dq}{dt} + \text{随机噪声}\]

过阻尼极限（高摩擦）
当惯性效应可忽略时：

\[\gamma \frac{dq}{dt} = -\nabla U(q) + \sqrt{2\gamma k_B T} \, \eta(t)\]

其中 \(\langle \eta(t) \eta(s) \rangle = \delta(t-s)\)

## 与采样的联系
设 \(\gamma = 1\)，\(k_B T = 1\) 即得到我们的采样方程！

===== Page 59 =====

数学表述

随机微分方程

\[dq(t) = -\nabla U(q)dt + \sqrt{2}dW(t)\]
\[U(q) = -\log \pi(q)\]

Fokker-Planck 方程
概率密度 \(\rho(q, t)\) 的演化：

\[\frac{\partial \rho}{\partial t} = \nabla \cdot (\rho \nabla U) + \Delta \rho\]

平稳分布
验证 \(\rho(q) = \pi(q) \propto e^{-U(q)} \) 是平稳的：

\[\nabla \cdot (\pi \nabla U) + \Delta \pi = \nabla \cdot (\pi \nabla U - \nabla \pi) = 0\] (因为 \( \nabla \pi = -\pi \nabla U \))

===== Page 60 =====

# 未调整的 Langevin 算法

## Euler-Maruyama 离散化

\[q_{k+1} = q_k - \epsilon \nabla U(q_k) + \sqrt{2\epsilon}\xi_k, \quad \xi_k \sim \mathcal{N}(0, I)\]

---

### 性质

- **偏差**：由于离散化，平稳分布 \(\pi_\epsilon \neq \pi\)
- **误差**：\(||\pi_\epsilon - \pi||_{TV} = O(\epsilon)\)
- **简单**：易于实现，无接受/拒绝步骤
- 用于优化和近似采样

---

### 局限性

需要递减步长 \(\epsilon_k \to 0\) 以实现精确收敛：

\[\sum \epsilon_k = \infty, \quad \sum \epsilon_k^2 < \infty\]

===== Page 61 =====

Metropolis 调整的 Langevin 算法

算法

1. 提议：\( q^* = q_k - \epsilon \nabla U(q_k) + \sqrt{2\epsilon} \xi_k \)

2. 以概率接受：

\[\alpha = \min \left( 1, \frac{\pi (q^*) T(q_k | q^*)}{\pi (q_k) T(q^* | q_k)} \right)\]

其中 \( T(x'|x) = \mathcal{N}(x'; x - \epsilon \nabla U(x), 2\epsilon I) \)

最优缩放
对于 \( d \) 维分布：

- 接受率 \(\approx 57.4\%\)（最优）
- 步长 \(\epsilon = O(d^{-1/3})\)
- 比随机游走好得多：\(\epsilon = O(d^{-1})\)

===== Page 62 =====

# 预条件的 Langevin

## 病态问题
当目标具有不同长度尺度时：

\[q_{k+1} = q_k - \epsilon P \nabla U(q_k) + \sqrt{2\epsilon P}\xi_k\]

---

### 预条件子的选择

- **对角**：\(P = \text{diag}(\sigma_1^{-2}, \ldots, \sigma_d^{-2})\)
- **Fisher 信息矩阵**：\(P = I(\theta)^{-1}\)
- **经验协方差**：\(P = \text{Cov}(q)\)

---

## 带预条件的 MALA
提议变为：

\[q^* = q_k - \epsilon P \nabla U(q_k) + \sqrt{2\epsilon P}\xi_k\]

需要小心处理提议分布的不对称性。

===== Page 63 =====

随机梯度 Langevin 动力学

大数据设置
当 \( U(q) = \frac{1}{N} \sum_{i=1}^N U_i(q) \) 计算昂贵时：

\[q_{k+1} = q_k - \epsilon_k \nabla \hat{U}_B(q_k) + \sqrt{2\epsilon_k} \xi_k\]

其中 \(\nabla \hat{U}_B(q) = \frac{1}{|B|} \sum_{i \in B} \nabla U_i(q)\)

理论保证
随着递减步长 \(\epsilon_k \to 0\)：

- **收敛到真实平稳分布**
- 有误差分析可用
- 实际权衡：固定步长 vs 递减步长

===== Page 64 =====

# MCMC 方法总结

采样问题
我们想从目标分布 \(\pi(\theta)\) 中采样，我们可以评估 \(\pi(\theta)\)（可能差一个常数），但不能直接采样。

1. **随机游走 MCMC**：基本的 Metropolis-Hastings
2. **Langevin 动力学**：梯度引导的随机游走
3. **哈密顿 MCMC**：物理启发的动量动力学

**共同目标**
所有方法都构建具有平稳分布 \(\pi(\theta)\) 的马尔可夫链

===== Page 65 =====

随机游走 MCMC

算法

1. 提议：\(\theta^* \sim q(\theta^*|\theta_t)\)

2. 以概率接受：

\[\alpha = \min\left(1, \frac{\pi(\theta^*)q(\theta_t|\theta^*)}{\pi(\theta_t)q(\theta^*|\theta_t)}\right)\]

典型提议
随机游走：\(\theta^* = \theta_t + \epsilon\xi\)，\(\xi \sim \mathcal{N}(0,I)\)

性质

► 简单：易于实现
► 灵活：适用于任何提议分布
► 缓慢：随机游走行为
► 缩放：对于最优接受率，\(\epsilon = O(d^{-1})\)

===== Page 66 =====

# Langevin 动力学

## 梯度引导的提议

**提议**：
\[\theta^* = \theta_t - \frac{\epsilon^2}{2} \nabla U(\theta_t) + \epsilon \xi\]
其中 \( U(\theta) = -\log \pi(\theta) \)，
\(\xi \sim \mathcal{N}(0, I)\)

### 直觉
结合了梯度下降和随机噪声：

- 漂移向高概率区域
- 扩散用于探索

### 性质

- **更快**：梯度信息改善了混合
- **缩放**：
  \[  \epsilon = O(d^{-1/3})\]
  对于最优接受率
- **需要**：目标分布的梯度

===== Page 67 =====

# 哈密顿蒙特卡洛

## 物理启发的动力学

引入动量变量 \( p \) 并使用哈密顿动力学：

\[\frac{d\theta}{dt} = M^{-1}p\]
\[\frac{dp}{dt} = -\nabla U(\theta)\]

## Leapfrog 积分

具有体积守恒的离散模拟：

\[p \leftarrow p - \frac{\epsilon}{2}\nabla U(\theta)\]
\[\theta \leftarrow \theta + \epsilon M^{-1}p\]
\[p \leftarrow p - \frac{\epsilon}{2}\nabla U(\theta)\]

===== Page 68 =====

# 理论比较

| 属性         | 随机游走         | Langevin 动力学     | 哈密顿蒙特卡洛       |
|--------------|------------------|---------------------|----------------------|
| 提议机制     | 随机             | 梯度引导            | 哈密顿动力学         |
| 所需梯度     | 否               | 是                  | 是                   |
| 最优缩放     | \( O(d^{-1}) \)  | \( O(d^{-1/3}) \)   | \( O(d^{-1/4}) \)    |
| 接受率       | 23.4%            | 57.4%               | 65% (典型)           |
| 混合时间     | 慢               | 中等                | 快                   |
| 每步复杂度   | 低               | 中等                | 高                   |

---

**表：针对 \( d \) 维问题的理论性质**

## 关键见解
更复杂的方法使用更多信息（梯度）以在维度上实现更好的缩放

===== Page 69 =====

# 计算需求

## 每次迭代成本
- **RWM**：1 次目标函数评估
- **MALA**：1 次梯度 + 1 次目标函数
- **HMC**：\( L \) 次梯度 + \( L \) 次目标函数

(\( L = \) leapfrog 步数)

## 内存
- **RWM**：存储当前状态
- **MALA**：存储当前状态
- **HMC**：存储状态 + 动量

---

## 有效样本量
- **RWM**：每次评估的 ESS 低
- **MALA**：每次评估的 ESS 中等
- **HMC**：每次评估的 ESS 高

## 调参复杂度
- **RWM**：仅步长
- **MALA**：仅步长
- **HMC**：步长 + 轨迹长度

===== Page 70 =====

# 不同场景下的性能

## 高维问题

- **RWM**：对于 \( d > 20 \) 变得不实用
- **MALA**：适用于中等维度
- **HMC**：适用于高维复杂分布

## 相关分布

- **RWM**：难以处理强相关性
- **MALA**：可以处理轻度相关性
- **HMC**：自然地遵循相关结构

## 多峰分布

- **RWM**：可能陷入局部众数
- **MALA**：更好的众数切换能力
- **HMC**：可以在远距离众数间跳跃

===== Page 71 =====

# 何时使用每种方法

## 在以下情况使用随机游走 MCMC：
- 目标是低维的 (\(d < 10\))
- 梯度不可用或计算昂贵
- 实现简单性是优先考虑因素
- 分布简单且良态

## 在以下情况使用 Langevin 动力学：
- 中等维度 (\(d \approx 10 - 100\))
- 梯度可用
- 需要简单性和效率的良好平衡
- 可以接受步长调参

## 在以下情况使用哈密顿蒙特卡洛：
- 高维复杂分布
- 梯度可用
- 计算效率至关重要
- 愿意投入调参（或使用 NUTS）

===== Page 72 =====

# 总结与建议

## 采样方法的演进

- **RWM**：基础，简单但低效
- **MALA**：梯度信息提高了效率
- **HMC**：物理直觉实现了最优探索

## 现代最佳实践

- 如果梯度可用，从 HMC/NUTS 开始
- 对于中等问题使用 MALA
- 仅对简单的低维情况使用 RWM
- 考虑计算成本与混合时间的权衡

## 关键权衡

**简单性 vs 效率**：更复杂的方法需要更多的实现工作和调参，但对于复杂问题能提供显著更好的性能。

---