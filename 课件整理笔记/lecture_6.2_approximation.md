## Lecture 6.2: Inference Part 2

> - 潜变量模型、ELBO 与 EM 算法
> - 变分推断、mean-field 与 Ising 模型
> - 采样方法的基本谱系
> - MCMC、HMC、Langevin 与方法选择

### Section 2: EM Algorithm

### 问题设定

- 讲义开头考虑的是带潜变量的概率模型。
- 设：
  - 观测数据为 $X=\{x_1,x_2,\dots,x_N\}$；
  - 潜变量为 $Z=\{z_1,z_2,\dots,z_N\}$；
  - 模型参数为 $\theta$。

- 我们希望最大化观测数据的边缘似然，也叫 evidence：

$$
\log p_\theta(X)=\log \int p_\theta(X,Z)\,dZ.
$$

- 难点在于：这个积分通常算不出来，尤其是在复杂模型中。

### 变分视角的切入

- 讲义从 variational inference 的角度引入 EM。
- 做法是引入一个近似分布 $q_\phi(Z)$，用它来近似真实后验 $p_\theta(Z\mid X)$。

- 于是

$$
\log p_\theta(X)
=
\log \int q_\phi(Z)\frac{p_\theta(X,Z)}{q_\phi(Z)}\,dZ.
$$

- 接下来对上式应用 Jensen 不等式，就得到 ELBO。

### ELBO 的推导

#### 第一步：引入辅助分布

- 只是把积分中的被积函数乘除以 $q_\phi(Z)$，本身没有做任何近似。

#### 第二步：Jensen 不等式

- 因为 $\log$ 是凹函数，所以

$$
\log \mathbb{E}_{q_\phi}\!\left[\frac{p_\theta(X,Z)}{q_\phi(Z)}\right]
\ge
\mathbb{E}_{q_\phi}\!\left[\log \frac{p_\theta(X,Z)}{q_\phi(Z)}\right].
$$

- 右边就是 evidence lower bound，记作

$$
\mathcal{L}(\theta,\phi)
=
\mathbb{E}_{q_\phi(Z)}[\log p_\theta(X,Z)]
-\mathbb{E}_{q_\phi(Z)}[\log q_\phi(Z)].
$$

### ELBO 的等价形式

- 讲义给出两种常见写法。

#### 形式 1

$$
\mathcal{L}(\theta,\phi)
=
\mathbb{E}_{q_\phi(Z)}[\log p_\theta(X,Z)]
-\mathbb{E}_{q_\phi(Z)}[\log q_\phi(Z)].
$$

#### 形式 2

$$
\mathcal{L}(\theta,\phi)
=
\mathbb{E}_{q_\phi(Z)}[\log p_\theta(X\mid Z)]
-D_{\mathrm{KL}}(q_\phi(Z)\Vert p_\theta(Z)).
$$

- 这两个形式都很重要，因为它们分别强调了不同的含义。

### ELBO 的解释

#### 从“重构 + 正则”的角度看

- 第一项 $\mathbb{E}_{q_\phi}[\log p_\theta(X\mid Z)]$ 可以理解成 reconstruction term。
- 它衡量的是：如果用潜变量 $Z$ 去解释数据 $X$，解释得有多好。

- 第二项 $D_{\mathrm{KL}}(q_\phi(Z)\Vert p_\theta(Z))$ 是 regularization term。
- 它要求近似后验不要偏离先验太远。

#### 从优化目标看

- 最大化 ELBO，等价于在“解释数据”和“保持分布规整”之间做平衡。
- 这也是变分方法后来在深度学习里非常常见的原因。

### ELBO 与后验 KL 的恒等式

- 讲义给出了最关键的恒等式：

$$
\log p_\theta(X)
=
\mathcal{L}(\theta,\phi)
+D_{\mathrm{KL}}(q_\phi(Z)\Vert p_\theta(Z\mid X)).
$$

- 因为 KL divergence 总是非负，所以

$$
\mathcal{L}(\theta,\phi)\le \log p_\theta(X).
$$

- 这就是 ELBO 名字里 “lower bound” 的来源。

#### 这个式子的意义

- 如果 $q_\phi(Z)$ 刚好等于真实后验 $p_\theta(Z\mid X)$，那么 KL 项为 0。
- 此时 ELBO 就与对数证据完全相等。

- 所以：
  - 做变分推断，本质是在尽量把 $q$ 推向真实后验；
  - 做 EM，本质是在交替地让这个下界变紧并把下界抬高。

### EM 的两步

### E-step

- 固定旧参数 $\theta^{\text{old}}$，最大化 ELBO 关于 $q$ 的部分。
- 最优选择恰好是真实后验：

$$
q^{\text{opt}}(Z)=p(Z\mid X,\theta^{\text{old}}).
$$

- 这一步等价于：在当前参数下，先推断潜变量“应该长什么样”。

- 讲义把对应的目标写成 $Q$-function：

$$
Q(\theta,\theta^{\text{old}})
=
\mathbb{E}_{p(Z\mid X,\theta^{\text{old}})}[\log p(X,Z\mid \theta)] + H(q^{\text{opt}}),
$$

其中 $H(q)$ 是熵项。

### M-step

- 固定刚才得到的后验分布，再更新参数：

$$
\theta^{\text{new}}=\arg\max_\theta Q(\theta,\theta^{\text{old}}).
$$

- 这一步通常比直接最大化 $\log p_\theta(X)$ 容易得多，因为它把“潜变量带来的困难”转移到了 E-step 中。

### 为什么 EM 有效

- 讲义证明了一个重要性质：EM 每一步都不会让对数似然下降。

$$
\log p(X\mid \theta^{\text{new}})
\ge
\log p(X\mid \theta^{\text{old}}).
$$

- 理由是：
  - E-step 让当前下界在旧参数处变得最紧；
  - M-step 再把这个下界往上推；
  - 而真实对数似然始终在下界之上。

- 所以 EM 具有单调上升保证。

### EM 的一般伪代码

1. 初始化参数 $\theta^{\text{old}}$。
2. 重复直到收敛：
   - E-step：计算 $p(Z\mid X,\theta^{\text{old}})$ 或相关期望；
   - M-step：最大化 $Q(\theta,\theta^{\text{old}})$ 得到 $\theta^{\text{new}}$；
   - 用新参数替换旧参数。

- 这是一种典型的交替优化策略。

### GMM 例子

- 讲义用 GMM 说明 EM 如何真正落地。

#### E-step：责任度

$$
\gamma_{ik}
=
\frac{\pi_k\mathcal{N}(X_i\mid \mu_k,\Sigma_k)}
{\sum_{j=1}^K \pi_j\mathcal{N}(X_i\mid \mu_j,\Sigma_j)}.
$$

- 这表示样本 $X_i$ 属于第 $k$ 个高斯分量的后验概率。

#### M-step：参数更新

- 记

$$
N_k=\sum_{i=1}^N \gamma_{ik},
$$

然后更新

$$
\pi_k^{\text{new}}=\frac{N_k}{N},
\qquad
\mu_k^{\text{new}}=\frac{1}{N_k}\sum_{i=1}^N \gamma_{ik}X_i.
$$

- 协方差的更新与此同理，也是用责任度做加权平均。

- 这个例子很重要，因为它清楚展示了：
  - E-step 是“软分配”；
  - M-step 是“带权重的参数重估计”。

### 这一节的意义

- 讲义最后总结了 EM 的典型应用：
  - Gaussian Mixture Models；
  - Hidden Markov Models；
  - Topic models；
  - 缺失数据填补；
  - 各类带潜变量的模型。

---

### Section 3: Variational Methods

### 什么是变分方法

- 变分方法的核心思想是：
  - 不直接去算难解的后验分布；
  - 而是在一个较简单的分布族 $q(z;\lambda)$ 里找一个最接近真实后验的成员。

- 它把“难以解析求积分”的问题，转成了“可以用优化解决的问题”。

### 关键公式

- 这一部分的中心仍然是

$$
\log p_\theta(X)=\mathcal{L}(\theta,q_\phi)+D_{\mathrm{KL}}(q_\phi(Z)\Vert p_\theta(Z\mid X)).
$$

- 于是：
  - 近似后验越接近真实后验，KL 越小；
  - ELBO 越大；
  - 近似也就越好。

### Variational Inference Framework

#### 目标

- 用简单分布 $q(z)$ 逼近复杂后验 $p(z\mid x)$。

#### 做法

- 选择一个可处理的分布族 $q(z;\lambda)$；
- 以 $\lambda$ 为变量最大化 ELBO；
- 这等价于最小化

$$
D_{\mathrm{KL}}(q(z)\Vert p(z\mid x)).
$$

#### 和 EM 的关系

- EM 是变分方法的一个特例。
- 在 EM 中，E-step 实际上是在所有可能的 $q$ 中，直接取真实后验，因此下界会被“拉紧”到最优。

### 例子：Ising Model 的 Mean Field

### 真分布

- 对 Ising model，讲义写出的 Boltzmann 分布为

$$
p(s)=\frac1Z e^{\beta J\sum_{\langle ij\rangle}s_is_j+\beta h\sum_i s_i}.
$$

- 这里 $s_i\in\{-1,+1\}$。

### Mean-field 假设

- 为了近似这个复杂耦合分布，讲义取一个完全因子化的近似：

$$
q(s)=\prod_{i=1}^N q_i(s_i).
$$

- 这就是 mean-field assumption：用“彼此独立”的近似分布去逼近本来强耦合的真实分布。

### 参数化方式

- 对二值变量，讲义把每个因子写成

$$
q_i(s_i)=\frac{1+m_i s_i}{2},
\qquad
\mathbb{E}_q[s_i]=m_i.
$$

- 所以优化变量从整个分布函数，变成了每个位置上的均值参数 $m_i$。

### Mean-field ELBO

- 目标仍然是最大化

$$
\mathcal{L}[q]=\mathbb{E}_q[\log p(s)]-\mathbb{E}_q[\log q(s)].
$$

- 第一项是 energy term；
- 第二项负号后对应 entropy term。

- 讲义最后得到完整 ELBO 形式

$$
\mathcal{L}[\{m_i\}]
=
\beta J\sum_{\langle ij\rangle} m_i m_j
+\beta h\sum_i m_i
-\sum_i S(m_i)-\log Z,
$$

其中 $S(m_i)$ 是二值熵。

### 自洽方程

- 对 $m_i$ 求偏导并令其为 0，讲义推得

$$
\frac12 \log\frac{1+m_i}{1-m_i}
=
2\beta J\sum_{j\in \text{n.n.}(i)} m_j+\beta h.
$$

- 利用 $\operatorname{arctanh}(x)=\frac12\log\frac{1+x}{1-x}$，可写成

$$
m_i=
\tanh\!\left(
2\beta J\sum_{j\in \text{n.n.}(i)} m_j+\beta h
\right).
$$

- 这就是 mean-field 的 self-consistency equation。

#### 它在说什么

- 每个位置的均值磁化 $m_i$，由其邻居的均值磁化共同决定。
- 也就是说，复杂耦合并没有消失，而是被压缩进一个自洽固定点方程里。

### CAVI 实现

- 讲义给出了 coordinate ascent variational inference 的形式：
  1. 初始化 $m_i$；
  2. 逐个位置计算邻居和；
  3. 更新

$$
m_i^{\text{new}}=\tanh(2\beta J\cdot \text{neighbor sum}+\beta h_i);
$$

  4. 直到所有 $m_i$ 基本稳定。

- 这就是“用局部更新不断提高 ELBO”的典型变分算法。

---

### Section 4: Sampling Methods

### 为什么需要采样

- 当积分太复杂、归一化常数难算、后验过于复杂时，采样方法提供了第三条路：
  - 不强求解析公式；
  - 而是用样本来近似期望。

- 例如

$$
\mathbb{E}_{p(x)}[f(x)]
=
\int f(x)p(x)\,dx
\approx
\frac1N\sum_{i=1}^N f(x_i).
$$

- 问题就变成了：怎样高效地产生来自目标分布的样本。

### 方法谱系

- 讲义把采样方法分成两大类：
  - 基本采样方法：inverse transform、rejection sampling、importance sampling；
  - MCMC 方法：Metropolis-Hastings、Gibbs、HMC、Langevin 等。

### Inverse Transform Sampling

#### 思想

- 如果能计算并反演 CDF，那么只要先采样

$$
u\sim \operatorname{Uniform}(0,1),
$$

再令

$$
x=F^{-1}(u),
$$

就能得到目标分布样本。

#### 例子

- 对指数分布 $p(x)=\lambda e^{-\lambda x}$，

$$
F(x)=1-e^{-\lambda x},
\qquad
F^{-1}(u)=-\frac{\ln(1-u)}{\lambda}.
$$

#### 优缺点

- 优点：精确、直观。
- 缺点：必须能求逆 CDF，高维下通常不现实。

### Rejection Sampling

#### 基本做法

1. 找一个 proposal $q(x)$ 和常数 $M$，使得

$$
p(x)\le M q(x).
$$

2. 先从 $q(x)$ 采样 $x$。
3. 再采样 $u\sim \operatorname{Uniform}(0,1)$。
4. 若

$$
u<\frac{p(x)}{M q(x)},
$$

则接受，否则拒绝。

#### 关键点

- 接受率约为 $1/M$。
- proposal 越贴近目标分布，效率越高。
- 高维下常常因为 $M$ 很大而变得低效。

### Importance Sampling

#### 核心思想

- 不要求直接从目标分布 $p(x)$ 采样，而是从容易采样的 $q(x)$ 采样，再用权重修正。

#### 算法

1. 采样 $x_i\sim q(x)$。
2. 计算权重

$$
w_i=\frac{p(x_i)}{q(x_i)}.
$$

3. 归一化

$$
\tilde w_i=\frac{w_i}{\sum_j w_j}.
$$

4. 估计期望

$$
\mathbb{E}_{p(x)}[f(x)]\approx \sum_{i=1}^N \tilde w_i f(x_i).
$$

#### 有效样本数

- 讲义给出

$$
\text{ESS}=\frac{1}{\sum_{i=1}^N \tilde w_i^2}.
$$

- 它衡量真正“有用”的样本量。

#### 主要问题

- 如果大多数权重都很小、少数样本权重极大，就会出现 weight degeneracy。
- 高维下这个问题尤其严重。

---

### MCMC Fundamentals

### 核心想法

- MCMC 的目标不是独立采样，而是构造一个 Markov chain，使其平稳分布正好是目标分布 $p(x)$。

#### 重要概念

- Markov property：

$$
p(x_{t+1}\mid x_t,x_{t-1},\dots,x_1)=p(x_{t+1}\mid x_t).
$$

- stationary distribution：链长期运行后收敛到的分布。
- burn-in：前期未稳定样本通常要丢弃。
- ergodicity：保证链最终会收敛到目标分布。

### Metropolis-Hastings

#### 算法

1. 从当前状态 $x_t$ 出发，用 proposal 分布采样候选点

$$
x^*\sim q(x^*\mid x_t).
$$

2. 计算接受率

$$
\alpha
=
\min\left(
1,\frac{p(x^*)q(x_t\mid x^*)}{p(x_t)q(x^*\mid x_t)}
\right).
$$

3. 采样 $u\sim \operatorname{Uniform}(0,1)$。
4. 若 $u<\alpha$，接受 $x^*$；否则留在原点。

#### 对称 proposal 的简化

- 若 $q(x^*\mid x)=q(x\mid x^*)$，则

$$
\alpha=\min\left(1,\frac{p(x^*)}{p(x_t)}\right).
$$

- 这就是最常见的 random walk MH 形式。

### Gibbs Sampling

- Gibbs sampling 是 Metropolis-Hastings 的特例。
- 它一次只更新一个变量，而且从对应的 full conditional 里直接采样。

- 对 $p(x_1,\dots,x_D)$，其更新模式是
  - 先采 $x_1\mid x_2,\dots,x_D$；
  - 再采 $x_2\mid x_1,x_3,\dots,x_D$；
  - 依此循环。

- 优点是接受率恒为 1；
- 缺点是必须能方便地从条件分布采样，而且混合速度可能慢。

### MCMC 的理论基础

### Transition Kernel

- 讲义把链的转移规律写成 transition kernel $T$。
- 离散情形下可写成 $T(x\to x')$，连续情形下可写成核函数。

### Stationary Distribution

- 若分布 $\pi$ 满足

$$
\pi(x')=\sum_x \pi(x)T(x\to x'),
$$

则称 $\pi$ 是该链的平稳分布。

### Detailed Balance

- 全局平衡常不好直接验证，所以引入更强但更容易检查的条件：

$$
\pi(x)T(x\to x')=\pi(x')T(x'\to x).
$$

- 若 detailed balance 成立，则 $\pi$ 一定是平稳分布。

#### 为什么它重要

- 它把“整个分布不变”这种全局性质，变成了任意两个状态之间的局部流量平衡。
- 这正是 MH 能系统构造采样链的原因。

### MH 为什么满足 detailed balance

- 讲义专门说明，若定义接受率为

$$
A(x,x')=
\min\left(
1,\frac{\pi(x')q(x\mid x')}{\pi(x)q(x'\mid x)}
\right),
$$

并令

$$
T(x\to x')=q(x'\mid x)A(x,x'),
$$

- 那么就能验证 detailed balance 成立。

- 因而 MH 是一个“为任意目标分布制造平稳链”的通用机器。

---

### Hamiltonian Monte Carlo

### 为什么随机游走不够好

- 讲义指出，普通 random walk MCMC 在高维或强相关分布下常常很慢：
  - 候选点几乎是盲走；
  - 拒绝率可能很高；
  - 探索范围小。

- HMC 的目标就是：利用梯度信息，让采样更像“沿着高概率区域滑行”，而不是瞎撞。

### 物理类比

- HMC 引入一组辅助动量变量 $p$。
- 对应关系是：
  - 位置 $q$：参数；
  - 势能 $U(q)$：取为 $-\log \pi(q)$；
  - 动量 $p$：辅助变量；
  - 动能 $K(p)$：常取二次型。

- 总能量，也就是 Hamiltonian，为

$$
H(q,p)=U(q)+K(p).
$$

### Hamilton 方程

$$
\frac{dq}{dt}=+\frac{\partial H}{\partial p},
\qquad
\frac{dp}{dt}=-\frac{\partial H}{\partial q}.
$$

- 连续理想动力学下，总能量守恒。
- 这意味着如果我们能精确模拟运动，轨迹会沿着近似等能曲线长距离移动。

### Leapfrog Integrator

- 真正计算时不能精确解动力学，所以讲义使用 leapfrog：

$$
p\leftarrow p-\frac{\varepsilon}{2}\frac{\partial U}{\partial q},
$$

$$
q\leftarrow q+\varepsilon\frac{\partial K}{\partial p},
$$

$$
p\leftarrow p-\frac{\varepsilon}{2}\frac{\partial U}{\partial q}.
$$

- 它的好处是：
  - time-reversible；
  - volume-preserving；
  - 近似保持 Hamiltonian。

### 完整 HMC 算法

1. 先采动量 $p\sim \mathcal{N}(0,M)$。
2. 做 $L$ 步 leapfrog：

$$
p\leftarrow p-\frac{\varepsilon}{2}\nabla U(q),\qquad
q\leftarrow q+\varepsilon M^{-1}p,\qquad
p\leftarrow p-\frac{\varepsilon}{2}\nabla U(q).
$$

3. 用 Metropolis 步校正离散化误差：

$$
\alpha=\min\bigl(1,\exp(H(q,p)-H(q^*,p^*))\bigr).
$$

### HMC 的优势

- 候选点不再是无方向的随机扰动；
- 能在高概率区域上走得更远；
- 接受率通常较高；
- 在高维问题上扩展性更好。

### 关键参数

#### 步长 $\varepsilon$

- 太大：数值积分误差大，接受率低。
- 太小：每步走得太短，计算浪费。

#### 轨迹长度 $L$

- 太小：又退化回 random walk。
- 太大：会浪费计算，甚至开始来回折返。

### NUTS

- No-U-Turn Sampler 的思路是：让轨迹自动增长，直到开始“掉头”为止。
- 它的意义是自动决定合适的轨迹长度，减少手动调参负担。

### 何时适合用 HMC

- 高维复杂分布；
- 梯度可得；
- 对采样效率要求高；
- 愿意为算法调参，或者使用 NUTS 这类自适应版本。

---

### Langevin Dynamics

### 基本思想

- Langevin 类方法把随机噪声和梯度漂移项结合起来。
- 它不像 random walk 那样纯随机，也不像 HMC 那样引入完整动量系统，而是走一条中间路线。

### 连续时间形式

- 讲义把其来源写成布朗运动在势场中的随机微分方程。
- 当 $U(q)=-\log \pi(q)$ 时，系统自然会倾向于往高概率区域移动。

### ULA

- Euler-Maruyama 离散化得到

$$
q_{k+1}=q_k-\varepsilon \nabla U(q_k)+\sqrt{2\varepsilon}\,\xi_k,
\qquad \xi_k\sim \mathcal{N}(0,I).
$$

- 这就是 unadjusted Langevin algorithm。

#### 特点

- 优点：简单，不需要接受拒绝步骤。
- 缺点：因为离散化误差，平稳分布不完全等于目标分布，存在 bias。

### MALA

- 在 Langevin proposal 外再加一个 Metropolis 校正，就得到 MALA。

#### proposal

$$
q^*=q_k-\varepsilon \nabla U(q_k)+\sqrt{2\varepsilon}\,\xi_k.
$$

#### 接受率

$$
\alpha=
\min\left(
1,\frac{\pi(q^*)T(q_k\mid q^*)}{\pi(q_k)T(q^*\mid q_k)}
\right),
$$

其中 $T(\cdot\mid\cdot)$ 是对应高斯 proposal 的转移密度。

#### 讲义中的比较

- 对高维问题，MALA 的最优尺度优于 random walk MH。
- 讲义给出的量级是：
  - Random Walk：步长标度约为 $O(d^{-1})$；
  - MALA：约为 $O(d^{-1/3})$。

- 这说明利用梯度确实能显著改善高维表现。

### Preconditioned Langevin

- 当目标分布各方向尺度差异很大时，可引入预条件矩阵 $P$：

$$
q_{k+1}=q_k-\varepsilon P\nabla U(q_k)+\sqrt{2\varepsilon P}\,\xi_k.
$$

- 它的作用是按不同方向的几何结构重新缩放更新步长。

### SGLD

- 大数据场景下，完整梯度太贵。
- 讲义因此引入 stochastic gradient Langevin dynamics：

$$
q_{k+1}=q_k-\varepsilon_k \nabla \hat U_B(q_k)+\sqrt{2\varepsilon_k}\,\xi_k,
$$

其中 $\nabla \hat U_B(q)$ 是 mini-batch 梯度估计。

- 这样就把随机优化里的小批量思想带进了采样算法。

### MCMC 方法比较

### 理论比较

- 讲义给出的比较是：
  - Random Walk：最简单，但混合最慢；
  - Langevin / MALA：利用梯度，速度明显改善；
  - HMC：利用更完整的动力学结构，在高维上通常最好。

- 对维度标度，讲义列出：
  - Random Walk：$O(d^{-1})$；
  - Langevin：$O(d^{-1/3})$；
  - Hamiltonian：$O(d^{-1/4})$。

### 不同场景下的表现

- 高维问题：
  - RWM 往往很快失效；
  - MALA 适合中等维度；
  - HMC 最有优势。

- 强相关分布：
  - RWM 很难沿着相关方向高效移动；
  - HMC 更容易顺着几何结构前进。

- 多模态分布：
  - 所有局部方法都会有挑战，但纯随机游走通常最容易卡住。

### 什么时候用哪种方法

- Random Walk MCMC：
  - 低维；
  - 梯度不可得；
  - 追求实现简单。

- Langevin / MALA：
  - 中等维度；
  - 梯度可得；
  - 希望在复杂度与效率间取平衡。

- HMC / NUTS：
  - 高维；
  - 梯度可得；
  - 强调采样质量与效率。

### 本讲复习时可以抓住的主线

- 第一条主线是：面对潜变量，直接最大化似然很难，所以 EM 用“先估潜变量、再更参数”的方式迭代优化下界。
- 第二条主线是：如果后验本身太复杂，就用变分方法找一个简单分布去逼近它，mean-field 是其中最典型的例子。
- 第三条主线是：如果连近似分布优化也不想做，或者更想通过样本来表达不确定性，那就转向 sampling methods。
- 最后在 MCMC 家族里，方法的发展脉络很清楚：
  - 从 random walk；
  - 到利用梯度的 Langevin；
  - 再到利用动力学结构的 HMC。

- 整讲真正贯穿始终的问题只有一个：
  - 当精确推断做不到时，我们到底是“优化一个近似分布”，还是“构造一个能正确采样的随机过程”。
