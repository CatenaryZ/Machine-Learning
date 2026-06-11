## Lecture 1: Probability I

> - 机器学习课程的基本组成
> - 一维与多维概率分布
> - 变量变换、期望、方差与协方差
> - 熵、互信息、KL 散度与 Gibbs 不等式
> - 常见概率分布的回顾

### 机器学习的核心组成

- 训练机器学习模型的基本 ingredients 包括：
  1. Model：通常是概率模型，用来反映现实世界的概率性。
  2. Data：通常表示成向量、矩阵或张量。
  3. Training：通过优化来寻找函数极小值。
  4. Inference：
     - 在新数据上做预测；
     - 在生成式 AI 中生成新的数据。

- 机器学习在本质上就是借助计算机来仔细调节参数，而现代训练又高度依赖并行计算和 GPU。

### 机器学习的应用

- 机器学习方法可以解决多种问题：
  1. 回归分析，包括线性与非线性曲线拟合。
  2. 分类任务。
  3. 聚类问题。
  4. 生成式 AI：
     - 翻译；
     - 文本、图像、音频、视频生成。

#### 学习范式
- Supervised learning：回归与分类。
- Unsupervised learning：聚类。
- Reinforcement learning：强化学习。

### 机器学习的新阶段

#### 大语言模型的兴起
- 大语言模型（LLMs）的快速发展释放出了前所未有的人工智能能力，也推动了整个领域的深刻变化。

#### 本课程的目标
- 本课程希望帮助我们深入理解机器学习所需的数学基础，用于：
  - 理解和构建传统机器学习模型；
  - 理解大语言模型背后的核心原理。

### 课程信息

#### 主要参考书
- C.M. Bishop, *Pattern Recognition and Machine Learning*, 2006.
- C. Bishop, H. Bishop, *Deep Learning: Foundations and Concepts*, 2024.
- D.J. MacKay, *Information Theory, Inference and Learning Algorithms*, 2003.

#### 额外资源
- 高级大语言模型可以用来做探索和代码辅助。

### 常用 Python 库

#### 核心机器学习库
- Scikit-learn：通用机器学习算法。
- PyTorch：灵活的深度学习研究框架。
- Transformers：先进的自然语言处理工具。

#### 使用场景
- Scikit-learn：传统机器学习任务。
- PyTorch：自定义神经网络和研究原型。
- Transformers：NLP、文本生成、翻译等任务。

### 实用信息

- Office Hours：周四 2:00–4:00 PM。
- 成绩构成：
  - 40% Homework
  - 60% Final Exam and projects

---

### Chapter 1: Probability I

### 概率基础：简要回顾

- 本课程的核心假设是：我们观测到的数据来自某个潜在的概率分布。

#### 一维概率模型
- 一维概率模型由概率密度函数（pdf）或概率质量函数（pmf）$p(x)$ 描述，并满足：

$$
p(x) \ge 0, \qquad \int p(x)\,dx = 1.
$$

- 这些函数可分成两类：
  - Continuous：$p(x)$ 定义在连续变量上。
  - Discrete：$p(x)$ 只在有限或可数多个点上非零。

### 多元分布

- 概率的概念可以自然推广到高维情形。

#### 联合概率

$$
p(x,y)
$$

#### 边缘分布

$$
p(x)=\sum_y p(x,y)
$$

或

$$
p(x)=\int p(x,y)\,dy.
$$

#### 条件概率

$$
p(x\mid y)=\frac{p(x,y)}{p(y)}, \qquad p(y)>0.
$$

#### 乘法法则

$$
p(x,y)=p(x\mid y)p(y)=p(y\mid x)p(x).
$$

### 与物理的联系

- 讲义提到 statistical physics。
- 例如统计分布可写成

$$
\rho(p,q)=\frac{\exp(-E(p,q))}{Z}.
$$

- 其中 $p,q$ 分别是动量与位置变量，$Z$ 是配分函数。
- 许多机器学习中的洞见都与物理学有联系。

### 变量变换

#### 一维情形
- 若 $Y=g(X)$，其逆变换为 $X=h(Y)$，则

$$
f_Y(y)=f_X(h(y))\cdot \left|\frac{dh}{dy}\right|.
$$

- 这里：
  - $f_X(h(y))$ 是把原 PDF 代入逆变换后的值；
  - $\left|\frac{dh}{dy}\right|$ 是 Jacobian 的绝对值；
  - 绝对值保证无论变换单调增减都成立。

#### 例子：线性变换
- 设

$$
X \sim \mathrm{Uniform}(0,1), \qquad f_X(x)=1, \quad 0<x<1.
$$

- 定义

$$
Y = 2X + 5.
$$

则：
1. 逆变换为

$$
X = h(Y)=\frac{Y-5}{2}.
$$

2. 导数为

$$
\frac{dh}{dy}=\frac12.
$$

3. 因此

$$
f_Y(y)=1\cdot\frac12=\frac12.
$$

4. 支撑区间是

$$
5<y<7.
$$

- 所以

$$
Y \sim \mathrm{Uniform}(5,7).
$$

#### 多维情形
- 若 $Y=g(X)$，逆变换为 $X=h(Y)$，则

$$
f_Y(y)=f_X(h(y))\cdot |J|,
$$

其中 $J$ 是 Jacobian 行列式。
- $|J|$ 表示体积缩放因子。

### 概率分布的刻画量

#### 均值

$$
\mu=\mathbb E[x]=\int x p(x)\,dx.
$$

#### 方差

$$
\sigma^2 = \mathbb E[(x-\mu)^2]
= \int (x-\mu)^2 p(x)\,dx.
$$

#### 函数的期望

$$
\mathbb E[f(X)] = \int f(x)p(x)\,dx.
$$

### 二维分布中的协方差

#### 定义

$$
\mathrm{Cov}(X,Y)=\mathbb E[(X-\mu_X)(Y-\mu_Y)].
$$

其中 $\mu_X=\mathbb E[X]$，$\mu_Y=\mathbb E[Y]$。

#### 等价形式

$$
\mathrm{Cov}(X,Y)=\mathbb E[XY]-\mathbb E[X]\mathbb E[Y].
$$

#### 解释
- 正协方差：$X$ 和 $Y$ 倾向于一起变化。
- 负协方差：$X$ 和 $Y$ 倾向于反向变化。
- 零协方差：没有线性关系，但仍可能存在非线性依赖。

### 什么是熵

- 熵衡量随机变量的不确定性或随机性。
- 对离散随机变量，

$$
H(X) = -\sum_{x\in\mathcal X} P(x) \log_2 P(x).
$$

- 若使用底数 2 的对数，单位就是 bit。
- 熵越大，不确定性越高；熵越小，可预测性越强。

### 例子：公平六面骰子

- 若 $X$ 表示掷骰子的结果，且

$$
P(X=i)=\frac16, \qquad i=1,\dots,6,
$$

则

$$
H(X)
= -\sum_{i=1}^6 P(i)\log_2 P(i)
= -6\cdot \frac16 \log_2 \frac16
= \log_2 6
\approx 2.585 \text{ bits}.
$$

#### 解释
- 这表示平均而言，需要约 2.585 bit 来编码一次掷骰结果。
- 对 6 个等概率结果而言，这已经是最大不确定性。
- 如果骰子偏置更强，熵就会更小。

### 联合熵

$$
H(X,Y) = -\sum_x \sum_y P(x,y) \log_2 P(x,y).
$$

- 讲义用两次公平抛硬币举例说明：

$$
H(X,Y)=2 \text{ bits}.
$$

### 条件熵

$$
H(Y\mid X) = -\sum_x \sum_y P(x,y) \log_2 P(y\mid x).
$$

- 还有一个等价形式：

$$
H(Y\mid X)=H(X,Y)-H(X).
$$

- 若 $Y=X$，则 $H(Y\mid X)=0$。
- 若 $X$ 与 $Y$ 独立，则 $H(Y\mid X)=H(Y)$。

### 熵的链式法则

- 一般形式：

$$
H(X_1,\dots,X_n)=\sum_{i=1}^n H(X_i \mid X_1,\dots,X_{i-1}).
$$

- 对两个变量：

$$
H(X,Y)=H(X)+H(Y\mid X)=H(Y)+H(X\mid Y).
$$

### 互信息

#### 定义

$$
I(X;Y)=\sum_x\sum_y P(x,y)\log_2\frac{P(x,y)}{P(x)P(y)}.
$$

#### 与熵的关系

$$
I(X;Y)=H(X)+H(Y)-H(X,Y)
$$

$$
= H(X)-H(X\mid Y)=H(Y)-H(Y\mid X).
$$

- 它也等于联合分布与边缘分布乘积之间的 KL 散度。

### Relative Entropy

#### 定义

$$
D(P\|Q)=\sum_x\sum_y P(x,y)\log_2\frac{P(x,y)}{Q(x,y)}.
$$

#### 性质
- $D(P\|Q)\ge 0$。
- 当且仅当 $P=Q$ 时取 0。
- 它不对称。

### Gibbs 不等式

- Gibbs 不等式说明 KL divergence 总是非负：

$$
D_{KL}(P\|Q) \ge 0.
$$

#### 用到的基本不等式

$$
\log t \le t-1, \qquad t>0.
$$

- 讲义分步骤说明了如何从这个对数不等式推出 Gibbs inequality。

### 重要概率分布

#### Gaussian / Normal Distribution

$$
p(x\mid \mu,\sigma)=\frac{1}{\sqrt{2\pi}\sigma}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).
$$

- 参数是均值 $\mu$ 和方差 $\sigma^2$。

#### Bernoulli Distribution

$$
\mathrm{Ber}(x\mid \mu)=\mu^x(1-\mu)^{1-x}.
$$

- $P(x=1)=\mu$，$P(x=0)=1-\mu$。
- 均值是 $\mu$，方差是 $\mu(1-\mu)$。

#### 多元高斯分布

$$
P(x\mid \mu,\Sigma)
= \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}
\exp\left(-\frac12 (x-\mu)^T\Sigma^{-1}(x-\mu)\right).
$$

- 参数是均值向量 $\mu$ 与协方差矩阵 $\Sigma$。

#### Categorical Distribution

$$
P(t=i)=p_i, \qquad i=1,\dots,K, \qquad \sum_{i=1}^K p_i = 1.
$$
