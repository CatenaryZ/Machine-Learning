## Lecture 3: Non-linear Effects

> - 线性模型的限制与基函数扩展
> - 广义线性模型与链接函数
> - 神经网络的基本结构与前向传播
> - 损失函数、反向传播与 softmax 分类
> - 深层网络的表达能力与计算代价

### 线性模型：假设与局限 I

#### 线性回归的核心概率假设
- 标准线性模型建立在两个基本假设上：
  1. Gaussian Noise：目标变量 $y$ 在其均值附近服从高斯分布。
  2. Linear Mean：这个分布的均值是输入特征 $x$ 的线性组合。

也就是说：

$$
\mathbb E[y \mid x] = w^T x = w_0 + w_1 x_1 + \cdots + w_D x_D.
$$

于是得到熟悉的模型：

$$
y = w^T x + \epsilon, \qquad \epsilon \sim \mathcal N(0, \sigma^2).
$$

#### 如何应对这些局限
- 这些假设对真实数据往往过于严格。
- 一个强有力的办法是把输入投影到特征空间，使用非线性基函数 $\phi_j(x)$：

$$
\mathbb E[y \mid x] = \sum_{j=0}^M w_j \phi_j(x).
$$

- 模型对参数 $w_j$ 仍然是线性的，因此优化仍然容易。
- 但它已经能够表示复杂的非线性关系。

### 广义线性模型与常见基函数

- 形如

$$
\mathbb E[y \mid x] = \sum_j w_j \phi_j(x)
$$

的模型，是 generalized linear models 和 linear basis function models 的基础。

#### 常见基函数
- Polynomial：

$$
\phi_j(x)=x^j
$$

适合简单曲线，影响范围是全局性的。

- Gaussian：

$$
\phi_j(x)=\exp\left(-\frac{(x-\mu_j)^2}{2s_j^2}\right)
$$

对应局部的“凸起”，也是通用逼近器。

- Sigmoidal：
- Fourier：

$$
\sin(\omega_j x), \quad \cos(\omega_j x)
$$

适合周期数据。

#### 特征工程与现代做法
- 特征工程的关键往往在于选择合适的基函数类型及其参数，如 $\mu_j$、$s_j$。
- 深度学习提供了另一种思路：不再手工设计特征，而是直接从数据中学习分层特征 $\phi(x)$。

### 什么是广义线性模型

- GLM 是对线性回归的推广。
- 它允许响应变量不再必须服从正态分布。

#### 三个组成部分
1. Random component：响应变量的概率分布。
2. Systematic component：线性预测子

$$
\eta = w^T \phi(x).
$$

3. Link function：

$$
g(\mu)=\eta.
$$

于是：

$$
\mathbb E[y\mid x] = \mu = g^{-1}(w^T\phi(x)).
$$

### 训练

- 损失函数可以像线性模型那样，从最大似然估计推导出来。
- 一般没有精确闭式解，因此需要使用 SGD 等迭代优化方法。

### 分类：链接函数

- 在 logistic regression 中，我们用 sigmoid 函数来建模概率。

#### 一般化形式
- 我们可以把它推广为任意一个把实数映射到 $[0,1]$ 的函数

$$
f : \mathbb R \to [0,1],
$$

从而写成：

$$
p(t=1 \mid w,x)=f(w^T x).
$$

- 在 GLM 语境下，这种函数叫 link function；在神经网络语境下，也常称 activation function。

#### 概率解释
- 任意连续随机变量的 CDF $F$ 都可以作为合法的链接函数：

$$
f(x) = \int_{-\infty}^x p(\theta)\,d\theta = F(x).
$$

- sigmoid 函数就是 logistic 分布的 CDF。

### 二分类中常见的链接函数

- Logistic (Sigmoid)。
- Probit。
- Log-Log。
- Complementary Log-Log。

#### 说明
- 不同链接函数会影响模型在分布尾部的表现。

#### 用基函数处理非线性
- 如果需要更复杂的非线性决策边界，可以先做基函数展开：

$$
p(t=1 \mid w,x) = f\left(\sum_{j=0}^M w_j \phi_j(x)\right).
$$

- 这样模型仍然对参数线性，但在原始输入空间中可以表现出非线性。

### 神经网络的思路

> 对随机变量 $y$ 的均值进行编码，另一种办法是使用神经网络。事实证明，神经网络是极其强大的函数逼近器。

### 神经网络基础

#### 定义
- 神经网络定义了一个带参数的函数 $y(x;w)$，其中：
  - $x$：输入变量。
  - $y$：输出变量。
  - $w$：网络参数。

#### 基本组成
1. Nodes / neurons：计算单元。
2. Edges：节点之间带权连接。

#### 分层结构
- 对于分层网络，第 $i$ 层中某个节点的函数形式为：

$$
z_k^{(i)}
= h\left(\sum_j w_{kj}^{(i)} z_j^{(i-1)} + b_k^{(i)}\right).
$$

### 网络实现

#### 简化记号
- 如果在每一层加入一个固定为 1 的偏置节点，就可以把表达式写得更紧凑。

#### 前向传播过程
- 输入层取 $i=0$。
- 递归地计算各个隐藏层。
- 最后一层得到输出。
- 若有多个输出节点，就得到向量输出。

### 网络结构的选择

#### 1. 结构设计
- 需要确定层数。
- 需要确定每层的节点数。
- 输入层规模由数据维度决定。
- 如果隐藏层很多，就得到 deep neural network。

#### 2. 激活函数
- 常见激活函数包括：
  - Logistic Sigmoid
  - Tanh
  - Hard Tanh
  - Softplus
  - ReLU
  - Leaky ReLU

### 神经网络的概率模型

- 对回归问题，可以写成：

$$
P(y \mid x, w, \sigma^2) = \mathcal N(y \mid y(x,w), \sigma^2).
$$

- 其中 $y(x,w)$ 由神经网络给出。
- 类似地，也可以构造分类问题下的概率模型。

### 神经网络的损失函数

#### 一般形式
- 无论是回归还是分类，损失函数都可以自然推广：

$$
E(w)=\sum_{n=1}^N (t_n - y(x_n,w))^2 + \lambda \sum_i w_i^2.
$$

其中：
- $y(x_n,w)$：神经网络输出。
- $\lambda$：正则化参数。

### 单样本损失

- 总损失可以分解为逐样本损失之和：

$$
E = \sum_n E_n.
$$

- 对于单个样本，

$$
E_n = \frac12 [t_n - y(x_n,w)]^2.
$$

#### 优化
- 可以用 gradient descent 来最小化 $E(w)$。
- 梯度通过 backpropagation algorithm 高效计算。

### 反向传播

- 对某个参数 $w_{kj}^{(i)}$ 求偏导，利用链式法则可得：

$$
\frac{\partial E_n}{\partial w_{kj}^{(i)}}
= \delta_k^{(i)} z_j^{(i-1)}.
$$

- 其中局部误差项定义为：

$$
\delta_k^{(i)} = \frac{\partial E_n}{\partial a_k^{(i)}}.
$$

- 这些误差项可以递归地从输出层往前传播。
- 初始条件来自输出端。

### 用神经网络做分类

#### softmax 输出
- 若网络有 $K$ 个输出节点 $y_i$，则可通过 softmax 得到分类概率：

$$
p_i = \frac{\exp(y_i)}{\sum_{j=1}^K \exp(y_j)}.
$$

#### 温度参数
- 还可以引入 temperature $T$：

$$
p_i(T) = \frac{\exp(y_i/T)}{\sum_{j=1}^K \exp(y_j/T)}.
$$

- $T$ 较高时，输出概率更平滑。
- $T$ 较低时，输出分布更尖锐。

### 生成式应用

- 神经网络还可以参数化许多概率分布。
- 这为生成式 AI 提供了基础，例如文本生成与图像扩散模型。

### 深层神经网络

#### 函数表示
- 神经网络提供了一种几何化的方式来表示复杂的非线性函数。
- 它们是 universal function approximator。

#### 深度的优势
- 从经验上看，更深的网络在许多任务上表现更好。

#### 训练挑战
1. 梯度问题：
   - Vanishing gradients。
   - Exploding gradients。
2. 计算复杂度：
   - 参数规模可从百万到十亿，甚至更高。
   - 计算资源需求巨大。

### 对策

- 架构创新用来缓解梯度问题。
- GPU 加速使大规模训练成为可能。

### 计算上的考虑

#### 矩阵运算
- 神经网络的主要计算是矩阵乘法。
- 这类运算非常适合并行化。
- GPU 特别擅长这类并行计算。

#### 效率
- 现代 GPU 可以同时执行成千上万次运算。
- 专门的 tensor cores 还能进一步加速训练。

### Practical Example

讲义最后预告了下一部分：神经网络实现的 hands-on demonstration。
