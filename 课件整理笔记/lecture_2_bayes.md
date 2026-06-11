## Lecture 2: Probability Foundations, Bayesian View, and Linear Models

> - 概率基础与最大熵分布
> - 多元高斯分布的边缘与条件性质
> - i.i.d. 假设与贝叶斯观点
> - MAP、Laplace 近似与数值稳定性
> - 线性回归、逻辑回归与梯度下降

### 概率论基础 I

#### 概率密度函数
- 对连续随机变量 $X$，概率分布由函数 $p(x)$ 描述，并满足：
  - 非负性：$p(x) \ge 0$；
  - 归一化：$\int p(x)\,dx = 1$。

#### 多元分布中的几个基本量
- 联合密度：$p(x,y)$。
- 边缘密度：$p(x)=\int p(x,y)\,dy$，也就是把另一个变量“积分掉”。
- 条件密度：$p(x\mid y)$ 与 $p(y\mid x)$。

### 概率论基础 II

#### 乘法法则
- 这些密度之间的基本关系是：

$$
p(x,y)=p(x\mid y)p(y)=p(y\mid x)p(x).
$$

- 这正是 Bayes 定理的基础：

$$
p(y\mid x)=\frac{p(x\mid y)p(y)}{p(x)}.
$$

### 概率论基础 III

#### 分布的重要刻画量
- Mean（均值）$\mu$：描述中心位置。
- Variance（方差）$\sigma^2$：描述离散程度。
- Entropy（熵）$H$：描述平均不确定性或信息量。

#### 相关定义
- 均值：$\mathbb E[X]=\mu=\int x p(x)\,dx$。
- 方差：$\mathrm{Var}(X)=\sigma^2=\int (x-\mu)^2 p(x)\,dx$。
- 对离散变量，熵写成 $H(X)=-\sum p(x)\log_2 p(x)$。
- 此外还有联合熵、条件熵、互信息，以及相对熵 $D(P\|Q)$。

### 熵的作用

#### 最大熵分布定理
- 在所有均值固定为 $\mu$、方差固定为 $\sigma^2$ 的连续概率分布中，高斯分布

$$
q(x)=\mathcal N(x\mid \mu,\sigma^2)
$$

具有最大的微分熵。

- 也就是说：

$$
h(p)=-\int p(x)\log p(x)\,dx \le \frac12 \log(2\pi e\sigma^2)=h(q).
$$

- 当且仅当 $p(x)=q(x)$ 时取等号。

> 结论：在均值和方差给定的前提下，高斯分布是“信息最少”的分布。

### 证明思路：KL 散度

#### KL divergence
- 从 $q$ 到 $p$ 的 KL 散度定义为：

$$
D_{KL}(p\|q)=\int p(x)\log \frac{p(x)}{q(x)}\,dx.
$$

- 关键性质是：

$$
D_{KL}(p\|q) \ge 0,
$$

且仅当 $p(x)=q(x)$ 几乎处处成立时取等号。

#### Step 1: 展开 KL 散度
- 对目标高斯分布 $q$，有

$$
D_{KL}(p\|q)
= \int p(x)\log p(x)\,dx - \int p(x)\log q(x)\,dx
= -h(p) - \int p(x)\log q(x)\,dx.
$$

- 因为 $D_{KL}(p\|q)\ge 0$，于是得到：

$$
h(p) \le -\int p(x)\log q(x)\,dx.
$$

#### Step 2: 计算上界
- 把高斯分布 $q(x)$ 的对数写开，再带入上式。
- 最终需要计算的就是 $-\int p(x)\log q(x)\,dx$。

#### Step 3: 使用约束
- 利用 $p(x)$ 的两个已知约束：
  - 归一化：$\int p(x)\,dx=1$；
  - 方差约束：$\int p(x)(x-\mu)^2\,dx=\sigma^2$。

- 代入后得到：

$$
-\int p(x)\log q(x)\,dx = \frac12 \log(2\pi\sigma^2) + \frac12.
$$

#### Step 4: 化简
- 进一步化简为：

$$
\frac12 \log(2\pi e\sigma^2)=h(q).
$$

#### Step 5: 结论
- 于是

$$
h(p) \le h(q).
$$

- 只有当 $D_{KL}(p\|q)=0$ 时取等号，也就是 $p=q$。

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

- $P(x=0)=1-\mu$，$P(x=1)=\mu$。
- 均值为 $\mu$，方差为 $\mu(1-\mu)$。

### 多元分布

#### 多元高斯分布

$$
p(x\mid \mu,\Sigma)=
\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}
\exp\left(-\frac12 (x-\mu)^T\Sigma^{-1}(x-\mu)\right).
$$

- 其中 $\mu$ 是均值向量，$\Sigma$ 是协方差矩阵。

#### Categorical Distribution
- 对 $K$ 个类别，

$$
P(t=i)=p_i, \quad i=1,\ldots,K, \qquad \sum_{i=1}^K p_i=1.
$$

#### One-Hot 编码
- $t$ 可以表示成 one-hot 向量。
- 例如：
  - Class 1：$[1,0,0,\ldots,0]$；
  - Class 2：$[0,1,0,\ldots,0]$；
  - Class $K$：最后一个位置为 1。

### 多元高斯：结构与性质

#### 基本记号
- 若 $X=[X_1,\ldots,X_D]^T$ 服从多元高斯，则记作

$$
X \sim \mathcal N(\mu,\Sigma).
$$

- 分布完全由均值和协方差决定：
  - $\mathbb E[X]=\mu$；
  - $\mathrm{Cov}[X]=\Sigma$。

#### 分块表示
- 为了讨论边缘分布和条件分布，把向量分块：
- 可以把 $X$ 写成由 $X_a$ 和 $X_b$ 组成的分块向量，把 $\mu$ 写成由 $\mu_a$ 和 $\mu_b$ 组成的分块均值向量。
- 相应地，协方差矩阵 $\Sigma$ 也按相同方式分块为 $\Sigma_{aa}, \Sigma_{ab}, \Sigma_{ba}, \Sigma_{bb}$。

#### 边缘分布
- 边缘分布仍然是高斯：
  - $p(X_a)=\mathcal N(X_a\mid \mu_a,\Sigma_{aa})$；
  - $p(X_b)=\mathcal N(X_b\mid \mu_b,\Sigma_{bb})$。

#### 条件分布
- 条件分布也仍然是高斯：

$$
p(X_a\mid X_b)=\mathcal N(X_a\mid \mu_{a\mid b}, \Sigma_{a\mid b}),
$$

其中

$$
\mu_{a\mid b}=\mu_a + \Sigma_{ab}\Sigma_{bb}^{-1}(X_b-\mu_b),
$$

$$
\Sigma_{a\mid b}=\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}.
$$

- 条件均值是 $X_b$ 的线性函数。
- 条件协方差与 $X_b$ 的具体数值无关，这是高斯分布的特殊性质。

#### 小结
- 高斯族在线性变换、边缘化和条件化下都是封闭的。
- 若 $\Sigma_{ab}=0$，则两部分独立，此时 $p(X_a\mid X_b)=p(X_a)$。

### 高维分布建模

- 除了多元高斯外，课程后续主要讨论三类高维模型：
  1. i.i.d. 模型；
  2. Markov Chains；
  3. Graphical Models。

- 当前先聚焦第一类：i.i.d. 模型。

### 什么是 i.i.d.

- i.i.d. 表示 Independent and Identically Distributed。
- 这是统计与机器学习中非常常见、也非常关键的假设。

#### Identically Distributed
- 所有变量服从同一个分布。
- 因而它们有相同的均值、相同的方差，以及相同的概率规律。

#### Independent
- 一个变量的结果不会影响另一个变量。
- 联合概率可以写成边缘概率的乘积。

#### 合在一起
- 若 $X_1,\ldots,X_n$ 既相互独立，又来自同一分布，就称它们构成一个 i.i.d. 序列。

### i.i.d. 假设为什么重要

- 它极大简化了数学处理，并支撑了很多核心定理：
  - Law of Large Numbers；
  - Central Limit Theorem。

- 在机器学习中，很多统计推断与学习算法都建立在这个假设上。

### 贝叶斯观点 I

- 机器学习模型通常从一个带参数的概率模型出发。
- 核心目标是：根据观测数据估计参数 $w$。

#### 贝叶斯估计
- 贝叶斯方法把参数 $w$ 本身也看作随机变量，因此有

$$
p(w\mid x)=\frac{p(x\mid w)p(w)}{p(x)}.
$$

其中：
- $p(w)$：先验；
- $p(x\mid w)$：似然；
- $p(w\mid x)$：后验；
- $p(x)$：evidence / marginal likelihood。

### 贝叶斯观点 II

- 后验分布是看到数据之后对参数的更新认识。
- evidence 是归一化项。

### 后验难以求解的问题

- 对复杂模型而言，我们希望得到完整的后验分布

$$
p(w\mid D)=\frac{p(D\mid w)p(w)}{p(D)}.
$$

- 难点在于：

$$
p(D)=\int p(D\mid w)p(w)\,dw
$$

通常不可解析。

#### 常见处理办法
- MCMC；
- Variational Inference；
- MAP 与 Laplace Approximation。

### MAP 与 Laplace Approximation

#### MAP 估计

$$
w^*_{MAP}=\arg\max_w p(w\mid x)=\arg\max_w p(x\mid w)p(w).
$$

#### MLE 是特例
- 当先验是均匀分布时，MAP 退化为 MLE：

$$
w^*_{MLE}=\arg\max_w p(x\mid w)=\arg\max_w \prod_{i=1}^N p(x_i\mid w).
$$

#### Laplace 近似的思想
- 用一个高斯分布 $q(w)$ 去近似真实后验 $p(w\mid D)$。
- 具体做法：
  1. 先找到后验众数，也就是 MAP 点；
  2. 再匹配该点附近的曲率。

#### 数学推导
- 定义

$$
E(w) = -\log p(D\mid w) - \log p(w).
$$

- 在 $w_{MAP}$ 附近做二阶 Taylor 展开，记 Hessian 为 $H$，则

$$
q(w) = \mathcal N(w\mid w_{MAP}, H^{-1}).
$$

#### 优点与局限
- 优点：
  - 直观；
  - 把积分问题转成优化问题；
  - 给出完整近似分布，而不仅是一个点估计。

- 局限：
  - 只是局部近似；
  - 对多峰、偏斜或重尾后验不理想；
  - 需要计算和求逆 Hessian。

### 极大似然中的数值问题

- 直接连乘许多小概率，容易发生下溢。
- 标准做法是改为最小化负对数似然。

#### 好处
- 乘积变和；
- 数值更稳定；
- 在优化框架中通常也更方便。

---

### Chapter 2: Linear Model

### 线性回归 I

- 监督学习中的基本任务之一，是给定 $N$ 个输入输出对：

$$
D=\{(x_1,y_1),(x_2,y_2),\ldots,(x_N,y_N)\}.
$$

- 其中：
  - $x_n \in \mathbb R^D$ 是输入特征；
  - $y_n \in \mathbb R$ 是实值目标。

### 线性回归 II

#### 概率模型
- 假设目标变量是输入的线性函数加上高斯噪声：

$$
P(y\mid x,w,\sigma^2)=\mathcal N(y\mid w^T x, \sigma^2).
$$

- 均值是

$$
\mu = w^T x = w_0 + w_1 x_1 + \cdots + w_D x_D.
$$

- 这里 $w_0$ 是 bias，常把输入扩展成 $x=[1,x_1,\ldots,x_D]^T$。

### 线性回归 III

- 最大化似然（对 $w$ 采用均匀先验）等价于最小化负对数似然，也就得到平方损失：

$$
E(w;D)=\frac12 \sum_{n=1}^N (y_n - w^T x_n)^2.
$$

- 最大似然解满足 normal equations：

$$
w_{ML}=(X^T X)^{-1} X^T y.
$$

### 防止过拟合：Regularization I

- 为了抑制过拟合并提高泛化，引入惩罚项：

$$
E(w;D,\lambda)=\sum_{n=1}^N (y_n - w^T x_n)^2 + \lambda \lVert w \rVert_2^2.
$$

- 这就是 L2 regularization，也就是 ridge regression。

#### 说明
- $\lambda \ge 0$ 是控制正则强度的超参数。

### 防止过拟合：Regularization II

- 在贝叶斯框架下，这等价于给参数放一个高斯先验：

$$
p(w)=\mathcal N(w\mid 0, \lambda^{-1} I).
$$

- 最大化后验就对应最小化上面的正则化目标。

#### 解

$$
w^*=(X^T X + \lambda I)^{-1}X^T y.
$$

- 这里 $(X^T X + \lambda I)$ 总是可逆，这是 L2 正则化的一个重要优点。

### Logistic Regression

- 现在数据写成 $(t_n,x_n)$，其中 $t_n \in \{0,1\}$。

#### 二分类模型

$$
P(t\mid x,w)=\sigma(w^T x)^t (1-\sigma(w^T x))^{1-t},
$$

其中

$$
\sigma(x)=\frac{1}{1+\exp(-x)}.
$$

#### 联合似然

$$
p(D\mid w)=\prod_{n=1}^N \sigma(w^T x_n)^{t_n}
\bigl(1-\sigma(w^T x_n)\bigr)^{1-t_n}.
$$

#### 损失函数
- 负对数似然为

$$
E(w;D)= -\sum_{n=1}^N
\left[
t_n \log \sigma(w^T x_n)
+ (1-t_n) \log(1-\sigma(w^T x_n))
\right].
$$

### 多分类

- 若 $t_n$ 可取 $K$ 个离散值，则用 softmax regression：

$$
p_i(w;x)=\frac{\exp(w_i^T x)}{\sum_j \exp(w_j^T x)}.
$$

- 相应的损失函数来自负对数似然：

$$
E(D;w)= -\sum_{n=1}^N \sum_{i=1}^K t_{ni}\log p_i(w;x_n).
$$

- 这个损失就是 cross entropy。

### 小结

- 给定概率假设，并用 Bayes / MAP 的思路，就会得到参数的损失函数 $E(D;w)$。
- 下一步就是求它的最小值 $w^*$。
- 实际中通常很难得到精确解析解。

### 优化问题

- 机器学习的大部分工作都可以写成：

$$
w^*=\arg\min_w J(w).
$$

- 对复杂模型，解析解往往不存在，因此需要迭代算法。
- 最基本的算法就是 Gradient Descent。

### 梯度下降：直觉

- 梯度 $\nabla J(w)$ 指向最陡上升方向。
- 为了最小化目标，我们朝相反方向走，即 $-\nabla J(w)$。
- 步长由学习率 $\eta$ 控制。

### 核心更新公式

$$
w^{(k+1)} = w^{(k)} - \eta \nabla_w J(w^{(k)}).
$$

#### 停止准则
- 达到最大迭代次数；
- 梯度足够小；
- 损失函数变化足够小。

### 梯度下降的变体

#### Batch Gradient Descent
- 用整个训练集计算梯度，方向准确，但对大数据较慢。

#### Stochastic Gradient Descent
- 每次只用一个随机样本，单步很快，但更新噪声大。

#### Mini-batch Gradient Descent
- 每次用一个小批量样本，是最常用的折中方案。

### 学习率的重要性

- 太小：收敛很慢。
- 太大：会震荡、越界，甚至发散。
- 合适：稳定且高效收敛。

### 回归中的概率推断

- 在得到最优参数 $w^*$ 后，对新输入 $x^*$，最简单的点预测是均值：

$$
\mathbb E[y^*\mid x^*,D]=w^{*T}x^*.
$$

- 若要反映不确定性，则考虑完整预测分布：

$$
p(y^*\mid x^*,D)=\int p(y^*\mid x^*,w)p(w\mid D)\,dw.
$$

### 分类中的概率推断

- 对新输入 $x^*$，可以先用 $w^*$ 算出类别概率，再选择概率最大的类。
- 贝叶斯方式则同样是对参数后验积分：

$$
p(y^*=c\mid x^*,D)=\int p(y^*=c\mid x^*,w)p(w\mid D)\,dw.
$$

### 机器学习模型的三个支柱

1. Model Formulation：建立概率模型，并得到损失函数。
2. Parameter Estimation：用训练算法求参数。
3. Prediction / Inference：在新数据上做预测。

#### 训练阶段
- 通常采用 SGD 及其变体。
- 常见超参数包括学习率、batch size、epoch 数。

#### 推断阶段
- 输入是新的自变量 $x_{new}$。
- 输出是 $y_{new}$ 的预测，或者完整预测分布。
