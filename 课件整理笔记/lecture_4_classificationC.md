## Lecture 4.3: Kernel Method

> - 核方法与核技巧
> - 凸优化、原始问题与对偶问题
> - 核岭回归与核 SVM
> - 高斯过程的先验与后验预测

### 处理非线性

我们可以通过两种主要方式来处理非线性效应。

#### 1. 显式特征变换
- 用一组基函数（basis functions）$\phi(x)$ 对输入特征做变换。
- 目标变量也可以通过链接函数（link function）$g(\mu)$ 来变换。
- 模型对参数仍然保持线性：

$$
\mathbb E[y] = w^T \phi(x).
$$

#### 2. 隐式表示学习
- 均值由输入和参数的一个复杂非线性函数给出。
- 这个函数由神经网络之类的模型直接学习。

接下来，讲义转向另一条强有力的路线：基于核（kernel）的方法，以及更一般的 Gaussian Process。

### 特征映射 $\phi$ 与核技巧

#### 显式特征映射
- 先计算 $\phi(x)$，再计算

$$
\langle \phi(x), \phi(x') \rangle.
$$

- 这种做法可能计算代价很高。

#### 核技巧
- 跳过显式变换，直接在输入空间中通过核函数计算内积：

$$
k(x, x') = \langle \phi(x), \phi(x') \rangle.
$$

#### 核技巧的定义
- 如果一个算法能够完全用内积来表述，那么我们就可以把每一个

$$
\langle x_i, x_j \rangle
$$

替换成

$$
k(x_i, x_j),
$$

从而把它变成非线性的算法。

### 常见核函数

#### 线性核
- 最简单的核函数，不需要额外映射。
- 它等价于普通点积：

$$
k(x, x') = x^T x'.
$$

#### 多项式核
- 它对应次数为 $d$ 的多项式特征映射，可学习多项式决策边界：

$$
k(x, x') = (x^T x' + c)^d.
$$

#### 径向基函数核 / 高斯核
- 这是最常用的一类核。
- 它把数据隐式映射到一个无限维特征空间，灵活性很高：

$$
k(x, x') = \exp\bigl(-\gamma \lVert x - x' \rVert^2\bigr).
$$

- 参数 $\gamma$ 控制单个训练样本的影响范围，也对应决策边界的平滑程度。

### 应用与小结

#### 核化算法
- 许多经典算法都有 kernelized version：
  - 支持向量机（SVM）：最经典的应用。
  - Kernel Ridge Regression：用于非线性回归。
  - Gaussian Processes：一种完整的贝叶斯核方法。

#### 核方法的力量
- Efficiency：在高维空间中工作，但不显式承担高维计算代价。
- Flexibility：能够建模复杂的非线性关系。
- Generality：只要能设计出衡量相似性的核函数，就可以把机器学习应用到图、序列等非向量数据上。

> 核方法把“设计特征空间”与“在这个空间中学习”这两件事分离开来。

### 凸优化与对偶

- 机器学习中经常出现约束优化问题。
- 我们希望在满足约束的同时最小化损失。
- 例子包括：
  - SVM：在正确分类的同时最大化间隔。
  - 正则化：在保持权重较小的同时降低误差。

#### 基本问题
- 我们怎样高效求解约束优化问题？

### 原始问题（Primal Problem）的形式

一般约束优化可以写成：

**最小化**

$$
f(w)
$$

**满足约束**

$$
g_i(w) \le 0, \quad i = 1, \ldots, k
$$

$$
h_j(w) = 0, \quad j = 1, \ldots, m.
$$

其中：
- $w$：模型参数。
- $f(w)$：目标函数 / 损失函数。
- $g_i(w)$：不等式约束。
- $h_j(w)$：等式约束。

### 拉格朗日方法

#### 核心思想
- 通过引入拉格朗日乘子（Lagrange multipliers），把有约束问题转化成无约束问题。

#### 拉格朗日乘子
- 对于 $g_i(w) \le 0$，引入 $\alpha_i \ge 0$。
- 对于 $h_j(w) = 0$，引入 $\beta_j$。

#### 拉格朗日函数

$$
L(w, \alpha, \beta)
= f(w) + \sum_{i=1}^k \alpha_i g_i(w) + \sum_{j=1}^m \beta_j h_j(w).
$$

### 拉格朗日乘子的直观理解

- 约束决定了可行域（feasible region）。
- 拉格朗日乘子可以看作“违反约束的价格”。
- 它体现了目标优化与约束满足之间的平衡。

### 从 Primal 到 Dual

#### 拉格朗日对偶函数

$$
G(\alpha, \beta) = \min_w L(w, \alpha, \beta).
$$

#### 对偶问题

**最大化**

$$
G(\alpha, \beta)
$$

**满足**

$$
\alpha_i \ge 0, \quad i = 1, \ldots, k.
$$

- 我们从 minimization 切换成了 maximization。
- 约束往往会变得更简单。
- 因而很多时候更容易求解。

### 对偶定理

#### 弱对偶
- 对任意可行的 $w$ 和 $\alpha \ge 0$，都有

$$
G(\alpha, \beta) \le f(w).
$$

- 也就是说，对偶问题总能给原问题提供一个下界。

#### 强对偶
- 在某些条件下（如凸性与 Slater 条件），有

$$
d^* = p^*.
$$

- 这里 $p^*$ 是 primal 最优值，$d^*$ 是 dual 最优值。

### 对偶性的图像理解

- primal 的目标值会向最优点下降。
- dual 的目标值会向最优点上升。
- 在最优点处，若强对偶成立，就有

$$
p^* = d^*.
$$

### 支持向量机

#### SVM 的原始问题

$$
\min \; \frac12 \lVert w \rVert^2 + C \sum_{i=1}^n \xi_i
$$

满足

$$
y_i (w \cdot x_i + b) \ge 1 - \xi_i, \qquad \xi_i \ge 0.
$$

#### SVM 的对偶问题

$$
\max \; \sum_{i=1}^n \alpha_i
- \frac12 \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \, x_i \cdot x_j
$$

满足

$$
0 \le \alpha_i \le C, \qquad \sum_{i=1}^n \alpha_i y_i = 0.
$$

### 核技巧与 SVM

- 对偶形式暴露出样本之间的点积 $x_i \cdot x_j$。
- 这正是 kernel trick 可以介入的位置：

$$
x_i \cdot x_j \longrightarrow K(x_i, x_j).
$$

- 这样就能得到非线性决策边界。

### 对偶形式的优点

- 约束更简单，很多时候只是对 $\alpha_i$ 的边界约束。
- 可以自然引入 kernel trick。
- 能揭示问题结构，比如支持向量。
- 数值上往往条件更好。
- 可以把模型解释为在高维特征空间中的隐式学习。

### 局限与实际考虑

#### 计算上的问题
- 变量个数往往等于约束个数。
- 对大规模数据集，这个规模可能很大。
- 有时需要专门的求解器。

#### 理论上的局限
- 强对偶并不总能保证成立。
- 可能存在 duality gap。

#### 实际上的挑战
- 需要从 dual 解恢复 primal 解。
- 对偶变量本身未必容易解释。
- 实现起来也可能更复杂。

#### 何时使用对偶形式
- 当原始约束很复杂时。
- 当需要 kernel method 时。
- 当问题具有适合对偶求解的特殊结构时。

### 小结

- 拉格朗日对偶把约束优化问题转化成另一种更易处理的形式。
- 弱对偶给出下界，强对偶在良好条件下给出精确最优值。
- 它对 SVM 和核方法尤其关键。

---

### Kernel Ridge Regression

#### 问题形式
- Ridge Regression 最小化带惩罚项的最小二乘目标：

$$
J(w) = \lVert y - Xw \rVert^2 + \lambda \lVert w \rVert^2.
$$

其中：
- $X \in \mathbb R^{n \times d}$ 是设计矩阵。
- $y \in \mathbb R^n$ 是目标向量。
- $w \in \mathbb R^d$ 是权重向量。
- $\lambda \ge 0$ 是正则化参数。

#### 推导

$$
\nabla_w J(w) = -2X^T(y - Xw) + 2\lambda w
$$

$$
0 = -X^T y + X^T X w + \lambda w
$$

$$
X^T y = (X^T X + \lambda I) w
$$

#### 闭式解

$$
\hat w_{ridge} = (X^T X + \lambda I)^{-1} X^T y.
$$

- 当 $\lambda > 0$ 时，矩阵 $X^T X + \lambda I$ 总是可逆。
- 相比普通最小二乘，正则化提升了数值稳定性。
- 当 $\lambda = 0$ 时，解退化为 OLS。

### Ridge Regression 的对偶形式

#### 原始问题

$$
\min_w \; \frac12 \lVert y - Xw \rVert^2 + \frac\lambda2 \lVert w \rVert^2.
$$

其中 $X \in \mathbb R^{n \times d}$，$y \in \mathbb R^n$，$w \in \mathbb R^d$，且 $\lambda > 0$。

#### Representer Theorem 的启发
- 解可以写成训练样本的线性组合：

$$
w = X^T \alpha = \sum_{i=1}^n \alpha_i x_i,
$$

其中 $\alpha \in \mathbb R^n$ 是对偶变量。

#### 代入原始问题

$$
\lVert y - Xw \rVert^2 = \lVert y - XX^T \alpha \rVert^2 = \lVert y - K\alpha \rVert^2
$$

$$
\lVert w \rVert^2 = \alpha^T XX^T \alpha = \alpha^T K \alpha
$$

其中 $K = XX^T$ 是 Gram 矩阵。

#### 对偶问题

$$
\min_\alpha \; \frac12 \lVert y - K\alpha \rVert^2 + \frac\lambda2 \alpha^T K \alpha.
$$

其解为：

$$
\hat \alpha = (K + \lambda I)^{-1} y.
$$

对新点 $x$ 的预测为：

$$
\hat y = w^T x = \alpha^T Xx = \sum_{i=1}^n \alpha_i x_i^T x.
$$

- 这时再把普通内积替换成一般核函数 $K(x, y)$，就得到 kernel ridge regression。

### Kernel Support Vector Machine

#### 目标
- 寻找超平面

$$
w^T x + b = 0
$$

使其以最大间隔分离两类样本，其中 $y_i \in \{-1, +1\}$。

#### 点到超平面的距离

$$
\frac{|w^T x_i + b|}{\lVert w \rVert}
= \frac{y_i(w^T x_i + b)}{\lVert w \rVert}.
$$

- 因而间隔为

$$
M = \frac{2}{\lVert w \rVert}.
$$

最大化间隔等价于求解：

$$
\min_{w,b} \frac12 \lVert w \rVert^2
$$

满足

$$
y_i(w^T x_i + b) \ge 1, \quad \forall i.
$$

- 这是一个二次规划（QP）问题。

### 软间隔 SVM

- 真实数据通常并不是完全可分的，因此引入松弛变量 $\xi_i$ 允许误分类：

$$
\min_{w,b,\xi} \frac12 \lVert w \rVert^2 + C \sum_{i=1}^n \xi_i
$$

满足

$$
y_i(w^T x_i + b) \ge 1 - \xi_i, \qquad \xi_i \ge 0.
$$

- 参数 $C$ 控制“大间隔”与“正确分类”之间的权衡。

### 软间隔 SVM 的对偶形式

#### 拉格朗日函数

$$
L(w, b, \xi, \alpha, \beta)
= \frac12 \lVert w \rVert^2
+ C \sum_{i=1}^n \xi_i
- \sum_{i=1}^n \alpha_i [y_i(w^T x_i + b) - 1 + \xi_i]
- \sum_{i=1}^n \beta_i \xi_i
$$

其中 $\alpha_i \ge 0$，$\beta_i \ge 0$。

#### 对偶问题

$$
\max_\alpha \sum_{i=1}^n \alpha_i
- \frac12 \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j
$$

满足

$$
\sum_{i=1}^n \alpha_i y_i = 0
$$

$$
0 \le \alpha_i \le C, \quad i = 1, \ldots, n.
$$

- 再次把 $x_i^T x_j$ 替换为核函数 $K(x_i, x_j)$，就得到 kernel SVM。

---

### 超越参数模型

#### 参数模型
- 例如线性回归：

$$
y = w^T \phi(x) + \epsilon.
$$

- 它学习的是固定维度的参数 $w$，灵活性有限。

#### 非参数模型
- 例如 Gaussian Process。
- 它不学习一个固定的参数向量 $w$。
- 它直接在可能的函数 $f(x)$ 上定义一个概率分布。
- 模型复杂度会随着数据量增加而增长。

### Gaussian Process 的核心思想

#### 定义
- Gaussian Process 是一组随机变量的集合，并且任意有限个随机变量都服从一致的联合高斯分布。

#### 类比
- Gaussian distribution：向量上的分布。
- Gaussian process：函数上的分布。

#### 一个 GP 由两部分完全决定
- 均值函数：

$$
m(x) = \mathbb E[f(x)].
$$

- 协方差函数（核函数）：

$$
k(x, x') = \mathbb E[(f(x)-m(x))(f(x')-m(x'))].
$$

记作：

$$
f(x) \sim \mathcal{GP}(m(x), k(x, x')).
$$

### GP 的核心：核函数

- 核函数 $k(x, x')$ 决定了函数值 $f(x)$ 与 $f(x')$ 之间的协方差。
- 它编码了我们对函数性质的先验假设。

### 先验分布

- 我们在函数空间上放置先验。
- 常见做法是取零均值：

$$
f(x) \sim \mathcal{GP}(0, k(x, x')).
$$

- 对于任意有限点集 $X = \{x_1, \ldots, x_N\}$，函数值向量

$$
f = [f(x_1), \ldots, f(x_N)]^T
$$

服从多元高斯分布：

$$
f \sim \mathcal N(0, K),
$$

其中 $K_{ij} = k(x_i, x_j)$。

### 含噪观测

- 实际观测满足

$$
y_i = f(x_i) + \epsilon_i, \qquad \epsilon_i \sim \mathcal N(0, \sigma_n^2).
$$

- 于是观测目标 $y$ 与潜在函数值 $f$ 的联合分布可写成相应的高斯形式，其中观测部分的协方差是 $K + \sigma_n^2 I$。

### 后验预测分布

- 对新测试点 $x_*$，我们关注

$$
p(f(x_*) \mid X, y, x_*).
$$

- 该后验预测分布仍然是高斯分布：

$$
p(f(x_*) \mid X, y, x_*) \sim \mathcal N(\bar f_*, \mathbb V[f_*]).
$$

#### 预测均值

$$
\bar f_* = k_*^T (K + \sigma_n^2 I)^{-1} y.
$$

#### 预测方差

$$
\mathbb V[f_*]
= k(x_*, x_*) - k_*^T (K + \sigma_n^2 I)^{-1} k_*.
$$

其中

$$
k_* = [k(x_*, x_1), \ldots, k(x_*, x_N)]^T.
$$

### Gaussian Process 的应用

- 小数据集回归。
- 对不确定性敏感的问题，例如校准、传感器数据建模。
- Bayesian Optimization。
- 地统计学中的 Kriging。
- 更复杂状态空间模型中的组成模块。

### 最后总结

- Gaussian Process 提供了一种非参数、贝叶斯式的回归方法。
- 它直接定义函数上的分布。
- 核函数编码了关于函数性质的先验知识。
- 它给出的不是单点预测，而是完整预测分布：均值加不确定性。
- 它的主要局限是计算复杂度通常为

$$
O(N^3).
$$

- 当数据不多、但不确定性刻画很重要时，它往往是非常合适的选择。
