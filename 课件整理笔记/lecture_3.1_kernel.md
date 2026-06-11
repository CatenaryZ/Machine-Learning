## Lecture 4: More on Classification

> - 线性分类方法：最小二乘、感知机、SVM
> - 生成式与判别式分类方法的比较
> - 核技巧与 Mercer 定理
> - 核岭回归、核 SVM 与高斯过程
> - 对偶优化与拉格朗日方法

### 分类问题

- 目标是把输入数据 $x$ 分到 $K$ 个类别中的某一个。
- 输入是特征向量 $x \in \mathbb R^D$。
- 输出是类别标签 $y \in \{1,2,\ldots,K\}$。

#### 三类基本方法
- Discriminant functions。
- Generative models。
- Discriminative models，例如 Logistic Regression。

### 判别函数

#### 定义
- 对每个类别 $k$，定义一个函数 $f_k(x)$，并按

$$
y = \arg\max_k f_k(x)
$$

来决定类别。

#### 线性判别与非线性判别
- 线性判别：$f_k(x)=w_k^T x+b_k$。
- 非线性判别：在此基础上再加非线性变换。

- 这一类方法直接刻画 decision boundary，而不显式估计概率分布。

### 线性判别函数 I

- 基本形式是

$$
f(x)=w^T x + b.
$$

- 其中：
  - $w$：权重向量；
  - $b$：偏置；
  - $x$：输入特征。

#### 二分类规则
- 当 $f(x) \ge 0$ 时，判为正类 $+1$；
- 当 $f(x) < 0$ 时，判为负类 $-1$。

### 线性判别函数 II

#### 多分类扩展

$$
f_k(x)=w_k^T x+b_k, \qquad y=\arg\max_k f_k(x).
$$

- 类别由函数值最大的那个类别来决定。

### 方法 1：最小二乘

#### 目标函数

$$
J(w)=\sum_{i=1}^N (y_i-(w^T x_i+b))^2.
$$

- 矩阵形式可写成 $(y-Xw)^T(y-Xw)$。

#### 闭式解

$$
w=(X^T X)^{-1}X^T y.
$$

#### 特点
- 要求 $X^T X$ 可逆；
- 对异常值敏感；
- 对小数据集计算效率高。

### 方法 2：感知机损失

#### 定义
- 感知机使用如下损失：

$$
L(w)=\sum_{i\in M} -y_i(w\cdot x_i + b),
$$

其中 $M$ 是被误分类的样本集合。

#### 关键性质
- Convex。
- Piecewise linear。
- 对分类正确的样本损失为 0。
- 对误分类样本损失为正。

#### 梯度

$$
\nabla L(w)=\sum_{i\in M} -y_i x_i, \qquad
\frac{\partial L}{\partial b}=\sum_{i\in M} -y_i.
$$

#### 更新规则
- 因而会得到典型的 perceptron update：

$$
w \leftarrow w + \eta y_i x_i.
$$

### 方法 3：支持向量机

### 从几何问题到优化问题 I

- 原始几何目标是最大化间隔。
- 在可分情形下，约束为

$$
y_i(w\cdot x_i + b) \ge 1.
$$

- 最大化间隔等价于最小化 $\lVert w \rVert$。

### 从几何问题到优化问题 II

- 为了计算方便，进一步写成二次规划：

$$
\min_{w,b} \frac12 w^T w
$$

满足

$$
y_i(w^T x_i + b) \ge 1, \qquad i=1,\ldots,n.
$$

#### 特点
- 目标函数是凸的；
- 约束是线性的；
- 因而全局最优解有保证。

### 从几何问题到优化问题 III

- 实际上可以用通用 QP 求解器，也可以用专门的 SVM 库来求解。

### Soft-Margin SVM

- 真实数据通常不是完美可分，因此引入松弛变量 $\xi_i$：

$$
\min_{w,b,\xi} \frac12 \lVert w \rVert^2 + C\sum_{i=1}^n \xi_i
$$

满足

$$
y_i(w^T x_i+b) \ge 1-\xi_i, \qquad \xi_i \ge 0.
$$

- 参数 $C$ 控制大间隔与正确分类之间的权衡。

### 多分类训练策略

#### One-vs-Rest
- 训练 $K$ 个二分类器，每个分类器负责把某一类和其它所有类分开。

#### One-vs-One
- 训练 $\frac{K(K-1)}{2}$ 个二分类器，每个分类器区分一对类别。

### 生成式模型

- 生成式方法从联合分布 $p(x,y)$ 出发：

$$
p(y\mid x)=\frac{p(x\mid y)p(y)}{p(x)}.
$$

- 其中包括：
  - 类先验 $p(y)$；
  - 类条件分布 $p(x\mid y)$；
  - 后验 $p(y\mid x)$。

#### 例子
- LDA。
- QDA。
- Naive Bayes。

### 线性判别分析 LDA

#### 假设
- 每个类别的类条件分布都是高斯：

$$
p(x\mid y=k)=\mathcal N(x\mid \mu_k, \Sigma).
$$

- 各类别共享同一个协方差矩阵。

#### 判别形式

$$
\log p(y=k\mid x)
\propto
-\frac12 (x-\mu_k)^T \Sigma^{-1}(x-\mu_k)+\log \pi_k.
$$

- 由于协方差相同，decision boundary 对 $x$ 是线性的。

### 二次判别分析 QDA

#### 假设
- 各类同样服从高斯分布，但每一类有自己的协方差矩阵 $\Sigma_k$。

#### 判别形式

$$
\log p(y=k\mid x)
\propto
-\frac12 (x-\mu_k)^T \Sigma_k^{-1}(x-\mu_k)
-\frac12 \log |\Sigma_k| + \log \pi_k.
$$

- 因为协方差不同，decision boundary 对 $x$ 一般是二次型。

### 什么是 Naive Bayes

- Naive Bayes 是一种基于 Bayes 定理的概率分类器。
- 它做了一个很强的假设：给定类别之后，特征之间条件独立。

#### 为什么叫 “Naive”
- 因为现实中的特征往往并不真正独立。
- 但这种假设常常 surprisingly effective。

### Naive Bayes 的概率模型

- 对特征 $X_1,\ldots,X_d$ 与类别 $Y$，

$$
P(Y\mid X_1,\ldots,X_d)
=
\frac{P(Y)\prod_{j=1}^d P(X_j\mid Y)}{P(X_1,\ldots,X_d)}.
$$

#### 分类规则

$$
\hat y = \arg\max_y P(y)\prod_{j=1}^d P(x_j\mid y).
$$

- 由于分母对所有类别相同，比较时可以忽略。

### Gaussian Naive Bayes

- 对连续特征，假设每个特征在每个类别下服从高斯分布。
- 均值和方差可由该类别样本的经验统计量估计。

### Multinomial Naive Bayes

- 对离散计数特征，常用于文本分类。
- 使用带平滑的频率估计。

### Bernoulli Naive Bayes

- 对二元特征，建模特征“出现/不出现”。
- 常用于用词是否出现这类文档表示。

### 训练算法

#### Step 1：估计先验

$$
P(Y=y_k)=\frac{\text{class } y_k \text{ 的样本数}}{\text{总样本数}}.
$$

#### Step 2：估计似然
- Gaussian：估计各类各特征的均值与方差。
- Multinomial：估计频数。
- Bernoulli：估计出现概率。

### 预测算法

#### Step 1：计算各类概率

$$
P(y_k\mid x) \propto P(y_k)\prod_{j=1}^d P(x_j\mid y_k).
$$

#### Step 2：用 log 概率避免下溢

$$
\log P(y_k\mid x) = \log P(y_k) + \sum_{j=1}^d \log P(x_j\mid y_k).
$$

#### Step 3：作出预测

$$
\hat y = \arg\max_{y_k} \log P(y_k\mid x).
$$

### 优点与局限

#### 优点
- 训练快，预测快；
- 适合高维数据；
- 同时可处理连续与离散特征；
- 给出概率输出；
- 对无关特征相对稳健。

#### 局限
- 条件独立假设过强；
- 可能被更复杂模型超越；
- 有 zero-frequency problem，需要 smoothing；
- 不适合复杂特征交互。

### 判别式模型：Logistic Regression

- 这一类方法直接建模后验 $p(y\mid x)$，而不去建模 $p(x\mid y)$。

#### Binary case

$$
p(y=1\mid x)=\sigma(w^T x + b).
$$

#### Multiclass case

$$
p(y=k\mid x)=\frac{\exp(w_k^T x + b_k)}{\sum_j \exp(w_j^T x + b_j)}.
$$

- 它对数据分布做的假设比生成式模型更少。

### 最大似然估计

- 训练时最大化对数似然：

$$
L(w)=\sum_{i=1}^N
\left[
y_i\log p(y_i\mid x_i,w)
+ (1-y_i)\log (1-p(y_i\mid x_i,w))
\right].
$$

- 二分类下其梯度为

$$
\nabla_w L = \sum_{i=1}^N
\bigl(y_i - p(y_i=1\mid x_i,w)\bigr)x_i.
$$

#### 常见优化方法
- Gradient Descent；
- Newton-Raphson；
- SGD。

### 方法比较

- Discriminant Functions：非概率式，边界灵活。
- LDA：生成式，边界线性。
- QDA：生成式，边界二次。
- Logistic Regression：判别式，边界线性。

#### 一般经验
- 生成式方法在小数据集上往往更有优势，也更容易处理缺失数据。
- 判别式方法在大数据集上通常表现更好，因为它直接关注决策边界。

### 什么时候用什么方法

#### Discriminant Functions
- 当不需要概率解释时；
- 当计算效率很重要时；
- 当希望模型简单可解释时。

#### Generative Models
- 当数据集较小时；
- 当希望生成样本时；
- 当特征大致满足高斯假设时；
- 当需要处理缺失数据时。

#### Logistic Regression
- 当数据集较大时；
- 当需要较好校准的概率时；
- 当高斯假设不成立时；
- 当需要一个强基线模型时。

### 处理非线性

- 讲义最后转向非线性处理：
  1. 显式特征变换；
  2. 隐式表示学习。

- 接下来便进入 kernel methods 与 Gaussian Processes。
