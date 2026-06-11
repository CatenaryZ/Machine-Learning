## Lecture 4.1: Classification - Linear Decision

> - 线性判别函数与多分类规则
> - 最小二乘、感知机、支持向量机
> - 硬间隔、软间隔与 hinge loss
> - LDA、QDA、Naive Bayes、Logistic Regression

### 分类问题

- 本讲考虑的任务是：把输入数据 $x$ 分配到 $K$ 个类别中的某一个。
- 输入是特征向量 $x\in\mathbb{R}^D$，输出是类别标签 $y\in\{1,2,\dots,K\}$。
- 从方法论上看，讲义把分类方法分成三类：
  - 直接构造判别函数的做法；
  - 先建模联合分布 $p(x,y)$ 的生成式方法；
  - 直接建模后验 $p(y\mid x)$ 的判别式方法。

### 判别函数

#### 基本定义

- 对每个类别 $k$ 定义一个打分函数 $f_k(x)$。
- 预测时选择得分最大的类别：

$$
\hat y=\arg\max_k f_k(x).
$$

- 这类方法的特点是：直接针对决策边界建模，而不必先显式估计概率分布。

#### 线性判别函数

- 在线性情形下，常写成

$$
f_k(x)=w_k^T x+b_k.
$$

- 二分类时通常写成单个函数 $f(x)=w^T x+b$。
- 其中：
  - $w$ 决定超平面的方向；
  - $b$ 决定超平面的位置。

#### 二分类规则

- 若 $f(x)\ge 0$，判为正类；
- 若 $f(x)<0$，判为负类。

- 因而学习一个线性分类器，本质上就是学习一个把特征空间切开的超平面。

#### 多分类扩展

- 多分类并不是本质上不同的问题，只是需要为每个类各写一个判别函数。
- 最终仍然是比较所有类别的得分并取最大者。

### 三种典型的线性判别方法

### 方法 1：最小二乘法

#### 目标函数

- 讲义先把分类问题写成平方误差拟合问题：

$$
J(w)=\sum_{i=1}^N\bigl(y_i-(w^T x_i+b)\bigr)^2.
$$

- 如果用矩阵记号表示，也可以写成

$$
J(w)=(y-Xw)^T(y-Xw).
$$

- 这说明它实际上把“分类”先改写成了一个“回归型”优化问题。

#### 闭式解

- 在 $X^T X$ 可逆时，参数满足

$$
w=(X^T X)^{-1}X^T y.
$$

- 这是它最大的优点：形式直接，计算方便，小规模数据上尤其容易实现。

#### 优缺点

- 优点：
  - 推导简单；
  - 有闭式解；
  - 计算代价相对低。
- 缺点：
  - 平方误差并不是专门为分类设计的目标；
  - 对离群点较敏感；
  - 结果会受到标签编码方式影响。

#### 正则化版本

- 讲义还提到可以加上正则项，例如 ridge classification：

$$
J_{\lambda}(w)=\sum_{i=1}^N\bigl(y_i-(w^T x_i+b)\bigr)^2+\lambda\sum_i w_i^2.
$$

- 这等于在拟合误差之外，再额外约束参数不要过大。

### 方法 2：感知机损失

#### 损失的写法

- 感知机只对误分类样本累积损失：

$$
L(w)=\sum_{i\in M}-y_i(w\cdot x_i+b),
$$

其中 $M$ 是当前被误分类的样本集合，且 $y_i\in\{-1,+1\}$。

- 直观上说，如果某个样本已经分对了，它就不再对损失作贡献；只有分错的点才推动参数更新。

#### 梯度与更新

- 对应梯度为

$$
\nabla_w L=\sum_{i\in M}-y_i x_i,\qquad
\frac{\partial L}{\partial b}=\sum_{i\in M}-y_i.
$$

- 单样本版本的更新规则可写成

$$
w\leftarrow w+\eta y_i x_i,
$$

其中 $\eta$ 是学习率。

#### 这一方法在做什么

- 如果 $y_i=+1$ 却被分成负类，那么 $y_i x_i=x_i$，更新会把 $w$ 往 $x_i$ 的方向推。
- 如果 $y_i=-1$ 却被分成正类，那么更新方向相反。
- 所以感知机的思想非常朴素：谁分错了，就朝着能把它纠正回来的方向改。

> 注：讲义此处把感知机损失描述为一种 hinge-style 的分段线性损失；更严格地区分时，它与后面 SVM 使用的标准 hinge loss 形式并不完全相同。

### 方法 3：支持向量机

#### 几何目标

- SVM 的核心不是“只要分对就行”，而是“在分对的前提下，尽量让边界离样本更远”。
- 这个“更远”对应的就是 margin。
- 当数据线性可分时，约束写成

$$
y_i(w^T x_i+b)\ge 1.
$$

#### 硬间隔 SVM

- 原始优化问题是

$$
\min_{w,b}\frac12\lVert w\rVert^2
\quad
\text{s.t. } y_i(w^T x_i+b)\ge 1,\ \forall i.
$$

- 最小化 $\lVert w\rVert^2$ 等价于最大化间隔。
- 所以 SVM 的目标可以理解成：在所有能正确分类训练集的超平面中，选出 margin 最大的那个。

#### 为什么大间隔重要

- 间隔越大，说明决策边界离训练样本越远。
- 这通常意味着：
  - 小扰动不容易把样本推到边界另一侧；
  - 对噪声更稳健；
  - 泛化能力往往更好。

### 从硬间隔到软间隔

#### 现实问题

- 讲义强调，真实数据很少是完全线性可分的。
- 如果仍然强行要求所有点满足硬间隔约束，模型会很脆弱，甚至无解。

#### 引入松弛变量

- 为此引入松弛变量 $\xi_i\ge 0$，允许某些样本落到间隔内，甚至被误分类：

$$
\min_{w,b,\xi}\frac12\lVert w\rVert^2+C\sum_{i=1}^n \xi_i
$$

满足

$$
y_i(w^T x_i+b)\ge 1-\xi_i,\qquad \xi_i\ge 0.
$$

- 这里：
  - $\frac12\lVert w\rVert^2$ 负责让间隔大；
  - $\sum_i \xi_i$ 负责惩罚违反间隔或误分类；
  - $C>0$ 控制两者之间的折中。

#### 参数 $C$ 的作用

- $C$ 大：更强调训练样本不要分错。
- $C$ 小：更强调大间隔和整体稳健性。

### hinge loss 的出现

#### 从约束到无约束目标

- 由约束可知，最优的 $\xi_i$ 实际上会取

$$
\xi_i=\max\bigl(0,\,1-y_i(w^T x_i+b)\bigr).
$$

- 代回目标函数可得

$$
\min_{w,b}\frac12\lVert w\rVert^2
+C\sum_{i=1}^n \max\bigl(0,\,1-y_i(w^T x_i+b)\bigr).
$$

#### 标准 hinge loss

- 若记 $f(x_i)=w^T x_i+b$，则每个样本的 hinge loss 为

$$
L_{\text{hinge}}(y_i,f(x_i))=\max\bigl(0,\,1-y_i f(x_i)\bigr).
$$

- 讲义进一步写成平均损失加正则项的形式：

$$
\min_{w,b}\lambda\lVert w\rVert^2+\frac1n\sum_{i=1}^n \max\bigl(0,\,1-y_i f(x_i)\bigr),
$$

其中 $\lambda=\frac{1}{2nC}$。

- 因而 soft-margin SVM 可以看成“正则化 + hinge loss”的组合。

### 多分类训练策略

- 对多分类任务，讲义提到常见策略包括：
  - one-vs-rest：训练 $K$ 个二分类器，每个类对其余所有类；
  - one-vs-one：任意两个类别训练一个分类器。

- 这类策略本质上是把多分类问题拆成若干二分类问题来做。

### 生成式模型

#### 基本思路

- 生成式方法先建模联合分布 $p(x,y)$，再由贝叶斯公式得到后验：

$$
p(y\mid x)=\frac{p(x\mid y)p(y)}{p(x)}.
$$

- 这里：
  - $p(y)$ 是类别先验；
  - $p(x\mid y)$ 是类条件分布。

- 和前面的判别函数方法相比，它更关注“数据是如何生成出来的”。

### LDA

#### 模型假设

- LDA 假设：
  1. 每一类的类条件分布都是高斯：

$$
p(x\mid y=k)=\mathcal{N}(x\mid \mu_k,\Sigma);
$$

  2. 所有类别共享同一个协方差矩阵 $\Sigma$；
  3. 类别先验为 $p(y=k)=\pi_k$，并满足 $\sum_{k=1}^K \pi_k=1$。

- 需要估计的参数是：
  - 类先验 $\pi_k$；
  - 类均值 $\mu_k$；
  - 共享协方差 $\Sigma$。

#### MLE 结果

- 讲义逐页推导了这些参数的极大似然估计，最终得到

$$
\hat\pi_k=\frac{N_k}{N},
\qquad
\hat\mu_k=\frac{1}{N_k}\sum_{i\in C_k} x_i,
$$

以及

$$
\hat\Sigma=
\frac1N
\sum_{k=1}^K\sum_{i\in C_k}(x_i-\hat\mu_k)(x_i-\hat\mu_k)^T.
$$

- 其中 $C_k$ 表示第 $k$ 类样本集合，$N_k$ 表示该类样本个数。
- 讲义还提到常见的无偏版本会把分母写成 $N-K$。

#### 分类规则为什么是线性的

- Bayes 分类器是

$$
\hat y=\arg\max_k \hat\pi_k\cdot \mathcal{N}(x\mid \hat\mu_k,\hat\Sigma).
$$

- 由于所有类别共享协方差矩阵，二次项会相互抵消，所以判别函数可写成

$$
\delta_k(x)=x^T\hat\Sigma^{-1}\hat\mu_k
-\frac12 \hat\mu_k^T\hat\Sigma^{-1}\hat\mu_k+\log \hat\pi_k.
$$

- 这个式子对 $x$ 是线性的，因此 LDA 的决策边界是线性的。

### QDA

#### 和 LDA 的区别

- QDA 仍然假设类条件分布是高斯，但允许每个类别有自己的协方差矩阵 $\Sigma_k$：

$$
p(x\mid y=k)=\mathcal{N}(x\mid \mu_k,\Sigma_k).
$$

- 对应的后验对数打分中会出现

$$
-\frac12(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)-\frac12\log|\Sigma_k|+\log\pi_k.
$$

- 因为不同类的二次项不能消掉，所以决策边界一般是二次的。

#### 直观理解

- LDA 更强地假设“各类形状差不多，只是中心不同”；
- QDA 则允许不同类别拥有不同形状与方向，因此更灵活，但参数也更多。

### Naive Bayes

#### 朴素假设

- 朴素贝叶斯的关键假设是：在给定类别 $Y$ 的条件下，各特征彼此条件独立。
- 若特征为 $X_1,\dots,X_d$，则有

$$
P(X_1,\dots,X_d\mid Y)=\prod_{j=1}^d P(X_j\mid Y).
$$

- 因而后验可写为

$$
P(Y\mid X_1,\dots,X_d)
=
\frac{P(Y)\prod_{j=1}^d P(X_j\mid Y)}{P(X_1,\dots,X_d)}.
$$

- 做分类时，分母对所有类别都相同，所以只需比较

$$
\hat y=\arg\max_y P(y)\prod_{j=1}^d P(x_j\mid y).
$$

#### 三种常见版本

##### Gaussian Naive Bayes

- 适用于连续特征。
- 假设每个特征在每个类别条件下都服从一维高斯分布：

$$
P(X_j\mid Y=y_k)
=
\frac{1}{\sqrt{2\pi\sigma_{jk}^2}}
\exp\!\left(
-\frac{(x_j-\mu_{jk})^2}{2\sigma_{jk}^2}
\right).
$$

- 参数估计就是分别求各类中每个特征的均值和方差。

##### Multinomial Naive Bayes

- 适用于计数型离散特征，文本分类里很常见。
- 讲义给出的形式是

$$
P(X_j\mid Y=y_k)=
\frac{\operatorname{count}(X_j,Y=y_k)+\alpha}
{\sum_{l=1}^d \operatorname{count}(X_l,Y=y_k)+\alpha d}.
$$

- 其中 $\alpha$ 是平滑参数，$\alpha=1$ 时就是 Laplace smoothing。

##### Bernoulli Naive Bayes

- 适用于二值特征，例如某个词是否出现。
- 概率模型写成

$$
P(X_j\mid Y=y_k)=P(j\mid y_k)^{x_j}\bigl(1-P(j\mid y_k)\bigr)^{1-x_j}.
$$

#### 这一方法的特点

- 优点：
  - 简单；
  - 训练快；
  - 小样本下往往也能工作；
  - 对高维稀疏数据比较友好。
- 局限：
  - 条件独立假设通常过强；
  - 当特征之间强相关时，模型可能偏差较大。

### 判别式模型：Logistic Regression

#### 基本思想

- 与 LDA / QDA / Naive Bayes 不同，逻辑回归不先建模 $p(x\mid y)$，而是直接建模后验概率 $p(y\mid x)$。

#### 二分类

- 二分类时，

$$
p(y=1\mid x)=\sigma(w^T x+b),
\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

- 所以线性函数 $w^T x+b$ 先给出一个实数打分，再通过 sigmoid 映射成概率。

#### 多分类

- 多分类时使用 softmax：

$$
p(y=k\mid x)=
\frac{\exp(w_k^T x+b_k)}
{\sum_{j=1}^K \exp(w_j^T x+b_j)}.
$$

#### 最大似然训练

- 二分类下，训练目标通常写成对数似然

$$
L(w)=
\sum_{i=1}^N
\Bigl[
y_i\log p(y_i\mid x_i,w)
+(1-y_i)\log\bigl(1-p(y_i\mid x_i,w)\bigr)
\Bigr].
$$

- 对应梯度为

$$
\nabla_w L=
\sum_{i=1}^N
\bigl(y_i-p(y_i=1\mid x_i,w)\bigr)x_i.
$$

- 这说明逻辑回归是在直接推动预测概率靠近真实标签。

#### 与生成式模型的差异

- 逻辑回归对输入分布的假设更弱。
- 因此当高斯假设并不成立时，它常常比 LDA / QDA 更稳健。
- 同时，它还能给出较自然的概率输出。

### 方法比较

#### 按建模方式分类

- Discriminant functions：直接拟合决策边界，通常不显式给概率。
- LDA / QDA / Naive Bayes：生成式方法，先建模分布再做分类。
- Logistic Regression：判别式方法，直接建模后验概率。

#### 边界形状与假设

- LDA：线性边界，依赖“高斯 + 共享协方差”假设。
- QDA：二次边界，允许不同类有不同协方差。
- Logistic Regression：线性边界，但不要求高斯生成假设。
- SVM：强调最大间隔，不直接建模概率。

#### 什么时候用

- 讲义最后给出了一些经验判断：
  - 若不需要概率解释，且追求简单快速，可用判别函数类方法；
  - 若数据较小、想利用分布假设，LDA / QDA 往往有优势；
  - 若数据更多、关注分类表现和概率校准，逻辑回归常是很好的基线；
  - 若希望边界鲁棒、泛化强，SVM 往往是重要选择。

### 本讲复习时可以抓住的主线

- 第一条主线是：线性分类器到底怎么构造。
- 这一部分从最小二乘、感知机一路走到 SVM，体现的是三种不同的优化视角：
  - 拟合数值误差；
  - 修正误分类；
  - 最大化间隔。
- 第二条主线是：分类时要不要显式建模概率分布。
- 这就把方法分成了生成式与判别式两大类，并由此引出 LDA、QDA、Naive Bayes 和 Logistic Regression。

### 下一步

- 本讲最后一页把内容引向 non-parametric methods，例如 K-nearest neighbor、decision tree 和 random forest。
