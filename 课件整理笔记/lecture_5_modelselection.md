## Lecture 4: Decision Theory and Model Selection

> - 决策论中的风险最小化
> - 0-1、平方与绝对值损失
> - 偏差-方差分解与泛化误差
> - 正则化、交叉验证与集成学习
> - 贝叶斯模型选择与 Bayesian Model Averaging

### Chapter 5: Decision Theory and Model Selection

### Section 1: Decision Theory

### 机器学习中的根本问题

- 机器学习模型会给出预测，但这些预测天然带有不确定性。
- 在这种不确定性下，什么才是最好的行动？

> Decision Theory 提供了一个严格的数学框架，用损失函数和概率来回答这个问题。

### 决策问题的四个核心组成

- 一个决策论问题通常由四元组 $(Y,A,P,L)$ 给出：
  - $Y$：状态空间，表示所有可能的真实状态；
  - $A$：动作空间，表示所有可能的决策或预测；
  - $P(y\mid x)$：不确定性模型，即给定观测 $x$ 时对真实状态 $y$ 的条件概率；
  - $L(y,a)$：损失函数，表示真实状态为 $y$ 时采取动作 $a$ 的代价。

### 从损失到期望损失（Risk）

- 由于真实的 $y$ 未知，我们不能直接最小化 $L(y,a)$。
- 我们转而最小化它在可能状态上的期望，也就是风险：

$$
R(a\mid x)=\mathbb E_{y\sim P(y\mid x)}[L(y,a)].
$$

- 离散情形下是求和，连续情形下是积分。

### 最优决策规则

- 最优动作 $a^*$ 就是让条件风险最小的动作：

$$
a^* = \arg\min_{a\in A} R(a\mid x)
= \arg\min_{a\in A} \mathbb E_{y\sim P(y\mid x)}[L(y,a)].
$$

- 这个规则称为 Bayes decision rule。
- 最小可达到的风险称为 Bayes risk。

### 情形 1：0-1 损失

- 对分类问题，0-1 loss 定义为：
- 若预测正确，也就是 $y=a$，则损失为 $0$；
- 若预测错误，也就是 $y\ne a$，则损失为 $1$。

- 若动作固定为 $a=j$，则风险为

$$
R(a=j\mid x)=1-P(y=j\mid x).
$$

- 因而为了最小化风险，应该最大化后验概率：

$$
a^* = \arg\max_j P(y=j\mid x).
$$

- 这就是 MAP 规则。

### 情形 2：平方损失

- 对回归问题，平方损失是

$$
L(y,a)=(y-a)^2.
$$

- 风险为

$$
R(a\mid x)=\mathbb E[(y-a)^2\mid x].
$$

- 对 $a$ 求导并令其为 0，可得最优动作是后验均值：

$$
a^* = \mathbb E[y\mid x] = \int y p(y\mid x)\,dy.
$$

### 情形 3：绝对值损失

- 绝对值损失为

$$
L(y,a)=|y-a|.
$$

- 相应风险是

$$
R(a\mid x)=\int |y-a| p(y\mid x)\,dy.
$$

- 它的最优解是后验分布的中位数。

### 一个 principled workflow

- 决策论可以自然嵌入贝叶斯框架：
  1. Prior：先验 $P(y)$；
  2. Data：观测数据 $D$；
  3. Posterior：由 Bayes 定理得到 $P(y\mid D)$；
  4. Loss：根据任务定义损失函数 $L(y,a)$；
  5. Decision：选择最小化 posterior expected loss 的动作。

### 非对称损失与复合损失

- 现实问题中，错误代价常常不是对称的。

#### 例子：医学诊断
- 假阴性（把病人判成健康）代价极高；
- 假阳性代价较低。

- 在这种情况下，最优决策阈值不再是 0.5，而会被代价矩阵改变。

### Rejection Option

- 有时最优行动不是做出某个类别判断，而是拒绝决策。
- 引入一个 reject action $a_R$，其代价是固定常数 $\lambda_R$。

#### 规则
- 只有当某个类别的风险既最小、又小于拒绝代价时，才真正输出该类别；
- 否则就选择 reject。

- 这能避免在模型极不确定时做出武断预测。

### Section 1 小结

- 决策论的核心是：把“模型相信什么”转化为“我们该做什么”。
- 不同损失函数对应不同的最优决策：
  - 0-1 Loss 对应 mode / MAP；
  - Squared Loss 对应 mean；
  - Absolute Loss 对应 median。

---

### Section 2: Model Selection - Bias-Variance Trade-off

### 泛化才是目标

- 我们不只希望模型在训练集上表现好。
- 更希望它在新的、未见过的数据上表现好，这才是 generalization error / test error。

- 如果有很多模型可选，那么问题就是：应该选择哪个模型？

### 两条失败路径

#### Underfitting
- 模型过于简单；
- 无法捕捉数据模式；
- 训练误差就已经较高。

#### Overfitting
- 模型过于复杂；
- 把噪声也当成模式学进去；
- 训练误差低，但测试误差高。

### 真实模型与近似模型

- 设真实关系是

$$
y = f(x) + \epsilon,
$$

其中：
- $f(x)$ 是真实但未知的函数；
- $\epsilon$ 是随机噪声，满足 $\mathbb E[\epsilon]=0$，$\mathrm{Var}(\epsilon)=\sigma_\epsilon^2$。

- 我们用有限训练集学到一个估计器 $\hat f_D(x)$ 去近似 $f(x)$。

### 定义预测误差

- 对固定测试点 $x$，期望预测误差定义为：

$$
\mathrm{Error}(x)=\mathbb E_D \bigl(y - \hat f_D(x)\bigr)^2.
$$

- 这里期望是对所有可能训练集取的。

### Bias-Variance Decomposition

- 经过展开与取期望，可以得到经典分解：

$$
\mathbb E_D \bigl(y-\hat f(x)\bigr)^2
=
\bigl(f(x)-\mathbb E[\hat f(x)]\bigr)^2
+ \mathbb E_D\bigl(\hat f(x)-\mathbb E[\hat f(x)]\bigr)^2
+ \sigma_\epsilon^2.
$$

- 三项分别对应：
  - Bias$^2$；
  - Variance；
  - Irreducible Error。

#### 解释
- Bias$^2$：平均预测离真实函数有多远。
- Variance：模型对训练集变化有多敏感。
- Irreducible Error：数据本身噪声带来的不可消除误差。

### 如何管理这个 trade-off

#### 减少 Bias
- 用更复杂的模型；
- 加入更相关的特征；
- 降低正则化强度。

#### 减少 Variance
- 用更简单的模型；
- 增加训练数据；
- 做特征选择或降维；
- 增强正则化；
- 使用 ensemble methods。

### Bias-Variance 小结

- 总误差可以看作 Bias$^2$ + Variance + Irreducible Error。
- 高 Bias 对应 underfitting；
- 高 Variance 对应 overfitting。

---

### Method 1: Regularization

- 通过修改损失函数来控制模型复杂度：

$$
E(w)=E_0(w) + \lambda \sum_i w_i^2.
$$

- 这样会把参数 $w$ 压小，从而有效降低模型复杂度。
- $\lambda$ 是需要调节的超参数。

### Method 2: Cross-Validation

#### 单次划分的问题
- 如果只做一次 train-test split，结果可能很不稳定；
- 模型表现会严重依赖哪一部分样本恰好落在测试集中。

#### Cross-Validation 的想法
- 让所有数据在多次系统化划分中既参与训练，也参与验证。

### k-Fold Cross-Validation

#### 流程
1. 随机打乱数据并分成 $k$ 份。
2. 轮流把其中一份作为 validation set，其余 $k-1$ 份作为 training set。
3. 在 $k$ 次结果上取平均。

#### 常见取值
- $k=5$；
- $k=10$；
- $k=n$，也就是 Leave-One-Out。

### Cross-Validation 的类型

- k-Fold Cross-Validation；
- Leave-One-Out；
- Time Series Cross-Validation；
- Leave-P-Out；
- Repeated k-Fold。

### 应用：模型选择与超参数调优

- 对每个超参数组合，都跑一整轮 k-fold CV；
- 然后选择平均表现最好的组合。

#### 示例
- 讲义给了不同正则参数 $\lambda$ 的 5-fold CV score。
- 选择原则不只是平均分高，也要考虑方差小。

#### Nested Cross-Validation
- 外层负责评估模型性能；
- 内层负责调超参数；
- 这样能减少 optimistic bias。

### Grid Search

- Grid Search 是对超参数做系统穷举搜索。
- 它会测试预定义网格中的所有组合，并用交叉验证来比较性能。

#### 特点
- 在给定网格内保证能找到最优组合；
- 实现简单；
- 各组参数之间可以并行。

### Method 3: Ensemble

- 集成学习的核心直觉是：多个模型合起来，往往比单个模型更稳定。

> None of us is as smart as all of us.

### 集成的方差降低原理

- 设 $M$ 个模型的预测分别是 $f_1(x),\ldots,f_M(x)$，则平均预测为

$$
f_{avg}(x)=\frac1M \sum_{i=1}^M f_i(x).
$$

- 如果这些模型彼此不强相关，那么平均后的方差会显著下降。

### Bagging

- Bagging 的流程是：
  1. 构造多个 bootstrap samples；
  2. 在每个样本上训练一个模型；
  3. 回归取平均，分类取投票。

- 它特别适合高方差模型，例如 decision trees。

### Random Forest

- Random Forest = Bagging + 随机特征选择。
- 每次节点划分时，不看全部特征，只看随机抽出的一个子集。

#### 结果
- 树之间差异更大；
- 相关性更低；
- 方差进一步降低。

### 什么时候集成最有效

- 基模型需要足够 diverse；
- 预测误差最好尽量不相关；
- 单个模型至少要有一定能力；
- 集成规模足够大。

### Gradient Boosting

- 与 Bagging 不同，Boosting 是 sequential 的。
- 每个新模型都试图纠正前一个模型的错误。

#### 核心过程
1. 从一个简单预测开始；
2. 计算残差；
3. 用弱学习器拟合残差；
4. 以小步长更新模型；
5. 重复很多次。

### Gradient Boosting Algorithm

- 讲义给出了标准 boosting 的数学流程：
  - 定义损失函数；
  - 初始化模型；
  - 计算 pseudo-residuals；
  - 拟合弱学习器 $h_m(x)$；
  - 求步长 $\gamma_m$；
  - 更新

$$
F_m(x)=F_{m-1}(x)+\nu \gamma_m h_m(x).
$$

### XGBoost

- XGBoost 是高度优化的 gradient boosting 实现。
- 其优势包括：
  - L1 + L2 regularization；
  - parallel processing；
  - 自动处理缺失值；
  - pruning；
  - 训练中交叉验证。

#### 目标函数
- 讲义把目标写成 loss 加上树复杂度正则项。

#### 适用场景
- 结构化 / 表格数据；
- 大规模数据；
- 对速度和准确率都要求较高时。

---

### Section 3: Bayes Perspective

### Frequentist vs Bayesian

- Frequentist 方法（如 AIC, BIC）通常用参数个数惩罚拟合优度，然后选一个“最优模型”。
- Bayesian 方法则计算模型后验概率，使用 marginal likelihood，自带 Occam’s razor，并且可以做 model averaging。

### 模型后验概率

- 对模型 $M_i$ 应用 Bayes 定理：

$$
p(M_i\mid D)=\frac{p(D\mid M_i)p(M_i)}{p(D)}.
$$

- 其中最关键的是 marginal likelihood $p(D\mid M_i)$。

### Marginal Likelihood

- 对模型 $M_i$ 及其参数 $\theta_i$，

$$
p(D\mid M_i)=\int p(D\mid \theta_i, M_i)p(\theta_i\mid M_i)\,d\theta_i.
$$

- 它是似然在先验下的平均值。

### Automatic Occam’s Razor

- 复杂模型虽然更灵活，但其先验概率质量会摊得更开。
- 简单模型若能较好解释数据，边缘似然反而会更大。

- 这说明 marginal likelihood 会自动惩罚不必要的复杂性。

### Bayes Factor

- 比较两个模型时，看 posterior odds：

$$
\frac{p(M_1\mid D)}{p(M_2\mid D)}
=
\frac{p(D\mid M_1)}{p(D\mid M_2)}
\cdot
\frac{p(M_1)}{p(M_2)}.
$$

- 其中

$$
BF_{12}=\frac{p(D\mid M_1)}{p(D\mid M_2)}
$$

称为 Bayes factor。

- 若先验模型概率相同，则 posterior odds 就等于 Bayes factor。

### Bayes Factor 的解释

- 讲义给出了一张常见解释表，从 anecdotal evidence 到 decisive evidence。

### Occam’s Razor 的 Laplace 推导

- 用 Laplace approximation 近似 marginal likelihood 时，有

$$
p(D\mid M)
\approx
p(D\mid \hat\theta, M)
\cdot p(\hat\theta\mid M)
\cdot (2\pi)^{d/2}|H|^{-1/2}.
$$

- 可以看成：
  - goodness-of-fit；
  - prior at best-fit point；
  - Occam factor。

- 维数越高、模型越复杂，Occam factor 往往越小。

### 为什么不只选一个模型

- 如果只选一个“最佳模型”，就忽略了 model uncertainty。
- 这会导致预测过度自信，也可能降低泛化性能。

### Bayesian Model Averaging

- 对新输入 $x^*$，预测 $y^*$ 时，不只用一个模型，而是对所有模型加权平均：

$$
p(y^*\mid x^*,D)=\sum_{i=1}^K p(y^*\mid x^*, M_i, D) p(M_i\mid D).
$$

- 权重就是模型后验概率。
- BMA 往往比单模型预测更稳健、校准更好。

### 计算上的大挑战

- 问题仍然在于 marginal likelihood 中的高维积分通常难以解析求出。

#### 常见近似
- Laplace Approximation；
- Variational Inference；
- MCMC；
- 以及 Chib’s method、Bridge sampling、Nested sampling 等更专门的方法。

### 总结

- Bayesian model selection 基于 posterior model probabilities。
- 关键量是 marginal likelihood。
- 它自动体现 Occam’s razor。
- 模型比较可通过 Bayes factor 完成。
- 若不想只选单个模型，可以用 Bayesian Model Averaging。
- 实践中的最大难点是计算 marginal likelihood。
