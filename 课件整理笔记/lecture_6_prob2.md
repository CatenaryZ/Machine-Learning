## Lecture 5: Probability II

> - 从 i.i.d. 模型走向图模型
> - 贝叶斯网络、d-separation 与 MRF
> - K-means、soft K-means 与选簇数
> - GMM、潜变量表示与 EM 算法

### Chapter 6: Graphical Models and Mixture Models

### Section 1: Graphical Probability Model

### 为什么需要图模型

- 之前课程里常用的概率模型大多建立在 i.i.d. 假设上，也就是各观测之间彼此独立。
- 这类模型通常也带有明显的监督学习结构，把数据写成 $(X,y)$，目标是从特征预测标签。
- 但现实中很多问题并不满足这种独立性假设，例如：
  - 自然语言；
  - 音频信号；
  - 序列数据；
  - 更一般的结构化数据。

- 在这些场景里，不同变量之间往往存在显式依赖关系，仅靠 i.i.d. 模型难以表达。
- 图模型的价值就在于：它不仅告诉我们“变量之间有关联”，还告诉我们“这种关联是怎样组织起来的”。

### Beyond Supervised Learning

- 讲义随后提醒我们：很多时候数据里甚至没有标签，只有特征向量 $X$。
- 这时学习目标常常变成：
  - 发现数据内部结构，例如 clustering；
  - 学习数据分布本身，例如 generative modeling。

- 所以后半讲的聚类和 mixture model，其实正是在这个背景下出现的。

### 什么是概率图

#### 有向图的含义

- 在概率论里，有向图用来表达随机变量之间的条件依赖方向。
- 图中的：
  - 节点表示随机变量；
  - 有向边表示条件依赖。

- 它给出的并不是单个概率值，而是一个联合分布的结构化分解方式。

#### 基本术语

- 讲义先介绍了图模型里常见的语言：
  - nodes / vertices：节点；
  - edges / arcs：边；
  - parents：父节点；
  - children：子节点；
  - ancestors / descendants：祖先与后代。

- 这些术语后面会直接决定联合分布如何分解。

### Bayesian Networks

#### 定义

- 贝叶斯网络是有向无环图（DAG）上的概率模型。
- 它把联合分布分解为

$$
P(X_1,\dots,X_n)=\prod_{i=1}^n P(X_i\mid \operatorname{parents}(X_i)).
$$

- 一旦图结构给定，这个分解结构就固定了。

#### 这一分解的意义

- 它把本来可能非常复杂的高维联合分布拆成若干局部条件分布。
- 因而图结构一方面能帮助建模，另一方面也能帮助后续推断。

### 例子 1：Markov Chain

- 马尔可夫链是最简单的有向图模型之一。
- 图结构是

$$
X_1\to X_2\to X_3\to \cdots \to X_T.
$$

- 它表达的马尔可夫性质是

$$
P(X_t\mid X_{t-1},X_{t-2},\dots,X_1)=P(X_t\mid X_{t-1}).
$$

- 因而联合分布可以分解为

$$
P(x_1,\dots,x_T)=P(x_1)\prod_{t=2}^T P(x_t\mid x_{t-1}).
$$

- 这句话的核心含义是：给定现在，未来与更久远的过去条件独立。

### 例子 2：Hidden Markov Model

- HMM 在链式结构上引入了潜变量与观测变量。
- 讲义中的结构是：潜状态 $Z_1,\dots,Z_T$ 构成一条链，而每个 $Z_t$ 再生成一个观测 $X_t$。
- 联合分布因此分解为

$$
P(z_1)P(x_1\mid z_1)\prod_{t=2}^T P(z_t\mid z_{t-1})P(x_t\mid z_t).
$$

- 这类模型非常适合语言、序列标注等具有时间结构的任务。

### 图如何编码独立性

- 讲义特别强调，图模型最大的优点之一，不只是“能画图”，而是“可以从图上读出条件独立关系”。
- 这使得我们不需要每次都去手动操作联合分布公式。

### 三种基本结构

#### 1. Chain

$$
A\to B\to C
$$

- 这是头接尾的链式结构。
- 直觉上，$A$ 会通过 $B$ 影响 $C$。

#### 2. Fork

$$
A\leftarrow B\to C
$$

- 这是共同原因结构。
- $B$ 同时影响 $A$ 和 $C$，因此在不观测 $B$ 时，$A$ 与 $C$ 往往相关。

#### 3. Collider

$$
A\to B\leftarrow C
$$

- 这是共同结果结构，也叫 v-structure。
- 它最反直觉，因为在不观测 $B$ 时，$A$ 与 $C$ 反而是独立的。

### 路径何时被阻断

- 讲义把条件独立的判断组织成“路径是否被 blocked”的语言。
- 关键规则是：
  - 若路径上的中间节点属于 chain 或 fork，并且这个节点被条件化，则该路径被阻断；
  - 若中间节点是 collider，并且该节点及其后代都没有被观测，则该路径被阻断。

- 复习时最容易出错的地方就是 collider：
  - 不观测 collider 时，它会挡住路径；
  - 一旦观测了 collider 或其后代，原本关闭的路径反而会打开。

### d-Separation

#### 算法步骤

- 讲义把判断 $(X\perp\!\!\!\perp Y\mid Z)$ 的方法整理成 d-separation algorithm：
  1. 找出 $X$ 到 $Y$ 的所有无向路径；
  2. 逐条检查这些路径是否被 $Z$ 阻断；
  3. 如果所有路径都被阻断，则 $X$ 与 $Y$ 在给定 $Z$ 下条件独立；
  4. 只要有一条路径未被阻断，就不独立。

#### 三个例子

- Simple chain：
  - 不给定中间节点时，$A$ 与 $C$ 一般不独立；
  - 给定 $B$ 后，$A\perp\!\!\!\perp C\mid B$。

- Common cause：
  - 不给定共同原因 $B$ 时，$A$ 与 $C$ 一般相关；
  - 给定 $B$ 后，路径被阻断。

- Collider：
  - 不给定碰撞点时，$A\perp\!\!\!\perp C$；
  - 给定碰撞点或其后代后，$A$ 与 $C$ 反而会相关。

### Markov Random Fields

#### 为什么还需要无向图

- 有些依赖关系并不适合用“方向”来描述。
- 这时更自然的是无向图模型，也就是 Markov Random Field，简称 MRF。

#### 分解形式

- 讲义把无向图的联合分布写成

$$
P(X)=\frac1Z\prod_{C\in\mathcal{C}} \psi_C(X_C),
$$

其中：
- $\mathcal{C}$ 表示最大 clique 的集合；
- $\psi_C(X_C)$ 是 clique 上的 potential function；
- $Z$ 是 partition function，用来保证分布归一化。

#### potential 的写法

- 讲义还给出一种常见形式：

$$
\psi_C(X_C)=\exp\bigl(-E(X_C)\bigr),
$$

其中 $E(X_C)$ 可以理解为局部能量。

#### 三种 Markov 性质

- Local Markov property：

$$
X_i \perp X_{V\setminus N(i)} \mid X_{N(i)}.
$$

- Pairwise Markov property：
  - 如果 $i,j$ 之间没有边，那么在给定其余所有变量后，$X_i$ 与 $X_j$ 条件独立。

- Global Markov property：
  - 图上的分离结构对应条件独立关系。

### Ising Model

#### 图表示

- Ising model 是无向图模型的经典例子。
- 节点对应自旋变量 $\sigma_i\in\{-1,+1\}$。
- 边表示相邻粒子之间的相互作用。

#### 能量函数

- 讲义写出的哈密顿量为

$$
H(\sigma)=
-\sum_{(i,j)\in E} J_{ij}\sigma_i\sigma_j
-\mu\sum_i h_i\sigma_i.
$$

- 第一项描述邻居间相互作用；
- 第二项描述外场作用。

#### 概率分布

- 对应的 Gibbs / Boltzmann 分布为

$$
P(\sigma)=\frac1Z \exp\bigl(-\beta H(\sigma)\bigr).
$$

- 这里 $\beta$ 是逆温度，$Z$ 是配分函数。

- 这个例子很重要，因为它把“图结构”“能量函数”“概率分布”三者直接连在了一起。

### 从有向图到无向图

- 讲义还简要提到 moralization，把某些 directed graphical models 转成 undirected graphical models。
- 这一步的意义是：有时为了推断方便，我们愿意把方向信息转成 clique 结构来处理。

### 图模型中的推断

- 本讲最后先做了一个预告：
  - exact inference，例如 variable elimination、belief propagation；
  - approximate inference，例如 sampling、variational inference。

- 这些内容会在后面的 inference 讲义中继续展开。

---

### Section 2: Gaussian Mixture Model

### K-means 聚类

#### 问题设定

- 给定数据集 $X=\{x_1,\dots,x_n\}$ 和簇数 $K$，目标是把数据划分到 $K$ 个簇里。
- 讲义强调的目标是：
  - 同一簇里的点彼此相近；
  - 不同簇里的点彼此较远。

#### 目标函数

- K-means 最小化的是簇内平方和：

$$
J=\sum_{i=1}^K\sum_{x\in C_i}\lVert x-\mu_i\rVert^2,
$$

其中 $\mu_i$ 是第 $i$ 个簇的中心。

- 所以它做的事情可以概括成一句话：让每个样本尽量靠近自己所属簇的中心。

#### 基本算法

- 讲义给出的过程是交替优化：
  1. 随机初始化 $K$ 个中心 $\mu_1,\dots,\mu_K$；
  2. Assignment step：每个点分到最近的中心，

$$
c_j=\arg\min_i \lVert x_j-\mu_i\rVert^2;
$$

  3. Update step：对每个簇重新计算中心，

$$
\mu_i=\frac1{|C_i|}\sum_{x_j\in C_i} x_j;
$$

  4. 重复直到中心基本不再变化。

#### 为什么它可行

- Assignment step 在固定中心时，使目标函数尽可能小；
- Update step 在固定簇分配时，使每个簇内平方误差最小的中心恰好就是样本均值。

- 所以 K-means 虽然不是全局最优算法，但每一步都在降低目标函数。

### 初始化问题

- K-means 对初值敏感。
- 随机初始化可能导致：
  - 落入较差局部最优；
  - 不同次运行结果差异较大。

### k-means++

- 为了得到更好的初始中心，讲义介绍了 k-means++。
- 它的核心思想是：
  - 第一个中心随机选；
  - 后续中心优先从“离现有中心较远”的点中选出。

- 这样能减少一开始多个中心扎堆的问题，通常会得到更稳定的结果。

### 如何选择 $K$

#### Elbow Method

- 把训练后得到的簇内平方和 $J(K)$ 画成随 $K$ 变化的曲线。
- 寻找那个“继续增大 $K$ 已经没有太大改善”的拐点。
- 这种方法直观，但带有一定主观性。

#### Silhouette Score

- 讲义还给出了更定量的指标：

$$
s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}}.
$$

- 其中：
  - $a(i)$ 是样本 $i$ 到同簇其他点的平均距离；
  - $b(i)$ 是样本 $i$ 到最近其他簇的平均距离。

- 解释为：
  - $s(i)\approx 1$：聚得很好；
  - $s(i)\approx 0$：位于边界；
  - $s(i)\approx -1$：可能分错了。

- 实际上可以比较不同 $K$ 的平均 silhouette score，选择分数较高者。

### Hard K-means 的局限

- 每个样本只能属于一个簇；
- 对离群点敏感；
- 对重叠簇处理较差；
- 归属关系是硬分配的，不反映不确定性。

- 这正是讲义引出 soft K-means 的原因。

### Soft K-means

#### 核心思想

- soft K-means 允许一个样本同时对多个簇具有不同程度的归属。
- 这比硬聚类更符合许多现实数据的模糊边界。

#### 隶属度

- 讲义给出的 membership function 是

$$
w_{ij}
=
\frac{1}
{\sum_{k=1}^K\left(\frac{d_{ij}}{d_{ik}}\right)^{\frac{2}{m-1}}},
$$

其中：
- $w_{ij}$ 是样本 $i$ 对簇 $j$ 的隶属度；
- $d_{ij}$ 是样本 $i$ 到簇 $j$ 中心的距离；
- $m>1$ 是 fuzziness parameter。

#### 中心更新

- 簇中心用加权平均更新：

$$
c_j=
\frac{\sum_{i=1}^N w_{ij}^m x_i}
{\sum_{i=1}^N w_{ij}^m}.
$$

- 这里用的是“软分配权重”，而不是像 K-means 那样只看是否被分配到该簇。

#### 算法步骤

1. 初始化簇中心。
2. 计算所有点对所有簇的 membership weights。
3. 用加权平均更新中心。
4. 反复迭代，直到

$$
\lVert C^{(t)}-C^{(t-1)}\rVert<\varepsilon.
$$

#### 模糊参数 $m$

- $m=1$ 时退化到 hard K-means。
- $m\to\infty$ 时，各簇隶属度趋向平均。
- 讲义给出的常用范围是 $1.5\le m\le 3.0$。

#### 优点

- 更适合重叠簇；
- 对噪声和离群点更稳健；
- 能表达归属不确定性；
- 从思想上也更接近概率模型。

### Hard vs Soft K-means

- hard K-means：
  - 每个点只属于一个簇；
  - 计算更简单；
  - 但边界生硬。

- soft K-means：
  - 每个点对多个簇有连续权重；
  - 更灵活；
  - 但计算稍复杂。

### Gaussian Mixture Models

#### 模型定义

- GMM 用 $K$ 个高斯分布的加权和描述数据分布：

$$
P(x)=\sum_{k=1}^K \pi_k\,\mathcal{N}(x\mid \mu_k,\Sigma_k),
$$

其中：
- $\pi_k\ge 0$，且 $\sum_{k=1}^K \pi_k=1$；
- $\mu_k$ 是第 $k$ 个分量的均值；
- $\Sigma_k$ 是第 $k$ 个分量的协方差。

#### 与 K-means 的区别

- K-means 给的是“你属于哪个簇”；
- GMM 给的是“你属于每个簇的概率有多大”。

- 因此 GMM 是一个真正的概率模型，而不只是几何划分算法。

### 参数估计为什么困难

- 如果知道每个样本来自哪个高斯分量，那么参数估计会容易很多。
- 但现实里这个“分量标签”不可观测，所以直接最大化观测数据似然并不方便。

### 潜变量表示

- 为了描述“样本来自哪个 component”，讲义引入 one-hot 潜变量 $z$：

$$
z=[z_1,z_2,\dots,z_K]^T,\qquad z_k\in\{0,1\},\qquad \sum_k z_k=1.
$$

- 其中 $P(z_k=1)=\pi_k$，且在给定 $z_k=1$ 时，

$$
P(x\mid z_k=1)=\mathcal{N}(x\mid \mu_k,\Sigma_k).
$$

#### 完整数据联合分布

- 于是联合分布可以写成

$$
P(x,z)=P(z)P(x\mid z)
=
\prod_{k=1}^K \pi_k^{z_k}\mathcal{N}(x\mid \mu_k,\Sigma_k)^{z_k}.
$$

- 这个写法对后面的 EM 推导非常关键。

### GMM 的 EM 算法

#### E-step：责任度

- 对每个样本 $x_n$ 和每个分量 $k$，计算责任度

$$
\gamma(z_{nk})
=
\frac{\pi_k\mathcal{N}(x_n\mid \mu_k,\Sigma_k)}
{\sum_{j=1}^K \pi_j\mathcal{N}(x_n\mid \mu_j,\Sigma_j)}.
$$

- 它表示：在当前参数下，样本 $n$ 来自第 $k$ 个分量的后验概率。

#### M-step：参数更新

- 先记

$$
N_k=\sum_{n=1}^N \gamma(z_{nk}).
$$

- 然后更新

$$
\mu_k^{\text{new}}
=
\frac1{N_k}\sum_{n=1}^N \gamma(z_{nk})x_n,
$$

$$
\Sigma_k^{\text{new}}
=
\frac1{N_k}
\sum_{n=1}^N
\gamma(z_{nk})(x_n-\mu_k^{\text{new}})(x_n-\mu_k^{\text{new}})^T,
$$

$$
\pi_k^{\text{new}}=\frac{N_k}{N}.
$$

- 直观上看：
  - E-step 先做软分配；
  - M-step 再用这些软分配后的权重更新参数。

### 从 GMM 到 K-means

- 讲义最后说明，K-means 可以看成 GMM 的一个极限情形。
- 若施加以下约束：
  1. 所有协方差矩阵都相同且球形，即 $\Sigma_k=\varepsilon I$；
  2. 混合系数相等，即 $\pi_k=1/K$；
  3. 再令 $\varepsilon\to 0$，

- 那么：
  - 各高斯分量会变得极窄；
  - 每个样本几乎只对最近分量有非零后验；
  - GMM 的软分配就退化成 K-means 的硬分配。

- 因而 K-means 可以理解为 GMM 的一个特殊极限版本。

### 本讲复习时可以抓住的主线

- 前半部分的主线是：如何用图来表达随机变量之间的结构化依赖。
- 这一部分从 DAG、Bayesian network、d-separation 讲到 MRF 和 Ising model，重点是“结构如何编码独立性与分解”。
- 后半部分的主线是：如何从简单聚类走向概率聚类。
- 这里的逻辑链条是：
  - hard K-means 只做硬分配；
  - soft K-means 引入模糊归属；
  - GMM 则把这种归属彻底概率化，并通过潜变量和 EM 完成学习。
