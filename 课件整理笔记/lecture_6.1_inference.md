## Lecture 6: Inference Part 1

> - 层次聚类、树状图与 DBSCAN
> - collider 上的 explain away 现象
> - 链式图上的精确推断与前后向消息
> - factor graph、sum-product、MAP 与 max-sum

### 聚类方法概览

- 讲义一开始先回顾了两类已经学过的聚类方法：
  - K-means：围绕若干中心点做硬分配；
  - Gaussian Mixture Models：围绕若干高斯中心做软分配。

- 它们的共同点是：都带有明显的 centroid-based 假设，也就是默认簇大致围绕某些“中心”组织。
- 本讲随后引入另外两类思路：
  - 层次聚类：不直接假设单个中心，而是构造簇之间的树状层级；
  - DBSCAN：不用中心描述簇，而用局部密度来定义簇。

### Hierarchical Clustering

#### 核心想法

- 层次聚类不是一次性给出最终簇划分，而是构造一整套由粗到细的层级结构。
- 高层的簇包含低层的簇，因此最后得到的是一棵“簇的树”。

#### 两种基本路线

1. Agglomerative，也就是 bottom-up：
   - 一开始每个样本各自成簇；
   - 再不断合并最接近的两个簇。
2. Divisive，也就是 top-down：
   - 一开始把所有点看成一个大簇；
   - 然后递归地往下拆分。

- 讲义的重点放在 agglomerative hierarchical clustering 上。

### Agglomerative 算法

#### 基本步骤

1. 初始时有 $n$ 个簇，每个点一个簇。
2. 计算簇之间的 proximity matrix。
3. 重复执行：
   - 找到最近的两个簇 $C_i,C_j$；
   - 将它们合并成一个新簇；
   - 更新距离矩阵。
4. 直到所有样本都合并成一个簇。

- 这个过程记录下来的合并历史，就是后面 dendrogram 的来源。

### Linkage Methods

- 层次聚类里的关键问题是：“两个簇之间的距离”到底怎么定义。
- 讲义列出了几种 linkage 方法。

#### Single linkage

- 定义为两簇中最近两点之间的距离：

$$
d(C_i,C_j)=\min_{a\in C_i,\ b\in C_j} d(a,b).
$$

- 它容易形成“链式串联”的簇。

#### Complete linkage

- 定义为两簇中最远两点之间的距离：

$$
d(C_i,C_j)=\max_{a\in C_i,\ b\in C_j} d(a,b).
$$

- 这样得到的簇往往更紧凑。

#### Average linkage

- 定义为跨簇点对距离的平均值。
- 直觉上，它介于 single 与 complete 之间，没那么容易出现细长链，也不像 complete 那样过于保守。

### 用树状图选择簇数

- 层次聚类通常和 dendrogram 一起使用。
- dendrogram 记录的是“哪些簇在什么距离尺度下被合并了”。
- 讲义给出的经验规则是：
  - 在树状图中找最长的竖直间隔；
  - 在那里水平切一刀；
  - 被切出来的连通分支数就可以作为簇数候选。

- 这个方法本质上是在找“明显的合并跳跃点”。

### DBSCAN

#### 核心思想

- DBSCAN 不再围绕“中心点”定义簇，而是围绕“密度连通”来定义簇。
- 所以它特别适合：
  - 簇形状不规则；
  - 需要显式识别噪声点；
  - 簇不一定是球形或凸形。

#### 基本概念

- 给定半径参数 $\varepsilon$ 和阈值 $\text{MinPts}$。
- 对任一点 $p$，其 $\varepsilon$-邻域为

$$
N_\varepsilon(p)=\{q\in D\mid \operatorname{dist}(p,q)\le \varepsilon\}.
$$

- 然后定义：
  - 核心点 core point：若 $|N_\varepsilon(p)|\ge \text{MinPts}$；
  - 边界点 border point：自身邻域点数不足，但落在某个核心点的邻域内；
  - 噪声点 noise point：既不是核心点，也不是边界点。

#### 可达性

- 直接密度可达：
  - 若 $p$ 是核心点，且 $q\in N_\varepsilon(p)$，则 $q$ 从 $p$ 直接密度可达。

- 密度可达：
  - 若存在一条点链 $p_1,\dots,p_n$，其中相邻点逐步直接密度可达，则终点对起点密度可达。

- 密度连接：
  - 若存在某个点 $o$，使得 $p$ 和 $q$ 都从 $o$ 密度可达，则称 $p$ 和 $q$ 密度连接。

- 簇正是由这类密度连接关系组织出来的。

### DBSCAN 的算法步骤

1. 先把所有点标记成 core / border / noise 三类。
2. 对每个尚未分配的核心点：
   - 新建一个簇；
   - 把与它密度连接的点不断扩展进来。
3. 对未分簇的边界点：
   - 分配给最近的核心簇。
4. 噪声点保持不分簇，作为 outliers。

- 这使 DBSCAN 与 K-means 很不一样：它不要求事先指定簇数，而是让密度结构自己决定有几个簇。

### DBSCAN 与层次聚类比较

#### DBSCAN 的优势

- 能找到任意形状的簇；
- 能显式识别噪声点；
- 不需要预先指定簇数。

#### 层次聚类的优势

- 给出完整的层级结构；
- 可以在多个分辨率下观察数据。

#### 二者差别的一个简洁理解

- 层次聚类更像“看合并历史”；
- DBSCAN 更像“看局部密度是否足够支撑成簇”。

---

### Chapter 7

### Explain Away 现象

### 三节点例子

- 讲义用一个非常经典的 Bayesian network 来说明 explain away：

$$
\text{Rain}\to \text{Wet Grass}\leftarrow \text{Sprinkler}.
$$

- 这里：
  - Rain 和 Sprinkler 是两个原因；
  - Wet Grass 是共同结果。

- 这正是 collider，也就是 v-structure。

### 模型设定

- 先验为：

$$
P(R=T)=0.2,\qquad P(S=T)=0.1.
$$

- 而且在没有观察草地是否湿的时候，

$$
R\perp\!\!\!\perp S.
$$

- 条件概率表给出了 $P(W\mid R,S)$，其中典型值包括：
  - $P(W=T\mid R=F,S=F)=0$；
  - $P(W=T\mid R=F,S=T)=0.9$；
  - $P(W=T\mid R=T,S=F)=0.8$；
  - $P(W=T\mid R=T,S=T)=0.98$。

### 先算证据概率

- 观测到草湿的总概率是

$$
P(W=T)=\sum_{r,s} P(W=T\mid r,s)P(r)P(s).
$$

- 讲义一步步算出

$$
P(W=T)=0+0.072+0.144+0.0196=0.2356.
$$

### 再看单变量后验

- 有了草湿这个证据后，

$$
P(R\mid W)=\frac{P(W\mid R)P(R)}{P(W)}
\approx \frac{0.8\times 0.2}{0.2356}\approx 0.679,
$$

$$
P(S\mid W)=\frac{P(W\mid S)P(S)}{P(W)}
\approx \frac{0.9\times 0.1}{0.2356}\approx 0.382.
$$

- 和原先的 $0.2$ 与 $0.1$ 相比，两者后验都明显升高了。
- 这很自然，因为“草湿”支持“下雨了”或“洒水了”。

### explain away 真正发生在哪里

- 关键不是单变量后验，而是联合后验

$$
P(R=T,S=T\mid W=T)
=
\frac{P(W=T\mid R=T,S=T)P(R=T)P(S=T)}{P(W=T)}.
$$

- 讲义给出的数值是

$$
\frac{0.98\times 0.2\times 0.1}{0.2356}\approx 0.083.
$$

- 如果雨和洒水在观测 $W$ 后仍然独立，那么应当有

$$
P(R,S\mid W)=P(R\mid W)P(S\mid W)\approx 0.679\times 0.382\approx 0.259.
$$

- 但真实值只有 $0.083$，远小于 $0.259$。

### 这说明了什么

- 一旦知道草湿了，雨和洒水就不再独立。
- 更准确地说，它们会呈现负相关：
  - 如果已经得知“下雨”很可能是真的；
  - 那么“还需要洒水来解释草湿”这件事就没那么必要了。

- 这就是 explain away 的含义：一个原因一旦被确认，会削弱另一个原因的必要性。

### 一般化表述

- 对一般的 collider 结构 $A\to C\leftarrow B$：
  - 观测前，$A$ 与 $B$ 可以独立；
  - 观测 $C$ 之后，$A$ 与 $B$ 通常变得相关。

- 这也是为什么图模型中的推断算法必须认真处理 v-structure。

---

### Section 1: Exact Inference

### 链式图上的推断目标

- 现在考虑链式 Bayesian network：

$$
X_1\to X_2\to X_3\to \cdots \to X_N.
$$

- 联合分布分解为

$$
P(X)=P(X_1)\prod_{i=2}^N P(X_i\mid X_{i-1}).
$$

- 推断目标是求边缘分布，例如

$$
P(X_i)=\sum_{X\setminus X_i} P(X).
$$

### 为什么直接算很难

- 如果每个变量有 $K$ 个可能状态，暴力求和需要枚举所有状态组合，复杂度会随 $N$ 指数增长。

> 注：讲义对应页的复杂度排版看起来像 `O(KN)`，但结合“暴力枚举全部状态组合”的上下文，这里更可能意指随链长指数增长的复杂度。

- 真正的突破点在于：联合分布已经因子化，我们不必一次性把所有变量都一起求和。

### 利用分解结构

- 讲义强调的关键观察是：可以交换求和顺序，并把大问题拆成一连串局部消元。
- 例如在链上，先把最后一个变量求和掉，再往前推进，就能避免重复计算。

- 这一思想最后被统一写成 factor graph + message passing 的框架。

### Factor Graph

#### 链式结构的因子表示

- 对链式模型，可把联合分布写成若干因子的乘积，例如
  - $f_1(X_1)=P(X_1)$；
  - $f_2(X_1,X_2)=P(X_2\mid X_1)$；
  - $f_3(X_2,X_3)=P(X_3\mid X_2)$；
  - 依此类推。

- factor graph 中有两类节点：
  - 变量节点 variable nodes；
  - 因子节点 factor nodes。

- 它把“谁是变量、谁是局部函数”分开表示，因此非常适合写消息传递规则。

### 前向消息与后向消息

### Forward pass：$\alpha$

- 讲义把前向消息记成 $\alpha$。
- 例如

$$
\alpha_2(X_2)=\sum_{x_1} f_1(x_1)f_2(x_1,X_2),
$$

$$
\alpha_3(X_3)=\sum_{x_2}\alpha_2(x_2)f_3(x_2,X_3).
$$

- 一般形式为

$$
\alpha_i(X_i)=\sum_{x_{i-1}}\alpha_{i-1}(x_{i-1})f_i(x_{i-1},X_i).
$$

- 它的含义是：把目标节点左边所有信息压缩成一个关于当前节点的函数。

### Backward pass：$\beta$

- 后向消息记成 $\beta$。
- 初始化通常取

$$
\beta_N(X_N)=1.
$$

- 然后向左递推：

$$
\beta_i(X_i)=\sum_{x_{i+1}} P(x_{i+1}\mid X_i)\beta_{i+1}(x_{i+1}).
$$

- 它表示从右边剩余链条传回来的汇总信息。

### 如何得到边缘分布

- 一旦有了左右两边的消息，就可以在中间会合：

$$
P(X_i)\propto \alpha_i(X_i)\beta_i(X_i).
$$

- 归一化后得到

$$
P(X_i)=
\frac{\alpha_i(X_i)\beta_i(X_i)}
{\sum_{x_i}\alpha_i(x_i)\beta_i(x_i)}.
$$

- 这就是前向后向算法的核心。

### 完整 two-pass algorithm

#### Step 1：Forward pass

$$
\alpha_1(X_1)=P(X_1),
$$

$$
\alpha_i(X_i)=\sum_{x_{i-1}}\alpha_{i-1}(x_{i-1})P(X_i\mid x_{i-1}),\quad i=2,\dots,N.
$$

#### Step 2：Backward pass

$$
\beta_N(X_N)=1,
$$

$$
\beta_i(X_i)=\sum_{x_{i+1}}P(x_{i+1}\mid X_i)\beta_{i+1}(x_{i+1}),\quad i=N-1,\dots,1.
$$

#### Step 3：Compute marginals

$$
P(X_i)=
\frac{\alpha_i(X_i)\beta_i(X_i)}
{\sum_{x_i}\alpha_i(x_i)\beta_i(x_i)}.
$$

### 复杂度优势

- 讲义强调：message passing 把原来全局、指数级的求和，拆成了局部、可复用的消息。
- 对链式结构，这会把推断从暴力枚举大幅降到多项式复杂度。

### 一般树图上的精确推断

- 链只是树图的一个特殊例子。
- 更一般的树结构上，也可以照样做 message passing。

### 一般 factor graph 中的消息规则

#### 变量节点到因子节点

- 若变量节点 $X$ 要向因子节点 $f$ 发送消息，则

$$
\mu_{X\to f}(X)=\prod_{h\in \operatorname{ne}(X)\setminus\{f\}} \mu_{h\to X}(X).
$$

- 也就是说，变量节点只是把除目标因子以外的所有入射消息相乘后发出去。

#### 因子节点到变量节点

- 若因子节点 $f$ 要向变量节点 $X$ 发送消息，则

$$
\mu_{f\to X}(X)=
\sum_Y
\left(
f(Y,X)\prod_{Y\in \operatorname{ne}(f)\setminus\{X\}}\mu_{Y\to f}(Y)
\right).
$$

- 这里的意思是：
  - 先把本因子和其他变量发来的消息相乘；
  - 再对除 $X$ 外的变量求和；
  - 得到一个只依赖 $X$ 的函数。

#### 叶节点初始化

- 讲义还给了叶节点的简化规则：
  - 变量叶子到因子的消息可以取 $1$；
  - 因子叶子到变量的消息就是该单变量因子本身。

### 树上的通用算法步骤

1. 任选一个节点当 root。
2. 从叶子向根做一次收集消息的 pass。
3. 再从根向叶子做一次分发消息的 pass。
4. 对每个变量节点，把所有流入它的因子消息相乘并归一化，得到边缘分布：

$$
P(X_i)=\frac1Z\prod_{f\in \operatorname{ne}(X_i)} \mu_{f\to X_i}(X_i).
$$

### 一个简单例子

- 若

$$
P(X_1,X_2,X_3)=f_A(X_1)f_B(X_1,X_2)f_C(X_2,X_3),
$$

- 那么要求 $P(X_2)$ 时，左右两侧都各自汇总成一个消息：

$$
\mu_{f_B\to X_2}=\sum_{x_1} f_B(x_1,X_2)f_A(x_1),
$$

$$
\mu_{f_C\to X_2}=\sum_{x_3} f_C(X_2,x_3).
$$

- 最终

$$
P(X_2)\propto \mu_{f_B\to X_2}\mu_{f_C\to X_2}.
$$

- 这正好展示了“左右两边的信息各压成一个函数，再在目标点汇合”的思想。

### MAP Inference

### 从边缘概率到最优赋值

- 到目前为止，我们算的是 marginal probabilities。
- 另一类重要问题是 MAP inference：

$$
x^*=\arg\max_x P(x).
$$

- 这时目标不再是“每个变量单独有多大概率”，而是“整组变量什么配置最可能”。

### Sum-product 到 Max-product

- 讲义指出，算法框架几乎不用改，只要把因子消息中的求和改成最大化即可：

$$
\mu_{f\to X}(X)=
\max_Y
\left(
f(Y,X)\prod_{Y\in \operatorname{ne}(f)\setminus\{X\}}\mu_{Y\to f}(Y)
\right).
$$

- 因此 sum-product 与 max-product 的差别非常集中：
  - 求边缘时，对其余变量求和；
  - 求 MAP 时，对其余变量取最大值。

### Max-product / Max-sum 的流程

1. 仍然选择 root。
2. 仍然做两遍消息传递。
3. 在 root 处得到 max-marginal。
4. 再通过 backtracking 把每个节点的最优取值依次找回来。

- 所以 MAP 问题比边缘问题多出来的关键步骤，是回溯。

### 为什么要进对数域

- 乘很多概率时容易数值下溢。
- 进入对数域后：
  - 乘法变成加法；
  - 最大值仍然保持最大值。

- 于是 max-product 常写成 max-sum。

### Sum-product 与 Max-product 的比较

- Sum-product：
  - 输出各节点的边缘分布；
  - 需要归一化；
  - 常用于 Bayesian inference 与估计问题。

- Max-product / max-sum：
  - 输出最优整体赋值；
  - 需要 backtracking；
  - 常用于 decoding、segmentation、optimal control 等任务。

### 本讲复习时可以抓住的主线

- 第一段内容是“聚类不一定非要围着中心点转”，因此讲了层次聚类和 DBSCAN。
- 第二段内容用 explain away 解释了 collider 为什么特殊，也说明了图模型中的条件依赖会随着观测而改变。
- 第三段内容是整讲的核心：如何利用因子分解，把原本巨大的全局推断拆成局部消息传递。
- 只要记住这句话，本讲的大部分公式都会变得容易理解：
  - $\alpha$ 和 $\beta$ 是链上的局部汇总；
  - sum-product 是求边缘；
  - max-product / max-sum 是求最优配置。
