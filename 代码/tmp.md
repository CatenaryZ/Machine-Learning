### 类内散度和类间散度的计算公式

在线性判别分析（LDA）中，类内散度矩阵（Within-class scatter matrix）和类间散度矩阵（Between-class scatter matrix）是核心概念。以下是它们的详细计算公式：

#### 1. 类内散度矩阵 \( S_W \)

类内散度矩阵衡量**同一类别内**样本的分散程度，计算公式为：

\[
S_W = \sum_{i=1}^{c} \sum_{x \in D_i} (x - \mu_i)(x - \mu_i)^T
\]
\[
S_B = \sum_{i=1}^{c} n_i (\mu_i - \mu)(\mu_i - \mu)^T
\]

其中：
- \( c \) 是类别总数
- \( S_i \) 是第 \( i \) 类的散度矩阵，计算公式为：

\[
S_i = \sum_{x \in D_i} (x - \mu_i)(x - \mu_i)^T
\]

这里：
- \( D_i \) 是属于第 \( i \) 类的样本集合
- \( \mu_i \) 是第 \( i \) 类样本的均值向量，计算公式为：

\[
\mu_i = \frac{1}{n_i} \sum_{x \in D_i} x
\]

- \( n_i \) 是第 \( i \) 类的样本数量

**物理意义**：\( S_W \) 越小，表示同一类别内的样本越紧凑，分类效果越好。

#### 2. 类间散度矩阵 \( S_B \)

类间散度矩阵衡量**不同类别之间**的分离程度，计算公式为：

\[
S_B = \sum_{i=1}^{c} n_i (\mu_i - \mu)(\mu_i - \mu)^T
\]

其中：
- \( \mu \) 是所有样本的总体均值向量，计算公式为：

\[
\mu = \frac{1}{n} \sum_{i=1}^{n} x_i = \frac{1}{n} \sum_{i=1}^{c} n_i \mu_i
\]

- \( n \) 是总样本数（\( n = \sum_{i=1}^{c} n_i \)）

**物理意义**：\( S_B \) 越大，表示不同类别的中心点相距越远，分类效果越好。

#### 3. 总体散度矩阵 \( S_T \)

还有一个相关的概念是总体散度矩阵，它衡量所有样本的总体分散程度：

\[
S_T = \sum_{i=1}^{n} (x_i - \mu)(x_i - \mu)^T
\]

这三个矩阵之间存在重要关系：

\[
S_T = S_W + S_B
\]

#### 应用示例

在LDA中，我们希望找到投影方向 \( w \) 使得类间散度与类内散度的比值最大化：

\[
J(w) = \frac{w^T S_B w}{w^T S_W w}
\]

通过求解广义特征值问题 \( S_B w = \lambda S_W w \)，我们可以找到最优的投影方向。

**总结**：
- \( S_W \)：衡量类别内部的紧凑程度（越小越好）
- \( S_B \)：衡量类别之间的分离程度（越大越好）
- LDA目标是最大化 \( S_B \) 与 \( S_W \) 的比值

我来使用平均场方法推导高斯混合模型边缘概率的近似。

## 1. 模型设定

高斯混合模型的联合分布为：
\[
p_\theta(X,Z,\pi,\mu,\Sigma) = p(Z|\pi)p(X|Z,\mu,\Sigma)p(\pi)p(\mu,\Sigma)
\]

其中：
- \( Z \)：离散隐变量（分配指示变量）
- \( \pi \)：混合权重
- \( \mu_k, \Sigma_k \)：第k个高斯分量的参数

## 2. 平均场变分分布

根据题意，变分分布为：
\[
q(Z,\pi,\mu_k,\Sigma_k) = q(Z)q(\pi)\prod_{k=1}^K q(\mu_k,\Sigma_k)
\]

## 3. 边缘概率的变分下界

边缘概率的对数为：
\[
\log p_\theta(X) = \log \int p_\theta(X,Z,\pi,\mu,\Sigma) dZd\pi d\mu d\Sigma
\]

使用Jensen不等式，得到变分下界(ELBO)：
\[
\log p_\theta(X) \geq \mathbb{E}_q\left[\log\frac{p_\theta(X,Z,\pi,\mu,\Sigma)}{q(Z,\pi,\mu,\Sigma)}\right] = \mathcal{L}(q)
\]

## 4. ELBO的具体形式

将变分分布代入：
\[
\mathcal{L}(q) = \mathbb{E}_q[\log p(X|Z,\mu,\Sigma)] + \mathbb{E}_q[\log p(Z|\pi)] + \mathbb{E}_q[\log p(\pi)] + \mathbb{E}_q[\log p(\mu,\Sigma)] - \mathbb{E}_q[\log q(Z)] - \mathbb{E}_q[\log q(\pi)] - \sum_{k=1}^K \mathbb{E}_q[\log q(\mu_k,\Sigma_k)]
\]

## 5. 最优变分分布的更新公式

通过固定其他变分因子，分别优化每个因子：

### 5.1 更新 \( q(Z) \)
\[
\log q^*(Z) = \mathbb{E}_{q(\pi)q(\mu,\Sigma)}[\log p(X,Z,\pi,\mu,\Sigma)] + \text{const}
\]
\[
q^*(Z) = \prod_{n=1}^N \prod_{k=1}^K r_{nk}^{z_{nk}}
\]
其中：
\[
r_{nk} \propto \exp\left(\mathbb{E}[\log \pi_k] - \frac{1}{2}\mathbb{E}[(\mathbf{x}_n-\mu_k)^T\Sigma_k^{-1}(\mathbf{x}_n-\mu_k)] - \frac{1}{2}\mathbb{E}[\log|\Sigma_k|]\right)
\]

### 5.2 更新 \( q(\pi) \)
\[
\log q^*(\pi) = \mathbb{E}_{q(Z)}[\log p(Z|\pi)] + \log p(\pi) + \text{const}
\]
如果先验 \( p(\pi) = \text{Dir}(\pi|\alpha) \)，则：
\[
q^*(\pi) = \text{Dir}(\pi|\alpha^*), \quad \alpha_k^* = \alpha_k + \sum_{n=1}^N \mathbb{E}[z_{nk}]
\]

### 5.3 更新 \( q(\mu_k,\Sigma_k) \)
\[
\log q^*(\mu_k,\Sigma_k) = \mathbb{E}_{q(Z)}[\log p(X|Z,\mu_k,\Sigma_k)] + \log p(\mu_k,\Sigma_k) + \text{const}
\]

对于高斯-逆Wishart先验，\( q(\mu_k,\Sigma_k) \) 也是高斯-逆Wishart分布。

## 6. 边缘概率的近似

通过变分推断，我们得到：
\[
p_\theta(X) \approx \exp(\mathcal{L}(q))
\]

其中 \( \mathcal{L}(q) \) 是收敛后的变分下界值。

## 7. 算法流程

1. 初始化变分分布参数
2. 交替更新：
   - \( q(Z) \)（E步）
   - \( q(\pi), q(\mu_k,\Sigma_k) \)（M步）
3. 计算ELBO \( \mathcal{L}(q) \)
4. 重复直到收敛
5. 用最终的 \( \mathcal{L}(q) \) 近似 \( \log p_\theta(X) \)

这种方法将复杂的边缘概率计算转化为相对简单的变分推断问题，通过坐标上升法可以高效求解。