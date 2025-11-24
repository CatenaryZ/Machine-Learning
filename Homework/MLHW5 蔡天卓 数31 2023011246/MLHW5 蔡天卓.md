1. 考虑如下概率图模型(图略)：  
   - 写出对应的因子图  
   - 使用和积算法（sum-product algorithm）计算边缘概率 \(p(x_2)\)

2. 详细推导高斯混合模型（Gaussian mixture model）的 **E 步** 与 **M 步**

3. 使用平均场方法（mean field method）近似高斯混合模型的边缘概率 \(p_\theta(X)\)。  
   变分分布为乘积形式：  
   \[
   q(Z,\pi,\mu_k,\Sigma_k) = q(Z)q(\pi)\prod_{k=1}^K q(\mu_k,\Sigma_k)
   \]

4. 编写一个简单的 **拒绝采样（rejection sampling）** Python 代码

5. 考虑一个转移矩阵为  
   \[
   P = \begin{bmatrix} p_{11} & p_{12} & p_{13} \\ p_{21} & p_{22} & p_{23} \\ p_{31} & p_{32} & p_{33} \end{bmatrix}
   \]  
   的马尔可夫链。设 \(\pi = (\pi_1, \pi_2, \pi_3)\) 是平稳分布，求 \(\pi\) 应满足的方程。

6. 为以下 MCMC 方法编写 Python 代码：  
   (a) 基本 MCMC  
   (b) 哈密顿蒙特卡洛（Hamiltonian MC）  
   (c) 朗之万动力学（Langevin dynamics）  
   用你的代码对一般的二维高斯分布进行采样，比较三种方法的结果（如接受率等）。

7. 证明：**吉布斯采样（Gibbs sampling）** 满足细致平衡条件（detailed balance condition）。

### Solution of T1

#### 因子图 

$$
p(Z_1,Z_2,Z_3,X_1,X_2,X_3)
= p(Z_1)p(Z_2\mid Z_1)p(Z_3\mid Z_2)p(X_1\mid Z_1)p(X_2\mid Z_2)p(X_3\mid Z_3).
$$

所以我们有 6 个因子（把每个括号内的项作为一个因子）：
$$
f_{Z1}=p(Z_1),
f_{Z2}=p(Z_2\mid Z_1),
f_{Z3}=p(Z_3\mid Z_2),
f_{X1}=p(X_1\mid Z_1),
f_{X2}=p(X_2\mid Z_2),
f_{X3}=p(X_3\mid Z_3).
$$

因子图如下:

```
[f_Z1]──(Z1)──[f_Z2]──(Z2)──[f_Z3]──(Z3)
          │            │              │
        [f_X1]       [f_X2]         [f_X3]
          │            │              │
         (X1)         (X2)          (X3)
```

#### 边际概率

叶节点的传播:

$$
\mu_{X_1\to f_{X1}}(x_1)=1,\qquad
\mu_{X_3\to f_{X3}}(x_3)=1.
$$

$$
\mu_{f_{X1}\to Z_1}(z_1)=\sum_{x_1}p(x_1|z_1)=1,\qquad
\mu_{f_{X3}\to Z_3}(z_3)=\sum_{x_3}p(x_3|z_3)=1.
$$

从左侧传来的部分:

$$
\mu_{f_{Z1}\to Z_1}(z_1)=p(z_1).
$$

$$
\mu_{Z_1\to f_{Z2}}(z_1)=p(z_1)\cdot 1=p(z_1).
$$

$$
\mu_{f_{Z2}\to Z_2}(z_2)
=\sum_{z_1} p(z_2|z_1)p(z_1)
= p(z_2).
$$

从右侧传来的部分:

$$
\mu_{Z_3\to f_{Z3}}(z_3)=1.
$$

$$
\mu_{f_{Z3}\to Z_2}(z_2)=\sum_{z_3} p(z_3|z_2)=1.
$$

$Z_2$的总消息:

$$
\mu_{Z_2\to f_{X2}}(z_2)
= \mu_{f_{Z2}\to Z_2}(z_2)\cdot \mu_{f_{Z3}\to Z_2}(z_2)
= p(z_2)\cdot 1
= p(z_2).
$$

从而$X_2$的边缘概率:

$$
p(x_2)
=\mu_{f_{X2}\to X_2}(x_2)
=\sum_{z_2}p(x_2|z_2)p(z_2).
$$

其中
$$
p(z_2)=\sum_{z_1}p(z_1)p(z_2|z_1).
$$

即:
$$
p(x_2)=\sum_{z_2} p(x_2\mid z_2)\left(\sum_{z_1}p(z_1)p(z_2\mid z_1)\right)
$$

---

### Solution of T2

回忆GMM模型的设定：假设我们有N个观测数据 $x_1, x_2, \dots, x_N$，假设它们独立同分布，服从K个高斯分布的混合：

$$
p(x_n \mid \Theta) = \sum_{k=1}^K \pi_k \mathcal{N}(x_n \mid \mu_k, \Sigma_k)
$$

其中：

* $\pi_k$ 是混合权重，满足 $\sum_k \pi_k = 1$ 且 $\pi_k \ge 0$
* $\mu_k$ 和 $\Sigma_k$ 分别是第k个高斯的均值和协方差矩阵
* $\Theta = {\pi_k, \mu_k, \Sigma_k}_{k=1}^K$ 是所有参数

隐变量 $z_n$表示第n个样本属于哪一类：

$$
z_n = [z_{n1}, z_{n2}, \dots, z_{nK}], \quad z_{nk} \in {0,1}, \quad \sum_{k} z_{nk} = 1
$$

则complete-data likelihood为：

$$
p(\{x_n, z_n\}_{n=1}^N \mid \Theta) = \prod_{n=1}^N \prod_{k=1}^K \big[\pi_k \mathcal{N}(x_n \mid \mu_k, \Sigma_k)\big]^{z_{nk}}
$$

对数似然：

$$
\ln p(X,Z \mid \Theta) = \sum_{n=1}^N \sum_{k=1}^K z_{nk} \Big[ \ln \pi_k + \ln \mathcal{N}(x_n \mid \mu_k, \Sigma_k) \Big]
$$

#### E-Step

在E-step，我们希望计算隐变量的后验期望

$$
\gamma_{nk} \equiv \mathbb{E}[z_{nk} \mid x_n, \Theta^{\text{old}}] = p(z_{nk}=1 \mid x_n, \Theta^{\text{old}})
$$

根据贝叶斯公式：

$$
\gamma_{nk} = p(z_{nk}=1 \mid x_n, \Theta^{\text{old}})
= \frac{ \pi_k \mathcal{N}(x_n \mid \mu_k, \Sigma_k) }{ \sum_{j=1}^K \pi_j \mathcal{N}(x_n \mid \mu_j, \Sigma_j) }
$$

#### M-Step

在M-step，我们希望最大化期望对数似然

定义Q 函数：

$$
Q(\Theta, \Theta^{\text{old}}) = \mathbb{E}_{Z \mid X, \Theta^{\text{old}}}[\ln p(X,Z \mid \Theta)]
= \sum_{n=1}^N \sum_{k=1}^K \gamma_{nk} \Big[ \ln \pi_k + \ln \mathcal{N}(x_n \mid \mu_k, \Sigma_k) \Big]
$$

以下分别对 $\pi_k$、$\mu_k$、$\Sigma_k$ 求最大化。

由于有约束条件 $\sum_k \pi_k = 1$。引入拉格朗日乘子：

$$
\mathcal{L} = \sum_{n=1}^N \sum_{k=1}^K \gamma_{nk} \ln \pi_k + \lambda \left( \sum_{k=1}^K \pi_k -1 \right)
$$

Q 对 $\pi_k$ 求导导数为 0：

$$
\frac{\partial \mathcal{L}}{\partial \pi_k} = \sum_{n=1}^N \frac{\gamma_{nk}}{\pi_k} + \lambda = 0 \quad \Rightarrow \quad \pi_k = -\frac{1}{\lambda} \sum_{n=1}^N \gamma_{nk}
$$

利用约束 $\sum_k \pi_k = 1$：

$$
\sum_k \pi_k = -\frac{1}{\lambda} \sum_k \sum_n \gamma_{nk} = -\frac{1}{\lambda} \sum_n \sum_k \gamma_{nk} = -\frac{1}{\lambda} N = 1 \quad \Rightarrow \lambda = -N
$$

所以更新公式：

$$
\pi_k^{\text{new}} = \frac{1}{N} \sum_{n=1}^N \gamma_{nk}
$$

因为：

$$
\ln \mathcal{N}(x_n \mid \mu_k, \Sigma_k) = -\frac{1}{2} (x_n - \mu_k)^T \Sigma_k^{-1} (x_n - \mu_k) + \text{const}
$$

因此 Q 对 $\mu_k$ 求导并令导数为 0：

$$
\sum_{n=1}^N \gamma_{nk} \Sigma_k^{-1} (x_n - \mu_k) = 0 \quad \Rightarrow \quad \mu_k^{\text{new}} = \frac{\sum_{n=1}^N \gamma_{nk} x_n}{\sum_{n=1}^N \gamma_{nk}}
$$

Q 对 $\Sigma_k$ 进行矩阵求导并令导数为 0：考虑到 $\Sigma_k$ 是对称正定矩阵，使用矩阵求导公式 $\frac{\partial \ln |\Sigma|}{\partial \Sigma} = (\Sigma^{-1})^T = \Sigma^{-1}$，$\frac{\partial}{\partial \Sigma} \big[ x^T \Sigma^{-1} x \big] = - \Sigma^{-1} x x^T \Sigma^{-1}$，因此求导结果如下：

$$
\frac{\partial L}{\partial \Sigma_k} = -\frac{1}{2} \sum_{n=1}^N \gamma_{nk} \left[ \Sigma_k^{-1} - \Sigma_k^{-1} (x_n - \mu_k)(x_n - \mu_k)^T \Sigma_k^{-1} \right] = 0
$$

化简后解得：

$$
\Sigma_k^{\text{new}} = \frac{\sum_{n=1}^N \gamma_{nk} (x_n - \mu_k^{\text{new}})(x_n - \mu_k^{\text{new}})^T}{\sum_{n=1}^N \gamma_{nk}}
$$

综上所述：

**E 步**：

$$
\gamma_{nk} = \frac{ \pi_k \mathcal{N}(x_n \mid \mu_k, \Sigma_k) }{ \sum_{j=1}^K \pi_j \mathcal{N}(x_n \mid \mu_j, \Sigma_j) }
$$

**M 步**：

$$
\begin{aligned}
\pi_k^{\text{new}} &= \frac{1}{N} \sum_{n=1}^N \gamma_{nk} \\
\mu_k^{\text{new}} &= \frac{\sum_{n=1}^N \gamma_{nk} x_n}{\sum_{n=1}^N \gamma_{nk}} \\
\Sigma_k^{\text{new}} &= \frac{\sum_{n=1}^N \gamma_{nk} (x_n - \mu_k^{\text{new}})(x_n - \mu_k^{\text{new}})^T}{\sum_{n=1}^N \gamma_{nk}}
\end{aligned}
$$

---

### Solution of T3