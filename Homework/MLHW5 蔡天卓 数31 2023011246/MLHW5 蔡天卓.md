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