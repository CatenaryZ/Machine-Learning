
## 第 1 页

第 7 讲：深度神经网络的训练

谢丹
清华大学数学系
2025 年 11 月 12 日

## 第 2 页

第 7 章：深度网络的训练技术

## 第 3 页

深度神经网络训练 I

深度网络的能力与复杂性

深度神经网络已经在各种机器学习应用中取得了显著成功，能够对高度复杂的概率关系进行建模。然而，深度网络的训练也带来了很大的挑战。本讲将从三个关键方面处理这些问题：

1. 高级优化方法

   * 克服非凸损失函数的复杂地形
   * 自适应学习率技术
   * 基于动量的优化方法

2. 梯度消失与梯度爆炸

   * 归一化技术，如 BatchNorm、LayerNorm
   * 残差连接
   * 谨慎的初始化策略

3. 缓解过拟合

   * 正则化方法，如 Dropout、权重衰减

## 第 4 页

深度神经网络训练 II

* 早停与模型选择

4. 网络结构

   * 卷积神经网络 CNN：图像
   * 循环神经网络 RNN：序列数据
   * Transformer：自然语言
   * 图神经网络：具有图结构的数据

## 第 5 页

第 1 节：优化方法

## 第 6 页

深度学习中的优化挑战

基本随机梯度下降 SGD：

$$
\theta_{t+1}=\theta_t-\eta\nabla J(\theta_t)
$$

其中 $J(\theta)$ 是损失函数。

基本 SGD 的问题：

* 所有参数使用固定学习率；
* 不记忆过去的梯度；
* 对特征缩放敏感；
* 在狭长谷地形中收敛缓慢；
* 在高曲率方向上容易振荡。

## 第 7 页

基于梯度的优化方法的发展

* 20 世纪 60 年代：动量方法；
* 20 世纪 80 年代：Nesterov 加速；
* 2010 年代：自适应方法，如 AdaGrad、RMSProp、Adam；
* 2010 年代以后：高级变体，如 AdamW、Lookahead 等。

## 第 8 页

动量 SGD

核心思想：
在梯度持续下降的方向上累积速度。

算法：

$$
v_t=\gamma v_{t-1}+\eta\nabla J(\theta_t)
$$

$$
\theta_{t+1}=\theta_t-v_t
$$

参数：

* $\eta$：学习率；
* $\gamma$：动量因子，典型值为 $0.9$。

## 第 9 页

动量的物理类比

小球沿山坡滚下：

* 梯度：山坡的斜率；
* 动量：小球的速度；
* 学习率：时间步长；
* 动量因子：摩擦力。

好处：

* 更新更加平滑；
* 收敛更快；
* 有助于逃离较浅的局部极小值。

## 第 10 页

Nesterov 加速梯度 NAG

相比普通动量方法的关键改进：
先“向前看”到未来位置，再计算梯度。

算法：

$$
v_t=\gamma v_{t-1}+\eta\nabla J(\theta_t-\gamma v_{t-1})
$$

$$
\theta_{t+1}=\theta_t-v_t
$$

直观理解：

* 在近似的未来位置计算梯度；
* 对损失地形的变化反应更灵敏；
* 具有更好的收敛保证。

## 第 11 页

AdaGrad：自适应梯度方法

核心思想：
根据历史梯度大小，为每个参数自适应调整学习率。

$$
G_t=G_{t-1}+(\nabla J(\theta_t))^2
$$

$$
\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{G_t+\epsilon}}\nabla J(\theta_t)
$$

优点：

* 自动调整学习率；
* 非常适合稀疏数据；
* 不需要手动设计学习率调度。

局限：

* 学习率下降过快；
* 不适合非凸问题；
* 记忆量随时间增长。

## 第 12 页

RMSProp：均方根传播

修正 AdaGrad 过度衰减的问题：
使用梯度平方的指数加权移动平均。

$$
E[g^2]*t=\gamma E[g^2]*{t-1}+(1-\gamma)g_t^2
$$

$$
\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{E[g^2]_t+\epsilon}}g_t
$$

* $\gamma$：衰减率，通常取 $0.9$；
* 防止学习率过快减小；
* 适用于非平稳问题；
* 在线学习中效果较好。

## 第 13 页

自适应学习率的可视化

* SGD：学习率恒定；
* AdaGrad：学习率快速下降，之后变得非常小；
* RMSProp：稳定地自适应调整；
* Adam：结合动量与自适应调整。

## 第 14 页

Adam：自适应矩估计

两类方法的结合：
Adam 结合了动量方法和自适应学习率。

关键组成：

* 一阶矩估计，即梯度均值，用于动量；
* 二阶矩估计，即梯度方差，用于自适应学习率；
* 针对初始化偏差的偏差修正。

## 第 15 页

Adam 算法细节

完整 Adam 算法：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

更新一阶矩。

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

更新二阶矩。

$$
\hat m_t=\frac{m_t}{1-\beta_1^t}
$$

偏差修正。

$$
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

偏差修正。

$$
\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{\hat v_t}+\epsilon}\hat m_t
$$

参数更新。

默认参数：

* $\beta_1=0.9$：动量衰减；
* $\beta_2=0.999$：梯度平方衰减；
* $\epsilon=10^{-8}$：保证数值稳定性；
* $\eta=0.001$：学习率。

## 第 16 页

Adam 的偏差修正

为什么需要偏差修正？

初始时矩估计会偏向于 0，尤其是在训练早期。

没有修正时：

$$
E[m_t]=E[g_t](1-\beta_1^t)
$$

当 $t$ 很小时，该值近似为 0。

修正后：

$$
E[\hat m_t]=E[g_t]
$$

## 第 17 页

AdamW：带权重衰减的 Adam

修正 Adam 中权重衰减的问题。

原始 Adam 对 L2 正则化的实现方式并不正确。

原始 Adam：

$$
\theta_t=\theta_{t-1}-\eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda\theta_{t-1}\right)
$$

AdamW：

$$
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

* 正确地将权重衰减与 Adam 更新分离；
* 泛化性能更好；
* 现在已经是大多数深度学习框架中的标准方法。

## 第 18 页

实用技术：梯度裁剪

防止梯度爆炸。
这在 RNN 和 Transformer 中尤其重要。

按值裁剪：

$$
g_t=\max(\min(g_t,\text{clip value}),-\text{clip value})
$$

按范数裁剪：

$$
g_t=
\begin{cases}
g_t, & |g_t|\leq \text{clip norm},\
g_t\cdot \dfrac{\text{clip norm}}{|g_t|}, & \text{otherwise}.
\end{cases}
$$

## 第 19 页

学习率调度

阶梯衰减：

* 每隔 $N$ 个 epoch 按一定比例降低学习率；
* 简单且有效。

指数衰减：

* 连续下降；
* $\eta_t=\eta_0e^{-kt}$。

余弦退火：

* 平滑的周期性重启；
* 更有助于找到较好的极小值。

## 第 20 页

Warmup 策略

逐渐增大学习率。
这种方法可以防止自适应优化方法在训练早期不稳定。

线性 Warmup：

$$
\eta_t=\eta_{\mathrm{final}}\times \frac{t}{T_{\mathrm{warmup}}}
$$

余弦 Warmup：

$$
\eta_t=\eta_{\mathrm{final}}\times \frac{1}{2}\left(1+\cos\left(\pi\left(1-\frac{t}{T_{\mathrm{warmup}}}\right)\right)\right)
$$

## 第 21 页

优化器性能比较

| 优化器            | 收敛速度 | 稳定性 | 内存 | 超参数 | 泛化性能 | 使用场景            |
| -------------- | ---- | --- | -- | --- | ---- | --------------- |
| SGD            | 慢    | 高   | 低  | 敏感  | 好    | 计算机视觉、大 batch   |
| SGD + Momentum | 快    | 高   | 低  | 中等  | 好    | 通用场景            |
| Adam           | 非常快  | 中等  | 中等 | 鲁棒  | 中等   | 默认选择            |
| AdamW          | 非常快  | 高   | 中等 | 鲁棒  | 好    | Transformer、NLP |
| RMSProp        | 快    | 中等  | 中等 | 中等  | 中等   | RNN             |

表：优化器特性比较。

## 第 22 页

什么时候使用哪种优化器？

计算机视觉：

* 带动量的 SGD；
* 泛化性能好；
* 适合大 batch。

自然语言处理：

* AdamW；
* Transformer 更偏好 Adam 变体；
* 适合稀疏梯度。

强化学习：

* Adam/RMSProp；
* 在线学习中稳定；
* 能够处理非平稳性。

建议：

1. 从 AdamW 开始；
2. 在计算机视觉任务中尝试 SGD + Momentum；
3. 使用 Lookahead 提高稳定性。

## 第 23 页

实践建议

超参数调节技巧：

* 学习率：使用 learning rate finder，即学习率范围测试；
* Batch size：更大的 batch 允许使用更高的学习率；
* 权重衰减：AdamW 可取 $0.01$ 到 $0.1$，SGD 可取 $0.0001$ 到 $0.001$；
* Warmup：对大模型和自适应方法非常重要；
* 梯度裁剪：对 RNN/Transformer 使用范数裁剪，常见范围为 $1.0$ 到 $5.0$。

常见陷阱：

* 使用 Adam 但没有正确处理权重衰减。

## 第 24 页

第 2 节：处理梯度消失与梯度爆炸问题

## 第 25 页

处理深度网络中的梯度挑战

深度学习中的梯度问题：

深度神经网络经常受到梯度消失或梯度爆炸的影响，这会阻碍训练稳定性和收敛。这里介绍几种有效解决方案：

1. 初始化

2. 批归一化 Batch Normalization

   * 对 mini-batch 中的激活值进行归一化；
   * 减少内部协变量偏移；
   * 允许使用更高学习率。

3. 层归一化 Layer Normalization

   * 对每个样本内部的特征进行归一化；
   * 与 batch size 无关；
   * 适合循环网络和小 batch。

4. 残差连接

   * 通过恒等映射提供跳跃连接；
   * 使梯度可以通过捷径路径传播；
   * 有助于训练非常深的网络。

## 第 26 页

为什么权重初始化很重要 I

问题：

糟糕的初始化会导致：

* 梯度消失：信号在深层网络中逐渐消失；
* 梯度爆炸：信号不受控制地增长；
* 收敛缓慢：训练耗时过长；
* 训练失败：网络完全学不到东西。

好的初始化可以带来：

* 稳定的前向和反向信号流；
* 更快的收敛；
* 更好的最终性能；
* 不同训练运行之间更一致。

关键原则：

## 第 27 页

为什么权重初始化很重要 II

在所有层之间，保持激活值和梯度的方差一致。

## 第 28 页

常见初始化方法 I

Xavier/Glorot 初始化

适用于：Tanh、Sigmoid 激活函数。

尺度：

$$
\text{Scale}=\sqrt{\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}}
$$

* 保持激活值方差；
* 适合平滑激活函数。

He 初始化：

## 第 29 页

常见初始化方法 II

适用于：ReLU、Leaky ReLU 激活函数。

尺度：

$$
\text{Scale}=\sqrt{\frac{2}{n_{\mathrm{in}}}}
$$

* 考虑了 ReLU 的“死亡区”；
* 是现代网络中的默认选择。

## 第 30 页

Xavier 初始化：问题设定

前向传播：

对第 $l$ 层，假设使用线性激活：

$$
z^l=W^la^{l-1}+b^l
$$

假设：

* $W_i^l$ 独立同分布，且 $E[W^l]=0$；
* $a_i^{l-1}$ 独立同分布，且 $E[a^{l-1}]=0$；
* $W^l$ 与 $a^{l-1}$ 相互独立。

于是：

$$
\mathrm{Var}(z^l)=n_{\mathrm{in}}\cdot \mathrm{Var}(W^l)\cdot \mathrm{Var}(a^{l-1})
$$

反向传播：

对梯度有：

$$
\frac{\partial L}{\partial a^{l-1}}=(W^l)^T\frac{\partial L}{\partial z^l}
$$

$$
\mathrm{Var}\left(\frac{\partial L}{\partial a^{l-1}}\right)=n_{\mathrm{out}}\cdot \mathrm{Var}(W^l)\cdot \mathrm{Var}\left(\frac{\partial L}{\partial z^l}\right)
$$

## 第 31 页

Xavier 初始化：推导

方差保持目标：

我们希望保持稳定的信号传播：

$$
\mathrm{Var}(z^l)=\mathrm{Var}(z^{l-1})
$$

前向传播中保持稳定。

$$
\mathrm{Var}\left(\frac{\partial L}{\partial a^l}\right)=\mathrm{Var}\left(\frac{\partial L}{\partial a^{l-1}}\right)
$$

反向传播中保持稳定。

两个约束：

由前向传播得到：

$$
n_{\mathrm{in}}\cdot \mathrm{Var}(W^l)=1
$$

由反向传播得到：

$$
n_{\mathrm{out}}\cdot \mathrm{Var}(W^l)=1
$$

但这两个条件无法同时满足。

Xavier 的折中方案：

取两个约束的平均：

$$
\mathrm{Var}(W^l)=\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}
$$

## 第 32 页

Xavier 初始化：实现

对于均匀分布：

若 $W\sim U[-a,a]$，则 $\mathrm{Var}(W)=\frac{a^2}{3}$。

令 $\frac{a^2}{3}=\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}$，得到：

$$
a=\frac{\sqrt{6}}{\sqrt{n_{\mathrm{in}}+n_{\mathrm{out}}}}
$$

因此：

$$
W\sim U\left[-\frac{\sqrt{6}}{\sqrt{n_{\mathrm{in}}+n_{\mathrm{out}}}},\frac{\sqrt{6}}{\sqrt{n_{\mathrm{in}}+n_{\mathrm{out}}}}\right]
$$

对于正态分布：

若 $W\sim N(0,\sigma^2)$，则 $\mathrm{Var}(W)=\sigma^2$。

令 $\sigma^2=\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}$，得到：

$$
W\sim N\left(0,\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}\right)
$$

最适合：

Tanh 和 Sigmoid 激活函数，因为它们是平滑的，并且在 0 附近近似线性。

## 第 33 页

He 初始化：ReLU 的挑战

ReLU 对方差的影响：

对于 ReLU 激活函数 $a=\max(0,z)$，如果 $z$ 关于 0 对称分布，则：

$$
E[a^2]=E[\max(0,z)^2]=\frac{1}{2}E[z^2]
$$

因此：

$$
\mathrm{Var}(a)=\frac{1}{2}\mathrm{Var}(z)
$$

对前向传播的影响：

由前面公式：

$$
\mathrm{Var}(z^l)=n_{\mathrm{in}}\cdot \mathrm{Var}(W^l)\cdot \mathrm{Var}(a^{l-1})
$$

对于 ReLU：

$$
\mathrm{Var}(a^{l-1})=\frac{1}{2}\mathrm{Var}(z^{l-1})
$$

所以：

$$
\mathrm{Var}(z^l)=n_{\mathrm{in}}\cdot \mathrm{Var}(W^l)\cdot \frac{1}{2}\mathrm{Var}(z^{l-1})
$$

ReLU 实际上会在每一层将方差减半。

## 第 34 页

He 初始化：推导

ReLU 下的方差保持：

我们希望：

$$
\mathrm{Var}(z^l)=\mathrm{Var}(z^{l-1})
$$

由上一页可得：

$$
\mathrm{Var}(z^l)=\frac{1}{2}n_{\mathrm{in}}\cdot \mathrm{Var}(W^l)\cdot \mathrm{Var}(z^{l-1})
$$

令 $\mathrm{Var}(z^l)=\mathrm{Var}(z^{l-1})$，得到：

$$
1=\frac{1}{2}n_{\mathrm{in}}\cdot \mathrm{Var}(W^l)
$$

因此：

$$
\mathrm{Var}(W^l)=\frac{2}{n_{\mathrm{in}}}
$$

反向传播方面的考虑：

He 初始化主要是为了解决 ReLU 在前向传播中造成的方差减半问题。在实践中，它对前向传播和反向传播都表现良好。

## 第 35 页

He 初始化：实现

对于均匀分布：

令 $\frac{a^2}{3}=\frac{2}{n_{\mathrm{in}}}$，得到：

$$
a=\frac{\sqrt{6}}{\sqrt{n_{\mathrm{in}}}}
$$

因此：

$$
W\sim U\left[-\frac{\sqrt{6}}{\sqrt{n_{\mathrm{in}}}},\frac{\sqrt{6}}{\sqrt{n_{\mathrm{in}}}}\right]
$$

对于正态分布：

令 $\sigma^2=\frac{2}{n_{\mathrm{in}}}$，得到：

$$
W\sim N\left(0,\frac{2}{n_{\mathrm{in}}}\right)
$$

变体：

* 只考虑前向传播：分母使用 $n_{\mathrm{in}}$；
* 只考虑反向传播：分母使用 $n_{\mathrm{out}}$；
* 默认选择：使用 $n_{\mathrm{in}}$，实践效果较好。

最适合：

ReLU 及其变体，如 Leaky ReLU、PReLU 等。

## 第 36 页

比较与总结

关键公式：

| 方法     | 均匀分布                                                                                                                               | 正态分布                                                         |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Xavier | $U\left[-\frac{\sqrt{6}}{\sqrt{n_{\mathrm{in}}+n_{\mathrm{out}}}},\frac{\sqrt{6}}{\sqrt{n_{\mathrm{in}}+n_{\mathrm{out}}}}\right]$ | $N\left(0,\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}\right)$ |
| He     | $U\left[-\frac{\sqrt{6}}{\sqrt{n_{\mathrm{in}}}},\frac{\sqrt{6}}{\sqrt{n_{\mathrm{in}}}}\right]$                                   | $N\left(0,\frac{2}{n_{\mathrm{in}}}\right)$                  |

理论基础：

* Xavier：为接近线性的激活函数平衡前向和反向方差；
* He：补偿 ReLU 在前向传播中造成的方差下降；
* 两者都假设权重分布以 0 为中心并且对称。

实践建议：

* Tanh/Sigmoid：使用 Xavier 初始化；
* ReLU 系列：使用 He 初始化；
* 现代默认：He 初始化，因为多数网络使用 ReLU。

## 第 37 页

归一化

* 内部协变量偏移：训练过程中输入分布发生变化；
* 训练挑战：

  * 梯度消失或梯度爆炸；
  * 收敛缓慢；
  * 对初始化敏感。

归一化的好处：

* 训练收敛更快；
* 可以使用更高学习率；
* 泛化性能更好；
* 降低对初始化的敏感性。

## 第 38 页

批归一化 BN

输入为 $B\times X$，其中 $B$ 是 batch size。

核心思想：

对每个特征，在 batch 维度上归一化激活值。

$$
\hat x_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

$$
y_i=\gamma \hat x_i+\beta
$$

* $\mu_B$：batch 均值；
* $\sigma_B^2$：batch 方差；
* $\gamma,\beta$：可学习参数。

## 第 39 页

批归一化：性质

优点：

* 收敛更快；
* 可以使用更高学习率；
* 减少过拟合；
* 梯度更稳定。

局限：

* 依赖 batch size；
* 对 RNN 来说可能有问题；
* 训练阶段和测试阶段的行为不同。

## 第 40 页

层归一化 LN

核心思想：

对每个样本内部的特征维度进行归一化。

$$
\hat x_i=\frac{x_i-\mu_L}{\sqrt{\sigma_L^2+\epsilon}}
$$

* 对单个样本的所有特征进行归一化；
* 与 batch size 无关；
* 适合循环神经网络。

其中 $x_i$ 是单层神经元上的取值。

## 第 41 页

深度学习悖论

* 直觉：更深的网络应该表现更好；
* 现实：非常深的网络往往表现更差；
* 观察：训练误差会随着深度增加而上升。

## 第 42 页

梯度消失问题

反向传播中的链式法则：

$$
\frac{\partial L}{\partial W^{(1)}}=\frac{\partial L}{\partial a^{(L)}}\prod_{k=2}^L\frac{\partial a^{(k)}}{\partial a^{(k-1)}}\frac{\partial a^{(1)}}{\partial W^{(1)}}
$$

问题：

* 许多小导数被连续相乘；
* 梯度会以指数速度趋近于 0；
* 早期层的权重无法有效更新。

## 第 43 页

根本性洞察

关键问题：

学习恒等映射更容易，还是学习零映射更容易？

* 传统方法：直接学习 $H(x)$；
* 残差方法：学习 $F(x)=H(x)-x$；
* 如果最优映射是恒等映射，则 $F(x)=0$；
* 将残差推向 0 比直接学习恒等映射更容易。

$$
H(x)=F(x)+x
$$

## 第 44 页

残差块示意图

图中结构表示：

输入 $x$ 经过卷积层形成残差分支 $F(x)$，同时输入 $x$ 通过一条恒等映射的捷径连接直接传到加法节点，最后输出：

$$
H(x)=F(x)+x
$$

图：基本残差块结构。

## 第 45 页

第 3 节：处理过拟合问题

## 第 46 页

问题：过拟合

* 模型把训练数据学得太好，甚至包括噪声；
* 在新的、未见过的数据，即测试集上表现较差；
* 复杂模型，比如大型神经网络，特别容易过拟合。

## 第 47 页

方法 1：正则化

目标：防止过拟合，提高泛化能力。

奥卡姆剃刀原则：

“在相互竞争的假设中，应选择假设最少的那个。”

在机器学习中：倾向于选择更简单的模型。

* 如何定义“简单”的模型？
* 一种理解是：权重 $w$ 的大小更小；
* 较大的权重会使模型对输入过于敏感。

## 第 48 页

数学形式：修改损失函数

原始损失函数：

$$
L(w)
$$

它衡量模型对数据的拟合程度，例如 MSE 或交叉熵。

正则化后的损失函数：

$$
L_{\mathrm{reg}}(w)=L(w)+\lambda R(w)
$$

* $R(w)$：正则化项；
* $\lambda$：正则化强度，是一个超参数。

现在我们同时最小化原始损失和权重大小。

## 第 49 页

L2 正则化：最常见的选择

最常见的形式是 L2 正则化：

$$
R(w)=\frac{1}{2}|w|*2^2=\frac{1}{2}\sum*{j=1}^p w_j^2
$$

* 对较大的权重惩罚更严重，因为有平方项；
* 使模型中的权重变得分散且较小；
* 系数 $\frac{1}{2}$ 用来简化求导。

最终的 L2 正则化损失为：

$$
L_{\mathrm{reg}}(w)=L(w)+\frac{\lambda}{2}\sum_{j=1}^p w_j^2
$$

## 第 50 页

与梯度下降的关系

看一下权重更新规则如何改变。

标准更新：

$$
w_j\leftarrow w_j-\eta\frac{\partial L}{\partial w_j}
$$

加入 L2 正则化后，新的梯度为：

$$
\frac{\partial L_{\mathrm{reg}}}{\partial w_j}=\frac{\partial L}{\partial w_j}+\lambda w_j
$$

因此：

$$
w_j\leftarrow w_j-\eta\left(\frac{\partial L}{\partial w_j}+\lambda w_j\right)
$$

这个式子可以重新整理为：

$$
w_j\leftarrow (1-\eta\lambda)w_j-\eta\frac{\partial L}{\partial w_j}
$$

也就是说，在主要更新之前，权重会先乘上一个衰减因子 $(1-\eta\lambda)$。

## 第 51 页

为什么这有帮助？

* 促进简单性：鼓励模型弱地使用所有输入，而不是过度依赖少数输入；
* 改善泛化能力：更平滑的模型对输入数据中的小波动或噪声不那么敏感；
* 数值稳定性：有助于稳定优化过程。

## 第 52 页

超参数 $\lambda$

$\lambda$ 控制惩罚强度。

* $\lambda=0$：没有影响，模型可能过拟合；
* $\lambda$ 太小：影响很弱，仍可能过拟合；
* $\lambda$ 适中：泛化性能较好；
* $\lambda$ 太大：欠拟合，权重被迫接近 0，模型无法学习复杂模式。

$\lambda$ 必须谨慎调节，例如通过交叉验证。

## 第 53 页

总结

* 权重衰减是一种用于对抗过拟合的正则化技术；
* 它通过向损失函数中加入惩罚项实现，通常使用 L2 惩罚；
* 该惩罚项鼓励较小的权重，从而得到更简单的模型；
* 在梯度下降过程中，它表现为每一步对权重进行乘法衰减；
* 关键超参数 $\lambda$ 控制正则化强度，需要调节。

## 第 54 页

方法 2：Dropout

* 灵感：大脑中的神经元并不是同时全部激活；
* 思想：训练过程中随机“丢弃”一些神经元；
* 效果：迫使网络学习冗余且稳健的路径；
* 防止某一个神经元成为关键瓶颈。

## 第 55 页

Dropout 操作

核心概念：

对于每个训练样本，随机将某一层中比例为 $p$ 的神经元置为 0。剩下的神经元会按比例放大，放大因子为 $\frac{1}{1-p}$。

## 第 56 页

带 Dropout 的训练阶段

对每个隐藏层 $l$ 和 mini-batch 中的每个训练样本：

1. 采样一个 mask 向量 $r$：

$$
r_j^{(l)}\sim \mathrm{Bernoulli}(1-p)
$$

对每个神经元 $j$ 都这样采样。

2. 将 mask 应用于该层输出 $y$：

$$
\tilde y^{(l)}=r^{(l)}\odot y^{(l)}
$$

3. 对激活值进行缩放，即 Inverted Dropout：

$$
\tilde y^{(l)}=\frac{r^{(l)}\odot y^{(l)}}{1-p}
$$

4. 使用 $\tilde y^{(l)}$ 继续进行前向传播和反向传播。

我们每次训练的实际上都是一个不同的“变薄”的子网络。

## 第 57 页

推理/测试阶段

Dropout 会被关闭。

* 我们希望使用完整网络的能力进行预测；
* 所有神经元都是激活的，不进行随机采样；
* 关键点：因为使用了 Inverted Dropout，即训练时已经进行了缩放，所以测试时不需要做任何特殊处理；
* 权重已经处于最终的、正确缩放的状态；
* 直接像平常一样进行前向传播即可。

为什么缩放有效？

训练时进行缩放，可以保证测试时所有神经元都激活时的期望输出，与训练时只有部分神经元激活时的期望输出一致。

## 第 58 页

Dropout 有效的原因

1. 防止共适应
   神经元不能依赖相邻神经元，它们必须变得独立有用。

2. 近似 Bagging
   相当于训练 $2^N$ 个子网络，并对它们的预测取平均，是一种模型集成形式。

3. 鲁棒性
   网络学会在数据缺失的情况下仍然做出准确预测，因此对噪声更加鲁棒。

## 第 59 页

Dropout 的实践使用

* 放置位置：通常放在全连接层之后，这是最常见的用法；有时也放在卷积层之后，不过在卷积层中 BatchNorm 往往更常用。
* Dropout 比例 $p$：

  * 隐藏层：一个较好的默认值是 $p=0.5$；
  * 输入层：使用更低的比例，例如 $p=0.1$ 或 $p=0.2$。
* 与其他技术的相互作用：

  * 与 L2 权重衰减等其他正则化方法结合效果很好；
  * 经常与 Batch Normalization 一起使用，不过 Dropout 放在 BN 前还是 BN 后属于设计选择。
* 对训练时间的影响：训练时间大约会变为原来的 2 到 3 倍，因为虽然部分神经元被 mask 掉，前向和反向传播仍然是在整个网络上进行的。不过，它能显著减少过拟合。

## 第 60 页

总结

* Dropout 是一种用于对抗过拟合和共适应的正则化技术；
* 它通过在训练过程中随机丢弃神经元，迫使网络学习稳健特征；
* 它使用 Inverted Dropout，即训练时缩放，从而提高推理阶段的效率；
* 它的力量来自于同时训练大量子网络组成的集成模型；
* 它是深度学习中简单、高效且广泛使用的工具。

## 第 61 页

什么是早停 Early Stopping？I

核心思想：

训练过程中监控验证集性能，当验证性能开始变差时停止训练，从而防止模型对训练数据过拟合。

关键组成：

* 训练集：用于更新权重；
* 验证集：用于监控性能；
* Patience：从上一次提升之后继续等待多久；
* 最佳权重：验证性能最优时的模型快照。

可视化概念：

## 第 62 页

什么是早停 Early Stopping？II

图中横轴是 epochs，纵轴是 loss。

* 训练损失通常持续下降；
* 验证损失先下降后上升；
* 当验证损失开始恶化并超过等待阈值时，触发 Early Stop。

## 第 63 页

早停如何防止过拟合

理论解释：

* 限制模型的有效复杂度；
* 起到隐式正则化作用；
* 控制优化过程；
* 防止模型过度专门适应训练数据。

## 第 64 页

与其他正则化方法的比较

| 方法                  | 计算开销 | 超参数 | 有效性 |
| ------------------- | ---- | --- | --- |
| Early Stopping      | 低    | 少   | 高   |
| L1/L2 正则化           | 中等   | 中等  | 中等  |
| Dropout             | 中等   | 少   | 高   |
| 数据增强                | 高    | 多   | 高   |
| Batch Normalization | 低    | 少   | 中等  |

表：正则化技术比较。

早停的独特优势：

* 无推理开销：测试时模型不变；
* 自动化：减少手动选择 epoch 的需求；
* 通用性：适用于任何网络结构；
* 节省计算：停止不必要的训练。
