## Lecture 7.3: Transformer

> - 语言模型的自回归形式
> - tokenizer、BPE 与 embedding
> - attention、multi-head attention 与位置编码
> - Transformer block、训练与采样策略

### 语言模型的基本任务

- 语言模型的核心任务，是在给定上下文的条件下预测下一个最可能的词。

#### 技术挑战
- 长距离依赖：要捕捉相隔较远的词之间的关系。
- 多义性：同一个词在不同上下文中有不同含义。
- 复杂逻辑：需要理解语法、常识与推理。

### 语言模型可以生成文本

- 一旦有了语言模型，就可以按顺序不断预测下一个 token，从而生成整段人类语言。

### 语言模型的数学表示

- 语言模型本质上是一个条件概率模型：

$$
p(x_n \mid x_1,\ldots,x_{n-1}).
$$

- 这就是 auto-regressive 建模。

#### 例子
- 给定上下文 “I can do”，模型会对词表中的候选词给出一个条件概率分布，并选择概率最高者。

---

### Section 3: Transformer

### 什么是 Tokenizer

- tokenizer 的作用，是把原始文本切分成模型能够处理的 token。
- 它是人类语言与模型数值表示之间的桥梁。

#### 为什么需要 tokenization
- 模型处理的是数字而不是原始文本；
- 词表规模需要可控；
- 要能较好处理 rare words；
- 要支持可变长度序列。

#### 常见方式
- word-based；
- character-based；
- subword-based。

### OpenAI 的 tokenizer

- 讲义给出了 OpenAI tokenizer 的链接作为参考。

### 什么是 BPE Tokenizer

- Byte Pair Encoding 是一种 subword tokenization 方法。
- 它会不断合并训练语料中最常见的字符对或子词对，构造子词词表。

#### 直觉
- BPE 在 word-level 与 character-level 之间取得折中：
  - 避免词级别词表过大；
  - 也避免字符级序列过长。

### BPE 的训练过程

1. 用字符或字节初始化词表。
2. 统计当前所有相邻 pair 的频率。
3. 合并最频繁的 pair。
4. 重复，直到达到目标词表大小。

### BPE 训练示例

- 讲义用 “low lower newest widest” 这个例子，展示了 merge 的逐步过程。

### BPE 的编码过程

- 对新文本，按照已经学好的 merges 规则，贪心地做最长可能的合并。

#### 例子
- “lowest” 会被切成 “low” + “est”。

### BPE 在现代 LLM 中的使用

- GPT 系列采用 byte-level BPE。
- BERT 使用 WordPiece 的变体。
- SentencePiece 是另一种广泛使用的方法。

### 优点与局限

#### 优点
- 没有 UNK；
- 词表紧凑；
- 支持多语言；
- 能部分保留词形结构。

#### 局限
- 贪心策略未必全局最优；
- 某些词可能有多种合理切分；
- 对某些语言可能会过度切分。

### 词嵌入 Embedding

- token 是离散 ID，本身不含语义结构。
- embedding 的作用就是把 token 映射到连续向量空间：

$$
\mathrm{embedding}: \mathbb N \to \mathbb R^d.
$$

- 相似词在向量空间中往往距离更近。

### 什么是 Embeddings in LLMs

- embedding 把离散 token 转成稠密向量表示。
- 这些向量在训练中自动学习，并逐渐编码出语义关系。

### Embeddings 是如何工作的

- 本质上是一个查表矩阵 $E \in \mathbb R^{V\times d}$。
- 每个 token 对应矩阵中的一行向量。

#### 关键性质
- semantic similarity；
- 词向量类比关系；
- 后续层会进一步加入上下文信息。

### 常见 embedding 方法概览

- Static embeddings：
  - Word2Vec；
  - GloVe；
  - FastText。

- Contextual embeddings：
  - BERT；
  - ELMo；
  - Transformer-based representations。

### Word2Vec

- Word2Vec 的核心是训练一个简单神经网络，让相似上下文中的词得到相似向量。

#### 两种结构
- Skip-gram：由中心词预测上下文。
- CBOW：由上下文预测中心词。

### Skip-gram 的数学形式

- 目标是最大化上下文词在给定中心词下的平均对数概率。
- 概率通常通过 softmax 来定义。

### 现代 LLM 的 embedding 方式

- 现代 Transformer 中通常使用 subword token；
- embedding 与 positional encoding 一起送入 Transformer；
- 整个表示是 end-to-end 学出来的。

---

### Attention Mechanism

### 注意力机制的核心思想

- 注意力机制为每个 token 分配它对其它 token 的关注权重。
- 权重越高，表示关系越重要。

#### 例子
- 在句子 “I swam across the river to get to the other bank” 中，模型会对 “river” 和 “swam” 赋予较高注意权重，从而理解 “bank” 在这里更可能是河岸而不是银行。

### 基本注意力公式

- 对输入矩阵 $Q,K,V$，注意力定义为：

$$
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V.
$$

- softmax 按行方向进行。

### Self-Attention

- 在 self-attention 中，query、key、value 都来自同一个输入序列：

$$
\mathrm{SelfAttention}(X)
=
\mathrm{Attention}(XW^Q, XW^K, XW^V).
$$

#### 缩放因子
- $\frac{1}{\sqrt{d_k}}$ 的作用，是防止点积过大，让 softmax 进入饱和区。

### Attention 的矩阵视角

- 输入序列记为 $X\in\mathbb R^{n\times d_{model}}$。
- $W^Q, W^K, W^V$ 是可学习参数矩阵。
- 输出中每个位置都聚合了其它位置的信息。

### Multi-Head Attention

#### Step 1：投影到多个子空间
- 对每个 head，分别计算 $Q_i, K_i, V_i$。

#### Step 2：每个 head 单独做 attention

- 每个 head 会学习不同的依赖模式。

### 为什么需要多个头

- 不同 head 可以同时关注不同位置、不同关系和不同特征子空间。

### Multi-Head 的输出

- 把所有 head 的结果拼接，再通过一个线性投影：

$$
\mathrm{MultiHead}(X)
=
[\mathrm{head}_1;\ldots;\mathrm{head}_h]W^O.
$$

### 参数量

- 讲义还给出了多头注意力中参数数量的估算方式。

### Positional Encoding

- 由于 attention 本身对位置是对称的，所以需要显式加入位置信息。

#### Sinusoidal Positional Encoding
- 对位置 $pos$ 和维度索引 $2i, 2i+1$，定义正弦/余弦编码。

- 这种编码是固定的、非学习式的，并且可以推广到更长序列。

### Advanced Position Representations

- 讲义还介绍了 relative position representations。
- 这里注意力分数中会显式加入相对位置信息。

### Rotary Position Embedding

- RoPE 用旋转矩阵把位置信息编码到 query 和 key 中。
- 一个关键性质是：attention score 只依赖相对位置差。

### Masked Self-Attention

- 在文本生成中，当前位置只能看见自己之前的 token。
- 因此需要在注意力分数上加一个 mask matrix $M$，把未来位置屏蔽掉。

#### Mask Matrix
- 它是下三角结构；
- 上三角位置填入 $-\infty$，从而在 softmax 后变成 0。

### Add & Norm

- Transformer 层中的一个核心组件是 “Add & Norm”。
- 它由两部分组成：
  1. residual connection；
  2. layer normalization。

### Residual Connection

- 子层输入与输出直接相加：

$$
x_{out}=x+\mathrm{SelfAttention}(x).
$$

- 它有助于保留梯度，缓解 vanishing gradient。

### Layer Normalization

- LayerNorm 在特征维度上做归一化：

$$
\mathrm{LayerNorm}(x)=
\gamma \frac{x-\mu}{\sqrt{\sigma^2 + \epsilon}} + \beta.
$$

- 它能够稳定激活值、加快收敛。

### Add & Norm 在哪里使用

- 一次在 self-attention 之后；
- 一次在 feedforward network 之后。

### Feedforward Layer

- FFN 本质上是标准前馈神经网络；
- 输入维度和输出维度通常都为 $d_{model}$。

### Transformer Block

- 把 self-attention、Add & Norm 和 FFN 组合起来，就得到一个 transformer block。
- 多个 block 可以堆叠。

### 输出层

- 堆叠若干 transformer blocks 后，接一个线性层，再用 softmax 得到词表大小为 $K$ 的输出概率。

### GPT-2 Architecture

- 讲义给出了 GPT-2 的结构示意。

### 训练过程

- 给定长度为 $n$ 的输入 token 序列，模型会输出 $n$ 个位置上的概率分布。
- 目标序列向后平移一位。

#### 损失函数
- 用所有位置上的平均交叉熵：

$$
L = \frac1n \sum_{i=1}^n \mathrm{CrossEntropy}(out_i, t_{i+1}).
$$

### Prediction

- 模型训练好后，就可以根据最后一个位置的输出分布来预测下一个 token。

---

### Sampling in LLMs

### 什么是采样

- 语言模型给出下一个词的概率分布后，还需要决定到底选哪个词。
- 这就引出 sampling 的问题。

#### 核心矛盾
- safe but boring；
- creative but risky。

### Temperature

- Temperature 用来调节分布的平滑程度，也就是“创造性”。

#### 低温
- 更确定；
- 更保守；
- 更适合事实性生成。

#### 高温
- 更发散；
- 更多样；
- 更适合创意性生成。

### Top-k Sampling

- Top-k 只保留概率最高的前 $k$ 个词，再从中采样。
- 它的作用是过滤掉概率极低、质量很差的候选词。

### Temperature 与 Top-k 如何配合

1. 模型先对所有词打分；
2. 用 temperature 调节分布形状；
3. 用 top-k 过滤掉尾部低质量词；
4. 再从剩余候选中采样。

### Quick Comparison

- Temperature 控制 creativity vs safety。
- Top-k 控制 quality vs variety。

### Summary

- Temperature 控制随机性；
- Top-k 过滤坏选项；
- 两者配合使用，能更好地控制生成风格与质量。
