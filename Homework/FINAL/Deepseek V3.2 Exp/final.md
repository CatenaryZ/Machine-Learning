## 1 Final Project

Choose one of the following two problems.

### Project A: Architectural Analysis of DeepSeek-V3.2

This project involves a comprehensive explanation of the architecture of DeepSeek-V3.2. You must explain the provided system in full technical detail, highlighting its key innovations. see https://arxiv.org/pdf/2512.02556, and https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference.

**Required Components:**

**(1) DeepSeek Sparse Attention (DSA) and the Lightning Indexer:**

* *Explain the prototype of DSA, which consists of two main components: a _lightning indexer_ and a _fine-grained token selection mechanism_.

* *Detail the function of the **lightning indexer**. Its purpose is to compute an index score \(I_{t,s}\) between a query token \(\mathbf{h}_{t}\) and preceding tokens \(\mathbf{h}_{s}\) to determine which tokens should be selected for attention. The score is calculated as: \[I_{t,s}=\sum_{j=1}^{H^{l}}w_{t,j}^{I}\cdot\text{ReLU}\left(\mathbf{q}_{t,j}^{I} \cdot\mathbf{k}_{s}^{I}\right),\] where \(H^{l}\) is the number of indexer heads, and \(\mathbf{q}_{t,j}^{I}\), \(w_{t,j}^{I}\), and \(\mathbf{k}_{s}^{I}\) are derived from the query and key tokens. ReLU is used for activation to improve throughput.

* *Describe the **fine-grained token selection mechanism**. This component retrieves only the key-value entries \(\{\mathbf{c}_{s}\}\) corresponding to the top-\(k\) index scores. The final attention output \(\mathbf{u}_{t}\) for token \(\mathbf{h}_{t}\) is then computed using standard attention on this sparse set: \[\mathbf{u}_{t}=\text{Attn}(\mathbf{h}_{t},\{\mathbf{c}_{s}\mid I_{t,s}\in\text{Top-k}(I_{t,:})\}).\]

* *Clarify how DSA is instantiated within the framework of **Multi-Head Latent Attention (MLA)** (from DeepSeek-V2). For computational efficiency, DSA is implemented based on the Multi-Query Attention (MQA) mode of MLA, where a single latent key-value vector is shared across all query heads.

* *Explain the two-stage continued pre-training process used to integrate DSA into the base model (DeepSeek-V3.1-Terminus): 1. (a)**Dense Warm-up Stage**: All model parameters are frozen except for the lightning indexer, which is trained for 1000 steps using a KL-divergence loss to align its output distribution with that of the main model's dense attention. 2. (b)**Sparse Training Stage**: The fine-grained token selection is activated, and all model parameters are optimized to adapt to the sparse attention pattern. The indexer is trained with a loss calculated only over the selected top-\(k\) tokens.

* *Discuss the efficiency gains: DSA reduces the core attention complexity from \(O(L^{2})\) to \(O(Lk)\) (where \(k\ll L\)), leading to significant inference speedups for long-context sequences.

**(2) The Mixture-of-Experts (MoE) Architecture:**

* *Describe the core principle of the **DeepSeekMoE** architecture used in DeepSeek-V2, which is a foundational component of the V3 series.

* *Highlight its key features designed for economical training and efficient inference: * **-****Fine-grained Expert Segmentation**: Experts are divided into smaller, more numerous units to allow for more specialized and flexible routing. * **-****Shared Expert Isolation**: A subset of experts is designated as "shared" and is always activated, ensuring stability and retaining common knowledge. * **-****Load Balancing Mechanisms**: Strategies like auxiliary loss and token-dropping are employed to ensure experts are utilized evenly during training.

* *Specify the scale: DeepSeek-V2 has 236B total parameters but activates only 21B per token, dramatically reducing computational costs during inference compared to dense models of similar total size.

* *Explain how this MoE design, combined with MLA, enables DeepSeek models to achieve top-tier performance while maintaining high training and inference efficiency.

### Deliverable:

Submit a detailed technical paper synthesizing information from the provided sources. Your analysis should connect the architectural innovations (DSA, MLA, DeepSeekMoE) to the model's stated goals of high efficiency and superior performance in reasoning and agent tasks.

### Project B: Train Your Own GPT Using the Muon Optimizer

This project involves a review of an alternative optimizer and a practical training exercise.

#### Required Components:

**(1) Review the Muon Optimizer:**

* ***Note:** (See https://kellerjordan.github.io/posts/muon/ ).

* *Your review should summarize the optimizer's proposed algorithm, its stated advantages over established optimizers (like AdamW), and its theoretical or empirical basis.

* *Discuss its potential benefits and drawbacks for training modern neural networks, particularly transformer-based LLMs.

**(2) Practical Training with nanoGPT:**

* *Use one of the referenced codebases (https://github.com/KellerJordan/modded-nanogpt or https://github.com/Deveraux-Parker/nanoGPT_1GPU_SPEEDRUN) or the original nanoGPT project to train a small-scale GPT model.

* *Your goal is to set up a training run, potentially incorporating the Muon optimizer as a comparative experiment if feasible.

* *Document your process, hyperparameters, and results.

#### Deliverable:

Submit a review paper covering the Muon optimizer and a separate report on your training exercise. The latter must include your training logs, the final model checkpoint, and a discussion of your results and observations.

#### Good Luck!

---

```python
class Indexer(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # ===== 基本维度与超参数 =====

        self.dim: int = args.dim
        # 原始 token embedding 维度（用于构造 index key 的输入）

        self.n_heads: int = args.index_n_heads
        # Index 专用的 head 数（独立于 Attention 的 n_heads）

        self.n_local_heads = args.index_n_heads // world_size
        # 分布式场景下，每个 rank 负责的 index head 数

        self.head_dim: int = args.index_head_dim
        # 每个 index head 的维度

        self.rope_head_dim: int = args.qk_rope_head_dim
        # 用于 RoPE 的子维度（index 也使用旋转位置编码）

        self.index_topk: int = args.index_topk
        # Lightning Index 最终选取的 Top-K 历史 token 数

        self.q_lora_rank: int = args.q_lora_rank
        # Query 的低秩维度（Index Query 复用 MLA 的 LoRA 表征）

        # ===== Index Query 投影 =====
        # 输入：qr ∈ R^{q_lora_rank}
        # 输出：q ∈ R^{n_heads * head_dim}
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)

        # ===== Index Key 投影 =====
        # 输入：x ∈ R^{dim}
        # 输出：k ∈ R^{head_dim}
        self.wk = Linear(self.dim, self.head_dim)

        # 对 index key 做 LayerNorm，稳定内积数值
        self.k_norm = LayerNorm(self.head_dim)

        # ===== Index Head 权重投影 =====
        # 每个 token 生成 n_heads 个标量权重，用于加权不同 head 的 index score
        # checkpoint 中是 bf16，这里用 fp32 以提高数值稳定性
        self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.float32)

        # Attention / Index 内积的标准缩放项
        self.softmax_scale = self.head_dim ** -0.5

        # FP8 量化所用的 scale 格式
        self.scale_fmt = args.scale_fmt

        # ===== Index Key Cache（FP8）=====
        # k_cache: 存储历史 token 的 index key（FP8）
        # 形状: [bsz, max_seq_len, head_dim]
        self.register_buffer(
            "k_cache",
            torch.zeros(
                args.max_batch_size,
                args.max_seq_len,
                self.head_dim,
                dtype=torch.float8_e4m3fn
            ),
            persistent=False
        )

        # k_scale_cache: 对应的 block-wise scale
        # 形状: [bsz, max_seq_len, head_dim // block_size]
        self.register_buffer(
            "k_scale_cache",
            torch.zeros(
                args.max_batch_size,
                args.max_seq_len,
                self.head_dim // block_size,
                dtype=torch.float32
            ),
            persistent=False
        )


    def forward(
        self,
        x: torch.Tensor,          # 当前层 token 表征 [bsz, seqlen, dim]
        qr: torch.Tensor,         # LoRA Query 表征 [bsz, seqlen, q_lora_rank]
        start_pos: int,           # KV cache 起始位置
        freqs_cis: torch.Tensor,  # RoPE 频率
        mask: Optional[torch.Tensor]  # causal / padding mask
    ):
        # ===== 基本形状信息 =====
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # ===== 构造 Index Query =====
        # 线性映射到 n_heads * head_dim
        q = self.wq_b(qr)
        # reshape 成多头形式
        # [bsz, seqlen, n_heads, head_dim]
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)

        # 将 query 拆分为 RoPE 部分和非 RoPE 部分
        q_pe, q_nope = torch.split(
            q,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1
        )

        # Indexer 中的 RoPE 使用非 interleaved 形式
        q_pe = apply_rotary_emb(q_pe, freqs_cis, False)

        # 重新拼接完整的 query
        q = torch.cat([q_pe, q_nope], dim=-1)

        # ===== 构造 Index Key =====
        # 从 token 表征直接线性映射
        k = self.wk(x)

        # LayerNorm，保证不同 token 的 key 分布稳定
        k = self.k_norm(k)

        # 同样拆分为 RoPE / 非 RoPE
        k_pe, k_nope = torch.split(
            k,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1
        )

        # 注意：key 没有 head 维，需要临时 unsqueeze
        k_pe = apply_rotary_emb(
            k_pe.unsqueeze(2),
            freqs_cis,
            False
        ).squeeze(2)

        # 拼回完整 key
        k = torch.cat([k_pe, k_nope], dim=-1)

        # ===== Hadamard 旋转（提升近似内积质量）=====
        # 通过快速 Hadamard 变换增强各维混合
        q = rotate_activation(q)
        k = rotate_activation(k)

        # ===== FP8 量化 =====
        # block-wise 量化，返回 FP8 张量和对应 scale
        q_fp8, q_scale = act_quant(q, block_size, self.scale_fmt)
        k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt)

        # ===== 写入 Index Key Cache =====
        # 仅缓存 key（query 是即时计算的）
        self.k_cache[:bsz, start_pos:end_pos] = k_fp8
        self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale

        # ===== 计算 Index Head 权重 =====
        # 每个 token -> n_heads 个标量
        weights = self.weights_proj(x.float()) * self.n_heads ** -0.5

        # 扩展维度并融合：
        # - q_scale：FP8 反量化所需
        # - softmax_scale：内积标准缩放
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale

        # ===== Lightning Index 核心算子 =====
        # 计算 q 与所有历史 k 的近似内积得分
        # 输出形状: [bsz, seqlen, end_pos]
        index_score = fp8_index(
            q_fp8.contiguous(),                         # 当前 query（FP8）
            weights,                                   # 动态 head 权重
            self.k_cache[:bsz, :end_pos].contiguous(), # 历史 key cache
            self.k_scale_cache[:bsz, :end_pos].contiguous()
        )

        # ===== 加入 mask（如 causal mask）=====
        if mask is not None:
            index_score += mask

        # ===== Top-K 索引选择 =====
        # 对每个 query，从历史 token 中选 Top-K
        topk_indices = index_score.topk(
            min(self.index_topk, end_pos),
            dim=-1
        )[1]

        # ===== 分布式一致性检查 =====
        # 所有 rank 应该得到完全一致的 index 结果
        topk_indices_ = topk_indices.clone()
        dist.broadcast(topk_indices_, src=0)
        assert torch.all(topk_indices == topk_indices_), \
            f"{topk_indices=} {topk_indices_=}"

        # 返回 Lightning Index 选中的 token 下标
        return topk_indices

```