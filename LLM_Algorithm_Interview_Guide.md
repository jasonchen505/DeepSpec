# LLM 算法实习面试深度准备指南

> **基于 DeepSpec + Speculators 项目深度分析**
> **适用对象**：找 LLM 算法实习的 MS 在读学生
> **重点覆盖**：Speculative Decoding、推理优化、后训练、模型架构设计

---

## 目录

- [第一部分：项目概述与技术定位](#第一部分项目概述与技术定位)
- [第二部分：Speculative Decoding 核心原理](#第二部分speculative-decoding-核心原理)
- [第三部分：三种算法深度对比](#第三部分三种算法深度对比)
- [第四部分：关键技术细节深挖](#第四部分关键技术细节深挖)
- [第五部分：训练系统与工程实践](#第五部分训练系统与工程实践)
- [第六部分：面试高频问题与深度回答](#第六部分面试高频问题与深度回答)
- [第七部分：算法实习面试考察点](#第七部分算法实习面试考察点)
- [第八部分：项目介绍话术模板](#第八部分项目介绍话术模板)
- [附录：关键代码路径速查](#附录关键代码路径速查)

---

## 第一部分：项目概述与技术定位

### 1.1 DeepSpec 项目定位

**DeepSpec** 是一个用于训练和评估 **Speculative Decoding draft 模型** 的全栈代码库。

**核心价值**：
- 实现了三种前沿的 speculative decoding 算法：DSpark、DFlash、Eagle3
- 提供完整的数据准备 → 训练 → 评估流水线
- 专注于 draft 模型的训练优化

**技术栈**：
```
PyTorch + FSDP + Triton + FlexAttention
Target Models: Qwen3-4B/8B/14B, Gemma-4-12B
```

### 1.2 Speculators 项目定位

**Speculators** 是 Red Hat 开发的、隶属于 vLLM 项目组织的开源库。

**核心价值**：
- 训练好的 draft 模型可直接部署到 vLLM
- 支持 EAGLE-3、DFlash、P-EAGLE、MTP 四种算法
- 生产级的训练和部署流程

### 1.3 两个项目的关系

| 维度 | DeepSpec | Speculators |
|------|----------|-------------|
| **定位** | 研究导向，算法创新 | 生产导向，工程落地 |
| **算法** | DSpark（创新）、DFlash、Eagle3 | EAGLE-3、DFlash、P-EAGLE、MTP |
| **集成** | 独立评估系统 | 深度集成 vLLM |
| **特色** | DSpark 的并行 block 预测 | 标准化模型格式、在线训练 |

**面试价值**：展示你既理解算法创新（DeepSpec），又了解工程落地（Speculators）。

---

## 第二部分：Speculative Decoding 核心原理

### 2.1 为什么需要 Speculative Decoding？

**问题背景**：
- LLM 推理是 **memory-bound** 的，受内存带宽限制而非计算限制
- 自回归生成每次只能产生 1 个 token，GPU 利用率低
- 大模型（如 70B）的单次 forward pass 延迟高

**核心思想**：
```
用小的 draft model 快速生成多个候选 token
      ↓
大的 target model 并行验证这些 token
      ↓
通过拒绝采样保证输出分布一致（lossless）
```

**数学保证**：
- 拒绝采样使得最终输出分布与直接使用 target model **完全一致**
- 这是 **无损加速**，不是近似

### 2.2 拒绝采样算法

**标准算法流程**：
```python
def speculative_decoding(draft_model, target_model, prompt, K):
    # K: 每次 draft 的 token 数
    output = []
    while not eos:
        # 1. Draft model 生成 K 个候选
        draft_tokens = draft_model.generate(prompt + output, k=K)
        draft_probs = [draft_model.prob(t) for t in draft_tokens]
        
        # 2. Target model 并行验证
        target_probs = target_model.verify(prompt + output + draft_tokens)
        
        # 3. 从左到右验证
        for i in range(K):
            accept_prob = min(1, target_probs[i] / draft_probs[i])
            if random() < accept_prob:
                output.append(draft_tokens[i])
            else:
                # 从修正分布重新采样
                residual = max(0, target_probs[i] - draft_probs[i])
                new_token = sample_from(residual)
                output.append(new_token)
                break
```

**关键指标**：
- **Acceptance Rate**：draft token 被接受的概率
- **Acceptance Length**：平均每次验证接受的 token 数
- **Speedup**：相比直接使用 target model 的加速比

### 2.3 加速效果分析

**理论加速比**：
```
Speedup = (1 + K * acceptance_rate) / (1 + K * draft_cost / target_cost)
```

**实际效果**：
- 当 acceptance_rate = 70%，K = 5 时，可获得 2-3x 加速
- Draft model 越小、target model 越大，加速效果越明显
- 对于 reasoning 任务（如数学），acceptance rate 通常更高

---

## 第三部分：三种算法深度对比

### 3.1 EAGLE-3：自回归 + Test-Time Training

**核心论文**：EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty

**架构设计**：
```
Target Model Hidden States (多层)
         ↓
    Concat + FC Layer (3*hidden_size → hidden_size)
         ↓
    Token Embedding (上一步的 token)
         ↓
    ┌────────────────┐
    │ Llama Decoder  │ ← 使用 FlexAttention
    │ Layer Stack    │
    └────────────────┘
         ↓
    LM Head → Draft Logits
```

**关键实现细节**：

1. **多层特征融合**：
```python
# 从 target model 的多个中间层提取 hidden states
# 默认使用 3 层，如 [2, 18, 33]
hidden_states = torch.cat([layer_2, layer_18, layer_33], dim=-1)
# 通过 FC 层降维：3 * hidden_size → hidden_size
```

2. **输入拼接**：
```python
# Draft model 的输入是 token embedding 和 hidden states 的拼接
hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)
# shape: [1, total_seq_len, 2 * hidden_size]
```

3. **TTT（Test-Time Training）步骤**：
```python
# 训练时模拟推理的自回归过程
for step in range(ttt_length):
    output = draft_model(input_ids, hidden_states, ...)
    hidden_states = output.hidden_states  # 更新 hidden states
    input_ids = output.next_token  # 使用预测的 token
```

**代码路径**：`src/speculators/models/eagle3/core.py`

### 3.2 DFlash：并行 Block 预测

**核心论文**：DFlash: Block Diffusion for Flash Speculative Decoding

**与 EAGLE-3 的关键区别**：

| 特性 | EAGLE-3 | DFlash |
|------|---------|--------|
| **生成方式** | 自回归（逐步） | 并行（一次生成所有） |
| **Attention** | 因果注意力 | 非因果注意力（bidirectional） |
| **架构** | Llama-style | Qwen3-style |
| **推测方式** | 逐 token | Block-based（anchor points） |
| **延迟** | 较高 | 同步请求下 2-3x 更低 |

**架构设计**：
```
Target Model Hidden States
         ↓
    FC Layer + LayerNorm
         ↓
    Mask Token Embeddings (anchor positions)
         ↓
    ┌────────────────────────────────────┐
    │ Qwen3 Decoder Layer Stack          │
    │ (Bidirectional Attention + MLP)    │
    └────────────────────────────────────┘
         ↓
    LM Head → All Draft Logits (并行)
```

**Anchor Point 机制**：
```python
# 1. 选择 anchor positions（序列中的锚点位置）
anchor_positions = select_anchors(loss_mask, block_size)

# 2. 构建 mask token 序列
# 每个 block 的第一个位置是 anchor token（真实 token）
# 后续位置是 mask token
mask_token_ids = torch.full((1, mask_tokens_size), mask_token_id)
mask_token_ids[:, ::block_size] = input_ids[:, anchor_positions]

# 3. 使用非因果注意力，每个 query 可以看到所有 context
```

**代码路径**：`src/speculators/models/dflash/core.py`

### 3.3 DSpark（DeepSpec 创新）：并行 Block + Markov Head

**核心创新**：在 DFlash 基础上增加 **Markov Head** 和 **Confidence Head**

**架构增强**：

1. **Markov Head**（二阶校正）：
```python
# 三种变体：VanillaMarkov、GatedMarkovHead、RNNHead
class VanillaMarkov(nn.Module):
    def __init__(self, vocab_size, rank):
        self.W1 = nn.Embedding(vocab_size, rank)
        self.W2 = nn.Linear(rank, vocab_size, bias=False)
    
    def forward(self, prev_token_id, logits):
        # 基于前一个 token 的 embedding 生成 logits 偏置
        prev_emb = self.W1(prev_token_id)
        bias = self.W2(prev_emb)
        return logits + bias  # 叠加到 draft logits 上
```

2. **Confidence Head**（置信度预测）：
```python
class AcceptRatePredictor(nn.Module):
    def __init__(self, input_dim):
        self.proj = nn.Linear(input_dim, 1)
    
    def forward(self, features):
        # 预测每个位置的接受率
        return self.proj(features).squeeze(-1)
```

3. **损失函数设计**：
```python
# DSpark 使用三项损失的加权和
loss = alpha_ce * CE_loss + alpha_l1 * L1_loss + alpha_conf * Confidence_loss

# 默认配置：ce=0.1, l1=0.9, conf=1.0
# L1 loss 为主，直接优化分布匹配
```

**代码路径**：`deepspec/modeling/dspark/`

### 3.4 算法对比总结

| 算法 | 生成方式 | 额外组件 | 损失函数 | 优势 |
|------|----------|----------|----------|------|
| **EAGLE-3** | 自回归 | 无 | Soft CE（多步加权） | 简单、稳定 |
| **DFlash** | 并行 | 无 | CE | 低延迟 |
| **DSpark** | 并行 | Markov Head + Confidence Head | CE + L1 + BCE | 高精度、可提前终止 |
| **P-EAGLE** | 并行多深度 | COD 采样 | Soft CE | 并行 + 高精度 |

---

## 第四部分：关键技术细节深挖

### 4.1 多层特征融合

**为什么需要多层特征？**

```python
# 不同层的 hidden states 编码不同层次的信息
# 浅层：局部语法、token 级别信息
# 深层：语义、长距离依赖

# DeepSpec 默认使用 5 层
target_layer_ids = [1, 9, 17, 25, 33]  # 对于 Qwen3-4B（36层）

# Speculators 默认使用 3 层
eagle_aux_hidden_state_layer_ids = [2, 18, 33]
```

**实现方式**：
```python
def extract_context_feature(hidden_states, layer_ids):
    return torch.cat(
        [hidden_states[0 if layer_id == -1 else layer_id + 1] for layer_id in layer_ids],
        dim=-1,
    )

# 然后通过 FC 层降维
self.fc = nn.Linear(len(target_layer_ids) * hidden_size, hidden_size, bias=False)
```

**面试深挖点**：
- Q: 为什么选择这些特定的层？
- A: 均匀分布以捕获不同深度的特征；实验验证这些层的组合效果最好。

### 4.2 Embedding/Head 共享

**为什么 draft model 要共享 target model 的 embedding 和 lm_head？**

```python
# 在训练初始化时
target_model = AutoModelForCausalLM.from_pretrained(target_model_name)
draft_model.initialize_embeddings_and_head(
    embed_tokens=target_model.get_input_embeddings(),
    lm_head=target_model.get_output_embeddings(),
    freeze=True,  # 冻结，不更新
)
```

**好处**：
1. Draft model 不需要学习词表映射
2. Draft logits 和 target logits 在同一个概率空间中
3. 减少 draft model 的参数量
4. 便于 L1 loss 和 confidence calibration

### 4.3 DSpark 的 Block-Level Parallelism

**核心设计**：每个 block 的第一个位置是 anchor token（已知），后续位置是 mask token

```python
# 创建特殊的注意力 mask
def dspark_mask_mod(b, h, q_idx, kv_idx):
    q_block_id = q_idx // block_size
    anchor_pos = anchor_positions[b, q_block_id]
    
    # Draft token 只能 attend to 之前的 context tokens
    is_context = kv_idx < seq_len
    mask_context = is_context & (kv_idx < anchor_pos)
    
    # 和同 block 内的 draft tokens
    is_draft = kv_idx >= seq_len
    kv_block_id = (kv_idx - seq_len) // block_size
    mask_draft = is_draft & (q_block_id == kv_block_id)
    
    return mask_context | mask_draft
```

**优势**：
- 训练时每个 batch 可以同时监督数百个分散的位置
- 数据效率极高

### 4.4 Eagle3 的 Test-Time Training

**TTT 机制**：在推理时通过多步迭代逐步精炼 hidden states

```python
# 训练时的 TTT 循环
for step in range(ttt_length):
    # 拼接 input_embeds 和 hidden_states
    hidden_states = torch.cat([input_layernorm(input_embeds), hidden_norm(hidden_states)], dim=-1)
    
    # 通过 decoder layers
    output = decoder_layers(hidden_states)
    
    # 更新 hidden states 用于下一步
    hidden_states = output.hidden_states
    
    # 计算 loss
    loss += step_loss * (step_loss_decay ** step)
```

**与传统自回归的区别**：
- 传统：每步只用 token embedding
- Eagle3：每步用 token embedding + 前一步的 hidden states

### 4.5 Triton 融合 Soft Cross-Entropy

**为什么需要 Triton kernel？**

```python
# 标准实现需要存储 [B, T, V] 的 fp32 log-probs
# 对于 vocab_size = 151936（Qwen3），这会消耗大量显存

# Triton 融合实现
class FusedLogSoftmaxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target_p, position_mask, normalizer):
        # 只保存 logits.detach() + 两个 fp32 标量 (m, d)
        # 不显式存储 [B, T, V] 的 log-probs
        ...
    
    @staticmethod
    def backward(ctx, grad_output):
        # 梯度直接写入 logits 存储（in-place）
        ...
```

**内存节省**：
- 标准实现：需要 2 * [B, T, V] * 4 bytes（fp32）
- Triton 实现：只需要 [B, T, V] * 2 bytes（bf16）+ 少量标量

### 4.6 Confidence Head 的作用

**训练时**：
```python
# Confidence Head 预测每个位置的接受率
confidence_targets = 1.0 - 0.5 * (draft_probs - target_probs).abs().sum(dim=-1)
confidence_loss = F.binary_cross_entropy_with_logits(confidence_pred, confidence_targets)
```

**推理时**：
```python
# 当 confidence_threshold > 0 时，提前截断低置信度的 block
if confidence < threshold:
    break  # 停止生成后续 draft tokens
```

**评估指标**：
- ECE（Expected Calibration Error）
- AUROC
- Brier Score

---

## 第五部分：训练系统与工程实践

### 5.1 数据准备流程

**三步流水线**：

```bash
# Step 1: 下载和拆分数据
python scripts/data/download_and_split.py

# Step 2: 用 Target 模型重新生成回答
python scripts/data/generate_train_data.py

# Step 3: 预计算 Target Cache（最耗时）
python scripts/data/prepare_target_cache.py
```

**Target Cache 存储**：
```python
# 每条训练数据包含
{
    "input_ids": [seq_len],                    # int32
    "attention_mask": [seq_len],               # uint8
    "loss_mask": [seq_len],                    # uint8
    "target_hidden_states": [seq_len, num_layers * hidden_size],  # bfloat16
    "target_last_hidden_states": [seq_len, hidden_size],          # bfloat16
}

# 存储警告：对于 Qwen3-4B，大约需要 38 TB 磁盘空间
```

### 5.2 训练循环核心流程

```python
class BaseTrainer:
    def train(self):
        for batch in prefetcher:
            # 1. 梯度累积控制
            should_sync = (micro_step + 1) % gradient_accumulation_steps == 0
            
            # 2. Forward pass
            with self.model.no_sync() if not should_sync else nullcontext():
                loss = self.run_batch(batch) / gradient_accumulation_steps
                loss.backward()
            
            # 3. 梯度裁剪 + 优化器更新
            if should_sync:
                grad_norm = FSDP.clip_grad_norm_(self.model, max_grad_norm)
                self.optimizer.step()
```

### 5.3 BF16 优化器

```python
class BF16Optimizer:
    def __init__(self, model, lr, total_steps, warmup_ratio, weight_decay):
        # 维护一份 fp32 的 master 参数副本
        self.fp32_params = [p.clone().float() for p in model.parameters()]
        
    def step(self):
        # 1. 梯度从 bf16 转为 fp32
        # 2. AdamW 更新 fp32 参数
        # 3. 拷贝回 bf16
```

### 5.4 FSDP 分布式训练

```python
# FSDP 配置
fsdp_kwargs = {
    "use_orig_params": True,
    "mixed_precision": MixedPrecision(
        param_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    "sharding_strategy": ShardingStrategy.NO_SHARD,  # 单机 DDP
}

# 包装模型
model = FSDP(model, **fsdp_kwargs)
```

### 5.5 训练超参数

**DeepSpec 默认配置**：
```python
lr = 6e-4
warmup_ratio = 0.04
weight_decay = 0.0
global_batch_size = 512
local_batch_size = 1  # 每 GPU
num_train_epochs = 10
max_grad_norm = 1.0
precision = "bf16"
```

**DSpark 特有配置**：
```python
block_size = 7
num_draft_layers = 5
target_layer_ids = [1, 9, 17, 25, 33]
num_anchors = 512
markov_rank = 256
ce_loss_alpha = 0.1
l1_loss_alpha = 0.9
confidence_head_alpha = 1.0
```

---

## 第六部分：面试高频问题与深度回答

### 6.1 基础概念类

**Q1: 什么是 Speculative Decoding？为什么能加速推理？**

**A**: Speculative Decoding 是一种无损加速 LLM 推理的技术。核心思想是用小的 draft model 快速生成多个候选 token，然后让大的 target model 并行验证。

加速原理：
1. LLM 推理是 memory-bound 的，GPU 利用率低
2. 自回归生成每次只能产生 1 个 token
3. 通过 draft model 预测多个 token，target model 可以一次验证
4. 拒绝采样保证输出分布与直接使用 target model 完全一致

**Q2: 什么是拒绝采样？如何保证输出分布一致？**

**A**: 拒绝采样是一种从目标分布采样的方法：

```python
# 对于每个 draft token
accept_prob = min(1, target_prob / draft_prob)
if random() < accept_prob:
    accept token
else:
    # 从修正分布重新采样
    residual = max(0, target_prob - draft_prob)
    new_token = sample_from(residual)
```

数学证明：通过这种机制，最终输出的 token 分布恰好等于 target model 的分布。

### 6.2 算法设计类

**Q3: EAGLE-3 的 TTT（Test-Time Training）是什么意思？**

**A**: TTT 是指 draft model 在推理时通过多步迭代逐步精炼 hidden states：

1. 传统自回归：每步只用 token embedding
2. Eagle3 TTT：每步用 token embedding + 前一步的 hidden states
3. 这使得 draft model 可以"看到"更丰富的上下文信息

好处：
- 提高 draft token 的准确性
- 每一步都能产生有意义的预测（不是只有最后一步）

**Q4: DFlash 的并行生成如何实现？与自回归有什么区别？**

**A**: DFlash 使用非因果注意力实现并行生成：

1. **自回归（EAGLE-3）**：生成 token k 需要 token 1 到 k-1 的结果
2. **并行（DFlash）**：所有 draft tokens 同时生成，每个 token 可以看到所有 context

实现方式：
- 使用 anchor points 将序列分成多个 block
- 每个 block 内使用非因果注意力（bidirectional）
- 不同 block 之间保持独立

**Q5: DSpark 的 Markov Head 有什么作用？**

**A**: Markov Head 提供二阶校正，提高 block 内 token 间的一致性：

```python
# 标准 draft model 只用 hidden states 预测
logits = lm_head(hidden_states)

# Markov Head 叠加基于前序 token 的偏置
bias = W2(W1(prev_token_id))
logits = logits + bias
```

好处：
- 捕获 token 之间的局部依赖关系
- 提高 block 内 token 的一致性
- 参数量很小（只有两个小矩阵）

### 6.3 训练优化类

**Q6: 为什么 draft model 要共享 target model 的 embedding 和 lm_head？**

**A**: 三个主要原因：

1. **概率空间一致**：Draft logits 和 target logits 在同一个空间中，便于计算 L1 loss 和 confidence calibration
2. **减少参数量**：Draft model 不需要学习词表映射
3. **训练稳定**：避免 draft model 学习到与 target model 不一致的表示

**Q7: 多层特征融合有什么好处？如何选择层数？**

**A**: 不同层的 hidden states 编码不同层次的信息：

- 浅层（如 layer 1-3）：局部语法、token 级别信息
- 中层（如 layer 10-20）：句子级别语义
- 深层（如 layer 30+）：长距离依赖、全局语义

选择策略：
- 均匀分布以捕获不同深度的特征
- 通常 3-5 层效果最好
- 通过实验验证具体选择

**Q8: Triton 融合 Soft Cross-Entropy 有什么优势？**

**A**: 内存效率是关键：

```python
# 标准实现
log_probs = torch.log_softmax(logits, dim=-1)  # [B, T, V] fp32
loss = -torch.sum(target * log_probs, dim=-1)

# Triton 融合实现
# 只保存 logits.detach() + 两个标量 (m, d)
# 不显式存储 [B, T, V] 的 log-probs
```

内存节省：对于 vocab_size = 151936，节省约 50% 的显存。

### 6.4 工程实践类

**Q9: 如何处理大规模的 Target Cache 存储？**

**A**: DeepSpec 使用二进制分片文件 + 索引文件：

```python
# 存储格式
shard_0.bin  # 64 GB 每个 shard
shard_1.bin
...
index.bin     # 索引文件，每条记录 44 字节
manifest.json # 元数据

# 使用 mmap 按需读取，支持 LRU 缓存
```

**Q10: 如何实现分布式训练的断点续训？**

**A**: 保存完整的训练状态：

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "next_micro_step": next_micro_step,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    ...
}
```

加载时恢复所有状态，从断点继续训练。

---

## 第七部分：算法实习面试考察点

### 7.1 基础知识考察

**考察点**：
1. Transformer 架构理解
2. 自回归生成原理
3. 注意力机制（因果 vs 非因果）
4. 分布式训练基础（DDP、FSDP）

**准备建议**：
- 能画出 Transformer 的完整架构图
- 理解 KV cache 的作用
- 知道 DDP 和 FSDP 的区别

### 7.2 算法理解考察

**考察点**：
1. Speculative Decoding 的原理和数学保证
2. 不同算法的优缺点对比
3. 损失函数设计的考量
4. 训练数据的构建方式

**准备建议**：
- 能解释拒绝采样的数学原理
- 能对比 EAGLE-3、DFlash、DSpark 的差异
- 理解 CE vs L1 vs KL loss 的适用场景

### 7.3 工程能力考察

**考察点**：
1. 代码阅读能力
2. 问题定位能力
3. 优化技巧
4. 系统设计能力

**准备建议**：
- 能解释关键代码段的作用
- 知道常见的训练问题和解决方案
- 理解显存优化的技巧

### 7.4 开放性问题

**常见问题**：
1. 如何提高 draft model 的 acceptance rate？
2. 如何处理不同长度的序列？
3. 如何平衡训练成本和效果？
4. Speculative Decoding 的局限性是什么？

**回答框架**：
1. 问题分析
2. 现有方案
3. 可能的改进方向
4. 实验验证方法

---

## 第八部分：项目介绍话术模板

### 8.1 一分钟版本

> 我参与了 DeepSpec 项目，这是一个用于训练 Speculative Decoding draft 模型的框架。Speculative Decoding 是一种无损加速 LLM 推理的技术，用小的 draft model 快速生成候选 token，再由大的 target model 并行验证。
>
> 我主要负责/研究了 DSpark 算法，它通过并行 block 预测和 Markov Head 实现了高精度的 draft 模型。相比传统的自回归方法，DSpark 在训练时可以同时监督数百个位置，数据效率很高。
>
> 这个项目让我深入理解了 LLM 推理优化、分布式训练、以及算法和工程的结合。

### 8.2 三分钟版本

> 我参与了 DeepSpec 项目，这是一个用于训练 Speculative Decoding draft 模型的全栈框架。Speculative Decoding 是一种无损加速 LLM 推理的技术，核心思想是用小的 draft model 快速生成多个候选 token，然后让大的 target model 并行验证，通过拒绝采样保证输出分布一致。
>
> 项目实现了三种算法：EAGLE-3、DFlash 和 DSpark。EAGLE-3 使用自回归方式，通过 Test-Time Training 逐步精炼 hidden states。DFlash 使用并行 block 预测，通过非因果注意力实现一次生成所有 draft tokens。DSpark 是我们的创新，在 DFlash 基础上增加了 Markov Head 和 Confidence Head。
>
> 我主要负责 DSpark 的实现。Markov Head 通过一个小型矩阵提供二阶校正，提高 block 内 token 间的一致性。Confidence Head 预测每个位置的接受率，可以用于提前终止低置信度的 block。损失函数使用 CE + L1 + BCE 的加权和，其中 L1 loss 直接优化分布匹配。
>
> 训练方面，我们使用 FSDP 进行分布式训练，BF16 混合精度优化显存使用。数据准备流程包括三步：下载数据、用 target model 重新生成回答、预计算 target cache。对于 Qwen3-4B，target cache 大约需要 38 TB 存储。
>
> 这个项目让我深入理解了 LLM 推理优化、分布式训练、算法和工程的结合，以及如何平衡训练成本和效果。

### 8.3 技术深挖准备

**可能的追问**：

1. **Q: DSpark 的 Markov Head 具体怎么工作？**
   > A: Markov Head 有两个小矩阵 W1 和 W2。W1 将前一个 token ID 映射为低维向量，W2 将其投影为 logits 偏置。这个偏置叠加到 draft model 的输出上，提供基于前序 token 的二阶校正。

2. **Q: 为什么 DSpark 用 L1 loss 而不是 CE loss？**
   > A: L1 loss 直接优化 draft 分布和 target 分布之间的距离，而 CE loss 只关注 argmax token。L1 loss 可以更好地校准整个分布，提高 confidence calibration 的准确性。

3. **Q: Confidence Head 如何用于提前终止？**
   > A: 在推理时，如果某个位置的 confidence score 低于阈值，就停止生成后续 draft tokens。这可以避免浪费计算在低置信度的预测上。

4. **Q: 多层特征融合的层数如何选择？**
   > A: 通常选择 3-5 层，均匀分布在 target model 的不同深度。具体选择通过实验验证，需要平衡特征丰富度和计算开销。

---

## 附录：关键代码路径速查

### DeepSpec

```
deepspec/
├── modeling/
│   ├── dspark/
│   │   ├── common.py          # 锚点采样、注意力 mask、置信度头
│   │   ├── markov_head.py     # Markov Head 实现
│   │   ├── loss.py            # DSpark 损失函数
│   │   ├── qwen3/modeling.py  # Qwen3 适配的 DSpark 模型
│   │   └── gemma4/modeling.py # Gemma4 适配
│   └── eagle3/
│       ├── common.py          # Eagle3 公共组件
│       ├── loss.py            # Triton 融合 Soft CE
│       ├── qwen3/modeling.py  # Qwen3 适配的 Eagle3 模型
│       └── gemma4/modeling.py # Gemma4 适配
├── trainer/
│   ├── base_trainer.py        # 训练器基类
│   ├── dspark_trainer.py      # DSpark 训练器
│   └── eagle3_trainer.py      # Eagle3 训练器
├── eval/
│   ├── base_evaluator.py      # 评估器基类
│   ├── dspark/                # DSpark 评估器
│   └── eagle3/                # Eagle3 评估器
└── data/
    ├── target_cache_dataset.py # Target Cache 数据集
    └── parser.py              # 对话模板解析
```

### Speculators

```
src/speculators/
├── config.py                  # 配置类
├── model.py                   # 基础模型类
├── models/
│   ├── eagle3/core.py         # EAGLE-3 实现
│   ├── dflash/core.py         # DFlash 实现
│   ├── peagle/core.py         # P-EAGLE 实现
│   └── mtp/core.py            # MTP 实现
├── train/
│   ├── trainer.py             # 训练器
│   ├── data.py                # 数据集
│   └── optimizers.py          # 优化器
└── data_generation/
    ├── offline.py             # 离线数据生成
    └── vllm_client.py         # vLLM 客户端
```

### 配置文件

```
# DeepSpec 配置示例
config/dspark/dspark_qwen3_4b.py
config/dflash/dflash_qwen3_4b.py
config/eagle3/eagle3_qwen3_4b.py
```

---

## 总结

这份面试准备指南涵盖了：

1. **技术原理**：Speculative Decoding 的数学基础和算法流程
2. **算法对比**：EAGLE-3、DFlash、DSpark 的设计差异和优缺点
3. **实现细节**：多层特征融合、Markov Head、Confidence Head 等关键技术
4. **工程实践**：分布式训练、显存优化、数据准备等实际问题
5. **面试准备**：高频问题、回答框架、项目介绍话术

**核心建议**：
- 理解原理比记忆细节更重要
- 能解释设计决策背后的思考
- 准备好项目介绍的话术
- 了解算法和工程的平衡

祝面试顺利！
