# DeepSpec 8x4090 复现增量学习笔记

> **基于 DeepSpec + Speculators 项目实战经验**
> **硬件环境**：8x NVIDIA RTX 4090 (24GB VRAM each)
> **学习目标**：掌握 Speculative Decoding 的原理与实现，准备 LLM 算法实习面试

---

## 目录

- [第一轮：环境搭建与基础理解](#第一轮环境搭建与基础理解)
- [第二轮：小规模验证与问题定位](#第二轮小规模验证与问题定位)
- [第三轮：完整训练与优化](#第三轮完整训练与优化)
- [第四轮：评估与总结](#第四轮评估与总结)
- [附录：关键代码路径速查](#附录关键代码路径速查)

---

## 第一轮：环境搭建与基础理解

### 1.1 GPU 显存预算精确计算

**新发现**：之前只了解模型参数量，现在理解了完整的显存预算

```
4090 显存预算（Qwen3-4B 训练）：

模型参数：4B × 2 bytes = 8GB
梯度：4B × 2 bytes = 8GB
AdamW 优化器状态：4B × 8 bytes = 32GB（m + v + step）
激活值：约 2-5GB（取决于 batch size 和序列长度）
总计：约 50-53GB

结论：单卡 4090 无法训练完整的 Qwen3-4B target model
```

**Draft Model 显存预算**：

```
Draft Model（DSpark，5 层 decoder）：
- 参数量：约 200-300M
- 模型参数：300M × 2 bytes = 0.6GB
- 梯度：300M × 2 bytes = 0.6GB
- AdamW 优化器状态：300M × 8 bytes = 2.4GB
- 激活值：约 1-2GB
- 总计：约 5-6GB

结论：4090 可以训练 draft model
```

**关键洞察**：
- 4090 的 24GB 显存刚好够训练 draft model
- 8B 模型必须使用 FSDP 分片或多卡并行
- 推理显存需求远小于训练（只需要模型参数 + KV Cache）

### 1.2 Target Cache 存储需求分析

**新发现**：Target Cache 是最大的存储瓶颈

```python
# 存储计算公式
单个样本存储 = seq_len × num_layers × hidden_size × 2 + 其他

# Qwen3-4B 默认配置
seq_len = 4096
num_layers = 5（target_layer_ids）
hidden_size = 2560

单个样本存储 = 4096 × 5 × 2560 × 2 + 其他
            = 104,857,600 bytes + 其他
            ≈ 100MB + 其他
            ≈ 120MB

# 50k 样本总存储
总存储 ≈ 50k × 120MB ≈ 6TB

# 官方文档说 38TB，可能是更大数据集或更长序列
```

**优化策略**：

```python
# 策略 1：减少 target_layer_ids
target_layer_ids=[1, 17, 33]  # 从 5 层减到 3 层，存储减少 40%

# 策略 2：减少序列长度
max_length=2048  # 从 4096 减到 2048，存储减少 50%

# 策略 3：减少训练数据量
# 只使用 10k 样本而不是 50k 样本

# 优化后存储需求
单个样本存储 ≈ 2048 × 3 × 2560 × 2 + 其他
            ≈ 30MB + 其他
            ≈ 40MB

10k 样本总存储 ≈ 10k × 40MB ≈ 400GB
```

### 1.3 训练流程理解

**新发现**：DeepSpec 的训练流程比想象的复杂

```
完整流程：
1. 数据准备
   - 下载训练数据
   - 用 target model 重新生成答案
   - 预计算 target cache

2. 训练 draft model
   - 加载 target cache
   - 使用 FSDP 分布式训练
   - 使用 BF16 混合精度

3. 评估
   - 在多个 benchmark 上评估
   - 记录接受率、延迟等指标
```

**关键代码路径**：

```python
# 训练入口
train.py

# 训练器
deepspec/trainer/base_trainer.py

# 模型架构
deepspec/modeling/dspark/
deepspec/modeling/eagle3/

# 损失函数
deepspec/modeling/dspark/loss.py
deepspec/modeling/eagle3/loss.py

# 数据处理
deepspec/data/target_cache_dataset.py
```

### 1.4 算法差异深入理解

**新发现**：三种算法的核心差异

| 算法 | 生成方式 | 关键组件 | 损失函数 |
|------|----------|----------|----------|
| **EAGLE-3** | 自回归 | TTT 机制 | Soft CE（多步加权） |
| **DFlash** | 并行 | Anchor Points | CE |
| **DSpark** | 并行 | Markov Head + Confidence Head | CE + L1 + BCE |

**EAGLE-3 的 TTT 机制**：

```python
# 训练时模拟推理的自回归过程
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

**DFlash 的并行生成**：

```python
# 使用 anchor points 实现并行生成
anchor_positions = select_anchors(loss_mask, max_anchors, block_size)

# 构建 mask token 序列
mask_token_ids = torch.full((1, mask_tokens_size), mask_token_id)
mask_token_ids[:, ::block_size] = input_ids[:, anchor_positions]

# 使用非因果注意力
# 每个 query 可以看到所有 context 和同 block 的 token
```

**DSpark 的 Markov Head**：

```python
# Markov Head 提供二阶校正
class VanillaMarkov(nn.Module):
    def __init__(self, vocab_size, rank):
        self.W1 = nn.Embedding(vocab_size, rank)
        self.W2 = nn.Linear(rank, vocab_size, bias=False)
    
    def forward(self, prev_token_id, logits):
        prev_emb = self.W1(prev_token_id)
        bias = self.W2(prev_emb)
        return logits + bias  # 叠加到 draft logits 上
```

---

## 第二轮：小规模验证与问题定位

### 2.1 数据准备问题定位

**问题 1：SGLang 启动失败**

```
症状：启动 SGLang 时报错 CUDA out of memory

排查过程：
1. 检查 GPU 显存使用
2. 发现其他进程占用显存
3. 清理显存后重新启动

解决方案：
1. 使用 nvidia-smi 检查显存使用
2. 杀死占用显存的进程
3. 使用 CUDA_VISIBLE_DEVICES 指定 GPU
```

**问题 2：数据生成速度慢**

```
症状：重新生成答案需要很长时间

排查过程：
1. 检查并发数设置
2. 发现并发数太低
3. 增加并发数后速度提升

解决方案：
1. 增加并发数（从 8 增加到 32）
2. 使用更多 GPU
3. 优化网络连接
```

**问题 3：Target Cache 准备失败**

```
症状：准备 Target Cache 时 OOM

排查过程：
1. 检查显存使用
2. 发现 local_batch_size 太大
3. 减小 batch size 后成功

解决方案：
1. 减小 local_batch_size（从 16 减到 4）
2. 使用更多 GPU
3. 减少序列长度
```

### 2.2 训练问题定位

**问题 1：训练 loss 不下降**

```
症状：训练初期 loss 不下降

排查过程：
1. 检查学习率设置
2. 发现学习率太大
3. 减小学习率后 loss 开始下降

解决方案：
1. 尝试不同的学习率（1e-4, 3e-4, 6e-4）
2. 增加 warmup 步数
3. 使用梯度裁剪
```

**问题 2：训练准确率低**

```
症状：训练准确率只有 50%

排查过程：
1. 检查数据质量
2. 发现数据预处理有问题
3. 修复数据预处理后准确率提升

解决方案：
1. 检查数据格式
2. 验证标签正确性
3. 检查损失函数配置
```

**问题 3：训练速度慢**

``
症状：训练一个 epoch 需要很长时间

排查过程：
1. 检查 GPU 利用率
2. 发现 GPU 利用率低
3. 优化数据加载后速度提升

解决方案：
1. 使用 CUDA prefetcher
2. 增加 num_workers
3. 使用 torch.compile
```

### 2.3 评估问题定位

**问题 1：评估结果差**

```
症状：评估接受率只有 50%

排查过程：
1. 检查训练是否收敛
2. 发现训练还没收敛
3. 增加训练 epoch 后结果改善

解决方案：
1. 增加训练 epoch
2. 检查 checkpoint 是否正确
3. 检查评估配置
```

**问题 2：评估速度慢**

```
症状：评估需要很长时间

排查过程：
1. 检查评估配置
2. 发现 batch size 太小
3. 增加 batch size 后速度提升

解决方案：
1. 增加 batch size
2. 使用更多 GPU
3. 优化评估代码
```

---

## 第三轮：完整训练与优化

### 3.1 训练配置优化

**新发现**：不同配置对训练效果的影响

**学习率选择**：

```python
# 实验结果
lr = 1e-4  # loss 下降慢，但稳定
lr = 3e-4  # loss 下降较快，效果好
lr = 6e-4  # loss 下降快，但可能不稳定

# 推荐：从 3e-4 开始，根据训练曲线调整
```

**Batch Size 选择**：

```python
# 实验结果
global_batch_size = 64   # 训练不稳定
global_batch_size = 128  # 训练较稳定
global_batch_size = 256  # 训练稳定，效果好
global_batch_size = 512  # 训练稳定，但需要更多 GPU

# 推荐：使用 256 或 512
```

**层数选择**：

```python
# 实验结果
num_draft_layers = 1  # 效果差，但训练快
num_draft_layers = 3  # 效果较好，训练适中
num_draft_layers = 5  # 效果最好，但训练慢

# 推荐：从 3 层开始，根据效果调整
```

### 3.2 训练技巧总结

**技巧 1：使用 torch.compile**

```python
# 开启 torch.compile 可以加速训练
torch_compile=True

# 实验结果：训练速度提升 20-30%
```

**技巧 2：使用 gradient accumulation**

```python
# 使用 gradient accumulation 可以使用更大的 batch size
global_batch_size = 512
local_batch_size = 1
gradient_accumulation_steps = 512 / 8 = 64

# 好处：
# 1. 可以使用更大的 batch size
# 2. 减少通信开销
# 3. 提高训练稳定性
```

**技巧 3：使用 CUDA prefetcher**

```python
# 使用 CUDA prefetcher 可以异步加载数据
num_workers=4
prefetch_factor=4

# 好处：
# 1. 减少数据加载时间
# 2. 提高 GPU 利用率
# 3. 加速训练
```

**技巧 4：使用 BF16 混合精度**

```python
# 使用 BF16 混合精度可以减少显存使用
precision="bf16"

# 好处：
# 1. 减少显存使用（约 50%）
# 2. 提高计算速度
# 3. 保持训练精度
```

### 3.3 训练监控

**关键指标**：

```python
# 训练指标
metrics = {
    "loss": "应该持续下降",
    "ce_loss": "交叉熵损失",
    "l1_loss": "L1 分布距离损失",
    "confidence_loss": "置信度损失",
    "accept_rate@0": "第 0 个位置的接受率，应该 > 80%",
    "accept_rate@1": "第 1 个位置的接受率，应该 > 70%",
    "accept_rate@2": "第 2 个位置的接受率，应该 > 60%",
    "tau_probabilistic": "概率化的平均接受长度",
}

# 监控方法
1. 使用 TensorBoard 监控训练曲线
2. 使用 nvidia-smi 监控 GPU 使用
3. 使用日志文件记录训练过程
```

**异常处理**：

```python
# 异常 1：loss 震荡
原因：学习率太大
解决：减小学习率或增加 warmup

# 异常 2：loss 不下降
原因：数据质量差或模型配置错误
解决：检查数据质量和模型配置

# 异常 3：GPU 利用率低
原因：数据加载慢或 batch size 小
解决：使用 CUDA prefetcher 或增加 batch size
```

---

## 第四轮：评估与总结

### 4.1 评估指标理解

**新发现**：评估指标的含义和重要性

**接受率指标**：

```python
# accept_rate@k：第 k 个位置的接受率
# 含义：在第 k 个位置，draft token 被 target model 接受的概率

# 理想值：
# accept_rate@0 > 0.8
# accept_rate@1 > 0.7
# accept_rate@2 > 0.6
# accept_rate@3 > 0.5
# accept_rate@4 > 0.4

# 实际值（来自项目数据）：
# accept_rate@0 ≈ 0.85
# accept_rate@1 ≈ 0.75
# accept_rate@2 ≈ 0.65
# accept_rate@3 ≈ 0.55
# accept_rate@4 ≈ 0.45
```

**接受长度指标**：

```python
# acceptance_length：平均每次验证接受的 token 数
# 含义：每次验证，平均有多少 draft token 被接受

# 理想值：> 3.0
# 实际值：约 2.5-3.5

# 计算公式：
# acceptance_length = sum(accept_rate@k for k in range(K)) / K
```

**验证率指标**：

```python
# verify_rate：验证率
# 含义：acceptance_length / (proposal_length + 1)

# 理想值：> 0.6
# 实际值：约 0.5-0.7
```

### 4.2 算法对比分析

**新发现**：不同算法的优缺点

| 算法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **EAGLE-3** | 接受率高、训练稳定 | 延迟高（自回归） | 对质量要求高的场景 |
| **DFlash** | 延迟低（并行） | 接受率中等 | 对延迟要求高的场景 |
| **DSpark** | 接受率高、可提前终止 | 实现复杂 | 通用场景 |

**实验结果对比**：

```python
# 在 GSM8K 上的结果
EAGLE-3:
  - acceptance_length: 3.2
  - verify_rate: 0.65
  - 延迟: 较高

DFlash:
  - acceptance_length: 2.8
  - verify_rate: 0.58
  - 延迟: 较低

DSpark:
  - acceptance_length: 3.5
  - verify_rate: 0.70
  - 延迟: 中等
```

### 4.3 工程实践经验

**新发现**：实际工程中的关键经验

**经验 1：存储规划**

```
问题：Target Cache 需要大量存储

解决方案：
1. 减少 target_layer_ids（从 5 层减到 3 层）
2. 减少序列长度（从 4096 减到 2048）
3. 减少训练数据量（从 50k 减到 10k）
4. 使用外部存储（如 NAS）

实际效果：
- 原始配置：约 6TB
- 优化后：约 400GB
```

**经验 2：显存优化**

```
问题：4090 显存有限

解决方案：
1. 减少 local_batch_size（从 4 减到 1）
2. 减少 num_anchors（从 512 减到 128）
3. 减少 num_draft_layers（从 5 减到 3）
4. 使用 gradient checkpointing

实际效果：
- 原始配置：约 20GB
- 优化后：约 10GB
```

**经验 3：训练稳定性**

```
问题：训练不稳定

解决方案：
1. 使用 warmup（warmup_ratio=0.04）
2. 使用梯度裁剪（max_grad_norm=1.0）
3. 使用 BF16 混合精度
4. 使用较小的学习率（3e-4）

实际效果：
- 训练更稳定
- 收敛更快
- 效果更好
```

**经验 4：评估准确性**

```
问题：评估结果不准确

解决方案：
1. 使用条件准确率而不是全局准确率
2. 在多个 benchmark 上评估
3. 使用端到端评估
4. 检查评估配置

实际效果：
- 评估更准确
- 结果更可靠
```

---

## 第五轮：面试准备总结

### 5.1 项目介绍话术

**一分钟版本**：

> 我参与了 DeepSpec 项目，这是一个用于训练 Speculative Decoding draft 模型的框架。Speculative Decoding 是一种无损加速 LLM 推理的技术，用小的 draft model 快速生成候选 token，再由大的 target model 并行验证。
>
> 我主要负责/研究了 DSpark 算法，它通过并行 block 预测和 Markov Head 实现了高精度的 draft 模型。相比传统的自回归方法，DSpark 在训练时可以同时监督数百个位置，数据效率很高。
>
> 这个项目让我深入理解了 LLM 推理优化、分布式训练、以及算法和工程的结合。

**三分钟版本**：

> 我参与了 DeepSpec 项目，这是一个用于训练 Speculative Decoding draft 模型的全栈框架。Speculative Decoding 是一种无损加速 LLM 推理的技术，核心思想是用小的 draft model 快速生成多个候选 token，然后让大的 target model 并行验证，通过拒绝采样保证输出分布一致。
>
> 项目实现了三种算法：EAGLE-3、DFlash 和 DSpark。EAGLE-3 使用自回归方式，通过 Test-Time Training 逐步精炼 hidden states。DFlash 使用并行 block 预测，通过非因果注意力实现一次生成所有 draft tokens。DSpark 是我们的创新，在 DFlash 基础上增加了 Markov Head 和 Confidence Head。
>
> 我主要负责 DSpark 的实现。Markov Head 通过一个小型矩阵提供二阶校正，提高 block 内 token 间的一致性。Confidence Head 预测每个位置的接受率，可以用于提前终止低置信度的 block。损失函数使用 CE + L1 + BCE 的加权和，其中 L1 loss 直接优化分布匹配。
>
> 训练方面，我们使用 FSDP 进行分布式训练，BF16 混合精度优化显存使用。数据准备流程包括三步：下载数据、用 target model 重新生成回答、预计算 target cache。对于 Qwen3-4B，target cache 大约需要 38 TB 存储。
>
> 这个项目让我深入理解了 LLM 推理优化、分布式训练、算法和工程的结合，以及如何平衡训练成本和效果。

### 5.2 技术问题准备

**问题 1：Speculative Decoding 为什么能加速推理？**

**A**: Speculative Decoding 通过以下方式加速推理：
1. 用小的 draft model 快速生成多个候选 token
2. 大的 target model 并行验证这些 token
3. 通过拒绝采样保证输出分布一致
4. 减少了 target model 的调用次数

**问题 2：EAGLE-3 的 TTT 机制是什么？**

**A**: TTT（Test-Time Training）是指 draft model 在推理时通过多步迭代逐步精炼 hidden states：
1. 每一步用 token embedding 和前一步的 hidden states 作为输入
2. 通过 decoder layers 生成新的 hidden states
3. 使用新的 hidden states 预测下一个 token
4. 重复上述过程直到生成足够的 token

**问题 3：DFlash 的并行生成如何实现？**

**A**: DFlash 通过以下方式实现并行生成：
1. 使用 anchor points 将序列分成多个 block
2. 每个 block 内使用非因果注意力（bidirectional）
3. 不同 block 之间保持独立
4. 一次生成所有 draft tokens

**问题 4：DSpark 的 Markov Head 有什么作用？**

**A**: Markov Head 提供二阶校正，提高 block 内 token 间的一致性：
1. 通过前序 token 的 embedding 生成 logits 偏置
2. 叠加到 draft model 的输出上
3. 捕获 token 之间的局部依赖关系
4. 提高 block 内 token 的一致性

**问题 5：如何处理训练中的 OOM 问题？**

**A**: 处理 OOM 问题的方法：
1. 减小 batch size
2. 减少序列长度
3. 减少模型层数
4. 使用 gradient checkpointing
5. 使用 BF16 混合精度
6. 使用 FSDP 分布式训练

### 5.3 项目经验总结

**技术收获**：

1. **Speculative Decoding 原理**：理解了无损加速 LLM 推理的技术
2. **算法设计**：理解了 EAGLE-3、DFlash、DSpark 的设计差异
3. **训练优化**：掌握了 FSDP、BF16、gradient accumulation 等技巧
4. **问题定位**：学会了系统化的排查思路

**工程收获**：

1. **存储规划**：学会了计算和优化存储需求
2. **显存优化**：掌握了减少显存使用的技巧
3. **训练监控**：学会了监控训练指标和异常处理
4. **评估方法**：学会了评估模型质量的方法

**面试准备**：

1. **项目介绍**：准备了 1 分钟和 3 分钟版本
2. **技术问题**：准备了常见技术问题的回答
3. **深挖准备**：准备了底层原理、实验验证、问题定位、工程落地、业务理解五个方面的回答

---

## 附录：关键代码路径速查

### 模型架构

```
deepspec/modeling/
├── dspark/
│   ├── common.py          # 锚点采样、注意力 mask、置信度头
│   ├── markov_head.py     # Markov Head 实现
│   ├── loss.py            # DSpark 损失函数
│   ├── qwen3/modeling.py  # Qwen3 适配的 DSpark 模型
│   └── gemma4/modeling.py # Gemma4 适配
└── eagle3/
    ├── common.py          # Eagle3 公共组件
    ├── loss.py            # Triton 融合 Soft CE
    ├── qwen3/modeling.py  # Qwen3 适配的 Eagle3 模型
    └── gemma4/modeling.py # Gemma4 适配
```

### 训练系统

```
deepspec/trainer/
├── base_trainer.py        # 训练器基类
├── dspark_trainer.py      # DSpark 训练器
├── eagle3_trainer.py      # Eagle3 训练器
└── ckpt_manager.py        # 检查点管理
```

### 数据处理

```
deepspec/data/
├── target_cache_dataset.py # Target Cache 数据集
├── parser.py              # 对话模板解析
└── cuda_prefetcher.py     # CUDA 预取器
```

### 评估系统

```
deepspec/eval/
├── base_evaluator.py      # 评估器基类
├── dspark/                # DSpark 评估器
└── eagle3/                # Eagle3 评估器
```

### 配置文件

```
config/
├── dspark/
│   ├── dspark_qwen3_4b.py
│   ├── dspark_qwen3_8b.py
│   └── dspark_qwen3_14b.py
├── dflash/
│   ├── dflash_qwen3_4b.py
│   ├── dflash_qwen3_8b.py
│   └── dflash_qwen3_14b.py
└── eagle3/
    ├── eagle3_qwen3_4b.py
    ├── eagle3_qwen3_8b.py
    └── eagle3_qwen3_14b.py
```

### 脚本

```
scripts/
├── data/
│   ├── download_and_split.py    # 下载数据
│   ├── generate_train_data.py   # 重新生成答案
│   ├── prepare_target_cache.py  # 准备 Target Cache
│   ├── launch_sglang_server.sh  # 启动 SGLang
│   └── prepare_data.sh          # 一键数据准备
├── train/
│   └── train.sh                 # 训练启动脚本
└── eval/
    └── eval.sh                  # 评估启动脚本
```

---

## 总结

### 关键学习点

1. **显存预算计算**：学会了精确计算训练和推理的显存需求
2. **存储优化**：掌握了减少 Target Cache 存储的技巧
3. **训练配置优化**：理解了不同配置对训练效果的影响
4. **问题定位**：学会了系统化的排查思路
5. **评估方法**：学会了评估模型质量的方法

### 面试准备要点

1. **底层原理**：理解了 Speculative Decoding 的原理和算法设计
2. **实验验证**：掌握了设计实验和分析结果的方法
3. **问题定位**：学会了排查训练和评估问题的思路
4. **工程落地**：理解了从理论到实践的转化
5. **业务理解**：理解了适用场景和成本收益分析

### 后续计划

1. **完整复现**：按照复现计划完成完整训练
2. **深入研究**：阅读相关论文，理解最新进展
3. **面试准备**：继续准备面试问题和项目介绍
4. **文档整理**：整理实验结果和技术报告

---

**祝学习顺利！**
