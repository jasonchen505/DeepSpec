# DeepSpec 8x4090 完整复现计划

> **硬件环境**：8x NVIDIA RTX 4090 (24GB VRAM each)
> **目标**：完整复现 DeepSpec 项目的全流程（数据准备 → 训练 → 评估）
> **适用对象**：找 LLM 算法实习的 MS 在读学生

---

## 目录

- [第一部分：资源评估与约束分析](#第一部分资源评估与约束分析)
- [第二部分：分阶段复现计划](#第二部分分阶段复现计划)
- [第三部分：详细执行步骤](#第三部分详细执行步骤)
- [第四部分：关键配置调整](#第四部分关键配置调整)
- [第五部分：学习路径规划](#第五部分学习路径规划)
- [第六部分：风险与应急预案](#第六部分风险与应急预案)
- [附录：命令速查表](#附录命令速查表)

---

## 第一部分：资源评估与约束分析

### 1.1 硬件资源评估

**4090 显存分析**：

```
4090 显存：24GB GDDR6X

Qwen3-4B 模型显存需求（bf16）：
- 模型参数：4B × 2 bytes = 8GB
- 梯度：4B × 2 bytes = 8GB
- AdamW 优化器状态：4B × 8 bytes = 32GB（m + v + step）
- 激活值：约 2-5GB（取决于 batch size 和序列长度）
- 总计：约 50-53GB（单卡无法容纳完整训练）

Draft Model 显存需求：
- 参数量：约 100-300M（取决于配置）
- 显存：约 1-3GB（bf16）
- 梯度 + 优化器：约 4-12GB
- 总计：约 5-15GB
```

**关键发现**：
- ❌ **单卡 4090 无法训练完整的 Qwen3-4B target model**
- ✅ **可以训练 draft model**（参数量小）
- ✅ **可以运行推理**（只需要模型参数 + KV Cache）

### 1.2 存储需求分析

**Target Cache 存储需求**：

```python
# 计算公式
# 单个样本存储 = seq_len × (num_layers × hidden_size × 2 + 其他)
# 对于 Qwen3-4B：
# - hidden_size = 2560
# - num_layers = 5（默认 target_layer_ids）
# - seq_len = 4096

单个样本存储 ≈ 4096 × 5 × 2560 × 2 bytes + 其他
             ≈ 100MB + 其他
             ≈ 120MB

# 默认训练数据集大小约 50k 样本
总存储 ≈ 50k × 120MB ≈ 6TB

# 官方文档说 38TB，可能是更大数据集或更长序列
```

**4090 环境存储策略**：
- 使用 **少量数据** 进行验证（1k-5k 样本）
- 减少 `target_layer_ids` 数量（从 5 层减到 2-3 层）
- 使用 **外部存储**（如有 NAS 或云存储）

### 1.3 时间成本估算

| 阶段 | 任务 | 预计时间 | 瓶颈 |
|------|------|----------|------|
| 环境搭建 | 安装依赖、下载模型 | 2-4 小时 | 网络带宽 |
| 数据准备 Step 1 | 下载数据集 | 30 分钟 | HuggingFace 访问 |
| 数据准备 Step 2 | 重新生成答案 | 4-8 小时 | GPU 推理速度 |
| 数据准备 Step 3 | 准备 Target Cache | 8-24 小时 | GPU 计算 + 磁盘 I/O |
| 训练 | 训练 Draft Model | 12-48 小时 | GPU 计算 |
| 评估 | 评估模型质量 | 2-4 小时 | GPU 推理 |
| **总计** | | **2-5 天** | |

---

## 第二部分：分阶段复现计划

### 阶段 0：环境搭建与代码理解（Day 1）

**目标**：
- 搭建开发环境
- 理解项目架构
- 阅读关键代码

**任务清单**：

```markdown
## 环境搭建
- [ ] 创建 Python 虚拟环境
- [ ] 安装 PyTorch 2.9.1（CUDA 12.x）
- [ ] 安装项目依赖
- [ ] 下载 Qwen3-4B 模型
- [ ] 验证 GPU 环境

## 代码阅读
- [ ] 阅读 README.md
- [ ] 理解项目结构
- [ ] 阅读 config 文件
- [ ] 理解训练流程
- [ ] 理解评估流程
```

**执行命令**：

```bash
# 1. 创建虚拟环境
conda create -n deepspec python=3.10 -y
conda activate deepspec

# 2. 安装 PyTorch（根据 CUDA 版本选择）
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装项目依赖
cd /data/home/yizhou/DeepSpec
pip install -r requirements.txt

# 4. 下载 Qwen3-4B 模型
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B')"

# 5. 验证 GPU
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')"
```

**学习重点**：
- 理解 Speculative Decoding 的原理
- 理解三种算法的差异（EAGLE-3、DFlash、DSpark）
- 理解训练数据格式

### 阶段 1：小规模验证（Day 2-3）

**目标**：
- 使用 **100-500 样本** 验证流程
- 确认代码可以正常运行
- 估算完整复现的时间和资源

**任务清单**：

```markdown
## 数据准备
- [ ] 下载少量训练数据（100 样本）
- [ ] 重新生成答案（使用 SGLang）
- [ ] 准备小规模 Target Cache

## 训练验证
- [ ] 使用小规模数据训练
- [ ] 验证训练流程
- [ ] 监控 GPU 使用情况

## 评估验证
- [ ] 评估训练结果
- [ ] 记录基线指标
```

**关键配置调整**：

```python
# 创建小规模配置文件
# config/experiments/small_scale_4090.py

import os

project_name = "deepspec_4090_test"
exp_name = "small_scale_test"

model = dict(
    target_model_name_or_path="Qwen/Qwen3-4B",
    block_size=7,
    num_draft_layers=3,  # 减少层数
    target_layer_ids=[1, 17, 33],  # 减少到 3 层
    mask_token_id=151669,
    num_anchors=128,  # 减少 anchor 数量
    markov_rank=128,  # 减少 rank
    confidence_head_alpha=1.0,
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=64,  # 减小 global batch size
    num_train_epochs=3,  # 减少 epoch
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=False,  # 关闭 compile 加速调试
)

logging = dict(
    logging_steps=5,
    checkpointing_steps=500,
)

data = dict(
    target_cache_path=None,
    chat_template="qwen",
    max_length=2048,  # 减少序列长度
    num_workers=2,
)
```

**执行命令**：

```bash
# 1. 准备小规模数据
mkdir -p train_datasets/small_scale
head -100 train_datasets/perfectblend_train.jsonl > train_datasets/small_scale/train_100.jsonl

# 2. 启动 SGLang（4 个 worker）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/data/launch_sglang_server.sh

# 3. 重新生成答案（小规模）
python scripts/data/generate_train_data.py \
    --model Qwen/Qwen3-4B \
    --server-address 127.0.0.1:30000 127.0.0.1:30001 127.0.0.1:30002 127.0.0.1:30003 \
    --concurrency 8 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --min-p 0 \
    --max-tokens 2048 \
    --disable-thinking \
    --input-file-path train_datasets/small_scale/train_100.jsonl \
    --output-file-path train_datasets/small_scale/train_100_regen.jsonl

# 4. 准备小规模 Target Cache
python scripts/data/prepare_target_cache.py \
    --config config/experiments/small_scale_4090.py \
    --train-data-path train_datasets/small_scale/train_100_regen.jsonl \
    --output-dir ~/.cache/deepspec/small_scale_target_cache \
    --local-batch-size 4

# 5. 训练（使用 4 卡）
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    --config config/experiments/small_scale_4090.py \
    --opts "data.target_cache_path=~/.cache/deepspec/small_scale_target_cache"

# 6. 评估
CUDA_VISIBLE_DEVICES=0,1 python eval.py \
    --target_name_or_path Qwen/Qwen3-4B \
    --draft_name_or_path ~/checkpoints/deepspec_4090_test/small_scale_test/step_latest
```

### 阶段 2：中等规模验证（Day 4-5）

**目标**：
- 使用 **1k-5k 样本** 验证
- 优化训练配置
- 估算完整训练时间

**任务清单**：

```markdown
## 数据准备
- [ ] 下载中等规模数据（1k-5k 样本）
- [ ] 重新生成答案
- [ ] 准备中等规模 Target Cache

## 训练优化
- [ ] 调整学习率
- [ ] 调整 batch size
- [ ] 监控训练曲线

## 结果分析
- [ ] 分析训练指标
- [ ] 评估模型质量
- [ ] 记录最佳配置
```

**配置优化**：

```python
# config/experiments/medium_scale_4090.py

model = dict(
    target_model_name_or_path="Qwen/Qwen3-4B",
    block_size=7,
    num_draft_layers=5,  # 恢复到 5 层
    target_layer_ids=[1, 9, 17, 25, 33],  # 恢复到 5 层
    mask_token_id=151669,
    num_anchors=256,  # 增加到 256
    markov_rank=256,
    confidence_head_alpha=1.0,
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=256,  # 增加到 256
    num_train_epochs=5,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=True,  # 开启 compile
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=1000,
)

data = dict(
    target_cache_path=None,
    chat_template="qwen",
    max_length=4096,  # 恢复到 4096
    num_workers=4,
)
```

### 阶段 3：完整训练（Day 6-10）

**目标**：
- 使用 **10k-50k 样本** 完整训练
- 获得高质量的 draft model
- 在多个 benchmark 上评估

**任务清单**：

```markdown
## 数据准备
- [ ] 下载完整训练数据（10k-50k 样本）
- [ ] 重新生成答案（可能需要 8-24 小时）
- [ ] 准备完整 Target Cache（可能需要 200GB-1TB 存储）

## 完整训练
- [ ] 配置训练参数
- [ ] 启动训练（可能需要 24-48 小时）
- [ ] 监控训练过程

## 评估与分析
- [ ] 在多个 benchmark 上评估
- [ ] 分析接受率
- [ ] 对比不同算法
```

**存储优化策略**：

```bash
# 1. 使用外部存储（如有 NAS）
export TARGET_CACHE_DIR=/mnt/nas/deepspec_cache

# 2. 或使用本地大容量磁盘
export TARGET_CACHE_DIR=/data/deepspec_cache

# 3. 减少 target_layer_ids（从 5 层减到 3 层）
# 在配置文件中修改：
# target_layer_ids=[1, 17, 33]  # 只保留 3 层
```

**训练命令**：

```bash
# 使用全部 8 卡训练
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TARGET_CACHE_DIR=/data/deepspec_cache

python train.py \
    --config config/dspark/dspark_qwen3_4b.py \
    --opts "data.target_cache_path=${TARGET_CACHE_DIR}" \
    --opts "train.global_batch_size=512" \
    --opts "train.num_train_epochs=10"
```

### 阶段 4：评估与部署（Day 11-12）

**目标**：
- 评估训练结果
- 对比不同算法
- 准备面试材料

**任务清单**：

```markdown
## 评估
- [ ] 在 GSM8K 上评估
- [ ] 在 MATH 上评估
- [ ] 在 HumanEval 上评估
- [ ] 记录详细指标

## 对比分析
- [ ] 对比 EAGLE-3、DFlash、DSpark
- [ ] 分析各算法优缺点
- [ ] 准备面试话术

## 文档整理
- [ ] 整理实验结果
- [ ] 编写技术报告
- [ ] 准备项目介绍
```

**评估命令**：

```bash
# 评估 DSpark
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py \
    --target_name_or_path Qwen/Qwen3-4B \
    --draft_name_or_path ~/checkpoints/deepspec/dspark_block8_qwen3_4b/step_latest

# 评估 DFlash
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py \
    --target_name_or_path Qwen/Qwen3-4B \
    --draft_name_or_path ~/checkpoints/deepspec/dflash_block8_qwen3_4b/step_latest

# 评估 EAGLE-3
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py \
    --target_name_or_path Qwen/Qwen3-4B \
    --draft_name_or_path ~/checkpoints/deepspec/eagle3_ttt7_qwen3_4b/step_latest
```

---

## 第三部分：详细执行步骤

### 3.1 环境搭建详细步骤

**Step 1：检查 GPU 环境**

```bash
# 检查 GPU 数量和显存
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 预期输出：
# 8x RTX 4090
# 每卡 24GB 显存
# CUDA 12.x
```

**Step 2：创建虚拟环境**

```bash
# 使用 conda
conda create -n deepspec python=3.10 -y
conda activate deepspec

# 或使用 venv
python -m venv ~/deepspec_env
source ~/deepspec_env/bin/activate
```

**Step 3：安装 PyTorch**

```bash
# 根据 CUDA 版本选择
# CUDA 12.1
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu124

# 验证安装
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

**Step 4：安装项目依赖**

```bash
cd /data/home/yizhou/DeepSpec
pip install -r requirements.txt

# 安装 SGLang（用于数据准备）
pip install "sglang[all]"
```

**Step 5：下载模型**

```bash
# 下载 Qwen3-4B
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
print('Model downloaded successfully')
"

# 检查模型大小
du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B*
```

### 3.2 数据准备详细步骤

**Step 1：下载训练数据**

```bash
# 创建数据目录
mkdir -p train_datasets
mkdir -p eval_datasets

# 下载数据集
python scripts/data/download_and_split.py \
    --dataset-name mlabonne/open-perfectblend \
    --test-size 0.05 \
    --train-output-path train_datasets/perfectblend_train.jsonl \
    --test-output-dir eval_datasets \
    --skip-existing

# 检查数据量
wc -l train_datasets/perfectblend_train.jsonl
```

**Step 2：启动 SGLang 服务器**

```bash
# 修改启动脚本以使用 8 卡
# 编辑 scripts/data/launch_sglang_server.sh
# 确保 num_workers=8

# 启动服务器
bash scripts/data/launch_sglang_server.sh

# 等待服务器启动（检查日志）
tail -f logs/sglang_qwen3_4b/worker_*.log

# 验证服务器
curl http://127.0.0.1:30000/v1/models
```

**Step 3：重新生成答案**

```bash
# 生成训练数据
python scripts/data/generate_train_data.py \
    --model Qwen/Qwen3-4B \
    --server-address \
        127.0.0.1:30000 \
        127.0.0.1:30001 \
        127.0.0.1:30002 \
        127.0.0.1:30003 \
        127.0.0.1:30004 \
        127.0.0.1:30005 \
        127.0.0.1:30006 \
        127.0.0.1:30007 \
    --concurrency 32 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --min-p 0 \
    --max-tokens 4096 \
    --disable-thinking \
    --resume \
    --input-file-path train_datasets/perfectblend_train.jsonl \
    --output-file-path train_datasets/qwen3_4b/perfectblend_train_regen.jsonl

# 检查生成结果
wc -l train_datasets/qwen3_4b/perfectblend_train_regen.jsonl
```

**Step 4：准备 Target Cache**

```bash
# 停止 SGLang 释放 GPU
pkill -f sglang

# 准备 Target Cache
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=1

python scripts/data/prepare_target_cache.py \
    --config config/dspark/dspark_qwen3_4b.py \
    --train-data-path train_datasets/qwen3_4b/perfectblend_train_regen.jsonl \
    --output-dir ~/.cache/deepspec/qwen3_4b_target_cache \
    --local-batch-size 8

# 检查缓存大小
du -sh ~/.cache/deepspec/qwen3_4b_target_cache
```

### 3.3 训练详细步骤

**Step 1：配置训练参数**

```bash
# 检查配置文件
cat config/dspark/dspark_qwen3_4b.py

# 关键参数：
# - local_batch_size: 每 GPU 的 batch size（默认 1）
# - global_batch_size: 全局 batch size（默认 512）
# - num_train_epochs: 训练轮数（默认 10）
# - lr: 学习率（默认 6e-4）
```

**Step 2：启动训练**

```bash
# 使用全部 8 卡
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TARGET_CACHE_DIR=~/.cache/deepspec/qwen3_4b_target_cache

python train.py \
    --config config/dspark/dspark_qwen3_4b.py \
    --opts "data.target_cache_path=${TARGET_CACHE_DIR}"

# 监控训练
# 1. 查看 GPU 使用
watch -n 1 nvidia-smi

# 2. 查看 TensorBoard
tensorboard --logdir ~/tensorboard --port 6006

# 3. 查看训练日志
tail -f ~/checkpoints/deepspec/dspark_block8_qwen3_4b/train.log
```

**Step 3：监控训练指标**

```python
# 关键指标
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
```

### 3.4 评估详细步骤

**Step 1：评估训练结果**

```bash
# 评估 DSpark
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py \
    --target_name_or_path Qwen/Qwen3-4B \
    --draft_name_or_path ~/checkpoints/deepspec/dspark_block8_qwen3_4b/step_latest

# 预期输出：
# - gsm8k: acceptance_length, verify_rate
# - math500: acceptance_length, verify_rate
# - humaneval: acceptance_length, verify_rate
# - 等等
```

**Step 2：分析评估结果**

```python
# 评估指标解释
metrics = {
    "acceptance_length": "平均每次验证接受的 token 数，越高越好",
    "draft_tokens_per_proposal": "平均每次 proposal 的 draft token 数",
    "verify_rate": "acceptance_length / (proposal_length + 1)",
    "accept_rate@k": "第 k 个位置的接受率",
}

# 理想结果：
# - acceptance_length > 3.0
# - verify_rate > 0.6
# - accept_rate@0 > 0.8
# - accept_rate@1 > 0.7
# - accept_rate@2 > 0.6
```

---

## 第四部分：关键配置调整

### 4.1 显存优化配置

**问题**：4090 显存有限，需要优化配置

**解决方案**：

```python
# config/experiments/optimized_4090.py

model = dict(
    target_model_name_or_path="Qwen/Qwen3-4B",
    block_size=7,
    num_draft_layers=3,  # 减少到 3 层（减少显存）
    target_layer_ids=[1, 17, 33],  # 减少到 3 层（减少存储）
    mask_token_id=151669,
    num_anchors=128,  # 减少到 128（减少显存）
    markov_rank=128,  # 减少到 128（减少显存）
    confidence_head_alpha=1.0,
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,  # 保持 1（避免 OOM）
    global_batch_size=128,  # 减少到 128（减少显存）
    num_train_epochs=10,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=True,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=1000,
)

data = dict(
    target_cache_path=None,
    chat_template="qwen",
    max_length=2048,  # 减少到 2048（减少显存）
    num_workers=2,
)
```

### 4.2 存储优化配置

**问题**：Target Cache 需要大量存储

**解决方案**：

```python
# 减少 target_layer_ids
target_layer_ids=[1, 17, 33]  # 从 5 层减到 3 层，存储减少 40%

# 减少序列长度
max_length=2048  # 从 4096 减到 2048，存储减少 50%

# 减少训练数据量
# 只使用 10k 样本而不是 50k 样本
```

**存储计算**：

```python
# 优化后的存储需求
# 原始配置：5 层 × 4096 序列长度 × 50k 样本 ≈ 6TB
# 优化后：3 层 × 2048 序列长度 × 10k 样本 ≈ 360GB

# 计算公式
单个样本存储 = seq_len × num_layers × hidden_size × 2 + 其他
            = 2048 × 3 × 2560 × 2 + 其他
            ≈ 30MB + 其他
            ≈ 40MB

10k 样本总存储 ≈ 10k × 40MB ≈ 400GB
```

### 4.3 训练速度优化

**问题**：训练速度慢

**解决方案**：

```python
# 1. 使用 torch.compile
torch_compile=True

# 2. 使用 gradient accumulation
global_batch_size=512
local_batch_size=1
# gradient_accumulation_steps = 512 / 8 = 64

# 3. 使用 CUDA prefetcher
num_workers=4
prefetch_factor=4

# 4. 使用 Multipack Batch Sampler（如果支持）
# 减少 padding 浪费
```

---

## 第五部分：学习路径规划

### 5.1 第一周：基础理解

**目标**：理解项目架构和核心概念

**学习内容**：

```markdown
## Day 1-2：项目概述
- 阅读 README.md
- 理解 Speculative Decoding 原理
- 理解三种算法的差异

## Day 3-4：代码结构
- 阅读 config 文件
- 理解训练流程
- 理解评估流程

## Day 5-7：动手实践
- 搭建环境
- 运行小规模测试
- 理解数据格式
```

**关键代码阅读**：

```python
# 1. 理解配置文件
config/dspark/dspark_qwen3_4b.py

# 2. 理解训练入口
train.py

# 3. 理解评估入口
eval.py

# 4. 理解模型架构
deepspec/modeling/dspark/
deepspec/modeling/eagle3/

# 5. 理解训练器
deepspec/trainer/
```

### 5.2 第二周：深入理解

**目标**：理解算法细节和实现

**学习内容**：

```markdown
## Day 8-9：算法细节
- 理解 EAGLE-3 的 TTT 机制
- 理解 DFlash 的并行生成
- 理解 DSpark 的 Markov Head

## Day 10-11：损失函数
- 理解 CE Loss
- 理解 L1 Loss
- 理解 Confidence Loss

## Day 12-14：训练技巧
- 理解 FSDP 分布式训练
- 理解 BF16 混合精度
- 理解梯度累积
```

**关键代码阅读**：

```python
# 1. 理解 DSpark 损失函数
deepspec/modeling/dspark/loss.py

# 2. 理解 Eagle3 损失函数
deepspec/modeling/eagle3/loss.py

# 3. 理解 Markov Head
deepspec/modeling/dspark/markov_head.py

# 4. 理解注意力掩码
deepspec/modeling/dspark/common.py

# 5. 理解训练器
deepspec/trainer/base_trainer.py
```

### 5.3 第三周：实践验证

**目标**：完整复现项目

**学习内容**：

```markdown
## Day 15-17：数据准备
- 下载训练数据
- 重新生成答案
- 准备 Target Cache

## Day 18-20：训练
- 配置训练参数
- 启动训练
- 监控训练过程

## Day 21：评估
- 评估训练结果
- 分析指标
- 对比不同算法
```

### 5.4 第四周：总结提升

**目标**：总结学习成果

**学习内容**：

```markdown
## Day 22-23：结果分析
- 分析训练曲线
- 分析评估结果
- 总结最佳配置

## Day 24-25：面试准备
- 整理项目介绍
- 准备技术问题
- 模拟面试

## Day 26-28：文档整理
- 编写技术报告
- 整理代码注释
- 准备演示材料
```

---

## 第六部分：风险与应急预案

### 6.1 风险识别

| 风险 | 概率 | 影响 | 应急预案 |
|------|------|------|----------|
| 显存不足 | 高 | 无法训练 | 减少 batch size、减少层数、使用 gradient checkpointing |
| 存储不足 | 高 | 无法准备 Target Cache | 减少数据量、减少层数、使用外部存储 |
| 训练时间过长 | 中 | 延期 | 减少 epoch、减少数据量、使用更多 GPU |
| 代码错误 | 中 | 无法运行 | 检查日志、搜索 issue、寻求帮助 |
| 网络问题 | 低 | 无法下载模型 | 使用代理、离线安装 |

### 6.2 应急预案

**显存不足**：

```bash
# 方案 1：减少 batch size
--opts "train.local_batch_size=1"
--opts "train.global_batch_size=64"

# 方案 2：减少层数
--opts "model.num_draft_layers=3"
--opts "model.target_layer_ids=[1,17,33]"

# 方案 3：减少序列长度
--opts "data.max_length=2048"

# 方案 4：使用 gradient checkpointing
# 在配置文件中添加
model.gradient_checkpointing = True
```

**存储不足**：

```bash
# 方案 1：减少数据量
head -5000 train_datasets/perfectblend_train.jsonl > train_datasets/small_train.jsonl

# 方案 2：减少层数
--opts "model.target_layer_ids=[1,17,33]"  # 从 5 层减到 3 层

# 方案 3：使用外部存储
export TARGET_CACHE_DIR=/mnt/nas/deepspec_cache

# 方案 4：使用压缩
# 在配置文件中启用压缩
data.compress_cache = True
```

**训练时间过长**：

```bash
# 方案 1：减少 epoch
--opts "train.num_train_epochs=3"

# 方案 2：减少数据量
# 只使用 5k 样本

# 方案 3：使用更多 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 方案 4：使用更大的 batch size
--opts "train.global_batch_size=1024"
```

### 6.3 常见问题解决

**问题 1：CUDA OOM**

```bash
# 解决方案
1. 减少 local_batch_size
2. 减少 max_length
3. 减少 num_anchors
4. 使用 gradient checkpointing
5. 使用更小的模型（如 Qwen3-1.5B）
```

**问题 2：训练 loss 不下降**

```bash
# 解决方案
1. 检查学习率（尝试 1e-4, 3e-4, 6e-4）
2. 检查数据质量
3. 检查梯度范数
4. 检查损失函数配置
```

**问题 3：评估结果差**

```bash
# 解决方案
1. 检查训练是否收敛
2. 检查 checkpoint 是否正确
3. 检查评估配置
4. 增加训练数据量
```

---

## 附录：命令速查表

### 环境搭建

```bash
# 创建环境
conda create -n deepspec python=3.10 -y
conda activate deepspec

# 安装依赖
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "sglang[all]"

# 验证环境
python -c "import torch; print(torch.cuda.device_count())"
```

### 数据准备

```bash
# 下载数据
python scripts/data/download_and_split.py \
    --dataset-name mlabonne/open-perfectblend \
    --test-size 0.05 \
    --train-output-path train_datasets/perfectblend_train.jsonl \
    --test-output-dir eval_datasets

# 启动 SGLang
bash scripts/data/launch_sglang_server.sh

# 重新生成答案
python scripts/data/generate_train_data.py \
    --model Qwen/Qwen3-4B \
    --server-address 127.0.0.1:30000 127.0.0.1:30001 127.0.0.1:30002 127.0.0.1:30003 \
    --concurrency 32 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --min-p 0 \
    --max-tokens 4096 \
    --disable-thinking \
    --input-file-path train_datasets/perfectblend_train.jsonl \
    --output-file-path train_datasets/qwen3_4b/perfectblend_train_regen.jsonl

# 准备 Target Cache
python scripts/data/prepare_target_cache.py \
    --config config/dspark/dspark_qwen3_4b.py \
    --train-data-path train_datasets/qwen3_4b/perfectblend_train_regen.jsonl \
    --output-dir ~/.cache/deepspec/qwen3_4b_target_cache \
    --local-batch-size 8
```

### 训练

```bash
# 启动训练
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train.py \
    --config config/dspark/dspark_qwen3_4b.py \
    --opts "data.target_cache_path=~/.cache/deepspec/qwen3_4b_target_cache"

# 监控训练
watch -n 1 nvidia-smi
tensorboard --logdir ~/tensorboard --port 6006
```

### 评估

```bash
# 评估模型
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py \
    --target_name_or_path Qwen/Qwen3-4B \
    --draft_name_or_path ~/checkpoints/deepspec/dspark_block8_qwen3_4b/step_latest
```

### 配置调整

```bash
# 调整 batch size
--opts "train.local_batch_size=1"
--opts "train.global_batch_size=128"

# 调整层数
--opts "model.num_draft_layers=3"
--opts "model.target_layer_ids=[1,17,33]"

# 调整序列长度
--opts "data.max_length=2048"

# 调整学习率
--opts "train.lr=3e-4"
```

---

## 总结

### 关键成功因素

1. **存储规划**：确保有足够的存储空间（至少 500GB-1TB）
2. **显存优化**：使用较小的 batch size 和层数
3. **时间规划**：预留充足的时间（至少 2 周）
4. **风险预案**：准备好应急方案

### 预期成果

1. **完整复现**：成功训练出 draft model
2. **性能验证**：在多个 benchmark 上验证效果
3. **深入理解**：掌握 Speculative Decoding 的原理和实现
4. **面试准备**：准备好项目介绍和技术问题

### 时间安排

| 阶段 | 时间 | 任务 |
|------|------|------|
| 阶段 0 | Day 1 | 环境搭建 |
| 阶段 1 | Day 2-3 | 小规模验证 |
| 阶段 2 | Day 4-5 | 中等规模验证 |
| 阶段 3 | Day 6-10 | 完整训练 |
| 阶段 4 | Day 11-12 | 评估与总结 |
| **总计** | **约 2 周** | |

---

**祝复现顺利！**
