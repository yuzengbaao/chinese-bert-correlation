# 训练指南 | Training Guide

本文档详细介绍如何使用本项目进行BERT模型训练和实验。

This document provides a comprehensive guide on how to train BERT models using this project.

---

## 📋 目录 | Table of Contents

- [训练流程](#训练流程--training-workflow)
- [数据准备](#数据准备--data-preparation)
- [配置参数](#配置参数--configuration)
- [训练监控](#训练监控--training-monitoring)
- [故障排除](#故障排除--troubleshooting)
- [最佳实践](#最佳实践--best-practices)

---

## 🚀 训练流程 | Training Workflow

### 1. 环境准备

```bash
# 1. 克隆项目
git clone https://github.com/yuzengbaao/chinese-bert-correlation.git
cd chinese-bert-correlation

# 2. 安装依赖
pip install -r requirements.txt

# 3. 验证环境
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 2. 数据准备

```bash
# 准备数据集（参考 docs/DATASET.md）
python examples/prepare_dataset.py

# 验证数据集格式
python -c "
import json
with open('large_wikipedia_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(f'句子数: {len(data[\"sentences\"])}')
    print(f'词汇量: {len(data[\"vocabulary\"])}')
"
```

### 3. 开始训练

```bash
# 基础训练（默认配置）
python main_train.py

# 自定义配置训练
python main_train.py --steps 100000 --batch-size 16 --learning-rate 5e-5
```

### 4. 训练完成后分析

```bash
# 分析训练结果
python analyze_100k.py

# 生成可视化
python visualize_results.py

# 查看自定义分析
python examples/load_and_analyze.py
```

---

## 📊 数据准备 | Data Preparation

### 数据集格式要求

项目使用JSON格式的数据集，结构如下：

```json
{
  "sentences": [
    "这是第一个句子。",
    "这是第二个句子。",
    "..."
  ],
  "vocabulary": {
    "这": 1523,
    "是": 1342,
    "第一": 234,
    "...": 0
  }
}
```

### 数据质量要求

| 指标 | 最小值 | 推荐值 | 本项目 |
|------|--------|--------|--------|
| 句子数量 | 50,000 | 300,000+ | 325,537 ✅ |
| 词汇量 | 5,000 | 8,000+ | 10,049 ✅ |
| 平均句长 | 10字符 | 15-30字符 | ~25字符 ✅ |
| 领域覆盖 | 10个 | 50+ | 59个 ✅ |

### 数据准备脚本

使用提供的脚本准备数据：

```python
# examples/prepare_dataset.py 的核心逻辑

import jieba
from collections import Counter

def build_vocab(sentences, min_freq=2, max_size=10000):
    """构建词汇表"""
    word_counts = Counter()
    for sentence in sentences:
        words = jieba.lcut(sentence)
        word_counts.update(words)
    
    # 过滤低频词
    vocab = {word: count for word, count in word_counts.items() 
             if count >= min_freq}
    
    # 按频率排序
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_vocab[:max_size])
```

---

## ⚙️ 配置参数 | Configuration

### 核心训练参数

#### 1. 学习率 (Learning Rate)

```python
learning_rate = 5e-5  # 默认值

# 推荐范围：
# - 小数据集 (< 100K): 1e-4
# - 中等数据集 (100K-500K): 5e-5 ✅
# - 大数据集 (> 500K): 2e-5
```

**影响**:
- 过高：训练不稳定，损失震荡
- 过低：收敛太慢，需要更多步数
- **本项目选择**: 5e-5（在100K步内达到良好收敛）

#### 2. 批大小 (Batch Size)

```python
batch_size = 16  # 默认值

# 显存与批大小关系：
# - RTX 3070 (8GB): 8-16 ✅
# - RTX 3080 (10GB): 16-24
# - RTX 3090 (24GB): 32-64
# - A100 (40GB): 64-128
```

**权衡**:
- **大批次** (32+): 稳定，但显存需求高
- **小批次** (8-16): 显存友好，但可能不稳定
- **本项目**: 16（RTX 3070最优配置）

#### 3. 训练步数 (Training Steps)

```python
total_steps = 100000  # 本项目设置

# 经验公式：
# steps = (sentences × epochs) / batch_size
# 100K = (325,537 × 5) / 16 ≈ 101K

# 建议：
# - 快速验证: 10,000步
# - 中等训练: 50,000步
# - 完整训练: 100,000步 ✅
# - 大规模训练: 200,000+步
```

#### 4. 学习率调度 (LR Scheduler)

```python
# 线性预热 + 余弦衰减
warmup_steps = 10000  # 前10K步预热
max_steps = 100000

# LR变化：
# [0-10K]:   0 → 5e-5 (线性增长)
# [10K-100K]: 5e-5 → 1e-6 (余弦衰减)
```

**可视化**:
```
LR
 |
5e-5 ┤     ╭────╮
     │    ╱      ╲
     │   ╱        ╲___
     │  ╱             ╲___
1e-6 ┤ ╱                   ╲___
     └─────────────────────────> Steps
       0   10K      50K      100K
```

---

## 📈 训练监控 | Training Monitoring

### 实时监控指标

训练过程中，每100步输出以下指标：

```
Step 50000/100000 | Loss: 3.45 | MLM Acc: 45.32% | NSP Acc: 87.56% | LR: 2.5e-5
```

#### 关键指标解读

| 指标 | 含义 | 健康范围 | 异常信号 |
|------|------|----------|----------|
| **Loss** | 总损失函数 | 持续下降 | 突然上升或NaN |
| **MLM Acc** | 遮蔽词预测准确率 | 逐步提升至50%+ | 停滞在<30% |
| **NSP Acc** | 句子对预测准确率 | 快速达到85%+ | 徘徊在50%附近 |
| **LR** | 当前学习率 | 按调度变化 | 保持不变 |

### 训练曲线分析

#### 1. 正常训练模式 ✅

```
MLM Accuracy
    60% ┤              ╭────
    50% ┤         ╭───╯
    40% ┤     ╭──╯
    30% ┤  ╭─╯
    20% ┤╭╯
        └────────────────────> Steps
```

**特征**:
- 平滑上升
- 最终收敛
- 无剧烈波动

#### 2. 过拟合模式 ⚠️

```
MLM Accuracy
    60% ┤      ╭─╮╭─╮╭─╮
    50% ┤    ╭╯ ╰╯ ╰╯ ╰╮
    40% ┤ ╭─╯          ╰─
        └────────────────────> Steps
```

**特征**:
- 剧烈波动
- 训练集acc高，验证集acc低
- **解决**: 增加dropout或数据量

#### 3. 学习率过高 ❌

```
MLM Accuracy
    60% ┤    ╱╲    ╱╲
    50% ┤   ╱  ╲  ╱  ╲
    40% ┤  ╱    ╲╱    ╲
    30% ┤ ╱            ╲
        └────────────────────> Steps
```

**特征**:
- 大幅震荡
- 损失不收敛
- **解决**: 降低学习率至1e-5或更低

### 检查点保存

```python
# 自动保存检查点
checkpoint_steps = [10000, 25000, 50000, 75000, 100000]

# 保存内容：
{
    'step': current_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': current_loss,
    'mlm_accuracy': current_mlm_acc
}
```

**文件命名规则**:
- `checkpoint_step_10000.pth`
- `checkpoint_step_50000.pth`
- `stage4_large_100k_final.pth` ← 最终模型

---

## 🔧 故障排除 | Troubleshooting

### 问题1: CUDA内存不足

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**解决方案**:

1. **降低批大小**:
   ```python
   batch_size = 8  # 从16降到8
   ```

2. **启用梯度累积**:
   ```python
   accumulation_steps = 2
   effective_batch_size = batch_size × accumulation_steps
   ```

3. **使用混合精度训练**:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   with autocast():
       outputs = model(**inputs)
       loss = outputs.loss
   scaler.scale(loss).backward()
   ```

4. **清理缓存**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### 问题2: 损失为NaN

```
Step 15000 | Loss: nan | MLM Acc: 0.00%
```

**原因**:
- 学习率过高
- 梯度爆炸
- 数据问题

**解决方案**:

1. **降低学习率**:
   ```python
   learning_rate = 1e-5  # 从5e-5降到1e-5
   ```

2. **梯度裁剪**:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

3. **检查数据**:
   ```python
   # 验证数据中没有异常值
   for sentence in sentences[:100]:
       assert len(sentence) > 0, "空句子"
       assert len(sentence) < 512, "句子过长"
   ```

### 问题3: 训练速度慢

**症状**: 100K步预计需要40小时+ (本项目仅需22.5小时)

**优化方案**:

1. **启用DataLoader多进程**:
   ```python
   dataloader = DataLoader(
       dataset, 
       batch_size=16, 
       num_workers=4,  # 启用4个进程
       pin_memory=True
   )
   ```

2. **使用更高效的tokenizer**:
   ```python
   from transformers import BertTokenizerFast  # Fast版本
   tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
   ```

3. **减少日志频率**:
   ```python
   log_interval = 500  # 从100改为500
   ```

### 问题4: 模型不收敛

**症状**: 50K步后MLM准确率仍 < 30%

**诊断步骤**:

1. **检查数据质量**:
   ```bash
   python examples/prepare_dataset.py
   # 确认词汇量 > 5000
   # 确认句子数 > 50000
   ```

2. **检查标签生成**:
   ```python
   # 确保MLM标签正确
   print(f"Masked tokens: {(labels != -100).sum()}")
   print(f"Total tokens: {labels.numel()}")
   # 比例应在15%左右
   ```

3. **降低任务难度**:
   ```python
   # 临时降低mask比例测试
   mask_prob = 0.10  # 从0.15降到0.10
   ```

---

## ✨ 最佳实践 | Best Practices

### 1. 渐进式训练策略

```
阶段1 (0-10K):   快速验证配置
├─ 目标: 损失开始下降
├─ 检查点: 10K步
└─ 决策: 继续 或 调整参数

阶段2 (10K-50K): 初步收敛
├─ 目标: MLM准确率 > 35%
├─ 检查点: 25K, 50K步
└─ 决策: 评估是否需要更多数据

阶段3 (50K-100K): 精细优化
├─ 目标: MLM准确率 > 50%
├─ 检查点: 75K, 100K步
└─ 结果: 最终模型 ✅
```

### 2. 超参数搜索

```python
# 网格搜索示例
configs = [
    {'lr': 5e-5, 'batch': 16, 'steps': 100000},  # 本项目配置 ✅
    {'lr': 2e-5, 'batch': 32, 'steps': 100000},
    {'lr': 1e-4, 'batch': 8, 'steps': 100000},
]

for config in configs:
    print(f"测试配置: {config}")
    train_model(**config)
    evaluate_and_save(config)
```

### 3. 实验记录

创建实验日志 `experiments.json`:

```json
{
  "experiment_1": {
    "date": "2025-10-17",
    "config": {
      "learning_rate": 5e-5,
      "batch_size": 16,
      "steps": 100000
    },
    "results": {
      "final_loss": 2.97,
      "mlm_accuracy": 50.53,
      "pearson_r": 0.7869,
      "training_time": "22.5 hours"
    },
    "notes": "达到目标的92.6%，结果可接受"
  }
}
```

### 4. GPU利用率优化

```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 目标指标：
# - GPU利用率: > 90%
# - 显存使用: 70-90%
# - 温度: < 85°C
```

**优化技巧**:
- 批大小尽可能大（不超显存）
- 启用TF32加速（Ampere架构）
- 使用混合精度训练

### 5. 数据增强技巧

```python
# 句子变换增强
def augment_sentence(sentence):
    """数据增强"""
    # 同义词替换
    # 随机插入
    # 随机删除
    # 句子重组
    return augmented_sentence

# 动态mask策略
def dynamic_masking(tokens, step):
    """根据训练进度调整mask比例"""
    if step < 10000:
        mask_prob = 0.10  # 初期降低难度
    elif step < 50000:
        mask_prob = 0.15  # 标准难度
    else:
        mask_prob = 0.20  # 后期增加难度
    return apply_mask(tokens, mask_prob)
```

---

## 📝 训练检查清单 | Training Checklist

### 开始训练前 ✓

- [ ] GPU驱动已更新至最新版本
- [ ] CUDA和PyTorch版本兼容
- [ ] 数据集格式正确（JSON，包含sentences和vocabulary）
- [ ] 数据集大小足够（>50K句子，>5K词汇）
- [ ] 磁盘空间充足（>10GB用于模型和日志）
- [ ] 已设置正确的`CUDA_VISIBLE_DEVICES`

### 训练中监控 ✓

- [ ] 每10K步检查损失下降趋势
- [ ] 监控MLM准确率是否稳步提升
- [ ] 观察GPU利用率（应>80%）
- [ ] 检查显存使用（不应频繁OOM）
- [ ] 保存关键步数的检查点

### 训练后分析 ✓

- [ ] 运行`analyze_100k.py`计算Pearson相关系数
- [ ] 运行`visualize_results.py`生成训练曲线
- [ ] 检查R²是否>0.6（说明拟合良好）
- [ ] 验证最终模型在测试集上的表现
- [ ] 记录实验结果和最佳配置

---

## 🎯 下一步 | Next Steps

完成训练后，你可以：

1. **分析结果**: 使用 `examples/load_and_analyze.py`
2. **测试模型**: 使用 `examples/quick_start.py`
3. **微调模型**: 在特定任务上继续训练
4. **分享经验**: 提交你的配置和结果到社区

---

## 📚 相关文档 | Related Documentation

- [数据集说明](DATASET.md) - 了解数据准备细节
- [分析方法](ANALYSIS.md) - 理解结果分析方法
- [使用示例](../examples/README.md) - 查看更多代码示例
- [项目README](../README.md) - 项目概览

---

## 💡 需要帮助？ | Need Help?

- 📝 查看 [FAQ](../README.md#常见问题)
- 💬 提交 [GitHub Issue](https://github.com/yuzengbaao/chinese-bert-correlation/issues)
- 📧 联系维护者

---

**祝训练顺利！Happy Training!** 🚀
