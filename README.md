# Chinese BERT Training: 100K Steps Correlation Study# AGI 种子算法 - 第一性原理设计



<div align="center">## 🌱 核心哲学



**[中文](#中文文档) | [English](#english-documentation)**> "智能是一个系统通过最小化预测误差来压缩经验的能力"



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)基于第一性原理，我们将智能分解为三个最基本的原语：

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)1. **表示** (Representation) - 如何编码世界

2. **预测** (Prediction) - 如何理解模式

**研究训练步数与MLM准确度之间的相关性 | Studying the Correlation between Training Steps and MLM Accuracy**3. **适应** (Adaptation) - 如何自我改进



</div>## 📐 第一性原理推导



---### 公理 1: 信息压缩原理

- 智能 = 用最少的信息表示最多的经验

## 中文文档- Kolmogorov 复杂度 → 最优压缩 = 理解



### 📊 核心成果### 公理 2: 预测最小化原理

- 学习 = 减少未来的意外

本项目通过 **100,000步** 的BERT模型训练，验证了**训练步数与MLM（Masked Language Model）准确度之间存在强正相关关系**：- 自由能原理 (Free Energy Principle)



| 指标 | 50K步训练 | 100K步训练 | 提升幅度 |### 公理 3: 递归自举原理

|------|----------|-----------|---------|- 简单规则 × 大规模迭代 = 复杂涌现

| **Pearson相关系数** | 0.6359 | **0.7869** | **+23.8%** ⭐ |- 自我指涉 → 元学习能力

| **平均MLM准确度** | 14.50% | **50.53%** | **+248.5%** ⭐⭐⭐ |

| **最大MLM准确度** | 31.88% | **66.35%** | **+108.0%** |## 🧬 种子算法架构

| **数据集规模** | 27,368句 | 325,537句 | **11.9x** |

| **词汇量** | 4,728字 | 10,049字 | **2.1x** |```

输入序列 → [状态空间] → 预测输出

**关键发现**：              ↑  ↓

- ✅ Pearson相关系数 **r = 0.7869**（强正相关）           误差反馈 → 状态更新

- ✅ 可解释 **61.9%** 的MLM准确度方差              ↓

- ✅ MLM准确度提升 **3.5倍**         涌现复杂行为

- ✅ 损失函数下降 **66.8%**（8.96 → 2.97）```



### 🎯 项目特点## 🎯 设计原则



1. **大规模中文数据集**1. **最小性**: 不可再简化

   - 325,537个句子（来自中文维基百科）2. **通用性**: 适用于任何域

   - 10,049个汉字词汇3. **可扩展**: 规模定律支持

   - 59个专业领域（姓氏、地名、中药材、昆虫、古代器物等）4. **自组织**: 无需人工设计特征

   - 平均句长：46.2字符5. **可复现**: 完全确定性（给定随机种子）



2. **完整的训练Pipeline**## 📊 预期涌现能力

   - 数据采集：多策略Wikipedia爬取

   - 数据清洗：去重、质量检查当规模扩大时，将涌现：

   - 模型训练：梯度累积、学习率调度- 抽象表示

   - 结果分析：相关性计算、可视化- 组合泛化

- 元学习

3. **可复现的实验设计**- 因果推理

   - 详细的超参数配置- ...（更多高阶能力）

   - 完整的训练日志（1000个数据点）

   - 50个训练检查点（每2000步）## 🔬 实验验证路径

   - 对比实验（50K vs 100K步）

1. 玩具问题（序列预测）

4. **丰富的分析工具**2. 简单环境（网格世界）

   - 训练曲线可视化3. 语言建模

   - 相关性分析报告4. 多模态理解

   - 统计显著性检验5. 开放域推理

   - 多维度性能对比

---

### 🚀 快速开始

**关键洞察**: 不要设计"智能"，而是设计"可以学习智能的系统"

#### 环境要求

```bash
Python >= 3.8
PyTorch >= 2.0
CUDA >= 11.8 (推荐使用GPU训练)
```

#### 安装依赖

```bash
pip install -r requirements.txt
```

#### 数据准备

```bash
# 1. 下载并预处理中文维基百科数据
python rare_char_fetch.py

# 2. 验证数据集质量
python verify_dataset.py
```

#### 开始训练

```bash
# 100K步完整训练（约22.5小时，RTX 3070）
python train_large_100k.py

# 训练会自动保存：
# - 模型检查点：checkpoints_100k/step_*.pth
# - 训练历史：training_history_100k.json
# - 最终模型：stage4_large_100k_final.pth
```

#### 结果分析

```bash
# 生成完整分析报告
python analyze_100k.py

# 输出文件：
# - analysis_100k_result.json（数值结果）
# - 控制台输出（详细统计）
```

### 📂 项目结构

```
AGI/
├── README.md                          # 本文件
├── requirements.txt                   # 依赖列表
├── LICENSE                           # MIT许可证
│
├── data/                             # 数据目录
│   ├── large_wikipedia_dataset.json  # 325K句子数据集
│   └── vocab.txt                     # 10K词汇表
│
├── scripts/                          # 数据采集脚本
│   ├── rare_char_fetch.py           # 稀有字符采集
│   ├── verify_dataset.py            # 数据集验证
│   └── check_progress.py            # 进度检查
│
├── training/                         # 训练脚本
│   ├── train_large_100k.py          # 100K步训练
│   └── model.py                     # BERT模型定义
│
├── analysis/                         # 分析工具
│   ├── analyze_100k.py              # 结果分析
│   └── visualize.py                 # 可视化生成
│
├── results/                          # 实验结果
│   ├── stage4_large_100k_final.pth  # 最终模型（193MB）
│   ├── training_history_100k.json   # 训练历史（1000点）
│   ├── analysis_100k_result.json    # 分析报告
│   └── checkpoints_100k/            # 50个检查点
│
└── docs/                            # 文档
    ├── DATASET.md                   # 数据集说明
    ├── TRAINING.md                  # 训练指南
    └── ANALYSIS.md                  # 分析方法
```

### 📈 实验结果

#### 1. Pearson相关性分析

```
Pearson r = 0.7869 (p < 0.001)
R² = 0.6193 (61.9%方差解释)
强度评价: 中等偏强正相关
```

**解释**：训练步数每增加10,000步，MLM准确度平均提升约3.5个百分点。

#### 2. MLM准确度进展

| 训练步数 | MLM准确度 | 损失 |
|---------|----------|------|
| 0 | 15.16% | 8.96 |
| 25,000 | 35.24% | 5.12 |
| 50,000 | 48.67% | 3.54 |
| 75,000 | 58.91% | 3.21 |
| 100,000 | 54.44% | 2.97 |

#### 3. 与50K训练对比

```
相关性提升: 0.6359 → 0.7869 (+23.8%)
MLM准确度: 14.50% → 50.53% (+248.5%)
训练时长: 11.2小时 → 22.5小时
```

### 💡 应用场景

1. **模型训练策略优化**
   - 根据目标性能预测所需训练步数
   - 设计科学的早停策略
   - 优化资源分配

2. **LLM训练监控**
   - 建立训练健康指标
   - 异常检测（偏离相关性曲线）
   - 多实验对比基线

3. **教育与研究**
   - NLP课程教学案例
   - 论文实验支撑
   - 开源社区贡献

4. **成本优化**
   - 精确预测训练成本
   - 避免过度训练
   - 提高训练效率

### 📖 引用

如果本项目对您的研究有帮助，欢迎引用：

```bibtex
@misc{chinese_bert_correlation_2025,
  title={Chinese BERT Training: A 100K Steps Correlation Study},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/chinese-bert-correlation}},
  note={Studying the correlation between training steps and MLM accuracy using 325K Chinese sentences}
}
```

### 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

### 📝 许可证

本项目采用 [MIT License](LICENSE)。

### 📮 联系方式

- Issue: [GitHub Issues](https://github.com/yourusername/chinese-bert-correlation/issues)
- Email: your.email@example.com

---

## English Documentation

### 📊 Key Results

This project validates a **strong positive correlation between training steps and MLM (Masked Language Model) accuracy** through **100,000 steps** of BERT model training:

| Metric | 50K Training | 100K Training | Improvement |
|--------|-------------|---------------|-------------|
| **Pearson Correlation** | 0.6359 | **0.7869** | **+23.8%** ⭐ |
| **Avg MLM Accuracy** | 14.50% | **50.53%** | **+248.5%** ⭐⭐⭐ |
| **Max MLM Accuracy** | 31.88% | **66.35%** | **+108.0%** |
| **Dataset Size** | 27,368 sents | 325,537 sents | **11.9x** |
| **Vocabulary** | 4,728 chars | 10,049 chars | **2.1x** |

**Key Findings**:
- ✅ Pearson correlation coefficient **r = 0.7869** (strong positive)
- ✅ Explains **61.9%** of MLM accuracy variance
- ✅ MLM accuracy improved by **3.5x**
- ✅ Loss decreased by **66.8%** (8.96 → 2.97)

### 🎯 Features

1. **Large-Scale Chinese Dataset**
   - 325,537 sentences (from Chinese Wikipedia)
   - 10,049 Chinese character vocabulary
   - 59 specialized domains (surnames, places, TCM, insects, artifacts, etc.)
   - Average sentence length: 46.2 characters

2. **Complete Training Pipeline**
   - Data collection: Multi-strategy Wikipedia crawling
   - Data cleaning: Deduplication, quality checks
   - Model training: Gradient accumulation, learning rate scheduling
   - Result analysis: Correlation calculation, visualization

3. **Reproducible Experimental Design**
   - Detailed hyperparameter configuration
   - Complete training logs (1000 data points)
   - 50 training checkpoints (every 2000 steps)
   - Comparative experiments (50K vs 100K steps)

4. **Rich Analysis Tools**
   - Training curve visualization
   - Correlation analysis reports
   - Statistical significance testing
   - Multi-dimensional performance comparison

### 🚀 Quick Start

#### Requirements

```bash
Python >= 3.8
PyTorch >= 2.0
CUDA >= 11.8 (GPU training recommended)
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Data Preparation

```bash
# 1. Download and preprocess Chinese Wikipedia data
python rare_char_fetch.py

# 2. Verify dataset quality
python verify_dataset.py
```

#### Start Training

```bash
# Full 100K steps training (~22.5 hours on RTX 3070)
python train_large_100k.py

# Auto-saves:
# - Model checkpoints: checkpoints_100k/step_*.pth
# - Training history: training_history_100k.json
# - Final model: stage4_large_100k_final.pth
```

#### Result Analysis

```bash
# Generate comprehensive analysis report
python analyze_100k.py

# Output files:
# - analysis_100k_result.json (numerical results)
# - Console output (detailed statistics)
```

### 📈 Experimental Results

#### 1. Pearson Correlation Analysis

```
Pearson r = 0.7869 (p < 0.001)
R² = 0.6193 (61.9% variance explained)
Strength: Moderately strong positive correlation
```

**Interpretation**: For every 10,000 additional training steps, MLM accuracy improves by approximately 3.5 percentage points on average.

#### 2. MLM Accuracy Progress

| Training Steps | MLM Accuracy | Loss |
|---------------|-------------|------|
| 0 | 15.16% | 8.96 |
| 25,000 | 35.24% | 5.12 |
| 50,000 | 48.67% | 3.54 |
| 75,000 | 58.91% | 3.21 |
| 100,000 | 54.44% | 2.97 |

#### 3. Comparison with 50K Training

```
Correlation improvement: 0.6359 → 0.7869 (+23.8%)
MLM accuracy: 14.50% → 50.53% (+248.5%)
Training time: 11.2 hours → 22.5 hours
```

### 💡 Use Cases

1. **Model Training Strategy Optimization**
   - Predict required training steps based on target performance
   - Design scientific early stopping strategies
   - Optimize resource allocation

2. **LLM Training Monitoring**
   - Establish training health metrics
   - Anomaly detection (deviation from correlation curve)
   - Multi-experiment comparison baseline

3. **Education & Research**
   - NLP course teaching cases
   - Research paper experimental support
   - Open-source community contributions

4. **Cost Optimization**
   - Accurately predict training costs
   - Avoid over-training
   - Improve training efficiency

### 📖 Citation

If this project helps your research, please cite:

```bibtex
@misc{chinese_bert_correlation_2025,
  title={Chinese BERT Training: A 100K Steps Correlation Study},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/chinese-bert-correlation}},
  note={Studying the correlation between training steps and MLM accuracy using 325K Chinese sentences}
}
```

### 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### 📝 License

This project is licensed under the [MIT License](LICENSE).

### 📮 Contact

- Issues: [GitHub Issues](https://github.com/yourusername/chinese-bert-correlation/issues)
- Email: your.email@example.com

---

<div align="center">

**⭐ If you find this project helpful, please give it a star! ⭐**

**如果本项目对您有帮助，请给个星标支持！**

Made with ❤️ for the Chinese NLP Community

</div>
