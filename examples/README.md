# 使用示例 | Usage Examples

这个目录包含了多个实用的代码示例，帮助你快速上手使用本项目。

This directory contains practical code examples to help you get started with this project.

## 📚 示例列表 | Example List

### 1. 快速开始 | Quick Start
**文件**: `quick_start.py`

最简单的入门示例，展示如何加载模型并进行预测。

The simplest starter example showing how to load the model and make predictions.

```bash
python examples/quick_start.py
```

**功能 | Features:**
- 加载训练好的模型 | Load trained model
- 使用BERT进行词语预测 | Use BERT for word prediction
- 展示Top-5预测结果 | Show Top-5 predictions

---

### 2. 结果分析 | Result Analysis
**文件**: `load_and_analyze.py`

深入分析训练结果，生成自定义可视化图表。

In-depth analysis of training results with custom visualizations.

```bash
python examples/load_and_analyze.py
```

**功能 | Features:**
- 加载训练历史数据 | Load training history
- 计算Pearson相关系数和R² | Calculate Pearson r and R²
- 生成4种自定义图表 | Generate 4 custom plots
  * MLM准确率散点图 | MLM accuracy scatter plot
  * 滑动窗口平均 | Moving average
  * 损失函数演变 | Loss function evolution
  * 准确率分布直方图 | Accuracy distribution histogram

**输出 | Output:**
- `results/custom_analysis.png` (2400x1800 像素)

---

### 3. 数据集准备 | Dataset Preparation
**文件**: `prepare_dataset.py`

演示如何处理中文文本数据并准备MLM训练。

Demonstrates how to process Chinese text data for MLM training.

```bash
python examples/prepare_dataset.py
```

**功能 | Features:**
- 加载原始数据集 | Load raw dataset
- 使用jieba进行分词 | Tokenization with jieba
- 构建词汇表 | Build vocabulary
- 准备MLM训练样本 | Prepare MLM training samples
- 数据集统计分析 | Dataset statistics

---

## 🚀 运行环境 | Environment Setup

### 安装依赖 | Install Dependencies

```bash
pip install -r requirements.txt
```

### 必需文件 | Required Files

确保以下文件存在于项目根目录：

Make sure the following files exist in the project root:

- `stage4_large_100k_final.pth` - 训练好的模型 | Trained model
- `training_history_100k.json` - 训练历史 | Training history
- `large_wikipedia_dataset.json` - 原始数据集 | Raw dataset

---

## 📖 使用流程 | Usage Workflow

### 方案A：快速体验 | Quick Experience

```bash
# 1. 快速开始，测试模型
python examples/quick_start.py

# 2. 分析结果
python examples/load_and_analyze.py
```

### 方案B：完整流程 | Full Workflow

```bash
# 1. 准备数据集
python examples/prepare_dataset.py

# 2. 训练模型（使用主脚本）
python main_train.py

# 3. 分析结果
python examples/load_and_analyze.py

# 4. 测试模型
python examples/quick_start.py
```

---

## 🎯 自定义示例 | Custom Examples

### 创建你自己的MLM预测脚本

```python
from examples.quick_start import load_model, predict_masked_word

# 加载模型
model, tokenizer = load_model()

# 自定义句子
my_sentence = "深度学习是[MASK]的重要技术。"
result = predict_masked_word(model, tokenizer, my_sentence)
print(f"预测结果: {result}")
```

### 自定义分析脚本

```python
from examples.load_and_analyze import load_training_history, calculate_statistics

# 加载数据
steps, mlm_acc, loss = load_training_history()

# 计算你关心的指标
correlation, r_squared, slope = calculate_statistics(steps, mlm_acc)

# 使用数据进行自定义分析
import numpy as np
print(f"中位数准确率: {np.median(mlm_acc):.2f}%")
print(f"准确率范围: {mlm_acc.max() - mlm_acc.min():.2f}%")
```

---

## 💡 提示 | Tips

1. **GPU加速**: 如果有GPU，在`quick_start.py`中修改：
   ```python
   model.to('cuda')  # 使用GPU
   ```

2. **批量预测**: 修改`quick_start.py`支持批量处理：
   ```python
   sentences = ["句子1", "句子2", "句子3"]
   for sent in sentences:
       predict_masked_word(model, tokenizer, sent)
   ```

3. **保存结果**: 在分析脚本中添加保存功能：
   ```python
   results = {
       'correlation': correlation,
       'r_squared': r_squared,
       'slope': slope
   }
   with open('my_analysis.json', 'w') as f:
       json.dump(results, f, indent=2)
   ```

---

## 🐛 故障排除 | Troubleshooting

### 问题1: 找不到模型文件

```
FileNotFoundError: [Errno 2] No such file or directory: 'stage4_large_100k_final.pth'
```

**解决**: 确保模型文件在项目根目录，或修改路径：
```python
model, tokenizer = load_model('path/to/your/model.pth')
```

### 问题2: 内存不足

```
RuntimeError: CUDA out of memory
```

**解决**: 使用CPU加载：
```python
checkpoint = torch.load(model_path, map_location='cpu')
```

### 问题3: 中文显示乱码

**解决**: 安装中文字体：
```bash
# Windows: 确保安装了SimHei字体
# Linux: sudo apt-get install fonts-wqy-zenhei
# Mac: 系统自带支持
```

---

## 📞 需要帮助？ | Need Help?

- 📝 查看完整文档: [README.md](../README.md)
- 📊 数据集说明: [docs/DATASET.md](../docs/DATASET.md)
- 💬 提交Issue: [GitHub Issues](https://github.com/yuzengbaao/chinese-bert-correlation/issues)

---

**Happy Coding! 祝编程愉快！** 🎉
