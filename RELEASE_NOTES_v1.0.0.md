# Chinese BERT 100K Training Correlation Study - v1.0.0

**首次正式发布 | First Official Release**

发布日期: 2025年10月17日 | Release Date: October 17, 2025

---

## 🎯 项目概述 | Project Overview

本项目深入研究中文BERT模型训练步数与遮蔽语言模型（MLM）准确率之间的相关性。通过100,000步的系统训练和严格的统计分析，证实了二者之间存在**强正相关关系** (r=0.7869)。

This project investigates the correlation between training steps and Masked Language Modeling (MLM) accuracy in Chinese BERT models. Through 100,000 systematic training steps and rigorous statistical analysis, we confirmed a **strong positive correlation** (r=0.7869).

---

## 📊 核心成果 | Key Results

| 指标 | 数值 | 说明 |
|------|------|------|
| **Pearson r** | 0.7869 | 强正相关 (目标0.85的92.6%) |
| **R²** | 0.6193 | 解释61.9%的方差 |
| **P-value** | < 0.001 | 极显著统计学意义 |
| **MLM准确率** | 50.53% | 平均准确率，最高66.35% |
| **损失降低** | 66.8% | 从8.96降至2.97 |
| **训练步数** | 100,000 | 完整训练周期 |
| **训练时间** | 22.5小时 | RTX 3070 GPU |

---

## 🎉 主要特性 | Main Features

### 1. 完整的训练流程
- ✅ 100K步系统化训练
- ✅ 动态学习率调度（预热+余弦衰减）
- ✅ 多阶段检查点保存
- ✅ 实时训练监控

### 2. 严格的统计分析
- ✅ Pearson相关系数计算
- ✅ 95%置信区间 [0.76, 0.81]
- ✅ R²决定系数分析
- ✅ 残差和显著性检验

### 3. 丰富的数据集
- ✅ 325,537条中文句子
- ✅ 10,049词汇量
- ✅ 59个专业领域覆盖
- ✅ 高质量数据清洗

### 4. 专业可视化
- ✅ 训练曲线图（4合1）
- ✅ 相关性分析图
- ✅ 50K vs 100K对比
- ✅ 损失函数演变

### 5. 完善的文档
- ✅ 双语README
- ✅ 数据集详细说明
- ✅ 训练指南（520行）
- ✅ 分析方法论（660行）
- ✅ 4个使用示例

---

## 📦 发布内容 | Release Contents

### 核心文件 | Core Files
- `stage4_large_100k_final.pth` - 最终训练模型（48.40M参数）
- `training_history_100k.json` - 完整训练历史（1000个数据点）
- `analysis_100k_result.json` - 统计分析结果

### 训练脚本 | Training Scripts
- `main_train.py` - 主训练脚本
- `analyze_100k.py` - 结果分析工具
- `visualize_results.py` - 可视化生成器

### 文档 | Documentation
- `README.md` - 项目主页（双语）
- `docs/DATASET.md` - 数据集说明（278行）
- `docs/TRAINING.md` - 训练指南（520行）
- `docs/ANALYSIS.md` - 分析方法论（660行）
- `CHANGELOG.md` - 版本历史

### 示例代码 | Examples
- `examples/quick_start.py` - 快速开始
- `examples/load_and_analyze.py` - 自定义分析
- `examples/prepare_dataset.py` - 数据准备
- `examples/README.md` - 示例指南

### 可视化 | Visualizations
- `results/training_curves.png` - 训练曲线（780 KB）
- `results/correlation_analysis.png` - 相关性分析（1.06 MB）
- `results/comparison_50k_100k.png` - 对比图（157 KB）
- `results/loss_analysis.png` - 损失分析（577 KB）

---

## 🚀 快速开始 | Quick Start

### 1. 克隆仓库
```bash
git clone https://github.com/yuzengbaao/chinese-bert-correlation.git
cd chinese-bert-correlation
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行示例
```bash
# 测试模型预测
python examples/quick_start.py

# 分析训练结果
python examples/load_and_analyze.py
```

### 4. 查看结果
- 打开 `results/` 目录查看可视化图表
- 阅读 `analysis_100k_result.json` 查看详细指标

---

## 📈 与50K训练对比 | Comparison with 50K Training

| 指标 | 50K训练 | 100K训练 | 提升 |
|------|---------|----------|------|
| Pearson r | 0.6355 | 0.7869 | +23.8% ✅ |
| R² | 0.4038 | 0.6193 | +53.4% ✅ |
| MLM准确率 | 14.50% | 50.53% | +248.5% ✅ |
| 最大准确率 | 31.88% | 66.35% | +108.0% ✅ |
| 损失 | 5.12 | 2.97 | -42.0% ✅ |

---

## 🎓 学术引用 | Citation

如果本项目对你的研究有帮助，请引用：

```bibtex
@misc{chinese_bert_correlation_2025,
  author = {Your Name},
  title = {Chinese BERT 100K Training Correlation Study},
  year = {2025},
  publisher = {GitHub},
  version = {v1.0.0},
  url = {https://github.com/yuzengbaao/chinese-bert-correlation},
  note = {Pearson r = 0.7869, MLM accuracy = 50.53\%}
}
```

---

## 💡 应用场景 | Applications

1. **模型训练策略** - 根据相关性优化训练计划
2. **LLM监控系统** - 实时评估训练进度
3. **教育研究** - 深度学习课程案例
4. **成本优化** - 预测训练时间和资源需求
5. **AutoML集成** - 自动超参数调优
6. **模型压缩** - 指导知识蒸馏策略

---

## 🔧 技术栈 | Tech Stack

- **深度学习**: PyTorch 2.0+, Transformers 4.30+
- **NLP工具**: BERT, jieba
- **数据分析**: NumPy, SciPy, Pandas
- **可视化**: Matplotlib, Seaborn
- **数据集**: 325K中文维基百科句子

---

## 📝 更新日志 | Changelog

查看 [CHANGELOG.md](CHANGELOG.md) 了解详细更新历史。

---

## 🤝 贡献 | Contributing

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献指南。

可能的贡献方向：
- 🔬 扩展到200K+训练步数
- 📊 增加更多统计分析方法
- 🌐 支持其他语言（英文、日文等）
- 🎨 改进可视化效果
- 📚 补充更多使用示例

---

## 📞 联系方式 | Contact

- **GitHub Issues**: [提交问题](https://github.com/yuzengbaao/chinese-bert-correlation/issues)
- **Email**: yuzengbaao@gmail.com
- **项目主页**: https://github.com/yuzengbaao/chinese-bert-correlation

---

## 📄 开源协议 | License

本项目采用 [MIT License](LICENSE) 开源协议。

---

## 🙏 致谢 | Acknowledgments

- **BERT** - Google Research团队的开创性工作
- **Hugging Face Transformers** - 优秀的NLP工具库
- **中文维基百科** - 高质量数据来源
- **开源社区** - 各类依赖库的贡献者

---

## 📊 项目统计 | Project Stats

- **代码行数**: 3,600+ 行
- **文档字数**: 50,000+ 字
- **可视化图表**: 4 张高清PNG
- **训练数据**: 325,537 句子
- **词汇量**: 10,049 词
- **训练步数**: 100,000 步
- **训练时间**: 22.5 小时
- **模型参数**: 48.40M 参数

---

## 🔮 未来计划 | Future Plans

### v1.1.0 (计划中)
- [ ] 扩展到150K训练步数
- [ ] 添加Spearman和Kendall相关性分析
- [ ] 增加多GPU训练支持
- [ ] 提供预训练模型下载

### v2.0.0 (长期目标)
- [ ] 支持英文和其他语言
- [ ] 实现在线训练监控Dashboard
- [ ] 集成AutoML自动调优
- [ ] 发布研究论文

---

## ⚡ 性能指标 | Performance Metrics

### 训练性能
- **GPU**: NVIDIA RTX 3070 (8GB)
- **批大小**: 16
- **步数/秒**: ~1.23 steps/s
- **总训练时间**: 22.5小时
- **GPU利用率**: 92% 平均

### 模型性能
- **参数量**: 48.40M
- **推理速度**: ~100 sentences/s (CPU)
- **模型大小**: ~185 MB (.pth文件)
- **内存占用**: ~500 MB (加载后)

---

## 🌟 亮点总结 | Highlights

✨ **首个系统化研究** - 中文BERT训练步数与MLM准确率的相关性

📊 **严格统计分析** - Pearson r=0.7869，p<0.001，95% CI [0.76, 0.81]

🎯 **实用价值高** - 可用于训练规划、成本估算、AutoML集成

📚 **完善文档** - 2400+行双语文档，覆盖所有细节

💻 **开箱即用** - 4个示例脚本，5分钟快速开始

🔬 **可复现性** - 完整代码、数据、训练历史全部公开

---

**感谢使用本项目！欢迎Star⭐和Fork🍴**

**Thank you for using this project! Star⭐ and Fork🍴 are welcome!**

---

**下载附件 | Download Assets:**
- `training_history_100k.json` - 完整训练历史
- `analysis_100k_result.json` - 统计分析结果

（模型文件因GitHub大小限制，请联系获取或自行训练）
