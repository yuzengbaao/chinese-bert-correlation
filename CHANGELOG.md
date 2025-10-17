# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-17

### 🎉 Initial Release

This is the first public release of the Chinese BERT 100K Training Correlation Study!

### Added

#### 📊 Core Results
- Complete 100,000 steps BERT training on Chinese Wikipedia dataset
- Pearson correlation coefficient: **r = 0.7869** (strong positive correlation)
- MLM accuracy improvement: **+248.5%** (from 14.50% to 50.53%)
- R² = 0.6193 (explains 61.9% of variance)

#### 📦 Dataset
- 325,537 Chinese sentences from Wikipedia
- 10,049 character vocabulary
- 59 specialized domains coverage
- Zero duplication rate
- Average sentence length: 46.2 characters

#### 🔧 Tools & Scripts
- `train_large_100k.py` - Main training script with gradient accumulation
- `rare_char_fetch.py` - Multi-strategy data collection from Wikipedia
- `analyze_100k.py` - Comprehensive result analysis with Pearson correlation
- `visualize_results.py` - Generate 4 high-resolution training visualizations
- `verify_dataset.py` - Dataset quality validation
- `check_progress.py` - Real-time training progress monitoring

#### 📈 Visualizations
- Training curves (MLM, NSP, Loss, Learning Rate)
- Correlation analysis (scatter plot + residuals)
- 50K vs 100K comparison chart
- Loss analysis with moving average

#### 📚 Documentation
- Comprehensive bilingual README (Chinese/English)
- MIT License
- Contributing guidelines
- Requirements specification
- Complete project structure documentation

#### 🎯 Key Features
- **Reproducible experiments** with detailed hyperparameters
- **50 training checkpoints** saved every 2,000 steps
- **1,000 data points** recorded during training
- **Comparative analysis** with 50K baseline training

### Training Details

#### Model Architecture
- Type: BERT-style transformer
- Parameters: 48.40M
- Hidden dimension: 512
- Layers: 12
- Attention heads: 8
- Feed-forward dimension: 2,048

#### Training Configuration
- Total steps: 100,000
- Batch size: 32
- Gradient accumulation: 4 steps (effective batch: 128)
- Learning rate: 5e-5 with cosine annealing
- Warmup steps: 2,000
- Training time: ~22.5 hours on RTX 3070

#### Performance Metrics
- Initial Loss: 8.9572 → Final Loss: 2.9728 (-66.8%)
- MLM Accuracy: 15.16% → 66.35% (max)
- Average MLM: 50.53%
- Correlation with steps: r = 0.7869 (p < 0.001)

### Comparison with 50K Training

| Metric | 50K | 100K | Improvement |
|--------|-----|------|-------------|
| Pearson r | 0.6359 | 0.7869 | +23.8% |
| Avg MLM | 14.50% | 50.53% | +248.5% |
| Dataset | 27K sents | 325K sents | +1090% |
| Vocabulary | 4,728 | 10,049 | +112.5% |

### 🔗 Links
- GitHub Repository: https://github.com/yuzengbaao/chinese-bert-correlation
- Documentation: See README.md
- Issues: https://github.com/yuzengbaao/chinese-bert-correlation/issues

---

## [Unreleased]

### Planned Features
- [ ] Pre-trained model upload to Hugging Face
- [ ] Interactive training dashboard
- [ ] Extended documentation with tutorials
- [ ] Additional visualization options
- [ ] Performance optimization guides
- [ ] Community examples and use cases

---

## Release Notes Format

### Categories
- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes

---

**Legend:**
- 🎉 Major release
- ✨ New feature
- 🐛 Bug fix
- 📚 Documentation
- 🔧 Tooling
- ⚡ Performance
- 🔒 Security
