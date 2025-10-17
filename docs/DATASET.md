# Dataset Documentation | 数据集说明

## 📚 Overview | 概览

The Chinese BERT Correlation Study dataset consists of **325,537 Chinese sentences** collected from Chinese Wikipedia, covering **59 specialized domains** with a vocabulary of **10,049 unique characters**.

本数据集包含从中文维基百科采集的 **325,537个中文句子**，涵盖 **59个专业领域**，词汇量达 **10,049个汉字**。

---

## 📊 Dataset Statistics | 数据集统计

| Metric | Value | Description |
|--------|-------|-------------|
| **Total Sentences** | 325,537 | 总句子数 |
| **Unique Characters** | 10,049 | 独特汉字数 |
| **Average Sentence Length** | 46.2 chars | 平均句长 |
| **Duplication Rate** | 0.00% | 去重率 |
| **Specialized Domains** | 59 | 专业领域数 |
| **Source** | Chinese Wikipedia | 数据来源 |
| **File Size** | ~44.5 MB | 文件大小 |
| **Format** | JSON | 文件格式 |

---

## 🏷️ Domain Coverage | 领域覆盖

The dataset covers 59 specialized domains to ensure vocabulary diversity:

### 1. **People & Society | 人物与社会**
- 中国姓氏 (Chinese surnames)
- 历史人物 (Historical figures)
- 现代名人 (Modern celebrities)

### 2. **Geography | 地理**
- 郡县 (Counties and prefectures)
- 城市 (Cities)
- 山川河流 (Mountains and rivers)
- 名胜古迹 (Scenic spots)

### 3. **Biology | 生物**
- 昆虫名 (Insect names) - 86+ rare characters
- 鱼类名 (Fish names) - 68+ rare characters
- 鸟类 (Birds)
- 植物 (Plants)
- 中药材 (Traditional Chinese medicine) - 68+ rare characters

### 4. **Culture & History | 文化历史**
- 古代器物 (Ancient artifacts)
- 青铜器名 (Bronze ware) - 52+ rare characters
- 书法作品 (Calligraphy works)
- 文学作品 (Literary works)
- 成语典故 (Idioms and allusions)

### 5. **Science & Technology | 科技**
- 天文学 (Astronomy)
- 物理学 (Physics)
- 化学 (Chemistry)
- 数学 (Mathematics)
- 计算机科学 (Computer science)

### 6. **Arts & Entertainment | 艺术娱乐**
- 音乐 (Music)
- 绘画 (Painting)
- 电影 (Movies)
- 戏曲 (Traditional opera)

### 7. **Other Domains | 其他领域**
- 饮食文化 (Culinary culture)
- 建筑 (Architecture)
- 节日习俗 (Festivals and customs)
- 宗教哲学 (Religion and philosophy)
- And 30+ more...

---

## 🔧 Data Collection Process | 数据采集过程

### Strategy | 策略

The dataset was collected using a **multi-stage rare character collection strategy**:

```
Stage 1: General Collection (27K sentences, 4.7K vocab)
   ↓
Stage 2: Professional Domain Expansion (104K sentences, 6.4K vocab)
   ↓
Stage 3: Enhanced Vocabulary Expansion (150K sentences, 9.2K vocab)
   ↓
Stage 4: Rare Character Targeted Collection (325K sentences, 10K vocab) ✅
```

### Script | 脚本

The main data collection script is `rare_char_fetch.py`, which:

1. **Targets rare characters** from 59 specialized domains
2. **Fetches full articles** from Chinese Wikipedia
3. **Extracts quality sentences** (length 30-100 characters)
4. **Removes duplicates** to ensure uniqueness
5. **Validates character coverage** to reach 10K vocabulary goal

### Key Breakthroughs | 关键突破

- **Category "郡县" (Counties)**: +109 new characters
- **Category "中国姓氏" (Surnames)**: +92 new characters
- **Category "昆虫名" (Insects)**: +86 new characters
- **Category "鱼类名" (Fish)**: +68 new characters
- **Category "中药材" (TCM)**: +68 new characters
- **Category "青铜器名" (Bronzeware)**: +52 new characters (final push to 10,049!)

---

## 📁 Data Format | 数据格式

### File Structure

```json
{
  "sentences": [
    "中国是世界上历史最悠久的文明古国之一。",
    "北京是中华人民共和国的首都。",
    ...
  ]
}
```

### Fields

- **sentences** (array): List of Chinese sentences
  - Each sentence is a string
  - Length range: 30-100 characters
  - No duplicates
  - UTF-8 encoding

---

## ✅ Quality Assurance | 质量保证

### Validation Checks

The dataset passes all quality checks via `verify_dataset.py`:

```python
✅ Sentence count: 325,537 / 100,000 (target)
✅ Vocabulary: 10,049 / 10,000 (target)
✅ Average sentence length: 46.2 (within 30-100 range)
✅ Duplication rate: 0.00% (< 5% threshold)
✅ Data completeness: 100%
```

### Filtering Criteria

Sentences are included only if they:
1. Contain 30-100 characters
2. Are not duplicates
3. Come from verified Wikipedia articles
4. Pass UTF-8 encoding validation

---

## 🚀 Usage Examples | 使用示例

### Loading the Dataset

```python
import json

# Load dataset
with open('large_wikipedia_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

sentences = data['sentences']
print(f"Total sentences: {len(sentences)}")
print(f"Sample: {sentences[0]}")
```

### Building Vocabulary

```python
from collections import Counter

# Count character frequency
all_chars = ''.join(sentences)
char_freq = Counter(all_chars)

print(f"Unique characters: {len(char_freq)}")
print(f"Top 10 characters: {char_freq.most_common(10)}")
```

### Preparing Training Data

```python
# For BERT MLM training
def prepare_mlm_data(sentences, mask_prob=0.15):
    """Prepare masked language modeling data"""
    masked_sentences = []
    for sent in sentences:
        # Implement masking logic
        # ...
    return masked_sentences
```

---

## 📊 Comparison with Other Datasets | 与其他数据集对比

| Dataset | Sentences | Vocabulary | Domains | Duplication | Source |
|---------|-----------|-----------|---------|-------------|--------|
| **Ours** | **325,537** | **10,049** | **59** | **0.00%** | Wikipedia |
| Previous (50K) | 27,368 | 4,728 | 20 | 0.02% | Wikipedia |
| Common Crawl | Millions | ~8,000 | Mixed | High | Web |
| News Corpus | ~100K | ~5,000 | News | Medium | News sites |

**Advantages | 优势:**
- ✅ Large vocabulary (10K+ characters)
- ✅ High quality (zero duplication)
- ✅ Diverse domains (59 specialized areas)
- ✅ Verified source (Wikipedia)
- ✅ Balanced rare character coverage

---

## 🔬 Research Applications | 研究应用

This dataset is particularly useful for:

1. **Chinese NLP Model Training**
   - Pre-training BERT models
   - Language model fine-tuning
   - Transfer learning experiments

2. **Rare Character Handling**
   - Testing model performance on rare characters
   - Vocabulary coverage analysis
   - Character-level modeling

3. **Domain-Specific Studies**
   - Cross-domain generalization
   - Domain adaptation
   - Specialized vocabulary learning

4. **Correlation Studies**
   - Training dynamics analysis
   - Performance prediction
   - Optimal training step calculation

---

## 📝 Citation | 引用

If you use this dataset in your research:

```bibtex
@misc{chinese_wikipedia_dataset_2025,
  title={Chinese Wikipedia Dataset for BERT Training},
  author={Yuzengbaao},
  year={2025},
  howpublished={Chinese BERT Correlation Study},
  note={325,537 sentences, 10,049 vocabulary, 59 domains}
}
```

---

## 📮 Contact | 联系方式

For dataset-related questions:
- GitHub Issues: https://github.com/yuzengbaao/chinese-bert-correlation/issues
- Email: yuzengbaao@gmail.com

---

## 📜 License | 许可证

The dataset is released under MIT License. Chinese Wikipedia content is under CC BY-SA 3.0.

数据集采用MIT许可证。中文维基百科内容遵循 CC BY-SA 3.0 许可。
