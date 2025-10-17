"""
数据集准备示例 | Dataset Preparation Example
==========================================

演示如何准备和处理中文数据集用于训练。
Demonstrates how to prepare and process Chinese dataset for training.
"""

import json
import jieba
from collections import Counter
import random

def load_raw_data(filepath='large_wikipedia_dataset.json'):
    """加载原始数据集 | Load raw dataset"""
    print(f"📥 加载原始数据 | Loading raw data from {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sentences = data.get('sentences', [])
    print(f"✅ 加载完成 | Loaded {len(sentences)} sentences")
    return sentences

def build_vocabulary(sentences, min_freq=2, max_vocab_size=10000):
    """构建词汇表 | Build vocabulary"""
    print(f"\n🔨 构建词汇表 | Building vocabulary...")
    print(f"   最小频率 | Min frequency: {min_freq}")
    print(f"   最大词汇量 | Max vocab size: {max_vocab_size}")
    
    # 分词并统计词频
    word_counts = Counter()
    for sentence in sentences:
        words = jieba.lcut(sentence)
        word_counts.update(words)
    
    # 过滤低频词
    vocab = {word: count for word, count in word_counts.items() 
             if count >= min_freq}
    
    # 按频率排序并截断
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    final_vocab = dict(sorted_vocab[:max_vocab_size])
    
    print(f"✅ 词汇表构建完成 | Vocabulary built")
    print(f"   总词数 | Total words: {len(word_counts)}")
    print(f"   过滤后 | After filtering: {len(vocab)}")
    print(f"   最终词汇量 | Final vocab size: {len(final_vocab)}")
    
    return final_vocab

def prepare_mlm_data(sentences, mask_prob=0.15):
    """准备MLM训练数据 | Prepare MLM training data"""
    print(f"\n🎭 准备MLM数据 | Preparing MLM data...")
    print(f"   遮蔽概率 | Mask probability: {mask_prob}")
    
    mlm_samples = []
    
    for sentence in sentences[:5]:  # 只展示前5个示例
        words = jieba.lcut(sentence)
        
        # 随机遮蔽
        masked_sentence = []
        original_words = []
        
        for word in words:
            if random.random() < mask_prob:
                masked_sentence.append('[MASK]')
                original_words.append(word)
            else:
                masked_sentence.append(word)
                original_words.append(word)
        
        mlm_samples.append({
            'original': ''.join(words),
            'masked': ''.join(masked_sentence),
            'targets': original_words
        })
    
    print(f"✅ MLM数据准备完成 | MLM data prepared")
    return mlm_samples

def print_statistics(sentences, vocab):
    """打印数据集统计信息 | Print dataset statistics"""
    print("\n" + "=" * 60)
    print("📊 数据集统计 | Dataset Statistics")
    print("=" * 60)
    
    # 句子统计
    total_chars = sum(len(s) for s in sentences)
    avg_length = total_chars / len(sentences)
    
    print(f"句子数量 | Total sentences: {len(sentences):,}")
    print(f"总字符数 | Total characters: {total_chars:,}")
    print(f"平均长度 | Average length: {avg_length:.2f} chars")
    
    # 词汇统计
    print(f"\n词汇量 | Vocabulary size: {len(vocab):,}")
    
    # Top 10高频词
    top_words = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 高频词 | Top 10 frequent words:")
    for i, (word, count) in enumerate(top_words, 1):
        print(f"  {i:2d}. {word:8s} : {count:6,} 次")

def save_processed_data(sentences, vocab, output_prefix='processed'):
    """保存处理后的数据 | Save processed data"""
    print(f"\n💾 保存处理后的数据 | Saving processed data...")
    
    # 保存句子
    sentences_file = f"{output_prefix}_sentences.json"
    with open(sentences_file, 'w', encoding='utf-8') as f:
        json.dump({'sentences': sentences}, f, ensure_ascii=False, indent=2)
    print(f"✅ 句子已保存 | Sentences saved to {sentences_file}")
    
    # 保存词汇表
    vocab_file = f"{output_prefix}_vocab.json"
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"✅ 词汇表已保存 | Vocabulary saved to {vocab_file}")

def main():
    """主函数 | Main function"""
    print("=" * 60)
    print("Chinese BERT 100K - 数据集准备 | Dataset Preparation")
    print("=" * 60)
    
    # 加载数据
    sentences = load_raw_data()
    
    # 构建词汇表
    vocab = build_vocabulary(sentences, min_freq=2, max_vocab_size=10000)
    
    # 打印统计信息
    print_statistics(sentences, vocab)
    
    # 准备MLM数据示例
    mlm_samples = prepare_mlm_data(sentences)
    
    print("\n" + "=" * 60)
    print("📝 MLM数据示例 | MLM Data Examples")
    print("=" * 60)
    for i, sample in enumerate(mlm_samples[:3], 1):
        print(f"\n示例 {i} | Example {i}:")
        print(f"  原始 | Original: {sample['original']}")
        print(f"  遮蔽 | Masked: {sample['masked']}")
    
    # 保存处理后的数据（可选）
    # save_processed_data(sentences, vocab)
    
    print("\n" + "=" * 60)
    print("✅ 数据准备完成！| Data preparation completed!")
    print("=" * 60)

if __name__ == '__main__':
    main()
