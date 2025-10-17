#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证large_wikipedia_dataset.json的质量
"""

import json
import sys

def verify_dataset(file_path='large_wikipedia_dataset.json'):
    """验证数据集质量"""
    
    print("=" * 80)
    print("数据集质量验证")
    print("=" * 80)
    
    try:
        # 加载数据
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sentences = data.get('sentences', [])
        vocab = data.get('vocab', [])
        metadata = data.get('metadata', {})
        
        # 基础统计
        print("\n【基础统计】")
        print(f"✓ 句子总数:       {len(sentences):,}")
        print(f"✓ 词汇表大小:     {len(vocab):,}")
        print(f"✓ 平均句长:       {metadata.get('avg_sentence_length', 0):.1f} 字符")
        print(f"✓ 去重率:         {metadata.get('deduplication_rate', 0):.2f}%")
        
        # 质量检查
        print("\n【质量检查】")
        
        # 1. 数量达标
        sentence_ok = len(sentences) >= 100000
        vocab_ok = len(vocab) >= 10000
        
        print(f"{'✅' if sentence_ok else '❌'} 句子数量: {len(sentences):,} / 100,000 (目标)")
        print(f"{'✅' if vocab_ok else '❌'} 词汇量:   {len(vocab):,} / 10,000 (目标)")
        
        # 2. 句长分布
        lengths = [len(s) for s in sentences[:1000]]  # 采样前1000句
        avg_len = sum(lengths) / len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
        
        length_ok = 30 <= avg_len <= 100
        print(f"{'✅' if length_ok else '⚠️ '} 平均句长: {avg_len:.1f} (建议: 30-100)")
        print(f"  - 最短: {min_len}")
        print(f"  - 最长: {max_len}")
        
        # 3. 去重率
        dedup_rate = metadata.get('deduplication_rate', 0)
        dedup_ok = dedup_rate < 5.0
        print(f"{'✅' if dedup_ok else '⚠️ '} 去重率: {dedup_rate:.2f}% (建议: < 5%)")
        
        # 4. 数据完整性
        complete_ok = all([
            'sentences' in data,
            'vocab' in data,
            'metadata' in data
        ])
        print(f"{'✅' if complete_ok else '❌'} 数据完整性: {'完整' if complete_ok else '缺失字段'}")
        
        # 总结
        print("\n【验证结果】")
        all_ok = sentence_ok and vocab_ok and length_ok and dedup_ok and complete_ok
        
        if all_ok:
            print("🎉 所有检查通过！数据集质量合格，可以开始训练。")
            return True
        else:
            print("⚠️  部分检查未通过，建议:")
            if not sentence_ok:
                print("  - 重新运行 fetch_large_wikipedia.py 继续采集")
            if not vocab_ok:
                print("  - 增加采集轮数以扩充词汇量")
            if not length_ok:
                print("  - 数据可用，但句长分布略有偏差")
            if not dedup_ok:
                print("  - 去重率略高，但仍在可接受范围")
            return False
            
    except FileNotFoundError:
        print(f"❌ 错误: 文件 {file_path} 不存在")
        print("  请先运行: python fetch_large_wikipedia.py")
        return False
    except json.JSONDecodeError:
        print(f"❌ 错误: 文件 {file_path} 格式错误")
        return False
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        return False

if __name__ == '__main__':
    success = verify_dataset()
    sys.exit(0 if success else 1)
