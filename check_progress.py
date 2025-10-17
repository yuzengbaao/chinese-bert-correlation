#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速检查数据采集进度
"""

import os
import json

def check_progress():
    """检查数据采集进度"""
    
    dataset_file = 'large_wikipedia_dataset.json'
    
    if os.path.exists(dataset_file):
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            sentences = data.get('sentences', [])
            vocab = data.get('vocab', [])
            
            print(f"✅ 当前句子数: {len(sentences):,}")
            print(f"✅ 当前词汇数: {len(vocab):,}")
            print(f"📊 完成度: {len(sentences)/100000*100:.1f}%")
            
            if len(sentences) >= 100000:
                print("🎉 数据采集已完成！")
                return True
            else:
                print(f"⏳ 还需: {100000-len(sentences):,} 句子")
                return False
        except:
            print("⚠️  文件正在写入中...")
            return False
    else:
        print("⏳ 数据采集进行中，文件尚未生成...")
        return False

if __name__ == '__main__':
    check_progress()
