#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生僻字专项采集 - 针对性增加罕见汉字
策略：采集生僻字集中的特定条目
"""

import requests
import json
import time
import re
from collections import Counter
import os

class RareCharFetcher:
    def __init__(self, existing_file='large_wikipedia_dataset.json'):
        self.base_url = "https://zh.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # 加载现有数据
        print(f"📂 加载现有数据: {existing_file}")
        with open(existing_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.all_sentences = data['sentences']
        self.vocab_counter = Counter()
        
        # 重新统计词汇
        for sent in self.all_sentences:
            for char in sent:
                if '\u4e00' <= char <= '\u9fff':
                    self.vocab_counter[char] += 1
        
        print(f"   已有句子: {len(self.all_sentences):,}")
        print(f"   已有词汇: {len(self.vocab_counter):,}")
        print(f"   需增加: {max(0, 10000 - len(self.vocab_counter)):,}\n")
        
        # 生僻字集中的关键词
        self.rare_char_keywords = [
            # 姓氏（生僻姓）
            '中国姓氏', '复姓', '罕见姓氏', '百家姓',
            
            # 地名（含生僻字）
            '中国地名', '古代地名', '郡县', '州府', '乡镇名',
            
            # 动植物（学名含生僻字）
            '植物学名', '动物学名', '昆虫名', '鸟类名', '鱼类名',
            '药用植物', '中药材', '草本植物', '木本植物',
            
            # 矿物化石
            '矿物名称', '宝石名称', '化石名称', '岩石名称',
            
            # 古代文化
            '古代器物', '青铜器名', '陶瓷器', '玉器名称',
            '古代官职', '科举制度', '古代礼仪',
            
            # 中医药
            '中药方剂', '穴位名称', '经络名称', '病症名称',
            
            # 古籍典故
            '诗经', '尔雅', '说文解字', '康熙字典',
            '古代典籍', '文献名称',
            
            # 宗教文化
            '佛教术语', '道教术语', '寺庙名称', '道观名称',
            
            # 建筑名称
            '古代建筑', '宫殿名称', '城池名称', '园林名称',
            
            # 艺术门类
            '书法字体', '篆刻印章', '绘画技法', '戏曲剧种',
            
            # 天文历法
            '星宿名称', '节气', '干支', '天文术语',
            
            # 专业术语
            '纺织品名', '工艺名称', '乐器名称', '兵器名称'
        ]
    
    def get_specific_articles(self, keyword, count=50):
        """获取特定关键词的文章"""
        print(f"🔍 {keyword}")
        
        articles = []
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': keyword,
            'srnamespace': 0,
            'srlimit': count
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            for item in data.get('query', {}).get('search', []):
                articles.append(item['title'])
            
            print(f"   找到 {len(articles)} 篇")
        except Exception as e:
            print(f"   出错: {e}")
        
        return articles
    
    def fetch_and_extract_chars(self, title):
        """获取文章并提取新汉字"""
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            'explaintext': True
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if 'extract' in page_data:
                    text = page_data['extract']
                    
                    # 提取新汉字
                    new_chars = set()
                    for char in text:
                        if '\u4e00' <= char <= '\u9fff':
                            if char not in self.vocab_counter:
                                new_chars.add(char)
                            self.vocab_counter[char] += 1
                    
                    # 提取句子
                    sentences = re.split(r'[。！？\n]', text)
                    for sent in sentences:
                        sent = sent.strip()
                        if 10 <= len(sent) <= 200 and re.search(r'[\u4e00-\u9fff]', sent):
                            self.all_sentences.append(sent)
                    
                    return len(new_chars)
        except:
            pass
        
        return 0
    
    def process_category(self, keyword):
        """处理一个类别"""
        articles = self.get_specific_articles(keyword, count=50)
        
        if not articles:
            return 0
        
        new_char_count = 0
        for title in articles:
            new_chars = self.fetch_and_extract_chars(title)
            new_char_count += new_chars
        
        print(f"   +{new_char_count}新字 → 总词汇: {len(self.vocab_counter):,}\n")
        return new_char_count
    
    def save_dataset(self, filename='large_wikipedia_dataset.json'):
        """保存数据集"""
        print(f"\n💾 保存数据集...")
        
        # 去重句子
        self.all_sentences = list(set(self.all_sentences))
        
        # 构建词汇表（所有出现过的字）
        vocab_list = list(self.vocab_counter.keys())
        
        dataset = {
            'sentences': self.all_sentences,
            'vocab': vocab_list,
            'metadata': {
                'sentence_count': len(self.all_sentences),
                'vocab_size': len(vocab_list),
                'avg_sentence_length': sum(len(s) for s in self.all_sentences) / len(self.all_sentences),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'min_freq': 1
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"   ✓ 已保存\n")
        return dataset

def main():
    print('='*80)
    print('生僻字专项采集')
    print('='*80 + '\n')
    
    fetcher = RareCharFetcher()
    
    target = 10000
    current = len(fetcher.vocab_counter)
    
    if current >= target:
        print(f"✅ 已达标！({current:,} >= {target:,})")
        return
    
    print(f"📌 当前: {current:,} | 目标: {target:,} | 还需: {target - current:,}\n")
    print('='*80 + '\n')
    
    for idx, keyword in enumerate(fetcher.rare_char_keywords, 1):
        current = len(fetcher.vocab_counter)
        remaining = target - current
        
        print(f"[{idx}/{len(fetcher.rare_char_keywords)}] 还需{remaining}字")
        
        fetcher.process_category(keyword)
        
        current = len(fetcher.vocab_counter)
        if current >= target:
            print(f"\n🎉 词汇量达标！({current:,} >= {target:,})\n")
            break
        
        time.sleep(1)
    
    # 保存
    print('='*80)
    dataset = fetcher.save_dataset()
    
    print('='*80)
    print('最终统计')
    print('='*80)
    print(f"✅ 句子总数:   {dataset['metadata']['sentence_count']:,}")
    print(f"✅ 词汇量:     {dataset['metadata']['vocab_size']:,}")
    print('='*80 + '\n')
    
    if dataset['metadata']['vocab_size'] >= 10000:
        print("🎉🎉🎉 恭喜！词汇量达标！可以开始训练！")
    else:
        print(f"⚠️  还差 {10000 - dataset['metadata']['vocab_size']:,} 个词汇")

if __name__ == '__main__':
    main()
