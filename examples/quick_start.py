"""
快速开始示例 | Quick Start Example
=================================

这个脚本演示如何快速加载和使用训练好的模型。
This script demonstrates how to quickly load and use the trained model.
"""

import torch
from transformers import BertTokenizer, BertForMaskedLM
import json

def load_model(model_path='stage4_large_100k_final.pth'):
    """加载训练好的模型 | Load the trained model"""
    print(f"📥 加载模型 | Loading model from {model_path}...")
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 初始化模型
    model = BertForMaskedLM.from_pretrained('bert-base-chinese')
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ 模型加载成功！| Model loaded successfully!")
    return model, tokenizer

def predict_masked_word(model, tokenizer, text):
    """预测mask位置的词 | Predict the masked word"""
    print(f"\n🔍 输入 | Input: {text}")
    
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt')
    
    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    
    # 获取mask位置
    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    
    if len(mask_token_index) == 0:
        print("❌ 未找到[MASK]标记 | No [MASK] token found")
        return None
    
    # 获取top-5预测
    mask_token_logits = predictions[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    
    print("📊 Top 5 预测 | Predictions:")
    for i, token_id in enumerate(top_5_tokens, 1):
        token = tokenizer.decode([token_id])
        print(f"   {i}. {token}")
    
    return tokenizer.decode([top_5_tokens[0]])

def main():
    """主函数 | Main function"""
    print("=" * 60)
    print("Chinese BERT 100K - 快速开始 | Quick Start")
    print("=" * 60)
    
    # 加载模型
    model, tokenizer = load_model()
    
    # 示例1：简单预测
    print("\n" + "=" * 60)
    print("示例 1 | Example 1: Simple Prediction")
    print("=" * 60)
    predict_masked_word(model, tokenizer, "今天天气很[MASK]。")
    
    # 示例2：更复杂的句子
    print("\n" + "=" * 60)
    print("示例 2 | Example 2: Complex Sentence")
    print("=" * 60)
    predict_masked_word(model, tokenizer, "人工智能是计算机[MASK]的一个重要分支。")
    
    # 示例3：专业领域
    print("\n" + "=" * 60)
    print("示例 3 | Example 3: Domain-Specific")
    print("=" * 60)
    predict_masked_word(model, tokenizer, "深度学习模型的训练需要大量[MASK]。")
    
    print("\n" + "=" * 60)
    print("✅ 演示完成！| Demo completed!")
    print("=" * 60)

if __name__ == '__main__':
    main()
