#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大规模训练脚本 - 100K步训练
目标: 在100K+句子数据集上训练，达到Pearson r > 0.85
优化: 更大批量, 更多epoch, 梯度累积
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import logging
import os
import random
import numpy as np
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_100k_internal.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 修复PyTorch随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==================== 模型定义 ====================

class BERTModel(nn.Module):
    """BERT模型 - 与之前保持一致"""
    def __init__(self, vocab_size, hidden_size=512, num_layers=12, num_heads=8, ff_size=2048, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(512, hidden_size)
        self.token_type_embedding = nn.Embedding(2, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        self.nsp_head = nn.Linear(hidden_size, 2)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, token_type_ids, attention_mask=None):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        
        embeddings = self.embedding(input_ids) + self.position_embedding(position_ids) + self.token_type_embedding(token_type_ids)
        embeddings = self.dropout(embeddings)
        
        if attention_mask is not None:
            attention_mask = attention_mask.float()
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        hidden_states = self.transformer(embeddings, src_key_padding_mask=attention_mask)
        
        mlm_logits = self.mlm_head(hidden_states)
        cls_output = hidden_states[:, 0, :]
        nsp_logits = self.nsp_head(cls_output)
        
        return mlm_logits, nsp_logits

# ==================== 数据集 ====================

class LargeBERTDataset(Dataset):
    """大规模BERT数据集"""
    def __init__(self, sentences, vocab, max_len=128):
        self.sentences = sentences
        self.vocab = vocab
        self.max_len = max_len
        
        # 构建字符映射
        self.char2idx = {char: idx + 4 for idx, char in enumerate(vocab)}
        self.char2idx.update({
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': len(vocab) + 4
        })
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        
        logger.info(f"Dataset initialized: {len(sentences):,} sentences, vocab size: {len(self.char2idx):,}")
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        # 获取两个句子（50%相邻，50%随机）
        sent1 = self.sentences[idx]
        
        if random.random() > 0.5 and idx + 1 < len(self.sentences):
            sent2 = self.sentences[idx + 1]
            nsp_label = 0  # IsNext
        else:
            sent2 = self.sentences[random.randint(0, len(self.sentences) - 1)]
            nsp_label = 1  # NotNext
        
        # 编码
        tokens1 = [self.char2idx.get(c, self.char2idx['[UNK]']) for c in sent1]
        tokens2 = [self.char2idx.get(c, self.char2idx['[UNK]']) for c in sent2]
        
        # 截断
        max_len_per_sent = (self.max_len - 3) // 2
        tokens1 = tokens1[:max_len_per_sent]
        tokens2 = tokens2[:max_len_per_sent]
        
        # 拼接: [CLS] sent1 [SEP] sent2 [SEP]
        tokens = [self.char2idx['[CLS]']] + tokens1 + [self.char2idx['[SEP]']] + tokens2 + [self.char2idx['[SEP]']]
        token_type_ids = [0] * (len(tokens1) + 2) + [1] * (len(tokens2) + 1)
        
        # Padding
        padding_len = self.max_len - len(tokens)
        tokens += [self.char2idx['[PAD]']] * padding_len
        token_type_ids += [0] * padding_len
        
        return {
            'input_ids': torch.tensor(tokens[:self.max_len], dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids[:self.max_len], dtype=torch.long),
            'nsp_label': torch.tensor(nsp_label, dtype=torch.long)
        }

def create_mlm_task(batch, mask_prob=0.15, device='cuda'):
    """创建MLM任务"""
    input_ids = batch['input_ids'].clone().to(device)
    labels = input_ids.clone()
    
    # 随机mask
    prob_matrix = torch.rand(input_ids.shape, device=device)
    mask_arr = (prob_matrix < mask_prob).float()
    
    # 不mask特殊token
    special_tokens = [0, 2, 3]  # [PAD], [CLS], [SEP]
    for token_id in special_tokens:
        mask_arr = mask_arr * (input_ids != token_id).float()
    
    # 应用mask
    mask_idx = mask_arr.bool()
    labels[~mask_idx] = -100
    
    # 80% mask, 10% random, 10% keep
    mask_token_id = input_ids.max().item() + 1
    random_tokens = torch.randint(4, input_ids.max().item(), input_ids.shape, device=device)
    
    prob_replace = torch.rand(input_ids.shape, device=device)
    input_ids[mask_idx] = torch.where(
        prob_replace[mask_idx] < 0.8,
        torch.tensor(mask_token_id, device=device),
        torch.where(
            prob_replace[mask_idx] < 0.9,
            random_tokens[mask_idx],
            input_ids[mask_idx]
        )
    )
    
    return input_ids, labels

# ==================== 训练器 ====================

class Trainer:
    """训练器 - 支持梯度累积"""
    def __init__(self, model, train_loader, device='cuda', lr=5e-5, total_steps=100000, 
                 warmup_steps=2000, accumulation_steps=4, checkpoint_dir='checkpoints_100k'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.total_steps = total_steps
        self.accumulation_steps = accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 优化器
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        # 学习率调度器
        self.scheduler = self.get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )
        
        # 损失函数
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.nsp_criterion = nn.CrossEntropyLoss()
        
        # 统计
        self.history = {
            'steps': [],
            'mlm_acc': [],
            'nsp_acc': [],
            'loss': [],
            'lr': []
        }
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  Total steps: {total_steps:,}")
        logger.info(f"  Warmup steps: {warmup_steps:,}")
        logger.info(f"  Gradient accumulation: {accumulation_steps}")
        logger.info(f"  Effective batch size: {train_loader.batch_size * accumulation_steps}")
    
    @staticmethod
    def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
        """余弦退火学习率"""
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def train(self, start_step=0, log_interval=100, save_interval=1000):
        """训练主循环"""
        self.model.train()
        global_step = start_step
        self.optimizer.zero_grad()
        
        logger.info(f"\n{'='*80}")
        logger.info("Starting training...")
        logger.info(f"{'='*80}\n")
        
        epoch = 0
        while global_step < self.total_steps:
            epoch += 1
            logger.info(f"Epoch {epoch}")
            
            for batch_idx, batch in enumerate(self.train_loader):
                # MLM任务
                input_ids, mlm_labels = create_mlm_task(batch, device=self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                nsp_labels = batch['nsp_label'].to(self.device)
                
                # 前向传播
                mlm_logits, nsp_logits = self.model(input_ids, token_type_ids)
                
                # 计算损失
                mlm_loss = self.mlm_criterion(
                    mlm_logits.view(-1, mlm_logits.size(-1)),
                    mlm_labels.view(-1)
                )
                nsp_loss = self.nsp_criterion(nsp_logits, nsp_labels)
                loss = mlm_loss + nsp_loss
                
                # 反向传播（梯度累积）
                loss = loss / self.accumulation_steps
                loss.backward()
                
                # 更新参数
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # 记录指标
                    if global_step % log_interval == 0:
                        mlm_acc = self.calculate_mlm_accuracy(mlm_logits, mlm_labels)
                        nsp_acc = (nsp_logits.argmax(dim=1) == nsp_labels).float().mean().item()
                        
                        self.history['steps'].append(global_step)
                        self.history['mlm_acc'].append(mlm_acc)
                        self.history['nsp_acc'].append(nsp_acc)
                        self.history['loss'].append(loss.item() * self.accumulation_steps)
                        self.history['lr'].append(self.scheduler.get_last_lr()[0])
                        
                        logger.info(
                            f"Step: {global_step:,} | "
                            f"MLM Acc: {mlm_acc:.4f} | "
                            f"NSP Acc: {nsp_acc:.4f} | "
                            f"Loss: {loss.item() * self.accumulation_steps:.4f} | "
                            f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                        )
                    
                    # 保存检查点
                    if global_step % save_interval == 0:
                        self.save_checkpoint(global_step)
                    
                    # 检查是否完成
                    if global_step >= self.total_steps:
                        break
            
            if global_step >= self.total_steps:
                break
        
        # 保存最终模型
        self.save_final_model()
        logger.info(f"\n✅ Training completed! Total steps: {global_step:,}\n")
    
    @staticmethod
    def calculate_mlm_accuracy(logits, labels):
        """计算MLM准确度"""
        preds = logits.argmax(dim=-1)
        mask = labels != -100
        if mask.sum() == 0:
            return 0.0
        correct = (preds == labels) & mask
        return (correct.sum() / mask.sum()).item()
    
    def save_checkpoint(self, step):
        """保存检查点"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'step_{step}_checkpoint.pth')
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def save_final_model(self):
        """保存最终模型"""
        model_path = 'stage4_large_100k_final.pth'
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"💾 Final model saved: {model_path}")
        
        # 保存训练历史
        history_path = 'training_history_100k.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"💾 Training history saved: {history_path}")

# ==================== 主函数 ====================

def main():
    """主函数"""
    logger.info('\n' + '='*80)
    logger.info('大规模BERT训练 - 100K步实验')
    logger.info('='*80 + '\n')
    
    # 1. 加载数据
    logger.info("Loading dataset...")
    with open('large_wikipedia_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sentences = data['sentences']
    vocab = data['vocab']
    
    logger.info(f"✅ Dataset loaded:")
    logger.info(f"   Sentences: {len(sentences):,}")
    logger.info(f"   Vocab size: {len(vocab):,}\n")
    
    # 2. 创建数据集和加载器
    dataset = LargeBERTDataset(sentences, vocab, max_len=128)
    
    # 更大的批量大小（从16增至32）
    train_loader = DataLoader(
        dataset,
        batch_size=32,  # 增大批量
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    logger.info(f"✅ DataLoader created:")
    logger.info(f"   Batch size: 32")
    logger.info(f"   Gradient accumulation: 4")
    logger.info(f"   Effective batch size: 128\n")
    
    # 3. 创建模型
    vocab_size = len(dataset.char2idx)
    model = BERTModel(
        vocab_size=vocab_size,
        hidden_size=512,
        num_layers=12,
        num_heads=8,
        ff_size=2048,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Model created:")
    logger.info(f"   Parameters: {total_params/1e6:.2f}M\n")
    
    # 4. 创建训练器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}\n")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        device=device,
        lr=5e-5,
        total_steps=100000,
        warmup_steps=2000,
        accumulation_steps=4,
        checkpoint_dir='checkpoints_100k'
    )
    
    # 5. 开始训练
    trainer.train(
        start_step=0,
        log_interval=100,
        save_interval=2000  # 每2000步保存一次
    )
    
    logger.info("\n🎉 All done!\n")

if __name__ == '__main__':
    main()
