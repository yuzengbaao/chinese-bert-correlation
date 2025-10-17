#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练结果可视化脚本
Visualization script for training results

生成图表:
1. 训练曲线 (Training curves)
2. 相关性散点图 (Correlation scatter plot)
3. 50K vs 100K对比图 (Comparison chart)
4. 损失函数曲线 (Loss curve)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_training_history(file_path='training_history_100k.json'):
    """加载训练历史数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_training_curves(history, save_path='results/training_curves.png'):
    """绘制完整训练曲线"""
    steps = np.array(history['steps'])
    mlm_acc = np.array(history['mlm_acc'])
    nsp_acc = np.array(history['nsp_acc'])
    losses = np.array(history['loss'])
    lr = np.array(history['lr'])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('100K Steps Training Curves | 100K步训练曲线', fontsize=16, fontweight='bold')
    
    # 1. MLM准确度
    axes[0, 0].plot(steps, mlm_acc, linewidth=2, color='#2E86AB', alpha=0.8)
    axes[0, 0].set_xlabel('Training Steps | 训练步数', fontsize=12)
    axes[0, 0].set_ylabel('MLM Accuracy | MLM准确度', fontsize=12)
    axes[0, 0].set_title('MLM Accuracy Progress | MLM准确度进展', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% baseline')
    axes[0, 0].legend()
    
    # 2. NSP准确度
    axes[0, 1].plot(steps, nsp_acc, linewidth=2, color='#A23B72', alpha=0.8)
    axes[0, 1].set_xlabel('Training Steps | 训练步数', fontsize=12)
    axes[0, 1].set_ylabel('NSP Accuracy | NSP准确度', fontsize=12)
    axes[0, 1].set_title('NSP Accuracy Progress | NSP准确度进展', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random guess')
    axes[0, 1].legend()
    
    # 3. 损失函数
    axes[1, 0].plot(steps, losses, linewidth=2, color='#F18F01', alpha=0.8)
    axes[1, 0].set_xlabel('Training Steps | 训练步数', fontsize=12)
    axes[1, 0].set_ylabel('Loss | 损失', fontsize=12)
    axes[1, 0].set_title('Training Loss | 训练损失', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 学习率
    axes[1, 1].plot(steps, lr, linewidth=2, color='#C73E1D', alpha=0.8)
    axes[1, 1].set_xlabel('Training Steps | 训练步数', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate | 学习率', fontsize=12)
    axes[1, 1].set_title('Learning Rate Schedule | 学习率调度', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练曲线已保存: {save_path}")
    plt.close()

def plot_correlation_analysis(history, save_path='results/correlation_analysis.png'):
    """绘制相关性分析图"""
    steps = np.array(history['steps'])
    mlm_acc = np.array(history['mlm_acc'])
    
    # 计算Pearson相关系数
    correlation, p_value = stats.pearsonr(steps, mlm_acc)
    
    # 线性拟合
    slope, intercept = np.polyfit(steps, mlm_acc, 1)
    fit_line = slope * steps + intercept
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Correlation Analysis | 相关性分析 (Pearson r = {correlation:.4f})', 
                 fontsize=16, fontweight='bold')
    
    # 1. 散点图 + 拟合线
    axes[0].scatter(steps, mlm_acc, alpha=0.5, s=20, color='#2E86AB', label='Actual data')
    axes[0].plot(steps, fit_line, 'r--', linewidth=2, label=f'Linear fit (r={correlation:.4f})')
    axes[0].set_xlabel('Training Steps | 训练步数', fontsize=12)
    axes[0].set_ylabel('MLM Accuracy | MLM准确度', fontsize=12)
    axes[0].set_title('Steps vs MLM Accuracy | 步数与准确度关系', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 添加统计信息
    r_squared = correlation ** 2
    text_str = f'Pearson r: {correlation:.4f}\n'
    text_str += f'R²: {r_squared:.4f}\n'
    text_str += f'p-value: {p_value:.2e}\n'
    text_str += f'Slope: {slope:.2e}'
    axes[0].text(0.05, 0.95, text_str, transform=axes[0].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. 残差图
    residuals = mlm_acc - fit_line
    axes[1].scatter(steps, residuals, alpha=0.5, s=20, color='#A23B72')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Training Steps | 训练步数', fontsize=12)
    axes[1].set_ylabel('Residuals | 残差', fontsize=12)
    axes[1].set_title('Residual Plot | 残差图', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 相关性分析图已保存: {save_path}")
    plt.close()

def plot_comparison_50k_vs_100k(save_path='results/comparison_50k_100k.png'):
    """绘制50K vs 100K对比图"""
    # 数据
    metrics = ['Pearson r', 'Avg MLM\nAccuracy', 'Max MLM\nAccuracy', 'Loss\nReduction']
    values_50k = [0.6359, 0.145, 0.3188, 0.668]  # 标准化的值
    values_100k = [0.7869, 0.5053, 0.6635, 0.668]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width/2, values_50k, width, label='50K Training', 
                   color='#FFB6C1', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, values_100k, width, label='100K Training', 
                   color='#87CEEB', alpha=0.8, edgecolor='black')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Metrics | 指标', fontsize=13)
    ax.set_ylabel('Value | 数值', fontsize=13)
    ax.set_title('50K vs 100K Training Comparison | 训练对比', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加改进百分比
    improvements = [(values_100k[i] - values_50k[i])/values_50k[i]*100 for i in range(len(metrics))]
    for i, improvement in enumerate(improvements):
        ax.text(i, max(values_50k[i], values_100k[i]) + 0.05, 
               f'+{improvement:.1f}%',
               ha='center', fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图已保存: {save_path}")
    plt.close()

def plot_loss_analysis(history, save_path='results/loss_analysis.png'):
    """绘制损失函数详细分析"""
    steps = np.array(history['steps'])
    losses = np.array(history['loss'])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Loss Function Analysis | 损失函数分析', fontsize=16, fontweight='bold')
    
    # 1. 损失曲线 + 移动平均
    window_size = 50
    losses_smooth = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
    steps_smooth = steps[:len(losses_smooth)]
    
    axes[0].plot(steps, losses, alpha=0.3, color='gray', linewidth=0.5, label='Raw loss')
    axes[0].plot(steps_smooth, losses_smooth, linewidth=2, color='#F18F01', label=f'MA({window_size})')
    axes[0].set_xlabel('Training Steps | 训练步数', fontsize=12)
    axes[0].set_ylabel('Loss | 损失', fontsize=12)
    axes[0].set_title('Loss Curve with Moving Average | 损失曲线（含移动平均）', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 损失下降率
    loss_change = np.diff(losses)
    axes[1].plot(steps[1:], loss_change, linewidth=1, color='#C73E1D', alpha=0.6)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Training Steps | 训练步数', fontsize=12)
    axes[1].set_ylabel('Loss Change Rate | 损失变化率', fontsize=12)
    axes[1].set_title('Loss Change Rate | 损失变化率', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 损失分析图已保存: {save_path}")
    plt.close()

def main():
    """主函数"""
    print("=" * 80)
    print("训练结果可视化 | Training Results Visualization")
    print("=" * 80)
    print()
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 加载训练历史
    print("【1】加载训练历史数据...")
    history = load_training_history()
    print(f"  ✓ 已加载 {len(history['steps'])} 个数据点")
    print()
    
    # 生成图表
    print("【2】生成可视化图表...")
    plot_training_curves(history)
    plot_correlation_analysis(history)
    plot_comparison_50k_vs_100k()
    plot_loss_analysis(history)
    print()
    
    print("=" * 80)
    print("✅ 所有图表已生成完成！")
    print("   查看 results/ 目录获取所有图表")
    print("=" * 80)

if __name__ == '__main__':
    main()
