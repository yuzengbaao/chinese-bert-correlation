"""
加载和分析训练结果 | Load and Analyze Training Results
=====================================================

这个脚本演示如何加载训练历史并进行自定义分析。
This script demonstrates how to load training history and perform custom analysis.
"""

import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_training_history(filepath='training_history_100k.json'):
    """加载训练历史 | Load training history"""
    print(f"📥 加载训练历史 | Loading training history from {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    steps = [item['step'] for item in data]
    mlm_acc = [item['mlm_accuracy'] for item in data]
    loss = [item['loss'] for item in data]
    
    print(f"✅ 加载完成 | Loaded {len(steps)} training steps")
    return np.array(steps), np.array(mlm_acc), np.array(loss)

def calculate_statistics(steps, mlm_acc):
    """计算统计指标 | Calculate statistics"""
    print("\n" + "=" * 60)
    print("📊 统计分析 | Statistical Analysis")
    print("=" * 60)
    
    # Pearson相关系数
    correlation, p_value = stats.pearsonr(steps, mlm_acc)
    r_squared = correlation ** 2
    
    print(f"Pearson相关系数 | Pearson r: {correlation:.4f}")
    print(f"R² (决定系数) | R-squared: {r_squared:.4f}")
    print(f"P值 | P-value: {p_value:.2e}")
    print(f"显著性 | Significance: {'✅ 显著' if p_value < 0.001 else '❌ 不显著'}")
    
    # 趋势分析
    coeffs = np.polyfit(steps, mlm_acc, deg=1)
    slope = coeffs[0]
    print(f"\n趋势斜率 | Trend slope: {slope:.6f}")
    print(f"每1000步提升 | Improvement per 1000 steps: {slope*1000:.4f}%")
    
    # 准确率统计
    print(f"\n准确率统计 | Accuracy Statistics:")
    print(f"  最小值 | Min: {mlm_acc.min():.2f}%")
    print(f"  最大值 | Max: {mlm_acc.max():.2f}%")
    print(f"  平均值 | Mean: {mlm_acc.mean():.2f}%")
    print(f"  标准差 | Std: {mlm_acc.std():.2f}%")
    
    return correlation, r_squared, slope

def plot_custom_analysis(steps, mlm_acc, loss):
    """自定义可视化 | Custom visualization"""
    print("\n📈 生成自定义图表 | Generating custom plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('自定义训练分析 | Custom Training Analysis', fontsize=16, fontweight='bold')
    
    # 图1：MLM准确率散点图
    ax1 = axes[0, 0]
    scatter = ax1.scatter(steps, mlm_acc, c=steps, cmap='viridis', alpha=0.6, s=10)
    ax1.plot(steps, np.poly1d(np.polyfit(steps, mlm_acc, 1))(steps), 
             'r--', linewidth=2, label='线性拟合')
    ax1.set_xlabel('训练步数 | Training Steps')
    ax1.set_ylabel('MLM准确率 (%) | MLM Accuracy (%)')
    ax1.set_title('MLM准确率变化趋势 | MLM Accuracy Trend')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='步数')
    
    # 图2：滑动窗口平均
    window = 100
    mlm_smooth = np.convolve(mlm_acc, np.ones(window)/window, mode='valid')
    ax2 = axes[0, 1]
    ax2.plot(steps[:len(mlm_smooth)], mlm_smooth, linewidth=2, color='blue')
    ax2.fill_between(steps[:len(mlm_smooth)], mlm_smooth, alpha=0.3)
    ax2.set_xlabel('训练步数 | Training Steps')
    ax2.set_ylabel('平滑MLM准确率 (%) | Smoothed MLM Accuracy (%)')
    ax2.set_title(f'滑动平均 (窗口={window}) | Moving Average (window={window})')
    ax2.grid(True, alpha=0.3)
    
    # 图3：损失函数分析
    ax3 = axes[1, 0]
    ax3.semilogy(steps, loss, color='red', linewidth=1, alpha=0.7)
    ax3.set_xlabel('训练步数 | Training Steps')
    ax3.set_ylabel('损失 (对数刻度) | Loss (log scale)')
    ax3.set_title('损失函数变化 | Loss Function Evolution')
    ax3.grid(True, alpha=0.3, which='both')
    
    # 图4：准确率分布直方图
    ax4 = axes[1, 1]
    ax4.hist(mlm_acc, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax4.axvline(mlm_acc.mean(), color='red', linestyle='--', 
                linewidth=2, label=f'平均值: {mlm_acc.mean():.2f}%')
    ax4.set_xlabel('MLM准确率 (%) | MLM Accuracy (%)')
    ax4.set_ylabel('频数 | Frequency')
    ax4.set_title('准确率分布 | Accuracy Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = 'results/custom_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存 | Plot saved to {output_path}")
    plt.close()

def main():
    """主函数 | Main function"""
    print("=" * 60)
    print("Chinese BERT 100K - 结果分析 | Result Analysis")
    print("=" * 60)
    
    # 加载数据
    steps, mlm_acc, loss = load_training_history()
    
    # 统计分析
    correlation, r_squared, slope = calculate_statistics(steps, mlm_acc)
    
    # 自定义可视化
    plot_custom_analysis(steps, mlm_acc, loss)
    
    print("\n" + "=" * 60)
    print("✅ 分析完成！| Analysis completed!")
    print("=" * 60)

if __name__ == '__main__':
    main()
