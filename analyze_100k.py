#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析100K步训练结果并计算Pearson相关系数
"""

import json
import numpy as np
from datetime import datetime

def analyze_100k_training():
    """分析100K步训练结果"""
    
    print("=" * 80)
    print("100K步训练 - 最终分析报告")
    print("=" * 80)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 加载训练历史
        with open('training_history_100k.json', 'r') as f:
            history = json.load(f)
        
        steps = np.array(history['steps'])
        mlm_acc = np.array(history['mlm_acc'])
        nsp_acc = np.array(history['nsp_acc'])
        losses = np.array(history['loss'])
        
        # 基础统计
        print("【1】训练完成状态")
        print(f"  总步数:     {len(steps):,}")
        print(f"  最终步数:   {steps[-1]:,}")
        print(f"  数据点数:   {len(steps):,}")
        print()
        
        # MLM准确度统计
        print("【2】MLM准确度统计")
        print(f"  平均值:     {np.mean(mlm_acc):.4f} ({np.mean(mlm_acc)*100:.2f}%)")
        print(f"  最大值:     {np.max(mlm_acc):.4f} ({np.max(mlm_acc)*100:.2f}%)")
        print(f"  最小值:     {np.min(mlm_acc):.4f} ({np.min(mlm_acc)*100:.2f}%)")
        print(f"  标准差:     {np.std(mlm_acc):.4f}")
        print(f"  提升幅度:   {(np.max(mlm_acc) - np.min(mlm_acc))*100:.2f} 百分点")
        print()
        
        # 计算Pearson相关系数
        print("【3】Pearson相关性分析")
        correlation = np.corrcoef(steps, mlm_acc)[0, 1]
        
        print(f"  Pearson r:  {correlation:.4f}")
        print(f"  r²:         {correlation**2:.4f} ({correlation**2*100:.1f}% 方差解释)")
        
        # 判断相关性强度
        if correlation >= 0.90:
            strength = "极强正相关 ⭐⭐⭐"
        elif correlation >= 0.85:
            strength = "很强正相关 ⭐⭐"
        elif correlation >= 0.80:
            strength = "强正相关 ⭐"
        elif correlation >= 0.70:
            strength = "中等偏强正相关"
        else:
            strength = "中等正相关"
        
        print(f"  强度评价:   {strength}")
        print()
        
        # 目标验证
        print("【4】实验目标验证")
        print(f"  目标相关系数: r >= 0.85")
        print(f"  实际相关系数: r = {correlation:.4f}")
        
        if correlation >= 0.85:
            achievement = (correlation - 0.85) / 0.15 * 100
            print(f"  达成状态:     ✅ 已达成目标！")
            print(f"  超额完成:     {achievement:.1f}% (相对于0.85-1.0区间)")
        else:
            achievement = correlation / 0.85 * 100
            gap = 0.85 - correlation
            print(f"  达成状态:     ⚠️  未达目标")
            print(f"  完成度:       {achievement:.1f}%")
            print(f"  差距:         {gap:.4f}")
        print()
        
        # 与50K训练对比
        print("【5】与50K训练对比")
        
        baseline_r = 0.6359
        baseline_mlm = 0.1450
        
        r_improvement = correlation - baseline_r
        mlm_improvement = np.mean(mlm_acc) - baseline_mlm
        
        print(f"  50K训练 Pearson r:     {baseline_r:.4f}")
        print(f"  100K训练 Pearson r:    {correlation:.4f}")
        print(f"  相关性提升:            {r_improvement:+.4f} ({r_improvement/baseline_r*100:+.1f}%)")
        print()
        print(f"  50K训练 平均MLM:       {baseline_mlm:.4f}")
        print(f"  100K训练 平均MLM:      {np.mean(mlm_acc):.4f}")
        print(f"  准确度提升:            {mlm_improvement:+.4f} ({mlm_improvement/baseline_mlm*100:+.1f}%)")
        print()
        
        # 损失统计
        print("【6】损失函数统计")
        print(f"  起始损失:   {losses[0]:.4f}")
        print(f"  最终损失:   {losses[-1]:.4f}")
        print(f"  损失下降:   {losses[0] - losses[-1]:.4f}")
        print()
        
        # 最近10步详情
        print("【7】最近10步详情")
        for i in range(-10, 0):
            print(f"  Step {steps[i]:,}: MLM={mlm_acc[i]:.4f}, Loss={losses[i]:.4f}")
        print()
        
        # 保存分析结果
        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'total_steps': int(steps[-1]),
            'data_points': len(steps),
            'pearson_correlation': float(correlation),
            'r_squared': float(correlation**2),
            'mlm_stats': {
                'mean': float(np.mean(mlm_acc)),
                'max': float(np.max(mlm_acc)),
                'min': float(np.min(mlm_acc)),
                'std': float(np.std(mlm_acc))
            },
            'loss_stats': {
                'initial': float(losses[0]),
                'final': float(losses[-1]),
                'decrease': float(losses[0] - losses[-1])
            },
            'comparison_with_50k': {
                '50k_pearson_r': baseline_r,
                '100k_pearson_r': float(correlation),
                'improvement': float(r_improvement),
                'improvement_percentage': float(r_improvement/baseline_r*100)
            },
            'goal_achieved': bool(correlation >= 0.85)  # 显式转换为bool
        }
        
        with open('analysis_100k_result.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        
        print("【8】文件输出")
        print("  ✓ analysis_100k_result.json (分析结果)")
        print()
        
        # 总结
        print("=" * 80)
        print("【总结】")
        if correlation >= 0.85:
            print("🎉 恭喜！实验成功达成目标 (Pearson r >= 0.85)")
            print("   通过扩大数据集规模和增加训练步数，相关性显著提升。")
            print("   证明了训练步数与MLM准确度之间存在很强的正相关关系。")
        elif correlation >= 0.80:
            print("✅ 实验接近目标 (Pearson r >= 0.80)")
            print(f"   当前相关性 {correlation:.4f}，距离目标 0.85 还差 {0.85-correlation:.4f}。")
            print("   建议: 继续增加数据量或延长训练步数。")
        else:
            print("⚠️  实验未达目标")
            print(f"   当前相关性 {correlation:.4f}，建议:")
            print("   1. 进一步扩大数据集至150K+句子")
            print("   2. 延长训练至150K步")
            print("   3. 调整学习率schedule")
        
        print("=" * 80)
        
        return correlation >= 0.85
        
    except FileNotFoundError:
        print("❌ 错误: 找不到 training_history_100k.json")
        print("   请确认训练已完成")
        return False
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    import sys
    success = analyze_100k_training()
    sys.exit(0 if success else 1)
