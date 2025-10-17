"""
准备GitHub Release附件文件
Prepare GitHub Release Assets
"""

import json
import os
from datetime import datetime

def prepare_release_assets():
    """准备发布附件"""
    print("=" * 60)
    print("准备GitHub Release v1.0.0附件 | Preparing Release Assets")
    print("=" * 60)
    
    assets = []
    
    # 1. 训练历史文件
    history_file = "training_history_100k.json"
    if os.path.exists(history_file):
        size = os.path.getsize(history_file) / 1024  # KB
        print(f"✅ {history_file}: {size:.2f} KB")
        assets.append(history_file)
    else:
        print(f"❌ {history_file}: 文件不存在")
    
    # 2. 分析结果文件
    analysis_file = "analysis_100k_result.json"
    if os.path.exists(analysis_file):
        size = os.path.getsize(analysis_file) / 1024  # KB
        print(f"✅ {analysis_file}: {size:.2f} KB")
        assets.append(analysis_file)
    else:
        print(f"❌ {analysis_file}: 文件不存在")
    
    # 3. 创建项目元信息文件
    metadata = {
        "version": "v1.0.0",
        "release_date": "2025-10-17",
        "project_name": "Chinese BERT 100K Training Correlation Study",
        "repository": "https://github.com/yuzengbaao/chinese-bert-correlation",
        "key_metrics": {
            "pearson_r": 0.7869,
            "r_squared": 0.6193,
            "p_value": "< 0.001",
            "mlm_accuracy": 50.53,
            "max_mlm_accuracy": 66.35,
            "training_steps": 100000,
            "training_time_hours": 22.5,
            "dataset_sentences": 325537,
            "vocabulary_size": 10049
        },
        "files": {
            "model": "stage4_large_100k_final.pth",
            "training_history": "training_history_100k.json",
            "analysis_results": "analysis_100k_result.json",
            "visualizations": [
                "results/training_curves.png",
                "results/correlation_analysis.png",
                "results/comparison_50k_100k.png",
                "results/loss_analysis.png"
            ]
        },
        "citation": {
            "format": "bibtex",
            "text": "@misc{chinese_bert_correlation_2025,\n  author = {Your Name},\n  title = {Chinese BERT 100K Training Correlation Study},\n  year = {2025},\n  publisher = {GitHub},\n  version = {v1.0.0},\n  url = {https://github.com/yuzengbaao/chinese-bert-correlation}\n}"
        }
    }
    
    metadata_file = "release_metadata_v1.0.0.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    size = os.path.getsize(metadata_file) / 1024
    print(f"✅ {metadata_file}: {size:.2f} KB (新创建)")
    assets.append(metadata_file)
    
    # 4. 可视化文件
    print("\n可视化文件 | Visualization Files:")
    viz_files = [
        "results/training_curves.png",
        "results/correlation_analysis.png",
        "results/comparison_50k_100k.png",
        "results/loss_analysis.png"
    ]
    
    for viz_file in viz_files:
        if os.path.exists(viz_file):
            size = os.path.getsize(viz_file) / 1024
            print(f"  ✅ {viz_file}: {size:.2f} KB")
        else:
            print(f"  ❌ {viz_file}: 文件不存在")
    
    # 5. 生成发布检查清单
    print("\n" + "=" * 60)
    print("GitHub Release检查清单 | Release Checklist")
    print("=" * 60)
    
    checklist = [
        ("创建Release", "访问: https://github.com/yuzengbaao/chinese-bert-correlation/releases/new"),
        ("设置Tag", "v1.0.0"),
        ("设置标题", "Chinese BERT 100K Training Correlation Study - v1.0.0"),
        ("复制说明", "从 RELEASE_NOTES_v1.0.0.md 复制内容"),
        ("上传附件1", f"training_history_100k.json"),
        ("上传附件2", f"analysis_100k_result.json"),
        ("上传附件3", f"release_metadata_v1.0.0.json"),
        ("设置选项", "✅ Set as the latest release"),
        ("发布", "点击 'Publish release'")
    ]
    
    for i, (task, detail) in enumerate(checklist, 1):
        print(f"{i}. {task}")
        print(f"   {detail}\n")
    
    # 6. 生成发布命令（如果安装了gh cli）
    print("=" * 60)
    print("使用GitHub CLI发布 (可选) | Using GitHub CLI (Optional)")
    print("=" * 60)
    
    gh_command = f"""gh release create v1.0.0 \\
  --title "Chinese BERT 100K Training Study - v1.0.0" \\
  --notes-file RELEASE_NOTES_v1.0.0.md \\
  {' '.join(assets)}"""
    
    print(gh_command)
    
    # 7. 保存发布命令到文件
    with open("github_release_command.sh", 'w', encoding='utf-8') as f:
        f.write("#!/bin/bash\n")
        f.write("# GitHub Release v1.0.0 创建命令\n")
        f.write("# 需要先安装 GitHub CLI: https://cli.github.com/\n\n")
        f.write(gh_command + "\n")
    
    print(f"\n✅ 命令已保存到: github_release_command.sh")
    
    # 8. 总结
    print("\n" + "=" * 60)
    print("准备完成 | Preparation Complete")
    print("=" * 60)
    print(f"可用附件文件: {len(assets)} 个")
    for asset in assets:
        print(f"  - {asset}")
    
    print("\n下一步:")
    print("1. 访问 https://github.com/yuzengbaao/chinese-bert-correlation/releases/new")
    print("2. 按照上述检查清单操作")
    print("3. 或使用 GitHub CLI 运行: bash github_release_command.sh")

if __name__ == '__main__':
    prepare_release_assets()
