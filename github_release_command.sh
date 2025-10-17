#!/bin/bash
# GitHub Release v1.0.0 创建命令
# 需要先安装 GitHub CLI: https://cli.github.com/

gh release create v1.0.0 \
  --title "Chinese BERT 100K Training Study - v1.0.0" \
  --notes-file RELEASE_NOTES_v1.0.0.md \
  training_history_100k.json analysis_100k_result.json release_metadata_v1.0.0.json
