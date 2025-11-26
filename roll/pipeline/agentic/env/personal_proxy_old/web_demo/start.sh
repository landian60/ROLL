#!/bin/bash

# Personal Proxy Web Demo 启动脚本（在 roll conda 环境中）

echo "正在启动 Personal Proxy Web Demo..."

# # 激活 conda 环境
# conda activate roll

# 切换到脚本所在目录
cd "$(dirname "$0")"

export DASHSCOPE_API_KEY=sk-6b9b2d78b40040959f7914801c14714d

# 启动应用
echo "启动 Flask 应用..."
echo "=========================================="
python app.py

