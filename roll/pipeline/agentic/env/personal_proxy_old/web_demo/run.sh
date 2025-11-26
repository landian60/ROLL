#!/bin/bash

# Personal Proxy Web Demo 启动脚本

echo "Personal Proxy Web Demo 启动中..."

# 检查是否设置了 API Key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "警告: 未设置 DEEPSEEK_API_KEY 环境变量"
    echo "请设置: export DEEPSEEK_API_KEY='your-api-key'"
    echo ""
fi

# 检查依赖
if ! command -v python &> /dev/null; then
    echo "错误: 未找到 Python"
    exit 1
fi

# 安装依赖（如果需要）
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python -m venv venv
fi

echo "激活虚拟环境..."
source venv/bin/activate

echo "安装依赖..."
pip install -r requirements.txt -q

echo "启动应用..."
python app.py

