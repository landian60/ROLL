#!/bin/bash

# Personal Proxy Web Demo 启动脚本（在 roll conda 环境中）
# 使用本地 Qwen2.5-14B-Instruct-GPTQ-Int8 模型进行意图判断

echo "=========================================="
echo "Personal Proxy Web Demo (使用本地 LLM)"
echo "=========================================="

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 设置 API Key（用于多模态模型，意图判断使用本地模型）
export DASHSCOPE_API_KEY=sk-6b9b2d78b40040959f7914801c14714d

echo ""
echo "正在加载本地 LLM 模型..."
echo "模型路径: ~/.cache/modelscope/hub/qwen/Qwen2___5-14B-Instruct-GPTQ-Int8/"
echo "注意：首次启动时会加载模型，可能需要30秒左右"
echo ""

# 启动应用
echo "启动 Flask 应用..."
echo "=========================================="
CUDA_VISIBLE_DEVICES=1 python app.py

