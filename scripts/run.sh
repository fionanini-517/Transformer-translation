#!/bin/bash

# 设置字符编码
export LANG=en_US.UTF-8

echo "==============================================="
echo "      Transformer Translation Training"
echo "==============================================="
echo ""

# 切换到项目根目录（scripts的父目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Script directory: $SCRIPT_DIR"
echo "Project root directory: $PROJECT_ROOT"
echo ""

# 检查操作系统并设置Python路径
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    # Linux/macOS/WSL
    PYTHON_PATH="/mnt/e/Anaconda/envs/pytorch-gpu/python"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows Git Bash
    PYTHON_PATH="E:/Anaconda/envs/pytorch-gpu/python.exe"
else
    # 默认使用Windows路径
    PYTHON_PATH="E:/Anaconda/envs/pytorch-gpu/python.exe"
fi

echo "Using Python: $PYTHON_PATH"
echo ""

# 检查Python是否存在
if ! [ -f "$PYTHON_PATH" ]; then
    echo "Error: Python not found at $PYTHON_PATH"
    echo "Please check your Python path in the script."
    echo "Trying to use 'python' command instead..."
    PYTHON_PATH="python"
fi

echo "Checking GPU status..."
"$PYTHON_PATH" -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU device:', torch.cuda.get_device_name(0))
    print('GPU count:', torch.cuda.device_count())
else:
    print('Using CPU')
"

echo ""
echo "Setting up environment and starting training..."

# 检查数据目录是否存在
if [ -d ".data" ]; then
    echo "✓ Data directory found: .data/"
else
    echo "✗ Data directory not found: .data/"
    echo "Please make sure your data is in the correct location."
fi

# 创建结果目录
mkdir -p results
echo "✓ Results directory ready: results/"

echo "==============================================="
echo "           Starting Training..."
echo "==============================================="

# 运行训练脚本
echo "Running: $PYTHON_PATH src/train.py"
"$PYTHON_PATH" src/train.py

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed with exit code: $EXIT_CODE"
    echo "Please check the error messages above."
fi

# 等待用户按键
echo ""
read -p "Press Enter to exit..."