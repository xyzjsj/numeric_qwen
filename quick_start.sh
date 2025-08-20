#!/bin/bash

# 数值增强Qwen2.5-VL模型快速启动脚本

set -e

echo "=== 数值增强Qwen2.5-VL模型快速启动 ==="

# 检查conda环境
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到conda命令"
    exit 1
fi

# 激活环境
echo "1. 激活conda环境..."
eval "$(conda shell.bash hook)"
conda activate llava || {
    echo "错误: 无法激活llava环境"
    echo "请确保已创建llava conda环境"
    exit 1
}

# 检查Python包
echo "2. 检查依赖包..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
    echo "正在安装PyTorch..."
    pip install torch torchvision torchaudio
}

python -c "import transformers; print(f'Transformers: {transformers.__version__}')" || {
    echo "正在安装Transformers..."
    pip install transformers>=4.44.0
}

python -c "import accelerate; print('Accelerate: OK')" || {
    echo "正在安装Accelerate..."
    pip install accelerate
}

python -c "import deepspeed; print('DeepSpeed: OK')" || {
    echo "正在安装DeepSpeed..."
    pip install deepspeed
}

python -c "import wandb; print('WandB: OK')" || {
    echo "正在安装WandB..."
    pip install wandb
}

python -c "from PIL import Image; print('Pillow: OK')" || {
    echo "正在安装Pillow..."
    pip install pillow
}

# 设置工作目录
cd /data1/wangzhiye/1a1a11/original

echo "3. 准备训练数据..."
if [ ! -f "data/numeric_training_data.json" ]; then
    echo "创建示例训练数据..."
    python prepare_data.py
else
    echo "训练数据已存在"
fi

echo "4. 检查模型配置..."
python -c "
import sys
sys.path.append('.')
from numeric_qwen2_5_vl import NumericQwen2_5_VLConfig
print('✓ 模型配置检查通过')
"

echo "5. 开始训练..."
echo "是否开始训练？(y/N): "
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "启动训练..."
    python train.py
else
    echo "跳过训练"
fi

echo ""
echo "=== 可用命令 ==="
echo "训练模型:     python train.py"
echo "准备数据:     python prepare_data.py"
echo "推理测试:     python inference.py"
echo "查看数据:     ls -la data/"
echo ""
echo "=== 训练配置建议 ==="
echo "小显存(8GB):  batch_size=1, grad_accum=16"
echo "中显存(16GB): batch_size=2, grad_accum=8"
echo "大显存(24GB): batch_size=4, grad_accum=4"
echo ""
echo "启动完成！"
