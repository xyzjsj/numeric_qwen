# 数值增强Qwen2.5-VL模型

基于原生Qwen2.5-VL架构的数值增强多模态大模型，支持 `<num><value>` 格式的数值token处理。

## 特性

- **原生架构**: 基于Qwen2_5_VLForConditionalGeneration，无需复杂的组件拼接
- **数值增强**: 支持专门的数值token `<num><value>` 格式
- **双重输出**: 同时输出文本logits和数值预测
- **混合损失**: 结合交叉熵损失和均方误差损失
- **端到端训练**: 支持直接的端到端训练，无需分阶段训练

## 文件结构

```
/data1/wangzhiye/1a1a11/original/
├── numeric_qwen2_5_vl.py      # 核心模型实现
├── training_config.py         # 训练配置和数据处理
├── train.py                   # 训练脚本
├── inference.py               # 推理脚本
├── prepare_data.py            # 数据准备工具
├── README.md                  # 说明文档
└── data/                      # 数据目录
    ├── numeric_training_data.json
    └── images/
```

## 快速开始

### 1. 数据准备

```bash
cd /data1/wangzhiye/1a1a11/original
python prepare_data.py
```

这将创建示例训练数据和图像。

### 2. 训练模型

```bash
python train.py
```

### 3. 推理测试

```bash
python inference.py
```

## 数据格式

训练数据采用JSON格式，支持多轮对话和图像：

```json
{
  "id": "sample_001",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n这个图表中的数值是多少？"
    },
    {
      "from": "gpt", 
      "value": "根据图表显示，主要数值包括：\n1. 销售额：<num><1234.56>万元\n2. 增长率：<num><15.8>%"
    }
  ],
  "image": "chart_001.jpg"
}
```

## 数值Token格式

使用 `<num><value>` 格式表示数值：

- `<num><3.14159>` - 表示π的近似值
- `<num><-273.15>` - 表示绝对零度
- `<num><1.618>` - 表示黄金比例

## 模型架构

### 核心组件

1. **NumericQwen2_5_VLConfig**: 扩展配置类
   - 添加数值处理相关参数
   - 保持与原生Qwen2.5-VL的兼容性

2. **NumericQwen2_5_VLProcessor**: 数值增强处理器
   - 自动提取文本中的数值token
   - 提供数值位置信息

3. **NumericQwen2_5_VLForConditionalGeneration**: 核心模型类
   - 数值嵌入网络：1 → 512 → hidden_size
   - 回归头：hidden_size → 1
   - 混合损失计算

### 损失函数

```python
total_loss = token_loss + α * numeric_loss
```

- `token_loss`: 交叉熵损失（文本生成）
- `numeric_loss`: 均方误差损失（数值预测）
- `α`: 数值损失权重（默认1.0）

## 配置参数

### 训练参数

```python
training_args = NumericTrainingArguments(
    output_dir="./output",
    model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
    
    # 学习率
    learning_rate=1e-5,
    vision_lr=2e-6,       # 视觉编码器学习率
    numeric_lr=1e-4,      # 数值层学习率
    
    # 数值特定参数
    numeric_loss_weight=1.0,
    
    # 训练设置
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    bf16=True,
    gradient_checkpointing=True
)
```

### 模型参数

```python
numeric_config = {
    'numeric_embedding_dim': 512,    # 数值嵌入维度
    'numeric_token': '<num>',        # 数值token
    'numeric_loss_weight': 1.0       # 损失权重
}
```

## 使用示例

### 训练自定义数据

```python
from training_config import create_model_and_processor, NumericDataset

# 创建模型
model, processor = create_model_and_processor(
    model_path="Qwen/Qwen2.5-VL-3B-Instruct",
    numeric_config={'numeric_loss_weight': 1.5}
)

# 创建数据集
dataset = NumericDataset(
    data_path="your_data.json",
    processor=processor,
    image_folder="your_images/"
)
```

### 推理使用

```python
from inference import NumericQwen2_5_VLInference

# 创建推理引擎
inference = NumericQwen2_5_VLInference("./output")

# 文本推理
result = inference.generate_response("π的值是多少？")
print(result['text'])  # "π的近似值是<num><3.14159>"
print(result['numeric_predictions'])  # [{"value": 3.14159, ...}]

# 图像推理
result = inference.generate_response(
    "分析这个图表中的数据", 
    image="chart.jpg"
)
```

## 性能优化

### DeepSpeed配置

自动生成DeepSpeed ZeRO-2配置：

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "cpu"}
  }
}
```

### 内存优化

- 使用`bf16`精度训练
- 启用`gradient_checkpointing`
- 支持ZeRO优化器状态分片

## 与原始实现的对比

| 特性 | 原始custom_qwen.py | 新实现 |
|------|------------------|--------|
| 基础架构 | 从头实现Qwen2.5-VL | 继承原生Qwen2_5_VLForConditionalGeneration |
| 兼容性 | 需要手动适配 | 完全兼容Transformers |
| 训练复杂度 | 需要深度定制 | 标准训练流程 |
| 维护成本 | 高 | 低 |
| 扩展性 | 有限 | 良好 |

## 环境要求

```bash
# 激活conda环境
conda activate llava

# 安装依赖
pip install torch transformers accelerate deepspeed wandb
pip install pillow datasets
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少`per_device_train_batch_size`
   - 增加`gradient_accumulation_steps`
   - 启用DeepSpeed ZeRO-3

2. **数值token未被识别**
   - 检查tokenizer是否正确添加了`<num>`token
   - 验证`num_token_id`配置

3. **损失不收敛**
   - 调整`numeric_loss_weight`
   - 检查学习率设置
   - 验证数据格式

### 调试模式

在训练脚本中启用调试信息：

```python
# 在训练过程中会打印损失详情
# Token Loss: 2.3456, Float Loss: 0.1234
```

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 许可证

Apache 2.0 License

## 致谢

- 基于Qwen2.5-VL原生架构
- 参考LLaVA-NeXT训练框架
- 感谢Transformers库的强大支持
