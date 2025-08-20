# 数值增强Qwen2.5-VL模型 - SwanLab可视化训练

本项目集成了SwanLab可视化工具，用于实时监控和分析数值增强Qwen2.5-VL模型的训练过程。

## ? SwanLab项目信息

- **项目名称**: `qsinghua`
- **查看地址**: https://swanlab.cn/qsinghua
- **功能特点**: 训练损失可视化、模型参数监控、数据样本展示、实验对比

## ? 快速开始

### 1. 环境准备

确保已安装SwanLab：
```bash
pip install swanlab
```

登录SwanLab账号：
```bash
swanlab login
# 或在Python中登录
# import swanlab
# swanlab.login(api_key="your_api_key")
```

### 2. 测试SwanLab集成

在开始正式训练前，建议先测试SwanLab集成是否正常：

```bash
python test_swanlab.py
```

或使用启动脚本的测试模式：
```bash
python start_swanlab_training.py --test_only
```

### 3. 启动可视化训练

#### 方式一：使用启动脚本（推荐）

```bash
# 使用默认配置启动训练
python start_swanlab_training.py

# 自定义配置
python start_swanlab_training.py \
    --data_path /path/to/your/data.json \
    --image_folder /path/to/images \
    --output_dir /path/to/output \
    --swanlab_project qsinghua \
    --swanlab_experiment my_experiment_v1 \
    --epochs 5 \
    --batch_size 4 \
    --learning_rate 2e-5
```

#### 方式二：直接运行训练脚本

```bash
python train.py
```

## ? SwanLab可视化功能

### 训练监控指标

训练过程中会自动记录以下指标到SwanLab：

1. **损失指标**
   - `train/loss` - 总体训练损失
   - `loss/total_loss` - 详细总损失
   - `loss/numeric_loss` - 数值损失组件
   - `loss/language_loss` - 语言模型损失组件

2. **训练进度**
   - `train/epoch` - 当前训练轮数
   - `train/learning_rate` - 当前学习率
   - `time/elapsed_time` - 总训练时间
   - `time/step_time` - 单步训练时间

3. **模型信息**
   - `model/total_parameters` - 模型总参数量
   - `model/trainable_parameters` - 可训练参数量
   - `model/vocab_size` - 词汇表大小
   - `model/num_token_id` - 数值token ID

4. **训练配置**
   - 所有训练超参数
   - 数据集信息
   - 模型架构参数

5. **数据样本展示**
   - 训练数据样本预览
   - 数值标注信息
   - 图像处理信息

### 实验管理功能

- **实验对比**: 在SwanLab界面中对比不同实验的性能
- **实时监控**: 远程查看训练进度，支持手机查看
- **硬件监控**: 自动记录GPU、CPU、内存使用情况
- **日志记录**: 完整的训练日志记录
- **模型版本管理**: 自动记录检查点信息

## ?? 配置选项

### 训练配置

在 `training_config.py` 中可以配置SwanLab相关参数：

```python
@dataclass
class NumericTrainingArguments(TrainingArguments):
    # SwanLab可视化参数
    swanlab_project: str = "qsinghua"  # SwanLab项目名称
    swanlab_experiment: str = None     # 实验名称（自动生成）
    enable_swanlab: bool = True        # 是否启用SwanLab
```

### 启动脚本选项

```bash
python start_swanlab_training.py --help
```

可用参数：
- `--swanlab_project`: SwanLab项目名称（默认: qsinghua）
- `--swanlab_experiment`: 实验名称（默认: 自动生成时间戳）
- `--disable_swanlab`: 禁用SwanLab可视化
- `--test_only`: 仅测试SwanLab集成
- `--data_path`: 训练数据路径
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--learning_rate`: 学习率

## ? 查看训练结果

### 在线查看

访问 https://swanlab.cn/qsinghua 查看所有实验结果。

### 主要监控内容

1. **训练曲线**
   - 损失函数变化趋势
   - 学习率调度曲线
   - 训练时间分析

2. **模型分析**
   - 参数量统计
   - 数值token处理效果
   - 词汇表扩展情况

3. **数据分析**
   - 数据集样本分布
   - 数值标注统计
   - 图像处理情况

4. **硬件性能**
   - GPU利用率
   - 内存使用情况
   - 训练效率分析

## ? 故障排除

### 常见问题

1. **SwanLab连接失败**
   ```bash
   # 检查网络连接
   ping swanlab.cn
   
   # 重新登录
   swanlab login
   ```

2. **指标记录失败**
   - 检查SwanLab API密钥是否正确
   - 确认项目名称是否存在
   - 查看终端错误信息

3. **实验不显示**
   - 确认实验名称没有重复
   - 检查项目权限设置
   - 刷新浏览器页面

### 调试模式

启用详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 离线模式

如果网络不稳定，可以禁用SwanLab：
```bash
python start_swanlab_training.py --disable_swanlab
```

## ? 更多资源

- [SwanLab官方文档](https://docs.swanlab.cn/)
- [SwanLab GitHub](https://github.com/SwanHubX/SwanLab)
- [Qwen2.5-VL模型文档](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

## ? 贡献

欢迎提交Issue和Pull Request来改进SwanLab集成功能！
