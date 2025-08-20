# SwanLab状态码说明

由于SwanLab只支持数值类型的图表数据，我们使用数值状态码来表示不同的训练状态：

## 训练状态码 (training/status)

- **1.0**: 训练开始 (Started)
- **2.0**: 训练成功完成 (Completed Successfully)  
- **0.0**: 训练被中断 (Interrupted)
- **-1.0**: 训练失败 (Failed)

## 评估状态码 (eval/status)

- **1.0**: 评估开始 (Evaluation Started)
- **2.0**: 评估完成 (Evaluation Completed)
- **-1.0**: 评估失败 (Evaluation Failed)

## 布尔值状态码

- **1.0**: True/是/成功
- **0.0**: False/否/失败

## 示例指标

```python
# 训练开始
{
    "training/status": 1.0,
    "training/resume_from_checkpoint": 1.0  # 如果从检查点恢复
}

# 训练完成
{
    "training/status": 2.0,
    "training/model_saved": 1.0,
    "training/total_time": 3600.5,
    "training/total_steps": 1000
}

# 错误状态
{
    "training/status": -1.0,
    "training/error_occurred": 1.0
}
```

这样设计的好处是：
1. 所有状态都能在SwanLab中正确显示为图表
2. 可以很容易地在图表中看到训练的不同阶段
3. 数值状态码有明确的含义且易于理解
