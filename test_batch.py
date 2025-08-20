#!/usr/bin/env python3
"""
简单的训练测试脚本，用于验证修复
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from training_config import create_model_and_processor, NumericDataset, NumericDataCollator

def test_single_batch():
    """测试单个批次的前向传播"""
    print("=== 测试单个批次前向传播 ===")
    
    try:
        # 创建模型和处理器
        model, processor = create_model_and_processor(
            model_path="Qwen/Qwen2.5-VL-3B-Instruct"
        )
        
        # 创建一个小数据集
        dataset = NumericDataset(
            data_path="/data1/wangzhiye/1a1a11/original/data/numeric_training_data.json",
            processor=processor,
            image_folder="/data1/wangzhiye/1a1a11/original/data/images",
            max_length=512  # 使用较小的长度进行测试
        )
        
        # 创建数据整理器
        collator = NumericDataCollator(processor=processor)
        
        # 获取第一个样本
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{key}: {type(value)} = {value}")
        
        # 创建一个小批次
        batch = collator([sample])
        print(f"\nBatch keys: {batch.keys()}")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{key}: {type(value)}")
        
        # 测试前向传播
        print("\n测试前向传播...")
        model.eval()
        with torch.no_grad():
            outputs = model(**batch, return_dict=True)
            print(f"Outputs type: {type(outputs)}")
            if hasattr(outputs, 'loss'):
                print(f"Loss: {outputs.loss}")
            if hasattr(outputs, 'logits'):
                print(f"Logits shape: {outputs.logits.shape}")
        
        print("✅ 单个批次测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 单个批次测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_batch()
    if success:
        print("\n✅ 测试通过，可以尝试继续训练")
    else:
        print("\n❌ 测试失败，需要进一步调试")
