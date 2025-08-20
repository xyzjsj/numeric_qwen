#!/usr/bin/env python3
"""
快速测试LoRA训练配置和模型加载
"""

import os
import sys
import torch
from transformers import set_seed

# 添加当前目录到path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training_config import get_training_config, create_model_and_processor


def test_lora_config():
    """测试LoRA配置"""
    print("=== 测试LoRA配置 ===")
    
    # 获取LoRA训练配置
    training_args = get_training_config(
        output_dir="./test_output",
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        data_path="./data/numeric_training_data.json",
        image_folder="./data/images",
        
        # 启用LoRA
        use_lora=True,
        lora_r=8,  # 较小的r值用于测试
        lora_alpha=16,
        lora_dropout=0.1,
        
        # 较小的训练参数用于测试
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        
        exp_name="lora_test"
    )
    
    print(f"使用LoRA: {training_args.use_lora}")
    print(f"LoRA r: {training_args.lora_r}")
    print(f"LoRA alpha: {training_args.lora_alpha}")
    print(f"LoRA dropout: {training_args.lora_dropout}")
    print(f"目标模块: {training_args.lora_target_modules}")
    print(f"学习率: {training_args.learning_rate}")
    print(f"批次大小: {training_args.per_device_train_batch_size}")
    
    return training_args


def test_lora_model_loading():
    """测试LoRA模型加载"""
    print("\n=== 测试LoRA模型加载 ===")
    
    try:
        # 创建LoRA模型
        model, processor = create_model_and_processor(
            model_path="Qwen/Qwen2.5-VL-3B-Instruct",
            use_lora=True,
            lora_config={
                'r': 8,
                'lora_alpha': 16,
                'lora_dropout': 0.1,
                'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"]
            }
        )
        
        print("✅ LoRA模型加载成功")
        
        # 检查PEFT配置
        if hasattr(model, 'peft_config'):
            print(f"✅ PEFT配置存在: {list(model.peft_config.keys())}")
            
            # 打印LoRA配置详情
            for adapter_name, config in model.peft_config.items():
                print(f"适配器 {adapter_name}:")
                print(f"  - r: {config.r}")
                print(f"  - lora_alpha: {config.lora_alpha}")
                print(f"  - lora_dropout: {config.lora_dropout}")
                print(f"  - target_modules: {config.target_modules}")
                print(f"  - modules_to_save: {getattr(config, 'modules_to_save', None)}")
        else:
            print("❌ PEFT配置不存在")
            return False
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print(f"可训练参数比例: {100 * trainable_params / total_params:.4f}%")
        
        # 检查特定模块是否可训练
        print("\n可训练模块:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  - {name}: {param.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ LoRA模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_model_loading():
    """测试完整模型加载（对比）"""
    print("\n=== 测试完整模型加载（对比） ===")
    
    try:
        # 创建完整模型
        model, processor = create_model_and_processor(
            model_path="Qwen/Qwen2.5-VL-3B-Instruct",
            use_lora=False
        )
        
        print("✅ 完整模型加载成功")
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print(f"可训练参数比例: {100 * trainable_params / total_params:.4f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ 完整模型加载失败: {e}")
        return False


def main():
    """主测试函数"""
    print("LoRA配置和模型加载测试")
    print("=" * 50)
    
    # 设置随机种子
    set_seed(42)
    
    # 测试1: LoRA配置
    training_args = test_lora_config()
    
    # 测试2: LoRA模型加载
    lora_success = test_lora_model_loading()
    
    # 测试3: 完整模型加载（对比）
    full_success = test_full_model_loading()
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结:")
    print(f"LoRA模型加载: {'✅ 成功' if lora_success else '❌ 失败'}")
    print(f"完整模型加载: {'✅ 成功' if full_success else '❌ 失败'}")
    
    if lora_success:
        print("\n✅ LoRA配置正确，可以开始训练！")
        print("\n下一步:")
        print("1. 检查数据集路径是否正确")
        print("2. 运行 python train.py 开始LoRA训练")
        print("3. 训练完成后使用 inference_lora.py 进行推理")
    else:
        print("\n❌ LoRA配置有问题，请检查依赖和配置")
    
    return lora_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
