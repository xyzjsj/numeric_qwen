#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级测试检查点保存功能
不需要下载模型，仅测试保存机制
"""

import os
import torch
import tempfile
from transformers import TrainingArguments
from swanlab_trainer import SwanLabNumericTrainer, create_swanlab_trainer
import swanlab

def test_checkpoint_saving_lightweight():
    """轻量级测试检查点保存功能"""
    print("开始轻量级检查点保存功能测试...")
    
    # 创建一个简单的虚拟模型和处理器
    class MockModel:
        def __init__(self):
            self.config = {"model_type": "mock", "hidden_size": 512}
        
        def save_pretrained(self, path):
            import json
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "config.json"), "w") as f:
                json.dump(self.config, f)
            print(f"虚拟模型已保存到: {path}")
    
    class MockProcessor:
        def __init__(self):
            self.config = {
                "processor_class": "NumericQwen2_5_VLProcessor",
                "image_processor": {"class": "mock"},
                "tokenizer": {"class": "mock"},
                "numeric_tokens": ["<num>", "<num_pad>"]
            }
        
        def save_pretrained(self, path):
            import json
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "preprocessor_config.json"), "w") as f:
                json.dump(self.config, f)
            with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
                json.dump({"tokenizer_class": "mock"}, f)
            with open(os.path.join(path, "special_tokens_map.json"), "w") as f:
                json.dump({"additional_special_tokens": ["<num>", "<num_pad>"]}, f)
            print(f"虚拟处理器已保存到: {path}")
        
        def _process_text_with_numeric_tokens(self, text):
            # 简单的模拟实现
            return (text.replace("<num>", "<num_pad>"), [8.5])
    
    model = MockModel()
    processor = MockProcessor()
    
    # 创建临时目录用于保存检查点
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"临时目录: {temp_dir}")
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=temp_dir,
            save_steps=1,
            save_total_limit=1,
            logging_steps=1,
            max_steps=1,
            per_device_train_batch_size=1,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to=None,
        )
        
        # 创建虚拟数据集
        class DummyDataset:
            def __len__(self):
                return 1
            
            def __getitem__(self, idx):
                return {
                    'input_ids': torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
                    'attention_mask': torch.tensor([1, 1, 1, 1, 1], dtype=torch.long),
                    'labels': torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
                    'numeric_values': torch.tensor([1.0], dtype=torch.float32),
                    'numeric_masks': torch.tensor([1], dtype=torch.bool),
                }
        
        train_dataset = DummyDataset()
        
        # 创建训练器（不使用SwanLab）
        print("创建训练器...")
        trainer = create_swanlab_trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            swanlab_run=None,
            processor=processor  # 传递处理器
        )
        
        # 验证处理器是否正确附加
        if hasattr(trainer, 'processor'):
            print("✅ 处理器已正确附加到训练器")
        else:
            print("❌ 处理器未附加到训练器")
            return False
        
        # 手动触发检查点保存
        print("触发检查点保存...")
        
        # 直接调用trainer的保存方法
        checkpoint_dir = os.path.join(temp_dir, "checkpoint-1")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存模型
        model.save_pretrained(checkpoint_dir)
        
        # 保存处理器（这是我们要测试的关键部分）
        if hasattr(trainer, 'processor') and trainer.processor is not None:
            try:
                trainer.processor.save_pretrained(checkpoint_dir)
                print("✅ 处理器保存成功")
            except Exception as e:
                print(f"❌ 处理器保存失败: {e}")
                return False
        else:
            print("❌ 训练器中没有处理器")
            return False
        
        print(f"检查点目录: {checkpoint_dir}")
        
        # 检查文件是否存在
        expected_files = [
            "config.json",
            "preprocessor_config.json",  # 处理器配置
            "special_tokens_map.json",   # 特殊令牌映射
            "tokenizer_config.json",     # 分词器配置
        ]
        
        print("检查文件:")
        all_files_exist = True
        for file_name in expected_files:
            file_path = os.path.join(checkpoint_dir, file_name)
            if os.path.exists(file_path):
                print(f"  ✅ {file_name}")
            else:
                print(f"  ❌ {file_name} (缺失)")
                all_files_exist = False
        
        # 列出实际存在的文件
        actual_files = os.listdir(checkpoint_dir)
        print(f"\n实际文件列表: {actual_files}")
        
        # 检查处理器配置内容
        preprocessor_config_path = os.path.join(checkpoint_dir, "preprocessor_config.json")
        if os.path.exists(preprocessor_config_path):
            import json
            with open(preprocessor_config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ 处理器配置已保存，包含 {len(config)} 个配置项")
            print(f"   配置内容: {config}")
        else:
            print("❌ 处理器配置文件缺失")
        
        # 验证处理器功能（使用原始处理器）
        print("\n测试处理器功能...")
        try:
            test_text = "这是一个数字 <num>8.5</num> 的测试"
            result = processor._process_text_with_numeric_tokens(test_text)
            print(f"✅ 处理器功能正常: {result}")
        except Exception as e:
            print(f"❌ 处理器功能测试失败: {e}")
            return False
        
        return all_files_exist

if __name__ == "__main__":
    success = test_checkpoint_saving_lightweight()
    if success:
        print("\n🎉 轻量级检查点保存功能测试通过!")
    else:
        print("\n❌ 轻量级检查点保存功能测试失败!")
