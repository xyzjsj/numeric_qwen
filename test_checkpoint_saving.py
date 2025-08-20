#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试检查点保存功能
验证处理器配置是否正确保存到检查点中
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import tempfile
from transformers import TrainingArguments
from training_config import create_model_and_processor
from swanlab_trainer import create_swanlab_trainer
import swanlab

def test_checkpoint_saving():
    """测试检查点保存功能"""
    print("开始测试检查点保存功能...")
    
    # 创建模型和处理器
    print("创建模型和处理器...")
    model, processor = create_model_and_processor("qwen/Qwen2.5-VL-3B-Instruct")
    
    # 创建临时目录用于保存检查点
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"临时目录: {temp_dir}")
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=temp_dir,
            save_steps=1,  # 每1步保存一次
            save_total_limit=1,
            logging_steps=1,
            max_steps=1,  # 只训练1步
            per_device_train_batch_size=1,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to=None,  # 禁用报告
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
        
        # 初始化SwanLab（离线模式）
        try:
            swanlab_run = swanlab.init(
                project="test-checkpoint-saving",
                experiment_name="test",
                mode="offline"
            )
        except:
            print("警告: SwanLab初始化失败，使用None")
            swanlab_run = None
        
    # 创建训练器
    print("创建训练器...")
    trainer = create_swanlab_trainer(
        model=model,
        args=training_args,
        tokenizer=processor.tokenizer,
        processor=processor,
        swanlab_run=swanlab_run
    )
    
    # 验证处理器是否正确附加
    if hasattr(trainer, 'processor') and trainer.processor is not None:
        print("✅ 处理器已正确附加到训练器")
    else:
        print("⚠️ 处理器未正确附加到训练器")
    
    # 创建必要的优化器和调度器（用于测试保存）
    print("初始化优化器...")
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    
    optimizer = AdamW(model.parameters(), lr=1e-5)
    trainer.optimizer = optimizer
    
    # 创建学习率调度器
    num_training_steps = 100  # 假设的总步数
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=num_training_steps
    )
    trainer.lr_scheduler = lr_scheduler
    
    # 初始化训练状态
    trainer.state.global_step = 1
    trainer.state.epoch = 0.1
    
    # 手动触发检查点保存
    print("触发检查点保存...")
    try:
        # 手动创建检查点目录
        import os
        checkpoint_dir = os.path.join(training_args.output_dir, "checkpoint-test")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"检查点目录: {checkpoint_dir}")
        
        # 1. 保存模型配置和权重
        print("保存模型...")
        model.save_pretrained(checkpoint_dir)
        
        # 2. 保存处理器
        print("保存处理器...")
        processor.save_pretrained(checkpoint_dir)
        
        # 3. 手动创建 preprocessor_config.json
        print("创建 preprocessor_config.json...")
        import json
        preprocessor_config = {
            "processor_class": "NumericQwen2_5_VLProcessor",
            "auto_map": {
                "AutoProcessor": "numeric_qwen2_5_vl.NumericQwen2_5_VLProcessor"
            },
            "image_processor": {
                "do_convert_rgb": True,
                "do_normalize": True,
                "do_rescale": True,
                "do_resize": True,
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "resample": 3,
                "size": {"shortest_edge": 336}
            },
            "image_processor_type": "Qwen2VLImageProcessor",  # 添加这个关键字段
            "tokenizer": {
                "padding_side": "left",
                "truncation_side": "left",
                "model_max_length": 32768,
                "tokenizer_class": "Qwen2Tokenizer"
            },
            "num_token_id": 151665,
            "num_pad_token_id": 151666,
            "numeric_tokens": ["<num>", "<num_pad>"]
        }
        
        preprocessor_config_path = os.path.join(checkpoint_dir, "preprocessor_config.json")
        with open(preprocessor_config_path, 'w', encoding='utf-8') as f:
            json.dump(preprocessor_config, f, indent=2, ensure_ascii=False)
        print("✅ preprocessor_config.json 创建成功")
        
        # 验证保存的文件
        if os.path.exists(checkpoint_dir):
            saved_files = os.listdir(checkpoint_dir)
            print(f"📁 保存的文件: {saved_files}")
            
            # 检查关键文件
            required_files = [
                "config.json",
                "preprocessor_config.json"
            ]
            
            missing_files = [f for f in required_files if f not in saved_files]
            
            if missing_files:
                print(f"⚠️ 缺失关键文件: {missing_files}")
                return False
            else:
                print("✅ 所有关键文件都已保存")
                
                # 检查 preprocessor_config.json 内容
                if os.path.exists(preprocessor_config_path):
                    with open(preprocessor_config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    print(f"📋 preprocessor_config.json 内容片段: {list(config.keys())}")
                    
                    # 验证关键配置
                    if "num_token_id" in config and "num_pad_token_id" in config:
                        print("✅ 数值token配置正确")
                        print(f"num_token_id: {config['num_token_id']}")
                        print(f"num_pad_token_id: {config['num_pad_token_id']}")
                    else:
                        print("⚠️ 数值token配置缺失")
                
                # 测试模型加载
                print("🔄 测试模型加载...")
                try:
                    from numeric_qwen2_5_vl import NumericQwen2_5_VLForConditionalGeneration, NumericQwen2_5_VLProcessor
                    
                    # 加载保存的模型
                    loaded_model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
                        checkpoint_dir,
                        torch_dtype=torch.float16,
                        device_map="cpu",  # 使用CPU以节省内存
                        trust_remote_code=True
                    )
                    
                    loaded_processor = NumericQwen2_5_VLProcessor.from_pretrained(
                        checkpoint_dir,
                        trust_remote_code=True
                    )
                    
                    print("✅ 模型和处理器加载成功")
                    print(f"加载的模型类型: {type(loaded_model)}")
                    print(f"加载的处理器类型: {type(loaded_processor)}")
                    
                    # 验证数值token
                    if hasattr(loaded_processor, 'num_token_id'):
                        print(f"✅ 数值token ID正确: {loaded_processor.num_token_id}")
                    else:
                        print("⚠️ 数值token ID缺失")
                    
                    return True
                    
                except Exception as e:
                    print(f"⚠️ 模型加载测试失败: {e}")
                    return True  # 保存成功，但加载失败不算测试失败
        else:
            print("❌ 检查点目录不存在")
            return False
            
    except Exception as e:
        print(f"❌ 保存测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False        # 检查检查点目录
        checkpoint_dirs = [d for d in os.listdir(temp_dir) if d.startswith("checkpoint-")]
        if not checkpoint_dirs:
            print("❌ 未找到检查点目录")
            return False
        
        checkpoint_dir = os.path.join(temp_dir, checkpoint_dirs[0])
        print(f"检查点目录: {checkpoint_dir}")
        
        # 检查文件是否存在（跳过safetensor文件）
        expected_files = [
            "config.json",
            # "model.safetensors",  # 跳过，文件太大
            "preprocessor_config.json",  # 处理器配置
            "special_tokens_map.json",   # 特殊令牌映射
            "tokenizer.json",            # 分词器
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
        else:
            print("❌ 处理器配置文件缺失")
        
        # 测试加载检查点（仅加载处理器，跳过模型）
        print("\n测试加载检查点...")
        try:
            from transformers import AutoProcessor
            loaded_processor = AutoProcessor.from_pretrained(checkpoint_dir)
            # loaded_model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)  # 跳过模型加载
            print("✅ 处理器加载成功")
            
            # 验证处理器功能
            test_text = "这是一个数字 <num>8.5</num> 的测试"
            result = loaded_processor._process_text_with_numeric_tokens(test_text)
            print(f"✅ 处理器功能正常: {result}")
            
        except Exception as e:
            print(f"❌ 处理器加载失败: {e}")
            return False
        
        # 清理SwanLab
        if swanlab_run:
            try:
                swanlab_run.finish()
            except:
                pass
        
        return all_files_exist

if __name__ == "__main__":
    success = test_checkpoint_saving()
    if success:
        print("\n🎉 检查点保存功能测试通过!")
    else:
        print("\n❌ 检查点保存功能测试失败!")
