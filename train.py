#!/usr/bin/env python3
"""
数值增强Qwen2.5-VL模型训练脚本

基于原生Qwen2.5-VL架构的端到端训练
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import torch
import swanlab  # 替换wandb为swanlab
from transformers import Trainer, set_seed
from transformers.trainer_utils import get_last_checkpoint

# 添加当前目录到path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from numeric_qwen2_5_vl import (
    NumericQwen2_5_VLConfig,
    NumericQwen2_5_VLProcessor,
    NumericQwen2_5_VLForConditionalGeneration
)
from training_config import (
    NumericTrainingArguments,
    NumericDataset,
    NumericDataCollator,
    create_model_and_processor,
    get_training_config,
    create_deepspeed_config,
    init_swanlab,
    log_model_info_to_swanlab
)
from swanlab_trainer import (
    create_swanlab_trainer,
    log_sample_data_to_swanlab,
    log_training_progress_to_swanlab
)


class NumericTrainer(Trainer):
    """
    自定义训练器，支持数值增强功能
    """
    
    def __init__(self, numeric_loss_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.numeric_loss_weight = numeric_loss_weight
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        自定义损失计算
        """
        # 前向传播
        outputs = model(**inputs, return_dict=True)
        
        # 获取损失
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        else:
            # 如果模型没有返回loss，我们需要手动计算
            # 这种情况下通常意味着没有传入labels
            loss = torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
        
        return (loss, outputs) if return_outputs else loss


def main():
    """
    主训练函数
    """
    
    # 设置随机种子
    set_seed(42)
    
    # 获取训练配置
    training_args = get_training_config(
        output_dir="/data1/wangzhiye/1a1a11/original/output_new_jyb",
        model_path="/data1/wangzhiye/qwen253B",
        data_path="/data1/wangzhiye/LLaMA-Factory/data/6vqa_data_extracted_converted_numeric.json",
        val_data_path=None,
        test_data_path=None,
        image_folder=None,
        enable_swanlab=True,
        swanlab_project="qsinghua",
        swanlab_experiment="numeric_qwen2_5_vl_vqa_v1",
        exp_name="numeric_qwen2_5_vl_vqa_v1"
    )
    
    print("训练配置:")
    print(f"- 输出目录: {training_args.output_dir}")
    print(f"- 模型路径: {training_args.model_name_or_path}")
    print(f"- 训练数据路径: {training_args.data_path}")
    print(f"- 验证数据路径: {training_args.val_data_path}")
    print(f"- 测试数据路径: {training_args.test_data_path}")
    print(f"- 数值损失权重: {training_args.numeric_loss_weight}")
    print(f"- SwanLab项目: {training_args.swanlab_project}")
    print(f"- SwanLab实验: {training_args.swanlab_experiment}")
    print("- 评估策略: {training_args.eval_strategy}")
    if training_args.eval_strategy != "no":
        print(f"- 评估步数: {training_args.eval_steps}")
    
    # 初始化SwanLab
    swanlab_run = None
    if training_args.enable_swanlab:
        print("\n正在初始化SwanLab...")
        swanlab_run = init_swanlab(training_args)
    
    # 创建输出目录
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # 创建DeepSpeed配置
    deepspeed_config = create_deepspeed_config(
        output_file=os.path.join(training_args.output_dir, "ds_config.json"),
        gradient_accumulation_steps=training_args.gradient_accumulation_steps
    )
    training_args.deepspeed = deepspeed_config
    
    # 创建模型和处理器
    print("正在加载模型和处理器...")
    model, processor = create_model_and_processor(
        model_path=training_args.model_name_or_path,
        numeric_config={
            'numeric_embedding_dim': 512,
            'numeric_token': '<num>',
            'numeric_loss_weight': training_args.numeric_loss_weight
        }
    )
    
    print(f"模型配置: {model.config}")
    print(f"模型参数数量: {model.num_parameters():,}")
    
    # 记录模型信息到SwanLab
    if swanlab_run is not None:
        log_model_info_to_swanlab(swanlab_run, model, processor)
    
    # 创建数据集
    print("正在加载训练数据集...")
    train_dataset = NumericDataset(
        data_path=training_args.data_path,
        processor=processor,
        image_folder=training_args.image_folder,
        max_length=8192  # 增加到8192避免图像token被截断
    )
    print(f"加载了 {len(train_dataset)} 条训练数据")
    
    # 创建验证数据集（如果指定了）
    eval_dataset = None
    if training_args.val_data_path:
        print("正在加载验证数据集...")
        eval_dataset = NumericDataset(
            data_path=training_args.val_data_path,
            processor=processor,
            image_folder=training_args.image_folder,
            max_length=8192
        )
        print(f"加载了 {len(eval_dataset)} 条验证数据")
    else:
        print("未指定验证数据集")
    
    # 记录测试数据集信息（不加载，只记录路径）
    if training_args.test_data_path:
        print(f"测试数据集路径: {training_args.test_data_path}")
    else:
        print("未指定测试数据集")
    
    # 创建数据整理器
    data_collator = NumericDataCollator(processor=processor)
    
    # 记录数据样本到SwanLab
    if swanlab_run is not None:
        print("正在记录数据样本到SwanLab...")
        log_sample_data_to_swanlab(swanlab_run, train_dataset, processor, num_samples=5)
    
    # 检查是否有断点续训
    last_checkpoint = None
    
    
    # 创建训练器
    print("正在创建SwanLab集成训练器...")
    trainer = create_swanlab_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        swanlab_run=swanlab_run,
        processor=processor
    )
    
    # 开始训练
    print("开始训练...")
    try:
        # 记录训练开始信息到SwanLab
        if swanlab_run is not None:
            log_training_progress_to_swanlab(swanlab_run, {
                "training/dataset_size": len(train_dataset),
                "training/batch_size": training_args.per_device_train_batch_size,
                "training/gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "training/effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
                "training/total_epochs": training_args.num_train_epochs,
                "training/learning_rate": training_args.learning_rate,
                "training/vision_lr": training_args.vision_lr,
                "training/numeric_lr": training_args.numeric_lr,
            })
        
        if last_checkpoint is not None:
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            trainer.train()
        
        # 训练成功完成，保存最终模型
        print("保存最终模型...")
        
        # 1. 保存模型本身
        trainer.save_model()
        
        # 2. 保存处理器（完整保存）
        print("保存处理器...")
        processor.save_pretrained(training_args.output_dir)
        
        # 3. 保存生成配置
        if hasattr(model, 'generation_config') and model.generation_config is not None:
            try:
                model.generation_config.save_pretrained(training_args.output_dir)
                print("✅ generation_config 保存成功")
            except Exception as e:
                print(f"⚠️ generation_config 保存失败: {e}")
        
        # 4. 创建/更新 preprocessor_config.json（确保完整）
        import json
        preprocessor_config_path = os.path.join(training_args.output_dir, "preprocessor_config.json")
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
        
        with open(preprocessor_config_path, 'w', encoding='utf-8') as f:
            json.dump(preprocessor_config, f, indent=2, ensure_ascii=False)
        print("✅ preprocessor_config.json 已更新")
        
        # 5. 保存训练状态
        trainer.save_state()
        
        # 6. 验证保存结果
        saved_files = os.listdir(training_args.output_dir)
        print(f"最终模型保存的文件: {saved_files}")
        
        required_files = [
            "config.json",
            "added_tokens.json", 
            "special_tokens_map.json",
            "tokenizer_config.json",
            "preprocessor_config.json"
        ]
        
        missing_files = [f for f in required_files if f not in saved_files]
        if missing_files:
            print(f"⚠️ 缺失文件: {missing_files}")
        else:
            print("✅ 所有必要文件都已保存")
        
        # 记录训练完成信息到SwanLab
        if swanlab_run is not None:
            log_training_progress_to_swanlab(swanlab_run, {
                "training/status": 2.0,  # 2.0表示成功完成
                "training/model_saved": 1.0
            })
            print("✅ 训练信息已完整记录到SwanLab")
            
    except KeyboardInterrupt:
        print("训练被用户中断")
        # 记录中断信息到SwanLab
        if swanlab_run is not None:
            log_training_progress_to_swanlab(swanlab_run, {
                "training/status": 0.0,  # 0.0表示中断
                "training/interrupted": 1.0
            })
        # 即使中断也尝试保存当前状态
        print("尝试保存当前训练状态...")
        try:
            trainer.save_state()
        except:
            print("保存训练状态失败")
            
    except Exception as e:
        print(f"训练过程中出错: {e}")
        # 记录错误信息到SwanLab
        if swanlab_run is not None:
            log_training_progress_to_swanlab(swanlab_run, {
                "training/status": -1.0,  # -1.0表示失败
                "training/error_occurred": 1.0
            })
        # 即使出错也尝试保存当前状态
        print("尝试保存当前训练状态...")
        try:
            trainer.save_state()
        except:
            print("保存训练状态失败")
        raise
    
    print(f"训练完成！模型已保存到: {training_args.output_dir}")
    
    # 输出SwanLab链接
    if swanlab_run is not None:
        print(f"🔗 查看训练过程: https://swanlab.cn/{training_args.swanlab_project}")
        print(f"📊 实验名称: {training_args.swanlab_experiment}")
        try:
            swanlab_run.finish()
        except:
            pass

if __name__ == "__main__":
    main()
