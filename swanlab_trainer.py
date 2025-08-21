#!/usr/bin/env python3
"""
集成SwanLab的数值增强训练器

专门为数值增强Qwen2.5-VL模型设计的SwanLab集成训练器
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from transformers import Trainer
import swanlab
import time
from training_config import log_to_swanlab


class SwanLabNumericTrainer(Trainer):
    """
    集成SwanLab的数值增强训练器
    """
    
    def __init__(self, swanlab_run=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.swanlab_run = swanlab_run
        self.start_time = time.time()
        self.step_start_time = time.time()
        
    def log(self, logs: Dict[str, float], start_time=None) -> None:
        """
        重写log方法，同时记录到SwanLab
        兼容Transformers传递的start_time参数
        """
        try:
            # 调用父类的log方法，传递所有参数
            if start_time is not None:
                super().log(logs, start_time)
            else:
                super().log(logs)
            
            if self.swanlab_run is not None and logs:
                # 添加时间信息
                current_time = time.time()
                logs_with_time = logs.copy()
                logs_with_time.update({
                    "time/elapsed_time": current_time - self.start_time,
                    "time/step_time": current_time - self.step_start_time,
                })
                
                # 记录到SwanLab
                log_to_swanlab(
                    self.swanlab_run, 
                    logs_with_time, 
                    step=self.state.global_step
                )
                
                self.step_start_time = current_time
        except Exception as e:
            print(f"⚠️  日志记录失败: {e}")
            import traceback
            traceback.print_exc()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算损失，包括数值损失
        """
        try:
            # 获取模型输出，确保返回字典格式
            outputs = model(**inputs, return_dict=True)
            
            # 获取损失
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            elif isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            else:
                # 如果没有预计算的loss，需要手动计算
                if 'labels' in inputs:
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
                    labels = inputs['labels']
                    
                    # 计算交叉熵损失
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                else:
                    raise ValueError("无法计算损失：模型输出中没有loss，输入中也没有labels")
            
            # 记录损失组件到SwanLab
            if self.swanlab_run is not None:
                loss_components = {
                    "loss/total_loss": loss.item(),
                }
                
                # 如果有数值损失组件，也记录
                if hasattr(outputs, 'numeric_loss') and outputs.numeric_loss is not None:
                    loss_components["loss/numeric_loss"] = outputs.numeric_loss.item()
                if hasattr(outputs, 'language_loss') and outputs.language_loss is not None:
                    loss_components["loss/language_loss"] = outputs.language_loss.item()
                
                log_to_swanlab(
                    self.swanlab_run,
                    loss_components,
                    step=self.state.global_step
                )
            
            # 确保正确处理返回值
            if return_outputs:
                return loss, outputs
            else:
                return loss
            
        except Exception as e:
            print(f"计算损失时出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回一个默认损失
            dummy_loss = torch.tensor(0.0, requires_grad=True)
            if hasattr(model, 'device'):
                dummy_loss = dummy_loss.to(model.device)
            elif hasattr(model, 'parameters'):
                dummy_loss = dummy_loss.to(next(model.parameters()).device)
            
            if return_outputs:
                return dummy_loss, None
            else:
                return dummy_loss
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """
        保存检查点时记录到SwanLab，并保存处理器
        参考 HuggingFace 最佳实践，确保保存所有必要组件
        """
        import json
        import os
        
        # 调用父类方法保存模型权重
        checkpoint_path = super()._save_checkpoint(model, trial)
        
        if checkpoint_path:
            print(f"正在保存完整检查点到: {checkpoint_path}")
            
            # 1. 保存处理器（包含tokenizer和image_processor）
            if hasattr(self, 'processor') and self.processor is not None:
                try:
                    print("保存处理器...")
                    self.processor.save_pretrained(checkpoint_path)
                    print("✅ 处理器保存成功")
                except Exception as e:
                    print(f"⚠️ 处理器保存失败: {e}")
            
            # 2. 确保 preprocessor_config.json 存在
            preprocessor_config_path = os.path.join(checkpoint_path, "preprocessor_config.json")
            if not os.path.exists(preprocessor_config_path):
                print("创建 preprocessor_config.json...")
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
                print("✅ preprocessor_config.json 创建成功")
            else:
                # 如果文件已存在，检查并更新必要字段
                print("更新 preprocessor_config.json...")
                with open(preprocessor_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 确保包含数值token配置和image_processor_type
                config.update({
                    "processor_class": "NumericQwen2_5_VLProcessor",
                    "auto_map": {
                        "AutoProcessor": "numeric_qwen2_5_vl.NumericQwen2_5_VLProcessor"
                    },
                    "image_processor_type": "Qwen2VLImageProcessor",  # 确保有这个字段
                    "num_token_id": 151665,
                    "num_pad_token_id": 151666,
                    "numeric_tokens": ["<num>", "<num_pad>"]
                })
                
                with open(preprocessor_config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                print("✅ preprocessor_config.json 更新成功")
            
            # 3. 保存生成配置（如果存在）
            if hasattr(model, 'generation_config') and model.generation_config is not None:
                try:
                    model.generation_config.save_pretrained(checkpoint_path)
                    print("✅ generation_config 保存成功")
                except Exception as e:
                    print(f"⚠️ generation_config 保存失败: {e}")
            
            # 4. 验证保存的文件
            saved_files = os.listdir(checkpoint_path)
            print(f"检查点保存的文件: {saved_files}")
            
            # 检查关键文件
            required_files = [
                "config.json",
                "added_tokens.json", 
                "special_tokens_map.json",
                "tokenizer_config.json",
                "preprocessor_config.json"
            ]
            
            missing_files = []
            for file_name in required_files:
                if file_name not in saved_files:
                    missing_files.append(file_name)
            
            if missing_files:
                print(f"⚠️ 缺失文件: {missing_files}")
            else:
                print("✅ 所有必要文件都已保存")
        
        # 记录检查点信息到SwanLab
        if self.swanlab_run is not None and checkpoint_path:
            checkpoint_info = {
                "checkpoint/path": str(checkpoint_path),
                "checkpoint/step": self.state.global_step,
                "checkpoint/epoch": self.state.epoch,
            }
            
            if metrics:
                for key, value in metrics.items():
                    checkpoint_info[f"checkpoint/{key}"] = value
            
            log_to_swanlab(
                self.swanlab_run,
                checkpoint_info,
                step=self.state.global_step
            )
        
        return checkpoint_path
    
    def train(self, resume_from_checkpoint: Optional[str] = None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """
        重写训练方法，添加SwanLab集成
        """
        # 记录训练开始
        if self.swanlab_run is not None:
            log_to_swanlab(
                self.swanlab_run,
                {
                    "training/status": 1.0,  # 1.0表示开始
                    "training/resume_from_checkpoint": 1.0 if resume_from_checkpoint is not None else 0.0,
                },
                step=0
            )
        
        try:
            # 调用父类的训练方法
            result = super().train(
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
                **kwargs
            )
            
            # 记录训练完成
            if self.swanlab_run is not None:
                final_metrics = {
                    "training/status": 2.0,  # 2.0表示完成
                    "training/total_time": time.time() - self.start_time,
                    "training/total_steps": self.state.global_step,
                    "training/total_epochs": self.state.epoch,
                }
                
                # 添加最终的训练指标
                if hasattr(result, 'training_loss'):
                    final_metrics["training/final_loss"] = result.training_loss
                
                log_to_swanlab(
                    self.swanlab_run,
                    final_metrics,
                    step=self.state.global_step
                )
            
            return result
            
        except Exception as e:
            # 记录训练失败
            if self.swanlab_run is not None:
                log_to_swanlab(
                    self.swanlab_run,
                    {
                        "training/status": -1.0,  # -1.0表示失败
                        "training/error_occurred": 1.0,
                        "training/total_time": time.time() - self.start_time,
                    },
                    step=self.state.global_step
                )
            raise
    
    def evaluation_loop(self, dataloader, description: str, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        重写评估循环，记录评估指标到SwanLab
        """
        if self.swanlab_run is not None:
            log_to_swanlab(
                self.swanlab_run,
                {f"{metric_key_prefix}/status": 1.0},  # 1.0表示开始评估
                step=self.state.global_step
            )
        
        try:
            # 调用父类的评估方法
            result = super().evaluation_loop(
                dataloader=dataloader,
                description=description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix
            )
            
            # 记录评估结果到SwanLab
            if self.swanlab_run is not None and result.metrics:
                eval_metrics = {}
                for key, value in result.metrics.items():
                    if isinstance(value, (int, float, np.number)):
                        eval_metrics[key] = float(value)
                
                log_to_swanlab(
                    self.swanlab_run,
                    eval_metrics,
                    step=self.state.global_step
                )
            
            return result
            
        except Exception as e:
            if self.swanlab_run is not None:
                log_to_swanlab(
                    self.swanlab_run,
                    {f"{metric_key_prefix}/status": -1.0, f"{metric_key_prefix}/error_occurred": 1.0},
                    step=self.state.global_step
                )
            raise


def create_swanlab_trainer(
    model,
    args,
    train_dataset=None,
    eval_dataset=None,
    tokenizer=None,
    data_collator=None,
    compute_metrics=None,
    swanlab_run=None,
    processor=None,  # 添加处理器参数
    **kwargs
):
    """
    创建集成SwanLab的训练器
    
    Args:
        model: 要训练的模型
        args: 训练参数
        train_dataset: 训练数据集
        eval_dataset: 评估数据集
        tokenizer: 分词器
        data_collator: 数据整理器
        compute_metrics: 指标计算函数
        swanlab_run: SwanLab运行对象
        processor: 处理器（用于保存）
        **kwargs: 其他参数
        
    Returns:
        SwanLabNumericTrainer: 集成SwanLab的训练器
    """
    
    trainer = SwanLabNumericTrainer(
        swanlab_run=swanlab_run,
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        **kwargs
    )
    
    # 将处理器附加到训练器上，以便在保存时使用
    if processor is not None:
        trainer.processor = processor
    
    return trainer


def log_sample_data_to_swanlab(swanlab_run, dataset, processor, num_samples=5):
    """
    记录数据样本到SwanLab
    
    Args:
        swanlab_run: SwanLab运行对象
        dataset: 数据集
        processor: 处理器
        num_samples: 要记录的样本数量
    """
    if swanlab_run is None or len(dataset) == 0:
        return
    
    try:
        sample_data = []
        for i in range(min(num_samples, len(dataset))):
            item = dataset[i]
            
            # 准备样本信息
            sample_info = {
                "sample_id": i,
                "input_length": len(item.get('input_ids', [])),
                "has_image": 'pixel_values' in item,
                "has_numeric": len(item.get('numeric_values', [])) > 0,
            }
            
            if item.get('numeric_values'):
                if item['numeric_values']:
                    first_group = item['numeric_values'][0]
                    if not isinstance(first_group, (list, tuple)):
                        first_group = [first_group]
                    sample_info["numeric_count"] = len(first_group)
                    sample_info["numeric_values"] = str(first_group[:3])  # 只显示前3个数值
                else:
                    sample_info["numeric_count"] = 0
            
            # 解码文本样本
            if 'input_ids' in item and processor:
                try:
                    text_sample = processor.tokenizer.decode(
                        item['input_ids'][:100],  # 只显示前100个token
                        skip_special_tokens=False
                    )
                    sample_info["text_preview"] = text_sample[:200] + "..." if len(text_sample) > 200 else text_sample
                except:
                    sample_info["text_preview"] = "无法解码"
            
            sample_data.append(sample_info)
        
        # 记录到SwanLab - 只记录数值统计信息
        sample_stats = {
            "data/total_samples": len(sample_data),
            "data/avg_input_length": sum(item["input_length"] for item in sample_data) / len(sample_data),
            "data/samples_with_image": sum(1 for item in sample_data if item["has_image"]),
            "data/samples_with_numeric": sum(1 for item in sample_data if item["has_numeric"]),
        }
        
        log_to_swanlab(
            swanlab_run,
            sample_stats
        )
        
        print(f"📊 已记录 {len(sample_data)} 个数据样本到SwanLab")
        
    except Exception as e:
        print(f"⚠️  记录数据样本失败: {e}")


def log_training_progress_to_swanlab(swanlab_run, progress_info: Dict[str, Any]):
    """
    记录训练进度信息到SwanLab
    
    Args:
        swanlab_run: SwanLab运行对象
        progress_info: 进度信息字典
    """
    if swanlab_run is None:
        return
    
    try:
        log_to_swanlab(swanlab_run, progress_info)
    except Exception as e:
        print(f"⚠️  记录训练进度失败: {e}")
