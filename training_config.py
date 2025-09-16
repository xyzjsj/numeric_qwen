#!/usr/bin/env python3
"""
数值增强Qwen2.5-VL模型的训练配置和数据处理

基于原生Qwen2.5-VL架构的训练框架，支持端到端训练
"""

import os
import json
import torch
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from transformers import TrainingArguments, AutoTokenizer
from torch.utils.data import Dataset
import copy
from PIL import Image
import swanlab
from torch.nn import CrossEntropyLoss, MSELoss

# 导入我们的自定义模型
from numeric_qwen2_5_vl import (
    NumericQwen2_5_VLConfig,
    NumericQwen2_5_VLProcessor,
    NumericQwen2_5_VLForConditionalGeneration
)


@dataclass
class NumericTrainingArguments(TrainingArguments):
    """
    数值增强训练参数
    """
    # 数值相关参数
    numeric_loss_weight: float = field(default=1.0, metadata={"help": "数值损失权重"})
    
    # 模型相关参数
    model_name_or_path: Optional[str] = field(default=None)
    
    # 数据相关参数
    data_path: Optional[str] = field(default=None, metadata={"help": "训练数据路径"})
    val_data_path: Optional[str] = field(default=None, metadata={"help": "验证数据路径"})
    test_data_path: Optional[str] = field(default=None, metadata={"help": "测试数据路径"})
    image_folder: Optional[str] = field(default=None, metadata={"help": "图像文件夹路径"})
    
    # 训练相关参数
    vision_lr: Optional[float] = field(default=2e-6, metadata={"help": "视觉编码器学习率"})
    numeric_lr: Optional[float] = field(default=1e-4, metadata={"help": "数值层学习率"})
    
    # SwanLab可视化参数
    swanlab_project: Optional[str] = field(default="qsinghua", metadata={"help":  "SwanLab项目名称"})
    swanlab_experiment: Optional[str] = field(default=None, metadata={"help":  "SwanLab实验名称"})
    enable_swanlab: bool = field(default=True, metadata={"help":  "是否启用SwanLab可视化"})


class NumericDataset(Dataset):
    """
    数值增强的数据集类
    支持多模态数据和数值标注
    """
    
    def __init__(
        self,
        data_path: str,
        processor: NumericQwen2_5_VLProcessor,
        image_folder: Optional[str] = None,
        max_length: int = 8192
    ):
        self.processor = processor
        self.image_folder = image_folder
        self.max_length = max_length
        
        # 加载数据
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        elif data_path.endswith('.jsonl'):
            self.data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line.strip()))
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")
            
        print(f"加载了 {len(self.data)} 条训练数据")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return self._process_item(item)
    
    def _process_item(self, item: Dict) -> Dict:
        """
        处理单个数据项 - 使用Qwen2.5-VL的标准对话模板
        """
        # 新格式：{"images": [...], "messages": [...]}
        if 'messages' in item:
            messages = item.get('messages', [])
            images = item.get('images', [])
        else:
            # 兼容原有格式：{"conversations": [...], "image": "..."}
            conversations = item.get('conversations', [])
            images = [item.get('image')] if item.get('image') else []
            
            # 转换为标准messages格式
            messages = []
            for turn in conversations:
                role = turn.get('from', '')
                content = turn.get('value', '')
                
                if role == 'human':
                    messages.append({"role": "user", "content": content})
                elif role == 'gpt':
                    messages.append({"role": "assistant", "content": content})
        
        # 处理图像
        image_list = []
        if images:
            # 限制图像数量，避免过多图像导致内存和处理问题
            max_images = 6  # 最多使用6张图像
            for i, img_path in enumerate(images[:max_images]):
                if not img_path:
                    continue
                    
                # 如果有image_folder，则拼接路径
                if self.image_folder and not os.path.isabs(img_path):
                    img_path = os.path.join(self.image_folder, img_path)
                
                try:
                    if os.path.exists(img_path):
                        image = Image.open(img_path).convert('RGB')
                        image_list.append(image)
                    else:
                        print(f"警告: 图像文件不存在: {img_path}")
                except Exception as e:
                    print(f"警告: 无法加载图像 {img_path}: {e}")
        
        # 使用Qwen2.5-VL标准的处理方式
        try:
            if image_list:
                # 转换messages格式，确保包含图像信息
                formatted_messages = []
                for msg in messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    
                    if role == 'user' and '<image>' in content:
                        # 如果用户消息包含<image>，转换为正确的格式
                        content_parts = []
                        # 为每个图像添加图像部分
                        for i, _ in enumerate(image_list):
                            content_parts.append({"type": "image"})
                        
                        # 添加文本部分（移除<image>标记）
                        text_content = content.replace('<image>', '').strip()
                        if text_content:
                            content_parts.append({"type": "text", "text": text_content})
                        
                        formatted_messages.append({
                            "role": role,
                            "content": content_parts
                        })
                    else:
                        formatted_messages.append({
                            "role": role,
                            "content": content
                        })
                
                # 使用tokenizer的apply_chat_template
                text = self.processor.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                # 然后使用processor处理
                processed = self.processor(
                    text=[text],
                    images=image_list,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
            else:
                # 对于纯文本，使用标准处理
                text = self.processor.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                
                # 处理纯文本
                processed = self.processor(
                    text=[text],
                    images=None,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
            
            # 将tensor转为合适的格式
            result = {}
            for key, value in processed.items():
                if isinstance(value, torch.Tensor):
                    # 移除batch维度
                    result[key] = value.squeeze(0)
                    
                    # 特殊处理image_grid_thw
                    if key == 'image_grid_thw':
                        if value.dim() == 0:
                            del result[key]
                        elif value.dim() == 1 and value.numel() == 0:
                            del result[key]
                        elif value.dim() == 1 and value.numel() == 3:
                            result[key] = value.unsqueeze(0)
                else:
                    result[key] = value
            
            # 创建labels (用于语言建模)
            if 'input_ids' in result:
                result['labels'] = result['input_ids'].clone()
                # 获取处理器生成的数值标注并扁平化
                numeric_values = result.get('numeric_values', [])
                if isinstance(numeric_values, list) and len(numeric_values) == 1:
                    numeric_values = numeric_values[0]
                numeric_positions = result.get('numeric_positions', [])
                if isinstance(numeric_positions, list) and len(numeric_positions) == 1:
                    numeric_positions = numeric_positions[0]
                # 查找 <num_pad> 的 token id
                num_pad_token_id = self.processor.tokenizer.convert_tokens_to_ids('<num_pad>')
                if num_pad_token_id is not None:
                    input_ids_list = result['input_ids'].tolist()
                    positions = [i for i, token_id in enumerate(input_ids_list) if token_id == num_pad_token_id]
                    # 对齐数值与位置数量
                    if numeric_values and positions:
                        min_len = min(len(numeric_values), len(positions))
                        numeric_values = numeric_values[:min_len]
                        numeric_positions = positions[:min_len]
                    else:
                        numeric_values = []
                        numeric_positions = []
                else:
                    numeric_values = []
                    numeric_positions = []
                result['numeric_values'] = numeric_values
                result['numeric_positions'] = numeric_positions

            return result
            
        except Exception as e:
            print(f"处理数据项时出错: {e}")
            # 返回一个最小的有效样本
            return {
                'input_ids': torch.tensor([self.processor.tokenizer.eos_token_id]),
                'labels': torch.tensor([self.processor.tokenizer.eos_token_id]),
                'attention_mask': torch.tensor([1])
            }


class NumericDataCollator:
    """
    数值增强的数据整理器
    """
    
    def __init__(self, processor: NumericQwen2_5_VLProcessor, padding_side: str = "right"):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.padding_side = padding_side
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        整理批次数据
        """
        batch = {}
        
        # 收集所有key
        all_keys = set()
        for feature in features:
            all_keys.update(feature.keys())
        
        # 处理每个key
        for key in all_keys:
            values = [feature.get(key) for feature in features if key in feature]
            
            if not values:
                continue
                
            if key in ['input_ids', 'labels', 'attention_mask']:
                # 文本相关的tensor需要padding
                batch[key] = self._pad_sequence(values, key)
            elif key.startswith('pixel_values'):
                # 图像数据
                if all(v is not None for v in values):
                    batch[key] = torch.stack(values)
            elif key == 'image_grid_thw':
                # 图像网格维度信息，需要特殊处理
                if all(v is not None for v in values):
                    valid_values = []
                    for v in values:
                        if isinstance(v, torch.Tensor):
                            if v.dim() == 1:
                                v = v.unsqueeze(0)
                            elif v.dim() == 0:
                                continue
                            valid_values.append(v)
                    
                    if valid_values:
                        batch[key] = torch.cat(valid_values, dim=0)
                    else:
                        batch[key] = torch.empty(0, 3, dtype=torch.long)
            elif key in ['numeric_values', 'numeric_positions']:
                # 数值相关的列表数据
                batch[key] = values
            elif isinstance(values[0], torch.Tensor):
                try:
                    batch[key] = torch.stack(values)
                except:
                    batch[key] = values
            else:
                batch[key] = values
        
        return batch
    
    def _pad_sequence(self, sequences: List[torch.Tensor], key: str) -> torch.Tensor:
        """
        对序列进行padding
        """
        if not sequences:
            return torch.tensor([])
        
        max_len = max(len(seq) for seq in sequences)
        
        if key == 'labels':
            pad_value = -100
        elif key == 'attention_mask':
            pad_value = 0
        else:  # input_ids
            pad_value = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
        padded_sequences = []
        for seq in sequences:
            seq_len = len(seq)
            if seq_len < max_len:
                pad_len = max_len - seq_len
                if self.padding_side == "right":
                    padded = torch.cat([seq, torch.full((pad_len,), pad_value, dtype=seq.dtype)])
                else:
                    padded = torch.cat([torch.full((pad_len,), pad_value, dtype=seq.dtype), seq])
            else:
                padded = seq
            padded_sequences.append(padded)
        
        return torch.stack(padded_sequences)


def init_swanlab(config: NumericTrainingArguments) -> Optional[object]:
    """
    初始化SwanLab实验跟踪
    
    Args:
        config: 训练配置
        
    Returns:
        SwanLab run对象或None
    """
    if not config.enable_swanlab:
        print("SwanLab可视化已禁用")
        return None
    
    try:
        experiment_name = config.swanlab_experiment
        if experiment_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"numeric_qwen2_5_vl_{timestamp}"
        
        run = swanlab.init(
            project=config.swanlab_project,
            experiment_name=experiment_name,
            config={
                "model_name": config.model_name_or_path,
                "architecture": "Numeric-Qwen2.5-VL",
                "num_train_epochs": config.num_train_epochs,
                "per_device_train_batch_size": config.per_device_train_batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "learning_rate": config.learning_rate,
                "vision_lr": config.vision_lr,
                "numeric_lr": config.numeric_lr,
                "weight_decay": config.weight_decay,
                "warmup_ratio": config.warmup_ratio,
                "lr_scheduler_type": config.lr_scheduler_type,
                "numeric_loss_weight": config.numeric_loss_weight,
                "bf16": config.bf16,
                "tf32": config.tf32,
                "gradient_checkpointing": config.gradient_checkpointing,
                "data_path": config.data_path,
                "image_folder": config.image_folder,
                "output_dir": config.output_dir,
                "save_strategy": config.save_strategy,
                "save_steps": config.save_steps,
                "logging_steps": config.logging_steps,
            },
            description=f"数值增强Qwen2.5-VL模型训练实验 - {experiment_name}"
        )
        
        print(f"✅ SwanLab实验已初始化:")
        print(f"   项目: {config.swanlab_project}")
        print(f"   实验: {experiment_name}")
        print(f"   查看链接: https://swanlab.cn/{config.swanlab_project}")
        
        return run
        
    except Exception as e:
        print(f"⚠️  SwanLab初始化失败: {e}")
        print("   训练将继续进行，但不会记录到SwanLab")
        return None

def log_to_swanlab(run: object, metrics: Dict[str, Any], step: Optional[int] = None):
    """
    记录指标到SwanLab
    
    Args:
        run: SwanLab run对象
        metrics: 要记录的指标字典
        step: 训练步数
    """
    if run is None:
        return
    
    try:
        if step is not None:
            run.log(metrics, step=step)
        else:
            run.log(metrics)
    except Exception as e:
        print(f"⚠️  SwanLab记录失败: {e}")


def log_model_info_to_swanlab(run: object, model: object, processor: object):
    """
    记录模型信息到SwanLab
    
    Args:
        run: SwanLab run对象
        model: 模型对象
        processor: 处理器对象
    """
    if run is None:
        return
    
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/vocab_size": processor.tokenizer.vocab_size,
        }
        
        if hasattr(model.config, 'num_token_id'):
            model_info["model/num_token_id"] = model.config.num_token_id
        if hasattr(model.config, 'numeric_embedding_dim'):
            model_info["model/numeric_embedding_dim"] = model.config.numeric_embedding_dim
        
        run.log(model_info)
        
        print(f"📊 模型信息已记录到SwanLab:")
        print(f"   总参数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        
    except Exception as e:
        print(f"⚠️  模型信息记录失败: {e}")


def create_model_and_processor(
    model_path: str,
    numeric_config: Dict = None
) -> tuple:
    """
    创建模型和处理器
    """
    default_config = {
        'numeric_embedding_dim': 512,
        'numeric_token': '<num>',
        'numeric_loss_weight': 1.0
    }
    
    if numeric_config:
        default_config.update(numeric_config)
    
    try:
        config = NumericQwen2_5_VLConfig.from_pretrained(model_path)
    except:
        from transformers import Qwen2_5_VLConfig as BaseConfig
        base_config = BaseConfig.from_pretrained(model_path)
        config = NumericQwen2_5_VLConfig(**base_config.to_dict())
    
    for key, value in default_config.items():
        setattr(config, key, value)
    
    try:
        processor = NumericQwen2_5_VLProcessor.from_pretrained(model_path)
    except:
        from transformers import Qwen2_5_VLProcessor as BaseProcessor
        base_processor = BaseProcessor.from_pretrained(model_path)
        processor = NumericQwen2_5_VLProcessor(
            image_processor=base_processor.image_processor,
            tokenizer=base_processor.tokenizer
        )
    
    model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float32,  # 强制使用fp32
        device_map="auto",
        trust_remote_code=True
    )
    
    numeric_token = default_config['numeric_token']
    numeric_pad_token = '<num_pad>'
    
    vocab = processor.tokenizer.get_vocab()
    tokens_to_add = []
    
    if numeric_token not in vocab:
        tokens_to_add.append(numeric_token)
    if numeric_pad_token not in vocab:
        tokens_to_add.append(numeric_pad_token)
    
    if tokens_to_add:
        processor.tokenizer.add_special_tokens({
            'additional_special_tokens': tokens_to_add
        })
        model.resize_token_embeddings(len(processor.tokenizer))
        config.vocab_size = len(processor.tokenizer)
        config.num_token_id = processor.tokenizer.convert_tokens_to_ids(numeric_token)
        config.num_pad_token_id = processor.tokenizer.convert_tokens_to_ids(numeric_pad_token)
        model.config.vocab_size = len(processor.tokenizer)
        model.config.num_token_id = processor.tokenizer.convert_tokens_to_ids(numeric_token)
        model.config.num_pad_token_id = processor.tokenizer.convert_tokens_to_ids(numeric_pad_token)
    else:
        config.num_token_id = processor.tokenizer.convert_tokens_to_ids(numeric_token)
        config.num_pad_token_id = processor.tokenizer.convert_tokens_to_ids(numeric_pad_token)
        model.config.num_token_id = processor.tokenizer.convert_tokens_to_ids(numeric_token)
        model.config.num_pad_token_id = processor.tokenizer.convert_tokens_to_ids(numeric_pad_token)
    
    return model, processor


# 在 training_config.py 中添加这些修改

def get_training_config(
    output_dir: str = "./numeric_qwen2_5_vl_output",
    model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    data_path: str = "./data/numeric_training_data.json",
    val_data_path: str = None,
    test_data_path: str = None,
    image_folder: str = "./data/images",
    enable_swanlab: bool = True,
    swanlab_project: str = "qsinghua",
    swanlab_experiment: str = None,
    **kwargs
) -> NumericTrainingArguments:
    """
    获取训练配置，添加防NaN设置
    """
    
    exp_name = kwargs.pop('exp_name', 'exp1')
    
    conflicting_params = [
        'use_lora', 'lora_rank', 'lora_alpha', 'lora_dropout',
        'num_train_epochs', 'per_device_train_batch_size', 'per_device_eval_batch_size',
        'gradient_accumulation_steps', 'eval_strategy', 'save_strategy', 'save_steps',
        'save_total_limit', 'learning_rate', 'weight_decay', 'warmup_ratio',
        'lr_scheduler_type', 'bf16', 'tf32', 'dataloader_num_workers',
        'gradient_checkpointing', 'logging_steps', 'report_to', 'remove_unused_columns'
    ]
    
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in conflicting_params}
    
    report_to_value = "swanlab" if enable_swanlab else "none"
    eval_strategy = "steps" if val_data_path else "no"
    
    config = NumericTrainingArguments(
        output_dir=output_dir,
        model_name_or_path=model_path,
        data_path=data_path,
        val_data_path=val_data_path,
        test_data_path=test_data_path,
        image_folder=image_folder,
        enable_swanlab=enable_swanlab,
        swanlab_project=swanlab_project,
        swanlab_experiment=swanlab_experiment,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # 减小batch size防止内存溢出
        per_device_eval_batch_size=1,   # 减小eval batch size
        gradient_accumulation_steps=16,  # 增加梯度累积来补偿小batch
        eval_strategy=eval_strategy,
        eval_steps=100,  # 增加评估间隔
        save_strategy="steps",
        save_steps=100,  # 增加保存间隔
        save_total_limit=3,
        metric_for_best_model="eval_loss" if val_data_path else None,
        greater_is_better=False,
        load_best_model_at_end=True if val_data_path else False,
        learning_rate=5e-6,  # 降低学习率防止梯度爆炸
        vision_lr=1e-6,      # 降低视觉编码器学习率
        numeric_lr=5e-5,     # 降低数值层学习率
        weight_decay=0.01,
        warmup_ratio=0.1,    # 增加warmup比例
        lr_scheduler_type="cosine",
        numeric_loss_weight=0.1,  # 降低数值损失权重
        bf16=False,          # 暂时禁用bf16，使用fp32防止精度问题
        tf32=True,
        dataloader_num_workers=0,  # 减少worker数量避免并发问题
        gradient_checkpointing=True,
        logging_steps=10,
        report_to=report_to_value,
        run_name=f"numeric_qwen2_5_vl_{exp_name}",
        remove_unused_columns=False,
        max_grad_norm=1.0,   # 添加梯度裁剪
        **filtered_kwargs
    )
    
    return config


# 修改 _compute_mixed_loss 方法，添加更多安全检查
def _compute_mixed_loss_safe(
    self,
    logits: torch.Tensor,
    predicted_floats: torch.Tensor,
    labels: torch.LongTensor,
    input_ids: torch.LongTensor,
    numeric_values: Optional[List[List[float]]] = None,
    numeric_positions: Optional[List[List[int]]] = None
) -> torch.Tensor:
    """
    计算混合损失：交叉熵损失 + 数值回归损失，添加安全检查
    """
    import math
    
    batch_size, seq_len = input_ids.shape
    
    # 检查输入是否包含NaN
    if torch.isnan(logits).any():
        print("ERROR: logits包含NaN，替换为0")
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if torch.isnan(predicted_floats).any():
        print("ERROR: predicted_floats包含NaN，替换为0") 
        predicted_floats = torch.nan_to_num(predicted_floats, nan=0.0, posinf=1e6, neginf=-1e6)
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_predicted_floats = predicted_floats[..., :-1].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    prev_input_ids = input_ids[..., :-1].contiguous()
    
    num_pad_token_id = getattr(self.config, 'num_pad_token_id', None) or getattr(self, 'num_pad_token_id', None)
    
    if num_pad_token_id is not None and num_pad_token_id >= 0:
        is_float_target_mask = (prev_input_ids == num_pad_token_id)
    else:
        is_float_target_mask = torch.zeros_like(prev_input_ids, dtype=torch.bool)
    
    # 计算token损失（交叉熵）
    loss_fct_token = CrossEntropyLoss()
    token_labels = shift_labels.clone()
    token_labels[is_float_target_mask] = -100
    
    try:
        actual_vocab_size = shift_logits.size(-1)
        loss_token = loss_fct_token(
            shift_logits.view(-1, actual_vocab_size), token_labels.view(-1)
        )
        
        # 检查token损失是否为NaN
        if torch.isnan(loss_token):
            print("ERROR: token损失为NaN，设置为0")
            loss_token = torch.tensor(0.0, device=logits.device, requires_grad=True)
            
    except Exception as e:
        print(f"ERROR: 计算token损失失败: {e}")
        loss_token = torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    # 计算数值损失（均方误差）
    loss_float = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    
    if numeric_values is not None and numeric_positions is not None:
        try:
            loss_fct_float = MSELoss()
            float_labels = torch.zeros_like(shift_predicted_floats)
            
            valid_numeric_count = 0
            
            for i in range(batch_size):
                if i < len(numeric_values) and numeric_values[i]:
                    values = numeric_values[i]
                    positions = numeric_positions[i]
                    
                    if not isinstance(values, list) or not isinstance(positions, list):
                        continue
                        
                    if isinstance(values, torch.Tensor):
                        values = values.flatten().tolist()
                    
                    for j, (pos, val) in enumerate(zip(positions, values)):
                        if isinstance(pos, (list, tuple)):
                            pos = pos[0] if len(pos) > 0 else 0
                        
                        try:
                            target_pos = int(pos) - 1
                            val_scalar = float(val)
                            
                            # 检查数值有效性
                            if math.isnan(val_scalar) or math.isinf(val_scalar):
                                print(f"WARNING: 无效数值标签 {val_scalar}")
                                val_scalar = 0.0
                            
                            # 限制数值范围
                            val_scalar = max(-1e3, min(1e3, val_scalar))
                            
                            if 0 <= target_pos < float_labels.shape[1]:
                                float_labels[i, target_pos] = val_scalar
                                valid_numeric_count += 1
                                
                        except (ValueError, TypeError, IndexError) as e:
                            print(f"WARNING: 处理数值标签失败: pos={pos}, val={val}, error={e}")
                            continue
            
            if is_float_target_mask.any() and valid_numeric_count > 0:
                is_float_target_mask = is_float_target_mask.to(shift_predicted_floats.device)
                float_labels = float_labels.to(shift_predicted_floats.device)
                
                valid_float_preds = shift_predicted_floats[is_float_target_mask]
                valid_float_labels = float_labels[is_float_target_mask]
                
                if valid_float_preds.numel() > 0:
                    # 裁剪预测值防止溢出
                    valid_float_preds = torch.clamp(valid_float_preds, -1e3, 1e3)
                    
                    loss_float = loss_fct_float(valid_float_preds, valid_float_labels)
                    
                    # 检查数值损失是否为NaN
                    if torch.isnan(loss_float):
                        print("ERROR: 数值损失为NaN，设置为0")
                        loss_float = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
                        
        except Exception as e:
            print(f"ERROR: 计算数值损失失败: {e}")
            loss_float = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    
    # 合并损失
    try:
        target_device = loss_token.device
        if loss_float.device != target_device:
            loss_float = loss_float.to(target_device)
        
        # 确保损失权重不会导致溢出
        numeric_weight = min(max(self.numeric_loss_weight, 0.01), 10.0)
        total_loss = loss_token + numeric_weight * loss_float
        
        # 最终检查
        if torch.isnan(total_loss):
            print("ERROR: 总损失为NaN，回退到仅token损失")
            total_loss = loss_token
            
    except Exception as e:
        print(f"ERROR: 合并损失失败: {e}")
        total_loss = loss_token
    
    # 安全地打印损失信息
    try:
        token_loss_val = loss_token.item() if not torch.isnan(loss_token) else 0.0
        float_loss_val = loss_float.item() if not torch.isnan(loss_float) else 0.0
        print(f"Token Loss: {token_loss_val:.4f}, Float Loss: {float_loss_val:.4f}")
    except:
        print("Token Loss: [error], Float Loss: [error]")
    
    return total_loss

def create_deepspeed_config(
    output_file: str = "ds_config_zero2.json",
    zero_stage: int = 2,
    gradient_accumulation_steps: int = 8
) -> str:
    """
    创建DeepSpeed配置
    """
    config = {
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto"}
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {"warmup_min_lr": "auto", "warmup_max_lr": "auto", "warmup_num_steps": "auto"}
        },
        "zero_optimization": {
            "stage": zero_stage,
            "offload_optimizer": {"device": "cpu" if zero_stage >= 2 else "none"},
            "offload_param": {"device": "cpu" if zero_stage >= 3 else "none"},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto"
        },
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False
    }
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return output_file
