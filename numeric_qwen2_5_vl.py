#!/usr/bin/env python3
"""
基于原生Qwen2.5-VL架构的数值增强多模态模型

这个实现基于原生Qwen2_5_VLForConditionalGeneration，添加了数值处理能力，
支持 <num><value> 格式的数值token，具有双重输出：文本logits + 数值预测
"""

import re
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from typing import Dict, List, Optional, Tuple, Union, Any

from transformers import (
    Qwen2_5_VLConfig, 
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoProcessor
)
from transformers.modeling_outputs import CausalLMOutputWithPast


class NumericQwen2_5_VLConfig(Qwen2_5_VLConfig):
    """
    数值增强Qwen2.5-VL配置类
    在原生Qwen2.5-VL配置基础上添加数值处理相关参数
    """
    model_type = "numeric_qwen2_5_vl"

    def __init__(
        self,
        # 数值处理相关参数
        numeric_embedding_dim: int = 512,
        numeric_token: str = "<num>",
        numeric_loss_weight: float = 1.0,
        # 继承所有原生Qwen2.5-VL参数
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # 数值增强配置
        self.numeric_embedding_dim = numeric_embedding_dim
        self.numeric_token = numeric_token
        self.numeric_loss_weight = numeric_loss_weight
        
        # token ID将在tokenizer配置时正确设置
        if not hasattr(self, 'num_token_id'):
            self.num_token_id = None
        if not hasattr(self, 'num_pad_token_id'):
            self.num_pad_token_id = None


class NumericQwen2_5_VLProcessor(Qwen2_5_VLProcessor):
    """
    数值增强的处理器
    在原生Qwen2.5-VL处理器基础上添加数值token提取和处理功能
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 定义数值相关的特殊标记
        self.num_token = "<num>"
        self.num_pad_token = "<num_pad>"
        
        # 追加而不是覆盖已有 additional_special_tokens
        existing_additional = []
        if hasattr(self.tokenizer, 'additional_special_tokens') and self.tokenizer.additional_special_tokens:
            existing_additional = list(self.tokenizer.additional_special_tokens)
        vision_tokens = ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
        to_add = []
        for tok in vision_tokens + [self.num_token, self.num_pad_token]:
            if tok not in existing_additional:
                to_add.append(tok)
        if to_add:
            self.tokenizer.add_special_tokens({'additional_special_tokens': existing_additional + to_add})
        
        self.num_token_id = self.tokenizer.convert_tokens_to_ids(self.num_token)
        self.num_pad_token_id = self.tokenizer.convert_tokens_to_ids(self.num_pad_token)
        
        # 数值token正则表达式
        self.numeric_pattern = re.compile(r'<num><([+-]?\d*\.?\d+)>')
        
        if hasattr(self, 'num_additional_image_tokens'):
            pass
        else:
            self.num_additional_image_tokens = getattr(self, 'num_additional_image_tokens', 1)
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        processor = super().from_pretrained(*args, **kwargs)
        try:
            import json
            import os
            pretrained_model_name_or_path = args[0] if args else kwargs.get('pretrained_model_name_or_path')
            if pretrained_model_name_or_path:
                config_path = os.path.join(pretrained_model_name_or_path, 'preprocessor_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    if 'num_additional_image_tokens' in config:
                        processor.num_additional_image_tokens = config['num_additional_image_tokens']
                        print(f"✅ 从配置加载 num_additional_image_tokens: {processor.num_additional_image_tokens}")
        except Exception as e:
            processor.num_additional_image_tokens = 1
        return processor
        
    def extract_numeric_values(self, text: str) -> Tuple[List[float], List[int]]:
        """
        从文本中提取数值和位置
        """
        values = []
        positions = []
        
        matches = list(self.numeric_pattern.finditer(text))
        for match in matches:
            try:
                value = float(match.group(1))
                values.append(value)
                
                start_char = match.start()
                token_pos = len(self.tokenizer.encode(text[:start_char], add_special_tokens=False))
                positions.append(token_pos)
            except ValueError:
                continue
                
        return values, positions
    
    def _process_text_with_numeric_tokens(self, text: Union[str, List[str]]) -> Tuple[Union[str, List[str]], List[List[float]]]:
        is_batched = isinstance(text, list)
        if not is_batched:
            text = [text]
            
        processed_texts = []
        all_values = []
        
        for t in text:
            processed_t, values = self._process_single_text(t)
            processed_texts.append(processed_t)
            all_values.append(values)
        
        return processed_texts if is_batched else processed_texts[0], all_values
    
    def _process_single_text(self, text: str) -> Tuple[str, List[float]]:
        values = []
        matches = list(self.numeric_pattern.finditer(text))
        
        if not matches:
            return text, []
        
        result_text = text
        for match in reversed(matches):
            start, end = match.span()
            num_str = match.group(1)
            try:
                value = float(num_str)
            except ValueError:
                value = 0.0
            
            result_text = result_text[:start] + f"{self.num_token}{self.num_pad_token}" + result_text[end:]
            values.insert(0, value) # 保持原始顺序
        
        return result_text, values
    
    def __call__(
        self,
        text: Union[str, List[str]] = None,
        images=None,
        videos=None,
        **kwargs
    ):
        """
        处理输入，添加数值信息提取和文本转换
        """
        processed_text = text
        batch_numeric_values = []
        batch_numeric_positions = []
        
        if text is not None:
            processed_text, batch_numeric_values = self._process_text_with_numeric_tokens(text)
            
            if isinstance(processed_text, str):
                text_list = [processed_text]
            else:
                text_list = processed_text
                
            for txt in text_list:
                positions = []
                input_ids = self.tokenizer.encode(txt, add_special_tokens=False)
                # 寻找 <num><num_pad> 序列，记录 <num> 的位置
                for i in range(len(input_ids) - 1):
                    if (input_ids[i] == self.num_token_id and 
                        input_ids[i + 1] == self.num_pad_token_id):
                        positions.append(i)  # 记录<num>的位置，不是<num_pad>的位置
                batch_numeric_positions.append(positions)
        
        processed = super().__call__(text=processed_text, images=images, videos=videos, **kwargs)
        
        if text is not None:
            processed['numeric_values'] = batch_numeric_values
            processed['numeric_positions'] = batch_numeric_positions
        
        return processed


class NumericQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    数值增强的Qwen2.5-VL条件生成模型
    在原生架构基础上添加数值嵌入层和回归头
    """
    config_class = NumericQwen2_5_VLConfig
    
    def __init__(self, config: NumericQwen2_5_VLConfig):
        super().__init__(config)
        
        self.config = config
        self.numeric_loss_weight = getattr(config, 'numeric_loss_weight', 1.0)
        
        self.num_token_id = getattr(config, 'num_token_id', config.vocab_size - 1)
        
        numeric_dim = getattr(config, 'numeric_embedding_dim', 512)
        hidden_size = config.hidden_size
        
        self.numeric_embedding = nn.Sequential(
            nn.Linear(1, numeric_dim),
            nn.GELU(),
            nn.Linear(numeric_dim, hidden_size),
            nn.LayerNorm(hidden_size))
        
        # 数值回归头
        self.regression_head = nn.Linear(hidden_size, 1)
        
        # 初始化权重
        self._init_numeric_weights()
    
    def _format_float(self, v: float) -> str:
        """格式化数值（带符号与两位小数）。"""
        try:
            return f"{float(v):+0.2f}"
        except Exception:
            return "+0.00"
    
    def _fill_numeric_in_text(self, text: str, values: list) -> str:
        """将文本中的 <num><num_pad> 按出现顺序替换为 <num><+x.xx>。"""
        if not text or not values:
            return text
        pattern = re.compile(r"<num><num_pad>", re.IGNORECASE)
        idx = 0
        def repl(_):
            nonlocal idx
            if idx >= len(values):
                return "<num><num_pad>"
            rep = f"<num><{self._format_float(values[idx])}>"
            idx += 1
            return rep
        return pattern.sub(repl, text)
    
    @torch.no_grad()
    def generate_with_numeric(
        self,
        tokenizer,
        *args,
        format_numbers: bool = True,
        return_text: bool = True,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
        **kwargs
    ):
        """包装生成: 返回 (sequences, texts, filled_texts)"""
        sequences = super().generate(*args, **kwargs)
        if not return_text:
            return sequences, None, None

        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
        raw_texts = [
            tokenizer.decode(seq, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces)
            for seq in sequences
        ]
        
        filled_texts = raw_texts
        if format_numbers:
            outputs = super().forward(
                input_ids=sequences,
                output_hidden_states=True,
                return_dict=True
            )
            predicted_floats = self.regression_head(outputs.hidden_states[-1]).squeeze(-1)
            
            filled_texts_new = []
            for b_idx, seq in enumerate(sequences):
                seq_list = seq.tolist()
                num_token_id = getattr(self.config, 'num_token_id', None) or getattr(self, 'num_token_id', None)
                num_pad_id = getattr(self.config, 'num_pad_token_id', None) or getattr(self, 'num_pad_token_id', None)
                collected_vals = []
                if num_token_id is not None and num_pad_id is not None:
                    # 寻找<num><num_pad>序列，从<num>位置获取数值预测
                    for pos in range(len(seq_list) - 1):
                        if seq_list[pos] == num_token_id and seq_list[pos + 1] == num_pad_id:
                            try:
                                # 从<num>位置的predicted_floats获取数值
                                collected_vals.append(float(predicted_floats[b_idx, pos].item()))
                            except Exception:
                                collected_vals.append(0.0)
                filled_texts_new.append(self._fill_numeric_in_text(raw_texts[b_idx], collected_vals))
            filled_texts = filled_texts_new

        return sequences, raw_texts, filled_texts
    
    def _compute_mixed_loss(
        self,
        logits: torch.Tensor,
        predicted_floats: torch.Tensor,
        labels: torch.LongTensor,
        input_ids: torch.LongTensor,
        numeric_values: Optional[List[List[float]]] = None,
        numeric_positions: Optional[List[List[int]]] = None
    ) -> torch.Tensor:
        """
        计算混合损失：交叉熵损失 + 数值回归损失，修复设备不一致问题
        """
        # 获取主设备（基于logits的设备）
        device = logits.device
        
        # 确保所有tensor都在同一设备上
        if labels.device != device:
            labels = labels.to(device)
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        if predicted_floats.device != device:
            predicted_floats = predicted_floats.to(device)
        
        batch_size, seq_len = input_ids.shape
        
        # 标准的语言模型shift：用位置i预测位置i+1
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 对于数值预测，我们需要不同的策略：
        # 位置i是<num>时，用位置i的hidden_state预测对应的数值
        # 所以数值预测不需要shift
        unshift_predicted_floats = predicted_floats.contiguous()  # 不shift，直接使用
        unshift_input_ids = input_ids.contiguous()  # 不shift，直接使用
        
        # 获取token IDs
        num_token_id = getattr(self.config, 'num_token_id', None) or getattr(self, 'num_token_id', None)
        num_pad_token_id = getattr(self.config, 'num_pad_token_id', None) or getattr(self, 'num_pad_token_id', None)
        
        # 创建数值预测掩码：标记<num>位置（这些位置需要预测数值）
        if num_token_id is not None and num_token_id >= 0:
            is_float_target_mask = (unshift_input_ids == num_token_id)
        else:
            is_float_target_mask = torch.zeros_like(unshift_input_ids, dtype=torch.bool, device=device)
        
        # 计算token损失（交叉熵）
        loss_fct_token = CrossEntropyLoss()
        token_labels = shift_labels.clone()
       
        
        actual_vocab_size = shift_logits.size(-1)
        if actual_vocab_size != self.config.vocab_size:
            loss_token = loss_fct_token(
                shift_logits.view(-1, actual_vocab_size), token_labels.view(-1)
            )
        else:
            loss_token = loss_fct_token(
                shift_logits.view(-1, self.config.vocab_size), token_labels.view(-1)
            )
        
        # 计算数值损失（均方误差）
        loss_float = torch.tensor(0.0, device=device, dtype=logits.dtype)
        
        if numeric_values is not None and numeric_positions is not None:
            loss_fct_float = MSELoss()
            # 使用unshift的predicted_floats来构建float_labels
            float_labels = torch.zeros_like(unshift_predicted_floats, device=device)
            
            for i in range(batch_size):
                if i < len(numeric_values) and numeric_values[i]:
                    values = torch.tensor(numeric_values[i], device=device, dtype=logits.dtype)
                    positions = numeric_positions[i]
                    
                    if values.dim() > 1:
                        values = values.flatten()
                    
                    for j, (pos, val) in enumerate(zip(positions, values)):
                        if isinstance(pos, (list, tuple)):
                            pos = pos[0] if len(pos) > 0 else 0
                        
                        # pos是<num>的位置，直接在这个位置设置数值标签
                        target_pos = int(pos)
                        
                        if 0 <= target_pos < float_labels.shape[1]:
                            if isinstance(val, torch.Tensor):
                                if val.dim() > 0:
                                    val_scalar = val.item() if val.numel() == 1 else val[0].item()
                                else:
                                    val_scalar = val.item()
                            else:
                                val_scalar = float(val)
                            
                            float_labels[i, target_pos] = val_scalar
                        else:
                            pass
            
            if is_float_target_mask.any():
                # 确保mask也在正确设备上
                is_float_target_mask = is_float_target_mask.to(device)
                float_labels = float_labels.to(device)
                
                # 使用unshift的predicted_floats和mask
                valid_float_preds = unshift_predicted_floats[is_float_target_mask]
                valid_float_labels = float_labels[is_float_target_mask]
                
                if valid_float_preds.numel() > 0:
                    # 确保所有tensor都在同一设备上
                    if valid_float_labels.device != valid_float_preds.device:
                        valid_float_labels = valid_float_labels.to(valid_float_preds.device)
                    
                    loss_float = loss_fct_float(valid_float_preds, valid_float_labels)
        
        # 合并损失 - 确保在同一设备上
        if loss_float.device != device:
            loss_float = loss_float.to(device)
        if loss_token.device != device:
            loss_token = loss_token.to(device)
        
        total_loss = loss_token + self.numeric_loss_weight * loss_float
        
        # 调试信息
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(f"Token Loss: {loss_token.item():.8f}, Float Loss: {loss_float.item():.8f}")
                print(f"Device - logits: {logits.device}, loss: {total_loss.device}")
        else:
            print(f"Token Loss: {loss_token.item():.8f}, Float Loss: {loss_float.item():.8f}")
            print(f"Device - logits: {logits.device}, loss: {total_loss.device}")
        
        return total_loss

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        **kwargs
    ) -> Dict[str, Any]:
        # 解码阶段移除视觉输入
        for k in ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"]:
            kwargs.pop(k, None)
        
        # 处理数值embedding的生成时一致性
        if input_ids is not None:
            kwargs.update(self._prepare_numeric_inputs_for_generation(input_ids))
        
        return super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, **kwargs)
    
    def _prepare_numeric_inputs_for_generation(self, input_ids: torch.LongTensor) -> Dict[str, Any]:
        """
        为生成过程准备数值输入，确保训练和推理时的一致性
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        num_token_id = getattr(self.config, 'num_token_id', None) or getattr(self, 'num_token_id', None)
        num_pad_token_id = getattr(self.config, 'num_pad_token_id', None) or getattr(self, 'num_pad_token_id', None)
        
        if num_token_id is None or num_pad_token_id is None:
            return {}
        
        # 寻找需要数值预测的<num><num_pad>序列
        numeric_values = []
        numeric_positions = []
        
        for batch_idx in range(batch_size):
            batch_values = []
            batch_positions = []
            seq = input_ids[batch_idx].tolist()
            
            # 寻找<num><num_pad>序列
            for pos in range(len(seq) - 1):
                if seq[pos] == num_token_id and seq[pos + 1] == num_pad_token_id:
                    # 对<num>位置做forward pass获取数值预测
                    with torch.no_grad():
                        # 构建到当前位置的序列进行预测
                        partial_seq = input_ids[batch_idx:batch_idx+1, :pos+1]
                        outputs = super().forward(
                            input_ids=partial_seq,
                            output_hidden_states=True,
                            return_dict=True
                        )
                        hidden_state = outputs.hidden_states[-1][0, -1]  # 最后一个位置的hidden state
                        predicted_value = self.regression_head(hidden_state).squeeze(-1).item()
                    
                    batch_values.append(predicted_value)
                    batch_positions.append(pos)  # <num>的位置
            
            numeric_values.append(batch_values)
            numeric_positions.append(batch_positions)
        
        return {
            'numeric_values': numeric_values,
            'numeric_positions': numeric_positions
        }


class NumericCausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    扩展的输出类，包含数值预测
    """
    def __init__(
        self,
        loss=None,
        logits=None,
        predicted_floats=None,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
    ):
        super().__init__(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
        )
        self.predicted_floats = predicted_floats


        
    def _init_numeric_weights(self):
        """初始化数值相关层的权重"""
        for module in [self.numeric_embedding, self.regression_head]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        numeric_values: Optional[List[List[float]]] = None,
        numeric_positions: Optional[List[List[int]]] = None,
        **kwargs
    ):
        """
        前向传播，支持数值增强功能，修复设备不一致问题
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 获取主设备
        if input_ids is not None:
            device = input_ids.device
        elif inputs_embeds is not None:
            device = inputs_embeds.device
        else:
            device = next(self.parameters()).device
        
        # 确保所有输入tensor都在同一设备上
        if input_ids is not None and input_ids.device != device:
            input_ids = input_ids.to(device)
        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device)
        if position_ids is not None and position_ids.device != device:
            position_ids = position_ids.to(device)
        if labels is not None and labels.device != device:
            labels = labels.to(device)
        if pixel_values is not None and pixel_values.device != device:
            pixel_values = pixel_values.to(device)
        if pixel_values_videos is not None and pixel_values_videos.device != device:
            pixel_values_videos = pixel_values_videos.to(device)
        if image_grid_thw is not None and image_grid_thw.device != device:
            image_grid_thw = image_grid_thw.to(device)
        if video_grid_thw is not None and video_grid_thw.device != device:
            video_grid_thw = video_grid_thw.to(device)
        
        # 处理数值embedding融入
        modified_inputs_embeds = inputs_embeds
        numeric_replaced_count = 0
        
        if input_ids is not None and numeric_values is not None and numeric_positions is not None:
            # 获取基础embeddings
            if inputs_embeds is None:
                modified_inputs_embeds = self.get_input_embeddings()(input_ids)
            else:
                modified_inputs_embeds = inputs_embeds.clone()
            
            # 确保embeddings在正确设备上
            if modified_inputs_embeds.device != device:
                modified_inputs_embeds = modified_inputs_embeds.to(device)
            
            # 检查输入是否包含NaN
            if torch.isnan(modified_inputs_embeds).any():
                print("embeddings包含NaN!!!!!!")
                modified_inputs_embeds = torch.nan_to_num(modified_inputs_embeds, nan=0.0)
            
            # 获取num_pad_token_id
            num_pad_token_id = getattr(self.config, 'num_pad_token_id', None) or getattr(self, 'num_pad_token_id', None)
            
            if num_pad_token_id is not None:
                batch_size = input_ids.shape[0] if input_ids is not None else modified_inputs_embeds.shape[0]
                
                # 遍历batch中的每个样本
                for batch_idx in range(batch_size):
                    if batch_idx < len(numeric_values) and numeric_values[batch_idx]:
                        values = numeric_values[batch_idx]
                        positions = numeric_positions[batch_idx]
                        
                        # 数据有效性检查
                        if not isinstance(values, list):
                            print(f"WARNING: numeric_values[{batch_idx}] 不是列表: {type(values)}")
                            continue
                        if not isinstance(positions, list):
                            print(f"WARNING: numeric_positions[{batch_idx}] 不是列表: {type(positions)}")
                            continue
                        
                        # 确保values和positions长度一致
                        min_len = min(len(values), len(positions))
                        values = values[:min_len]
                        positions = positions[:min_len]
                        
                        # 为每个数值计算embedding并替换对应位置
                        for value, pos in zip(values, positions):
                            if isinstance(pos, (list, tuple)):
                                pos = pos[0] if len(pos) > 0 else 0
                            
                            try:
                                pos = int(pos)
                                value = float(value)
                                
                                # 数值有效性检查
                                if math.isnan(value) or math.isinf(value):
                                    print(f"WARNING: 无效数值 {value} 在位置 {pos}")
                                    value = 0.0
                                
                                # 数值范围限制，防止溢出
                                value = max(-1e6, min(1e6, value))
                                
                            except (ValueError, TypeError) as e:
                                print(f"WARNING: 数值转换失败: value={value}, pos={pos}, error={e}")
                                continue
                            
                            # 检查位置是否有效 - pos是<num>的位置，我们要替换pos+1位置(<num_pad>)的embedding
                            pad_pos = pos + 1
                            if 0 <= pad_pos < modified_inputs_embeds.shape[1]:
                                # 检查该位置是否确实是num_pad_token
                                if input_ids is not None and input_ids[batch_idx, pad_pos].item() == num_pad_token_id:
                                    try:
                                        # 计算数值embedding - 确保在正确设备上
                                        value_tensor = torch.tensor([[value]], 
                                                                device=device,  # 使用统一设备
                                                                dtype=modified_inputs_embeds.dtype)
                                        
                                        numeric_emb = self.numeric_embedding(value_tensor).squeeze(0).squeeze(0)
                                        
                                        # 检查embedding是否包含NaN
                                        if torch.isnan(numeric_emb).any():
                                            print(f"WARNING: 数值embedding包含NaN! value={value}")
                                            numeric_emb = torch.zeros_like(numeric_emb)
                                        
                                        # 检查embedding是否包含Inf
                                        if torch.isinf(numeric_emb).any():
                                            print(f"WARNING: 数值embedding包含Inf! value={value}")
                                            numeric_emb = torch.clamp(numeric_emb, -1e6, 1e6)
                                        
                                        # 替换<num_pad>位置的embedding
                                        modified_inputs_embeds[batch_idx, pad_pos] = numeric_emb
                                        numeric_replaced_count += 1
                                        
                                    except Exception as e:
                                        print(f"ERROR: 计算数值embedding失败: value={value}, pos={pos}, error={e}")
                                        continue
        
        if numeric_replaced_count > 0:
            print(f"DEBUG: 成功替换了 {numeric_replaced_count} 个数值embedding")
        
        # 调用父类forward，传入修改后的inputs_embeds
        outputs = super().forward(
            input_ids=None,  # 使用inputs_embeds时设为None
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=modified_inputs_embeds,
            labels=None,  # 暂时不传labels，后续手动计算loss
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # 确保返回hidden_states用于数值预测
            return_dict=True,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs
        )
        
        logits = outputs.logits
        
        # 检查logits是否包含NaN
        if torch.isnan(logits).any():
            print("ERROR: logits包含NaN!")
            logits = torch.nan_to_num(logits, nan=0.0)
        
        # 获取hidden states用于数值预测
        if outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
            hidden_states = outputs.hidden_states[-1]
        else:
            # fallback: 如果没有hidden_states，重新forward一次获取
            temp_outputs = super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=modified_inputs_embeds,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                **kwargs
            )
            hidden_states = temp_outputs.hidden_states[-1]
        
        # 确保hidden_states在正确设备上
        if hidden_states.device != device:
            hidden_states = hidden_states.to(device)
        
        # 检查hidden_states是否包含NaN
        if torch.isnan(hidden_states).any():
            print("ERROR: hidden_states包含NaN!")
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0)
        
        # 数值embedding安全处理
        if numeric_values is not None:
            # 替换nan/inf为安全值
            safe_numeric_values = []
            for batch in numeric_values:
                safe_batch = [float(v) if isinstance(v, (int, float)) and torch.isfinite(torch.tensor(v)) else 0.0 for v in batch]
                safe_numeric_values.append(safe_batch)
            numeric_values = safe_numeric_values

        predicted_floats = self.regression_head(hidden_states).squeeze(-1)
        
        # 确保predicted_floats在正确设备上
        if predicted_floats.device != device:
            predicted_floats = predicted_floats.to(device)
        
        # 检查predicted_floats是否包含NaN
        if torch.isnan(predicted_floats).any():
            print("ERROR: predicted_floats包含NaN!")
            predicted_floats = torch.nan_to_num(predicted_floats, nan=0.0)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss = self._compute_mixed_loss(
                logits=logits,
                predicted_floats=predicted_floats,
                labels=labels,
                input_ids=input_ids if input_ids is not None else None,
                numeric_values=numeric_values,
                numeric_positions=numeric_positions
            )
            
            # 检查损失是否为NaN
            if loss is not None and torch.isnan(loss):
                print("ERROR: 计算出的损失为NaN!")
                loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        if not return_dict:
            output = (logits, predicted_floats) + outputs[1:]
            return (loss,) + output if loss is not None else output
            
        # 确保所有返回的tensor都在同一设备上
        return NumericCausalLMOutputWithPast(
            loss=loss,
            logits=logits.to(device) if logits.device != device else logits,
            predicted_floats=predicted_floats.to(device) if predicted_floats.device != device else predicted_floats,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 注册自定义模型组件
AutoConfig.register("numeric_qwen2_5_vl", NumericQwen2_5_VLConfig)
AutoModelForCausalLM.register(NumericQwen2_5_VLConfig, NumericQwen2_5_VLForConditionalGeneration)
AutoProcessor.register(NumericQwen2_5_VLConfig, NumericQwen2_5_VLProcessor)

print(">>> 数值增强Qwen2.5-VL模型组件已注册")
print(f">>> 配置类: {NumericQwen2_5_VLConfig}")
print(f">>> 模型类: {NumericQwen2_5_VLForConditionalGeneration}")
print(f">>> 处理器类: {NumericQwen2_5_VLProcessor}")