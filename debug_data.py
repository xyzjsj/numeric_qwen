#!/usr/bin/env python3
"""
调试数据处理问题
"""

import os
import json
import torch
from PIL import Image
import sys

# 添加当前目录到path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from numeric_qwen2_5_vl import NumericQwen2_5_VLProcessor
from transformers import Qwen2_5_VLProcessor as OriginalProcessor

def debug_data_processing():
    """
    调试数据处理过程
    """
    print("=== 调试数据处理 ===")
    
    # 加载原始处理器和我们的处理器
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    try:
        original_processor = OriginalProcessor.from_pretrained(model_path)
        our_processor = NumericQwen2_5_VLProcessor.from_pretrained(model_path)
        
        print("? 处理器加载成功")
        
        # 测试数据
        data_path = "/data1/wangzhiye/1a1a11/original/data/numeric_training_data.json"
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 测试第一个有图像的样本
        image_sample = None
        for item in data[:10]:
            if 'image' in item:
                image_sample = item
                break
        
        if image_sample is None:
            print("? 没有找到包含图像的样本")
            return
            
        print(f"测试样本: {image_sample['id']}")
        print(f"图像: {image_sample.get('image', 'None')}")
        
        # 构建对话文本
        conversations = image_sample.get('conversations', [])
        full_text = ""
        for turn in conversations:
            role = turn.get('from', '')
            content = turn.get('value', '')
            if role == 'human':
                full_text += f"Human: {content}\n"
            elif role == 'gpt':
                full_text += f"Assistant: {content}\n"
        
        print(f"完整文本:\n{full_text}")
        
        # 加载图像
        image_path = os.path.join("/data1/wangzhiye/1a1a11/original/data/images", image_sample['image'])
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            print(f"? 图像加载成功: {image.size}")
        else:
            print(f"? 图像文件不存在: {image_path}")
            return
        
        # 测试原始处理器
        print("\n=== 测试原始处理器 ===")
        try:
            original_result = original_processor(
                text=full_text,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            print("? 原始处理器成功")
            print(f"Keys: {list(original_result.keys())}")
            for key, value in original_result.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
                    
            # 检查是否有image_embeds
            if 'image_embeds' in original_result:
                print(f"  image_embeds shape: {original_result['image_embeds'].shape}")
            
            # 检查input_ids中的图像tokens
            input_ids = original_result['input_ids']
            vision_start_token_id = getattr(original_processor.tokenizer, 'vision_start_token_id', 151652)
            vision_end_token_id = getattr(original_processor.tokenizer, 'vision_end_token_id', 151653)
            vision_token_id = getattr(original_processor.tokenizer, 'vision_token_id', 151654)
            
            vision_start_count = (input_ids == vision_start_token_id).sum().item()
            vision_end_count = (input_ids == vision_end_token_id).sum().item()  
            vision_token_count = (input_ids == vision_token_id).sum().item()
            
            print(f"  Vision start tokens: {vision_start_count}")
            print(f"  Vision end tokens: {vision_end_count}")
            print(f"  Vision tokens: {vision_token_count}")
            
        except Exception as e:
            print(f"? 原始处理器失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 测试我们的处理器
        print("\n=== 测试我们的处理器 ===")
        try:
            our_result = our_processor(
                text=full_text,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            print("? 我们的处理器成功")
            print(f"Keys: {list(our_result.keys())}")
            for key, value in our_result.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
                    
        except Exception as e:
            print(f"? 我们的处理器失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 测试只有文本的情况
        print("\n=== 测试只有文本的情况 ===")
        text_only = "计算 1 + 2 的结果是 <num><3>"
        try:
            text_result = our_processor(
                text=text_only,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            print("? 纯文本处理成功")
            print(f"Keys: {list(text_result.keys())}")
            for key, value in text_result.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
                    
        except Exception as e:
            print(f"? 纯文本处理失败: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"? 整体测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_processing()
