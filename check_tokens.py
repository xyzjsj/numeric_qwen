#!/usr/bin/env python3
"""
检查Qwen2.5-VL特殊token和正确的prompt格式
"""
import os
import torch
from PIL import Image

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加数值增强模块路径
import sys
sys.path.append('/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250')

from numeric_qwen2_5_vl import NumericQwen2_5_VLForConditionalGeneration, NumericQwen2_5_VLProcessor

def check_special_tokens():
    """检查处理器的特殊token"""
    checkpoint_path = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"
    
    processor = NumericQwen2_5_VLProcessor.from_pretrained(
        checkpoint_path,
        trust_remote_code=True
    )
    
    print("🔍 检查特殊Token:")
    tokenizer = processor.tokenizer
    
    # 检查关键的特殊token
    special_tokens = [
        'vision_start_token', 'vision_end_token', 'image_token', 
        'video_start_token', 'video_end_token', 'eos_token',
        'bos_token', 'pad_token', 'unk_token'
    ]
    
    for token_name in special_tokens:
        if hasattr(tokenizer, token_name):
            token = getattr(tokenizer, token_name)
            print(f"   {token_name}: {token}")
    
    print("\n📚 特殊token映射:")
    if hasattr(tokenizer, 'special_tokens_map'):
        for key, value in tokenizer.special_tokens_map.items():
            print(f"   {key}: {value}")
    
    # 检查vocab中的视觉相关token
    print("\n🔎 搜索vision相关token:")
    vocab = tokenizer.get_vocab()
    vision_tokens = {k: v for k, v in vocab.items() if 'vision' in k.lower() or 'image' in k.lower()}
    for token, token_id in sorted(vision_tokens.items(), key=lambda x: x[1]):
        print(f"   {token} (ID: {token_id})")

def test_correct_prompt_format():
    """测试正确的prompt格式"""
    checkpoint_path = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"
    
    processor = NumericQwen2_5_VLProcessor.from_pretrained(
        checkpoint_path,
        trust_remote_code=True
    )
    
    # 创建测试图像
    image = Image.new('RGB', (224, 224), color='red')
    
    # 测试不同的prompt格式
    prompt_formats = [
        # 格式1: 基础格式
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What color is this image?<|im_end|>\n<|im_start|>assistant\n",
        
        # 格式2: 使用image token
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|image|>What color is this image?<|im_end|>\n<|im_start|>assistant\n",
        
        # 格式3: 简化格式
        "What color is this image?<|image|>",
        
        # 格式4: 使用vision token
        "<|vision_start|><|image_pad|><|vision_end|>What color is this image?",
    ]
    
    for i, prompt in enumerate(prompt_formats, 1):
        print(f"\n📋 测试格式 {i}: {prompt[:50]}...")
        try:
            # 处理输入
            inputs = processor(
                text=[prompt],
                images=[image],
                return_tensors="pt"
            )
            
            print(f"   ✅ 处理成功")
            print(f"   📊 input_ids shape: {inputs.input_ids.shape}")
            if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
                print(f"   🖼️ pixel_values shape: {inputs.pixel_values.shape}")
            
            # 检查是否包含图像token
            input_text = processor.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)
            has_vision_tokens = any(token in input_text for token in ['<|vision_start|>', '<|vision_end|>', '<|image_pad|>', '<|image|>'])
            print(f"   🔍 包含视觉token: {has_vision_tokens}")
            
            if has_vision_tokens:
                print(f"   📝 解码的文本: {input_text}")
            
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")

if __name__ == "__main__":
    print("🚀 检查Qwen2.5-VL特殊token和prompt格式")
    print("=" * 60)
    
    check_special_tokens()
    test_correct_prompt_format()
