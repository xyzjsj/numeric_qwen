#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用慢速图像处理器测试图像推理
"""

import torch
from PIL import Image
import numpy as np
import os

# 注册自定义模型组件
from numeric_qwen2_5_vl import (
    NumericQwen2_5_VLConfig,
    NumericQwen2_5_VLForConditionalGeneration, 
    NumericQwen2_5_VLProcessor
)

print("🎯 使用慢速图像处理器测试图像推理")
print("=" * 60)

def create_test_image():
    """创建一个简单的测试图像"""
    # 创建 224x224 的简单图像
    image = Image.new('RGB', (224, 224), color='red')
    
    # 在图像上添加一些简单的形状
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    draw.rectangle([50, 50, 174, 174], fill='blue')
    draw.ellipse([75, 75, 149, 149], fill='yellow')
    
    return image

def test_image_inference_with_slow_processor():
    """使用慢速处理器测试图像推理"""
    try:
        print("🔧 加载模型和处理器...")
        
        # 加载模型
        model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
            "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 加载处理器，强制使用慢速图像处理器
        processor = NumericQwen2_5_VLProcessor.from_pretrained(
            "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output",
            use_fast=False  # 关键：使用慢速处理器
        )
        
        print("✅ 模型和慢速处理器加载成功")
        
        # 创建测试图像
        image = create_test_image()
        print("✅ 测试图像创建成功")
        
        # 构建对话
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": "描述这个图像中你看到的内容。"}
                ],
            }
        ]
        
        # 应用chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print("📝 Chat template应用成功")
        print("Generated text preview:", text[:200] + "..." if len(text) > 200 else text)
        
        # 处理输入
        print("🖼️ 处理图像输入...")
        image_inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # 移动到设备
        image_inputs = image_inputs.to(model.device)
        print("✅ 输入处理成功")
        print(f"输入形状: {image_inputs.input_ids.shape}")
        print(f"图像特征形状: {image_inputs.pixel_values.shape}")
        
        # 生成
        print("🚀 开始生成...")
        with torch.no_grad():
            generated_ids = model.generate(
                **image_inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7
            )
        
        # 提取新生成的tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(image_inputs.input_ids, generated_ids)
        ]
        
        # 解码输出
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        print("✅ 图像推理成功!")
        print("🎉 生成结果:")
        print("-" * 40)
        print(output_text[0])
        print("-" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ 图像推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text_inference():
    """测试文本推理以确保模型基本功能正常"""
    try:
        print("\n📝 测试文本推理...")
        
        # 加载模型
        model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
            "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        processor = NumericQwen2_5_VLProcessor.from_pretrained(
            "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output"
        )
        
        # 纯文本对话
        messages = [
            {"role": "user", "content": "计算 <num>25</num> + <num>37</num> = ?"}
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor(text=[text], return_tensors="pt")
        inputs = inputs.to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        print("✅ 文本推理成功!")
        print("🎉 计算结果:")
        print("-" * 40)
        print(output_text[0])
        print("-" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ 文本推理失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试...")
    
    # 先测试文本推理
    text_success = test_text_inference()
    
    # 再测试图像推理
    image_success = test_image_inference_with_slow_processor()
    
    print("\n" + "=" * 60)
    print("📋 测试总结:")
    print(f"✅ 文本推理: {'成功' if text_success else '失败'}")
    print(f"✅ 图像推理: {'成功' if image_success else '失败'}")
    
    if text_success and image_success:
        print("🎉 所有测试通过！模型可以正常处理文本和图像。")
    elif text_success:
        print("⚠️ 文本推理正常，图像推理需要进一步调试。")
    else:
        print("❌ 模型存在基础问题，需要检查配置。")
