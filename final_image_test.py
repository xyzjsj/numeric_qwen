#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最终图像推理测试 - 使用checkpoint-4250
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

print("🎯 最终图像推理测试")
print("=" * 60)

# 使用checkpoint-4250路径
MODEL_PATH = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"

def create_simple_test_image():
    """创建一个非常简单的测试图像"""
    # 创建 224x224 的图像，包含简单的几何图形
    image = Image.new('RGB', (224, 224), color='white')
    
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    
    # 画一个蓝色矩形
    draw.rectangle([50, 50, 150, 100], fill='blue')
    # 画一个红色圆圈
    draw.ellipse([100, 120, 180, 200], fill='red')
    
    return image

def test_model_basic_functionality():
    """测试模型基本功能"""
    try:
        print("🔧 加载模型进行基本测试...")
        
        # 只加载处理器来测试chat template
        processor = NumericQwen2_5_VLProcessor.from_pretrained(MODEL_PATH)
        print("✅ 处理器加载成功")
        
        # 测试chat template
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        
        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            print("✅ Chat template工作正常")
            print(f"Generated text: {text}")
        except Exception as e:
            print(f"❌ Chat template失败: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False

def test_image_inference_final():
    """最终图像推理测试"""
    try:
        print("🖼️ 开始图像推理测试...")
        
        # 加载处理器
        processor = NumericQwen2_5_VLProcessor.from_pretrained(MODEL_PATH)
        print("✅ 处理器加载成功")
        
        # 创建测试图像
        image = create_simple_test_image()
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
                    {"type": "text", "text": "这个图像中有什么颜色的形状？"}
                ],
            }
        ]
        
        # 应用chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print("📝 Chat template应用成功")
        
        # 处理输入（只处理，不生成）
        print("🔧 处理输入...")
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        print("✅ 输入处理成功！")
        print(f"输入shape: {inputs.input_ids.shape}")
        print(f"图像特征shape: {inputs.pixel_values.shape}")
        
        # 检查token数量
        print(f"Token数量: {inputs.input_ids.shape[1]}")
        
        # 现在尝试加载模型并生成
        print("🚀 加载模型进行推理...")
        model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ 模型加载成功")
        
        # 移动输入到设备
        inputs = inputs.to(model.device)
        
        # 生成
        print("🎯 开始生成...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7
            )
        
        # 提取新生成的tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # 解码输出
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        print("🎉 图像推理成功!")
        print("=" * 40)
        print("生成结果:")
        print(output_text[0])
        print("=" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ 图像推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"使用模型路径: {MODEL_PATH}")
    
    # 首先测试基本功能
    basic_success = test_model_basic_functionality()
    
    if basic_success:
        # 如果基本功能正常，则测试图像推理
        image_success = test_image_inference_final()
        
        print("\n" + "=" * 60)
        print("📋 最终测试结果:")
        print(f"✅ 基本功能: {'成功' if basic_success else '失败'}")
        print(f"✅ 图像推理: {'成功' if image_success else '失败'}")
        
        if image_success:
            print("🎉 恭喜！您的checkpoint-4250模型可以正常处理图像！")
        else:
            print("⚠️ 图像推理仍有问题，但模型基本功能正常。")
    else:
        print("❌ 基本功能测试失败，请检查模型配置。")
