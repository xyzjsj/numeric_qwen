#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用原生Qwen2.5-VL处理器测试
"""

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import json

# 注册自定义模型
from numeric_qwen2_5_vl import (
    NumericQwen2_5_VLConfig,
    NumericQwen2_5_VLForConditionalGeneration
)

print("🎯 使用原生处理器测试")
print("=" * 60)

MODEL_PATH = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"

def test_with_native_processor():
    """使用原生处理器测试"""
    try:
        print("🔧 加载原生处理器...")
        
        # 使用原生Qwen2.5-VL处理器
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        print("✅ 原生处理器加载成功")
        
        # 创建测试图像
        image = Image.new('RGB', (224, 224), color='blue')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        draw.rectangle([50, 50, 150, 150], fill='red')
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
                    {"type": "text", "text": "这个图像里有什么颜色？"}
                ],
            }
        ]
        
        # 应用chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print("✅ Chat template应用成功")
        print(f"Generated text: {text[:200]}...")
        
        # 处理输入
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        print("✅ 输入处理成功")
        print(f"Input shape: {inputs.input_ids.shape}")
        print(f"Image shape: {inputs.pixel_values.shape}")
        
        # 加载自定义模型
        print("🚀 加载自定义模型...")
        model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ 模型加载成功")
        
        # 移动到设备
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
        
        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        print("🎉 图像推理成功!")
        print("=" * 50)
        print("模型输出:")
        print(output_text[0])
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text_with_native_processor():
    """使用原生处理器测试文本"""
    try:
        print("📝 测试纯文本推理...")
        
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        
        messages = [
            {"role": "user", "content": "你好，请介绍一下你自己。"}
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = processor(text=[text], return_tensors="pt")
        
        model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
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
        print("=" * 50)
        print("模型输出:")
        print(output_text[0])
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ 文本推理失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试...")
    
    # 测试文本推理
    text_success = test_text_with_native_processor()
    
    # 测试图像推理
    image_success = test_with_native_processor()
    
    print("\n" + "=" * 60)
    print("📋 最终结果:")
    print(f"✅ 文本推理: {'成功' if text_success else '失败'}")
    print(f"✅ 图像推理: {'成功' if image_success else '失败'}")
    
    if text_success and image_success:
        print("🎉 完美！您的模型使用原生处理器完全正常！")
        print("💡 接下来可以考虑修复自定义处理器或直接用原生处理器进行评估。")
    else:
        print("⚠️ 还需要进一步调试。")
