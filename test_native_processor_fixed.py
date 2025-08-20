#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用原生Qwen2.5-VL处理器测试图片推理（应用GitHub issue修复）
"""

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import json
import os

# 注册自定义模型
from numeric_qwen2_5_vl import (
    NumericQwen2_5_VLConfig,
    NumericQwen2_5_VLForConditionalGeneration
)

print("🎯 使用原生处理器测试图片推理（应用GitHub修复）")
print("=" * 60)

MODEL_PATH = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"

def test_with_native_processor_fixed():
    """使用原生处理器测试（应用修复）"""
    try:
        print("🔧 加载原生处理器...")
        
        # 使用原生Qwen2.5-VL处理器
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        
        # 手动设置num_additional_image_tokens（根据GitHub issue修复）
        processor.num_additional_image_tokens = 1
        print(f"✅ 设置 num_additional_image_tokens: {processor.num_additional_image_tokens}")
        
        print("✅ 原生处理器加载成功")
        
        # 创建测试图像
        image = Image.new('RGB', (336, 336), color='blue')  # 使用336x336符合模型配置
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        draw.rectangle([50, 50, 150, 150], fill='red')
        draw.ellipse([200, 200, 280, 280], fill='yellow')
        print("✅ 测试图像创建成功")
        
        # 构建简单的对话
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": "这个图像里有什么？"}
                ],
            }
        ]
        
        # 应用chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print("✅ Chat template应用成功")
        print(f"Generated text preview: {text[:200]}...")
        
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
            device_map="cuda"
        )
        print("✅ 模型加载成功")
        
        # 移动到设备
        inputs = inputs.to(model.device)
        
        # 测试前向传播
        print("🧪 测试前向传播...")
        with torch.no_grad():
            outputs = model(**inputs)
            print(f"✅ 前向传播成功!")
        
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

def manual_prompt_test():
    """手动构建prompt测试"""
    try:
        print("\n🔧 手动构建prompt测试...")
        
        # 直接加载模型和tokenizer
        model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        processor.num_additional_image_tokens = 1
        
        # 创建图像
        image = Image.new('RGB', (336, 336), color='green')
        
        # 手动构建包含图像tokens的prompt
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述这个图像。<|im_end|>\n<|im_start|>assistant\n"
        
        print(f"使用手动prompt: {prompt}")
        
        # 处理输入
        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt"
        )
        
        inputs = inputs.to(model.device)
        
        print(f"Input IDs shape: {inputs.input_ids.shape}")
        print(f"Pixel values shape: {inputs.pixel_values.shape}")
        
        # 尝试生成
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=True,
                temperature=0.7
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        print("🎉 手动prompt测试成功!")
        print("=" * 50)
        print("生成结果:")
        print(output_text[0])
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ 手动prompt测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试...")
    
    # 测试原生处理器
    success1 = test_with_native_processor_fixed()
    
    # 测试手动prompt
    success2 = manual_prompt_test()
    
    print("\n" + "=" * 60)
    print("📋 最终结果:")
    print(f"✅ 原生处理器测试: {'成功' if success1 else '失败'}")
    print(f"✅ 手动prompt测试: {'成功' if success2 else '失败'}")
    
    if success1 or success2:
        print("🎉 至少一种方法成功！图像推理功能正常！")
    else:
        print("⚠️ 都失败了，需要进一步调试。")
