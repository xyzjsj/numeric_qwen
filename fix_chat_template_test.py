#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复chat template并测试图像推理
"""

import torch
from PIL import Image
import json

# 注册自定义模型组件
from numeric_qwen2_5_vl import (
    NumericQwen2_5_VLConfig,
    NumericQwen2_5_VLForConditionalGeneration, 
    NumericQwen2_5_VLProcessor
)

print("🎯 修复chat template测试")
print("=" * 60)

MODEL_PATH = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"

def load_and_fix_chat_template():
    """加载处理器并修复chat template"""
    try:
        # 加载处理器
        processor = NumericQwen2_5_VLProcessor.from_pretrained(MODEL_PATH)
        
        # 手动读取并设置chat template
        chat_template_path = f"{MODEL_PATH}/chat_template.json"
        with open(chat_template_path, 'r', encoding='utf-8') as f:
            chat_template_data = json.load(f)
        
        # 设置chat template到tokenizer
        processor.tokenizer.chat_template = chat_template_data["chat_template"]
        
        print("✅ Chat template修复成功")
        return processor
        
    except Exception as e:
        print(f"❌ Chat template修复失败: {e}")
        return None

def test_fixed_image_inference():
    """测试修复后的图像推理"""
    try:
        # 加载并修复处理器
        processor = load_and_fix_chat_template()
        if not processor:
            return False
        
        # 创建简单测试图像
        image = Image.new('RGB', (224, 224), color='red')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        draw.rectangle([50, 50, 150, 150], fill='blue')
        
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
                    {"type": "text", "text": "描述这个图像。"}
                ],
            }
        ]
        
        # 应用chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print("✅ Chat template应用成功")
        print(f"Generated prompt: {text[:200]}...")
        
        # 处理输入
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        print("✅ 输入处理成功")
        print(f"Input IDs shape: {inputs.input_ids.shape}")
        print(f"Pixel values shape: {inputs.pixel_values.shape}")
        
        # 加载模型
        print("🚀 加载模型...")
        model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 移动到设备
        inputs = inputs.to(model.device)
        
        # 生成
        print("🎯 开始生成...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                pad_token_id=processor.tokenizer.eos_token_id
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
        print(f"❌ 图像推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text_only():
    """测试纯文本推理"""
    try:
        processor = load_and_fix_chat_template()
        if not processor:
            return False
            
        messages = [
            {"role": "user", "content": "计算 <num>15</num> + <num>25</num>"}
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
        
        print("🎉 文本推理成功!")
        print("=" * 50)
        print("数学计算结果:")
        print(output_text[0])
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ 文本推理失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试...")
    
    # 先测试文本推理
    print("\n📝 测试文本推理...")
    text_success = test_text_only()
    
    # 再测试图像推理
    print("\n🖼️ 测试图像推理...")
    image_success = test_fixed_image_inference()
    
    print("\n" + "=" * 60)
    print("📋 最终结果:")
    print(f"✅ 文本推理: {'成功' if text_success else '失败'}")
    print(f"✅ 图像推理: {'成功' if image_success else '失败'}")
    
    if text_success and image_success:
        print("🎉 完美！您的checkpoint-4250模型完全正常工作！")
        print("💡 现在可以进行余弦相似度评估了。")
    elif text_success:
        print("✅ 文本功能正常，图像功能需要进一步调试。")
    else:
        print("❌ 需要检查模型配置和权重。")
