#!/usr/bin/env python3
"""
最终的图像Token匹配解决方案
基于GitCode博客文章和token检查结果的完整修复
"""
import os
import torch
from PIL import Image

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加数值增强模块路径
import sys
sys.path.append('/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250')

from numeric_qwen2_5_vl import NumericQwen2_5_VLForConditionalGeneration, NumericQwen2_5_VLProcessor

def test_final_solution():
    """最终解决方案测试"""
    print("🚀 最终图像Token匹配解决方案测试")
    print("=" * 60)
    
    checkpoint_path = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"
    
    # 加载模型和处理器
    print("📂 加载模型...")
    model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    processor = NumericQwen2_5_VLProcessor.from_pretrained(
        checkpoint_path,
        trust_remote_code=True
    )
    
    print("✅ 模型加载成功")
    
    # 创建超小图像以减少Token数量
    print("\n🎨 创建超小测试图像...")
    image = Image.new('RGB', (56, 56), color='red')  # 极小尺寸
    print(f"📊 图像尺寸: {image.size}")
    
    # 计算预期Token数量
    pixels = 56 * 56
    expected_tokens = pixels // (28 * 28)
    print(f"📊 预期Token数量: {expected_tokens}")
    
    # 测试多种prompt格式
    test_prompts = [
        # 方案1: 直接使用vision tokens
        "<|vision_start|><|image_pad|><|vision_end|>What color is this?",
        
        # 方案2: 使用image token  
        "What color is this image? <|image|>",
        
        # 方案3: 标准聊天格式 + image token
        "<|im_start|>user\n<|image|>What color is this?<|im_end|>\n<|im_start|>assistant\n",
        
        # 方案4: 仅使用image token
        "<|image|>Describe this image.",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📋 测试方案 {i}: {prompt[:50]}...")
        
        try:
            # 处理输入
            inputs = processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
                padding=True
            )
            
            print(f"   ✅ 输入处理成功")
            print(f"   📊 input_ids shape: {inputs.input_ids.shape}")
            
            if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
                print(f"   🖼️ pixel_values shape: {inputs.pixel_values.shape}")
            
            # 检查解码后的文本
            decoded = processor.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)
            image_pad_count = decoded.count('<|image_pad|>')
            vision_start_count = decoded.count('<|vision_start|>')
            vision_end_count = decoded.count('<|vision_end|>')
            
            print(f"   🔍 image_pad tokens: {image_pad_count}")
            print(f"   🔍 vision_start tokens: {vision_start_count}")
            print(f"   🔍 vision_end tokens: {vision_end_count}")
            
            if image_pad_count > 0:
                print(f"   ✅ 找到图像tokens！尝试推理...")
                
                # 移动到设备
                device_inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                # 生成
                with torch.no_grad():
                    outputs = model.generate(
                        **device_inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                
                # 解码回答
                generated_ids = outputs[0][device_inputs['input_ids'].shape[1]:]
                response = processor.decode(generated_ids, skip_special_tokens=True)
                
                print(f"   🎯 生成回答: {response}")
                print(f"   ✅ 方案 {i} 成功！")
                break
            else:
                print(f"   ⚠️ 没有找到图像tokens")
                
        except Exception as e:
            print(f"   ❌ 方案 {i} 失败: {e}")
    
    print("\n" + "=" * 60)
    print("✅ 图像Token匹配测试完成")

if __name__ == "__main__":
    test_final_solution()
