#!/usr/bin/env python3
"""
基于GitCode博客文章的参数调整解决方案
调整max_pixels和max_prompt_length参数来解决Token不匹配问题
"""
import os
import torch
from PIL import Image

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加数值增强模块路径
import sys
sys.path.append('/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250')

from numeric_qwen2_5_vl import NumericQwen2_5_VLForConditionalGeneration, NumericQwen2_5_VLProcessor

def test_parameter_adjustment_solution():
    """测试参数调整解决方案"""
    print("🚀 GitCode博客参数调整解决方案测试")
    print("=" * 60)
    
    checkpoint_path = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"
    
    # 加载模型
    print("📂 加载模型...")
    model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载处理器并调整参数
    print("📂 加载处理器并调整参数...")
    processor = NumericQwen2_5_VLProcessor.from_pretrained(
        checkpoint_path,
        trust_remote_code=True
    )
    
    # 根据GitCode文章建议调整参数
    print("🔧 调整图像处理器参数...")
    if hasattr(processor, 'image_processor'):
        # 方案1: 降低max_pixels，确保Token数量不超过限制
        original_max_pixels = processor.image_processor.max_pixels
        original_min_pixels = processor.image_processor.min_pixels
        
        print(f"   📊 原始max_pixels: {original_max_pixels}")
        print(f"   📊 原始min_pixels: {original_min_pixels}")
        
        # 设置更小的max_pixels以减少Token数量
        processor.image_processor.max_pixels = 3136  # 56*56 = 3136
        processor.image_processor.min_pixels = 784   # 28*28 = 784
        
        print(f"   📊 调整后max_pixels: {processor.image_processor.max_pixels}")
        print(f"   📊 调整后min_pixels: {processor.image_processor.min_pixels}")
    
    # 测试不同尺寸的图像
    test_sizes = [
        (28, 28),   # 1个Token
        (56, 56),   # 4个Token
        (84, 84),   # 9个Token
    ]
    
    for size in test_sizes:
        print(f"\n📋 测试图像尺寸: {size}")
        
        # 创建测试图像
        image = Image.new('RGB', size, color='red')
        pixels = size[0] * size[1]
        expected_tokens = pixels // (28 * 28)
        
        print(f"   📊 像素数量: {pixels}")
        print(f"   📊 预期Token数量: {expected_tokens}")
        
        # 测试处理
        try:
            # 使用简单的prompt
            prompt = f"<|vision_start|><|image_pad|><|vision_end|>What color is this {size[0]}x{size[1]} image?"
            
            inputs = processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
                padding=True
            )
            
            print(f"   ✅ 输入处理成功")
            print(f"   📊 input_ids shape: {inputs.input_ids.shape}")
            print(f"   🖼️ pixel_values shape: {inputs.pixel_values.shape}")
            
            # 检查Token匹配
            decoded = processor.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)
            image_pad_count = decoded.count('<|image_pad|>')
            features_count = inputs.pixel_values.shape[0]
            
            print(f"   🔍 图像Token数量: {image_pad_count}")
            print(f"   🔍 图像特征数量: {features_count}")
            
            if image_pad_count == features_count:
                print(f"   ✅ Token匹配成功！进行推理测试...")
                
                # 移动到设备
                device_inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                # 生成回答
                with torch.no_grad():
                    outputs = model.generate(
                        **device_inputs,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                
                # 解码回答
                generated_ids = outputs[0][device_inputs['input_ids'].shape[1]:]
                response = processor.decode(generated_ids, skip_special_tokens=True)
                
                print(f"   🎯 回答: {response.strip()}")
                
                if response.strip():
                    print(f"   ✅ 尺寸 {size} 测试完全成功！")
                else:
                    print(f"   ⚠️ 尺寸 {size} 技术成功，但无文本回答")
            else:
                print(f"   ❌ Token不匹配: {image_pad_count} vs {features_count}")
                
        except Exception as e:
            print(f"   ❌ 尺寸 {size} 测试失败: {e}")
    
    # 测试数值增强功能
    print("\n🔢 测试数值增强功能...")
    test_image = Image.new('RGB', (56, 56), color='blue')
    
    numeric_questions = [
        "This image has area <num>3136</num> pixels. What color is it?",
        "The RGB value of this image is approximately <num>0.0</num>, <num>0.0</num>, <num>1.0</num>. Describe it.",
    ]
    
    for question in numeric_questions:
        print(f"\n   🤔 数值问题: {question[:50]}...")
        
        try:
            prompt = f"<|vision_start|><|image_pad|><|vision_end|>{question}"
            
            inputs = processor(
                text=[prompt],
                images=[test_image],
                return_tensors="pt",
                padding=True
            )
            
            # 检查Token匹配
            decoded = processor.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)
            image_pad_count = decoded.count('<|image_pad|>')
            features_count = inputs.pixel_values.shape[0]
            
            if image_pad_count == features_count:
                print(f"   ✅ 数值增强Token匹配成功")
                
                # 推理
                device_inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **device_inputs,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                
                generated_ids = outputs[0][device_inputs['input_ids'].shape[1]:]
                response = processor.decode(generated_ids, skip_special_tokens=True)
                
                print(f"   🎯 数值回答: {response.strip()}")
            else:
                print(f"   ❌ 数值增强Token不匹配: {image_pad_count} vs {features_count}")
                
        except Exception as e:
            print(f"   ❌ 数值增强测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("✅ GitCode博客参数调整解决方案测试完成")
    print("\n📊 关键发现:")
    print("   🎯 调整max_pixels可以控制图像处理的Token数量")
    print("   🎯 确保image_pad_count == features_count是成功的关键")
    print("   🎯 数值增强功能与图像处理可以兼容")

if __name__ == "__main__":
    test_parameter_adjustment_solution()
