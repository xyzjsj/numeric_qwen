#!/usr/bin/env python3
"""
基于GitCode文章和DeepWiki建议的完整Qwen2.5-VL图像Token修复解决方案
参考链接:
- https://blog.gitcode.com/d41d68b8e2ccdd03c0c59a4ca19a517b.html
- https://blog.gitcode.com/ca29fd2798662a2888c8ed01fd2a4207.html
"""
import os
import torch
from PIL import Image
import json

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加数值增强模块路径
import sys
sys.path.append('/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250')

from numeric_qwen2_5_vl import NumericQwen2_5_VLForConditionalGeneration, NumericQwen2_5_VLProcessor

def fix_chat_template(checkpoint_path):
    """
    修复chat template配置
    基于GitCode文章的解决方案
    """
    tokenizer_config_path = os.path.join(checkpoint_path, "tokenizer_config.json")
    
    print(f"🔧 检查和修复chat template: {tokenizer_config_path}")
    
    try:
        # 读取现有配置
        with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 检查是否有chat_template
        if 'chat_template' not in config or not config['chat_template']:
            print("   ⚠️ 缺少chat_template，正在添加...")
            
            # 添加标准的Qwen2.5-VL chat template
            config['chat_template'] = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
            
            # 备份原文件
            import shutil
            shutil.copy2(tokenizer_config_path, f"{tokenizer_config_path}.backup")
            
            # 写入修复后的配置
            with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print("   ✅ chat_template已添加并保存")
        else:
            print("   ✅ chat_template已存在")
            
    except Exception as e:
        print(f"   ❌ 修复chat_template失败: {e}")

def load_checkpoint_model_with_fixes(checkpoint_path):
    """加载模型并应用所有修复"""
    print(f"📂 加载检查点并应用修复: {checkpoint_path}")
    
    # 首先修复chat template
    fix_chat_template(checkpoint_path)
    
    try:
        # 加载模型
        model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载处理器
        processor = NumericQwen2_5_VLProcessor.from_pretrained(
            checkpoint_path,
            trust_remote_code=True
        )
        
        print("✅ 数值增强检查点加载成功")
        return model, processor
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise

def create_optimal_image_with_exact_tokens(target_tokens=64):
    """
    创建精确Token数量的图像
    根据公式: Token数量 = 图像像素数 / (28 * 28)
    """
    target_pixels = target_tokens * 28 * 28
    
    # 计算接近正方形的尺寸
    import math
    side_length = int(math.sqrt(target_pixels))
    
    # 微调到最接近目标像素数
    width = side_length
    height = target_pixels // width
    
    actual_pixels = width * height
    actual_tokens = actual_pixels // (28 * 28)
    
    print(f"🎨 创建精确Token图像:")
    print(f"   目标Token数: {target_tokens}")
    print(f"   图像尺寸: {width}x{height}")
    print(f"   实际像素数: {actual_pixels}")
    print(f"   实际Token数: {actual_tokens}")
    
    # 创建测试图像
    image = Image.new('RGB', (width, height), color='lightblue')
    
    # 添加视觉特征
    import numpy as np
    img_array = np.array(image)
    
    # 添加彩色条纹
    for i in range(0, height, 20):
        img_array[i:i+10, :] = [255, 200, 100]  # 橙色
    
    for j in range(0, width, 30):
        img_array[:, j:j+15] = [100, 255, 100]  # 绿色
    
    # 添加对角线
    for k in range(min(width, height)):
        if k < height and k < width:
            img_array[k, k] = [255, 0, 0]  # 红色对角线
    
    return Image.fromarray(img_array)

def test_with_proper_chat_template(model, processor, image, question):
    """
    使用正确的chat template进行VQA测试
    """
    try:
        print(f"🤔 问题: {question}")
        
        # 方法1: 使用apply_chat_template（如果可用）
        try:
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # 尝试使用chat template
            text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            print("✅ 使用apply_chat_template成功")
            
        except Exception as e:
            print(f"⚠️ apply_chat_template失败: {e}")
            print("🔄 使用手动构建的格式...")
            
            # 方法2: 手动构建正确格式
            text = f"<|im_start|>user\n<|image|>{question}<|im_end|>\n<|im_start|>assistant\n"
        
        print(f"📝 最终prompt长度: {len(text)}")
        
        # 处理输入
        image_inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        )
        
        print("📦 输入处理成功")
        print(f"📊 input_ids shape: {image_inputs.input_ids.shape}")
        
        if hasattr(image_inputs, 'pixel_values') and image_inputs.pixel_values is not None:
            print(f"🖼️ pixel_values shape: {image_inputs.pixel_values.shape}")
        
        # 检查Token数量
        decoded_text = processor.tokenizer.decode(image_inputs.input_ids[0], skip_special_tokens=False)
        image_token_count = decoded_text.count('<|image_pad|>')
        print(f"🔢 实际图像Token数量: {image_token_count}")
        
        # 移动到设备
        device_inputs = {}
        for k, v in image_inputs.items():
            if hasattr(v, 'to'):
                device_inputs[k] = v.to(model.device)
            else:
                device_inputs[k] = v
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **device_inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # 解码回答
        generated_ids = outputs[0][device_inputs['input_ids'].shape[1]:]
        response = processor.decode(generated_ids, skip_special_tokens=True)
        
        print(f"🎯 回答: {response}")
        return response
        
    except Exception as e:
        print(f"❌ VQA推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def comprehensive_test_suite():
    """综合测试套件"""
    print("🚀 开始综合图像Token修复测试")
    print("=" * 70)
    
    checkpoint_path = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"
    
    try:
        # 加载模型（包含所有修复）
        model, processor = load_checkpoint_model_with_fixes(checkpoint_path)
        
        # 测试不同Token数量的图像
        test_configs = [
            {"tokens": 16, "question": "这张图片的主要颜色是什么？"},
            {"tokens": 64, "question": "描述这张图片中的图案和特征。"},
            {"tokens": 144, "question": "这张图片给你什么感觉？"}
        ]
        
        for config in test_configs:
            print(f"\n📋 测试配置: {config['tokens']} tokens")
            print("-" * 50)
            
            # 创建精确Token数量的图像
            image = create_optimal_image_with_exact_tokens(config['tokens'])
            
            # 进行VQA测试
            result = test_with_proper_chat_template(
                model, processor, image, config['question']
            )
            
            if result:
                print("✅ 测试成功")
            else:
                print("❌ 测试失败")
        
        print("\n🎉 综合测试完成！")
        
    except Exception as e:
        print(f"❌ 综合测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    comprehensive_test_suite()
