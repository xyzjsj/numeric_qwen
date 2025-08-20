#!/usr/bin/env python3
"""
专门修复图像推理的方案
基于前面的所有测试结果，创建能正常处理图像的版本
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

class InferenceOnlyQwen2_5_VL(NumericQwen2_5_VLForConditionalGeneration):
    """
    推理专用版本的模型
    绕过数值增强的forward，直接调用父类方法
    """
    
    def forward(self, **kwargs):
        """
        推理专用的forward方法
        直接调用Qwen2VLForConditionalGeneration的forward
        """
        # 移除我们自定义的参数
        clean_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['numeric_values', 'numeric_positions']}
        
        # 直接调用父类的父类（即原始Qwen2VLForConditionalGeneration）的forward
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration.forward(self, **clean_kwargs)

def load_inference_model():
    """加载推理专用模型"""
    print("🔧 加载推理专用模型...")
    
    checkpoint_path = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"
    
    # 先用普通方式加载
    base_model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 转换为推理专用模型
    inference_model = InferenceOnlyQwen2_5_VL.from_pretrained(
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
    
    # 设置chat template
    official_chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    
    processor.chat_template = official_chat_template
    
    print("✅ 推理专用模型加载成功")
    return inference_model, processor

def test_image_inference_fixed(model, processor):
    """测试修复后的图像推理"""
    print("\n🖼️ 测试修复后的图像推理...")
    
    # 创建测试图像
    test_image = Image.new('RGB', (224, 224), color='red')
    
    # 在图像上添加一些特征
    import numpy as np
    img_array = np.array(test_image)
    
    # 添加蓝色对角线
    for i in range(min(224, 224)):
        img_array[i, i] = [0, 0, 255]
    
    # 添加绿色条纹
    for i in range(0, 224, 30):
        img_array[:, i:i+5] = [0, 255, 0]
    
    test_image = Image.fromarray(img_array)
    
    try:
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image},
                    {"type": "text", "text": "这张图片是什么颜色？图片中有什么图案？"}
                ]
            }
        ]
        
        # 应用chat template
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        print(f"📝 Chat template应用成功")
        
        # 处理输入
        image_inputs = processor(
            text=[text],
            images=[test_image],
            return_tensors="pt"
        )
        
        print(f"📦 输入处理成功")
        print(f"📊 input_ids shape: {image_inputs.input_ids.shape}")
        print(f"🖼️ pixel_values shape: {image_inputs.pixel_values.shape}")
        
        # 检查vision tokens
        decoded_text = processor.tokenizer.decode(image_inputs.input_ids[0], skip_special_tokens=False)
        image_pad_count = decoded_text.count('<|image_pad|>')
        print(f"🔢 图像pad token数量: {image_pad_count}")
        
        # 移动到设备
        device_inputs = {}
        for k, v in image_inputs.items():
            if hasattr(v, 'to'):
                device_inputs[k] = v.to(model.device)
            else:
                device_inputs[k] = v
        
        # 生成回答
        print("🚀 开始图像推理...")
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
        
        print(f"🎯 图像推理结果: {response}")
        print("✅ 图像推理成功！")
        
        return response
        
    except Exception as e:
        print(f"❌ 图像推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def alternative_approach():
    """备选方案：手动构建prompt"""
    print("\n🔄 备选方案：手动构建图像prompt...")
    
    checkpoint_path = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"
    
    try:
        # 使用原始模型但手动处理
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
        
        # 创建简单图像
        test_image = Image.new('RGB', (224, 224), color='blue')
        
        # 手动构建prompt（不使用chat template）
        manual_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>这张图片是什么颜色？<|im_end|>\n<|im_start|>assistant\n"
        
        # 手动处理输入
        inputs = processor(
            text=[manual_prompt],
            images=[test_image],
            return_tensors="pt"
        )
        
        print(f"📦 手动输入处理成功")
        print(f"📊 input_ids shape: {inputs.input_ids.shape}")
        
        # 临时修改模型的forward方法
        original_forward = model.forward
        
        def bypass_forward(self, **kwargs):
            """绕过数值增强的forward"""
            clean_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['numeric_values', 'numeric_positions']}
            # 直接调用父类forward
            return super(NumericQwen2_5_VLForConditionalGeneration, self).forward(**clean_kwargs)
        
        import types
        model.forward = types.MethodType(bypass_forward, model)
        
        # 移动到设备并生成
        device_inputs = {}
        for k, v in inputs.items():
            if hasattr(v, 'to'):
                device_inputs[k] = v.to(model.device)
            else:
                device_inputs[k] = v
        
        with torch.no_grad():
            outputs = model.generate(
                **device_inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # 恢复原始forward
        model.forward = original_forward
        
        # 解码结果
        generated_ids = outputs[0][device_inputs['input_ids'].shape[1]:]
        response = processor.decode(generated_ids, skip_special_tokens=True)
        
        print(f"🎯 备选方案结果: {response}")
        print("✅ 备选方案成功！")
        
        return response
        
    except Exception as e:
        print(f"❌ 备选方案失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主测试函数"""
    print("🎯 专门修复图像推理问题")
    print("=" * 60)
    
    # 方案1：推理专用模型
    try:
        inference_model, processor = load_inference_model()
        result1 = test_image_inference_fixed(inference_model, processor)
        
        if result1:
            print("\n🎉 方案1成功：推理专用模型工作正常！")
        else:
            print("\n⚠️ 方案1失败，尝试备选方案...")
            
    except Exception as e:
        print(f"\n❌ 方案1加载失败: {e}")
        print("🔄 尝试备选方案...")
    
    # 方案2：备选方案
    result2 = alternative_approach()
    
    if result2:
        print("\n🎉 方案2成功：备选方案工作正常！")
    else:
        print("\n❌ 两个方案都失败了")
    
    print("\n📋 总结:")
    if result1 or result2:
        print("✅ 图像推理问题已解决！")
        print("💡 解决方案：创建绕过数值增强forward的推理模型")
    else:
        print("❌ 图像推理问题仍需进一步调试")
        print("💡 建议：检查模型权重和forward方法实现")

if __name__ == "__main__":
    main()
