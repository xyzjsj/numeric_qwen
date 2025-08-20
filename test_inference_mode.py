#!/usr/bin/env python3
"""
修复版本的图像Token匹配测试
绕过自定义模型的数值增强功能，直接进行推理
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

def test_inference_mode():
    """在纯推理模式下测试图像Token匹配"""
    print("🚀 测试纯推理模式...")
    
    checkpoint_path = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"
    
    try:
        # 加载模型
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
        
        # 设置chat template
        official_chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        
        processor.chat_template = official_chat_template
        
        print("✅ 模型和处理器加载成功")
        
        # 创建测试图像
        test_image = Image.new('RGB', (224, 224), color='blue')
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image},
                    {"type": "text", "text": "这张图片是什么颜色？"}
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
        
        # 移动到设备
        device_inputs = {}
        for k, v in image_inputs.items():
            if hasattr(v, 'to'):
                device_inputs[k] = v.to(model.device)
            else:
                device_inputs[k] = v
        
        # 方法1：直接调用父类的forward方法（绕过我们的自定义forward）
        print("\n🔄 方法1: 直接调用父类forward...")
        try:
            with torch.no_grad():
                # 直接调用Qwen2VLForConditionalGeneration的forward方法
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2VLForConditionalGeneration
                
                # 临时将模型转换为父类实例来调用forward
                parent_outputs = Qwen2VLForConditionalGeneration.forward(
                    model, 
                    **device_inputs
                )
                
                print("✅ 父类forward调用成功！")
                print(f"📊 输出logits shape: {parent_outputs.logits.shape}")
                
                # 尝试生成
                parent_generate_outputs = Qwen2VLForConditionalGeneration.generate(
                    model,
                    **device_inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
                
                # 解码回答
                generated_ids = parent_generate_outputs[0][device_inputs['input_ids'].shape[1]:]
                response = processor.decode(generated_ids, skip_special_tokens=True)
                
                print(f"🎯 父类模型回答: {response}")
                print("✅ 方法1成功！")
                
        except Exception as e:
            print(f"❌ 方法1失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 方法2：修改我们模型的forward行为
        print("\n🔄 方法2: 修改自定义模型forward行为...")
        try:
            # 清理device_inputs，移除可能的额外参数
            clean_device_inputs = {}
            valid_params = ['input_ids', 'attention_mask', 'pixel_values', 'pixel_values_videos', 
                           'image_grid_thw', 'video_grid_thw', 'position_ids']
            
            for k, v in device_inputs.items():
                if k in valid_params:
                    clean_device_inputs[k] = v
            
            print(f"🧹 清理后的输入参数: {list(clean_device_inputs.keys())}")
            
            # 暂时禁用数值增强功能
            original_forward = model.forward
            
            def simple_forward(self, **kwargs):
                """简化的forward方法，直接调用父类"""
                # 移除我们添加的自定义参数
                clean_kwargs = {k: v for k, v in kwargs.items() 
                              if k not in ['numeric_values', 'numeric_positions']}
                
                # 直接调用父类forward
                return super(NumericQwen2_5_VLForConditionalGeneration, self).forward(**clean_kwargs)
            
            # 临时替换forward方法
            import types
            model.forward = types.MethodType(simple_forward, model)
            
            with torch.no_grad():
                outputs = model.generate(
                    **clean_device_inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            # 解码回答
            generated_ids = outputs[0][clean_device_inputs['input_ids'].shape[1]:]
            response = processor.decode(generated_ids, skip_special_tokens=True)
            
            print(f"🎯 修改后模型回答: {response}")
            print("✅ 方法2成功！")
            
            # 恢复原始forward方法
            model.forward = original_forward
            
        except Exception as e:
            print(f"❌ 方法2失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 确保恢复原始forward方法
            model.forward = original_forward
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference_mode()
