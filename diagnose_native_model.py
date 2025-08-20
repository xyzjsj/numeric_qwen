#!/usr/bin/env python3
"""
测试原生Qwen2.5-VL模型是否能正常处理图像Token
用于诊断问题是否在我们的自定义模型中
"""
import os
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import json

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def test_native_qwen2_5_vl():
    """测试原生Qwen2.5-VL模型"""
    print("🔍 测试原生Qwen2.5-VL模型...")
    
    # 使用我们checkpoint的配置，但加载原生模型类
    checkpoint_path = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"
    
    try:
        # 使用原生模型类加载我们的checkpoint
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
        
        processor = AutoProcessor.from_pretrained(
            checkpoint_path,
            trust_remote_code=True
        )
        
        # 手动设置chat template
        official_chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        
        processor.chat_template = official_chat_template
        
        print("✅ 原生模型加载成功")
        
        # 创建测试图像
        test_image = Image.new('RGB', (224, 224), color='red')
        
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
        
        print(f"📝 Chat template应用成功，长度: {len(text)}")
        
        # 处理输入
        image_inputs = processor(
            text=[text],
            images=[test_image],
            return_tensors="pt",
            padding=True
        )
        
        print(f"📦 输入处理成功")
        print(f"📊 input_ids shape: {image_inputs.input_ids.shape}")
        print(f"🖼️ pixel_values shape: {image_inputs.pixel_values.shape}")
        
        # 检查token数量
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
        
        # 测试生成
        print("🚀 开始原生模型生成...")
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
        
        print(f"🎯 原生模型回答: {response}")
        print("✅ 原生模型测试成功！")
        
        return True
        
    except Exception as e:
        print(f"❌ 原生模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_custom_model():
    """对比我们的自定义模型"""
    print("\n🔄 加载自定义数值增强模型进行对比...")
    
    checkpoint_path = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"
    
    # 添加数值增强模块路径
    import sys
    sys.path.append('/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250')
    
    try:
        from numeric_qwen2_5_vl import NumericQwen2_5_VLForConditionalGeneration, NumericQwen2_5_VLProcessor
        
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
        
        print("✅ 自定义模型加载成功")
        print(f"📋 模型类型: {type(model)}")
        print(f"📋 处理器类型: {type(processor)}")
        
        # 检查forward方法的签名
        import inspect
        forward_sig = inspect.signature(model.forward)
        print(f"📋 Forward方法参数: {list(forward_sig.parameters.keys())}")
        
    except Exception as e:
        print(f"❌ 自定义模型加载失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("🚀 诊断图像Token匹配问题")
    print("=" * 60)
    
    # 测试原生模型
    native_success = test_native_qwen2_5_vl()
    
    # 对比自定义模型
    compare_with_custom_model()
    
    if native_success:
        print("\n✅ 诊断结论: 原生模型工作正常，问题可能在自定义模型实现中")
    else:
        print("\n❌ 诊断结论: 原生模型也有问题，可能是配置或环境问题")

if __name__ == "__main__":
    main()
