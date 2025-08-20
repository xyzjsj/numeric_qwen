#!/usr/bin/env python3
"""
基于官方Qwen2.5-VL Chat Template文档的完整解决方案
参考文档: Qwen2.5-VL Chat Template Format
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

def fix_chat_template_official(checkpoint_path):
    """
    修复chat template配置 - 使用官方文档中的完整模板
    """
    tokenizer_config_path = os.path.join(checkpoint_path, "tokenizer_config.json")
    chat_template_path = os.path.join(checkpoint_path, "chat_template.json")
    
    print(f"🔧 修复官方chat template配置...")
    
    # 官方完整的Chat Template
    official_chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    
    try:
        # 1. 更新tokenizer_config.json
        with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        config['chat_template'] = official_chat_template
        
        # 备份并保存
        import shutil
        shutil.copy2(tokenizer_config_path, f"{tokenizer_config_path}.backup")
        
        with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print("   ✅ tokenizer_config.json 已更新")
        
        # 2. 创建独立的chat_template.json文件
        chat_template_content = {
            "chat_template": official_chat_template
        }
        
        with open(chat_template_path, 'w', encoding='utf-8') as f:
            json.dump(chat_template_content, f, indent=2, ensure_ascii=False)
        
        print("   ✅ chat_template.json 已创建")
        
    except Exception as e:
        print(f"   ❌ 修复失败: {e}")

def load_model_with_official_template(checkpoint_path):
    """加载模型并应用官方模板修复"""
    print(f"📂 加载模型并应用官方Chat Template修复...")
    
    # 修复chat template
    fix_chat_template_official(checkpoint_path)
    
    try:
        # 加载模型
        model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载处理器 - 重新从修复后的配置加载
        processor = NumericQwen2_5_VLProcessor.from_pretrained(
            checkpoint_path,
            trust_remote_code=True
        )
        
        # 手动设置chat template（如果处理器仍然没有加载到）
        if not hasattr(processor, 'chat_template') or not processor.chat_template:
            print("🔄 手动设置chat template...")
            
            # 读取chat template
            chat_template_path = os.path.join(checkpoint_path, "chat_template.json")
            with open(chat_template_path, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
            
            # 设置到处理器
            processor.chat_template = template_data['chat_template']
            print("✅ 手动设置chat template成功")
        
        print("✅ 模型和处理器加载成功")
        return model, processor
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise

def create_test_image(size=(224, 224)):
    """创建测试图像"""
    print(f"🎨 创建测试图像: {size}")
    
    image = Image.new('RGB', size, color='lightblue')
    
    # 添加视觉特征
    import numpy as np
    img_array = np.array(image)
    
    # 添加红色对角线
    for i in range(min(size)):
        if i < img_array.shape[0] and i < img_array.shape[1]:
            img_array[i, i] = [255, 0, 0]
    
    # 添加绿色条纹
    for i in range(0, size[1], 20):
        if i < img_array.shape[1]:
            img_array[:, i:i+5] = [0, 255, 0]
    
    return Image.fromarray(img_array)

def test_official_chat_template(model, processor, image, question):
    """使用官方Chat Template格式进行测试"""
    try:
        print(f"🤔 问题: {question}")
        
        # 构建官方格式的消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        print("📝 构建官方格式消息成功")
        
        # 使用apply_chat_template
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            add_vision_id=False  # 不添加图片编号
        )
        
        print(f"✅ 官方Chat Template应用成功")
        print(f"📄 生成的模板长度: {len(text)}")
        
        # 显示生成的模板片段
        print(f"📋 模板预览: {text[:200]}...")
        
        # 处理输入
        image_inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )
        
        print("📦 输入处理成功")
        print(f"📊 input_ids shape: {image_inputs.input_ids.shape}")
        
        if hasattr(image_inputs, 'pixel_values') and image_inputs.pixel_values is not None:
            print(f"🖼️ pixel_values shape: {image_inputs.pixel_values.shape}")
        
        # 检查vision tokens
        decoded_text = processor.tokenizer.decode(image_inputs.input_ids[0], skip_special_tokens=False)
        vision_start_count = decoded_text.count('<|vision_start|>')
        vision_end_count = decoded_text.count('<|vision_end|>')
        image_pad_count = decoded_text.count('<|image_pad|>')
        
        print(f"🔍 Vision Token统计:")
        print(f"   <|vision_start|>: {vision_start_count}")
        print(f"   <|vision_end|>: {vision_end_count}")
        print(f"   <|image_pad|>: {image_pad_count}")
        
        # 移动到设备
        device_inputs = {}
        for k, v in image_inputs.items():
            if hasattr(v, 'to'):
                device_inputs[k] = v.to(model.device)
            else:
                device_inputs[k] = v
        
        # 生成回答
        print("🚀 开始生成回答...")
        with torch.no_grad():
            outputs = model.generate(
                **device_inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # 解码回答
        generated_ids = outputs[0][device_inputs['input_ids'].shape[1]:]
        response = processor.decode(generated_ids, skip_special_tokens=True)
        
        print(f"🎯 模型回答: {response}")
        return response
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主测试函数"""
    print("🚀 开始官方Chat Template格式测试")
    print("=" * 70)
    
    checkpoint_path = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"
    
    try:
        # 加载模型（应用官方模板修复）
        model, processor = load_model_with_official_template(checkpoint_path)
        
        # 创建测试图像
        test_image = create_test_image((224, 224))
        
        # 测试问题列表
        test_questions = [
            "这张图片中有什么颜色？",
            "描述一下这张图片的特征。",
            "你看到了什么图案？"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📋 测试 {i}/{len(test_questions)}")
            print("-" * 50)
            
            result = test_official_chat_template(model, processor, test_image, question)
            
            if result:
                print("✅ 测试成功！")
            else:
                print("❌ 测试失败")
        
        print("\n🎉 官方Chat Template测试完成！")
        
    except Exception as e:
        print(f"❌ 主测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
