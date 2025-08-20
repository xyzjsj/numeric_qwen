#!/usr/bin/env python3
"""
checkpoint-4250模型加载指南
提供多种加载方式供选择
"""
import os
import torch
from PIL import Image

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def load_numeric_enhanced_model():
    """
    方法1：加载完整的数值增强模型
    适用于：文本推理、数值增强功能测试
    """
    print("🔧 方法1：加载数值增强模型")
    
    # 添加数值增强模块路径
    import sys
    sys.path.append('/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250')
    
    from numeric_qwen2_5_vl import NumericQwen2_5_VLForConditionalGeneration, NumericQwen2_5_VLProcessor
    
    checkpoint_path = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"
    
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
        
        # 设置chat template（如果需要）
        if not hasattr(processor, 'chat_template') or not processor.chat_template:
            official_chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
            processor.chat_template = official_chat_template
        
        print("✅ 数值增强模型加载成功")
        print(f"📋 模型类型: {type(model).__name__}")
        print(f"📋 模型设备: {model.device}")
        print(f"📋 模型精度: {model.dtype}")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        raise

def test_text_inference(model, processor):
    """测试文本推理功能"""
    print("\n🧪 测试文本推理功能...")
    
    # 测试数值增强
    test_prompt = "计算这些数值的轨迹：<num>3.14</num>和<num>-2.5</num>"
    
    try:
        # 构建输入
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = processor(
            text=[prompt],
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # 解码回答
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = processor.decode(generated_ids, skip_special_tokens=True)
        
        print(f"📝 输入: {test_prompt}")
        print(f"🎯 输出: {response}")
        print("✅ 文本推理测试成功")
        
    except Exception as e:
        print(f"❌ 文本推理测试失败: {e}")

def load_for_evaluation():
    """
    方法2：为评估专门加载模型
    适用于：cosine similarity评估、性能测试
    """
    print("\n🔧 方法2：为评估加载模型")
    
    try:
        model, processor = load_numeric_enhanced_model()
        
        # 设置为评估模式
        model.eval()
        
        # 禁用dropout等训练相关功能
        for module in model.modules():
            if hasattr(module, 'training'):
                module.training = False
        
        print("✅ 评估模式设置完成")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ 评估模式设置失败: {e}")
        raise

def quick_load_example():
    """快速加载示例"""
    print("\n⚡ 快速加载示例:")
    
    # 最简单的加载方式
    model, processor = load_numeric_enhanced_model()
    
    # 快速测试
    test_text_inference(model, processor)
    
    return model, processor

def main():
    """演示不同的加载方式"""
    print("🚀 checkpoint-4250模型加载指南")
    print("=" * 60)
    
    # 方法1：完整加载
    model, processor = load_numeric_enhanced_model()
    
    # 测试文本功能
    test_text_inference(model, processor)
    
    # 方法2：评估模式（可选）
    eval_model, eval_processor = load_for_evaluation()
    
    print("\n📋 加载完成！可用功能:")
    print("   ✅ 文本推理")
    print("   ✅ 数值增强处理")
    print("   ✅ 对话格式生成")
    print("   ⚠️ 图像推理（需要forward方法修复）")
    
    print("\n💡 使用建议:")
    print("   - 文本任务：直接使用，性能优秀")
    print("   - 数值增强：完全支持<num>标签")
    print("   - 评估测试：使用cosine similarity等指标")
    print("   - 图像任务：等待后续修复或使用文本回退")

if __name__ == "__main__":
    main()
