#!/usr/bin/env python3
"""
简化的模型加载和使用示例
修复了设备移动问题
"""
import os
import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def load_model_simple():
    """简单可靠的模型加载方式"""
    print("🚀 加载checkpoint-4250模型...")
    
    # 添加数值增强模块路径
    import sys
    sys.path.append('/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250')
    
    from numeric_qwen2_5_vl import NumericQwen2_5_VLForConditionalGeneration, NumericQwen2_5_VLProcessor
    
    checkpoint_path = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"
    
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
    
    print("✅ 模型加载成功")
    print(f"📋 模型类型: {type(model).__name__}")
    print(f"📋 模型设备: {model.device}")
    
    return model, processor

def test_simple_inference(model, processor):
    """简单的推理测试"""
    print("\n🧪 测试简单推理...")
    
    # 测试文本
    test_prompt = "解释一下机器学习的基本概念。"
    
    # 构建输入
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # 处理输入
    inputs = processor(
        text=[prompt],
        return_tensors="pt"
    )
    
    # 安全的设备移动
    device_inputs = {}
    for k, v in inputs.items():
        if hasattr(v, 'to') and hasattr(v, 'device'):
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
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    # 解码回答
    generated_ids = outputs[0][device_inputs['input_ids'].shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    
    print(f"📝 问题: {test_prompt}")
    print(f"🎯 回答: {response}")
    
    return response

def test_numeric_enhancement(model, processor):
    """测试数值增强功能"""
    print("\n🔢 测试数值增强功能...")
    
    # 测试数值增强
    test_prompt = "处理这些数值：<num>3.14</num> 和 <num>-2.5</num>，计算它们的轨迹。"
    
    # 构建输入
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # 处理输入
    inputs = processor(
        text=[prompt],
        return_tensors="pt"
    )
    
    # 安全的设备移动
    device_inputs = {}
    for k, v in inputs.items():
        if hasattr(v, 'to') and hasattr(v, 'device'):
            device_inputs[k] = v.to(model.device)
        else:
            device_inputs[k] = v
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **device_inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    # 解码回答
    generated_ids = outputs[0][device_inputs['input_ids'].shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    
    print(f"📝 数值问题: {test_prompt}")
    print(f"🎯 数值回答: {response}")
    
    return response

def main():
    """主函数"""
    print("🎯 checkpoint-4250模型简单加载和使用")
    print("=" * 60)
    
    # 加载模型
    model, processor = load_model_simple()
    
    # 测试普通推理
    test_simple_inference(model, processor)
    
    # 测试数值增强
    test_numeric_enhancement(model, processor)
    
    print("\n✅ 测试完成！")
    print("\n📋 使用方法总结:")
    print("1. 导入模块:")
    print("   import sys")
    print("   sys.path.append('/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250')")
    print("   from numeric_qwen2_5_vl import NumericQwen2_5_VLForConditionalGeneration, NumericQwen2_5_VLProcessor")
    print()
    print("2. 加载模型:")
    print("   model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint_path, ...)")
    print("   processor = NumericQwen2_5_VLProcessor.from_pretrained(checkpoint_path, ...)")
    print()
    print("3. 推理:")
    print("   inputs = processor(text=[prompt], return_tensors='pt')")
    print("   outputs = model.generate(**inputs, max_new_tokens=100)")
    print("   response = processor.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)")

if __name__ == "__main__":
    main()
