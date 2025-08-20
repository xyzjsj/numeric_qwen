#!/usr/bin/env python3
"""
LoRA微调后的数值增强Qwen2.5-VL模型推理脚本
"""

import os
import sys
import torch
from PIL import Image
from transformers import AutoTokenizer
from peft import PeftModel
import json
import re

# 添加当前目录到path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from numeric_qwen2_5_vl import (
    NumericQwen2_5_VLConfig,
    NumericQwen2_5_VLProcessor,
    NumericQwen2_5_VLForConditionalGeneration
)


def load_lora_model(base_model_path: str, lora_adapter_path: str):
    """
    加载LoRA微调后的模型
    """
    print(f"加载基础模型: {base_model_path}")
    
    # 加载基础模型
    config = NumericQwen2_5_VLConfig.from_pretrained(base_model_path)
    base_model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"加载LoRA适配器: {lora_adapter_path}")
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        torch_dtype=torch.bfloat16
    )
    
    # 合并权重以提高推理速度（可选）
    # model = model.merge_and_unload()
    
    # 加载处理器
    processor = NumericQwen2_5_VLProcessor.from_pretrained(lora_adapter_path)
    
    return model, processor


def extract_numbers_from_text(text: str):
    """
    从文本中提取数值
    """
    # 匹配 <num><value> 格式的数值
    pattern = r'<num><([+-]?\d*\.?\d+)>'
    matches = re.findall(pattern, text)
    
    numbers = []
    for match in matches:
        try:
            # 尝试转换为浮点数
            number = float(match)
            numbers.append(number)
        except ValueError:
            continue
    
    return numbers


def format_conversation(messages):
    """
    格式化对话为Qwen2.5-VL的格式
    """
    formatted_messages = []
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if isinstance(content, list):
            # 多模态消息
            formatted_content = []
            for item in content:
                if item["type"] == "text":
                    formatted_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    formatted_content.append({"type": "image", "image": item["image"]})
            formatted_messages.append({"role": role, "content": formatted_content})
        else:
            # 纯文本消息
            formatted_messages.append({"role": role, "content": content})
    
    return formatted_messages


def inference_single_sample(model, processor, messages, max_new_tokens=512):
    """
    单样本推理
    """
    # 格式化输入
    formatted_messages = format_conversation(messages)
    
    # 准备输入
    text = processor.apply_chat_template(
        formatted_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 处理图像和文本
    inputs = processor(
        text=[text],
        images=[msg["content"] for msg in formatted_messages 
               if isinstance(msg["content"], list) 
               for item in msg["content"] 
               if item["type"] == "image"],
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    # 解码输出
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # 提取数值
    numbers = extract_numbers_from_text(response)
    
    return {
        "response": response,
        "extracted_numbers": numbers
    }


def batch_inference(model, processor, data_path: str, output_path: str = None):
    """
    批量推理
    """
    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    for i, item in enumerate(data):
        print(f"处理样本 {i+1}/{len(data)}")
        
        try:
            # 构建消息
            messages = item.get("messages", [])
            
            # 进行推理
            result = inference_single_sample(model, processor, messages)
            
            # 添加原始数据信息
            result.update({
                "sample_id": i,
                "original_messages": messages,
                "ground_truth": item.get("ground_truth", None)
            })
            
            results.append(result)
            
        except Exception as e:
            print(f"样本 {i+1} 处理失败: {e}")
            results.append({
                "sample_id": i,
                "error": str(e),
                "original_messages": item.get("messages", [])
            })
    
    # 保存结果
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_path}")
    
    return results


def main():
    """
    主函数
    """
    # 配置路径
    base_model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    lora_adapter_path = "/data1/wangzhiye/1a1a11/original/output_lora"  # LoRA适配器路径
    
    # 检查LoRA适配器是否存在
    if not os.path.exists(lora_adapter_path):
        print(f"LoRA适配器路径不存在: {lora_adapter_path}")
        print("请先完成LoRA训练")
        return
    
    # 加载模型
    print("正在加载LoRA模型...")
    model, processor = load_lora_model(base_model_path, lora_adapter_path)
    
    # 设置为评估模式
    model.eval()
    
    print("模型加载完成！")
    
    # 单样本推理示例
    print("\n=== 单样本推理示例 ===")
    
    # 示例消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图片中的车辆速度是多少？请用<num><速度值>的格式回答。"},
                {"type": "image", "image": "path/to/your/image.jpg"}  # 替换为实际图片路径
            ]
        }
    ]
    
    try:
        result = inference_single_sample(model, processor, messages)
        print(f"生成回复: {result['response']}")
        print(f"提取的数值: {result['extracted_numbers']}")
    except Exception as e:
        print(f"单样本推理失败: {e}")
    
    # 批量推理
    test_data_path = "/data1/wangzhiye/1a1a11/original/data/numeric_training_data.json"
    if os.path.exists(test_data_path):
        print(f"\n=== 批量推理测试 ===")
        output_path = "/data1/wangzhiye/1a1a11/original/inference_results.json"
        
        try:
            results = batch_inference(model, processor, test_data_path, output_path)
            print(f"批量推理完成，共处理 {len(results)} 个样本")
            
            # 简单统计
            successful_samples = [r for r in results if "error" not in r]
            failed_samples = [r for r in results if "error" in r]
            
            print(f"成功: {len(successful_samples)} 个")
            print(f"失败: {len(failed_samples)} 个")
            
            if successful_samples:
                # 计算平均提取的数值数量
                avg_numbers = sum(len(r["extracted_numbers"]) for r in successful_samples) / len(successful_samples)
                print(f"平均每个样本提取数值数量: {avg_numbers:.2f}")
                
        except Exception as e:
            print(f"批量推理失败: {e}")
    else:
        print(f"测试数据文件不存在: {test_data_path}")


if __name__ == "__main__":
    main()
