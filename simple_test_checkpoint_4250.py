#!/usr/bin/env python3
"""
简单测试checkpoint-4250模型的推理能力
"""

import json
import torch
import os
import sys
from PIL import Image

# 添加数值增强模块路径
sys.path.append('/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250')

# 设置HF镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from numeric_qwen2_5_vl import NumericQwen2_5_VLForConditionalGeneration, NumericQwen2_5_VLProcessor
import warnings
warnings.filterwarnings("ignore")

def load_checkpoint_model(checkpoint_path):
    """加载checkpoint-4250模型"""
    print(f"📂 加载检查点: {checkpoint_path}")
    
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
        print(f"❌ 检查点加载失败: {e}")
        raise

def simple_inference_test(model, processor, image_path, question):
    """简单的推理测试"""
    try:
        print(f"🖼️ 处理图像: {image_path}")
        print(f"❓ 问题: {question}")
        
        # 检查图像是否存在
        if not os.path.exists(image_path):
            print(f"❌ 图像文件不存在: {image_path}")
            return ""
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        print("🔄 生成回答中...")
        
        # 方法1: 尝试使用消息格式（如果支持）
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
            
            # 尝试应用聊天模板
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = processor.process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            print("✓ 使用标准消息格式")
            
        except (AttributeError, Exception) as e:
            print(f"⚠️ 标准格式失败: {e}")
            print("🔄 尝试简单格式...")
            
            # 方法2: 使用简单的文本+图像格式
            try:
                # 使用图像占位符
                text_with_image = f"<image>
{question}"
                inputs = processor(
                    text=text_with_image,
                    images=image,
                    return_tensors="pt",
                    padding=True
                )
                print("✓ 使用简单格式")
                
            except Exception as e2:
                print(f"⚠️ 简单格式也失败: {e2}")
                print("🔄 尝试最基础格式...")
                
                # 方法3: 最基础的格式
                inputs = processor(
                    text=question,
                    images=image,
                    return_tensors="pt"
                )
                print("✓ 使用基础格式")
        
        # 移动到GPU
        inputs = inputs.to("cuda")
        
        # 生成回答
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        # 解码回答
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        answer = output_text[0].strip()
        print(f"🤖 模型回答: {answer}")
        
        return answer
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        return ""

def test_numeric_capabilities(model, processor):
    """测试数值增强能力"""
    print("\n" + "="*60)
    print("🧮 测试数值增强能力")
    print("="*60)
    
    # 测试数值文本处理
    test_cases = [
        "这个轨迹包含坐标 <num><3.14> 和 <num><-2.5>",
        "位置信息: (<num><+11.3>, <num><-4.0>)",
        "轨迹 [PT, (<num><+3.41>, <num><-0.06>), (<num><+6.96>, <num><-0.20>)]"
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n测试数值文本 {i}:")
        print(f"原始: {test_text}")
        
        try:
            # 测试数值token处理
            result = processor._process_text_with_numeric_tokens(test_text)
            
            if isinstance(result, tuple):
                processed_text, numeric_values = result
                print(f"处理后: {processed_text}")
                print(f"数值: {numeric_values}")
            elif isinstance(result, dict):
                print(f"处理后: {result['text']}")
                print(f"数值: {result['numeric_values']}")
            else:
                print(f"结果: {result}")
                
        except Exception as e:
            print(f"❌ 数值处理失败: {e}")

def main():
    """主函数"""
    print("🚀 开始简单测试checkpoint-4250模型")
    print("="*60)
    
    # 配置路径
    checkpoint_path = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"
    
    # 1. 加载模型
    try:
        model, processor = load_checkpoint_model(checkpoint_path)
    except Exception as e:
        print(f"❌ 模型加载失败，退出测试: {e}")
        return
    
    # 2. 测试数值增强能力
    test_numeric_capabilities(model, processor)
    
    # 3. 简单的VQA测试
    print("\n" + "="*60)
    print("🖼️ 简单VQA推理测试")
    print("="*60)
    
    # 准备一些简单的测试用例
    test_cases = [
        {
            "image": "/data1/wangzhiye/data2/nuscenes/samples/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281439762460.jpg",
            "question": "描述一下这张图片中看到的内容。"
        },
        {
            "image": "/data1/wangzhiye/data2/nuscenes/samples/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281439762460.jpg", 
            "question": "这张图片中有什么交通元素？"
        },
        {
            "image": "/data1/wangzhiye/data2/nuscenes/samples/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281439762460.jpg",
            "question": "如果要规划一条轨迹，你会怎么建议？"
        }
    ]
    
    # 运行测试
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- 测试用例 {i} ---")
        answer = simple_inference_test(
            model, 
            processor, 
            test_case["image"], 
            test_case["question"]
        )
        
        if answer:
            print(f"✅ 测试用例 {i} 成功")
        else:
            print(f"❌ 测试用例 {i} 失败")
    
    print("\n" + "="*60)
    print("🎉 测试完成！")
    print("="*60)

if __name__ == "__main__":
    print(">>> 数值增强Qwen2.5-VL模型组件已注册")
    main()
