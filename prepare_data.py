#!/usr/bin/env python3
"""
数值增强Qwen2.5-VL模型的数据准备工具

用于创建训练数据和验证数据格式
"""

import json
import os
from typing import List, Dict, Any
from PIL import Image
import random


def create_sample_numeric_data(output_path: str, num_samples: int = 100):
    """
    创建示例数值训练数据
    
    Args:
        output_path: 输出文件路径
        num_samples: 样本数量
    """
    
    # 数学问题模板
    math_templates = [
        {
            "question": "计算 {a} + {b} 的结果",
            "answer": "计算结果是 <num><{result}>",
            "type": "addition"
        },
        {
            "question": "计算 {a} × {b} 的乘积", 
            "answer": "{a} × {b} = <num><{result}>",
            "type": "multiplication"
        },
        {
            "question": "计算 {a} ÷ {b} 的商",
            "answer": "{a} ÷ {b} = <num><{result}>",
            "type": "division"
        },
        {
            "question": "π的近似值是多少？",
            "answer": "π的近似值是 <num><3.14159>",
            "type": "constant"
        },
        {
            "question": "e的近似值是多少？",
            "answer": "自然常数e的近似值是 <num><2.71828>",
            "type": "constant"
        },
        {
            "question": "黄金比例的值是什么？",
            "answer": "黄金比例φ = <num><1.618>",
            "type": "constant"
        }
    ]
    
    # 图表分析模板
    chart_templates = [
        {
            "question": "这个柱状图显示的最大值是多少？",
            "answer": "从图表可以看出，最大值是 <num><{max_val}>",
            "image": "chart_{id}.jpg"
        },
        {
            "question": "图表中的平均值是多少？",
            "answer": "根据图表数据计算，平均值为 <num><{avg_val}>",
            "image": "chart_{id}.jpg"  
        },
        {
            "question": "这个图表显示的增长率是多少？",
            "answer": "图表显示的增长率为 <num><{growth_rate}>%",
            "image": "growth_chart_{id}.jpg"
        }
    ]
    
    samples = []
    
    for i in range(num_samples):
        sample_id = f"sample_{i:04d}"
        
        if i % 3 == 0:  # 数学问题
            template = random.choice(math_templates)
            
            if template["type"] in ["addition", "multiplication", "division"]:
                a = round(random.uniform(1, 100), 2)
                b = round(random.uniform(1, 100), 2)
                
                if template["type"] == "addition":
                    result = round(a + b, 2)
                elif template["type"] == "multiplication":
                    result = round(a * b, 2)
                else:  # division
                    result = round(a / b, 4)
                
                question = template["question"].format(a=a, b=b)
                answer = template["answer"].format(a=a, b=b, result=result)
            else:  # 常数
                question = template["question"]
                answer = template["answer"]
            
            sample = {
                "id": sample_id,
                "conversations": [
                    {
                        "from": "human",
                        "value": question
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ]
            }
            
        else:  # 图表分析问题
            template = random.choice(chart_templates)
            
            # 生成随机数值
            max_val = round(random.uniform(50, 1000), 1)
            avg_val = round(random.uniform(20, max_val), 1)
            growth_rate = round(random.uniform(5, 50), 1)
            
            question = template["question"]
            answer = template["answer"].format(
                max_val=max_val,
                avg_val=avg_val, 
                growth_rate=growth_rate
            )
            # 使用固定的图像索引来匹配实际存在的图像文件
            chart_index = i % 50  # 确保索引在0-49范围内
            image_name = template["image"].format(id=chart_index)
            
            sample = {
                "id": sample_id,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{question}"
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ],
                "image": image_name
            }
        
        samples.append(sample)
    
    # 保存数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"已创建 {num_samples} 条训练样本，保存到: {output_path}")
    return samples


def create_sample_images(image_folder: str, num_images: int = 50):
    """
    创建示例图像（占位符）
    
    Args:
        image_folder: 图像文件夹路径
        num_images: 图像数量
    """
    
    os.makedirs(image_folder, exist_ok=True)
    
    # 创建简单的占位符图像
    for i in range(num_images):
        # 为每个索引创建两种类型的图像文件
        # 这样确保模板中引用的所有图像都存在
        
        # chart_{i}.jpg
        chart_path = os.path.join(image_folder, f"chart_{i}.jpg")
        img = Image.new('RGB', (640, 480), color=(
            random.randint(100, 255),
            random.randint(100, 255), 
            random.randint(100, 255)
        ))
        img.save(chart_path)
        
        # growth_chart_{i}.jpg
        growth_chart_path = os.path.join(image_folder, f"growth_chart_{i}.jpg")
        img = Image.new('RGB', (640, 480), color=(
            random.randint(100, 255),
            random.randint(100, 255), 
            random.randint(100, 255)
        ))
        img.save(growth_chart_path)
    
    print(f"已创建 {num_images * 2} 张示例图像（chart_*.jpg 和 growth_chart_*.jpg），保存到: {image_folder}")


def validate_data_format(data_path: str) -> bool:
    """
    验证数据格式是否正确
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        是否格式正确
    """
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("错误: 数据应该是一个列表")
            return False
        
        for i, item in enumerate(data):
            # 检查必需字段
            if 'id' not in item:
                print(f"错误: 样本 {i} 缺少 'id' 字段")
                return False
                
            if 'conversations' not in item:
                print(f"错误: 样本 {i} 缺少 'conversations' 字段")
                return False
            
            conversations = item['conversations']
            if not isinstance(conversations, list):
                print(f"错误: 样本 {i} 的 'conversations' 应该是列表")
                return False
            
            # 检查对话格式
            for j, conv in enumerate(conversations):
                if 'from' not in conv or 'value' not in conv:
                    print(f"错误: 样本 {i} 对话 {j} 缺少 'from' 或 'value' 字段")
                    return False
                
                if conv['from'] not in ['human', 'gpt']:
                    print(f"警告: 样本 {i} 对话 {j} 的 'from' 字段值不标准: {conv['from']}")
        
        print(f"数据格式验证通过，共 {len(data)} 条样本")
        return True
        
    except Exception as e:
        print(f"验证数据格式时出错: {e}")
        return False


def analyze_numeric_patterns(data_path: str):
    """
    分析数据中的数值模式
    
    Args:
        data_path: 数据文件路径
    """
    
    import re
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    numeric_pattern = re.compile(r'<num><([+-]?\d*\.?\d+)>')
    
    total_samples = len(data)
    samples_with_numeric = 0
    numeric_values = []
    
    for item in data:
        found_numeric = False
        for conv in item['conversations']:
            if conv['from'] == 'gpt':  # 只分析回复中的数值
                matches = numeric_pattern.findall(conv['value'])
                if matches:
                    found_numeric = True
                    for match in matches:
                        try:
                            value = float(match)
                            numeric_values.append(value)
                        except ValueError:
                            continue
        
        if found_numeric:
            samples_with_numeric += 1
    
    print(f"=== 数值模式分析 ===")
    print(f"总样本数: {total_samples}")
    print(f"包含数值的样本: {samples_with_numeric}")
    print(f"数值覆盖率: {samples_with_numeric/total_samples*100:.1f}%")
    print(f"总数值数量: {len(numeric_values)}")
    
    if numeric_values:
        print(f"数值范围: {min(numeric_values):.4f} ~ {max(numeric_values):.4f}")
        print(f"平均值: {sum(numeric_values)/len(numeric_values):.4f}")


def main():
    """
    数据准备主函数
    """
    
    # 设置路径
    output_dir = "/data1/wangzhiye/1a1a11/original/data"
    os.makedirs(output_dir, exist_ok=True)
    
    data_path = os.path.join(output_dir, "numeric_training_data.json")
    image_folder = os.path.join(output_dir, "images")
    
    print("=== 数值增强Qwen2.5-VL数据准备 ===")
    
    # 创建训练数据
    print("1. 创建训练数据...")
    create_sample_numeric_data(data_path, num_samples=200)
    
    # 创建示例图像
    print("\n2. 创建示例图像...")
    create_sample_images(image_folder, num_images=100)
    
    # 验证数据格式
    print("\n3. 验证数据格式...")
    validate_data_format(data_path)
    
    # 分析数值模式
    print("\n4. 分析数值模式...")
    analyze_numeric_patterns(data_path)
    
    print(f"\n数据准备完成！")
    print(f"训练数据: {data_path}")
    print(f"图像数据: {image_folder}")
    print(f"\n可以使用以下命令开始训练:")
    print(f"cd /data1/wangzhiye/1a1a11/original")
    print(f"python train.py")


if __name__ == "__main__":
    main()
