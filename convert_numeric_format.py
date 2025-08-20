#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数字格式转换脚本

"""

import json
import re
import os
from typing import Dict, List, Any

def convert_numbers_to_token_format(text: str) -> str:
    """
    将文本中的数字转换为 <num><数字> 格式
    整数转换为两位小数，小数保持原有位数
    
    例子：
    5.6 -> <num><5.6>
    4 -> <num><4.00>
    -123 -> <num><-123.00>
    +45.678 -> <num><+45.678>
    
    Args:
        text: 输入文本
        
    Returns:
        转换后的文本
    """
    # 匹配各种数字格式：
    # 1. 正负号 + 数字（整数或小数）
    # 2. 纯数字（整数或小数）
    # 3. 科学计数法
    
    # 正负号数字模式：(+1.5, -2.3, +123, -456等)
    pattern_signed = r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
    
    def replace_number(match):
        number_str = match.group(1)
        
        # 检查是否为科学计数法
        if 'e' in number_str.lower():
            # 科学计数法保持原样
            return f"<num><{number_str}>"
        
        # 检查是否包含小数点
        if '.' in number_str:
            # 已经是小数，保持原有位数
            return f"<num><{number_str}>"
        else:
            # 是整数，添加.00
            return f"<num><{number_str}.00>"
    
    # 转换所有匹配的数字
    converted_text = re.sub(pattern_signed, replace_number, text)
    
    return converted_text

def process_conversation(conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    处理对话中的数字
    
    Args:
        conversation: 对话列表
        
    Returns:
        处理后的对话列表
    """
    processed_conversation = []
    
    for message in conversation:
        processed_message = message.copy()
        if 'content' in processed_message:
            processed_message['content'] = convert_numbers_to_token_format(processed_message['content'])
        processed_conversation.append(processed_message)
    
    return processed_conversation

def convert_dataset(input_path: str, output_path: str):
    """
    转换整个数据集
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
    """
    print(f"开始转换数据集: {input_path}")
    
    # 读取原始数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据包含 {len(data)} 条记录")
    
    # 处理每条记录
    processed_data = []
    for i, item in enumerate(data):
        if i % 100 == 0:
            print(f"处理进度: {i+1}/{len(data)}")
        
        processed_item = item.copy()
        
        # 处理messages字段
        if 'messages' in processed_item:
            processed_item['messages'] = process_conversation(processed_item['messages'])
        
        processed_data.append(processed_item)
    
    # 保存转换后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！输出文件: {output_path}")
    print(f"处理了 {len(processed_data)} 条记录")

def test_conversion():
    """测试转换功能"""
    print("测试数字转换功能...")
    
    test_cases = [
        "位置 (+3.6, -2.0)",
        "长度: 4.6, 宽度: 1.9, 高度: 1.7",
        "角度: +93.0 度",
        "速度: (+0.2, +0.1)",
        "坐标 (-17.1, +33.4)",
        "数值 123.456",
        "负数 -789.012",
        "整数 4",
        "正整数 +123",
        "负整数 -456",
        "科学计数法 1.23e-4",
        "混合文本 车辆位置在 (+25.6, +8.0)，速度为 (0.0, +0.2)",
        "长度: 4, 宽度: 1, 高度: 2"
    ]
    
    for test_case in test_cases:
        converted = convert_numbers_to_token_format(test_case)
        print(f"原文: {test_case}")
        print(f"转换: {converted}")
        print()

def main():
    """主函数"""
    # 测试转换功能
    test_conversion()
    
    # 转换数据集
    input_file = "/data1/wangzhiye/LLaMA-Factory/data/5vqa_data_extracted_test_converted.json"
    output_file = "/data1/wangzhiye/LLaMA-Factory/data/5vqa_data_extracted_test_converted_numeric.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    # 执行转换
    convert_dataset(input_file, output_file)
    
    # 验证转换结果
    print("\n验证转换结果...")
    with open(output_file, 'r', encoding='utf-8') as f:
        converted_data = json.load(f)
    
    # 显示转换前后的示例
    print("转换示例:")
    if converted_data:
        sample = converted_data[0]
        if 'messages' in sample and sample['messages']:
            for msg in sample['messages'][:2]:  # 显示前两条消息
                print(f"角色: {msg.get('role', 'unknown')}")
                content = msg.get('content', '')
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"内容: {content}")
                print()

if __name__ == "__main__":
    main()
