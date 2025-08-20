#!/usr/bin/env python3
"""
测试修复后的数值处理器
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
sys.path.append('/data1/wangzhiye/1a1a11/original')

from numeric_qwen2_5_vl import NumericQwen2_5_VLProcessor

def test_fixed_processor():
    print("=== 测试修复后的数值处理器 ===")
    
    # 创建处理器
    processor = NumericQwen2_5_VLProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
    
    # 测试文本
    test_text = "产品评分为<num><8.5>分，价格是<num><299.99>元。"
    print(f"原始文本: {test_text}")
    
    # 处理文本
    result = processor(text=test_text, return_tensors="pt")
    
    # 检查结果
    print(f"处理后的token: {processor.tokenizer.decode(result['input_ids'][0])}")
    print(f"提取的数值: {result.get('numeric_values', '未找到')}")
    print(f"数值位置: {result.get('numeric_positions', '未找到')}")
    
    # 检查token ID
    print(f"\n=== Token ID 信息 ===")
    print(f"<num> token ID: {processor.num_token_id}")
    print(f"<num_pad> token ID: {processor.num_pad_token_id}")
    
    # 检查处理后的文本中是否包含正确的token
    input_ids = result['input_ids'][0]
    num_count = (input_ids == processor.num_token_id).sum().item()
    num_pad_count = (input_ids == processor.num_pad_token_id).sum().item()
    
    print(f"输入中的<num>数量: {num_count}")
    print(f"输入中的<num_pad>数量: {num_pad_count}")
    
    return processor, result

if __name__ == "__main__":
    processor, result = test_fixed_processor()
    print("\n测试完成！")
