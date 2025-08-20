#!/usr/bin/env python3
"""
综合测试修复后的训练代码
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import torch
sys.path.append('/data1/wangzhiye/1a1a11/original')

from numeric_qwen2_5_vl import (
    NumericQwen2_5_VLConfig,
    NumericQwen2_5_VLProcessor,
    NumericQwen2_5_VLForConditionalGeneration
)
from training_config import create_model_and_processor

def test_complete_training_pipeline():
    print("=== 综合测试修复后的训练管道 ===")
    
    # 1. 测试处理器
    print("\n1. 测试处理器...")
    processor = NumericQwen2_5_VLProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
    
    test_text = "产品评分为<num><8.5>分，价格是<num><299.99>元。"
    result = processor(text=test_text, return_tensors="pt")
    
    print(f"原始文本: {test_text}")
    print(f"处理后的token: {processor.tokenizer.decode(result['input_ids'][0])}")
    print(f"提取的数值: {result.get('numeric_values', '未找到')}")
    print(f"<num> token ID: {processor.num_token_id}")
    print(f"<num_pad> token ID: {processor.num_pad_token_id}")
    
    # 2. 测试模型创建
    print("\n2. 测试模型和处理器创建...")
    try:
        model, proc = create_model_and_processor('Qwen/Qwen2.5-VL-3B-Instruct')
        print("✅ 模型和处理器创建成功")
        print(f"模型配置中的 num_token_id: {getattr(model.config, 'num_token_id', 'None')}")
        print(f"模型配置中的 num_pad_token_id: {getattr(model.config, 'num_pad_token_id', 'None')}")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False
    
    # 3. 测试前向传播
    print("\n3. 测试前向传播...")
    try:
        model.eval()
        with torch.no_grad():
            # 准备输入
            inputs = processor(text=test_text, return_tensors="pt")
            
            # 前向传播（显式设置return_dict=True）
            outputs = model(**inputs, return_dict=True)
            
            print(f"✅ 前向传播成功")
            print(f"Logits shape: {outputs.logits.shape}")
            if hasattr(outputs, 'predicted_floats') and outputs.predicted_floats is not None:
                print(f"Predicted floats shape: {outputs.predicted_floats.shape}")
            else:
                print("⚠️  predicted_floats 为 None 或不存在")
            
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 测试损失计算
    print("\n4. 测试损失计算...")
    try:
        # 创建假标签
        input_ids = inputs['input_ids']
        labels = input_ids.clone()
        
        # 添加数值信息
        inputs['labels'] = labels
        inputs['numeric_values'] = result['numeric_values']
        inputs['numeric_positions'] = result['numeric_positions']
        
        # 计算损失
        outputs = model(**inputs, return_dict=True)
        
        if outputs.loss is not None:
            print(f"✅ 损失计算成功: {outputs.loss.item():.4f}")
        else:
            print("⚠️  没有返回损失值")
            
    except Exception as e:
        print(f"❌ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== 所有测试通过！可以开始训练 ===")
    return True

if __name__ == "__main__":
    success = test_complete_training_pipeline()
    if success:
        print("\n🎉 训练代码已准备就绪！")
    else:
        print("\n⚠️  还有问题需要解决")
