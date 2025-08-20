#!/usr/bin/env python3
"""
测试修复后的配置是否正确
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from training_config import create_model_and_processor

def test_vocab_size_fix():
    """测试vocab_size修复"""
    print("=== 测试vocab_size修复 ===")
    
    try:
        # 创建模型和处理器
        model, processor = create_model_and_processor(
            model_path="Qwen/Qwen2.5-VL-3B-Instruct"
        )
        
        print(f"Tokenizer vocab size: {len(processor.tokenizer)}")
        print(f"Model config vocab_size: {model.config.vocab_size}")
        print(f"Model num_token_id: {model.config.num_token_id}")
        
        # 检查<num>token是否在词汇表中
        num_token = '<num>'
        if num_token in processor.tokenizer.get_vocab():
            token_id = processor.tokenizer.convert_tokens_to_ids(num_token)
            print(f"<num> token ID: {token_id}")
        else:
            print("<num> token not found in vocabulary")
        
        # 测试logits形状
        # 创建一个简单的输入
        test_text = "Hello world"
        inputs = processor(text=[test_text], images=None, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            # 确保返回dict格式
            outputs = model(**inputs, return_dict=True)
            print(f"Output type: {type(outputs)}")
            
            if hasattr(outputs, 'logits'):
                print(f"Logits shape: {outputs.logits.shape}")
                print(f"Expected vocab dim: {model.config.vocab_size}")
                
                # 检查是否匹配
                if outputs.logits.shape[-1] == model.config.vocab_size:
                    print("✅ Logits shape matches config vocab_size")
                else:
                    print("❌ Logits shape does not match config vocab_size")
            else:
                print("❌ Outputs does not have logits attribute")
                print(f"Available attributes: {dir(outputs)}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vocab_size_fix()
    if success:
        print("\n✅ 修复测试通过！")
    else:
        print("\n❌ 修复测试失败！")
