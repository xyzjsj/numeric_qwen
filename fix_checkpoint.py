#!/usr/bin/env python3
"""
修复检查点文件
为现有检查点添加缺失的 preprocessor_config.json
"""
import json
import os
import sys

def fix_checkpoint(checkpoint_path):
    """修复检查点，添加缺失的配置文件"""
    print(f"🔧 修复检查点: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点目录不存在: {checkpoint_path}")
        return False
    
    # 检查当前文件列表
    files = os.listdir(checkpoint_path)
    print(f"📁 当前文件: {files}")
    
    # 创建 preprocessor_config.json
    preprocessor_config_path = os.path.join(checkpoint_path, "preprocessor_config.json")
    
    if os.path.exists(preprocessor_config_path):
        print(f"⚠️ preprocessor_config.json 已存在，将更新")
    
    preprocessor_config = {
        "processor_class": "NumericQwen2_5_VLProcessor",
        "auto_map": {
            "AutoProcessor": "numeric_qwen2_5_vl.NumericQwen2_5_VLProcessor"
        },
        "image_processor": {
            "do_convert_rgb": True,
            "do_normalize": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "resample": 3,
            "size": {"shortest_edge": 336}
        },
        "image_processor_type": "Qwen2VLImageProcessor",
        "tokenizer": {
            "padding_side": "left",
            "truncation_side": "left",
            "model_max_length": 32768,
            "tokenizer_class": "Qwen2Tokenizer"
        },
        "num_token_id": 151665,
        "num_pad_token_id": 151666,
        "numeric_tokens": ["<num>", "<num_pad>"]
    }
    
    print(f"📝 创建/更新 preprocessor_config.json...")
    with open(preprocessor_config_path, 'w', encoding='utf-8') as f:
        json.dump(preprocessor_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ preprocessor_config.json 已创建/更新")
    
    # 验证文件
    if os.path.exists(preprocessor_config_path):
        with open(preprocessor_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"📋 验证配置文件:")
        print(f"- processor_class: {config.get('processor_class')}")
        print(f"- image_processor_type: {config.get('image_processor_type')}")
        print(f"- num_token_id: {config.get('num_token_id')}")
        print(f"- num_pad_token_id: {config.get('num_pad_token_id')}")
        return True
    
    return False

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        checkpoint_name = sys.argv[1]
        checkpoint_path = f"/data1/wangzhiye/1a1a11/original/output/{checkpoint_name}"
    else:
        checkpoint_path = "/data1/wangzhiye/1a1a11/original/output/checkpoint-4250"
    
    print(f"目标检查点: {checkpoint_path}")
    
    success = fix_checkpoint(checkpoint_path)
    
    if success:
        print(f"\n🎉 检查点修复成功！现在可以正常加载处理器了。")
        print(f"修复的检查点: {checkpoint_path}")
    else:
        print(f"\n❌ 检查点修复失败！")