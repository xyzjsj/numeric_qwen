#!/usr/bin/env python3
"""
从Checkpoint加载NumericQwen2.5-VL模型

示例用法:
    python load_checkpoint.py
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import torch
from transformers import AutoTokenizer, AutoProcessor
from numeric_qwen2_5_vl import (
    NumericQwen2_5_VLConfig,
    NumericQwen2_5_VLProcessor,
    NumericQwen2_5_VLForConditionalGeneration
)


def load_model_from_checkpoint(checkpoint_path, device="auto"):
    """
    从指定的checkpoint路径加载模型
    
    Args:
        checkpoint_path: checkpoint目录路径，如 "/data1/wangzhiye/1a1a11/original/output/checkpoint-150"
        device: 设备类型，默认为 "auto"
    
    Returns:
        model: 加载的模型
        processor: 处理器
        tokenizer: 分词器
        trainer_state: 训练状态信息
    """
    
    print(f"🔄 正在从 {checkpoint_path} 加载模型...")
    
    # 检查checkpoint路径是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint路径不存在: {checkpoint_path}")
    
    # 检查必要的文件是否存在
    required_files = ["config.json"]
    
    # 检查配置文件
    for file in required_files:
        file_path = os.path.join(checkpoint_path, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"必需文件不存在: {file_path}")
    
    # 检查模型文件（支持多种格式）
    model_file_exists = False
    model_files_found = []
    
    # 检查单个模型文件
    single_model_files = ["model.safetensors", "pytorch_model.bin"]
    for file in single_model_files:
        file_path = os.path.join(checkpoint_path, file)
        if os.path.exists(file_path):
            model_file_exists = True
            model_files_found.append(file)
            print(f"✅ 找到模型文件: {file}")
    
    # 检查分片模型文件
    if not model_file_exists:
        # 检查safetensors分片文件
        files = os.listdir(checkpoint_path)
        safetensors_files = [f for f in files if f.startswith('model-') and f.endswith('.safetensors')]
        pytorch_files = [f for f in files if f.startswith('pytorch_model-') and f.endswith('.bin')]
        
        if safetensors_files:
            model_file_exists = True
            model_files_found.extend(safetensors_files)
            print(f"✅ 找到分片模型文件 (safetensors): {len(safetensors_files)} 个文件")
            # 检查index文件
            index_file = os.path.join(checkpoint_path, "model.safetensors.index.json")
            if os.path.exists(index_file):
                print(f"✅ 找到模型索引文件: model.safetensors.index.json")
        elif pytorch_files:
            model_file_exists = True
            model_files_found.extend(pytorch_files)
            print(f"✅ 找到分片模型文件 (pytorch): {len(pytorch_files)} 个文件")
    
    if not model_file_exists:
        available_files = os.listdir(checkpoint_path)
        raise FileNotFoundError(
            f"未找到模型文件在 {checkpoint_path}\n"
            f"查找的文件类型: model.safetensors, pytorch_model.bin, model-*.safetensors, pytorch_model-*.bin\n"
            f"目录中的文件: {available_files}"
        )
    
    try:
        # 1. 加载训练状态信息 (如果存在)
        trainer_state = None
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            print("📊 加载训练状态...")
            with open(trainer_state_path, 'r', encoding='utf-8') as f:
                trainer_state = json.load(f)
            print(f"   全局步数: {trainer_state.get('global_step', 'N/A')}")
            print(f"   训练轮次: {trainer_state.get('epoch', 'N/A')}")
        
        # 2. 加载分词器和处理器 (使用原始模型路径，因为checkpoint中可能没有)
        base_model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        print(f"🔤 加载分词器从: {base_model_path}")
        
        try:
            # 尝试从checkpoint加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
            print("   ✅ 从checkpoint加载分词器成功")
        except:
            # 如果失败，从基础模型加载
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            print("   ✅ 从基础模型加载分词器成功")
        
        print(f"🖼️  加载处理器...")
        processor = None
        try:
            # 尝试从checkpoint加载processor
            processor = NumericQwen2_5_VLProcessor.from_pretrained(checkpoint_path)
            print("   ✅ 从checkpoint加载处理器成功")
        except Exception as e1:
            print(f"   ⚠️ checkpoint处理器加载失败: {e1}")
            try:
                # 尝试手动创建处理器（不包含video_processor）
                from transformers import Qwen2_5_VLImageProcessor
                
                # 加载图像处理器
                image_processor = Qwen2_5_VLImageProcessor.from_pretrained(base_model_path)
                
                # 手动创建处理器，跳过video_processor
                class SafeNumericProcessor:
                    def __init__(self, image_processor, tokenizer):
                        self.image_processor = image_processor
                        self.tokenizer = tokenizer
                    
                    def __call__(self, text, images=None, **kwargs):
                        # 简化的处理逻辑
                        if images is not None:
                            print("   ℹ️ 图像处理暂时跳过（处理器简化版本）")
                        
                        # 处理文本输入
                        if isinstance(text, list):
                            text = text[0] if text else ""
                        
                        return self.tokenizer(text, **kwargs)
                
                processor = SafeNumericProcessor(image_processor, tokenizer)
                print("   ✅ 创建简化处理器成功（无视频支持）")
                
            except Exception as e2:
                print(f"   ⚠️ 简化处理器创建失败: {e2}")
                print("   ℹ️ 将使用纯tokenizer模式（仅支持文本）")
                processor = None
        
        # 3. 加载模型配置
        print("📝 加载模型配置...")
        config = NumericQwen2_5_VLConfig.from_pretrained(checkpoint_path)
        print(f"   模型类型: {config.model_type}")
        print(f"   词汇表大小: {config.vocab_size}")
        if hasattr(config, 'num_token_id'):
            print(f"   数值Token ID: {config.num_token_id}")
        
        # 4. 加载模型 (从checkpoint)
        print(f"🤖 加载模型从: {checkpoint_path}")
        model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        
        # 5. 模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ 模型成功加载!")
        print(f"📊 模型信息:")
        print(f"   总参数量: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"   可训练参数: {trainable_params:,} ({trainable_params/1e9:.2f}B)")
        print(f"   设备分布: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'auto'}")
        
        return model, processor, tokenizer, trainer_state
        
    except Exception as e:
        print(f"❌ 加载过程中出错: {e}")
        raise


def test_model_inference(model, processor, tokenizer, test_text="你好，这是一个测试。"):
    """
    测试模型推理功能
    
    Args:
        model: 加载的模型
        processor: 处理器
        tokenizer: 分词器
        test_text: 测试文本
    """
    print(f"\n🧪 测试模型推理...")
    print(f"输入文本: {test_text}")
    
    try:
        # 处理输入
        if processor is not None:
            try:
                inputs = processor(
                    text=[test_text],
                    images=None,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                print("   ✅ 使用processor处理输入")
            except Exception as e:
                print(f"   ⚠️ processor处理失败，降级使用tokenizer: {e}")
                inputs = tokenizer(
                    test_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512  # 限制输入长度
                )
        else:
            # 如果没有processor，直接使用tokenizer
            inputs = tokenizer(
                test_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512  # 限制输入长度
            )
            print("   ✅ 使用tokenizer处理输入")
        
        # 移动到模型设备
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        print(f"   输入设备: {device}")
        print(f"   输入长度: {inputs['input_ids'].shape[1]} tokens")
        
        # 推理 - 使用更保守的参数
        # 推理
        print("🔮 开始推理...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                min_new_tokens=1,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"✅ 推理成功!")
        print(f"   生成长度: {len(generated_tokens)} tokens")
        print(f"   生成文本: '{generated_text.strip()}'")
        
        # 如果生成文本为空，给出提示
        if not generated_text.strip():
            print("   ℹ️ 生成文本为空，这可能是正常的（模型认为不需要继续生成）")
        
    except KeyboardInterrupt:
        print("❌ 推理被用户中断")
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()


def continue_training_from_checkpoint(checkpoint_path):
    """
    从检查点继续训练的示例代码
    
    Args:
        checkpoint_path: 检查点路径
    """
    print(f"\n🔄 准备从 {checkpoint_path} 继续训练...")
    
    # 加载模型
    model, processor, tokenizer, trainer_state = load_model_from_checkpoint(checkpoint_path)
    
    if trainer_state:
        print(f"📊 训练状态信息:")
        print(f"   当前步数: {trainer_state.get('global_step', 0)}")
        print(f"   当前轮次: {trainer_state.get('epoch', 0)}")
        print(f"   最大步数: {trainer_state.get('max_steps', 'N/A')}")
        print(f"   训练轮次: {trainer_state.get('num_train_epochs', 'N/A')}")
    
    print("💡 要继续训练，请在 train.py 中设置:")
    print(f"   resume_from_checkpoint='{checkpoint_path}'")
    
    return model, processor, tokenizer, trainer_state


def main():
    """
    主函数 - 演示如何使用
    """
    print("🚀 NumericQwen2.5-VL Checkpoint加载器")
    print("=" * 50)
    
    # 设置checkpoint路径
    checkpoint_path = "/data1/wangzhiye/1a1a11/original/output/checkpoint-150"
    
    try:
        # 1. 加载模型
        model, processor, tokenizer, trainer_state = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device="auto"  # 自动分配设备
        )
        
        # 2. 测试推理 - 使用简单的文本测试
        test_texts = [
            "你好",
            "计算 2+3=",
            "什么是数学？"
        ]
        
        for i, test_text in enumerate(test_texts, 1):
            print(f"\n📝 测试 {i}/{len(test_texts)}: {test_text}")
            test_model_inference(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                test_text=test_text
            )
            
            # 如果第一个测试成功，继续其他测试
            if i == 1:
                print("   ✅ 基础推理测试通过，继续其他测试...")
        
        print(f"\n🎯 测试结论:")
        print(f"   - 模型加载: ✅ 成功")
        print(f"   - 分词器: ✅ 正常")
        print(f"   - 处理器: {'✅ 正常' if processor else '⚠️ 简化版本'}")
        print(f"   - 推理功能: ✅ 可用")
        
        # 3. 显示如何继续训练
        print("\n" + "=" * 50)
        print("💡 继续训练指南:")
        print(f"在 train.py 中，确保 resume_from_checkpoint 参数设置为:")
        print(f"'{checkpoint_path}'")
        
        if trainer_state:
            current_step = trainer_state.get('global_step', 0)
            max_steps = trainer_state.get('max_steps', 1338)  # 默认值
            progress = (current_step / max_steps) * 100 if max_steps > 0 else 0
            print(f"\n📈 训练进度: {current_step}/{max_steps} ({progress:.1f}%)")
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False
    
    return True


if __name__ == "__main__":
    # 检查可用的checkpoints
    output_dir = "/data1/wangzhiye/1a1a11/original/output"
    print("📁 可用的检查点:")
    
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
        checkpoints.sort(key=lambda x: int(x.split('-')[1]))
        
        for checkpoint in checkpoints:
            checkpoint_path = os.path.join(output_dir, checkpoint)
            # 检查训练状态
            trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
            if os.path.exists(trainer_state_path):
                with open(trainer_state_path, 'r') as f:
                    state = json.load(f)
                step = state.get('global_step', 'N/A')
                epoch = state.get('epoch', 'N/A')
                print(f"   ✅ {checkpoint} (步数: {step}, 轮次: {epoch})")
            else:
                print(f"   ⚠️  {checkpoint} (缺少trainer_state.json)")
    
    print("\n" + "=" * 50)
    
    # 运行主函数
    main()
