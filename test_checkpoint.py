import os
import torch
from transformers import AutoTokenizer, AutoProcessor
from model.modeling_numeric_qwen2_5_vl import NumericQwen2_5_VLForConditionalGeneration
from training_config import get_training_config

def load_model_from_checkpoint(checkpoint_path, device="cuda"):
    """
    从指定的checkpoint路径加载模型
    
    Args:
        checkpoint_path: checkpoint目录路径，如 "/data1/wangzhiye/1a1a11/original/output/checkpoint-150"
        device: 设备类型，默认为 "cuda"
    
    Returns:
        model: 加载的模型
        processor: 处理器
        tokenizer: 分词器
    """
    
    print(f"🔄 正在从 {checkpoint_path} 加载模型...")
    
    # 检查checkpoint路径是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint路径不存在: {checkpoint_path}")
    
    # 检查必要的文件是否存在
    required_files = ["config.json", "model.safetensors", "trainer_state.json"]
    for file in required_files:
        file_path = os.path.join(checkpoint_path, file)
        if not os.path.exists(file_path):
            print(f"⚠️  警告: {file} 不存在于 {checkpoint_path}")
    
    try:
        # 1. 加载配置
        print("📝 加载模型配置...")
        
        # 2. 加载分词器和处理器 (使用原始模型路径)
        base_model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        print(f"🔤 加载分词器从: {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        
        print(f"🖼️  加载处理器从: {base_model_path}")
        processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        
        # 3. 加载模型 (从checkpoint)
        print(f"🤖 加载模型从: {checkpoint_path}")
        model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 4. 确保模型在正确的设备上
        if device != "auto":
            model = model.to(device)
        
        print(f"✅ 模型成功加载到设备: {model.device}")
        print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        
        # 5. 加载训练状态信息 (可选)
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            import json
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)
            print(f"📈 训练状态 - 全局步数: {trainer_state.get('global_step', 'Unknown')}")
            print(f"📈 训练状态 - 轮次: {trainer_state.get('epoch', 'Unknown')}")
        
        return model, processor, tokenizer
        
    except Exception as e:
        print(f"❌ 加载模型时出错: {str(e)}")
        raise e

def continue_training_from_checkpoint(checkpoint_path):
    """
    从checkpoint继续训练
    
    Args:
        checkpoint_path: checkpoint目录路径
    """
    
    print(f"🚀 准备从 {checkpoint_path} 继续训练...")
    
    # 获取训练配置
    training_args = get_training_config(
        output_dir="/data1/wangzhiye/1a1a11/original/output",
        data_path="/data1/wangzhiye/LLaMA-Factory/data/3bddx_train_converted1_train.json",
        val_data_path="/data1/wangzhiye/LLaMA-Factory/data/3bddx_train_converted1_val.json",
        test_data_path="/data1/wangzhiye/LLaMA-Factory/data/1bddx_test_converted.json",
        image_folder="/data1/wangzhiye/LLaMA-Factory/data"
    )
    
    # 加载模型
    model, processor, tokenizer = load_model_from_checkpoint(checkpoint_path)
    
    # 这里可以继续设置训练器和继续训练
    print("✅ 模型加载完成，可以继续训练")
    
    return model, processor, tokenizer, training_args

def test_model_inference(checkpoint_path, test_text="这是一个测试"):
    """
    测试从checkpoint加载的模型推理能力
    
    Args:
        checkpoint_path: checkpoint路径
        test_text: 测试文本
    """
    
    print(f"🧪 测试模型推理能力...")
    
    # 加载模型
    model, processor, tokenizer = load_model_from_checkpoint(checkpoint_path)
    
    # 设置为评估模式
    model.eval()
    
    try:
        # 准备输入
        inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
        
        # 推理
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"📝 输入: {test_text}")
        print(f"📤 输出: {response}")
        
        return response
        
    except Exception as e:
        print(f"❌ 推理测试失败: {str(e)}")
        return None

if __name__ == "__main__":
    # 使用示例
    
    # 方案1: 直接加载模型
    checkpoint_path = "/data1/wangzhiye/1a1a11/original/output/checkpoint-150"
    
    # 检查实际存在的checkpoint
    output_dir = "/data1/wangzhiye/1a1a11/original/output"
    available_checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    print(f"📂 可用的checkpoints: {available_checkpoints}")
    
    if available_checkpoints:
        # 使用最新的checkpoint
        latest_checkpoint = max(available_checkpoints, key=lambda x: int(x.split("-")[1]))
        checkpoint_path = os.path.join(output_dir, latest_checkpoint)
        print(f"🎯 使用最新的checkpoint: {checkpoint_path}")
        
        try:
            # 加载模型
            model, processor, tokenizer = load_model_from_checkpoint(checkpoint_path)
            
            # 测试推理
            test_model_inference(checkpoint_path, "图片中有多少个数字?")
            
            # 如果要继续训练，取消注释下面这行
            # continue_training_from_checkpoint(checkpoint_path)
            
        except Exception as e:
            print(f"❌ 执行失败: {str(e)}")
    else:
        print("❌ 没有找到可用的checkpoint")