#!/usr/bin/env python3
"""
基于GitCode博客文章解决方案的图像Token匹配修复测试
参考: https://blog.gitcode.com/d41d68b8e2ccdd03c0c59a4ca19a517b.html
"""
import os
import torch
from PIL import Image
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加数值增强模块路径
import sys
sys.path.append('/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250')

from numeric_qwen2_5_vl import NumericQwen2_5_VLForConditionalGeneration, NumericQwen2_5_VLProcessor

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
        print(f"❌ 模型加载失败: {e}")
        raise

def create_optimal_image(target_resolution=(224, 224)):
    """
    创建优化尺寸的测试图像
    根据GitCode文章建议，限制图像尺寸以避免Token数量过多
    """
    print(f"🎨 创建优化分辨率图像: {target_resolution}")
    
    # 创建简单的测试图像 - 小尺寸避免Token过多
    image = Image.new('RGB', target_resolution, color='lightblue')
    
    # 添加一些简单的视觉元素
    import numpy as np
    img_array = np.array(image)
    
    # 添加一些颜色条纹
    height, width = img_array.shape[:2]
    for i in range(0, height, 20):
        img_array[i:i+10, :] = [255, 200, 100]  # 橙色条纹
    
    for j in range(0, width, 30):
        img_array[:, j:j+15] = [100, 255, 100]  # 绿色条纹
    
    return Image.fromarray(img_array)

def calculate_image_tokens(image_size, patch_size=28):
    """
    计算图像Token数量
    根据GitCode文章: Token数量 = max_pixels/(28*28)
    """
    width, height = image_size
    total_pixels = width * height
    tokens = total_pixels // (patch_size * patch_size)
    
    print(f"📊 图像尺寸: {width}x{height}")
    print(f"📊 总像素数: {total_pixels}")
    print(f"📊 预计Token数量: {tokens}")
    
    return tokens

def test_image_inference_with_token_control(model, processor, max_prompt_length=2048):
    """
    带Token控制的图像推理测试
    根据GitCode文章建议调整参数避免Token不匹配
    """
    print("\n🔧 开始图像Token控制测试...")
    
    # 方案1: 使用小尺寸图像
    print("\n📋 方案1: 小尺寸图像测试")
    small_image = create_optimal_image((224, 224))
    small_tokens = calculate_image_tokens(small_image.size)
    
    if small_tokens < max_prompt_length // 4:  # 保留3/4空间给文本
        print("✅ 小图像Token数量合理，进行推理测试...")
        test_vqa_inference(model, processor, small_image, "这张图片中有什么颜色?")
    else:
        print("⚠️ 小图像Token数量仍然过多")
    
    # 方案2: 更小尺寸图像
    print("\n📋 方案2: 超小尺寸图像测试")
    tiny_image = create_optimal_image((112, 112))
    tiny_tokens = calculate_image_tokens(tiny_image.size)
    
    if tiny_tokens < max_prompt_length // 8:  # 保留7/8空间给文本
        print("✅ 超小图像Token数量合理，进行推理测试...")
        test_vqa_inference(model, processor, tiny_image, "描述这张图片中的图案。")
    else:
        print("⚠️ 超小图像Token数量仍然过多")

def test_vqa_inference(model, processor, image, question):
    """执行视觉问答推理"""
    try:
        print(f"🤔 问题: {question}")
        
        # 使用最简单有效的格式 - 基于check_tokens.py的发现
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|image|>{question}<|im_end|>\n<|im_start|>assistant\n"
        
        print(f"📝 构建prompt成功，文本长度: {len(prompt)}")
        
        # 处理输入 - 限制参数以避免Token过多
        image_inputs = processor(
            text=[prompt], 
            images=[image], 
            return_tensors="pt",
            max_length=1024,  # 限制最大长度
            truncation=True   # 允许截断
        )
        
        print("📦 输入处理成功")
        print(f"📊 输入形状: {image_inputs.input_ids.shape}")
        
        # 检查是否有图像Token
        if hasattr(image_inputs, 'pixel_values') and image_inputs.pixel_values is not None:
            print(f"🖼️ 图像特征形状: {image_inputs.pixel_values.shape}")
        
        # 解码并检查Token
        decoded_text = processor.tokenizer.decode(image_inputs.input_ids[0], skip_special_tokens=False)
        image_token_count = decoded_text.count('<|image_pad|>')
        print(f"🔢 图像Token数量: {image_token_count}")
        
        # 移动到设备 - 安全处理所有类型的数据
        device_inputs = {}
        for k, v in image_inputs.items():
            if hasattr(v, 'to'):  # 检查是否有to方法
                device_inputs[k] = v.to(model.device)
            else:
                device_inputs[k] = v  # 保持原样
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **device_inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # 解码回答
        generated_ids = outputs[0][device_inputs['input_ids'].shape[1]:]
        response = processor.decode(generated_ids, skip_special_tokens=True)
        
        print(f"🎯 回答: {response}")
        return response
        
    except Exception as e:
        print(f"❌ VQA推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_processor_configuration(processor):
    """测试处理器配置参数"""
    print("\n🔍 检查处理器配置...")
    
    # 检查图像处理器配置
    if hasattr(processor, 'image_processor'):
        img_proc = processor.image_processor
        print("🖼️ 图像处理器配置:")
        
        # 检查关键参数
        if hasattr(img_proc, 'size'):
            print(f"   - 默认尺寸: {img_proc.size}")
        if hasattr(img_proc, 'max_pixels'):
            print(f"   - 最大像素: {img_proc.max_pixels}")
        if hasattr(img_proc, 'min_pixels'):
            print(f"   - 最小像素: {img_proc.min_pixels}")
    
    # 检查分词器配置
    if hasattr(processor, 'tokenizer'):
        tokenizer = processor.tokenizer
        print("📝 分词器配置:")
        
        if hasattr(tokenizer, 'model_max_length'):
            print(f"   - 最大长度: {tokenizer.model_max_length}")
        if hasattr(tokenizer, 'pad_token_id'):
            print(f"   - 填充Token ID: {tokenizer.pad_token_id}")

def main():
    """主测试函数"""
    print("🚀 开始图像Token匹配修复测试")
    print("=" * 60)
    
    # 模型路径 - 使用output目录中的checkpoint-4250
    checkpoint_path = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"
    
    try:
        # 加载模型
        model, processor = load_checkpoint_model(checkpoint_path)
        
        # 检查处理器配置
        test_processor_configuration(processor)
        
        # 进行Token控制的图像推理测试
        test_image_inference_with_token_control(model, processor)
        
        print("\n✅ 图像Token匹配测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
