#!/usr/bin/env python3
"""
数值增强Qwen2.5-VL模型推理脚本

支持文本、图像输入，能够预测数值和生成文本
"""

import os
import sys
import torch
from PIL import Image
from typing import List, Dict, Union, Optional
import re

# 添加当前目录到path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from numeric_qwen2_5_vl import (
    NumericQwen2_5_VLConfig,
    NumericQwen2_5_VLProcessor,
    NumericQwen2_5_VLForConditionalGeneration
)


class NumericQwen2_5_VLInference:
    """
    数值增强Qwen2.5-VL推理类
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        初始化推理引擎
        
        Args:
            model_path: 模型路径
            device: 设备类型
        """
        self.device = device
        self.model_path = model_path
        
        print(f"正在加载模型: {model_path}")
        
        # 加载模型和处理器
        self.processor = NumericQwen2_5_VLProcessor.from_pretrained(model_path)
        self.model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        
        self.model.eval()
        print("模型加载完成")
        
        # 数值token提取模式
        self.numeric_pattern = re.compile(r'<num><([+-]?\d*\.?\d+)>')
    
    def generate_response(
        self,
        text: str,
        image: Optional[Union[str, Image.Image]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9
    ) -> Dict:
        """
        生成回复
        
        Args:
            text: 输入文本
            image: 输入图像（路径或PIL图像）
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            do_sample: 是否采样
            top_p: top-p采样参数
            
        Returns:
            包含生成文本和数值预测的字典
        """
        
        # 处理图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # 构建对话格式
        if image is not None:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text}
                    ]
                }
            ]
        else:
            conversation = [
                {
                    "role": "user", 
                    "content": [{"type": "text", "text": text}]
                }
            ]
        
        # 处理输入
        inputs = self.processor(
            text=conversation,
            images=image,
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # 解码生成的文本
        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0][input_length:]
        generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 提取数值预测
        numeric_predictions = self.extract_numeric_predictions(generated_text)
        
        return {
            "text": generated_text,
            "numeric_predictions": numeric_predictions,
            "full_conversation": text + "\n\nAssistant: " + generated_text
        }
    
    def extract_numeric_predictions(self, text: str) -> List[Dict]:
        """
        从生成文本中提取数值预测
        
        Args:
            text: 生成的文本
            
        Returns:
            数值预测列表
        """
        predictions = []
        
        for match in self.numeric_pattern.finditer(text):
            try:
                value = float(match.group(1))
                predictions.append({
                    "value": value,
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end()
                })
            except ValueError:
                continue
        
        return predictions
    
    def batch_inference(
        self,
        texts: List[str],
        images: Optional[List[Union[str, Image.Image]]] = None,
        **kwargs
    ) -> List[Dict]:
        """
        批量推理
        
        Args:
            texts: 文本列表
            images: 图像列表
            **kwargs: 生成参数
            
        Returns:
            结果列表
        """
        results = []
        
        if images is None:
            images = [None] * len(texts)
        
        for text, image in zip(texts, images):
            result = self.generate_response(text, image, **kwargs)
            results.append(result)
        
        return results


def main():
    """
    推理示例
    """
    
    # 模型路径
    model_path = "/data1/wangzhiye/1a1a11/original/output"
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        print("请先训练模型或指定正确的模型路径")
        return
    
    # 创建推理引擎
    inference = NumericQwen2_5_VLInference(model_path)
    
    # 示例1: 纯文本推理
    print("=== 示例1: 纯文本推理 ===")
    text1 = "计算 3.14 * 2.5 的结果，并用数值格式表示"
    result1 = inference.generate_response(text1)
    print(f"输入: {text1}")
    print(f"输出: {result1['text']}")
    print(f"数值预测: {result1['numeric_predictions']}")
    print()
    
    # 示例2: 图像+文本推理（如果有图像）
    print("=== 示例2: 图像分析 ===")
    image_path = "/data1/wangzhiye/data/images/sample.jpg"  # 请替换为实际图像路径
    
    if os.path.exists(image_path):
        text2 = "分析这个图像中的数值信息，并提取关键数据"
        result2 = inference.generate_response(text2, image_path)
        print(f"输入: {text2}")
        print(f"输出: {result2['text']}")
        print(f"数值预测: {result2['numeric_predictions']}")
    else:
        print(f"示例图像不存在: {image_path}")
        print("请提供有效的图像路径进行测试")
    print()
    
    # 示例3: 批量推理
    print("=== 示例3: 批量推理 ===")
    texts = [
        "π的近似值是多少？",
        "黄金比例的数值是什么？",
        "e的值大约是多少？"
    ]
    
    results = inference.batch_inference(texts, temperature=0.1)
    
    for i, (text, result) in enumerate(zip(texts, results)):
        print(f"问题{i+1}: {text}")
        print(f"回答: {result['text']}")
        print(f"数值: {result['numeric_predictions']}")
        print()


if __name__ == "__main__":
    main()
