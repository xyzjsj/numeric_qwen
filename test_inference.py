#!/usr/bin/env python3
"""
CIDEr评估脚本 - NumericQwen2.5-VL模型
简单版本，专注解决token计数问题
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, Qwen2VLProcessor
import re
from collections import Counter

# 添加当前目录到路径，以便导入本地模块
sys.path.append('/data1/wangzhiye/1a1a11/original')

# 导入自定义模型组件
from numeric_qwen2_5_vl import (
    NumericQwen2_5_VLConfig, 
    NumericQwen2_5_VLForConditionalGeneration, 
    NumericQwen2_5_VLProcessor
)


class CIDErScorer:
    """
    CIDEr (Consensus-based Image Description Evaluation) 评分器
    """
    
    def __init__(self, n: int = 4, sigma: float = 6.0):
        """
        初始化CIDEr评分器
        
        Args:
            n: n-gram的最大长度
            sigma: 高斯权重的标准差
        """
        self.n = n
        self.sigma = sigma
        self.document_frequency = {}
        
    def compute_cider(self, candidate: str, refs: List[str]) -> float:
        """
        计算单个候选答案的CIDEr分数
        
        Args:
            candidate: 候选答案
            refs: 参考答案列表
            
        Returns:
            CIDEr分数
        """
        score = 0.0
        
        # 对每个n-gram级别计算分数
        for n in range(1, self.n + 1):
            # 获取候选答案的n-gram
            candidate_ngrams = self._get_ngrams(candidate, n)
            candidate_counts = Counter(candidate_ngrams)
            
            # 计算所有参考答案的n-gram
            ref_ngrams_list = []
            for ref in refs:
                ref_ngrams = self._get_ngrams(ref, n)
                ref_ngrams_list.append(Counter(ref_ngrams))
            
            # 计算TF-IDF相似度
            similarity = self._compute_tfidf_similarity(
                candidate_counts, ref_ngrams_list, n
            )
            
            # 累加各级n-gram的分数
            score += similarity
        
        # 返回平均分数
        return score / self.n
    
    def _compute_tfidf_similarity(self, candidate_counts: Counter, 
                                  ref_counts_list: List[Counter], n: int) -> float:
        """
        计算TF-IDF相似度
        """
        # 计算候选答案的TF-IDF向量
        candidate_tfidf = self._compute_tfidf_vector(candidate_counts, n)
        
        # 计算每个参考答案的TF-IDF向量
        ref_tfidf_list = []
        for ref_counts in ref_counts_list:
            ref_tfidf = self._compute_tfidf_vector(ref_counts, n)
            ref_tfidf_list.append(ref_tfidf)
        
        # 计算余弦相似度
        similarities = []
        for ref_tfidf in ref_tfidf_list:
            sim = self._cosine_similarity(candidate_tfidf, ref_tfidf)
            similarities.append(sim)
        
        # 返回与所有参考答案的平均相似度
        return np.mean(similarities) if similarities else 0.0
    
    def _compute_tfidf_vector(self, ngram_counts: Counter, n: int) -> Dict[str, float]:
        """
        计算n-gram的TF-IDF向量
        """
        tfidf = {}
        total_ngrams = sum(ngram_counts.values())
        
        if total_ngrams == 0:
            return tfidf
        
        for ngram, count in ngram_counts.items():
            # 计算TF (词频)
            tf = count / total_ngrams
            
            # 计算IDF (逆文档频率)
            df = self.document_frequency.get((n, ngram), 1)
            total_docs = self.document_frequency.get(f'total_docs_{n}', 1)
            idf = np.log(total_docs / df)
            
            # 计算TF-IDF
            tfidf[ngram] = tf * idf
        
        return tfidf
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        计算两个向量的余弦相似度
        """
        # 获取共同的词汇
        common_words = set(vec1.keys()) & set(vec2.keys())
        
        if not common_words:
            return 0.0
        
        # 计算点积
        dot_product = sum(vec1[word] * vec2[word] for word in common_words)
        
        # 计算向量的模长
        norm1 = np.sqrt(sum(val**2 for val in vec1.values()))
        norm2 = np.sqrt(sum(val**2 for val in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _get_ngrams(self, text: str, n: int) -> List[str]:
        """
        提取文本中的n-gram
        """
        text = self._preprocess_text(text)
        words = text.split()
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def _preprocess_text(self, text: str) -> str:
        """
        文本预处理
        """
        # 转小写
        text = text.lower()
        
        # 移除标点符号
        text = re.sub(r'[^\w\s]', '', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


def test_with_original_qwen():
    """
    使用原始Qwen2.5-VL模型进行测试，看是否是自定义模型的问题
    """
    print("🧪 测试原始Qwen2.5-VL模型...")
    
    try:
        # 尝试加载原始Qwen2.5-VL模型
        checkpoint_path = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"
        
        # 使用原始模型
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        processor = Qwen2VLProcessor.from_pretrained(checkpoint_path)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ 原始Qwen2.5-VL模型加载成功")
        return model, processor, tokenizer, "original"
        
    except Exception as e:
        print(f"⚠️ 原始模型加载失败: {e}")
        print("🔄 回退到自定义模型...")
        
        # 回退到自定义模型
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        processor = NumericQwen2_5_VLProcessor.from_pretrained(checkpoint_path)
        model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ 自定义模型加载成功")
        return model, processor, tokenizer, "custom"


def simple_test_inference(model, processor, tokenizer, model_type):
    """
    简单的推理测试，只测试一个样本
    """
    print(f"🧪 简单推理测试 - {model_type} 模型")
    
    # 加载测试数据
    test_data_path = "/data1/wangzhiye/LLaMA-Factory/data/1bddx_test_converted1.json"
    with open(test_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_base_path = "/data1/wangzhiye/LLaMA-Factory/data"
    
    # 测试3个样本
    for i in range(3):
        try:
            sample = data[i]
            print(f"\n🔍 测试样本 {i}:")
            
            # 加载图像
            image_list = []
            image_paths = sample.get('images', [])[:4]  # 限制为4张
            
            for img_path in image_paths:
                full_img_path = os.path.join(image_base_path, img_path)
                if os.path.exists(full_img_path):
                    image = Image.open(full_img_path).convert('RGB')
                    image_list.append(image)
            
            # 构建文本
            messages = sample['messages']
            user_messages = [msg for msg in messages if msg.get('role') == 'user']
            if user_messages:
                original_text = user_messages[-1]['content']
                text_without_images = original_text.replace('<image>', '').strip()
                text_prompt = '<|image_pad|>' * len(image_list) + text_without_images
            else:
                text_prompt = '<|image_pad|>' * len(image_list) + "请描述这些图片。"
            
            print(f"   图像数量: {len(image_list)}")
            print(f"   文本: {text_prompt[:100]}...")
            
            # 处理输入
            inputs = processor(
                text=text_prompt,
                images=image_list if image_list else None,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # 移动到设备
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            print(f"   input_ids shape: {inputs['input_ids'].shape}")
            if 'pixel_values' in inputs:
                print(f"   pixel_values shape: {inputs['pixel_values'].shape}")
            
            # 检查图像token数量
            input_ids = inputs['input_ids'][0].cpu().numpy()
            image_token_id = 151655  # <|image_pad|> 的token id
            image_token_count = np.sum(input_ids == image_token_id)
            print(f"   图像token数量: {image_token_count}")
            
            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False
                )
            
            # 解码
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            print(f"   ✅ 预测: {prediction[:50]}...")
            
        except Exception as e:
            print(f"   ❌ 样本 {i} 失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    """
    主函数
    """
    print("🚀 CIDEr评估器 - 调试版本")
    print("=" * 60)
    
    # 测试模型加载和推理
    model, processor, tokenizer, model_type = test_with_original_qwen()
    
    # 简单推理测试
    simple_test_inference(model, processor, tokenizer, model_type)


if __name__ == "__main__":
    main()
