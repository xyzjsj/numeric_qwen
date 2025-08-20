#!/usr/bin/env python3
"""
从Checkpoint加载NumericQwen2.5-VL模型并在测试集上计算CIDEr分数

CIDEr (Consensus-based Image Description Evaluation) 评估方法
论文: https://arxiv.org/pdf/1411.5726

使用方法:
    python evaluate_cider.py
"""

import os
import json
import torch
import numpy as np
from collections import defaultdict, Counter
import math
import re
from typing import List, Dict, Any
from PIL import Image
from tqdm import tqdm

# 导入模型相关组件
from numeric_qwen2_5_vl import (
    NumericQwen2_5_VLConfig,
    NumericQwen2_5_VLProcessor,
    NumericQwen2_5_VLForConditionalGeneration
)
from transformers import AutoTokenizer


class CIDErScorer:
    """
    CIDEr分数计算器
    实现论文: CIDEr: Consensus-based Image Description Evaluation
    """
    
    def __init__(self, n=4, sigma=6.0):
        """
        初始化CIDEr计算器
        
        Args:
            n: n-gram的最大长度 (默认4-gram)
            sigma: 高斯核的标准差 (默认6.0)
        """
        self.n = n
        self.sigma = sigma
        self.document_frequency = defaultdict(float)
        self.ref_len = 0
        
    def cook_refs(self, refs: List[List[str]]):
        """
        预处理参考答案，计算文档频率
        
        Args:
            refs: 参考答案列表，每个元素是一个图像的多个参考描述
        """
        print("🔄 预处理参考答案...")
        
        # 统计所有n-gram的文档频率
        for ref_group in tqdm(refs, desc="处理参考答案"):
            # 对每个图像的所有参考描述
            for ref in ref_group:
                # 获取所有n-gram
                ngrams = self._get_ngrams(ref, self.n)
                # 计算该参考中出现的唯一n-gram
                unique_ngrams = set(ngrams)
                # 累加文档频率
                for ngram in unique_ngrams:
                    self.document_frequency[ngram] += 1
        
        # 计算参考长度 (用于长度惩罚)
        self.ref_len = np.log(float(len(refs)))
        
        print(f"✅ 预处理完成，共找到 {len(self.document_frequency)} 个唯一n-gram")
    
    def compute_cider(self, gts: List[List[str]], res: List[str]) -> Dict[str, float]:
        """
        计算CIDEr分数
        
        Args:
            gts: 真实参考答案 [[ref1_img1, ref2_img1, ...], [ref1_img2, ...], ...]
            res: 生成的答案 [gen_img1, gen_img2, ...]
            
        Returns:
            包含CIDEr分数的字典
        """
        assert len(gts) == len(res), f"参考答案数量 {len(gts)} 与生成答案数量 {len(res)} 不匹配"
        
        print(f"🔍 计算 {len(res)} 个样本的CIDEr分数...")
        
        # 如果还没有预处理参考答案，先处理
        if not self.document_frequency:
            self.cook_refs(gts)
        
        cider_scores = []
        
        for i, (gt_group, prediction) in enumerate(tqdm(zip(gts, res), total=len(res), desc="计算CIDEr")):
            # 计算单个样本的CIDEr分数
            score = self._compute_cider_single(gt_group, prediction)
            cider_scores.append(score)
        
        # 计算平均分数
        avg_cider = np.mean(cider_scores)
        
        return {
            'CIDEr': avg_cider,
            'scores': cider_scores,
            'count': len(cider_scores)
        }
    
    def _compute_cider_single(self, refs: List[str], candidate: str) -> float:
        """
        计算单个样本的CIDEr分数
        
        Args:
            refs: 该图像的所有参考描述
            candidate: 生成的描述
            
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
            # TF: 词频
            tf = count / total_ngrams
            
            # IDF: 逆文档频率
            df = self.document_frequency.get(ngram, 0)
            if df > 0:
                idf = np.log(len(self.document_frequency) / df)
            else:
                idf = 0
            
            # TF-IDF
            tfidf[ngram] = tf * idf
        
        return tfidf
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        计算两个向量的余弦相似度
        """
        # 获取共同的键
        common_keys = set(vec1.keys()) & set(vec2.keys())
        
        if not common_keys:
            return 0.0
        
        # 计算点积
        dot_product = sum(vec1[key] * vec2[key] for key in common_keys)
        
        # 计算向量模长
        norm1 = math.sqrt(sum(val**2 for val in vec1.values()))
        norm2 = math.sqrt(sum(val**2 for val in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _get_ngrams(self, text: str, n: int) -> List[str]:
        """
        从文本中提取n-gram
        
        Args:
            text: 输入文本
            n: n-gram的长度
            
        Returns:
            n-gram列表
        """
        # 文本预处理
        text = self._preprocess_text(text)
        
        # 分词
        words = text.split()
        
        # 提取n-gram
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


def load_model_from_checkpoint(checkpoint_path: str):
    """
    从checkpoint加载模型
    """
    print(f"🔄 从 {checkpoint_path} 加载模型...")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    
    # 加载处理器
    processor = NumericQwen2_5_VLProcessor.from_pretrained(checkpoint_path)
    
    # 加载模型
    model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("✅ 模型加载完成")
    return model, processor, tokenizer


def load_test_dataset(data_path: str):
    """
    加载测试数据集
    """
    print(f"📂 加载测试数据集: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ 加载了 {len(data)} 个测试样本")
    return data


def generate_predictions(model, processor, tokenizer, test_data: List[Dict]):
    """
    为测试数据生成预测结果
    """
    print(f"🔮 为 {len(test_data)} 个样本生成预测...")
    
    predictions = []
    references = []
    
    for i, item in enumerate(tqdm(test_data, desc="生成预测")):
        try:
            # 提取图像和对话
            if 'messages' in item:
                messages = item['messages']
                images = item.get('images', [])
            else:
                # 兼容旧格式
                conversations = item.get('conversations', [])
                messages = []
                for turn in conversations:
                    role = turn.get('from', '')
                    content = turn.get('value', '')
                    if role == 'human':
                        messages.append({"role": "user", "content": content})
                    elif role == 'gpt':
                        messages.append({"role": "assistant", "content": content})
                images = [item.get('image')] if item.get('image') else []
            
            # 加载图像 (使用数据集中的完整路径，最多4张图片)
            image_list = []
            max_images = 4  # 训练时设置的最大图片数量
            for img_path in images[:max_images]:  # 只取前4张图片
                if img_path and os.path.exists(img_path):
                    try:
                        image = Image.open(img_path).convert('RGB')
                        image_list.append(image)
                    except Exception as e:
                        print(f"⚠️ 加载图像失败 {img_path}: {e}")
                elif img_path:
                    print(f"⚠️ 图像文件不存在: {img_path}")
            
            # 如果原始样本有超过4张图片，给出提示
            if len(images) > max_images:
                print(f"📝 样本 {i} 有 {len(images)} 张图片，已限制为 {max_images} 张")
            
            # 构建输入文本 (使用聊天模板格式)
            user_messages = [msg for msg in messages if msg.get('role') == 'user']
            if user_messages:
                original_text = user_messages[-1]['content']  # 使用最后一个用户消息
                
                # 打印调试信息
                print(f"🔍 样本 {i} 原始输入: {str(original_text)[:100]}...")
                print(f"🔍 样本 {i} 图片数量: {len(image_list)}")
                
                # 如果有图像，构建conversation格式的消息
                if image_list:
                    # 确保original_text是字符串
                    if isinstance(original_text, str):
                        # 移除原有的所有<image>标记
                        cleaned_text = original_text.replace('<image>', '').strip()
                    else:
                        # 如果不是字符串，转换为字符串
                        cleaned_text = str(original_text)
                    
                    # 构建标准的conversation格式
                    content_parts = []
                    # 为每个图像添加图像部分
                    for _ in image_list:
                        content_parts.append({"type": "image"})
                    # 添加文本部分
                    if cleaned_text:
                        content_parts.append({"type": "text", "text": cleaned_text})
                    
                    conversation = [
                        {
                            "role": "user",
                            "content": content_parts
                        }
                    ]
                    
                    print(f"🔍 样本 {i} conversation格式: {conversation}")
                else:
                    # 确保original_text是字符串
                    text_content = str(original_text) if not isinstance(original_text, str) else original_text
                    conversation = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": text_content}]
                        }
                    ]
            else:
                if image_list:
                    content_parts = []
                    for _ in image_list:
                        content_parts.append({"type": "image"})
                    content_parts.append({"type": "text", "text": "请描述这些图片。"})
                    
                    conversation = [
                        {
                            "role": "user",
                            "content": content_parts
                        }
                    ]
                else:
                    conversation = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "请描述这张图片。"}]
                        }
                    ]
            
            # 处理输入 - 手动构建文本格式
            if image_list:
                # 方法: 手动构建文本，确保图像token正确
                text_parts = []
                for content_item in conversation[0]['content']:
                    if content_item['type'] == 'image':
                        text_parts.append('<image>')
                    else:
                        text_parts.append(content_item['text'])
                text_prompt = ''.join(text_parts)
                print(f"🔍 样本 {i} 手动构建文本: {text_prompt[:200]}...")
                
                # 验证图像token数量
                image_token_count = text_prompt.count('<image>')
                print(f"🔍 样本 {i} 图像token数量: {image_token_count}, 图像数量: {len(image_list)}")
                
                if image_token_count != len(image_list):
                    print(f"⚠️ 图像token数量不匹配，调整中...")
                    # 移除所有现有的<image>标记并重新添加正确数量的标记
                    text_without_images = text_prompt.replace('<image>', '')
                    image_tokens = '<image>' * len(image_list)
                    text_prompt = image_tokens + text_without_images
                    print(f"🔧 修正后的文本: {text_prompt[:200]}...")
                
                inputs = processor(
                    text=text_prompt,  # 传入处理后的字符串
                    images=image_list,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                # 调试：打印处理后的input_ids
                print(f"🔍 样本 {i} 处理器处理后的input_ids shape: {inputs['input_ids'].shape}")
                if 'pixel_values' in inputs:
                    print(f"🔍 样本 {i} pixel_values shape: {inputs['pixel_values'].shape}")
                
            else:
                # 对于无图像情况，使用简单文本
                text_content = conversation[0]['content'][0]['text']
                
                inputs = processor(
                    text=text_content,
                    images=None,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
            
            # 移动到模型设备
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # 生成预测
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 解码生成的文本
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            predictions.append(prediction)
            
            # 提取参考答案
            assistant_messages = [msg for msg in messages if msg.get('role') == 'assistant']
            ref_texts = [msg['content'] for msg in assistant_messages]
            if not ref_texts:
                ref_texts = [""]  # 如果没有参考答案，使用空字符串
            
            references.append(ref_texts)
            
        except Exception as e:
            print(f"⚠️ 处理样本 {i} 时出错: {e}")
            import traceback
            print(f"🔍 详细错误信息:")
            traceback.print_exc()
            predictions.append("")
            references.append([""])
    
    return predictions, references


def main():
    """
    主函数
    """
    print("🚀 CIDEr评估器 - NumericQwen2.5-VL")
    print("=" * 60)
    
    # 配置路径
    checkpoint_path = "/data1/wangzhiye/1a1a11/original/output/checkpoint-200"
    test_data_path = "/data1/wangzhiye/LLaMA-Factory/data/1bddx_test_converted1.json"
    
    try:
        # 1. 加载模型
        model, processor, tokenizer = load_model_from_checkpoint(checkpoint_path)
        
        # 2. 加载测试数据 (只测试前3个样本)
        test_data = load_test_dataset(test_data_path)[:3]
        
        # 3. 生成预测 (使用数据集中的完整图片路径)
        predictions, references = generate_predictions(
            model, processor, tokenizer, test_data
        )
        
        # 4. 计算CIDEr分数
        print("\n" + "=" * 60)
        print("📊 计算CIDEr分数...")
        
        cider_scorer = CIDErScorer(n=4, sigma=6.0)
        results = cider_scorer.compute_cider(references, predictions)
        
        # 5. 显示结果
        print("=" * 60)
        print("🎯 评估结果:")
        print(f"   样本数量: {results['count']}")
        print(f"   CIDEr分数: {results['CIDEr']:.4f}")
        print(f"   分数范围: [{min(results['scores']):.4f}, {max(results['scores']):.4f}]")
        print(f"   分数标准差: {np.std(results['scores']):.4f}")
        
        # 6. 保存详细结果
        output_file = "cider_evaluation_results.json"
        detailed_results = {
            "checkpoint": checkpoint_path,
            "test_dataset": test_data_path,
            "results": results,
            "sample_predictions": [
                {
                    "index": i,
                    "prediction": pred,
                    "references": refs,
                    "cider_score": score
                }
                for i, (pred, refs, score) in enumerate(zip(
                    predictions[:5], references[:5], results['scores'][:5]
                ))
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        print(f"📝 详细结果已保存到: {output_file}")
        
        # 7. 显示示例
        print("\n" + "=" * 60)
        print("📋 示例预测结果:")
        for i in range(min(3, len(predictions))):
            print(f"\n样本 {i+1}:")
            print(f"预测: {predictions[i]}")
            print(f"参考: {references[i]}")
            print(f"CIDEr: {results['scores'][i]:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
