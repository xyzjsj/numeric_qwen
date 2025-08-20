#!/usr/bin/env python3
"""
CIDEr评估脚本 - NumericQwen2.5-VL模型
简化版本，直接使用原始文本格式
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoTokenizer
import re
from collections import Counter

# 添加当前目录到路径，以便导入本地模块
sys.path.append('/data1/wangzhiye/1a1a11/original')

# 导入自定义模型组件 - 使用训练时的模块
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


def load_model_from_checkpoint(checkpoint_path: str):
    """
    从checkpoint加载模型
    """
    print(f"🔄 从 {checkpoint_path} 加载模型...")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    
    # 加载处理器
    processor = NumericQwen2_5_VLProcessor.from_pretrained(checkpoint_path)
    
    # 确保图像处理器使用正确的max_pixels设置
    if hasattr(processor, 'image_processor'):
        processor.image_processor.max_pixels = 12845056
        print(f"🔧 设置图像处理器max_pixels为: {processor.image_processor.max_pixels}")
    
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


def generate_predictions(model, processor, tokenizer, test_data: List[Dict], num_samples: int = 3):
    """
    为测试数据生成预测结果，使用简化的输入格式
    """
    print(f"🔮 为 {num_samples} 个样本生成预测...")
    
    predictions = []
    references = []
    
    image_base_path = "/data1/wangzhiye/LLaMA-Factory/data"
    
    for i, sample in enumerate(tqdm(test_data[:num_samples], desc="生成预测")):
        try:
            # 获取消息列表
            messages = sample['messages']
            
            # 加载图像
            image_list = []
            image_paths = sample.get('images', [])
            
            print(f"📝 样本 {i} 有 {len(image_paths)} 张图片")
            
            if len(image_paths) > 4:
                print(f"📝 样本 {i} 有 {len(image_paths)} 张图片，已限制为 4 张")
                image_paths = image_paths[:4]  # 限制为4张图片
            
            for img_path in image_paths:
                full_img_path = os.path.join(image_base_path, img_path)
                if os.path.exists(full_img_path):
                    image = Image.open(full_img_path).convert('RGB')
                    print(f"🔍 样本 {i} 图像 {len(image_list)}: {image.size} (W×H)")
                    image_list.append(image)
                else:
                    print(f"⚠️ 图片未找到: {full_img_path}")
            
            # 获取用户输入文本
            user_messages = [msg for msg in messages if msg.get('role') == 'user']
            if user_messages:
                original_text = user_messages[-1]['content']
                
                print(f"🔍 样本 {i} 原始输入: {str(original_text)[:100]}...")
                print(f"🔍 样本 {i} 图片数量: {len(image_list)}")
                
                # 使用正确的图像标记 <|image_pad|>
                if isinstance(original_text, str):
                    # 移除所有现有的图像标记
                    text_without_images = original_text.replace('<image>', '').strip()
                    needed_image_count = len(image_list)
                    
                    print(f"🔍 样本 {i} 需要图像标记数量: {needed_image_count}")
                    
                    # 使用正确的图像标记 <|image_pad|>
                    image_tokens = '<|image_pad|>' * needed_image_count
                    text_prompt = image_tokens + text_without_images
                    
                    print(f"🔍 样本 {i} 最终文本: {text_prompt[:200]}...")
                else:
                    # 如果不是字符串，转换
                    text_content = str(original_text)
                    image_tokens = '<|image_pad|>' * len(image_list)
                    text_prompt = image_tokens + text_content
                    
            else:
                # 没有用户消息的情况
                text_prompt = '<|image_pad|>' * len(image_list) + "请描述这些图片。"
                
            print(f"💫 样本 {i} 调用处理器...")
            
            # 使用处理器处理输入，明确设置max_pixels
            inputs = processor(
                text=text_prompt,
                images=image_list if image_list else None,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_pixels=12845056  # 明确设置最大像素数
            )
            
            print(f"🔍 样本 {i} 处理器处理后的input_ids shape: {inputs['input_ids'].shape}")
            if 'pixel_values' in inputs:
                print(f"🔍 样本 {i} pixel_values shape: {inputs['pixel_values'].shape}")
            
            # 检查tokenized text中的image token数量
            input_ids = inputs['input_ids'][0].cpu().numpy()
            image_token_id = 151655  # <|image_pad|> 的token id
            image_token_count = np.sum(input_ids == image_token_id)
            print(f"🔍 样本 {i} tokenized中的图像token数量: {image_token_count}")
            
            # 如果图像token数量为0，尝试手动添加
            if image_token_count == 0 and len(image_list) > 0:
                print(f"⚠️ 样本 {i} 没有找到图像token，跳过...")
                predictions.append("")
                references.append([""])
                continue
            
            # 移动到模型设备
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            print(f"💫 样本 {i} 开始生成...")
            
            # 清理GPU内存和模型缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 生成预测
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False  # 禁用缓存避免干扰
                )
            
            # 解码生成的文本
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            print(f"✅ 样本 {i} 预测: {prediction[:100]}...")
            
            # 获取参考答案
            assistant_messages = [msg for msg in messages if msg.get('role') == 'assistant']
            reference = assistant_messages[0]['content'] if assistant_messages else ""
            
            predictions.append(prediction)
            references.append([reference])  # CIDEr需要参考答案列表格式
            
        except Exception as e:
            print(f"⚠️ 处理样本 {i} 时出错: {e}")
            import traceback
            print(f"🔍 详细错误信息:")
            traceback.print_exc()
            predictions.append("")
            references.append([""])
    
    return predictions, references


def generate_predictions_separately(model, processor, tokenizer, test_data: List[Dict], num_samples: int = 10):
    """
    为每个样本单独生成预测，确保完全的状态隔离
    """
    print(f"🔮 为 {num_samples} 个样本单独生成预测...")
    
    predictions = []
    references = []
    
    image_base_path = "/data1/wangzhiye/LLaMA-Factory/data"
    
    for i in range(min(num_samples, len(test_data))):
        print(f"\n=== 处理样本 {i}/{num_samples} ===")
        
        try:
            sample = test_data[i]
            
            # 强制清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 重新初始化模型状态（如果有的话）
            model.eval()
            
            # 获取消息列表
            messages = sample['messages']
            
            # 加载图像
            image_list = []
            image_paths = sample.get('images', [])
            
            print(f"📝 样本 {i} 有 {len(image_paths)} 张图片")
            
            if len(image_paths) > 4:
                print(f"📝 样本 {i} 有 {len(image_paths)} 张图片，已限制为 4 张")
                image_paths = image_paths[:4]  # 限制为4张图片
            
            for img_path in image_paths:
                full_path = os.path.join(image_base_path, img_path)
                if os.path.exists(full_path):
                    try:
                        image = Image.open(full_path).convert('RGB')
                        image_list.append(image)
                        print(f"📷 加载图片: {img_path}, 尺寸: {image.size}")
                    except Exception as e:
                        print(f"⚠️ 图片 {img_path} 加载失败: {e}")
                else:
                    print(f"⚠️ 图片文件不存在: {full_path}")
            
            if not image_list:
                print(f"⚠️ 样本 {i} 没有有效图片")
                predictions.append("")
                references.append([""])
                continue
            
            # 构建文本输入 - 使用标准的图像token处理
            # 注意：我们的模型同时支持图像和数值，这里处理图像输入
            
            # 获取用户消息
            user_message = ""
            reference = ""
            for msg in messages:
                if msg['role'] == 'user':
                    user_message = msg['content']
                elif msg['role'] == 'assistant':
                    reference = msg['content']
            
            # 构建输入文本 - 使用Qwen2.5-VL的标准格式
            # 不需要手动添加图像token，让processor自动处理
            input_text = f"User: {user_message}\nAssistant:"
            
            print(f"📝 输入文本长度: {len(input_text)}")
            print(f"🖼️ 图片token数: {len(image_list)}")
            print(f"📝 实际处理的图片数量: {len(image_list)}")
            
            # 使用processor处理输入 - 确保完全重新处理
            with torch.no_grad():
                inputs = processor(
                    text=input_text,
                    images=image_list,
                    return_tensors="pt",
                    max_pixels=12845056
                )
                
                # 移动到GPU
                inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                print(f"🔍 输入形状检查:")
                for key, value in inputs.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: {value.shape}")
                
                # 检查image tokens
                input_ids = inputs['input_ids']
                # 使用正确的图像token ID
                image_token_id = 151655  # <|image_pad|> token (这是正确的图像token)
                image_token_count = (input_ids == image_token_id).sum().item()
                print(f"🔍 输入中的图像token数量: {image_token_count}")
                
                # 同时检查是否有数值token (可能和图像token混用)
                num_token_id = 151665  # <num> token  
                num_pad_token_id = 151666  # <num_pad> token
                num_token_count = (input_ids == num_token_id).sum().item()
                num_pad_token_count = (input_ids == num_pad_token_id).sum().item()
                print(f"🔍 输入中的数值token数量: <num>={num_token_count}, <num_pad>={num_pad_token_count}")
                
                # 生成响应
                print("🚀 开始生成...")
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False  # 禁用缓存以避免状态污染
                )
                
                # 解码新生成的tokens
                new_tokens = outputs[0][len(inputs['input_ids'][0]):]
                prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                
                print(f"🎯 生成的预测: {prediction[:100]}...")
                print(f"🎯 参考答案: {reference[:100]}...")
            
            predictions.append(prediction)
            references.append([reference])  # CIDEr需要参考答案列表格式
            print(f"✅ 样本 {i} 处理完成")
            
        except Exception as e:
            print(f"⚠️ 处理样本 {i} 时出错: {e}")
            import traceback
            print(f"🔍 详细错误信息:")
            traceback.print_exc()
            predictions.append("")
            references.append([""])
    
    print(f"\n✅ 总共处理了 {len(predictions)} 个样本")
    return predictions, references


def evaluate_with_cider(predictions: List[str], references: List[List[str]]) -> Dict:
    """
    使用CIDEr评估预测结果
    """
    print(f"📊 计算CIDEr分数...")
    print(f"🔍 计算 {len(predictions)} 个样本的CIDEr分数...")
    
    # 初始化CIDEr评分器
    scorer = CIDErScorer()
    
    # 预处理参考答案以计算文档频率
    print("🔄 预处理参考答案...")
    all_ngrams = {}
    
    for ref_list in tqdm(references, desc="处理参考答案"):
        for ref in ref_list:
            for n in range(1, scorer.n + 1):
                ngrams = scorer._get_ngrams(ref, n)
                for ngram in ngrams:
                    key = (n, ngram)
                    if key not in all_ngrams:
                        all_ngrams[key] = 0
                    all_ngrams[key] += 1
    
    # 设置文档频率
    for n in range(1, scorer.n + 1):
        scorer.document_frequency[f'total_docs_{n}'] = len(references)
    
    for key, freq in all_ngrams.items():
        scorer.document_frequency[key] = freq
    
    print(f"✅ 预处理完成，共找到 {len(all_ngrams)} 个唯一n-gram")
    
    # 计算每个预测的CIDEr分数
    scores = []
    for pred, ref_list in tqdm(zip(predictions, references), desc="计算CIDEr", total=len(predictions)):
        score = scorer.compute_cider(pred, ref_list)
        scores.append(score)
    
    # 计算统计信息
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    results = {
        'cider_score': mean_score,
        'std': std_score,
        'min_score': min_score,
        'max_score': max_score,
        'scores': scores,
        'predictions': predictions,
        'references': references
    }
    
    return results


def main():
    """
    主函数
    """
    print("🚀 CIDEr评估器 - NumericQwen2.5-VL")
    print("=" * 60)
    
    # 配置路径
    checkpoint_path = "/data1/wangzhiye/1a1a11/original/output/checkpoint-200"
    test_data_path = "/data1/wangzhiye/LLaMA-Factory/data/1bddx_test_converted1.json"
    
    # 加载模型
    model, processor, tokenizer = load_model_from_checkpoint(checkpoint_path)
    
    # 加载测试数据
    test_data = load_test_dataset(test_data_path)
    
    # 生成预测 - 完整测试但单独处理每个样本
    predictions, references = generate_predictions_separately(
        model, processor, tokenizer, test_data, num_samples=10  # 测试更多样本
    )
    
    # 评估
    results = evaluate_with_cider(predictions, references)
    
    # 打印结果
    print("=" * 60)
    print("🎯 评估结果:")
    print(f"   样本数量: {len(predictions)}")
    print(f"   CIDEr分数: {results['cider_score']:.4f}")
    print(f"   分数范围: [{results['min_score']:.4f}, {results['max_score']:.4f}]")
    print(f"   分数标准差: {results['std']:.4f}")
    
    # 保存结果
    with open('cider_evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"📝 详细结果已保存到: cider_evaluation_results.json")
    
    # 显示示例预测结果
    print("\n" + "=" * 60)
    print("📋 示例预测结果:")
    
    for i, (pred, ref, score) in enumerate(zip(predictions[:3], references[:3], results['scores'][:3])):
        print(f"\n样本 {i+1}:")
        print(f"预测: {pred}")
        print(f"参考: {ref}")
        print(f"CIDEr: {score:.4f}")


if __name__ == "__main__":
    main()
