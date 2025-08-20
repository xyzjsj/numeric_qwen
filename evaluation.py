#!/usr/bin/env python3
"""
数值增强Qwen2.5-VL模型评估模块

提供完整的模型性能评估和对比功能
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import swanlab

# BLEU和ROUGE评估
try:
    from nltk.translate.bleu_score import sentence_bleu
    from rouge import Rouge
    EVAL_AVAILABLE = True
except ImportError:
    print("警告: 未安装nltk或rouge，文本评估功能将被禁用")
    EVAL_AVAILABLE = False


class NumericModelEvaluator:
    """数值增强模型评估器"""
    
    def __init__(self, model, processor, device='cuda'):
        self.model = model
        self.processor = processor
        self.device = device
        self.model.eval()
        
    def evaluate_numeric_performance(self, dataset, batch_size=4) -> Dict[str, float]:
        """评估数值预测性能"""
        predictions = []
        targets = []
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                # 移动数据到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(**batch)
                
                # 提取数值预测和真实值
                if hasattr(outputs, 'predicted_floats'):
                    pred_floats = outputs.predicted_floats
                    
                    # 提取真实数值
                    if 'numeric_values' in batch and batch['numeric_values']:
                        for i, sample_values in enumerate(batch['numeric_values']):
                            if sample_values:  # 如果有数值标签
                                # 获取<num>token位置
                                num_token_id = getattr(self.model.config, 'num_token_id', None)
                                if num_token_id is not None:
                                    input_ids = batch['input_ids'][i]
                                    mask = (input_ids == num_token_id)
                                    
                                    if mask.any():
                                        pred_values = pred_floats[i][mask].cpu().numpy()
                                        true_values = np.array(sample_values[0])  # 展平嵌套列表
                                        
                                        # 确保长度匹配
                                        min_len = min(len(pred_values), len(true_values))
                                        predictions.extend(pred_values[:min_len])
                                        targets.extend(true_values[:min_len])
        
        if len(predictions) == 0 or len(targets) == 0:
            return {
                'MSE': float('inf'),
                'RMSE': float('inf'),
                'MAE': float('inf'),
                'MAPE': float('inf'),
                'R²': 0.0,
                'sample_count': 0
            }
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 计算指标
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        
        # 计算MAPE (避免除零)
        mask = targets != 0
        if mask.any():
            mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
        else:
            mape = float('inf')
        
        r2 = r2_score(targets, predictions)
        
        return {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape),
            'R²': float(r2),
            'sample_count': len(predictions)
        }
    
    def evaluate_text_generation(self, dataset, batch_size=2, max_new_tokens=128) -> Dict[str, float]:
        """评估文本生成质量"""
        if not EVAL_AVAILABLE:
            return {'BLEU-4': 0.0, 'ROUGE-L': 0.0, 'ROUGE-1': 0.0, 'ROUGE-2': 0.0}
        
        generated_texts = []
        reference_texts = []
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        rouge = Rouge()
        
        with torch.no_grad():
            for batch in dataloader:
                # 生成文本
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 如果有图像，也传递图像
                generation_kwargs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'max_new_tokens': max_new_tokens,
                    'do_sample': False,
                    'temperature': 1.0,
                    'pad_token_id': self.processor.tokenizer.eos_token_id
                }
                
                if 'pixel_values' in batch and batch['pixel_values'] is not None:
                    generation_kwargs['pixel_values'] = batch['pixel_values'].to(self.device)
                if 'image_grid_thw' in batch and batch['image_grid_thw'] is not None:
                    generation_kwargs['image_grid_thw'] = batch['image_grid_thw'].to(self.device)
                
                generated_ids = self.model.generate(**generation_kwargs)
                
                # 解码生成的文本
                for i, gen_ids in enumerate(generated_ids):
                    # 移除输入部分，只保留生成的新token
                    new_tokens = gen_ids[len(input_ids[i]):]
                    generated_text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
                    # 获取参考文本
                    if 'labels' in batch:
                        ref_ids = batch['labels'][i]
                        # 移除-100的padding
                        ref_ids = ref_ids[ref_ids != -100]
                        reference_text = self.processor.tokenizer.decode(ref_ids, skip_special_tokens=True)
                    else:
                        reference_text = "参考文本"  # 默认值
                    
                    generated_texts.append(generated_text.strip())
                    reference_texts.append(reference_text.strip())
        
        # 计算BLEU分数
        bleu_scores = []
        for gen, ref in zip(generated_texts, reference_texts):
            if gen and ref:
                # 简单的BLEU计算
                gen_tokens = gen.split()
                ref_tokens = [ref.split()]  # BLEU需要参考文本列表
                if len(gen_tokens) > 0 and len(ref_tokens[0]) > 0:
                    bleu = sentence_bleu(ref_tokens, gen_tokens)
                    bleu_scores.append(bleu)
        
        # 计算ROUGE分数
        rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
        for gen, ref in zip(generated_texts, reference_texts):
            if gen and ref:
                try:
                    scores = rouge.get_scores(gen, ref)[0]
                    rouge_scores['rouge-1'].append(scores['rouge-1']['f'])
                    rouge_scores['rouge-2'].append(scores['rouge-2']['f'])
                    rouge_scores['rouge-l'].append(scores['rouge-l']['f'])
                except:
                    continue
        
        return {
            'BLEU-4': np.mean(bleu_scores) if bleu_scores else 0.0,
            'ROUGE-L': np.mean(rouge_scores['rouge-l']) if rouge_scores['rouge-l'] else 0.0,
            'ROUGE-1': np.mean(rouge_scores['rouge-1']) if rouge_scores['rouge-1'] else 0.0,
            'ROUGE-2': np.mean(rouge_scores['rouge-2']) if rouge_scores['rouge-2'] else 0.0,
            'generation_count': len(generated_texts)
        }
    
    def evaluate_inference_speed(self, dataset, batch_size=4, num_samples=50) -> Dict[str, float]:
        """评估推理速度"""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        times = []
        
        with torch.no_grad():
            sample_count = 0
            for batch in dataloader:
                if sample_count >= num_samples:
                    break
                
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 预热
                if sample_count == 0:
                    _ = self.model(**batch)
                
                # 计时
                torch.cuda.synchronize() if self.device == 'cuda' else None
                start_time = time.time()
                
                outputs = self.model(**batch)
                
                torch.cuda.synchronize() if self.device == 'cuda' else None
                end_time = time.time()
                
                times.append(end_time - start_time)
                sample_count += batch_size
        
        return {
            'avg_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'samples_per_second': batch_size / np.mean(times) if times else 0.0
        }
    
    def comprehensive_evaluation(self, dataset, batch_size=4) -> Dict[str, Any]:
        """综合评估"""
        print("🔍 开始综合评估...")
        
        # 数值性能评估
        print("  📊 评估数值预测性能...")
        numeric_metrics = self.evaluate_numeric_performance(dataset, batch_size)
        
        # 文本生成评估
        print("  📝 评估文本生成质量...")
        text_metrics = self.evaluate_text_generation(dataset, batch_size=2)
        
        # 推理速度评估
        print("  ⚡ 评估推理速度...")
        speed_metrics = self.evaluate_inference_speed(dataset, batch_size)
        
        return {
            'numeric': numeric_metrics,
            'text': text_metrics,
            'speed': speed_metrics,
            'evaluation_time': datetime.now().isoformat()
        }


def compare_models(model_results: Dict[str, Dict], save_path: str = None) -> Dict[str, str]:
    """对比多个模型的性能"""
    
    comparison = {
        'winner_by_metric': {},
        'summary': {}
    }
    
    # 数值预测对比 (越小越好的指标)
    for metric in ['MSE', 'RMSE', 'MAE', 'MAPE']:
        best_model = min(model_results.keys(), 
                        key=lambda m: model_results[m]['numeric'].get(metric, float('inf')))
        comparison['winner_by_metric'][metric] = best_model
    
    # R²对比 (越大越好)
    if any('R²' in model_results[m]['numeric'] for m in model_results):
        best_model = max(model_results.keys(),
                        key=lambda m: model_results[m]['numeric'].get('R²', -float('inf')))
        comparison['winner_by_metric']['R²'] = best_model
    
    # 文本生成对比 (越大越好)
    for metric in ['BLEU-4', 'ROUGE-L']:
        if any(metric in model_results[m]['text'] for m in model_results):
            best_model = max(model_results.keys(),
                            key=lambda m: model_results[m]['text'].get(metric, 0.0))
            comparison['winner_by_metric'][metric] = best_model
    
    # 速度对比 (samples_per_second越大越好)
    best_model = max(model_results.keys(),
                    key=lambda m: model_results[m]['speed'].get('samples_per_second', 0.0))
    comparison['winner_by_metric']['speed'] = best_model
    
    # 生成对比报告
    report = generate_comparison_report(model_results, comparison)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 对比报告已保存到: {save_path}")
    
    return comparison


def generate_comparison_report(model_results: Dict, comparison: Dict) -> str:
    """生成对比报告"""
    
    report = []
    report.append("="*80)
    report.append("📊 模型性能对比报告")
    report.append("="*80)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 模型列表
    models = list(model_results.keys())
    report.append(f"📋 对比模型: {', '.join(models)}")
    report.append("")
    
    # 数值预测性能对比
    report.append("🔢 数值预测性能对比:")
    report.append("-" * 60)
    report.append(f"{'指标':<15} " + " ".join(f"{model:<15}" for model in models) + f" {'最优模型':<15}")
    report.append("-" * 60)
    
    for metric in ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²']:
        line = f"{metric:<15} "
        for model in models:
            value = model_results[model]['numeric'].get(metric, 0.0)
            if metric == 'MAPE' and value == float('inf'):
                line += f"{'∞':<15} "
            else:
                line += f"{value:<15.4f} "
        
        winner = comparison['winner_by_metric'].get(metric, 'N/A')
        line += f"{winner:<15}"
        report.append(line)
    
    report.append("")
    
    # 文本生成质量对比
    report.append("📝 文本生成质量对比:")
    report.append("-" * 60)
    for metric in ['BLEU-4', 'ROUGE-L', 'ROUGE-1', 'ROUGE-2']:
        line = f"{metric:<15} "
        for model in models:
            value = model_results[model]['text'].get(metric, 0.0)
            line += f"{value:<15.4f} "
        
        winner = comparison['winner_by_metric'].get(metric, 'N/A')
        line += f"{winner:<15}"
        report.append(line)
    
    report.append("")
    
    # 推理速度对比
    report.append("⚡ 推理速度对比:")
    report.append("-" * 60)
    for metric in ['avg_inference_time', 'samples_per_second']:
        line = f"{metric:<15} "
        for model in models:
            value = model_results[model]['speed'].get(metric, 0.0)
            line += f"{value:<15.4f} "
        report.append(line)
    
    report.append("")
    
    # 综合评分
    report.append("🏆 综合表现:")
    report.append("-" * 60)
    scores = calculate_overall_scores(model_results)
    for model, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        report.append(f"{model}: {score:.2f}分")
    
    best_overall = max(scores.items(), key=lambda x: x[1])
    report.append(f"\n🥇 最佳模型: {best_overall[0]} (综合得分: {best_overall[1]:.2f})")
    
    return "\n".join(report)


def calculate_overall_scores(model_results: Dict) -> Dict[str, float]:
    """计算综合得分"""
    scores = {}
    
    for model in model_results.keys():
        score = 0.0
        
        # 数值预测得分 (40%)
        numeric_score = 0.0
        rmse = model_results[model]['numeric'].get('RMSE', float('inf'))
        if rmse != float('inf') and rmse > 0:
            numeric_score = 100 / (1 + rmse)  # RMSE越小得分越高
        
        r2 = model_results[model]['numeric'].get('R²', 0.0)
        if r2 > 0:
            numeric_score += r2 * 100  # R²本身就是0-1的分数
        
        score += numeric_score * 0.4
        
        # 文本生成得分 (30%)
        bleu = model_results[model]['text'].get('BLEU-4', 0.0)
        rouge = model_results[model]['text'].get('ROUGE-L', 0.0)
        text_score = (bleu + rouge) * 50  # 归一化到100分
        score += text_score * 0.3
        
        # 速度得分 (30%)
        sps = model_results[model]['speed'].get('samples_per_second', 0.0)
        speed_score = min(sps * 10, 100)  # 速度得分，最高100分
        score += speed_score * 0.3
        
        scores[model] = score
    
    return scores


def log_evaluation_to_swanlab(swanlab_run, evaluation_results: Dict, prefix: str = "eval"):
    """将评估结果记录到SwanLab"""
    if swanlab_run is None:
        return
    
    try:
        # 展平结果用于记录
        flat_results = {}
        
        # 数值指标
        for metric, value in evaluation_results['numeric'].items():
            if isinstance(value, (int, float)) and not np.isinf(value):
                flat_results[f"{prefix}/numeric_{metric.lower()}"] = value
        
        # 文本指标
        for metric, value in evaluation_results['text'].items():
            if isinstance(value, (int, float)):
                flat_results[f"{prefix}/text_{metric.lower().replace('-', '_')}"] = value
        
        # 速度指标
        for metric, value in evaluation_results['speed'].items():
            if isinstance(value, (int, float)):
                flat_results[f"{prefix}/speed_{metric}"] = value
        
        swanlab_run.log(flat_results)
        print(f"✅ 评估结果已记录到SwanLab ({prefix})")
        
    except Exception as e:
        print(f"⚠️  SwanLab记录失败: {e}")


def save_evaluation_results(results: Dict, filepath: str):
    """保存评估结果到文件"""
    import json
    import os
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 转换numpy类型为Python原生类型
    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # 转换结果
    converted_results = convert_numpy_types(results)
    
    # 保存到文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(converted_results, f, indent=2, ensure_ascii=False)
    
    print(f"📄 评估结果已保存到: {filepath}")
    return filepath
