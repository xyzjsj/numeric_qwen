#!/usr/bin/env python3
"""
Numeric Qwen2.5-VL (checkpoint-4250) 余弦相似度简易评估脚本
核心特性:
 1. 使用自定义 NumericQwen2_5_VL 模型与处理器
 2. 与已验证可行的图像推理方式保持一致 (单 <|image_pad|> 占位符, 由处理器注入真实特征)
 3. 生成阶段关闭采样，保证稳定输出
 4. 提供两种文本相似度: TF-IDF & (可选) 模型隐藏向量平均池化
 5. 失败样本跳过，不中断整体评估
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numeric_qwen2_5_vl import (
    NumericQwen2_5_VLForConditionalGeneration,
    NumericQwen2_5_VLProcessor
)
import warnings
warnings.filterwarnings("ignore")

# ---------------- 配置 ---------------- #
CHECKPOINT_PATH = "/data1/wangzhiye/results/omini/custom_num/checkpoint-4200"
TEST_DATA_PATH = "/data1/wangzhiye/LLaMA-Factory/data/5vqa_data_extracted_test_converted.json"  # 数据项含 messages / images
NUM_SAMPLES = 15200          # 0 或负数表示遍历全部问答对
MAX_IMAGES_PER_Q = 6      # 每个问题最多使用的图像数量（场景有 6 视角）
USE_MULTI_IMAGE_PLACEHOLDERS = True  # True: 为每张图像插入一个 <|image_pad|> 占位符
USE_SINGLE_PLACEHOLDER_FOR_ALL = True  # True: 无论多少图像只放 1 组 <|vision_start|><|image_pad|><|vision_end|>
FORCE_MANUAL_TEMPLATE = True  # 强制使用手动 prompt，不尝试 apply_chat_template
USE_EMBEDDING_SIM = False  # 如需开启嵌入相似度改为 True（较慢）
MAX_NEW_TOKENS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_JSON_PATH = "/data1/wangzhiye/results/first_time/omini/custom_num/results.json"  # 结果保存路径
SAVE_EVERY = 50  # 每处理多少条增量写盘一次（0 表示仅最后写）


def load_model(checkpoint_path: str):
    print(f"🔄 加载自定义数值增强模型: {checkpoint_path}")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    processor = NumericQwen2_5_VLProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    print("✅ 模型与处理器加载完成")
    return model, processor


def load_data(path: str):
    print(f"📁 读取数据: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ 总样本数: {len(data)}")
    return data


def is_trajectory_question(q: str) -> bool:
    q_lower = q.lower()
    keys = ["trajectory", "planning", "plan", "[pt"]
    return any(k in q_lower for k in keys)


def build_manual_prompt(question: str, image_count: int) -> str:
    """构造手动 prompt, 可含多图像占位符。轨迹类问题才加 few-shot。"""
    if image_count < 1:
        image_count = 1
    if USE_SINGLE_PLACEHOLDER_FOR_ALL:
        img_segment = "<|vision_start|><|image_pad|><|vision_end|>"
    else:
        if USE_MULTI_IMAGE_PLACEHOLDERS:
            img_segment = ''.join(["<|vision_start|><|image_pad|><|vision_end|>" for _ in range(image_count)])
        else:
            img_segment = "<|vision_start|><|image_pad|><|vision_end|>"
    # few-shot：仅在轨迹问题时加入，避免普通问答被模板影响
    few_shot = ""
    if is_trajectory_question(question):
        few_shot = (
            "<|im_start|>user\nProvide the planning trajectory only.\n<|im_end|>\n"
            # 提供一个更完整的示例（6 个点），引导模型输出完整轨迹长度
            "<|im_start|>assistant\n[PT, (+4.10, +0.05), (+8.25, +0.37), (+12.40, +0.92), (+16.55, +1.65), (+20.70, +2.50), (+24.85, +3.40)]<|im_end|>\n"
        )
    system_inst = (
        "You are an autonomous driving perception and planning assistant. "
        "Given multi-view images, answer ONLY the user's request. "
        "If the user asks for a planning trajectory, output strictly a bracketed list in the form [PT, (+x1, +y1), (+x2, +y2), ...] with no extra words. "
        "Coordinates must keep the sign (use + or -) and two decimals when possible."
    )
    return (
        f"<|im_start|>system\n{system_inst}<|im_end|>\n" + few_shot +
        f"<|im_start|>user\n{img_segment}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate_answer(model, processor, image_paths, question: str) -> str:
    """支持多图像；优先使用 chat_template；失败则回退手动单图像模板。"""
    valid_images = []
    for p in image_paths[:MAX_IMAGES_PER_Q]:
        if not p or not os.path.exists(p):
            continue
        try:
            valid_images.append(Image.open(p).convert('RGB'))
        except Exception:
            continue
    if not valid_images:
        print("⚠️ 无可用图像，跳过")
        return ""

    # 统一使用手动 prompt；之前报错是因为自定义处理器没有保存 chat template。
    prompt = build_manual_prompt(question, len(valid_images))
    inputs = processor(text=[prompt], images=valid_images, padding=True, return_tensors="pt")

    inputs = inputs.to(DEVICE)
    # 调试视觉输入
    if 'pixel_values' in inputs:
        pv = inputs.pixel_values
        print(f"[调试] pixel_values.shape: {tuple(pv.shape)} (批, 图像, 通道, 高, 宽)" if pv.dim()==5 else f"[调试] pixel_values.shape: {tuple(pv.shape)}")
    if 'image_grid_thw' in inputs:
        print(f"[调试] image_grid_thw: {inputs.image_grid_thw}")
    image_token_id = getattr(model.config, 'image_token_id', None)
    if image_token_id is not None:
        cnt = (inputs.input_ids == image_token_id).sum().item()
        per_img = cnt / max(1, len(valid_images))
        placeholder_mode = 'single_placeholder_all' if USE_SINGLE_PLACEHOLDER_FOR_ALL else (
            'multi_placeholders' if USE_MULTI_IMAGE_PLACEHOLDERS else 'single_per_call'
        )
        print(f"[调试] image_token 总数: {cnt} | 图像数: {len(valid_images)} | 每图 token: {per_img:.1f} | 模式: {placeholder_mode}")
        if cnt == 0:
            print("❗ 未检测到任何视觉 token，结果不会利用图像")

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    new_tokens = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    return processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def cosine_tfidf(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        vec = TfidfVectorizer(lowercase=True)
        m = vec.fit_transform([a, b])
        return cosine_similarity(m[0:1], m[1:2])[0][0]
    except Exception:
        return 0.0


def cosine_embedding(a: str, b: str, model, processor) -> float:
    if not a or not b:
        return 0.0
    try:
        inp1 = processor(text=[a], return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        inp2 = processor(text=[b], return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            o1 = model(**inp1, output_hidden_states=True)
            o2 = model(**inp2, output_hidden_states=True)
        h1 = o1.hidden_states[-1]
        h2 = o2.hidden_states[-1]
        # 取 attention_mask 聚合（避免 pad 影响）
        def mean_pool(h, mask):
            if mask is None:
                return h.mean(dim=1)
            mask = mask.unsqueeze(-1).type_as(h)
            return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        e1 = mean_pool(h1, inp1.get('attention_mask'))
        e2 = mean_pool(h2, inp2.get('attention_mask'))
        e1 = torch.nn.functional.normalize(e1, dim=-1)
        e2 = torch.nn.functional.normalize(e2, dim=-1)
        return float((e1 @ e2.T).item())
    except Exception:
        return 0.0


def extract_qa_pairs_from_sample(sample: dict):
    """从单个样本 (messages + images) 提取多轮 (question, answer) 对。
    question 去掉 <image> 标记；answer 为紧随其后的 assistant 内容。
    返回 list[(images_list, question, answer)]
    """
    pairs = []
    messages = sample.get('messages', [])
    imgs = sample.get('images', [])
    for i in range(len(messages) - 1):
        m = messages[i]
        n = messages[i+1]
        if m.get('role') == 'user' and n.get('role') == 'assistant':
            raw_q = m.get('content', '')
            # 计数 <image> 以决定使用多少图像
            img_count = raw_q.count('<image>')
            question = raw_q.replace('<image>', '').strip()
            if not question:
                continue
            answer = n.get('content', '').strip()
            if not answer:
                continue
            use_paths = imgs[:img_count] if img_count > 0 else imgs[:1]
            if not use_paths:
                continue
            pairs.append((use_paths, question, answer))
    return pairs


import re
traj_re = re.compile(r"\[\s*PT\s*,(.*?)\]", re.IGNORECASE)
point_re = re.compile(r"\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)")

# -------- 数值包装清洗工具 -------- #
# 匹配 <num><+4.79> 或 <num><-12.3> 或 <num><4>
NUM_VALUE_WRAP = re.compile(r"<num>\s*<\s*([+-]?\d+(?:\.\d+)?)\s*>", re.IGNORECASE)
NUM_TOKEN = re.compile(r"<num>", re.IGNORECASE)
NUM_PAD_TOKEN = re.compile(r"<num_pad>", re.IGNORECASE)

def normalize_numeric_wrappers(text: str) -> str:
    """仅用于 L2 解析阶段: 将 <num><+4.79> 展开为 +4.79。
    保留其它 <num> / <num_pad> 标记原样（不在此处删除），以免影响原始可视化/存储。
    注意: 只在 parse_trajectory 内部调用，不改变模型输入与最终展示文本。
    """
    if not text:
        return text
    return NUM_VALUE_WRAP.sub(r"\1", text)

def parse_trajectory(text: str):
    """解析 [PT, (x,y), ...] 返回 list[(x,y)] (float)。自动清洗 <num> 包装。失败返回空列表。"""
    if not text:
        return []
    cleaned = normalize_numeric_wrappers(text)
    m = traj_re.search(cleaned)
    if not m:
        return []
    inner = m.group(1)
    pts = []
    for xm, ym in point_re.findall(inner):
        try:
            pts.append((float(xm), float(ym)))
        except ValueError:
            continue
    return pts

def trajectory_metrics(pred_pts, ref_pts):
    if not pred_pts or not ref_pts:
        return {"parsed": bool(pred_pts), "point_count_pred": len(pred_pts), "point_count_ref": len(ref_pts), "count_diff": len(pred_pts)-len(ref_pts), "avg_l2": None}
    import math
    n = min(len(pred_pts), len(ref_pts))
    l2s = [math.sqrt((pred_pts[i][0]-ref_pts[i][0])**2 + (pred_pts[i][1]-ref_pts[i][1])**2) for i in range(n)]
    return {"parsed": True, "point_count_pred": len(pred_pts), "point_count_ref": len(ref_pts), "count_diff": len(pred_pts)-len(ref_pts), "avg_l2": sum(l2s)/n if n>0 else None}


def main():
    print("🚀 开始 Numeric Qwen2.5-VL 余弦相似度评估 (messages 格式)")
    print("=" * 60)
    model, processor = load_model(CHECKPOINT_PATH)
    data_items = load_data(TEST_DATA_PATH)  # 全量载入

    # 聚合问答对
    qa_pairs = []
    for sample in data_items:
        qa_pairs.extend(extract_qa_pairs_from_sample(sample))
        if NUM_SAMPLES and NUM_SAMPLES > 0 and len(qa_pairs) >= NUM_SAMPLES:
            break
    if NUM_SAMPLES and NUM_SAMPLES > 0:
        qa_pairs = qa_pairs[:NUM_SAMPLES]
    print(f"✅ 收集到 {len(qa_pairs)} 个问答对用于评估 (NUM_SAMPLES={NUM_SAMPLES if NUM_SAMPLES else 'ALL'})")
    if not qa_pairs:
        print("⚠️ 没有问答对，退出")
        return

    tfidf_scores = []
    traj_avg_l2_list = []
    traj_parse_success = 0
    embed_scores = []
    results = []  # 保存每个样本结果
    for idx, (img_paths, q, ref) in enumerate(qa_pairs, 1):
        print(f"\n--- 样本 {idx}/{len(qa_pairs)} ---")
        print(f"🖼️ 使用图像数: {len(img_paths)}  (显示首图路径: {img_paths[0]})")
        is_traj = is_trajectory_question(q)
        if is_traj:
            print(f"❓ 问题: {q}")
            print(f"📚 参考: {ref}")
        else:
            print(f"❓ 问题: {q[:100]}{'...' if len(q)>100 else ''}")
            print(f"📚 参考: {ref[:100]}{'...' if len(ref)>100 else ''}")
        ans = generate_answer(model, processor, img_paths, q)
        if not ans:
            print("⚠️ 无生成答案，跳过")
            continue
        if is_traj:
            print(f"🤖 答案: {ans}")
        else:
            print(f"🤖 答案: {ans[:100]}{'...' if len(ans)>100 else ''}")
        s_t = cosine_tfidf(ans, ref)
        tfidf_scores.append(s_t)
        print(f"📈 TF-IDF 相似度: {s_t:.4f}")
        sample_record = {
            "index": idx,
            "image_paths": img_paths,
            "question": q,
            "reference": ref,
            "answer": ans,
            "tfidf": s_t,
        }
        # 轨迹专属评估
        if is_traj:
            pred_pts = parse_trajectory(ans)
            ref_pts = parse_trajectory(ref)
            tm = trajectory_metrics(pred_pts, ref_pts)
            if tm["parsed"] and tm["avg_l2"] is not None:
                traj_avg_l2_list.append(tm["avg_l2"])
            if tm["parsed"]:
                traj_parse_success += 1
            print(f"🛣️ TrajEval parsed={tm['parsed']} pred_pts={tm['point_count_pred']} ref_pts={tm['point_count_ref']} diff={tm['count_diff']} avg_l2={tm['avg_l2']}")
            # 调试清洗结果（只显示前 120 字）
            print(f"[TrajDebug] clean_ref: {normalize_numeric_wrappers(ref)[:120]}")
            print(f"[TrajDebug] clean_ans: {normalize_numeric_wrappers(ans)[:120]}")
            sample_record.update({
                "trajectory": {
                    "parsed": tm['parsed'],
                    "pred_point_count": tm['point_count_pred'],
                    "ref_point_count": tm['point_count_ref'],
                    "count_diff": tm['count_diff'],
                    "avg_l2": tm['avg_l2']
                }
            })
        if USE_EMBEDDING_SIM:
            s_e = cosine_embedding(ans, ref, model, processor)
            embed_scores.append(s_e)
            print(f"🧠 嵌入相似度: {s_e:.4f}")
            sample_record["embedding_sim"] = s_e
        results.append(sample_record)

        # 增量保存
        if SAVE_EVERY and SAVE_EVERY > 0 and idx % SAVE_EVERY == 0:
            try:
                with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                    json.dump({
                        "checkpoint": CHECKPOINT_PATH,
                        "total_processed": idx,
                        "num_samples_limit": NUM_SAMPLES,
                        "results": results
                    }, f, ensure_ascii=False, indent=2)
                print(f"💾 已临时保存 {idx} 条到 {OUTPUT_JSON_PATH}")
            except Exception as e:
                print(f"⚠️ 临时保存失败: {e}")

    print("\n" + "=" * 60)
    print("📊 汇总结果")
    if tfidf_scores:
        arr = np.array(tfidf_scores)
        print(f"TF-IDF 平均: {arr.mean():.4f} | 中位: {np.median(arr):.4f} | 最小: {arr.min():.4f} | 最大: {arr.max():.4f} | 样本: {len(arr)}")
    else:
        print("无 TF-IDF 结果")
    if traj_avg_l2_list:
        arr_l2 = np.array(traj_avg_l2_list)
        print(f"🛣️ 轨迹平均 L2: {arr_l2.mean():.4f} | 中位: {np.median(arr_l2):.4f} | 最小: {arr_l2.min():.4f} | 最大: {arr_l2.max():.4f} | 解析成功 {traj_parse_success} / 轨迹问答")
    if USE_EMBEDDING_SIM:
        if embed_scores:
            arr = np.array(embed_scores)
            print(f"嵌入 平均: {arr.mean():.4f} | 中位: {np.median(arr):.4f} | 最小: {arr.min():.4f} | 最大: {arr.max():.4f} | 样本: {len(arr)}")
        else:
            print("无 嵌入 结果")
    print("\n✅ 评估完成")

    # 最终保存
    try:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump({
                "checkpoint": CHECKPOINT_PATH,
                "total_processed": len(results),
                "num_samples_limit": NUM_SAMPLES,
                "tfidf_summary": {
                    "mean": float(np.mean(tfidf_scores)) if tfidf_scores else None,
                    "median": float(np.median(tfidf_scores)) if tfidf_scores else None,
                    "min": float(np.min(tfidf_scores)) if tfidf_scores else None,
                    "max": float(np.max(tfidf_scores)) if tfidf_scores else None,
                },
                "trajectory_summary": {
                    "parsed_success": traj_parse_success,
                    "avg_l2_mean": float(np.mean(traj_avg_l2_list)) if traj_avg_l2_list else None,
                    "avg_l2_median": float(np.median(traj_avg_l2_list)) if traj_avg_l2_list else None,
                },
                "embedding_summary": (
                    {
                        "mean": float(np.mean(embed_scores)),
                        "median": float(np.median(embed_scores)),
                        "min": float(np.min(embed_scores)),
                        "max": float(np.max(embed_scores)),
                    } if embed_scores else None
                ),
                "results": results
            }, f, ensure_ascii=False, indent=2)
        print(f"💾 已保存完整结果到: {OUTPUT_JSON_PATH}")
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")


if __name__ == "__main__":
    main()

def load_test_data(data_path, num_samples=10):
    """加载测试数据"""
    print(f"📁 加载测试数据: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 取前num_samples个样本
    test_samples = data[:num_samples]
    print(f"✅ 加载了 {len(test_samples)} 个测试样本")
    
    return test_samples

def generate_response(model, processor, image_path, question):
    """生成模型回答"""
    try:
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # 处理输入
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = processor.process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")
        
        # 生成回答
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0].strip()
        
    except Exception as e:
        print(f"❌ 生成回答失败: {e}")
        return ""

def calculate_text_cosine_similarity(text1, text2):
    """计算两个文本的余弦相似度"""
    if not text1 or not text2:
        return 0.0
    
    # 使用TF-IDF向量化
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    
    try:
        # 计算TF-IDF矩阵
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # 计算余弦相似度
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return similarity
        
    except Exception as e:
        print(f"❌ 相似度计算失败: {e}")
        return 0.0

def calculate_embedding_similarity(text1, text2, model, processor):
    """使用模型嵌入计算相似度"""
    try:
        # 简单的文本嵌入计算（使用模型的隐藏状态）
        inputs1 = processor(text=[text1], return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs2 = processor(text=[text2], return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        inputs1 = {k: v.to("cuda") for k, v in inputs1.items()}
        inputs2 = {k: v.to("cuda") for k, v in inputs2.items()}
        
        with torch.no_grad():
            outputs1 = model(**inputs1, output_hidden_states=True)
            outputs2 = model(**inputs2, output_hidden_states=True)
            
            # 使用最后一层的平均池化作为嵌入
            embed1 = outputs1.hidden_states[-1].mean(dim=1).cpu().numpy()
            embed2 = outputs2.hidden_states[-1].mean(dim=1).cpu().numpy()
            
            # 计算余弦相似度
            similarity = cosine_similarity(embed1, embed2)[0][0]
            
        return similarity
        
    except Exception as e:
        print(f"❌ 嵌入相似度计算失败: {e}")
        return 0.0

    # (旧残留代码已清理)
