#!/usr/bin/env python3
"""
自定义 Numeric Qwen2.5-VL 仅轨迹问题评估脚本
差异点:
 - 保留 <num>/<num_pad> 数值包装 token (PRESERVE_NUM_TOKENS)
 - 解析时将包装还原再提取坐标
 - 完整打印问题 / 参考 / 回答，不截断
 - 仅计算轨迹误差指标
"""
import os
import json
import re
import math
import torch
from PIL import Image
from numeric_qwen2_5_vl import (
    NumericQwen2_5_VLForConditionalGeneration,
    NumericQwen2_5_VLProcessor
)

# ---------------- 配置 ---------------- #
CHECKPOINT_PATH = "/data1/wangzhiye/1a1a11/custom_qwen_checkpoint_4250/output/checkpoint-4250"
TEST_DATA_PATH = "/data1/wangzhiye/LLaMA-Factory/data/5vqa_data_extracted_test_converted_numeric.json"
MAX_IMAGES_PER_Q = 6
MAX_NEW_TOKENS = 128
MAX_TRAJ_SAMPLES = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
PRESERVE_NUM_TOKENS = True

TRAJ_KEYS = ["trajectory", "planning", "plan", "[pt"]

def is_traj_question(q: str) -> bool:
    if not q: return False
    ql = q.lower()
    return any(k in ql for k in TRAJ_KEYS)

VISION_BLOCK_SINGLE = "<|vision_start|><|image_pad|><|vision_end|>"
USE_SINGLE_PLACEHOLDER_FOR_ALL = True  # 自定义数值模型可以单占位符承载多图

def build_prompt(question: str, image_count: int) -> str:
    if image_count < 1: image_count = 1
    if USE_SINGLE_PLACEHOLDER_FOR_ALL:
        img_segment = VISION_BLOCK_SINGLE
    else:
        img_segment = ''.join([VISION_BLOCK_SINGLE for _ in range(image_count)])
    few_shot = (
        "<|im_start|>user\nProvide only the planning trajectory.\n<|im_end|>\n"
        "<|im_start|>assistant\n[PT, (<num><+4.10>, <num><+0.05>), (<num><+8.25>, <num><+0.37>), (<num><+12.40>, <num><+0.92>)]<|im_end|>\n"
    )
    system_inst = (
        "You are an autonomous driving perception and planning assistant. "
        "If user asks for planning, output strictly [PT, (<num><+x1>, <num><+y1>), ...] preserving <num> wrappers."
    )
    return (
        f"<|im_start|>system\n{system_inst}<|im_end|>\n" + few_shot +
        f"<|im_start|>user\n{img_segment}{question}<|im_end|>\n" +
        "<|im_start|>assistant\n"
    )

# ------------- 数据提取 ------------- #

def extract_pairs(sample: dict):
    pairs = []
    messages = sample.get('messages', [])
    imgs = sample.get('images', [])
    for i in range(len(messages)-1):
        u = messages[i]
        a = messages[i+1]
        if u.get('role') == 'user' and a.get('role') == 'assistant':
            q_raw = u.get('content','')
            ans_raw = a.get('content','')
            if not isinstance(q_raw, str) or not isinstance(ans_raw, str):
                continue
            if not is_traj_question(q_raw):
                continue
            img_count = q_raw.count('<image>')
            q = q_raw.replace('<image>', '').strip()
            if not q or not ans_raw: continue
            use_paths = imgs[:img_count] if img_count>0 else imgs[:1]
            if not use_paths: continue
            pairs.append((use_paths, q, ans_raw.strip()))
    return pairs

# ------------- 轨迹解析 ------------- #
TRAJ_BLOCK_RE = re.compile(r"\[\s*PT\s*,(.*?)\]", re.IGNORECASE)
POINT_RE = re.compile(r"\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)")

def unwrap_num(text: str) -> str:
    if not text: return ""
    # 保留 <num> 供展示，但解析时复制一份去包装
    t = text
    t = re.sub(r"<num_pad>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<num>\s*<([+-]?\d+(?:\.\d+)?)>", r"\1", t, flags=re.IGNORECASE)
    t = re.sub(r"<num>\s*([+-]?\d+(?:\.\d+)?)", r"\1", t, flags=re.IGNORECASE)
    return t

def parse_traj(text: str):
    if not text: return []
    work = unwrap_num(text)
    m = TRAJ_BLOCK_RE.search(work)
    if not m: return []
    region = m.group(1)
    pts = []
    for xm, ym in POINT_RE.findall(region):
        try:
            pts.append((float(xm), float(ym)))
        except ValueError:
            continue
    return pts

def traj_metrics(pred, ref):
    if not pred or not ref:
        return {
            'parsed': bool(pred),
            'pred_pts': len(pred),
            'ref_pts': len(ref),
            'count_diff': len(pred)-len(ref),
            'avg_l2': None
        }
    n = min(len(pred), len(ref))
    l2s = [math.sqrt((pred[i][0]-ref[i][0])**2 + (pred[i][1]-ref[i][1])**2) for i in range(n)]
    return {
        'parsed': True,
        'pred_pts': len(pred),
        'ref_pts': len(ref),
        'count_diff': len(pred)-len(ref),
        'avg_l2': sum(l2s)/n if n>0 else None
    }

# ------------- 模型与推理 ------------- #

def load_model():
    print(f"加载自定义数值模型: {CHECKPOINT_PATH}")
    model = NumericQwen2_5_VLForConditionalGeneration.from_pretrained(
        CHECKPOINT_PATH, torch_dtype=DTYPE, device_map="auto", trust_remote_code=True
    )
    processor = NumericQwen2_5_VLProcessor.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)
    return model, processor

def generate(model, processor, images, question: str):
    valid = []
    for p in images[:MAX_IMAGES_PER_Q]:
        if p and os.path.exists(p):
            try:
                valid.append(Image.open(p).convert('RGB'))
            except Exception:
                pass
    if not valid:
        return "", None
    prompt = build_prompt(question, len(valid))
    inputs = processor(text=[prompt], images=valid, padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    new_tokens = [o[len(i):] for i,o in zip(inputs.input_ids, out_ids)]
    # 保留 <num> => skip_special_tokens=False
    ans = processor.batch_decode(new_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0].strip()
    return ans, inputs

# ------------- 主流程 ------------- #

def main():
    print("===== 仅轨迹评估 (自定义) =====")
    model, processor = load_model()
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    traj_pairs = []
    for sample in data:
        traj_pairs.extend(extract_pairs(sample))
        if len(traj_pairs) >= MAX_TRAJ_SAMPLES:
            break

    print(f"收集到轨迹问答: {len(traj_pairs)} 条")
    if not traj_pairs:
        return

    parsed_ok = 0
    l2_list = []

    for idx, (imgs, q, ref) in enumerate(traj_pairs, 1):
        print(f"\n--- Traj Sample {idx}/{len(traj_pairs)} ---")
        print(f"图像数: {len(imgs)} | 首图: {imgs[0] if imgs else 'N/A'}")
        print(f"问题: {q}")
        print(f"参考: {ref}")
        ans, inputs = generate(model, processor, imgs, q)
        print(f"答案: {ans}")
        pred_pts = parse_traj(ans)
        ref_pts = parse_traj(ref)
        m = traj_metrics(pred_pts, ref_pts)
        if m['parsed'] and m['avg_l2'] is not None:
            parsed_ok += 1
            l2_list.append(m['avg_l2'])
        print(f"指标: parsed={m['parsed']} pred_pts={m['pred_pts']} ref_pts={m['ref_pts']} diff={m['count_diff']} avg_l2={m['avg_l2']}")

    print("\n===== 汇总 =====")
    print(f"解析成功条数: {parsed_ok}")
    if l2_list:
        from statistics import mean, median
        print(f"平均 L2: {mean(l2_list):.4f} | 中位: {median(l2_list):.4f} | 最小: {min(l2_list):.4f} | 最大: {max(l2_list):.4f} | 样本: {len(l2_list)}")
    else:
        print("无 L2 样本")
    print("完成")

if __name__ == "__main__":
    main()
