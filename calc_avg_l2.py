#!/usr/bin/env python3
"""计算 cosine_eval_results.json 中 trajectory.avg_l2 的统计指标。

用法:
  python calc_avg_l2.py cosine_eval_results.json
或:
  cat cosine_eval_results.json | python calc_avg_l2.py

输出:
  有效值数量 / 缺失数量 / 均值 / 中位数 / 最小 / 最大 / 标准差
并可选择 --save raw_values.txt 保存所有有效值。
"""
import sys
import json
import math
import statistics
from typing import List, Tuple, Any


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_avg_l2(data: Any) -> List[float]:
    if not isinstance(data, dict):
        return []
    results = data.get('results', [])
    vals: List[float] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        traj = item.get('trajectory') if isinstance(item.get('trajectory'), dict) else None
        if traj is None:
            continue
        v = traj.get('avg_l2')
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                continue
            vals.append(float(v))
    return vals


def summarize(values: List[float]) -> dict:
    if not values:
        return {
            'count': 0,
            'mean': None,
            'median': None,
            'min': None,
            'max': None,
            'std': None,
        }
    return {
        'count': len(values),
        'mean': sum(values)/len(values),
        'median': statistics.median(values),
        'min': min(values),
        'max': max(values),
        'std': statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def main():
    args = sys.argv[1:]
    save_path = None
    json_path = None
    for a in list(args):
        if a == '--help':
            print(__doc__)
            return
        if a.startswith('--save='):
            save_path = a.split('=',1)[1].strip()
            args.remove(a)
    if args:
        json_path = args[0]
    # 读取 JSON
    if json_path:
        data = load_json(json_path)
    else:
        text = sys.stdin.read()
        if not text.strip():
            print('ERROR: 需要提供 JSON 文件路径或通过管道输入 JSON。', file=sys.stderr)
            sys.exit(1)
        data = json.loads(text)

    values = extract_avg_l2(data)
    summary = summarize(values)
    total_results = len(data.get('results', [])) if isinstance(data, dict) else None
    missing = (total_results - summary['count']) if (total_results is not None) else None

    print('=== avg_l2 统计 ===')
    if total_results is not None:
        print(f'总结果条目: {total_results}')
    print(f'有效 avg_l2 数: {summary["count"]}' + (f' | 缺失/无效: {missing}' if missing is not None else ''))
    if summary['count'] == 0:
        print('无有效 avg_l2 数值。')
    else:
        print(f"均值 mean: {summary['mean']:.6f}")
        print(f"中位 median: {summary['median']:.6f}")
        print(f"最小 min: {summary['min']:.6f}")
        print(f"最大 max: {summary['max']:.6f}")
        print(f"标准差 std: {summary['std']:.6f}")
    if save_path and values:
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                for v in values:
                    f.write(f'{v}\n')
            print(f'已保存 {len(values)} 个值到 {save_path}')
        except Exception as e:
            print(f'保存失败: {e}', file=sys.stderr)

if __name__ == '__main__':
    main()
