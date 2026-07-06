#!/usr/bin/env python3
"""
run_dictionary.py — 新华字典词 → pan-meme 管线 → WikiLine TSV
=============================================================
按背景树的上位词分组，分批跑管线，每批 ≈200 词。
策略：上位词 + 下位词 → 结构化文本 → 管线 → TSV

用法:
  python run_dictionary.py                    # 全部批处理
  python run_dictionary.py --limit 50         # 只跑前50个上位词组
  python run_dictionary.py --batch-size 150   # 自定义每批词数
"""

__version__ = "0.1.0"

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Tuple

_PM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PM_DIR not in sys.path:
    sys.path.insert(0, _PM_DIR)

_BG_TREE = os.path.join(_PM_DIR, 'data', 'background_tree.json')
_TSV_DIR = os.path.join(os.path.dirname(_PM_DIR), 'sighted-wiki', 'data', 'wiki_tsv', 'records')


def load_background_tree() -> dict:
    """加载背景层级树"""
    with open(_BG_TREE, 'r', encoding='utf-8') as f:
        return json.load(f)


def make_batches(tree: dict, batch_size: int = 200, min_children: int = 3) -> List[Tuple[str, List[str]]]:
    """
    按上位词分组，合并成批次。

    每组 = 上位词 + 其下位词列表。
    多个小组合并成一个批次，直到接近 batch_size。

    返回: [(批次标签, 词列表), ...]
    """
    # tree 本身就是 {上位词: [下位词列表]} 的映射
    # 过滤：只保留有足够下位词的上位词
    groups = []
    for hyper, children in sorted(tree.items(),
                                   key=lambda x: -len(x[1])):
        if len(children) >= min_children:
            groups.append((hyper, children))

    print(f"  有效上位词组: {len(groups)}（≥{min_children} 下位词）")

    # 合并成批次
    batches = []
    current_label_parts = []
    current_words = []

    for hyper, children in groups:
        group_words = [hyper] + children
        if len(current_words) + len(group_words) > batch_size and current_words:
            batches.append(('+'.join(current_label_parts), current_words))
            current_label_parts = []
            current_words = []
        current_label_parts.append(hyper)
        current_words.extend(group_words)

    if current_words:
        batches.append(('+'.join(current_label_parts), current_words))

    print(f"  合并为 {len(batches)} 个批次")
    return batches


def process_batch(label: str, words: List[str],
                  hypernym_dict: Dict[str, List[str]],
                  bridge: 'TsvBridge') -> Tuple[int, float]:
    """处理一个批次 → TSV（字典模式：跳过 jieba 分词）"""
    from run_pipeline import run_pipeline_from_words

    t0 = time.time()
    try:
        nodes, edges, weights, hierarchy = run_pipeline_from_words(
            words, hypernym_dict, max_nodes=300
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 0, time.time() - t0

    elapsed = time.time() - t0

    if not nodes or not edges:
        return 0, elapsed

    # 所有批次写入同一个 bridge，自动去重
    path = bridge.write_raw(nodes, edges, weights,
                            hierarchy=hierarchy,
                            source='dict')
    return len(edges), elapsed


def main():
    parser = argparse.ArgumentParser(description='新华字典 → pan-meme 管线 → TSV')
    parser.add_argument('--limit', type=int, default=0,
                        help='限制处理的批次数（0=全部）')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='每批最大词数')
    parser.add_argument('--min-children', type=int, default=3,
                        help='上位词最少下位词数')
    parser.add_argument('--output', default=None,
                        help='输出目录')
    args = parser.parse_args()

    output_dir = args.output or _TSV_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("新华字典 → pan-meme 管线")
    print("=" * 60)

    # 1. 加载背景树
    print("\n[1/3] 加载背景层级树...")
    t0 = time.time()
    bg = load_background_tree()
    print(f"  上位词: {bg['stats']['hypernym_count']}, "
          f"关系: {bg['stats']['total_relations']} "
          f"({time.time()-t0:.1f}s)")

    # 2. 构建批次
    print("\n[2/3] 构建批次...")
    batches = make_batches(bg['tree'], args.batch_size, args.min_children)

    if args.limit > 0:
        batches = batches[:args.limit]
        print(f"  限制为前 {args.limit} 批")

    # 3. 逐批处理
    print(f"\n[3/3] 处理 {len(batches)} 个批次...")
    from tsv_bridge import TsvBridge
    bridge = TsvBridge(output_dir)
    total_edges = 0
    total_time = 0.0

    for i, (label, words) in enumerate(batches):
        print(f"\n  [{i+1}/{len(batches)}] {label[:60]} ({len(words)} 词)")

        edge_count, elapsed = process_batch(
            label, words, bg['tree'], bridge
        )
        total_time += elapsed
        total_edges += edge_count
        print(f"    {edge_count} 边, {elapsed:.1f}s")

    # 统计最终 TSV 记录数
    dict_path = os.path.join(output_dir, 'dict.tsv')
    total_records = 0
    if os.path.exists(dict_path):
        with open(dict_path, 'r', encoding='utf-8') as f:
            total_records = sum(1 for _ in f) - 1

    print(f"\n{'='*60}")
    print(f"完成: {len(batches)} 批, {total_records} 条 TSV, {total_time:.1f}s")
    print(f"输出: {dict_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()