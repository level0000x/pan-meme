#!/usr/bin/env python3
"""
run_pipeline.py — pan-meme 四阶段管线 → ROSE WikiLine TSV
===========================================================
论文主导：以 Pipeline.run_forward() 为唯一数据通道。
模块一（浮现）→ 模块二（几何化）→ 模块三（模因化）→ 模块四（绑定）

数据流:
  输入文本 → Tokenizer → CycleEngine(↑↓+背景树+子串包含)
  → RelationExtractor → Reasoner → ConceptComposer
  → RuleExtractor → Consistency → MathModel
  → Geometrizer → Decomposer → Binder
  → 提取 PSI → TsvBridge → WikiLine TSV

用法:
  python run_pipeline.py --batch              # 批量处理所有数据源
  python run_pipeline.py --input text.txt     # 处理单个文件
"""

__version__ = "0.2.0"

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Tuple, Optional

_PM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PM_DIR not in sys.path:
    sys.path.insert(0, _PM_DIR)

_TSV_DIR = os.path.join(os.path.dirname(_PM_DIR), 'sighted-wiki', 'data', 'wiki_tsv', 'records')
_SIGHTED_DATA = os.path.join(os.path.dirname(_PM_DIR), 'sighted-wiki', 'data')
_BG_TREE = os.path.join(_PM_DIR, 'data', 'background_tree.json')


def load_background() -> Dict[str, List[str]]:
    """加载背景层级树，返回 {上位词: [下位词列表]}"""
    if not os.path.exists(_BG_TREE):
        print(f"  背景树未找到: {_BG_TREE}，跳过")
        return {}
    with open(_BG_TREE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tree = data.get('tree', {})
    print(f"  背景树: {len(tree)} 上位词, {data['stats']['total_relations']} 条关系")
    return tree


def run_pipeline_full(text: str, hypernym_dict: Dict[str, List[str]],
                      max_nodes: int = 300) -> Tuple[List[str], List, List, Dict]:
    """
    论文主导的管线：使用 Adapter.adapt() 执行模块一（浮现）。

    模块一流程：Tokenizer → CycleEngine(↑↓+背景树+子串包含)
    → RelationExtractor → Reasoner → ConceptComposer
    → RuleExtractor → Consistency → MathModel

    参数:
      text: 输入文本
      hypernym_dict: 背景上下位词典
      max_nodes: 最大节点数，超过则跳过 Reasoner（避免 O(n³) 瓶颈）

    返回: (node_texts, edges, weights, hierarchy)
    """
    from pan_meme.module1_input.adapter import InputAdapter, InputConfig

    config = InputConfig(
        cycle_mode='converge',
        cycle_max_rounds=20,
        threshold_default=0.5,
        transitive_decay=0.9,
        symmetric_decay=0.85,
        max_nodes_reason=max_nodes,
        hypernym_dict=hypernym_dict or {},
    )

    adapter = InputAdapter(config)
    data = adapter.adapt(text)

    if data.psi is None or not data.psi.nodes:
        return [], [], [], {}

    psi = data.psi

    # 从 HierarchyTree 还原原始文本
    tree = adapter._cycle_engine._last_tree
    node_texts = tree.token_texts if tree and tree.token_texts else psi.nodes

    print(f"  PSI: {len(node_texts)} 节点, {len(psi.edges)} 边")
    if data.math_model:
        print(f"  MathModel: {len(data.math_model.rules)} 规则, {len(data.math_model.constraints)} 约束")

    return node_texts, psi.edges, list(psi.weights), psi.hierarchy


def run_pipeline_from_words(words: List[str],
                             hypernym_dict: Dict[str, List[str]],
                             max_nodes: int = 300) -> Tuple[List[str], List, List, Dict]:
    """
    字典模式：跳过 jieba 分词，直接将词列表作为 token 注入管线。

    字典词本身就是 token，不需要 jieba 二次分词。
    此函数绕过 Tokenizer，直接构造 Token 对象送入 CycleEngine。

    参数:
      words: 预分词的词列表
      hypernym_dict: 背景上下位词典
      max_nodes: 最大节点数

    返回: (node_texts, edges, weights, hierarchy)
    """
    from pan_meme.module1_input.adapter import InputAdapter, InputConfig
    from pan_meme.core.types import Token

    config = InputConfig(
        cycle_mode='converge',
        cycle_max_rounds=20,
        threshold_default=0.5,
        transitive_decay=0.9,
        symmetric_decay=0.85,
        max_nodes_reason=max_nodes,
        hypernym_dict=hypernym_dict or {},
    )

    adapter = InputAdapter(config)

    # 直接构造 Token 对象，绕过 jieba 分词
    tokens = [Token(text=w, modality="dict", span=(0, len(w)), pos="") for w in words]

    # Step 2: CycleEngine（↑↓ 循环 + 虚拟节点注入）
    tree = adapter._step_cycle(tokens)

    # Step 3: RelationExtractor
    psi = adapter._step_relation_extract(tree, config.threshold_default)

    # 性能控制：节点过多时跳过后续步骤
    n_nodes = len(psi.nodes)
    if n_nodes > config.max_nodes_concept:
        pass  # 跳过 Reasoner/ConceptComposer 等

    if psi is None or not psi.nodes:
        return [], [], [], {}

    node_texts = tree.token_texts if tree and tree.token_texts else psi.nodes

    print(f"  PSI: {len(node_texts)} 节点, {len(psi.edges)} 边")
    return node_texts, psi.edges, list(psi.weights), psi.hierarchy


def process_text(text: str, source_label: str,
                 hypernym_dict: Dict[str, List[str]],
                 output_dir: str = None) -> str:
    """处理文本 → 完整管线 → TSV"""
    if output_dir is None:
        output_dir = _TSV_DIR

    print(f"\n--- {source_label} ---")
    print(f"  文本: {len(text)} 字符")

    t0 = time.time()
    try:
        nodes, edges, weights, hierarchy = run_pipeline_full(text, hypernym_dict)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  管线失败: {e}")
        return ''

    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.1f}s")

    if not nodes:
        print(f"  无有效内容")
        return ''

    from tsv_bridge import TsvBridge
    bridge = TsvBridge(output_dir)
    path = bridge.write_raw(nodes, edges, weights, hierarchy=hierarchy, source=source_label)
    return path


def process_batch(output_dir: str = None, skip_existing: bool = True):
    """批量处理所有数据源"""
    if output_dir is None:
        output_dir = _TSV_DIR

    os.makedirs(output_dir, exist_ok=True)
    existing = set(os.listdir(output_dir)) if os.path.isdir(output_dir) else set()

    # 加载背景树
    print("=" * 60)
    print("加载背景层级树")
    print("=" * 60)
    hypernym_dict = load_background()

    results = []

    # 1. 中文种子术语
    seeds_zh = os.path.join(_SIGHTED_DATA, 'seeds', 'science_zh.txt')
    if os.path.exists(seeds_zh) and (not skip_existing or 'seeds_zh.tsv' not in existing):
        print(f"\n{'='*60}")
        print(f"seeds_zh")
        print(f"{'='*60}")
        try:
            with open(seeds_zh, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]
            combined = '。'.join(lines)
            path = process_text(combined, 'seeds_zh', hypernym_dict, output_dir)
            results.append(('seeds_zh', path))
        except Exception as e:
            print(f"  失败: {e}")

    # 2. 英文种子术语
    seeds_en = os.path.join(_SIGHTED_DATA, 'seeds', 'science.txt')
    if os.path.exists(seeds_en) and (not skip_existing or 'seeds_en.tsv' not in existing):
        print(f"\n{'='*60}")
        print(f"seeds_en")
        print(f"{'='*60}")
        try:
            with open(seeds_en, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]
            combined = ' . '.join(lines)
            path = process_text(combined, 'seeds_en', hypernym_dict, output_dir)
            results.append(('seeds_en', path))
        except Exception as e:
            print(f"  失败: {e}")

    # 3. SUMO
    sumo_kif = os.path.join(_SIGHTED_DATA, 'raw', 'sumo_merge.kif')
    if os.path.exists(sumo_kif) and (not skip_existing or 'sumo.tsv' not in existing):
        print(f"\n{'='*60}")
        print(f"sumo")
        print(f"{'='*60}")
        try:
            import re
            concepts = []
            with open(sumo_kif, 'r', encoding='utf-8') as f:
                for line in f:
                    m = re.findall(r'\b[A-Z][a-zA-Z]+(?:Fn|Process|Attribute|Relation|Class|Entity|Set|Collection|List)?\b', line)
                    concepts.extend(m)
            combined = ' . '.join(concepts[:1000])
            path = process_text(combined, 'sumo', hypernym_dict, output_dir)
            results.append(('sumo', path))
        except Exception as e:
            print(f"  失败: {e}")

    # 4. 演示文本

    print(f"\n{'='*60}")
    print(f"完成: {len(results)} 个源")
    print(f"{'='*60}")
    for label, path in results:
        count = 0
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in f) - 1
        print(f"  {label}: {count} 条 → {path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='pan-meme 四阶段管线 → ROSE WikiLine TSV',
        epilog='python run_pipeline.py --batch',
    )
    parser.add_argument('--input', help='输入文件路径')
    parser.add_argument('--source', default='pan_meme', help='来源标签')
    parser.add_argument('--output', help='输出目录')
    parser.add_argument('--batch', action='store_true', help='批量处理')
    parser.add_argument('--no-skip', action='store_true', help='不跳过已有 TSV')

    args = parser.parse_args()

    if args.batch:
        process_batch(args.output, skip_existing=not args.no_skip)
        return

    if not args.input or not os.path.exists(args.input):
        print("用法: python run_pipeline.py --input <文件> --source <标签>")
        print("      python run_pipeline.py --batch")
        return

    hypernym_dict = load_background()
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()
    process_text(text, args.source, hypernym_dict, args.output)


if __name__ == '__main__':
    main()