#!/usr/bin/env python3
"""
run_dictionary_full.py — 新华字典全集 → pan-meme ↑↓循环 → 涌现结构
=================================================================
数学对应：
  I（原始信息）= 80k 词 + 5.7k 字 → ↑↓循环至自然收敛 → Ψ（关系网络）

核心原则：不注入任何外部知识。原始格式本身就是信息。
  - 每个词是一个 token（词的完整性被保留）
  - 每个字也是一个 token（字是最基础的符号原子）
  - CycleEngine 策略3（子串包含）自发现所有层级关系

用法:
  python run_dictionary_full.py
"""

import os
import sys
import json
import time

_PM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PM_DIR not in sys.path:
    sys.path.insert(0, _PM_DIR)

_BG_TREE = os.path.join(_PM_DIR, 'data', 'background_tree.json')
_TSV_DIR = os.path.join(os.path.dirname(_PM_DIR), 'sighted-wiki', 'data', 'wiki_tsv', 'records')


def is_chinese_char(ch: str) -> bool:
    return '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf'


def main():
    print("=" * 60)
    print("新华字典 → ↑↓循环 → 涌现结构")
    print("=" * 60)

    # ── Step 1: 加载原始信息 I ──
    print("\n[1] 加载原始信息 I ...")
    t0 = time.time()
    with open(_BG_TREE, 'r', encoding='utf-8') as f:
        bg = json.load(f)

    all_words = set()
    for hyper, children in bg['tree'].items():
        all_words.add(hyper)
        all_words.update(children)
    all_words = sorted(all_words)
    print(f"  词: {len(all_words)}")

    # 提取所有出现过的汉字
    all_chars = set()
    for w in all_words:
        for ch in w:
            if is_chinese_char(ch):
                all_chars.add(ch)
    all_chars = sorted(all_chars)
    print(f"  字: {len(all_chars)}")

    # ── Step 2: 构建论域 U — 词 + 字都在同一个空间中 ──
    all_tokens = all_words + all_chars
    print(f"  论域 U: {len(all_tokens)} 个符号")
    print(f"  ({time.time()-t0:.1f}s)")

    # ── Step 3: ↑↓循环 — 空字典，纯自发现 ──
    print("\n[2] ↑↓循环（无外部注入，纯自发现）...")
    from pan_meme.module1_input.adapter import InputAdapter, InputConfig
    from pan_meme.core.types import Token

    config = InputConfig(
        cycle_mode='converge',
        cycle_max_rounds=20,
        threshold_default=0.3,        # 低阈值，保留更多涌现关系
        transitive_decay=0.9,
        symmetric_decay=0.85,
        max_nodes_reason=0,            # 0 = 不跳过
        max_nodes_concept=0,           # 0 = 不跳过
        hypernym_dict={},              # 空字典 — 不注入任何外部知识
    )

    adapter = InputAdapter(config)
    tokens = [Token(text=w, modality="dict", span=(0, len(w)), pos="") for w in all_tokens]

    t0 = time.time()
    tree = adapter._step_cycle(tokens)
    cycle_time = time.time() - t0
    print(f"  循环完成: {tree.rounds} 轮, 深度 {tree.depth}")
    print(f"  节点: {len(tree.nodes)}")
    print(f"  收敛方式: {tree.terminated_by}")
    print(f"  ({cycle_time:.1f}s)")

    # ── Step 4: 关系提取 Ψ ──
    print("\n[3] 关系提取 Ψ ...")
    t0 = time.time()
    psi = adapter._step_relation_extract(tree, config.threshold_default)
    rel_time = time.time() - t0
    print(f"  Ψ: {len(psi.nodes)} 节点, {len(psi.edges)} 边")
    print(f"  ({rel_time:.1f}s)")

    if not psi.nodes or not psi.edges:
        print("  ERROR: 无涌现关系")
        return

    # ── Step 5: 后续步骤（Phase B）─
    step_time = time.time()
    if len(psi.nodes) <= config.max_nodes_concept or config.max_nodes_concept == 0:
        try:
            psi = adapter._step_reason(psi)
            print(f"  Step 4 Reasoner: done ({time.time()-step_time:.1f}s)")
            step_time = time.time()
            adapter._step_completeness(psi, None)
            print(f"  Step 5 Completeness: done ({time.time()-step_time:.1f}s)")
        except Exception as e:
            print(f"  Phase B 跳过: {e}")

    # ── Step 6: 输出 WikiLine TSV ──
    print("\n[4] 输出 TSV ...")
    from tsv_bridge import TsvBridge

    node_texts = tree.token_texts if tree and tree.token_texts else psi.nodes
    bridge = TsvBridge(_TSV_DIR)
    t0 = time.time()
    path = bridge.write_raw(node_texts, psi.edges, list(psi.weights),
                            hierarchy=psi.hierarchy, source='dict')
    ts_time = time.time() - t0

    with open(path, 'r', encoding='utf-8') as f:
        records = sum(1 for _ in f) - 1
    size_mb = os.path.getsize(path) / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"完成: {records} 条, {size_mb:.1f} MB")
    print(f"输出: {path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
