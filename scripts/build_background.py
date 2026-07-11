#!/usr/bin/env python3
"""
build_background.py — 新华字典 → ↑↓ 循环 → 背景层级树
=====================================================
从 chinese-dictionary 的 word.json（32万词语）出发，
通过子串包含策略自动发现层级结构，构建背景知识模型。

数学对应：公理3 — ↑↓ 自循环，前提0 — 有限层级收敛。
策略：长词是和短词的组合，短词是长词的上位词。
      "量子力学" ⊃ "力学" → ↑(量子力学) = 力学
      "力学" ⊃ "力"     → ↑(力学) = 力

输出：background_tree.json — 层级森林，供 CycleEngine 加载。
"""

__version__ = "0.1.0"

import json
import os
import sys
import time
import collections
from typing import Dict, List, Set, Optional, Tuple

# 路径
_PM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 从环境变量获取词典目录，未设置则回退到项目根目录
_DICT_DIR = os.environ.get('CHINESE_DICT_DIR', _PM_DIR)
_WORD_JSON = os.path.join(_DICT_DIR, 'word', 'word.json')
if not os.path.exists(_WORD_JSON):
    print(f"[WARN] 词典文件不存在: {_WORD_JSON}")
    print(f"      请设置环境变量 CHINESE_DICT_DIR 或确保 {_DICT_DIR}/word/word.json 存在")
_OUTPUT_DIR = os.path.join(_PM_DIR, 'data')


def load_words() -> List[str]:
    """从 word.json 提取所有词语"""
    with open(_WORD_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    words = [w['word'] for w in data if 'word' in w]
    # 去重，去纯英文/数字
    words = [w for w in words if any('\u4e00' <= c <= '\u9fff' for c in w)]
    return words


def build_substring_index(short_words: List[str]) -> Dict[str, Set[str]]:
    """
    构建子串→包含它的长词 的索引。
    
    对于每个短词（2-3字），找出所有包含它的更长的词。
    返回: {短词: {包含它的长词集合}}
    """
    short_set = set(short_words)
    # 按长度分组
    by_length: Dict[int, List[str]] = collections.defaultdict(list)
    for w in short_words:
        by_length[len(w)].append(w)
    
    # 只对2-3字的短词建索引（它们最可能成为上位词）
    candidate_hypers = set()
    for length in [2, 3]:
        candidate_hypers.update(by_length.get(length, []))
    
    print(f"  候选上位词（2-3字）: {len(candidate_hypers)} 个")
    
    # 对于长度≥4的词，检查它包含哪些短词
    longer_words = []
    for length in sorted(by_length.keys()):
        if length >= 4:
            longer_words.extend(by_length[length])
    
    print(f"  长词（≥4字）: {len(longer_words)} 个")
    
    # 构建索引：短词 → 包含它的长词
    index: Dict[str, Set[str]] = collections.defaultdict(set)
    
    for i, long_word in enumerate(longer_words):
        if i % 10000 == 0:
            print(f"    处理长词: {i}/{len(longer_words)}")
        # 生成所有长度≥2的子串
        L = len(long_word)
        for start in range(L):
            for end in range(start + 2, min(start + 5, L + 1)):  # 只检查2-4字子串
                sub = long_word[start:end]
                if sub in candidate_hypers:
                    index[sub].add(long_word)
    
    print(f"  索引构建完成: {len(index)} 个短词有上位关系")
    return index


def build_hierarchy_tree(words: List[str],
                         hypernym_index: Dict[str, Set[str]]) -> Dict[str, object]:
    """
    从子串包含索引构建层级树。
    
    层级分配算法：
    1. 没有出现在任何索引中的短词 → 层级0（最底层，原子词）
    2. 被索引引用但自身也是更长词的子串 → 层级1（中间层）
    3. 只作为子串出现，自己不包含任何短词 → 层级2+（上层）
    
    返回: {
        "nodes": {词: {"level": int, "children": [词], "parents": [词]}},
        "max_depth": int,
        "stats": {...}
    }
    """
    # 收集所有词
    all_words = set(words)
    # 被索引引用的短词（上位词候选）
    hypernyms = set(hypernym_index.keys())
    # 所有出现在长词位置的词
    hyponyms = set()
    for children in hypernym_index.values():
        hyponyms.update(children)
    
    print(f"  上位词候选: {len(hypernyms)}")
    print(f"  下位词: {len(hyponyms)}")
    print(f"  既是上位又是下位: {len(hypernyms & hyponyms)}")
    
    # 构建层级树
    tree: Dict[str, List[str]] = collections.defaultdict(list)  # parent → children
    reverse: Dict[str, List[str]] = collections.defaultdict(list)  # child → parents
    
    for hyper, children in hypernym_index.items():
        for child in children:
            tree[hyper].append(child)
            reverse[child].append(hyper)
    
    # 层级分配：BFS从最底层词开始
    # 底层词：不是任何词的上位词（没有children）
    level: Dict[str, int] = {}
    
    # 找到所有叶子节点（最短的词，没有下位词）
    leaves = [w for w in all_words if w not in tree or not tree[w]]
    for leaf in leaves:
        level[leaf] = 0
    
    print(f"  叶子节点（层级0）: {len(leaves)}")
    
    # 多轮BFS分配层级：父节点层级 = max(子节点层级) + 1
    changed = True
    round_count = 0
    while changed and round_count < 20:
        changed = False
        round_count += 1
        for parent, children in tree.items():
            if parent in level:
                continue
            child_levels = [level.get(c) for c in children if c in level]
            if child_levels:
                new_level = max(child_levels) + 1
                level[parent] = new_level
                changed = True
    
    # 未分配层级的词设为0
    unassigned = [w for w in all_words if w not in level]
    for w in unassigned:
        level[w] = 0
    
    max_depth = max(level.values()) if level else 0
    
    # 统计
    level_counts = collections.Counter(level.values())
    
    return {
        "nodes": {w: {"level": level.get(w, 0)} for w in all_words},
        "tree": {k: list(v) for k, v in tree.items()},
        "reverse": {k: list(v) for k, v in reverse.items()},
        "max_depth": max_depth,
        "stats": {
            "total_words": len(all_words),
            "total_relations": sum(len(v) for v in tree.values()),
            "hypernym_count": len(hypernyms),
            "hyponym_count": len(hyponyms),
            "level_distribution": dict(level_counts.most_common()),
            "rounds_to_converge": round_count,
        }
    }


def main():
    print("=" * 60)
    print("build_background: 新华字典 → ↑↓ 循环 → 背景层级树")
    print("=" * 60)
    
    # 1. 加载词语
    print("\n[1/4] 加载词语...")
    t0 = time.time()
    words = load_words()
    print(f"  有效中文词语: {len(words)} 个 ({time.time()-t0:.1f}s)")
    
    # 2. 构建子串包含索引
    print("\n[2/4] 构建子串包含索引...")
    t0 = time.time()
    index = build_substring_index(words)
    print(f"  索引完成 ({time.time()-t0:.1f}s)")
    
    # 3. 构建层级树
    print("\n[3/4] 构建层级树...")
    t0 = time.time()
    tree = build_hierarchy_tree(words, index)
    print(f"  层级树完成 ({time.time()-t0:.1f}s)")
    
    # 4. 保存
    print("\n[4/4] 保存背景模型...")
    output_path = os.path.join(_OUTPUT_DIR, 'background_tree.json')
    
    # 统计信息
    stats = tree['stats']
    print(f"\n{'='*60}")
    print(f"背景层级树统计")
    print(f"{'='*60}")
    print(f"  总词数: {stats['total_words']}")
    print(f"  总关系数: {stats['total_relations']}")
    print(f"  上位词数: {stats['hypernym_count']}")
    print(f"  下位词数: {stats['hyponym_count']}")
    print(f"  最大深度: {tree['max_depth']}")
    print(f"  收敛轮数: {stats['rounds_to_converge']}")
    print(f"  层级分布:")
    for level, count in sorted(stats['level_distribution'].items()):
        print(f"    层级{level}: {count} 个词")
    
    # 保存（只保存必要字段，压缩体积）
    output = {
        "tree": tree['tree'],
        "reverse": tree['reverse'],
        "max_depth": tree['max_depth'],
        "stats": stats,
        "version": __version__,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False)
    
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n  保存至: {output_path} ({size_mb:.1f} MB)")
    
    # 展示一些示例层级关系
    print(f"\n示例层级关系（前20条）:")
    shown = 0
    for parent, children in sorted(tree['tree'].items(), 
                                    key=lambda x: -len(x[1]))[:20]:
        sample_children = children[:5]
        more = f" ... +{len(children)-5}" if len(children) > 5 else ""
        print(f"  ↑ {parent}(L{tree['nodes'].get(parent,{}).get('level','?')})"
              f" → [{', '.join(sample_children)}{more}]")
        shown += 1
        if shown >= 20:
            break


if __name__ == '__main__':
    main()