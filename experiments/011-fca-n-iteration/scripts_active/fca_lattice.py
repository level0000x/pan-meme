"""
实验 011 — FCA 概念格构建器（v2：bigram 属性版本）
验证定理 11.3：D_i = |A_i| / |B_i| 在偏序下单调

改进：使用词内相邻字符二元组（bigrams）作为 FCA 属性，
而非单字符，以在英文文本中获得足够的区分度。

C_i ⪯ C_j ⇒ D_i ≥ D_j
"""

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


def tokenize_english(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
    stopwords = {
        "the", "and", "for", "are", "was", "were", "that", "this",
        "with", "from", "have", "been", "their", "which", "they",
        "not", "but", "has", "had", "its", "can", "all", "also",
        "than", "more", "some", "other", "each", "about", "would",
        "when", "will", "these", "such", "only", "over", "into",
        "most", "after", "where", "between", "being", "those", "them",
    }
    return [t for t in tokens if t not in stopwords]


def word_to_bigrams(word: str) -> List[str]:
    return [f"{word[i:i+2]}_{i}" for i in range(len(word) - 1)]


def build_formal_context(
    words: List[str],
    min_word_freq: int = 1,
    max_bigrams: int = 50,
) -> Tuple[List[str], List[str], Dict[int, Set[int]], Dict[int, Set[int]]]:
    """构建形式上下文，使用 bigrams 作为属性（M）。

    对象 G = 词（去重后）
    属性 M = 出现频率最高的 bigrams（取 top max_bigrams）
    关联 I = 词 w 中的 bigram 集合
    """
    word_counts = defaultdict(int)
    for w in words:
        word_counts[w] += 1

    unique_words = [w for w, c in word_counts.items() if c >= min_word_freq and len(w) >= 4]

    bigram_counts = defaultdict(int)
    word_bigram_sets: Dict[str, Set[str]] = {}
    for w in unique_words:
        bgs = set(word_to_bigrams(w))
        word_bigram_sets[w] = bgs
        for bg in bgs:
            bigram_counts[bg] += 1

    top_bigrams = sorted(bigram_counts.items(), key=lambda x: -x[1])[:max_bigrams]
    bigrams = [bg for bg, _ in top_bigrams]

    bigram_to_idx = {bg: i for i, bg in enumerate(bigrams)}

    word_to_bigram_idxs: Dict[int, Set[int]] = {}
    for wi, w in enumerate(unique_words):
        idxs = set()
        for bg in word_bigram_sets.get(w, set()):
            if bg in bigram_to_idx:
                idxs.add(bigram_to_idx[bg])
        if idxs:
            word_to_bigram_idxs[wi] = idxs

    bigram_to_word_idxs: Dict[int, Set[int]] = defaultdict(set)
    for wi, bg_idxs in word_to_bigram_idxs.items():
        for bi in bg_idxs:
            bigram_to_word_idxs[bi].add(wi)

    return unique_words, bigrams, bigram_to_word_idxs, word_to_bigram_idxs


def derivation_up(
    bigram_subset: Set[int],
    bigram_to_words: Dict[int, Set[int]],
    n_words: int,
) -> Set[int]:
    if not bigram_subset:
        return set(range(n_words))
    result = set(range(n_words))
    for bi in bigram_subset:
        result &= bigram_to_words.get(bi, set())
        if not result:
            break
    return result


def derivation_down(
    word_subset: Set[int],
    word_to_bigrams: Dict[int, Set[int]],
    n_bigrams: int,
) -> Set[int]:
    if not word_subset:
        return set(range(n_bigrams))
    result = set(range(n_bigrams))
    for wi in word_subset:
        result &= word_to_bigrams.get(wi, set())
        if not result:
            break
    return result


def compute_concepts(
    bigrams: List[str],
    bigram_to_words: Dict[int, Set[int]],
    word_to_bigrams: Dict[int, Set[int]],
    n_words: int,
    max_concepts: int = 2000,
    time_limit_sec: float = 60.0,
) -> List[Tuple[Set[int], Set[int]]]:
    """NextClosure 算法。

    概念 (A, B)：A ⊆ bigrams（内涵），B ⊆ words（外延）。
    """
    n_attrs = len(bigrams)
    t_start = time.time()

    def closure(attr_set: frozenset) -> frozenset:
        words_having_all = derivation_up(set(attr_set), bigram_to_words, n_words)
        common_attrs = derivation_down(words_having_all, word_to_bigrams, n_attrs)
        return frozenset(common_attrs)

    concepts_intent: List[frozenset] = [closure(frozenset())]
    concepts_extent: List[frozenset] = []
    for intent in concepts_intent:
        extent = frozenset(derivation_up(set(intent), bigram_to_words, n_words))
        concepts_extent.append(extent)

    current = frozenset()

    while len(concepts_intent) < max_concepts:
        if time.time() - t_start > time_limit_sec:
            break
        found = False
        for i in range(n_attrs - 1, -1, -1):
            if i not in current:
                candidate = frozenset(set(current) | {i})
                closed = closure(candidate)
                new_elements = closed - current
                if new_elements and min(new_elements) >= i:
                    current = closed
                    extent = frozenset(derivation_up(set(closed), bigram_to_words, n_words))
                    concepts_intent.append(closed)
                    concepts_extent.append(extent)
                    found = True
                    break
        if not found:
            break

    result = []
    for intent, extent in zip(concepts_intent, concepts_extent):
        result.append((set(intent), set(extent)))
    return result


def build_hasse_edges(concepts: List[Tuple[Set[int], Set[int]]]) -> List[Tuple[int, int]]:
    """构建 Hasse 图边列表。超概念 i → 子概念 j。"""
    n = len(concepts)
    edges: List[Tuple[int, int]] = []

    for i in range(n):
        Ai, Bi = concepts[i]
        for j in range(n):
            if i == j:
                continue
            Aj, Bj = concepts[j]
            if not (Ai.issuperset(Aj) and Bi.issubset(Bj)):
                continue
            if Ai == Aj and Bi == Bj:
                continue
            is_cover = True
            for k in range(n):
                if k == i or k == j:
                    continue
                Ak, Bk = concepts[k]
                if (Ai.issuperset(Ak) and Ak.issuperset(Aj) and
                    Bi.issubset(Bk) and Bk.issubset(Bj) and
                    (Ai != Ak or Bi != Bk) and (Ak != Aj or Bk != Bj)):
                    is_cover = False
                    break
            if is_cover:
                edges.append((i, j))

    return edges


def compute_d_values(concepts: List[Tuple[Set[int], Set[int]]]) -> List[float]:
    d_vals = []
    for Ai, Bi in concepts:
        nb = len(Bi)
        d_vals.append(len(Ai) / nb if nb > 0 else float("inf"))
    return d_vals


def verify_theorem_11_3(
    concepts: List[Tuple[Set[int], Set[int]]],
    d_vals: List[float],
    edges: List[Tuple[int, int]],
) -> Dict:
    """对每条 Hasse 覆盖边 (parent → child)，验证 D_parent ≥ D_child。"""
    passes = 0
    violations = []
    for parent_idx, child_idx in edges:
        Ai, Bi = concepts[parent_idx]
        Aj, Bj = concepts[child_idx]
        dp, dc = d_vals[parent_idx], d_vals[child_idx]
        if dp >= dc:
            passes += 1
        else:
            violations.append({
                "parent": parent_idx, "child": child_idx,
                "D_parent": dp, "D_child": dc, "diff": dc - dp,
                "|A_parent|": len(Ai), "|B_parent|": len(Bi),
                "|A_child|": len(Aj), "|B_child|": len(Bj),
            })
    total = len(edges)
    return {
        "total_edges": total,
        "passes": passes,
        "violations": len(violations),
        "pass_rate": passes / total if total > 0 else 1.0,
        "violation_details": violations[:20],
    }


def main():
    base_dir = Path(__file__).resolve().parent.parent
    extract_dir = base_dir.parent / "009-external-validation" / "data" / "extracts"

    if not extract_dir.exists():
        print(f"错误：数据目录不存在 {extract_dir}")
        sys.exit(1)

    extract_files = sorted(extract_dir.glob("*.json"))
    print(f"找到 {len(extract_files)} 个 Wikipedia 摘要文件")

    all_results = []
    for fpath in extract_files:
        concept_name = fpath.stem
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            continue
        page = next(iter(pages.values()))
        text = page.get("extract", "")
        if not text or len(text) < 300:
            continue

        words = tokenize_english(text)
        if len(words) < 30:
            continue

        unique_words, bigrams, bg_to_words, wd_to_bigrams = build_formal_context(
            words, min_word_freq=1, max_bigrams=200
        )

        if len(bigrams) < 5 or len(unique_words) < 10:
            continue

        print(f"  {concept_name}: {len(unique_words)}词, {len(bigrams)}bigrams", end="", flush=True)

        concepts = compute_concepts(
            bigrams=bigrams,
            bigram_to_words=bg_to_words,
            word_to_bigrams=wd_to_bigrams,
            n_words=len(unique_words),
            max_concepts=2000,
            time_limit_sec=30.0,
        )

        print(f" → {len(concepts)}概念", end="", flush=True)

        if len(concepts) < 2:
            print(" (跳过)")
            continue

        edges = build_hasse_edges(concepts)
        print(f", {len(edges)}边", end="", flush=True)

        d_vals = compute_d_values(concepts)
        result = verify_theorem_11_3(concepts, d_vals, edges)
        rate = result["pass_rate"]
        print(f", 通过率 {rate:.1%}", end="")
        if result["violations"] > 0:
            print(f" ⚠{result['violations']}违规")
        else:
            print()

        all_results.append({
            "concept": concept_name,
            "n_words": len(unique_words),
            "n_bigrams": len(bigrams),
            "n_concepts": len(concepts),
            "n_hasse_edges": len(edges),
            "d_min": float(min(d for d in d_vals if d != float("inf"))),
            "d_max": float(max(d for d in d_vals if d != float("inf"))),
            "verification": {
                "passes": result["passes"],
                "violations": result["violations"],
                "pass_rate": result["pass_rate"],
            },
        })

        out_path = base_dir / "results" / f"{concept_name}_lattice.json"
        os.makedirs(out_path.parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "concept_name": concept_name,
                "n_words": len(unique_words),
                "n_bigrams": len(bigrams),
                "n_concepts": len(concepts),
                "n_hasse_edges": len(edges),
                "d_min": float(min(d for d in d_vals if d != float("inf"))),
                "d_max": float(max(d for d in d_vals if d != float("inf"))),
                "d_values": [float(d) for d in d_vals],
                "concept_sizes": [
                    {"|A|": len(Ai), "|B|": len(Bi)} for Ai, Bi in concepts
                ],
                "edges": [[int(i), int(j)] for i, j in edges],
                "verification": result,
            }, f, indent=2)

    summary_path = base_dir / "results" / "e0_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n完成。处理了 {len(all_results)} 个概念。")

    total_edges = sum(r["n_hasse_edges"] for r in all_results)
    total_passes = sum(r["verification"]["passes"] for r in all_results)
    total_violations = sum(r["verification"]["violations"] for r in all_results)

    print(f"\n{'='*50}")
    print(f"定理 11.3 验证汇总")
    print(f"{'='*50}")
    print(f"处理概念数: {len(all_results)}")
    print(f"总概念数:   {sum(r['n_concepts'] for r in all_results)}")
    print(f"总 Hasse 边: {total_edges}")
    print(f"通过:       {total_passes}")
    print(f"违反:       {total_violations}")
    if total_edges > 0:
        print(f"通过率:     {total_passes/total_edges:.2%}")

    if total_violations > 0:
        print(f"\n违反详情:")
        for r in all_results:
            v = r["verification"]
            if v["violations"] > 0:
                print(f"  {r['concept']}: {v['violations']}违规 / {r['n_hasse_edges']}边")
                for detail in v.get("violation_details", [])[:3]:
                    print(f"    父D={detail['D_parent']:.4f} < 子D={detail['D_child']:.4f}")


if __name__ == "__main__":
    main()
