"""
F1: 模糊形式概念分析（Fuzzy FCA）模块
=====================================
将 crisp 形式上下文 I ⊆ G × M 推广到 fuzzy: I: G × M → [0, 1]。

模糊 Galois 连接:
  A↑_θ(x) = min_{g∈G} [I(g, x) ←θ A(g)]
  其中 ←θ 是模糊蕴涵算子（通常取 Gödel 或 Lukasiewicz）

  B↓_θ(g) = min_{m∈M} [I(g, m) ←θ B(m)]

标准选择: Gödel 蕴涵
  a →_G b = 1 if a ≤ b else b

用于模糊概念格构建。
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


def godel_implication(a: float, b: float) -> float:
    """Gödel 蕴涵: a → b = 1 if a <= b else b"""
    return 1.0 if a <= b else b


def lukasiewicz_implication(a: float, b: float) -> float:
    """Lukasiewicz 蕴涵: a → b = min(1, 1 - a + b)"""
    return min(1.0, 1.0 - a + b)


def build_fuzzy_context(
    texts: List[str],
    attributes: List[str],
    implication="godel",
    min_freq: int = 1,
) -> Tuple[List[str], np.ndarray]:
    """
    从文本列表构建模糊形式上下文。
    对象 = 文本（文档/段落），属性 = 词汇。

    I[g, m] = 词 m 在文档 g 中的归一化频率 × 词在全语料中的重要度。

    返回: (对象名, I 矩阵 n_objects × n_attrs)
    """
    n_objects = len(texts)
    n_attrs = len(attributes)

    I = np.zeros((n_objects, n_attrs))

    for g, text in enumerate(texts):
        words = text.lower().split()
        word_count = {}
        for w in words:
            word_count[w] = word_count.get(w, 0) + 1

        if not word_count:
            continue

        max_count = max(word_count.values())

        for m, attr in enumerate(attributes):
            if attr in word_count:
                tf = word_count[attr] / max_count
                df = sum(1 for t in texts if attr in t.lower().split())
                idf = math.log((n_objects + 1) / (df + 1))
                I[g, m] = min(1.0, tf * idf / math.log(n_objects + 1))

    return I


def fuzzy_up_operator(I, A_fuzzy, implication="godel"):
    """
    模糊 ↑ 算子: A↑_θ(m) = min_g [I(g, m) ←θ A(g)]
    其中 A_fuzzy[g] ∈ [0, 1] 是对象 g 的隶属度。
    """
    n_objects, n_attrs = I.shape
    impl = godel_implication if implication == "godel" else lukasiewicz_implication

    B = np.ones(n_attrs)
    for g in range(n_objects):
        a = A_fuzzy[g]
        for m in range(n_attrs):
            B[m] = min(B[m], impl(a, I[g, m]))
    return B


def fuzzy_down_operator(I, B_fuzzy, implication="godel"):
    """
    模糊 ↓ 算子: B↓_θ(g) = min_m [I(g, m) ←θ B(m)]
    """
    n_objects, n_attrs = I.shape
    impl = godel_implication if implication == "godel" else lukasiewicz_implication

    A = np.ones(n_objects)
    for m in range(n_attrs):
        b = B_fuzzy[m]
        for g in range(n_objects):
            A[g] = min(A[g], impl(b, I[g, m]))
    return A


def fuzzy_closure(I, A_fuzzy, implication="godel", tol=1e-8):
    """模糊闭包: ↓↑A"""
    B = fuzzy_up_operator(I, A_fuzzy, implication)
    return fuzzy_down_operator(I, B, implication)


def build_fuzzy_concepts(
    I,
    max_concepts: int = 50,
    implication="godel",
    n_thresholds: int = 10,
):
    """
    通过阈值扫描构建模糊概念格。

    方法: 对 α ∈ [0, 1] 均匀离散化，在每个 α 上运行 crisp FCA，
    合并不同 α 层的概念。

    返回: [(extent_fuzzy_vector, intent_fuzzy_vector, alpha)]
    """
    n_objects, n_attrs = I.shape

    concepts = []
    seen_extents = set()

    alphas = np.linspace(0.05, 0.95, n_thresholds)
    for alpha in alphas:
        crisp_I = (I >= alpha).astype(int)

        for g in range(min(n_objects, 10)):
            if not crisp_I[g].any():
                continue
            attr_indices = np.where(crisp_I[g] > 0)[0]
            B_up_crisp = np.ones(n_attrs, dtype=int)
            for m in attr_indices:
                objs_with_attr = np.where(crisp_I[:, m] > 0)[0]
                for o in objs_with_attr:
                    row = crisp_I[o]
                    B_up_crisp = B_up_crisp & row

            A_down_crisp = np.ones(n_objects, dtype=int)
            for m in range(n_attrs):
                if B_up_crisp[m]:
                    A_down_crisp = A_down_crisp & crisp_I[:, m]

            A_fuzzy = fuzzy_down_operator(I, B_up_crisp.astype(float), implication)
            B_fuzzy = fuzzy_up_operator(I, A_fuzzy, implication)

            extent_key = tuple((A_fuzzy >= 0.5).astype(int))
            if extent_key not in seen_extents and len(concepts) < max_concepts:
                seen_extents.add(extent_key)
                concepts.append({
                    "extent": A_fuzzy.tolist(),
                    "intent": B_fuzzy.tolist(),
                    "alpha": float(alpha),
                    "|A|": float(np.sum(A_fuzzy)),
                    "|B|": float(np.sum(B_fuzzy)),
                })

    concepts.sort(key=lambda c: c["|A|"], reverse=True)
    return concepts


def fuzzy_hasse_edges(concepts):
    """为模糊概念构建 Hasse 边（基于 intent 包含度）。"""
    n = len(concepts)
    edges = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            Ai = concepts[i]["|A|"]
            Aj = concepts[j]["|A|"]
            Bi = concepts[i]["|B|"]
            Bj = concepts[j]["|B|"]

            if Aj >= Ai + 0.01 and Bi >= Bj + 0.01:
                is_cover = True
                for k in range(n):
                    if k == i or k == j:
                        continue
                    Ak = concepts[k]["|A|"]
                    Bk = concepts[k]["|B|"]
                    if (Bj >= Bk >= Bi and Aj >= Ak >= Ai
                            and (Bj - Bk > 0.005 or Ak - Ai > 0.005)):
                        is_cover = False
                        break
                if is_cover:
                    edges.append((j, i))

    return edges


def fuzzy_d_values(concepts):
    """计算模糊 D 值 = |intent|/|extent| 的模糊版本。"""
    d_vals = []
    for c in concepts:
        b_len = c["|B|"]
        a_len = c["|A|"]
        d_vals.append(b_len / a_len if a_len > 0.001 else float("inf"))
    return d_vals


def run_fuzzy_n_iteration(
    concepts,
    edges,
    params,
    max_iter=500,
    tol=1e-8,
):
    """在模糊概念格上运行 N-迭代。"""
    n = len(concepts)
    parents = [set() for _ in range(n)]
    for p, c in edges:
        parents[c].add(p)

    heights = np.zeros(n, dtype=int)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if parents[i]:
                new_h = max(heights[p] for p in parents[i]) + 1
                if new_h != heights[i]:
                    heights[i] = new_h
                    changed = True
    topo = sorted(range(n), key=lambda i: -heights[i])

    results = [None] * n
    for ci in topo:
        d_val = concepts[ci]["|B|"] / max(concepts[ci]["|A|"], 0.001)
        max_d = max(c["|B|"] / max(c["|A|"], 0.001) for c in concepts)
        d_init = min(d_val / max(max_d, 0.001), 1.0)
        b_init = max(0.0, min(1.0, 1.0 - concepts[ci]["|B|"] / max(c["|A|"] for c in concepts)))
        rho_init = max(0.0, min(1.0, concepts[ci]["|A|"] / max(c["|B|"] for c in concepts)))
        m0 = np.array([d_init, b_init, rho_init, 0.5, 0.5])

        b_up = 0.0
        rho_up = 0.0
        cnt = 0
        for p in parents[ci]:
            if results[p] is not None:
                b_up += results[p][0][1]
                rho_up += results[p][0][2]
                cnt += 1
        if cnt > 0:
            b_up /= cnt
            rho_up /= cnt

        eps = params.get("eps", 0.01)
        a1, b1, g1, d1 = params["a1"], params["b1"], params["g1"], params["d1"]
        z1, e1, t1 = params["z1"], params["e1"], params["t1"]
        k1, k2, l1, m1 = params["k1"], params["k2"], params["l1"], params["m1"]

        M = m0.copy()
        traj = [M.copy()]
        for _ in range(max_iter):
            D, B, rho, R, S = M
            den_d = a1 * R + b1 * (B + b_up) + eps
            den_b = g1 * (R + b_up) + d1 * D + eps
            den_rho = z1 * (D + rho_up) + e1 * R + eps
            den_r = t1 * (rho + rho_up + b_up) + k1 * D + k2 * S + eps
            den_s = l1 * D + m1 * R + eps

            M_new = np.array([
                (a1 * R + eps) / den_d if den_d > 0 else 0.0,
                (g1 * (R + b_up) + eps) / den_b if den_b > 0 else 0.0,
                (z1 * (D + rho_up) + eps) / den_rho if den_rho > 0 else 0.0,
                (t1 * (rho + rho_up + b_up) + eps) / den_r if den_r > 0 else 0.0,
                (l1 * D + eps) / den_s if den_s > 0 else 0.0,
            ])
            traj.append(M_new.copy())
            if np.max(np.abs(M_new - M)) < tol:
                M = M_new
                break
            M = M_new

        results[ci] = (M, b_up, rho_up, traj)

    return results


def main():
    print("=" * 60)
    print("F1: 模糊 FCA（Fuzzy Formal Concept Analysis）")
    print("=" * 60)

    docs = [
        "calculus is the mathematical study of continuous change",
        "integral calculus studies the accumulation of quantities",
        "differential calculus studies rates of change and slopes",
        "algebra is the study of mathematical symbols and rules",
        "linear algebra studies vectors and matrices and linear transformations",
        "geometry studies shapes sizes and positions of figures",
        "probability theory studies random phenomena and uncertainty",
    ]

    attributes = [
        "calculus", "mathematical", "study", "continuous", "change",
        "integral", "differential", "algebra", "linear", "vectors",
        "matrices", "geometry", "shapes", "probability", "random",
    ]

    print(f"\n对象数: {len(docs)}")
    print(f"属性数: {len(attributes)}")

    I = build_fuzzy_context(docs, attributes, implication="godel")
    print(f"模糊上下文 I: {I.shape}")
    print(f"  值域: [{I.min():.3f}, {I.max():.3f}]")
    print(f"  非零率: {100.0 * (I > 0).sum() / I.size:.1f}%")
    print()

    print("构建模糊概念格...")
    concepts = build_fuzzy_concepts(I, max_concepts=30, n_thresholds=10)
    print(f"  概念数: {len(concepts)}")

    for ci, c in enumerate(concepts[:8]):
        print(f"  [{ci}] |A|={c['|A|']:.2f}  |B|={c['|B|']:.2f}  α={c['alpha']:.2f}")

    edges = fuzzy_hasse_edges(concepts)
    print(f"\nHasse 边数: {len(edges)}")
    for e in edges[:6]:
        print(f"  {e[0]} → {e[1]}  (|A|: {concepts[e[0]]['|A|']:.2f} → {concepts[e[1]]['|A|']:.2f})")

    d_vals = fuzzy_d_values(concepts)
    print(f"\nD 值域: [{min(d_vals):.3f}, {max(d_vals):.3f}]")
    for ci, d in enumerate(d_vals[:8]):
        print(f"  [{ci}] D = {d:.4f}")

    if len(edges) > 0:
        print("\n运行 N-迭代...")
        params = {
            "a1": 1.0, "b1": 1.0, "g1": 1.0, "d1": 1.0,
            "z1": 1.0, "e1": 1.0, "t1": 1.0,
            "k1": 1.0, "k2": 1.0, "l1": 1.0, "m1": 1.0,
            "eps": 0.01,
        }
        results = run_fuzzy_n_iteration(concepts, edges, params)

        print("  N-迭代结果:")
        for ci in range(min(8, len(concepts))):
            if results[ci] is not None:
                m_star, b_up, rho_up, traj = results[ci]
                tau_inv = 1.0 / (np.sum(m_star) + 0.01)
                print(f"  [{ci}] M*={[f'{x:.2f}' for x in m_star]}  τ⁻¹={tau_inv:.4f}  iters={len(traj)}")

        tau_invs = []
        for ci in range(len(concepts)):
            if results[ci] is not None:
                tau_invs.append(1.0 / (np.sum(results[ci][0]) + 0.01))
            else:
                tau_invs.append(float("inf"))

        violations = 0
        for p, c in edges:
            if tau_invs[p] < tau_invs[c]:
                violations += 1
                print(f"  ⚠ τ⁻¹ 违反: {p}→{c}  ({tau_invs[p]:.4f} < {tau_invs[c]:.4f})")

        print(f"\n  Theorem 11.1: {len(edges)-violations}/{len(edges)} = "
              f"{100.0*(len(edges)-violations)/max(len(edges),1):.0f}%")
        if violations == 0:
            print("  VERDICT: A/(A+B) 形式在模糊 FCA 格上成立 ✓")
        else:
            print(f"  VERDICT: {violations} 条边违反 τ⁻¹ 单调性——模糊推广需进一步调整")

    result_summary = {
        "n_docs": len(docs),
        "n_attrs": len(attributes),
        "n_concepts": len(concepts),
        "n_edges": len(edges),
        "fuzzy_sparsity": float((I > 0).sum() / I.size),
        "d_range": [float(min(d_vals)), float(max(d_vals))],
    }

    results_dir = Path(__file__).resolve().parent.parent / "results"
    summary_path = results_dir / "f1_fuzzy_fca_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result_summary, f, indent=2)
    print(f"\n结果已保存: {summary_path}")


if __name__ == "__main__":
    main()
