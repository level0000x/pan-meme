"""
实验 011 — v3 谱导出参数

对每个 FCA 概念 (A_i, B_i)：
1. 构建外延词集合的 bigram 共现子图
2. 计算归一化拉普拉斯谱 {λ_k}
3. 计算热迹 Θ(t) = Σ e^{-t λ_k} 在四个标度点的值
4. 导出 11 个 N-迭代参数（定理 6.8 映射）
5. 运行 N-迭代 + E-3 单调查证

与 v1（均匀参数 46.5%）和 v2（手工映射 39.6%）对比。
"""

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


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


def word_bigram_data(words: List[str]) -> Tuple[List[str], List[str], Dict[int, Set[str]], Dict[int, Set[int]]]:
    """返回 (unique_words, bigrams, word_bigram_sets, bigram_to_word_idxs)。"""
    word_counts = defaultdict(int)
    for w in words:
        word_counts[w] += 1
    unique = [w for w, c in word_counts.items() if len(w) >= 4]

    all_bigrams = set()
    w_bg_sets: Dict[int, Set[str]] = {}
    for wi, w in enumerate(unique):
        bgs = {w[i:i+2] for i in range(len(w) - 1)}
        w_bg_sets[wi] = bgs
        all_bigrams |= bgs

    bg_counts = defaultdict(int)
    for wi, bgs in w_bg_sets.items():
        for bg in bgs:
            bg_counts[bg] += 1

    bigrams = [bg for bg, _ in sorted(bg_counts.items(), key=lambda x: -x[1])[:40]]

    bg_to_idx = {bg: i for i, bg in enumerate(bigrams)}

    w_bg_idxs: Dict[int, Set[int]] = {}
    for wi, bgs in w_bg_sets.items():
        idxs = {bg_to_idx[bg] for bg in bgs if bg in bg_to_idx}
        if idxs:
            w_bg_idxs[wi] = idxs

    bg_to_w: Dict[int, Set[int]] = defaultdict(set)
    for wi, idxs in w_bg_idxs.items():
        for bi in idxs:
            bg_to_w[bi].add(wi)

    return unique, bigrams, bg_to_w, w_bg_idxs


def build_concepts(
    bigrams: List[str],
    bg_to_w: Dict[int, Set[int]],
    w_bg_idxs: Dict[int, Set[int]],
    n_words: int,
    max_concepts: int = 2000,
    time_limit: float = 60.0,
) -> Tuple[List[Set[int]], List[Set[int]]]:
    """NextClosure 算法。返回 (intents, extents)。"""
    n_attrs = len(bigrams)
    t0 = time.time()

    def closure(s: frozenset) -> frozenset:
        words_set = set(range(n_words))
        for bi in s:
            words_set &= bg_to_w.get(bi, set())
            if not words_set:
                break
        if not words_set:
            return frozenset(range(n_attrs))
        bg_set = set(range(n_attrs))
        for wi in words_set:
            bg_set &= w_bg_idxs.get(wi, set())
            if not bg_set:
                break
        return frozenset(bg_set)

    intents: List[frozenset] = [closure(frozenset())]
    extents: List[frozenset] = []

    for intent in intents:
        ws = set(range(n_words))
        for bi in intent:
            ws &= bg_to_w.get(bi, set())
        extents.append(frozenset(ws))

    current = frozenset()
    while len(intents) < max_concepts:
        if time.time() - t0 > time_limit:
            break
        found = False
        for i in range(n_attrs - 1, -1, -1):
            if i not in current:
                closed = closure(frozenset(set(current) | {i}))
                new = closed - current
                if new and min(new) >= i:
                    current = closed
                    ws = set(range(n_words))
                    for bi in closed:
                        ws &= bg_to_w.get(bi, set())
                    intents.append(closed)
                    extents.append(frozenset(ws))
                    found = True
                    break
        if not found:
            break

    results_intents = [set(it) for it in intents]
    results_extents = [set(ex) for ex in extents]
    return results_intents, results_extents


def build_concept_subgraph(
    word_indices: Set[int],
    w_bg_idxs: Dict[int, Set[int]],
    n_words_total: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """构建概念外延中词之间的 bigram 共现邻接矩阵。"""
    wlist = sorted(word_indices)
    m = len(wlist)
    if m <= 1:
        return np.eye(1), np.zeros((1, 1)), np.array([1.0]), np.array([0.0])

    old_to_new = {old: new for new, old in enumerate(wlist)}

    A = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        bg_i = w_bg_idxs.get(wlist[i], set())
        if not bg_i:
            continue
        for j in range(i + 1, m):
            bg_j = w_bg_idxs.get(wlist[j], set())
            if not bg_j:
                continue
            shared = len(bg_i & bg_j)
            if shared > 0:
                A[i, j] = shared
                A[j, i] = shared

    d = np.asarray(A.sum(axis=1)).flatten()
    d_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_sqrt)

    L = np.eye(m) - D_inv_sqrt @ A @ D_inv_sqrt

    return D_inv_sqrt, A, d, np.linalg.eigvalsh(L)


def spectral_params(
    L_eigvals: np.ndarray,
    d: np.ndarray,
    n_words: int,
) -> Dict[str, float]:
    """从归一化拉普拉斯谱导出 11 个 N-迭代参数（定理 6.8 映射）。

    6 个独立谱量 → 11 个耦合系数。
    """
    m = len(L_eigvals)
    if m == 1:
        return {
            "alpha1": 1.0, "beta1": 1.0, "gamma1": 1.0, "delta1": 1.0,
            "zeta1": 1.0, "eta1": 1.0, "theta1": 1.0, "kappa1": 1.0,
            "kappa2": 1.0, "lambda1": 1.0, "mu1": 1.0, "eps": 0.01,
        }

    beta0 = int(np.sum(L_eigvals < 1e-10))
    beta0 = max(beta0, 1)

    lambda1 = float(L_eigvals[beta0]) if beta0 < m else float(L_eigvals[-1])
    lambda_max = float(L_eigvals[-1]) if m > 0 else 2.0

    lambda1 = max(lambda1, 1e-10)
    lambda_max = max(lambda_max, 1.1 * lambda1)

    t_star = np.log(beta0 + 1) / lambda1 if lambda1 > 1e-10 else 1.0

    def theta(t: float) -> float:
        vals = np.exp(-t * L_eigvals)
        return float(np.sum(vals[np.isfinite(vals)]))

    Theta_half = theta(t_star / 2)
    Theta_star = theta(t_star)
    Theta_2 = theta(2 * t_star)
    Theta_3 = theta(3 * t_star)

    alpha1_val = lambda1
    beta1_val = lambda_max
    gamma1_val = Theta_half / max(n_words, 1)
    delta1_val = lambda1
    zeta1_val = Theta_half / max(n_words, 1)
    eta1_val = max(lambda_max - lambda1, 0.1)
    theta1_val = Theta_half / max(beta0, 1)
    kappa1_val = lambda_max
    kappa2_val = lambda1 / max(lambda_max, 0.1)
    lambda1_p_val = Theta_2 / max(beta0, 1)
    mu1_val = 1.0 - beta0 / max(n_words, 1)

    all_params = np.array([
        alpha1_val, beta1_val, gamma1_val, delta1_val,
        zeta1_val, eta1_val, theta1_val,
        kappa1_val, kappa2_val, lambda1_p_val, mu1_val,
    ])

    lo, hi = 0.5, 3.0
    p_min = all_params.min()
    p_max = all_params.max()
    if p_max > p_min:
        all_params = lo + (hi - lo) * (all_params - p_min) / (p_max - p_min)
    else:
        all_params = np.full_like(all_params, (lo + hi) / 2)

    return {
        "alpha1": float(all_params[0]),
        "beta1": float(all_params[1]),
        "gamma1": float(all_params[2]),
        "delta1": float(all_params[3]),
        "zeta1": float(all_params[4]),
        "eta1": float(all_params[5]),
        "theta1": float(all_params[6]),
        "kappa1": float(all_params[7]),
        "kappa2": float(all_params[8]),
        "lambda1": float(all_params[9]),
        "mu1": float(all_params[10]),
        "eps": 0.01,
    }


def n_operator(
    M: np.ndarray,
    B_up: float,
    rho_up: float,
    params: Dict[str, float],
) -> np.ndarray:
    D, B, rho, R, S = M
    eps = params.get("eps", 0.01)
    a1, b1 = params["alpha1"], params["beta1"]
    g1, d1 = params["gamma1"], params["delta1"]
    z1, e1 = params["zeta1"], params["eta1"]
    t1 = params["theta1"]
    k1, k2 = params["kappa1"], params["kappa2"]
    l1, m1 = params["lambda1"], params["mu1"]

    N_D = (a1 * R + eps) / (a1 * R + b1 * (B + B_up) + eps)
    N_B = (g1 * (R + B_up) + eps) / (g1 * (R + B_up) + d1 * D + eps)
    N_rho = (z1 * (D + rho_up) + eps) / (z1 * (D + rho_up) + e1 * R + eps)
    N_R = (t1 * (rho + rho_up + B_up) + eps) / (t1 * (rho + rho_up + B_up) + k1 * D + k2 * S + eps)
    N_S = (l1 * D + eps) / (l1 * D + m1 * R + eps)

    return np.array([N_D, N_B, N_rho, N_R, N_S])


def run_n_iteration(
    M0: np.ndarray, B_up: float, rho_up: float,
    params: Dict[str, float],
    max_iter: int = 500, tol: float = 1e-8,
) -> Dict:
    traj = [M0.copy()]
    M = M0.copy()
    converged = False
    conv_iter = max_iter

    for k in range(max_iter):
        M_next = n_operator(M, B_up, rho_up, params)
        traj.append(M_next.copy())
        if np.linalg.norm(M_next - M) < tol:
            converged = True
            conv_iter = k + 1
            M = M_next
            break
        M = M_next

    M_star = M
    traj_np = np.array(traj)

    if converged and conv_iter >= 5:
        deltas = np.array([np.linalg.norm(traj_np[i+1]-traj_np[i]) for i in range(conv_iter-2)])
        if len(deltas) >= 3:
            ratios = deltas[1:] / (deltas[:-1] + 1e-16)
            valid = ratios[(ratios > 0) & (ratios < 1)]
            rho_val = float(np.median(valid[-10:])) if len(valid) > 0 else 0.5
        else:
            rho_val = 0.5
    else:
        rho_val = 0.5

    rho_val = np.clip(rho_val, 0.001, 0.999)
    tau_inv = float(-np.log(rho_val))
    D_star, B_star, _, R_star, S_star = M_star

    w_diss = 0.0
    for i in range(len(traj_np) - 1):
        Di, _, _, _, Si = traj_np[i]
        Dj, _, _, _, Sj = traj_np[i + 1]
        w_diss += abs(Dj * (1.0 - Sj) - Di * (1.0 - Si))

    return {
        "converged": converged, "conv_iter": conv_iter,
        "M_star": [float(v) for v in M_star],
        "rho_spectral": float(rho_val), "tau_inv": tau_inv,
        "Phi_star": float(D_star * (1.0 - S_star)),
        "W_diss": float(w_diss),
        "D_init": float(M0[0]), "D_star": float(D_star),
    }


def main():
    base_dir = Path(__file__).resolve().parent.parent
    project_root = base_dir.parent.parent
    extract_dir = project_root / "experiments" / "009-external-validation" / "data" / "extracts"
    results_dir = base_dir / "results"

    if not extract_dir.exists():
        print(f"数据目录不存在: {extract_dir}")
        sys.exit(1)

    extract_files = sorted(extract_dir.glob("*.json"))
    print(f"找到 {len(extract_files)} 个 Wikipedia 摘要")

    results_dir.mkdir(parents=True, exist_ok=True)

    all_v1 = []
    all_v3 = []
    all_param_snapshots = []
    parent_map: Dict[str, Dict[int, List[int]]] = {}

    params_uniform = {
        "alpha1": 1.0, "beta1": 1.0, "gamma1": 1.0, "delta1": 1.0,
        "zeta1": 1.0, "eta1": 1.0, "theta1": 1.0, "kappa1": 1.0,
        "kappa2": 1.0, "lambda1": 1.0, "mu1": 1.0, "eps": 0.01,
    }

    for fpath in extract_files:
        concept_name = fpath.stem
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
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

        unique, bigrams, bg_to_w, w_bg_idxs = word_bigram_data(words)
        if len(bigrams) < 5 or len(unique) < 10:
            continue

        print(f"\n  {concept_name}: {len(unique)}词, {len(bigrams)}bg", end="", flush=True)

        intents, extents = build_concepts(
            bigrams, bg_to_w, w_bg_idxs, len(unique),
            max_concepts=2000, time_limit=30.0,
        )

        n_concepts = len(intents)
        print(f" → {n_concepts}概念", end="", flush=True)

        if n_concepts < 2:
            print(" (跳过)")
            continue

        d_values = []
        for Ai, Bi in zip(intents, extents):
            nb = len(Bi)
            d_values.append(len(Ai) / nb if nb > 0 else float("inf"))

        max_d = max(d for d in d_values if d != float("inf") and d < 1e6)
        max_d = max(max_d, 1.0)

        total_a = max(sum(len(Ai) for Ai in intents), 1)
        total_b = max(sum(len(Bi) for Bi in extents), 1)

        heights = [-1] * n_concepts
        edges = []
        for i in range(n_concepts):
            Ai, Bi = intents[i], extents[i]
            for j in range(n_concepts):
                if i == j:
                    continue
                Aj, Bj = intents[j], extents[j]
                if Ai.issuperset(Aj) and Bi.issubset(Bj) and (Ai != Aj or Bi != Bj):
                    is_cover = True
                    for k in range(n_concepts):
                        if k in (i, j):
                            continue
                        Ak, Bk = intents[k], extents[k]
                        if (Ai.issuperset(Ak) and Ak.issuperset(Aj) and
                            Bi.issubset(Bk) and Bk.issubset(Bj) and
                            (Ai != Ak or Bi != Bk) and (Ak != Aj or Bk != Bj)):
                            is_cover = False
                            break
                    if is_cover:
                        edges.append((i, j))

        print(f", {len(edges)}边", end="", flush=True)

        p2c = defaultdict(list)
        c2p = defaultdict(list)
        for pi, ci in edges:
            p2c[pi].append(ci)
            c2p[ci].append(pi)
        parent_map[concept_name] = dict(p2c)

        leaves = [i for i in range(n_concepts) if not p2c.get(i)]
        for leaf in leaves:
            heights[leaf] = 0
        queue = list(leaves)
        while queue:
            node = queue.pop(0)
            h = heights[node]
            for pi, children in p2c.items():
                pi = int(pi) if isinstance(pi, str) else pi
                if node in children and heights[pi] < h + 1:
                    heights[pi] = h + 1
                    queue.append(pi)
        for i in range(n_concepts):
            if heights[i] < 0:
                heights[i] = 0

        concept_params_list = []
        for ci in range(n_concepts):
            Bi = extents[ci]
            if len(Bi) <= 1:
                params = {
                    "alpha1": 1.0, "beta1": 1.0, "gamma1": 1.0, "delta1": 1.0,
                    "zeta1": 1.0, "eta1": 1.0, "theta1": 1.0, "kappa1": 1.0,
                    "kappa2": 1.0, "lambda1": 1.0, "mu1": 1.0, "eps": 0.01,
                }
            else:
                _, _, d_vec, eigvals = build_concept_subgraph(Bi, w_bg_idxs, len(unique))
                params = spectral_params(eigvals, d_vec, len(Bi))
            concept_params_list.append(params)

        v1_results = []
        v3_results = []

        for ci in range(n_concepts):
            Ai, Bi = intents[ci], extents[ci]
            n_a, n_b = len(Ai), len(Bi)

            raw_d = d_values[ci]
            D_init = min(raw_d / max_d, 1.0) if raw_d < 1e6 else 0.8
            D_init = np.clip(D_init, 0.0, 1.0)
            B_init = np.clip(1.0 - n_b / total_b, 0.0, 1.0)
            rho_init = np.clip(n_a / total_a, 0.0, 1.0)
            M0 = np.array([D_init, B_init, rho_init, 0.5, 0.5])

            children = p2c.get(ci, [])
            B_up, rho_up = 0.0, 0.0
            if children:
                for cidx in children:
                    if cidx < len(v1_results):
                        B_up += v1_results[cidx]["M_star"][1]
                        rho_up += v1_results[cidx]["M_star"][2]
                B_up /= len(children)
                rho_up /= len(children)

            r1 = run_n_iteration(M0.copy(), B_up, rho_up, params_uniform)
            r1["concept_idx"] = ci
            r1["name"] = concept_name
            r1["n_children"] = len(children)
            r1["n_a"] = n_a
            r1["n_b"] = n_b
            r1["height"] = heights[ci]
            r1["params"] = "v1_uniform"
            v1_results.append(r1)

            r3 = run_n_iteration(M0.copy(), B_up, rho_up, concept_params_list[ci])
            r3["concept_idx"] = ci
            r3["name"] = concept_name
            r3["n_children"] = len(children)
            r3["n_a"] = n_a
            r3["n_b"] = n_b
            r3["height"] = heights[ci]
            r3["params"] = "v3_spectral"
            v3_results.append(r3)

        all_v1.extend(v1_results)
        all_v3.extend(v3_results)

        for ci, p in enumerate(concept_params_list):
            all_param_snapshots.append({
                "concept": concept_name, "concept_idx": ci,
                "n_a": len(intents[ci]), "n_b": len(extents[ci]),
                "params": {k: float(v) for k, v in p.items() if k != "eps"},
            })

        conv_v1 = sum(1 for r in v1_results if r["converged"])
        conv_v3 = sum(1 for r in v3_results if r["converged"])
        tau_v1 = [r["tau_inv"] for r in v1_results]
        tau_v3 = [r["tau_inv"] for r in v3_results]
        print(f"\n    v1: 收敛{conv_v1}/{n_concepts}, τ⁻¹[{min(tau_v1):.3f},{max(tau_v1):.3f}]")
        print(f"    v3: 收敛{conv_v3}/{n_concepts}, τ⁻¹[{min(tau_v3):.3f},{max(tau_v3):.3f}]")

    def evaluate_hasse(results: List[Dict], label: str) -> Dict:
        idx = {(r["name"], r["concept_idx"]): r for r in results}
        hr = []
        for cn, p2c in parent_map.items():
            for pi_str, children in p2c.items():
                pi = int(pi_str)
                for ci in children:
                    ci = int(ci)
                    p, c = idx.get((cn, pi)), idx.get((cn, ci))
                    if p and c:
                        hr.append({
                            "concept": cn, "parent_idx": pi, "child_idx": ci,
                            "tau_inv_parent": p["tau_inv"], "tau_inv_child": c["tau_inv"],
                            "passes": p["tau_inv"] >= c["tau_inv"],
                            "D_star_parent": p["D_star"], "D_star_child": c["D_star"],
                        })
        passes = sum(1 for h in hr if h["passes"])
        total = len(hr)
        return {"label": label, "total": total, "passes": passes,
                "violations": total - passes,
                "pass_rate": passes / total if total > 0 else 0.0,
                "details": hr[:30]}

    e3_v1 = evaluate_hasse(all_v1, "v1_uniform")
    e3_v3 = evaluate_hasse(all_v3, "v3_spectral")

    print(f"\n{'='*60}")
    print(f"v1（均匀）vs v3（谱导出）对比")
    print(f"{'='*60}")

    for e3 in [e3_v1, e3_v3]:
        print(f"\n--- {e3['label']} ---")
        print(f"总 Hasse 对: {e3['total']}")
        print(f"通过:        {e3['passes']}")
        print(f"违反:        {e3['violations']}")
        print(f"通过率:      {e3['pass_rate']:.1%}")

    delta = e3_v3["pass_rate"] - e3_v1["pass_rate"]
    print(f"\n提升:       {delta:+.1%}")

    d_v1 = [r["D_star"] for r in all_v1]
    d_v3 = [r["D_star"] for r in all_v3]
    t_v1 = [r["tau_inv"] for r in all_v1]
    t_v3 = [r["tau_inv"] for r in all_v3]

    print(f"\n--- 统计对比 ---")
    print(f"D* 均值:   v1={np.mean(d_v1):.4f}  v3={np.mean(d_v3):.4f}")
    print(f"D* std:    v1={np.std(d_v1):.4f}  v3={np.std(d_v3):.4f}")
    print(f"τ⁻¹ 均值:  v1={np.mean(t_v1):.4f}  v3={np.mean(t_v3):.4f}")
    print(f"τ⁻¹ std:   v1={np.std(t_v1):.4f}  v3={np.std(t_v3):.4f}")

    out = {
        "v1_uniform": {"converged_rate": sum(1 for r in all_v1 if r["converged"])/max(len(all_v1),1), "e3": e3_v1},
        "v3_spectral": {"converged_rate": sum(1 for r in all_v3 if r["converged"])/max(len(all_v3),1), "e3": e3_v3},
        "improvement": delta,
        "param_snapshots": all_param_snapshots[:50],
    }

    out_path = results_dir / "e1_spectral.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n结果: {out_path}")


if __name__ == "__main__":
    main()
