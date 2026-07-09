"""
实验 011 — v4 FCA闭包图 + 谱导出参数

与 v3 的关键区别：词间边权使用 FCA 闭包交集大小，
而非 bigram 共现次数。

w_i^↑↓ = 在所有包含 w_i 中任一字符的的词中均出现的字符集
edge(w_i, w_j) = |w_i^↑↓ ∩ w_j^↑↓|

这保证了：超概念的外延词拥有更丰富的 FCA 闭包 →
子图 Laplacian 谱天然不同 → 参数区分为 → E-3 可能改善。
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


def build_formal_context(words: List[str]) -> Tuple[
    List[str], List[str],
    Dict[str, Set[str]], Dict[str, Set[str]],
    Dict[str, int], Dict[str, int],
]:
    """返回 (unique_words, chars, char_to_words, word_to_chars, char_idx, word_idx)。"""
    word_counts = defaultdict(int)
    for w in words:
        word_counts[w] += 1
    unique = [w for w, c in word_counts.items() if len(w) >= 4]

    char_set = set()
    for w in unique:
        char_set.update(w)
    chars = sorted(char_set)

    char_to_words: Dict[str, Set[str]] = defaultdict(set)
    word_to_chars: Dict[str, Set[str]] = {}
    for w in unique:
        wc = set(w)
        word_to_chars[w] = wc
        for c in wc:
            if c in char_set:
                char_to_words[c].add(w)

    char_idx = {c: i for i, c in enumerate(chars)}
    word_idx = {w: i for i, w in enumerate(unique)}
    return unique, chars, char_to_words, word_to_chars, char_idx, word_idx


def build_fca_closure_graph(
    unique_words: List[str],
    word_to_chars: Dict[str, Set[str]],
    char_to_words: Dict[str, Set[str]],
) -> np.ndarray:
    """构建 FCA 闭包交集图（美证版 §1.4）。

    词-词边权 = |w_i^↓↑ ∩ w_j^↓↑|
    w_i^↓↑ = {w_k : chars(w_i) ⊆ chars(w_k)}
    即包含 w_i 全部字符的所有词的集合。

    超概念外延中的词内涵更大 → ↓↑ 闭包更小 → 边权更低
    子概念外延中的词内涵更小 → ↓↑ 闭包更大 → 边权更高
    这一天然差异是 v4 的关键假设。
    """
    n = len(unique_words)
    down_up: List[Set[int]] = []

    for i, w in enumerate(unique_words):
        wc = word_to_chars.get(w, set())
        if not wc:
            down_up.append(set())
            continue
        du = set(range(n))
        for c in wc:
            du &= {j for j, w2 in enumerate(unique_words) if c in word_to_chars.get(w2, set())}
        down_up.append(du)

    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        di = down_up[i]
        if not di:
            continue
        for j in range(i + 1, n):
            dj = down_up[j]
            if not dj:
                continue
            shared = len(di & dj)
            if shared > 0:
                A[i, j] = shared
                A[j, i] = shared

    return A


def compute_concepts(
    unique_words: List[str],
    char_to_words: Dict[str, Set[str]],
    word_to_chars: Dict[str, Set[str]],
    max_concepts: int = 2000,
    time_limit: float = 60.0,
) -> Tuple[List[Set[str]], List[Set[str]]]:
    """NextClosure 算法，用词索引代替字符索引（对象为词，属性为字符）。"""
    chars = sorted({c for w in unique_words for c in w})
    char_idx = {c: i for i, c in enumerate(chars)}
    n_chars = len(chars)
    n_words = len(unique_words)

    bg_to_w: Dict[int, Set[int]] = defaultdict(set)
    w_to_bg: Dict[int, Set[int]] = {}
    for wi, w in enumerate(unique_words):
        idxs = {char_idx[c] for c in w if c in char_idx}
        w_to_bg[wi] = idxs
        for ci in idxs:
            bg_to_w[ci].add(wi)

    def closure(s: frozenset) -> frozenset:
        ws = set(range(n_words))
        for bi in s:
            ws &= bg_to_w.get(bi, set())
            if not ws:
                break
        if not ws:
            return frozenset(range(n_chars))
        cs = set(range(n_chars))
        for wi in ws:
            cs &= w_to_bg.get(wi, set())
            if not cs:
                break
        return frozenset(cs)

    t0 = time.time()
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
        for i in range(n_chars - 1, -1, -1):
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

    result_intents = []
    result_extents = []
    for intent, extent in zip(intents, extents):
        intent_names = {chars[i] for i in intent if i < len(chars)}
        extent_names = {unique_words[i] for i in extent if i < len(unique_words)}
        result_intents.append(intent_names)
        result_extents.append(extent_names)

    return result_intents, result_extents


def concept_subgraph_laplacian(
    word_indices: Set[int],
    global_A: np.ndarray,
    n_total: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从全局 FCA 闭包图中提取概念外延子图的归一化拉普拉斯及其谱。"""
    wlist = sorted(word_indices)
    m = len(wlist)
    if m <= 1:
        return np.eye(1), np.zeros(1), np.array([0.0])

    idx_map = {old: new for new, old in enumerate(wlist)}
    A_sub = np.zeros((m, m), dtype=np.float64)
    for i_old in wlist:
        for j_old in wlist:
            if i_old < j_old:
                w = global_A[i_old, j_old]
                if w > 0:
                    i_new, j_new = idx_map[i_old], idx_map[j_old]
                    A_sub[i_new, j_new] = w
                    A_sub[j_new, i_new] = w

    d = np.asarray(A_sub.sum(axis=1)).flatten()
    d_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    L = np.eye(m) - np.diag(d_sqrt) @ A_sub @ np.diag(d_sqrt)
    eigvals = np.linalg.eigvalsh(L)

    return np.diag(d_sqrt), d, eigvals


def spectral_params(
    L_eigvals: np.ndarray,
    d: np.ndarray,
    n_words: int,
) -> Dict[str, float]:
    """从归一化拉普拉斯谱导出 11 个 N-迭代参数。"""
    m = len(L_eigvals)
    if m <= 1:
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
    t_star = np.log(beta0 + 1) / lambda1

    def theta(t: float) -> float:
        vals = np.exp(-t * L_eigvals)
        return float(np.sum(vals[np.isfinite(vals)]))

    Th = theta(t_star / 2)
    T2 = theta(2 * t_star)
    T3 = theta(3 * t_star)

    all_p = np.array([
        lambda1, lambda_max,
        Th / max(n_words, 1), lambda1,
        Th / max(n_words, 1),
        max(lambda_max - lambda1, 0.1),
        Th / max(beta0, 1),
        lambda_max,
        lambda1 / max(lambda_max, 0.1),
        T2 / max(beta0, 1),
        1.0 - beta0 / max(n_words, 1),
    ])

    lo, hi = 0.5, 3.0
    p_min, p_max = all_p.min(), all_p.max()
    all_p = lo + (hi - lo) * (all_p - p_min) / (p_max - p_min) if p_max > p_min else np.full_like(all_p, (lo + hi) / 2)

    names = ["alpha1", "beta1", "gamma1", "delta1",
             "zeta1", "eta1", "theta1",
             "kappa1", "kappa2", "lambda1", "mu1"]
    return {n: float(v) for n, v in zip(names, all_p)} | {"eps": 0.01}


def n_operator(M: np.ndarray, B_up: float, rho_up: float, p: Dict[str, float]) -> np.ndarray:
    D, B, rho, R, S = M
    e = p.get("eps", 0.01)
    a1, b1 = p["alpha1"], p["beta1"]
    g1, d1 = p["gamma1"], p["delta1"]
    z1, e1 = p["zeta1"], p["eta1"]
    t1 = p["theta1"]
    k1, k2 = p["kappa1"], p["kappa2"]
    l1, m1 = p["lambda1"], p["mu1"]
    N_D = (a1 * R + e) / (a1 * R + b1 * (B + B_up) + e)
    N_B = (g1 * (R + B_up) + e) / (g1 * (R + B_up) + d1 * D + e)
    N_rho = (z1 * (D + rho_up) + e) / (z1 * (D + rho_up) + e1 * R + e)
    N_R = (t1 * (rho + rho_up + B_up) + e) / (t1 * (rho + rho_up + B_up) + k1 * D + k2 * S + e)
    N_S = (l1 * D + e) / (l1 * D + m1 * R + e)
    return np.array([N_D, N_B, N_rho, N_R, N_S])


def run_n(M0: np.ndarray, B_up: float, rho_up: float, p: Dict[str, float]) -> Dict:
    traj = [M0.copy()]
    M = M0.copy()
    converged, conv_iter = False, 500
    for k in range(500):
        Mn = n_operator(M, B_up, rho_up, p)
        traj.append(Mn.copy())
        if np.linalg.norm(Mn - M) < 1e-8:
            converged, conv_iter = True, k + 1
            M = Mn
            break
        M = Mn
    M_star = M
    traj_np = np.array(traj)
    if converged and conv_iter >= 5:
        deltas = np.array([np.linalg.norm(traj_np[i+1]-traj_np[i]) for i in range(conv_iter-2)])
        if len(deltas) >= 3:
            ratios = deltas[1:] / (deltas[:-1] + 1e-16)
            valid = ratios[(ratios > 0) & (ratios < 1)]
            rho_val = float(np.median(valid[-10:])) if len(valid) else 0.5
        else:
            rho_val = 0.5
    else:
        rho_val = 0.5
    tau_inv = float(-np.log(np.clip(rho_val, 0.001, 0.999)))
    Ds, _, _, Rs, Ss = M_star
    wd = 0.0
    for i in range(len(traj_np)-1):
        Di, _, _, _, Si = traj_np[i]; Dj, _, _, _, Sj = traj_np[i+1]
        wd += abs(Dj*(1.0-Sj)-Di*(1.0-Si))
    return {
        "converged": converged, "conv_iter": conv_iter,
        "M_star": [float(v) for v in M_star],
        "tau_inv": tau_inv,
        "Phi_star": float(Ds*(1.0-Ss)),
        "W_diss": float(wd),
        "D_init": float(M0[0]), "D_star": float(Ds),
    }


def main():
    base_dir = Path(__file__).resolve().parent.parent
    project_root = base_dir.parent.parent
    extract_dir = project_root / "experiments" / "009-external-validation" / "data" / "extracts"
    if not extract_dir.exists():
        print(f"数据目录不存在: {extract_dir}"); sys.exit(1)

    extract_files = sorted(extract_dir.glob("*.json"))
    print(f"加载 {len(extract_files)} 个 Wikipedia 摘要\n")

    p_uniform = {"alpha1":1.0,"beta1":1.0,"gamma1":1.0,"delta1":1.0,"zeta1":1.0,
                 "eta1":1.0,"theta1":1.0,"kappa1":1.0,"kappa2":1.0,"lambda1":1.0,"mu1":1.0,"eps":0.01}
    all_v1, all_v4 = [], []
    parent_map: Dict[str, Dict[int, List[int]]] = {}

    for fpath in extract_files:
        cn = fpath.stem
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        pages = data.get("query", {}).get("pages", {})
        if not pages: continue
        page = next(iter(pages.values()))
        text = page.get("extract", "")
        if not text or len(text) < 300: continue
        words = tokenize_english(text)
        if len(words) < 30: continue

        unique, chars, c2w, w2c, ci, wi = build_formal_context(words)
        if len(chars) < 5 or len(unique) < 10: continue

        A_global = build_fca_closure_graph(unique, w2c, c2w)
        total_edges = int(np.sum(A_global > 0) / 2)
        print(f"  {cn}: {len(unique)}词, {len(chars)}字符, {total_edges}边", end="", flush=True)

        intents, extents = compute_concepts(unique, c2w, w2c)
        n_c = len(intents)
        print(f" → {n_c}概念", end="", flush=True)
        if n_c < 2: print(" (跳过)"); continue

        d_vals = [len(Ai)/len(Bi) if len(Bi) else float("inf") for Ai, Bi in zip(intents, extents)]
        max_d = max(d for d in d_vals if d != float("inf") and d < 1e6)
        max_d = max(max_d, 1.0)
        total_b = max(sum(len(Bi) for Bi in extents), 1)
        total_a = max(sum(len(Ai) for Ai in intents), 1)

        edges = []
        for i in range(n_c):
            Ai, Bi = intents[i], extents[i]
            for j in range(n_c):
                if i == j: continue
                Aj, Bj = intents[j], extents[j]
                if Ai.issuperset(Aj) and Bi.issubset(Bj) and (Ai != Aj or Bi != Bj):
                    is_cover = True
                    for k in range(n_c):
                        if k in (i, j): continue
                        Ak, Bk = intents[k], extents[k]
                        if (Ai.issuperset(Ak) and Ak.issuperset(Aj) and
                            Bi.issubset(Bk) and Bk.issubset(Bj) and
                            (Ai != Ak or Bi != Bk) and (Ak != Aj or Bk != Bj)):
                            is_cover = False; break
                    if is_cover: edges.append((i, j))
        print(f", {len(edges)}边", end="", flush=True)

        p2c = defaultdict(list)
        for pi, ci in edges: p2c[pi].append(ci)
        parent_map[cn] = dict(p2c)

        word_to_idx = {w: i for i, w in enumerate(unique)}

        concept_params_v4 = []
        for ci in range(n_c):
            Bi = extents[ci]
            bidxs = {word_to_idx[w] for w in Bi if w in word_to_idx}
            if len(bidxs) <= 1:
                concept_params_v4.append(p_uniform.copy())
            else:
                _, d_vec, eigvals = concept_subgraph_laplacian(bidxs, A_global, len(unique))
                concept_params_v4.append(spectral_params(eigvals, d_vec, len(Bi)))

        v1r, v4r = [], []
        for ci in range(n_c):
            Ai, Bi = intents[ci], extents[ci]
            n_a, n_b = len(Ai), len(Bi)
            raw_d = d_vals[ci]
            D_init = min(raw_d/max_d, 1.0) if raw_d < 1e6 else 0.8
            D_init = np.clip(D_init, 0.0, 1.0)
            B_init = np.clip(1.0 - n_b/total_b, 0.0, 1.0)
            rho_init = np.clip(n_a/total_a, 0.0, 1.0)
            M0 = np.array([D_init, B_init, rho_init, 0.5, 0.5])
            children = p2c.get(ci, [])
            bu, ru = 0.0, 0.0
            if children:
                for cidx in children:
                    if cidx < len(v1r):
                        bu += v1r[cidx]["M_star"][1]; ru += v1r[cidx]["M_star"][2]
                bu /= len(children); ru /= len(children)

            r1 = run_n(M0.copy(), bu, ru, p_uniform)
            r1["cidx"], r1["name"] = ci, cn; v1r.append(r1)
            r4 = run_n(M0.copy(), bu, ru, concept_params_v4[ci])
            r4["cidx"], r4["name"] = ci, cn; v4r.append(r4)

        all_v1.extend(v1r); all_v4.extend(v4r)
        print(f"\n    v1: {sum(1 for r in v1r if r['converged'])}/{n_c}, "
              f"τ⁻¹[{min(r['tau_inv'] for r in v1r):.3f},{max(r['tau_inv'] for r in v1r):.3f}]")
        print(f"    v4: {sum(1 for r in v4r if r['converged'])}/{n_c}, "
              f"τ⁻¹[{min(r['tau_inv'] for r in v4r):.3f},{max(r['tau_inv'] for r in v4r):.3f}]")

    def eval_hasse(results, label):
        idx = {(r["name"], r["cidx"]): r for r in results}
        hr = []
        for cn, p2c in parent_map.items():
            for ps, children in p2c.items():
                pi = int(ps)
                for ci in children:
                    p, c = idx.get((cn, pi)), idx.get((cn, int(ci)))
                    if p and c:
                        hr.append({
                            "concept": cn, "parent_idx": pi, "child_idx": ci,
                            "tau_inv_parent": p["tau_inv"], "tau_inv_child": c["tau_inv"],
                            "passes": p["tau_inv"] >= c["tau_inv"],
                        })
        total = len(hr)
        passes = sum(1 for h in hr if h["passes"])
        return {"label": label, "total": total, "passes": passes,
                "violations": total - passes,
                "pass_rate": passes / total if total else 0.0}

    e3v1 = eval_hasse(all_v1, "v1_uniform")
    e3v4 = eval_hasse(all_v4, "v4_fca_closure")

    print(f"\n{'='*60}")
    print(f"v1（均匀）vs v4（FCA闭包图+谱导出）")
    print(f"{'='*60}")
    for e3 in [e3v1, e3v4]:
        print(f"\n--- {e3['label']} ---")
        print(f"总数: {e3['total']}  通过: {e3['passes']}  违反: {e3['violations']}  通过率: {e3['pass_rate']:.1%}")
    delta = e3v4["pass_rate"] - e3v1["pass_rate"]
    print(f"\n提升: {delta:+.1%}")

    dv1 = [r["D_star"] for r in all_v1]; dv4 = [r["D_star"] for r in all_v4]
    tv1 = [r["tau_inv"] for r in all_v1]; tv4 = [r["tau_inv"] for r in all_v4]
    print(f"\nD*: v1={np.mean(dv1):.4f}±{np.std(dv1):.4f}  v4={np.mean(dv4):.4f}±{np.std(dv4):.4f}")
    print(f"τ⁻¹: v1={np.mean(tv1):.4f}±{np.std(tv1):.4f}  v4={np.mean(tv4):.4f}±{np.std(tv4):.4f}")

    out = {"v1": e3v1, "v4": e3v4, "improvement": delta}
    out_path = base_dir / "results" / "e1_v4_fca_closure.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果: {out_path}")


if __name__ == "__main__":
    main()
