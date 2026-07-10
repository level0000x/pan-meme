"""
E-1 v5: 语义驱动的 N-迭代社区可分性
======================================
把 Wikipedia 文本语义特征注入 N-算子系数，
使 M* 携带语义信息 → 检验社区可分性。

管道:
  009 Wikipedia JSON → 语义特征 + FCA格
    → 语义调制 concept_specific_params
    → N-迭代 → M*
    → PCA + nearest-centroid 分类 + k-means 聚类纯度
"""
import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np


# ============================================================================
# Phase 0: 文本预处理
# ============================================================================

def tokenize_english(text):
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


# ============================================================================
# Phase 1: 语义特征提取
# ============================================================================

def extract_semantic_features(text):
    """从 Wikipedia 原始文本提取 6 个语义特征"""
    words = tokenize_english(text)
    total = max(len(words), 1)
    unique = list(set(words))
    n_unique = len(unique)

    word_counts = Counter(words)

    return {
        "vocab_richness": n_unique / total,
        "avg_word_len": np.mean([len(w) for w in unique]) if unique else 0.0,
        "text_len_log": np.log(max(len(text), 10)),
        "rare_ratio": sum(1 for w, c in word_counts.items() if c == 1) / max(total, 1),
        "long_word_ratio": sum(1 for w in unique if len(w) >= 7) / max(n_unique, 1),
        "avg_word_freq": total / max(n_unique, 1),
    }


# ============================================================================
# Phase 2: N-算子（从 n_iteration_v5.py 内联）
# ============================================================================

def n_operator(M, B_up, rho_up, params):
    D, B, rho, R, S = M
    eps = params.get("eps", 0.01)
    a1 = params["alpha1"]
    b1 = params["beta1"]
    g1 = params["gamma1"]
    d1 = params["delta1"]
    z1 = params["zeta1"]
    e1 = params["eta1"]
    t1 = params["theta1"]
    k1 = params["kappa1"]
    k2 = params["kappa2"]
    l1 = params["lambda1"]
    m1 = params["mu1"]

    Delta_D = a1 * R + b1 * (B + B_up) + eps
    Delta_B = g1 * (R + B_up) + d1 * D + eps
    Delta_rho = z1 * (D + rho_up) + e1 * R + eps
    Delta_R = t1 * (rho + rho_up + B_up) + k1 * D + k2 * S + eps
    Delta_S = l1 * D + m1 * R + eps

    D_next = (a1 * R) / Delta_D
    B_next = (g1 * (R + B_up)) / Delta_B
    rho_next = (z1 * (D + rho_up)) / Delta_rho
    R_next = (t1 * (rho + rho_up + B_up)) / Delta_R
    S_next = (l1 * D) / Delta_S

    return np.array([D_next, B_next, rho_next, R_next, S_next])


def compute_jacobian_at_fixed_point(M_star, B_up, rho_up, params):
    D, B, rho, R, S = M_star
    eps = params.get("eps", 0.01)
    a1 = params["alpha1"]
    b1 = params["beta1"]
    g1 = params["gamma1"]
    d1 = params["delta1"]
    z1 = params["zeta1"]
    e1 = params["eta1"]
    t1 = params["theta1"]
    k1 = params["kappa1"]
    k2 = params["kappa2"]
    l1 = params["lambda1"]
    m1 = params["mu1"]

    Delta_D = a1 * R + b1 * (B + B_up) + eps
    Delta_B = g1 * (R + B_up) + d1 * D + eps
    Delta_rho = z1 * (D + rho_up) + e1 * R + eps
    Delta_R = t1 * (rho + rho_up + B_up) + k1 * D + k2 * S + eps
    Delta_S = l1 * D + m1 * R + eps

    J = np.zeros((5, 5), dtype=np.float64)
    J[0, 1] = -b1 * D / Delta_D
    J[0, 3] = +a1 * (1.0 - D) / Delta_D
    J[1, 0] = -d1 * B / Delta_B
    J[1, 3] = +g1 * (1.0 - B) / Delta_B
    J[2, 0] = +z1 * (1.0 - rho) / Delta_rho
    J[2, 3] = -e1 * rho / Delta_rho
    J[3, 0] = -k1 * R / Delta_R
    J[3, 1] = +t1 * (1.0 - R) / Delta_R
    J[3, 2] = +t1 * (1.0 - R) / Delta_R
    J[3, 4] = -k2 * R / Delta_R
    J[4, 0] = +l1 * (1.0 - S) / Delta_S
    J[4, 3] = -m1 * S / Delta_S

    return J


def run_n_iteration(M0, B_up, rho_up, params, max_iter=500, tol=1e-12):
    M = M0.copy()
    for _ in range(max_iter):
        M_next = n_operator(M, B_up, rho_up, params)
        if np.max(np.abs(M_next - M)) < tol:
            M = M_next
            break
        M = M_next
    else:
        return {"converged": False, "M_star": M, "D_star": float(M[0]),
                "rho_spectral": 1.0, "tau_inv": 0.0}

    J = compute_jacobian_at_fixed_point(M, B_up, rho_up, params)
    eigvals = np.linalg.eigvals(J)
    rho_val = float(max(np.abs(eigvals)))
    tau_inv = float(-np.log(max(rho_val, 1e-10)))

    return {
        "converged": True,
        "M_star": M,
        "rho_spectral": rho_val,
        "tau_inv": tau_inv,
        "D_star": float(M[0]),
    }


# ============================================================================
# Phase 3: 结构参数（concept_specific_params，从 n_iteration_v5.py 内联）
# ============================================================================

def concept_specific_params(n_a, n_b, n_children, height,
                            max_a, max_b, max_children, max_height,
                            base=1.5, spread=1.5):
    d_norm = np.clip(n_a / max(max_a, 1), 0.05, 1.0)
    extent_norm = np.clip(n_b / max(max_b, 1), 0.05, 1.0)
    child_norm = np.clip(n_children / max(max_children, 1), 0.0, 1.0)
    depth_norm = np.clip(height / max(max_height, 1), 0.0, 1.0)

    sp = spread
    ba = base

    return {
        "alpha1": ba + sp * (1.0 - d_norm),
        "beta1":  ba + sp * d_norm,
        "gamma1": ba + sp * extent_norm,
        "delta1": ba + sp * d_norm,
        "zeta1":  ba + sp * d_norm,
        "eta1":   ba + sp * child_norm,
        "theta1": ba + sp * (1.0 - child_norm),
        "kappa1": ba + sp * depth_norm,
        "kappa2": ba + sp * (1.0 - depth_norm),
        "lambda1": ba + sp * d_norm,
        "mu1":    ba + sp * child_norm,
        "eps": 0.01,
    }


def compute_hasse_heights(n_concepts, parent_to_children):
    heights = np.full(n_concepts, -1, dtype=int)
    children_of = {}
    for p, ch_list in parent_to_children.items():
        for c in ch_list:
            children_of.setdefault(p, []).append(c)

    def dfs(node):
        if heights[node] >= 0:
            return heights[node]
        ch = children_of.get(node, [])
        if not ch:
            heights[node] = 0
        else:
            heights[node] = 1 + max(dfs(c) for c in ch)
        return heights[node]

    for i in range(n_concepts):
        dfs(i)
    return heights.tolist()


# ============================================================================
# Phase 4: 语义调制
# ============================================================================

def semantic_modulation(structural_params, sem_feats):
    """直接从语义特征生成 N-算子系数。
    不依赖格结构参数——语义是系数的唯一来源。
    """
    p = dict(structural_params)
    rich = sem_feats["vocab_richness"]
    tech = sem_feats["long_word_ratio"]
    rare = sem_feats["rare_ratio"]
    awl = sem_feats["avg_word_len"] / 8.0
    tlog = sem_feats["text_len_log"] / 8.0

    p["alpha1"] = 0.5 + 1.0 * rich
    p["beta1"] = 0.5 + 1.0 * (1.0 - tech)
    p["gamma1"] = 0.5 + 1.0 * (1.0 - rare)
    p["delta1"] = 0.5 + 1.0 * tech
    p["zeta1"] = 0.5 + 1.0 * rich
    p["eta1"] = 0.5 + 1.0 * (1.0 - awl)
    p["theta1"] = 0.5 + 1.0 * tlog
    p["kappa1"] = 0.5 + 1.0 * awl
    p["kappa2"] = 0.5 + 1.0 * (1.0 - awl)
    p["lambda1"] = 0.5 + 1.0 * tech
    p["mu1"] = 0.5 + 1.0 * rare

    return p


# ============================================================================
# Phase 5: 分类标签映射
# ============================================================================

def build_category_map():
    """语义分类（非时序分类）。"""
    return {
        "Newton": "Science", "Darwin": "Science", "DNA": "Science",
        "Evolution": "Science", "Calculus": "Science", "Algebra": "Science",
        "Carbon": "Science", "Oxygen": "Science", "Periodic_table": "Science",
        "Philosophy": "Science", "Photosynthesis": "Science", "Gravity": "Science",

        "ChatGPT": "Technology", "Bitcoin": "Technology",
        "Internet_Explorer": "Technology", "Windows_XP": "Technology",
        "BlackBerry": "Technology", "Nokia": "Technology",
        "MySpace": "Technology", "LimeWire": "Technology",
        "Vine_(service)": "Technology", "Second_Life": "Technology",
        "FarmVille": "Technology", "Clubhouse_(app)": "Technology",
        "Google%2B": "Technology", "Adobe_Flash": "Technology",

        "Barbie_(film)": "Entertainment", "Oppenheimer_(film)": "Entertainment",
        "Taylor_Swift": "Entertainment", "Academy_Awards": "Entertainment",
        "Grammy_Awards": "Entertainment",

        "Democracy": "Politics", "Brexit": "Politics",
        "Black_Lives_Matter": "Politics", "Ukraine": "Politics",

        "COVID-19": "Health",

        "Elon_Musk": "Business", "Tesla,_Inc.": "Business",
        "GameStop_short_squeeze": "Business",

        "Olympic_Games": "Sports", "FIFA_World_Cup": "Sports",
        "Super_Bowl": "Sports",

        "Christmas": "Culture", "Thanksgiving": "Culture",
        "Halloween": "Culture",
    }


# ============================================================================
# Phase 6: 主流水线
# ============================================================================

def main():
    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / "results"
    extract_dir = base_dir.parent / "009-external-validation" / "data" / "extracts"

    lattice_files = sorted(results_dir.glob("*_lattice.json"))
    if not lattice_files:
        print("错误：没有 lattice.json，先运行 fca_lattice.py")
        sys.exit(1)

    print(f"加载 {len(lattice_files)} 个概念格")

    CATEGORIES = build_category_map()
    sem_feats_all = {}

    print("\n" + "=" * 60)
    print("Phase 1: 语义特征提取")
    print("=" * 60)

    for lf in lattice_files:
        concept_name = lf.stem.replace("_lattice", "")

        json_path = extract_dir / f"{concept_name}.json"
        if not json_path.exists():
            print(f"  SKIP {concept_name}: 无 Wikipedia 摘要")
            continue

        with open(json_path, encoding="utf-8") as f:
            wp = json.load(f)

        pages = wp.get("query", {}).get("pages", {})
        page = list(pages.values())[0] if pages else {}
        text = page.get("extract", "")

        if not text:
            print(f"  SKIP {concept_name}: 空摘要")
            continue

        feats = extract_semantic_features(text)
        sem_feats_all[concept_name] = feats
        cat = CATEGORIES.get(concept_name, "Other")
        print(f"  {concept_name:<30s} [{cat:<13s}] "
              f"rich={feats['vocab_richness']:.3f} "
              f"tech={feats['long_word_ratio']:.3f} "
              f"rare={feats['rare_ratio']:.3f}")

    valid_concepts = list(sem_feats_all.keys())
    print(f"\n有效概念: {len(valid_concepts)}")
    print(f"类别分布: {dict(Counter(CATEGORIES.get(c, 'Other') for c in valid_concepts))}")

    if len(valid_concepts) < 15:
        print("概念太少，退出")
        sys.exit(1)

    # 归一化语义特征
    feat_names = ["vocab_richness", "avg_word_len", "text_len_log",
                  "rare_ratio", "long_word_ratio", "avg_word_freq"]
    X_sem_raw = np.array([[sem_feats_all[c][k] for k in feat_names]
                          for c in valid_concepts])
    X_sem = (X_sem_raw - X_sem_raw.mean(0)) / (X_sem_raw.std(0) + 1e-12)

    # PCA on semantic features to understand their structure
    U_sem, S_sem, Vt_sem = np.linalg.svd(X_sem, full_matrices=False)
    var_sem = S_sem**2 / np.sum(S_sem**2)
    print(f"\n语义特征 PCA: PC1={var_sem[0]:.3f} PC2={var_sem[1]:.3f} PC3={var_sem[2]:.3f}")
    for j in range(min(3, len(feat_names))):
        top = np.argsort(np.abs(Vt_sem[j]))[::-1][:3]
        top_names = [f"{feat_names[k]}={Vt_sem[j,k]:+.3f}" for k in top]
        print(f"  PC{j+1}: {' | '.join(top_names)}")

    print("\n" + "=" * 60)
    print("Phase 2: N-迭代（语义调制参数）")
    print("=" * 60)

    concept_Mstar = {}
    concept_stats = {}

    skipped_bad = 0
    for lf in lattice_files:
        concept_name = lf.stem.replace("_lattice", "")
        if concept_name not in sem_feats_all:
            continue

        with open(lf, encoding="utf-8") as f:
            lattice = json.load(f)

        if "concept_sizes" not in lattice or "d_values" not in lattice or "edges" not in lattice:
            skipped_bad += 1
            continue

        n_concepts = lattice["n_concepts"]
        d_values = lattice.get("d_values", [])
        edges = lattice.get("edges", [])
        size_info = lattice.get("concept_sizes", [])

        parent_to_children = defaultdict(list)
        for pi, ci in edges:
            parent_to_children[pi].append(ci)

        if n_concepts == 0:
            continue

        heights = compute_hasse_heights(n_concepts, dict(parent_to_children))
        max_h = max(heights) if heights else 1
        max_a = max((s.get("|A|", 1) for s in size_info), default=1)
        max_b = max((s.get("|B|", 1) for s in size_info), default=1)
        max_ch = max((len(parent_to_children.get(i, [])) for i in range(n_concepts)), default=1)
        total_extent = max(sum(s.get("|B|", 1) for s in size_info), 1)
        total_intent = max(sum(s.get("|A|", 1) for s in size_info), 1)
        max_d = max([d for d in d_values if d != float("inf") and d < 1e6]) if d_values else 1.0
        max_d = max(max_d, 1.0)

        sem_feats = sem_feats_all[concept_name]

        all_results = []
        for ci in range(n_concepts):
            si = size_info[ci] if ci < len(size_info) else {"|A|": 1, "|B|": 1}
            n_a, n_b = si["|A|"], si["|B|"]
            n_ch = len(parent_to_children.get(ci, []))

            raw_d = d_values[ci] if ci < len(d_values) else 1.0
            D_init = min(raw_d / max_d, 1.0) if raw_d < 1e6 else 0.8
            D_init = np.clip(float(D_init), 0.0, 1.0)

            B_init = np.clip(1.0 - n_b / total_extent, 0.0, 1.0)
            rho_init = np.clip(n_a / total_intent, 0.0, 1.0)
            M0 = np.array([D_init, B_init, rho_init, 0.5, 0.5])

            struct_params = concept_specific_params(
                n_a, n_b, n_ch, heights[ci],
                max_a, max_b, max_ch, max_h,
            )

            sem_params = semantic_modulation(struct_params, sem_feats)

            children = parent_to_children.get(ci, [])
            B_up, rho_up = 0.0, 0.0
            if children:
                vals_B, vals_r = [], []
                for cidx in children:
                    ckey = f"{concept_name}_{cidx}"
                    if ckey in concept_Mstar:
                        vals_B.append(concept_Mstar[ckey][0])
                        vals_r.append(concept_Mstar[ckey][1])
                if vals_B:
                    B_up = np.mean(vals_B)
                    rho_up = np.mean(vals_r)

            r = run_n_iteration(M0, B_up, rho_up, sem_params)
            r["cidx"] = ci
            r["name"] = concept_name
            concept_Mstar[f"{concept_name}_{ci}"] = (float(r["M_star"][1]),
                                                      float(r["M_star"][2]))
            all_results.append(r)

        D_vals = [r["D_star"] for r in all_results]
        tau_vals = [r["tau_inv"] for r in all_results]
        rho_vals = [r["rho_spectral"] for r in all_results]

        concept_stats[concept_name] = {
            "D_star": D_vals,
            "rho": rho_vals,
            "tau_inv": tau_vals,
        }

        cat = CATEGORIES.get(concept_name, "Other")
        if n_concepts <= 3:
            print(f"  {concept_name:<30s} [{cat:<13s}] "
                  f"D*=[{min(D_vals):.3f},{max(D_vals):.3f}] "
                  f"tau=[{min(tau_vals):.3f},{max(tau_vals):.3f}]")

    # Summary
    all_D = [v for s in concept_stats.values() for v in s["D_star"]]
    all_tau = [v for s in concept_stats.values() for v in s["tau_inv"]]
    print(f"\n总计: {len(concept_stats)} 概念, "
          f"过滤掉 {skipped_bad} 个坏格, "
          f"D*∈[{min(all_D):.4f},{max(all_D):.4f}] "
          f"τ⁻¹∈[{min(all_tau):.4f},{max(all_tau):.4f}]")
    print(f"D* std={np.std(all_D):.4f}, τ⁻¹ std={np.std(all_tau):.4f}")

    print("\n" + "=" * 60)
    print("Phase 3: 社区可分性检验")
    print("=" * 60)

    names = []
    y_str = []
    feats = []
    for name in valid_concepts:
        if name not in concept_stats:
            continue
        cat = CATEGORIES.get(name, "Other")
        s = concept_stats[name]
        D_arr = np.array(s["D_star"])
        r_arr = np.array(s["rho"])
        t_arr = np.array(s["tau_inv"])
        f = np.array([
            D_arr.mean(), D_arr.min(), D_arr.max(), D_arr.std(),
            r_arr.mean(), r_arr.min(), r_arr.max(), r_arr.std(),
            t_arr.mean(), t_arr.min(), t_arr.max(), t_arr.std(),
        ])
        names.append(name)
        y_str.append(cat)
        feats.append(f)

    X = np.array(feats)
    cats_list = sorted(set(y_str))
    y = np.array([cats_list.index(c) for c in y_str])
    n_cl = len(cats_list)
    chance = 1.0 / n_cl

    print(f"概念: {len(names)}, 特征: {X.shape[1]}, 类别: {n_cl}")
    print(f"类别分布: {dict(Counter(y_str))}")

    Xc = (X - X.mean(0)) / (X.std(0) + 1e-12)

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var_r = S[:3]**2 / np.sum(S**2)
    print(f"\nM* 特征 PCA: PC1={var_r[0]:.3f} PC2={var_r[1]:.3f} PC3={var_r[2]:.3f}")

    X_pca = U[:, :2] * S[:2]

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(y))
    N = len(y)
    K = 5
    preds = np.zeros(N, int)
    for fold in range(K):
        s, e = fold * N // K, min((fold + 1) * N // K, N)
        tst = np.zeros(N, bool)
        tst[s:e] = True
        trn = ~tst
        Xtr, Xte = Xc[trn], Xc[tst]
        cents = np.zeros((n_cl, Xtr.shape[1]))
        for ci in range(n_cl):
            mask = (y[trn] == ci)
            if mask.sum() > 0:
                cents[ci] = Xtr[mask].mean(0)
            else:
                cents[ci] = np.inf
        for i, j in enumerate(np.where(tst)[0]):
            d2 = np.sum((Xte[i] - cents)**2, axis=1)
            preds[j] = np.argmin(np.nan_to_num(d2, nan=1e10))

    acc = np.mean(preds == y)
    print(f"\nNearest-centroid 5-fold: {acc:.3f} (chance={chance:.3f}, {acc/chance:.1f}x)")

    cents = np.array([Xc[y == ci].mean(0) for ci in range(n_cl)])
    for _ in range(20):
        labs = np.argmin(np.sum((Xc[:, None, :] - cents[None, :, :])**2, axis=2), axis=1)
        for ci in range(n_cl):
            mask = labs == ci
            if mask.sum():
                cents[ci] = Xc[mask].mean(0)

    purities = []
    for cl in range(n_cl):
        mask = labs == cl
        if mask.sum() == 0:
            continue
        cnt = Counter(y[mask])
        dcat, dc = cnt.most_common(1)[0]
        purities.append(dc / mask.sum())
        print(f"  Cluster {cl}: {mask.sum()} samples, {cats_list[dcat]} purity={purities[-1]:.2f}")

    avg_pur = np.mean(purities) if purities else 0
    print(f"Avg purity: {avg_pur:.3f} (chance={chance:.3f}, {avg_pur/chance:.1f}x)")

    print(f"\nPer-category:")
    for ci, cname in enumerate(cats_list):
        mask = y == ci
        if mask.sum() == 0:
            continue
        ca = np.mean(preds[mask] == ci)
        pm = preds == ci
        pr = np.mean(y[pm] == ci) if pm.sum() > 0 else 0
        print(f"  {cname:15s} n={mask.sum():2d} acc={ca:.2f} prec={pr:.2f}")

    print(f"\nPCA 坐标（前 20 个概念）:")
    for i in range(min(20, len(names))):
        print(f"  {names[i]:<30s} {y_str[i]:<13s} "
              f"PC1={X_pca[i, 0]:>7.2f} PC2={X_pca[i, 1]:>7.2f}")

    top_loadings = np.argsort(np.abs(Vt[0]))[::-1][:5]
    feat_names = ["D*_mean", "D*_min", "D*_max", "D*_std",
                  "rho_mean", "rho_min", "rho_max", "rho_std",
                  "tau_mean", "tau_min", "tau_max", "tau_std"]
    print(f"\nTop PC1 loadings:")
    for k in top_loadings:
        print(f"  {feat_names[k]:12s} loading={Vt[0, k]:+.3f}")

    print(f"\n{'=' * 50}")
    print(f"VERDICT")
    print(f"{'=' * 50}")
    print(f"  Semantic-modulated N-iteration M* features:")
    print(f"  M* PCA: PC1={var_r[0]:.3f} PC2={var_r[1]:.3f} PC3={var_r[2]:.3f}")
    print(f"  Classification: {acc:.3f} (chance={chance:.3f}, {acc/chance:.1f}x)")
    print(f"  Cluster purity: {avg_pur:.3f} (chance={chance:.3f}, {avg_pur/chance:.1f}x)")
    print(f"  D* std = {np.std(all_D):.4f} (v5_d_monotonic = 0.1431)")

    if var_r[0] < 0.95 and acc > chance * 2.0:
        print(f"  PASS: 语义调制成功 → M* 携带社区信息")
    elif var_r[0] >= 0.95:
        print(f"  DEGENERATE: M* 仍退化 → 语义调制强度不足或方向不对")
    else:
        print(f"  WEAK: 有微弱信号但不够显著")
        print(f"  建议: 增大调制强度(scales), 尝试不同系数组合, "
              f"引入词向量语义距离")


if __name__ == "__main__":
    main()
