"""
实验 011 — N-迭代引擎 v2（概念特化参数）

M^{(k+1)} = N(M^{(k)})

每个 FCA 概念的 |A|, |B|, 孩子数, Hasse高度 → 11 个耦合系数
不再使用均匀参数。

验证：
  E-1：收敛性、作用量谱
  E-3：Hasse 覆盖对上的 tau_parent^{-1} >= tau_child^{-1}
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def compute_hasse_heights(
    n_concepts: int,
    parent_to_children: Dict[int, List[int]],
) -> List[int]:
    """计算每个概念在 Hasse 图中的高度（从叶子出发的最长路径）。

    叶子节点（无孩子）= 高度 0，向上每层 +1。
    如果形成 DAG，高度 = 到最深叶子的距离。
    """
    height = [-1] * n_concepts
    leaves = [i for i in range(n_concepts) if not parent_to_children.get(i)]

    for leaf in leaves:
        height[leaf] = 0

    queue = list(leaves)
    while queue:
        node = queue.pop(0)
        h = height[node]
        for pi, children in parent_to_children.items():
            pi = int(pi)
            if node in children and height[pi] < h + 1:
                height[pi] = h + 1
                queue.append(pi)

    for i in range(n_concepts):
        if height[i] < 0:
            height[i] = 0

    return height


def concept_params(
    n_a: int,
    n_b: int,
    n_children: int,
    height: int,
    max_a: int,
    max_b: int,
    max_children: int,
    max_height: int,
    base_lo: float = 1.0,
    param_range: float = 2.0,
) -> Dict[str, float]:
    """将 FCA 概念结构映射为 N 算子的 11 个耦合系数。

    经济含义：
      alpha1  — R压制B:       高 D → 更大 → D 衰减慢
      beta1   — B压制D:       低 D → 更大 → B 反压 D 强
      gamma1  — (R+B_up)增长B: 外延大 → 更大 → 关联增长快
      delta1  — D压制B:       高 D → 更大 → D 受 B 侵蚀少
      zeta1   — (D+rho_up)增长ρ: 高 D → 更大 → 能流注入多
      eta1    — R压制ρ:       叶子小(solo) / 非叶子大(下拉)
      theta1  — 组合增长R:    非叶子大 → R 靠下层注入增长
      kappa1  — D压制R:       深层大 → D 对 R 约束强
      kappa2  — S压制R:       浅层大 → S 对 R 约束强
      lambda1 — D增长S:       高 D → 更大 → 韧度增长快
      mu1     — R压制S:       叶子小 → 演化对韧度侵蚀弱
    """
    max_b = max(max_b, 1)
    max_a_val = max(max_a, 1)
    max_children_val = max(max_children, 1)
    max_height_val = max(max_height, 1)

    d_val = np.clip(n_a / max_a_val, 0.0, 1.0)
    extent_norm = np.clip(n_b / max_b, 0.0, 1.0)
    child_norm = np.clip(n_children / max_children_val, 0.0, 1.0)
    leaf_factor = 0.0 if n_children == 0 else child_norm
    depth_norm = np.clip(height / max_height_val, 0.0, 1.0)

    r = param_range

    return {
        "alpha1":  base_lo + r * d_val,
        "beta1":   base_lo + r * (1.0 - d_val),
        "gamma1":  base_lo + r * extent_norm,
        "delta1":  base_lo + r * d_val,
        "zeta1":   base_lo + r * d_val,
        "eta1":    base_lo + r * leaf_factor,
        "theta1":  base_lo + r * (1.0 - leaf_factor),
        "kappa1":  base_lo + r * depth_norm,
        "kappa2":  base_lo + r * (1.0 - depth_norm),
        "lambda1": base_lo + r * d_val,
        "mu1":     base_lo + r * leaf_factor,
        "eps": 0.01,
    }


def n_operator(
    M: np.ndarray,
    B_up: float,
    rho_up: float,
    params: Dict[str, float],
) -> np.ndarray:
    """定义 6.12 的 N 算子（概念特化参数版）。

    M = [D, B, rho, R, S]  (5维，∈ [0,1])
    """
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

    N_D = (a1 * R + eps) / (a1 * R + b1 * (B + B_up) + eps)
    N_B = (g1 * (R + B_up) + eps) / (g1 * (R + B_up) + d1 * D + eps)
    N_rho = (z1 * (D + rho_up) + eps) / (z1 * (D + rho_up) + e1 * R + eps)
    N_R = (t1 * (rho + rho_up + B_up) + eps) / (t1 * (rho + rho_up + B_up) + k1 * D + k2 * S + eps)
    N_S = (l1 * D + eps) / (l1 * D + m1 * R + eps)

    return np.array([N_D, N_B, N_rho, N_R, N_S])


def run_n_iteration(
    M0: np.ndarray,
    B_up: float,
    rho_up: float,
    params: Dict[str, float],
    max_iter: int = 500,
    tol: float = 1e-8,
) -> Dict:
    trajectory = [M0.copy()]
    M = M0.copy()
    converged = False
    conv_iter = max_iter

    for k in range(max_iter):
        M_next = n_operator(M, B_up, rho_up, params)
        trajectory.append(M_next.copy())
        delta = np.linalg.norm(M_next - M)
        if delta < tol:
            converged = True
            conv_iter = k + 1
            M = M_next
            break
        M = M_next

    M_star = M
    trajectory_np = np.array(trajectory)

    if converged and conv_iter >= 5:
        deltas = np.array([
            np.linalg.norm(trajectory_np[i + 1] - trajectory_np[i])
            for i in range(conv_iter - 2)
        ])
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
    Phi_star = D_star * (1.0 - S_star)

    w_diss = 0.0
    for i in range(len(trajectory_np) - 1):
        Di, _, _, _, Si = trajectory_np[i]
        Dj, _, _, _, Sj = trajectory_np[i + 1]
        w_diss += abs(Dj * (1.0 - Sj) - Di * (1.0 - Si))

    return {
        "converged": bool(converged),
        "conv_iter": int(conv_iter),
        "M_star": [float(v) for v in M_star],
        "rho_spectral": float(rho_val),
        "tau_inv": tau_inv,
        "Phi_star": float(Phi_star),
        "W_diss": float(w_diss),
        "D_init": float(M0[0]),
        "D_star": float(D_star),
    }


def run_v1_uniform(
    concept_name: str,
    lattice: Dict,
    parent_to_children: Dict[int, List[int]],
    n_concepts: int,
    n_words: int,
    d_values: List[float],
    results_dir: Path,
) -> Tuple[List[Dict], List[List]]:
    """v1: 均匀参数（coeff=1.0），作为对照。"""
    params_uniform = {"eps": 0.01}
    for k in ["alpha1", "beta1", "gamma1", "delta1", "zeta1",
              "eta1", "theta1", "kappa1", "kappa2", "lambda1", "mu1"]:
        params_uniform[k] = 1.0

    size_info_list = lattice.get("concept_sizes", [])
    total_extent = max(sum(s.get("|B|", 1) for s in size_info_list), 1)
    total_intent = max(sum(s.get("|A|", 1) for s in size_info_list), 1)

    results = []
    max_d = max([d for d in d_values if d != float("inf") and d < 1e6]) if d_values else 1.0
    max_d = max(max_d, 1.0)

    for ci in range(n_concepts):
        si = size_info_list[ci] if ci < len(size_info_list) else {"|A|": 1, "|B|": 1}
        n_a, n_b = si["|A|"], si["|B|"]

        if ci < len(d_values) and max_d > 0:
            raw_d = d_values[ci]
            D_init = min(raw_d / max_d, 1.0) if raw_d < 1e6 else 0.8
        else:
            D_init = 0.5
        D_init = np.clip(D_init, 0.0, 1.0)

        B_init = np.clip(1.0 - n_b / total_extent, 0.0, 1.0)
        rho_init = np.clip(n_a / total_intent, 0.0, 1.0)
        M0 = np.array([D_init, B_init, rho_init, 0.5, 0.5])

        children = parent_to_children.get(ci, [])
        B_up, rho_up = 0.0, 0.0
        if children:
            for cidx in children:
                if cidx < len(results):
                    B_up += results[cidx]["M_star"][1]
                    rho_up += results[cidx]["M_star"][2]
            B_up /= len(children)
            rho_up /= len(children)

        r = run_n_iteration(M0, B_up, rho_up, params_uniform)
        r["concept_idx"] = ci
        r["name"] = concept_name
        r["n_children"] = len(children)
        r["n_a"] = n_a
        r["n_b"] = n_b
        r["params"] = "uniform"
        results.append(r)

    conv = sum(1 for r in results if r["converged"])
    tau_invs = [r["tau_inv"] for r in results]
    d_means = [np.mean([r["D_init"] for r in results]), np.mean([r["D_star"] for r in results])]

    print(f"  [v1 均匀] {concept_name}: {n_concepts}概念, "
          f"收敛 {conv}/{n_concepts}, "
          f"D: {d_means[0]:.3f}→{d_means[1]:.3f}, "
          f"τ⁻¹ [{min(tau_invs):.3f}, {max(tau_invs):.3f}]")

    conv_info = [[concept_name, ci, "uniform"] for ci in range(n_concepts)]
    return results, conv_info


def run_v2_specific(
    concept_name: str,
    lattice: Dict,
    parent_to_children: Dict[int, List[int]],
    n_concepts: int,
    n_words: int,
    d_values: List[float],
    results_dir: Path,
) -> Tuple[List[Dict], List[List], List[Dict]]:
    """v2: 概念特化参数。"""
    size_info_list = lattice.get("concept_sizes", [])

    heights = compute_hasse_heights(n_concepts, parent_to_children)
    max_height = max(heights) if heights else 1

    max_a = max((s.get("|A|", 1) for s in size_info_list), default=1)
    max_b = max((s.get("|B|", 1) for s in size_info_list), default=1)
    max_children = max(
        (len(parent_to_children.get(i, [])) for i in range(n_concepts)),
        default=1,
    )

    total_extent = max(sum(s.get("|B|", 1) for s in size_info_list), 1)
    total_intent = max(sum(s.get("|A|", 1) for s in size_info_list), 1)
    max_d = max([d for d in d_values if d != float("inf") and d < 1e6]) if d_values else 1.0
    max_d = max(max_d, 1.0)

    results = []
    param_snapshots = []

    for ci in range(n_concepts):
        si = size_info_list[ci] if ci < len(size_info_list) else {"|A|": 1, "|B|": 1}
        n_a, n_b = si["|A|"], si["|B|"]
        n_ch = len(parent_to_children.get(ci, []))

        params = concept_params(
            n_a=n_a, n_b=n_b,
            n_children=n_ch,
            height=heights[ci],
            max_a=max_a, max_b=max_b,
            max_children=max_children,
            max_height=max_height,
        )

        if ci < len(d_values) and max_d > 0:
            raw_d = d_values[ci]
            D_init = min(raw_d / max_d, 1.0) if raw_d < 1e6 else 0.8
        else:
            D_init = 0.5
        D_init = np.clip(D_init, 0.0, 1.0)

        B_init = np.clip(1.0 - n_b / total_extent, 0.0, 1.0)
        rho_init = np.clip(n_a / total_intent, 0.0, 1.0)
        M0 = np.array([D_init, B_init, rho_init, 0.5, 0.5])

        children = parent_to_children.get(ci, [])
        B_up, rho_up = 0.0, 0.0
        if children:
            for cidx in children:
                if cidx < len(results):
                    B_up += results[cidx]["M_star"][1]
                    rho_up += results[cidx]["M_star"][2]
            B_up /= len(children)
            rho_up /= len(children)

        r = run_n_iteration(M0, B_up, rho_up, params)
        r["concept_idx"] = ci
        r["name"] = concept_name
        r["n_children"] = n_ch
        r["n_a"] = n_a
        r["n_b"] = n_b
        r["height"] = heights[ci]
        r["params"] = "specific"
        results.append(r)

        param_snapshots.append({
            "concept": concept_name,
            "concept_idx": ci,
            "n_a": n_a, "n_b": n_b,
            "n_children": n_ch,
            "height": heights[ci],
            "D_init": float(M0[0]),
            "params": {k: float(v) for k, v in params.items() if k != "eps"},
        })

    conv = sum(1 for r in results if r["converged"])
    tau_invs = [r["tau_inv"] for r in results]
    d_means = [np.mean([r["D_init"] for r in results]), np.mean([r["D_star"] for r in results])]

    print(f"  [v2 特化] {concept_name}: {n_concepts}概念, "
          f"收敛 {conv}/{n_concepts}, "
          f"D: {d_means[0]:.3f}→{d_means[1]:.3f}, "
          f"τ⁻¹ [{min(tau_invs):.3f}, {max(tau_invs):.3f}]")

    conv_info = [[concept_name, ci, "specific"] for ci in range(n_concepts)]
    return results, conv_info, param_snapshots


def evaluate_hasse(
    results: List[Dict],
    parent_to_children: Dict[str, Dict[int, List[int]]],
    label: str,
) -> Dict:
    result_index = {}
    for r in results:
        key = (r["name"], r["concept_idx"])
        result_index[key] = r

    hasse_results = []
    for concept_name, p2c in parent_to_children.items():
        for pi_str, children in p2c.items():
            pi = int(pi_str)
            for ci in children:
                ci = int(ci)
                parent = result_index.get((concept_name, pi))
                child = result_index.get((concept_name, ci))
                if parent and child:
                    hasse_results.append({
                        "concept": concept_name,
                        "parent_idx": pi,
                        "child_idx": ci,
                        "tau_inv_parent": float(parent["tau_inv"]),
                        "tau_inv_child": float(child["tau_inv"]),
                        "passes": bool(parent["tau_inv"] >= child["tau_inv"]),
                        "D_star_parent": float(parent["D_star"]),
                        "D_star_child": float(child["D_star"]),
                    })

    passes = sum(1 for h in hasse_results if h["passes"])
    total = len(hasse_results)
    return {
        "label": label,
        "total": total,
        "passes": passes,
        "violations": total - passes,
        "pass_rate": passes / total if total > 0 else 0.0,
        "details": hasse_results[:30],
    }


def main():
    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / "results"
    summary_path = results_dir / "e0_summary.json"

    if not summary_path.exists():
        print("错误：先运行 fca_lattice.py 生成 e0_summary.json")
        sys.exit(1)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    print(f"加载 {len(summary)} 个概念的 FCA 数据\n")

    all_v1 = []
    all_v2 = []
    all_param_snapshots = []
    parent_map: Dict[str, Dict[int, List[int]]] = {}

    for entry in summary:
        concept_name = entry["concept"]
        lattice_path = results_dir / f"{concept_name}_lattice.json"
        if not lattice_path.exists():
            continue

        with open(lattice_path, "r", encoding="utf-8") as f:
            lattice = json.load(f)

        n_concepts = lattice["n_concepts"]
        n_words = lattice["n_words"]
        d_values = lattice.get("d_values", [])
        edges = lattice.get("edges", [])

        parent_to_children = defaultdict(list)
        for pi, ci in edges:
            parent_to_children[pi].append(ci)

        parent_map[concept_name] = dict(parent_to_children)

        if n_concepts == 0:
            continue

        v1_results, _ = run_v1_uniform(
            concept_name, lattice, parent_to_children,
            n_concepts, n_words, d_values, results_dir,
        )
        all_v1.extend(v1_results)

        v2_results, _, snapshots = run_v2_specific(
            concept_name, lattice, parent_to_children,
            n_concepts, n_words, d_values, results_dir,
        )
        all_v2.extend(v2_results)
        all_param_snapshots.extend(snapshots)

    e3_v1 = evaluate_hasse(all_v1, parent_map, "v1_uniform")
    e3_v2 = evaluate_hasse(all_v2, parent_map, "v2_specific")

    print(f"\n{'='*60}")
    print(f"v1 vs v2 对比汇总")
    print(f"{'='*60}")

    for e3 in [e3_v1, e3_v2]:
        print(f"\n--- {e3['label']} ---")
        print(f"总 Hasse 对: {e3['total']}")
        print(f"通过:        {e3['passes']}")
        print(f"违反:        {e3['violations']}")
        print(f"通过率:      {e3['pass_rate']:.1%}")

    improvement = e3_v2["pass_rate"] - e3_v1["pass_rate"]
    print(f"\n提升: {improvement:+.1%}")

    all_d_stars_v1 = [r["D_star"] for r in all_v1]
    all_d_stars_v2 = [r["D_star"] for r in all_v2]
    all_tau_invs_v1 = [r["tau_inv"] for r in all_v1]
    all_tau_invs_v2 = [r["tau_inv"] for r in all_v2]

    print(f"\n--- 统计对比 ---")
    print(f"D* 均值:   v1={np.mean(all_d_stars_v1):.4f}  v2={np.mean(all_d_stars_v2):.4f}")
    print(f"D* std:    v1={np.std(all_d_stars_v1):.4f}  v2={np.std(all_d_stars_v2):.4f}")
    print(f"τ⁻¹ 均值:  v1={np.mean(all_tau_invs_v1):.4f}  v2={np.mean(all_tau_invs_v2):.4f}")
    print(f"τ⁻¹ std:   v1={np.std(all_tau_invs_v1):.4f}  v2={np.std(all_tau_invs_v2):.4f}")

    out = {
        "v1_uniform": {
            "converged_rate": sum(1 for r in all_v1 if r["converged"]) / max(len(all_v1), 1),
            "e3": e3_v1,
        },
        "v2_specific": {
            "converged_rate": sum(1 for r in all_v2 if r["converged"]) / max(len(all_v2), 1),
            "e3": e3_v2,
        },
        "improvement": improvement,
        "param_snapshots": all_param_snapshots[:50],
    }

    out_path = results_dir / "e1_n_iteration.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n详细结果: {out_path}")


if __name__ == "__main__":
    main()
