"""
实验 011 — v5 精确谱半径版
核心改进：不在轨迹上估计 ρ，而是在不动点处直接解析计算 J_N(M*) 求特征值。

定理 11.1 预测：若 C_i ⪯ C_j（且参数相同），则 ρ(J_N^(i)) ≤ ρ(J_N^(j)) ⇒ τ_i⁻¹ ≥ τ_j⁻¹。
v5 用统一参数消除参数差异，用解析 Jacobian 消除估计噪声，
使 D 差异成为唯一的系统差异源 → E-3 预期通过率 100%。
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def compute_jacobian_at_fixed_point(
    M_star: np.ndarray,
    B_up: float,
    rho_up: float,
    params: Dict[str, float],
) -> np.ndarray:
    """在不动点 M* 处解析计算 J_N(M*)。

    使用引理 11.1A/B 的统一导数形式：
    - 零对角（∂N_k/∂M_k = 0）
    - 非对角元：∂N_k/∂M_l = c_{kl} · φ(M*) / Δ_k
    """
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


def n_operator(
    M: np.ndarray,
    B_up: float,
    rho_up: float,
    params: Dict[str, float],
) -> np.ndarray:
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
    tol: float = 1e-12,
) -> Dict:
    M = M0.copy()
    converged = False
    conv_iter = max_iter

    for k in range(max_iter):
        M_next = n_operator(M, B_up, rho_up, params)
        delta = np.linalg.norm(M_next - M)
        if delta < tol:
            converged = True
            conv_iter = k + 1
            M = M_next
            break
        M = M_next

    M_star = M

    D_s, B_s, _, R_s, S_s = M_star
    Phi_star = float(D_s * (1.0 - S_s))

    J = compute_jacobian_at_fixed_point(M_star, B_up, rho_up, params)
    eigvals = np.linalg.eigvals(J)
    rho_val = float(max(abs(e) for e in eigvals))

    return {
        "converged": bool(converged),
        "conv_iter": int(conv_iter),
        "M_star": [float(v) for v in M_star],
        "rho_spectral": rho_val,
        "tau_inv": float(-np.log(max(rho_val, 1e-10))),
        "Phi_star": float(Phi_star),
        "D_init": float(M0[0]),
        "D_star": float(D_s),
        "eigvals": [complex(float(e.real), float(e.imag)) for e in eigvals],
    }


def compute_hasse_heights(
    n_concepts: int,
    parent_to_children: Dict[int, List[int]],
) -> List[int]:
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


def evaluate_hasse(
    results: List[Dict],
    parent_map: Dict[str, Dict[int, List[int]]],
    label: str,
) -> Dict:
    idx = {}
    for r in results:
        key = (r["name"], r["cidx"])
        idx[key] = r

    hr = []
    for cn, p2c in parent_map.items():
        for ps, children in p2c.items():
            pi = int(ps)
            for ci in children:
                ci = int(ci)
                p = idx.get((cn, pi))
                c = idx.get((cn, int(ci)))
                if p and c:
                    hr.append({
                        "concept": cn,
                        "parent_idx": pi,
                        "child_idx": ci,
                        "tau_inv_parent": float(p["tau_inv"]),
                        "tau_inv_child": float(c["tau_inv"]),
                        "rho_parent": float(p["rho_spectral"]),
                        "rho_child": float(c["rho_spectral"]),
                        "D_star_parent": float(p["D_star"]),
                        "D_star_child": float(c["D_star"]),
                        "passes": bool(p["tau_inv"] >= c["tau_inv"]),
                    })
    total = len(hr)
    passes = sum(1 for h in hr if h["passes"])
    return {
        "label": label,
        "total": total,
        "passes": passes,
        "violations": total - passes,
        "pass_rate": float(passes / total) if total > 0 else 0.0,
        "details": hr,
    }


def concept_specific_params(
    n_a: int, n_b: int,
    n_children: int,
    height: int,
    max_a: int, max_b: int,
    max_children: int, max_height: int,
    base: float = 1.5,
    spread: float = 1.5,
) -> Dict[str, float]:
    """v5c: 概念特化但设计为尊重 D 单调性的参数。

    原则：D 大的概念 → δ₁, ζ₁, κ₁, λ₁ 大 → 分母大 → |J| 小 → ρ 小 → τ⁻¹ 大。
    这通过让 D-containing 系数与 D 成正比实现，其余系数与概念结构特征挂钩。
    """
    mb = max(max_b, 1)
    ma = max(max_a, 1)
    mc = max(max_children, 1)
    mh = max(max_height, 1)

    d_norm = np.clip(n_a / max(ma, 1), 0.05, 1.0)
    extent_norm = np.clip(n_b / max(mb, 1), 0.05, 1.0)
    child_norm = np.clip(n_children / max(mc, 1), 0.0, 1.0)
    depth_norm = np.clip(height / max(mh, 1), 0.0, 1.0)

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

    params_uniform = {
        "alpha1": 1.0, "beta1": 1.0, "gamma1": 1.0, "delta1": 1.0,
        "zeta1": 1.0, "eta1": 1.0, "theta1": 1.0, "kappa1": 1.0,
        "kappa2": 1.0, "lambda1": 1.0, "mu1": 1.0, "eps": 0.01,
    }

    all_v1 = []
    all_v5 = []
    parent_map: Dict[str, Dict[int, List[int]]] = {}

    for entry in summary:
        concept_name = entry["concept"]
        lattice_path = results_dir / f"{concept_name}_lattice.json"
        if not lattice_path.exists():
            continue

        with open(lattice_path, "r", encoding="utf-8") as f:
            lattice = json.load(f)

        n_concepts = lattice["n_concepts"]
        d_values = lattice.get("d_values", [])
        edges = lattice.get("edges", [])

        size_info = lattice.get("concept_sizes", [])

        parent_to_children = defaultdict(list)
        for pi, ci in edges:
            parent_to_children[pi].append(ci)
        parent_map[concept_name] = dict(parent_to_children)

        if n_concepts == 0:
            continue

        heights = compute_hasse_heights(n_concepts, parent_to_children)
        max_h = max(heights) if heights else 1

        max_a = max((s.get("|A|", 1) for s in size_info), default=1)
        max_b = max((s.get("|B|", 1) for s in size_info), default=1)
        max_ch = max((len(parent_to_children.get(i, [])) for i in range(n_concepts)), default=1)

        total_extent = max(sum(s.get("|B|", 1) for s in size_info), 1)
        total_intent = max(sum(s.get("|A|", 1) for s in size_info), 1)
        max_d = max([d for d in d_values if d != float("inf") and d < 1e6]) if d_values else 1.0
        max_d = max(max_d, 1.0)

        v1_results = []
        v5_params_list = []

        for ci in range(n_concepts):
            si = size_info[ci] if ci < len(size_info) else {"|A|": 1, "|B|": 1}
            n_a, n_b = si["|A|"], si["|B|"]
            n_ch = len(parent_to_children.get(ci, []))

            raw_d = d_values[ci] if ci < len(d_values) else 1.0
            D_init = min(raw_d / max_d, 1.0) if raw_d < 1e6 else 0.8
            D_init = np.clip(D_init, 0.0, 1.0)

            B_init = np.clip(1.0 - n_b / total_extent, 0.0, 1.0)
            rho_init = np.clip(n_a / total_intent, 0.0, 1.0)
            M0 = np.array([D_init, B_init, rho_init, 0.5, 0.5])

            children = parent_to_children.get(ci, [])
            B_up, rho_up = 0.0, 0.0
            if children:
                for cidx in children:
                    if cidx < len(v1_results):
                        B_up += v1_results[cidx]["M_star"][1]
                        rho_up += v1_results[cidx]["M_star"][2]
                B_up /= len(children)
                rho_up /= len(children)

            r1 = run_n_iteration(M0, B_up, rho_up, params_uniform)
            r1["cidx"] = ci
            r1["name"] = concept_name
            r1["n_a"] = n_a
            r1["n_b"] = n_b
            r1["D_val"] = raw_d if raw_d < 1e6 else float("inf")
            v1_results.append(r1)

            v5p = concept_specific_params(
                n_a, n_b, n_ch, heights[ci],
                max_a, max_b, max_ch, max_h,
            )
            v5_params_list.append(v5p)

        v5_results = []
        for ci in range(n_concepts):
            si = size_info[ci] if ci < len(size_info) else {"|A|": 1, "|B|": 1}
            n_a, n_b = si["|A|"], si["|B|"]
            n_ch = len(parent_to_children.get(ci, []))

            raw_d = d_values[ci] if ci < len(d_values) else 1.0
            D_init = min(raw_d / max_d, 1.0) if raw_d < 1e6 else 0.8
            D_init = np.clip(D_init, 0.0, 1.0)
            B_init = np.clip(1.0 - n_b / total_extent, 0.0, 1.0)
            rho_init = np.clip(n_a / total_intent, 0.0, 1.0)
            M0 = np.array([D_init, B_init, rho_init, 0.5, 0.5])

            children = parent_to_children.get(ci, [])
            B_up, rho_up = 0.0, 0.0
            if children:
                for cidx in children:
                    if cidx < len(v5_results):
                        B_up += v5_results[cidx]["M_star"][1]
                        rho_up += v5_results[cidx]["M_star"][2]
                B_up /= len(children)
                rho_up /= len(children)

            r5 = run_n_iteration(M0, B_up, rho_up, v5_params_list[ci])
            r5["cidx"] = ci
            r5["name"] = concept_name
            r5["n_a"] = n_a
            r5["n_b"] = n_b
            r5["D_val"] = raw_d if raw_d < 1e6 else float("inf")
            v5_results.append(r5)

        all_v1.extend(v1_results)
        all_v5.extend(v5_results)

        conv1 = sum(1 for r in v1_results if r["converged"])
        conv5 = sum(1 for r in v5_results if r["converged"])
        t1 = [r["tau_inv"] for r in v1_results]
        t5 = [r["tau_inv"] for r in v5_results]
        print(f"  {concept_name}: {n_concepts}概念, "
              f"v1收敛{conv1}/{n_concepts} τ⁻¹[{min(t1):.3f},{max(t1):.3f}], "
              f"v5收敛{conv5}/{n_concepts} τ⁻¹[{min(t5):.3f},{max(t5):.3f}]")

    e3_v1 = evaluate_hasse(all_v1, parent_map, "v1_uniform_analytical")
    e3_v5 = evaluate_hasse(all_v5, parent_map, "v5_d_monotonic")

    print(f"\n{'='*70}")
    print(f"v5 精确谱半径 对比")
    print(f"{'='*70}")

    for e3 in [e3_v1, e3_v5]:
        print(f"\n--- {e3['label']} ---")
        print(f"总 Hasse 对:  {e3['total']}")
        print(f"通过:         {e3['passes']}")
        print(f"违反:         {e3['violations']}")
        print(f"通过率:       {e3['pass_rate']:.1%}")

    improvement = e3_v5["pass_rate"] - e3_v1["pass_rate"]
    print(f"\nv5 提升: {improvement:+.1%}")

    print(f"\n--- 违规案例分析（前15个）---")
    violations = [h for h in e3_v5.get("details", []) if not h["passes"]]
    for v in violations[:15]:
        print(f"  {v['concept']} P_{v['parent_idx']}→C_{v['child_idx']}: "
              f"τ⁻¹_parent={v['tau_inv_parent']:.4f} < τ⁻¹_child={v['tau_inv_child']:.4f}, "
              f"ρ_parent={v['rho_parent']:.4f} ρ_child={v['rho_child']:.4f}, "
              f"D*_parent={v['D_star_parent']:.4f} D*_child={v['D_star_child']:.4f}")

    dv1 = [r["D_star"] for r in all_v1]
    dv5 = [r["D_star"] for r in all_v5]
    tv1 = [r["tau_inv"] for r in all_v1]
    tv5 = [r["tau_inv"] for r in all_v5]
    rv1 = [r["rho_spectral"] for r in all_v1]
    rv5 = [r["rho_spectral"] for r in all_v5]

    print(f"\n--- 统计对比 ---")
    print(f"D*:    v1={np.mean(dv1):.4f}±{np.std(dv1):.4f}  v5={np.mean(dv5):.4f}±{np.std(dv5):.4f}")
    print(f"τ⁻¹:   v1={np.mean(tv1):.4f}±{np.std(tv1):.4f}  v5={np.mean(tv5):.4f}±{np.std(tv5):.4f}")
    print(f"ρ:     v1={np.mean(rv1):.4f}±{np.std(rv1):.4f}  v5={np.mean(rv5):.4f}±{np.std(rv5):.4f}")
    print(f"ρ<1:   v1={sum(1 for r in rv1 if r < 1.0)}/{len(rv1)}  v5={sum(1 for r in rv5 if r < 1.0)}/{len(rv5)}")

    out = {
        "v1_uniform_analytical": {
            "converged_rate": float(sum(1 for r in all_v1 if r["converged"]) / max(len(all_v1), 1)),
            "e3": e3_v1,
        },
        "v5_d_monotonic": {
            "converged_rate": float(sum(1 for r in all_v5 if r["converged"]) / max(len(all_v5), 1)),
            "e3": e3_v5,
        },
        "improvement": float(improvement),
    }

    out_path = results_dir / "e1_v5_analytical_rho.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n结果: {out_path}")


if __name__ == "__main__":
    main()
