"""
核心缺口 #1 验证脚本
任务：对所有 concept 用统一参数（all=1, ε=0.01）计算不动点，
检验 R_S = 1/(D*+R*+ε) < 1 是否恒成立。
等价于检验 D*+R* > 1-ε = 0.99 是否恒成立。
"""

import json
import sys
from pathlib import Path

import numpy as np


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

    N_D = (a1 * R + eps) / (a1 * R + b1 * (B + B_up) + eps)
    N_B = (g1 * (R + B_up) + eps) / (g1 * (R + B_up) + d1 * D + eps)
    N_rho = (z1 * (D + rho_up) + eps) / (z1 * (D + rho_up) + e1 * R + eps)
    N_R = (t1 * (rho + rho_up + B_up) + eps) / (t1 * (rho + rho_up + B_up) + k1 * D + k2 * S + eps)
    N_S = (l1 * D + eps) / (l1 * D + m1 * R + eps)

    return np.array([N_D, N_B, N_rho, N_R, N_S])


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
    for k in range(max_iter):
        M_next = n_operator(M, B_up, rho_up, params)
        delta = np.linalg.norm(M_next - M)
        if delta < tol:
            M = M_next
            break
        M = M_next
    return M


def compute_hasse_heights(n_concepts, parent_to_children):
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


def s_row_gershgorin_radius(D, R, S, l1, m1, eps):
    """直接计算 S 行 Gershgorin 半径（引理 11.1B 的导数形式）"""
    Delta_S = l1 * D + m1 * R + eps
    J_SD = l1 * (1.0 - S) / Delta_S
    J_SR = -m1 * S / Delta_S
    return abs(J_SD) + abs(J_SR)


def analytic_r_s(D, R, l1, m1, eps):
    """统一参数 (l1=m1=1) 下的简化公式"""
    Delta_S = D + R + eps
    return 1.0 / Delta_S


def main():
    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / "results"
    summary_path = results_dir / "e0_summary.json"

    if not summary_path.exists():
        print("错误：先运行 fca_lattice.py 生成 e0_summary.json")
        sys.exit(1)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    params_uniform = {
        "alpha1": 1.0, "beta1": 1.0, "gamma1": 1.0, "delta1": 1.0,
        "zeta1": 1.0, "eta1": 1.0, "theta1": 1.0, "kappa1": 1.0,
        "kappa2": 1.0, "lambda1": 1.0, "mu1": 1.0, "eps": 0.01,
    }
    eps = 0.01

    all_fixed_points = []

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

        from collections import defaultdict
        parent_to_children = defaultdict(list)
        for pi, ci in edges:
            parent_to_children[pi].append(ci)

        total_extent = max(sum(s.get("|B|", 1) for s in size_info), 1)
        total_intent = max(sum(s.get("|A|", 1) for s in size_info), 1)
        max_d = max([d for d in d_values if d != float("inf") and d < 1e6]) if d_values else 1.0
        max_d = max(max_d, 1.0)

        v1_results = []
        for ci in range(n_concepts):
            si = size_info[ci] if ci < len(size_info) else {"|A|": 1, "|B|": 1}
            n_a, n_b = si["|A|"], si["|B|"]
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

            M_star = run_n_iteration(M0, B_up, rho_up, params_uniform)
            result_entry = {
                "M_star": M_star.tolist(),
                "cidx": ci,
                "name": concept_name,
            }
            v1_results.append(result_entry)
            D_s, B_s, rho_s, R_s, S_s = M_star
            
            X = D_s + R_s
            R_S_analytic = analytic_r_s(D_s, R_s, 1.0, 1.0, eps)
            R_S_gershgorin = s_row_gershgorin_radius(D_s, R_s, S_s, 1.0, 1.0, eps)
            
            J = compute_jacobian_at_fixed_point(M_star, B_up, rho_up, params_uniform)
            eigvals = np.linalg.eigvals(J)
            rho_spectral = float(max(abs(e) for e in eigvals))
            
            all_fixed_points.append({
                "concept": concept_name,
                "cidx": ci,
                "D*": float(D_s),
                "B*": float(B_s),
                "rho*": float(rho_s),
                "R*": float(R_s),
                "S*": float(S_s),
                "X = D*+R*": float(X),
                "1-ε": 1.0 - eps,
                "X > 1-ε": bool(X > 1.0 - eps),
                "R_S (analytic)": float(R_S_analytic),
                "R_S < 1": bool(R_S_analytic < 1.0),
                "R_S (Gershgorin)": float(R_S_gershgorin),
                "rho_spectral": rho_spectral,
                "rho < 1": bool(rho_spectral < 1.0),
            })

    print(f"{'='*90}")
    print(f"核心缺口 #1 验证：R_S = 1/(D*+R*+ε) < 1 ⇔ D*+R* > 1-ε")
    print(f"统一参数：all=1.0, ε=0.01")
    print(f"{'='*90}\n")

    X_values = [fp["X = D*+R*"] for fp in all_fixed_points]
    RS_values = [fp["R_S (analytic)"] for fp in all_fixed_points]
    
    print(f"总不动点数: {len(all_fixed_points)}")
    print(f"\n--- D*+R* 统计 ---")
    print(f"  范围: [{min(X_values):.6f}, {max(X_values):.6f}]")
    print(f"  均值: {np.mean(X_values):.6f} ± {np.std(X_values):.6f}")
    
    above = sum(1 for x in X_values if x > 1.0 - eps)
    below = sum(1 for x in X_values if x <= 1.0 - eps)
    print(f"  > 1-ε (=0.99): {above}/{len(X_values)} ({100*above/len(X_values):.1f}%)")
    print(f"  ≤ 1-ε (=0.99): {below}/{len(X_values)} ({100*below/len(X_values):.1f}%)")
    
    print(f"\n--- R_S = 1/(X+ε) 统计 ---")
    print(f"  范围: [{min(RS_values):.6f}, {max(RS_values):.6f}]")
    print(f"  均值: {np.mean(RS_values):.6f} ± {np.std(RS_values):.6f}")
    
    rs_lt1 = sum(1 for r in RS_values if r < 1.0)
    rs_ge1 = sum(1 for r in RS_values if r >= 1.0)
    print(f"  < 1: {rs_lt1}/{len(RS_values)} ({100*rs_lt1/len(RS_values):.1f}%)")
    print(f"  ≥ 1: {rs_ge1}/{len(RS_values)} ({100*rs_ge1/len(RS_values):.1f}%)")
    
    rho_lt1 = sum(1 for fp in all_fixed_points if fp["rho_spectral"] < 1.0)
    print(f"\n--- ρ(J_N) 统计 ---")
    print(f"  < 1: {rho_lt1}/{len(all_fixed_points)} ({100*rho_lt1/len(all_fixed_points):.1f}%)")
    
    if below > 0:
        print(f"\n{'='*90}")
        print(f"⚠ 发现 {below} 个不满足 D*+R* > 1-ε 的不动点！")
        print(f"{'='*90}")
        for fp in all_fixed_points:
            if fp["X = D*+R*"] <= 1.0 - eps:
                print(f"  {fp['concept']}[{fp['cidx']}]: "
                      f"D*={fp['D*']:.6f}, R*={fp['R*']:.6f}, "
                      f"X={fp['X = D*+R*']:.6f}, "
                      f"R_S={fp['R_S (analytic)']:.6f}, "
                      f"ρ={fp['rho_spectral']:.6f}")
    else:
        print(f"\n✓ 所有不动点均满足 D*+R* > 1-ε！")

    if rs_ge1 > 0:
        print(f"\n⚠ 发现 {rs_ge1} 个 R_S ≥ 1 的不动点！")
        for fp in all_fixed_points:
            if fp["R_S (analytic)"] >= 1.0:
                print(f"  {fp['concept']}[{fp['cidx']}]: "
                      f"D*={fp['D*']:.6f}, R*={fp['R*']:.6f}, "
                      f"X={fp['X = D*+R*']:.6f}, "
                      f"R_S={fp['R_S (analytic)']:.6f}")

    print(f"\n--- 完整数据展示（前20个 + 边界20个）---")
    print(f"{'概念':<25} {'cidx':>4} {'D*':>8} {'R*':>8} {'S*':>8} {'X=D*+R*':>10} {'R_S':>8} {'ρ':>8}")
    print("-" * 90)
    
    shown = 0
    sorted_by_x = sorted(all_fixed_points, key=lambda fp: fp["X = D*+R*"])
    
    for fp in sorted_by_x[:10]:
        print(f"{fp['concept']:<25} {fp['cidx']:>4} {fp['D*']:>8.4f} {fp['R*']:>8.4f} "
              f"{fp['S*']:>8.4f} {fp['X = D*+R*']:>10.6f} {fp['R_S (analytic)']:>8.6f} {fp['rho_spectral']:>8.6f}")
        shown += 1
    if len(sorted_by_x) > 20:
        print("  ...")
    for fp in sorted_by_x[-10:]:
        print(f"{fp['concept']:<25} {fp['cidx']:>4} {fp['D*']:>8.4f} {fp['R*']:>8.4f} "
              f"{fp['S*']:>8.4f} {fp['X = D*+R*']:>10.6f} {fp['R_S (analytic)']:>8.6f} {fp['rho_spectral']:>8.6f}")

    out_path = results_dir / "e1_core_gap1_verification.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_fixed_points, f, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
