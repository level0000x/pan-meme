"""
验证：叶不动点下，对角相似变换能否使 Gershgorin 半径全 < 1？
（尝试 D=diag(d0,d1,d2,d3,d4) 缩放 J' = D^{-1} J D）
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize


def n_operator(M, B_up, rho_up, params):
    D, B, rho, R, S = M
    eps = params.get("eps", 0.01)
    a1, b1 = params["alpha1"], params["beta1"]
    g1, d1 = params["gamma1"], params["delta1"]
    z1, e1 = params["zeta1"], params["eta1"]
    t1, k1, k2 = params["theta1"], params["kappa1"], params["kappa2"]
    l1, m1 = params["lambda1"], params["mu1"]

    N_D = (a1 * R + eps) / (a1 * R + b1 * (B + B_up) + eps)
    N_B = (g1 * (R + B_up) + eps) / (g1 * (R + B_up) + d1 * D + eps)
    N_rho = (z1 * (D + rho_up) + eps) / (z1 * (D + rho_up) + e1 * R + eps)
    N_R = (t1 * (rho + rho_up + B_up) + eps) / (t1 * (rho + rho_up + B_up) + k1 * D + k2 * S + eps)
    N_S = (l1 * D + eps) / (l1 * D + m1 * R + eps)

    return np.array([N_D, N_B, N_rho, N_R, N_S])


def compute_jacobian_at_fixed_point(M_star, B_up, rho_up, params):
    D, B, rho, R, S = M_star
    eps = params.get("eps", 0.01)
    a1, b1 = params["alpha1"], params["beta1"]
    g1, d1 = params["gamma1"], params["delta1"]
    z1, e1 = params["zeta1"], params["eta1"]
    t1, k1, k2 = params["theta1"], params["kappa1"], params["kappa2"]
    l1, m1 = params["lambda1"], params["mu1"]

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


def max_gershgorin_radius(J, d):
    """对角缩放后的最大 Gershgorin 半径 d_k * sum_l |J_kl|/d_l"""
    n = J.shape[0]
    absJ = np.abs(J)
    max_r = 0.0
    for k in range(n):
        r = sum(d[k] / d[l] * absJ[k, l] for l in range(n))
        max_r = max(max_r, r)
    return max_r


def check_scaling_feasibility(J, verbose=False):
    """尝试找到 d 使所有 Gershgorin 半径 < 1"""
    n = 5
    
    def objective(log_d):
        d = np.exp(log_d)
        r = max_gershgorin_radius(J, d)
        return r if r >= 1.0 else r - 1.0  # 惩罚 r >= 1
    
    # 多起点全局搜索
    best_d = None
    best_r = float('inf')
    
    for _ in range(50):
        log_d0 = np.random.randn(5) * 0.5
        result = minimize(objective, log_d0, method='BFGS', options={'maxiter': 5000})
        d = np.exp(result.x)
        r = max_gershgorin_radius(J, d)
        if r < best_r:
            best_r = r
            best_d = d
        if r < 0.999:
            break
    
    return best_r, best_d


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

    print("=" * 90)
    print("叶不动点相似变换可行性分析")
    print("=" * 90)

    leaf_nodes = []

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

            is_leaf = len(children) == 0
            M_star = run_n_iteration(M0, B_up, rho_up, params_uniform)
            J = compute_jacobian_at_fixed_point(M_star, B_up, rho_up, params_uniform)
            eigvals = np.linalg.eigvals(J)
            rho_val = float(max(abs(e) for e in eigvals))
            
            if is_leaf:
                D_s, B_s, rho_s, R_s, S_s = M_star
                leaf_nodes.append({
                    "concept": concept_name,
                    "cidx": ci,
                    "M*": M_star.tolist(),
                    "S*-rho*": float(S_s - rho_s),
                    "B*+rho*": float(B_s + rho_s),
                    "rank(J)": np.linalg.matrix_rank(J),
                    "min_eigval": float(min(abs(e) for e in eigvals)),
                    "rho": rho_val,
                    "J_abs": np.abs(J).tolist(),
                })
            v1_results.append({"M_star": M_star.tolist(), "cidx": ci, "name": concept_name})

    print(f"\n找到 {len(leaf_nodes)} 个叶不动点\n")

    # 验证叶不动点恒等式
    print("--- 叶不动点恒等式验证 ---")
    s_eq_rho = all(abs(n["S*-rho*"]) < 1e-14 for n in leaf_nodes)
    bp_gt_1 = all(n["B*+rho*"] > 1.0 for n in leaf_nodes)
    has_eig_zero = all(n["min_eigval"] < 1e-14 for n in leaf_nodes)
    
    print(f"  S* = ρ* (容差 1e-14): {s_eq_rho}")
    print(f"  B* + ρ* > 1: {bp_gt_1}  (值范围: [{min(n['B*+rho*'] for n in leaf_nodes):.6f}, "
          f"{max(n['B*+rho*'] for n in leaf_nodes):.6f}])")
    print(f"  存在 λ≈0 特征值: {has_eig_zero}")

    # 对每个叶节点尝试对角缩放
    print(f"\n--- 对角相似变换搜索 ---")
    print(f"{'概念':<25} {'cidx':>4} {'原始max R_k':>12} {'缩放后max R_k':>14} {'ρ':>8} {'可行':>6}")
    print("-" * 75)

    feasible_count = 0
    for n in leaf_nodes[:5]:  # 前5个做搜索（搜索成本高）
        J = np.array(n["J_abs"])
        orig_max_r = max_gershgorin_radius(J, np.ones(5))
        best_r, best_d = check_scaling_feasibility(J)
        
        scaled_max = max_gershgorin_radius(J, best_d) if best_d is not None else float('inf')
        feasible = scaled_max < 0.999
        
        if feasible:
            feasible_count += 1
        
        print(f"{n['concept']:<25} {n['cidx']:>4} {orig_max_r:>12.6f} {scaled_max:>14.6f} "
              f"{n['rho']:>8.6f} {'✓' if feasible else '✗':>6}")

    if feasible_count == 0:
        print("\n结论：对角相似变换无法使所有 Gershgorin 半径 < 1。")

    # 解析反证
    print(f"\n--- 解析反证（双边不等式冲突）---")
    sample = leaf_nodes[0]
    J_s = np.array(sample["J_abs"])
    j_db, j_dr = J_s[0, 1], J_s[0, 3]
    j_bd, j_br = J_s[1, 0], J_s[1, 3]
    j_sd, j_sr = J_s[4, 0], J_s[4, 3]
    j_rd, j_rb, j_rrho, j_rs = J_s[3, 0], J_s[3, 1], J_s[3, 2], J_s[3, 4]
    
    print(f"  叶节点 Jacobian 非零元 (绝对值):")
    print(f"  J_DB={j_db:.4f}  J_DR={j_dr:.4f}")
    print(f"  J_BD={j_bd:.4f}  J_BR={j_br:.4f}")
    print(f"  J_RD={j_rd:.4f}  J_RB={j_rb:.4f}  J_Rρ={j_rrho:.4f}  J_RS={j_rs:.4f}")
    print(f"  J_SD={j_sd:.4f}  J_SR={j_sr:.4f}  (= J_ρD, J_ρR by S*=ρ*)")
    
    print(f"\n  行 0 约束: d₀/d₁·{j_db:.4f} + d₀/d₃·{j_dr:.4f} < 1")
    print(f"  行 1 约束: d₁/d₀·{j_bd:.4f} + d₁/d₃·{j_br:.4f} < 1")
    print(f"  行 4 约束: d₄·{j_sd:.4f} + d₄/d₃·{j_sr:.4f} < 1")
    print(f"  行 3 约束: d₃·{j_rd:.4f} + d₃/d₁·{j_rb:.4f} + d₃/d₄·{j_rrho:.4f} + d₃/d₄·{j_rs:.4f} < 1")
    
    # 缩减变量：行 0 + 行 1 ⇒ d₃ 下界
    d3_min = (j_bd + j_br) * j_dr + j_db * j_br
    print(f"\n  行 0+1 联立 ⇒ d₃ > {d3_min:.4f}")
    
    # 行 3 + 行 0 + 行 1 联立 ⇒ d₃ 上界
    sum_rd_rb = j_rd + j_rb
    sum_rrho_rs = j_rrho + j_rs
    # R'_3 = d₃·(j_rd + j_rb/d₁ + (j_rrho+j_rs)/d₄)
    # 代入 d₁ 和 d₄ 的上界 ⇒ d₃ 上界
    
    d4_max = 1.0 / (j_sd + j_sr / d3_min)
    d1_max = 1.0 / (j_bd + j_br / d3_min)
    d3_max = 1.0 / (j_rd + j_rb / d1_max + (j_rrho + j_rs) / d4_max)
    
    print(f"  行 3 联立 ⇒ d₃ < {d3_max:.4f}")
    print(f"  d₃ 上下界: [{d3_min:.4f}, {d3_max:.4f}] — {'相容' if d3_min < d3_max else '矛盾 ✗'}")
    
    eps_val = params_uniform["eps"]
    print(f"\n  数值: j_db={j_db:.6f}, j_dr={j_dr:.6f}, j_bd={j_bd:.6f}, j_br={j_br:.6f}")
    print(f"        j_rd={j_rd:.6f}, j_rb={j_rb:.6f}, j_rrho+s={j_rrho+j_rs:.6f}")
    print(f"        j_sd={j_sd:.6f}, j_sr={j_sr:.6f}")
    
    
if __name__ == "__main__":
    main()
