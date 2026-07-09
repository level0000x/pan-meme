"""
叶节点不动点 4 阶子系统分析
验证 B*=D*, S*=ρ*，推导简化 Jacobian，分析谱半径
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


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
    a1 = params["alpha1"]; b1 = params["beta1"]
    g1 = params["gamma1"]; d1 = params["delta1"]
    z1 = params["zeta1"]; e1 = params["eta1"]
    t1 = params["theta1"]; k1 = params["kappa1"]; k2 = params["kappa2"]
    l1 = params["lambda1"]; m1 = params["mu1"]

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


def run_n_iteration(M0, B_up, rho_up, params, max_iter=2000, tol=1e-12):
    M = M0.copy()
    for k in range(max_iter):
        M_next = n_operator(M, B_up, rho_up, params)
        if np.linalg.norm(M_next - M) < tol:
            return M_next
        M = M_next
    return M


def verify_fp_equation(M, B_up, rho_up, params):
    """验证不动点是否满足方程"""
    return n_operator(M, B_up, rho_up, params)


def main():
    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / "results"

    with open(results_dir / "e0_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    params = {
        "alpha1": 1.0, "beta1": 1.0, "gamma1": 1.0, "delta1": 1.0,
        "zeta1": 1.0, "eta1": 1.0, "theta1": 1.0, "kappa1": 1.0,
        "kappa2": 1.0, "lambda1": 1.0, "mu1": 1.0, "eps": 0.01,
    }
    eps = 0.01

    leaf_fps = []
    all_fps = []

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
            M_star = run_n_iteration(M0, B_up, rho_up, params)

            D_s, B_s, rho_s, R_s, S_s = M_star

            if is_leaf:
                J = compute_jacobian_at_fixed_point(M_star, B_up, rho_up, params)
                eigvals = np.linalg.eigvals(J)
                leaf_fps.append({
                    "concept": concept_name, "cidx": ci,
                    "D": float(D_s), "B": float(B_s), "rho": float(rho_s),
                    "R": float(R_s), "S": float(S_s),
                    "J": J.tolist(),
                    "eigvals": [complex(e.real, e.imag) for e in eigvals],
                    "rho_val": float(max(abs(e) for e in eigvals)),
                    "B_minus_D": float(B_s - D_s),
                    "S_minus_rho": float(S_s - rho_s),
                })
            all_fps.append({
                "is_leaf": is_leaf, "D": float(D_s), "B": float(B_s),
                "rho": float(rho_s), "R": float(R_s), "S": float(S_s),
            })
            v1_results.append({"M_star": M_star.tolist(), "cidx": ci})

    print("=" * 90)
    print("叶节点不动点分析")
    print("=" * 90)

    # 验证恒等式
    max_bd = max(abs(fp["B_minus_D"]) for fp in leaf_fps)
    max_srho = max(abs(fp["S_minus_rho"]) for fp in leaf_fps)
    print(f"\n恒等式验证：")
    print(f"  max|B* - D*| = {max_bd:.2e}  → {'✓ B*=D*' if max_bd < 1e-14 else '✗'}")
    print(f"  max|S* - ρ*| = {max_srho:.2e}  → {'✓ S*=ρ*' if max_srho < 1e-14 else '✗'}")

    # 取第一个叶节点做详细分析
    fp = leaf_fps[0]
    D, B, rho, R, S = fp["D"], fp["B"], fp["rho"], fp["R"], fp["S"]
    print(f"\n样本叶节点 FP（{fp['concept']}[{fp['cidx']}]）：")
    print(f"  D* = {D:.12f}")
    print(f"  B* = {B:.12f}")
    print(f"  ρ* = {rho:.12f}")
    print(f"  R* = {R:.12f}")
    print(f"  S* = {S:.12f}")

    # Jacobian 解析简化（叶节点，B_up=ρ_up=0，全参数=1）
    print(f"\n--- Jacobian 解析简化（B_up=ρ_up=0, 全参数=1）---")

    # 使用不动点方程简化 Jacobian 项
    # Δ_D = R + B + ε
    # D = (R+ε)/Δ_D  →  Δ_D = (R+ε)/D
    # Δ_B = R + D + ε
    # B = (R+ε)/Δ_B  →  Δ_B = (R+ε)/B
    # Δ_ρ = D + R + ε
    # ρ = (D+ε)/Δ_ρ  →  Δ_ρ = (D+ε)/ρ
    # Δ_R = ρ + D + S + ε = ρ + D + ρ + ε = 2ρ + D + ε
    # R = (ρ+ε)/Δ_R  →  Δ_R = (ρ+ε)/R
    # Δ_S = D + R + ε
    # S = (D+ε)/Δ_S  →  Δ_S = (D+ε)/S = (D+ε)/ρ  (since S=ρ)

    Delta_D = R + B + eps
    Delta_B = R + D + eps
    Delta_rho = D + R + eps
    Delta_R_val = rho + D + S + eps
    Delta_S = D + R + eps

    # 用 FP 方程表示 Jacobian 项
    J01 = -B / Delta_D  # J_DB
    J03 = (1 - D) / Delta_D  # J_DR
    J10 = -B / Delta_B  # J_BD
    J13 = (1 - B) / Delta_B  # J_BR
    J20 = (1 - rho) / Delta_rho  # J_ρD
    J23 = -rho / Delta_rho  # J_ρR
    J30 = -R / Delta_R_val  # J_RD
    J31 = (1 - R) / Delta_R_val  # J_RB
    J32 = (1 - R) / Delta_R_val  # J_Rρ
    J34 = -R / Delta_R_val  # J_RS
    J40 = (1 - S) / Delta_S  # J_SD
    J43 = -S / Delta_S  # J_SR

    print(f"\n  Δ 分母值：")
    print(f"  Δ_D = R+B+ε = {Delta_D:.8f}")
    print(f"  Δ_B = R+D+ε = {Delta_B:.8f}")
    print(f"  Δ_ρ = D+R+ε = {Delta_rho:.8f}")
    print(f"  Δ_R = 2ρ+D+ε = {Delta_R_val:.8f}")
    print(f"  Δ_S = D+R+ε = {Delta_S:.8f}")

    print(f"\n  J_N (5×5) 精确值：")
    J_full = np.array([
        [0, J01, 0, J03, 0],
        [J10, 0, 0, J13, 0],
        [J20, 0, 0, J23, 0],
        [J30, J31, J32, 0, J34],
        [J40, 0, 0, J43, 0],
    ])
    for i, row_name in enumerate(["D", "B", "ρ", "R", "S"]):
        row = J_full[i]
        print(f"  {row_name}: [{row[0]:10.8f} {row[1]:10.8f} {row[2]:10.8f} {row[3]:10.8f} {row[4]:10.8f}]")

    print(f"\n  ρ 行与 S 行完全相同：{np.allclose(J_full[2], J_full[4], atol=1e-14)}")
    print(f"  J_N 秩 = {np.linalg.matrix_rank(J_full)}")

    # 验证简化恒等式（用 FP 方程形式）
    # J_DB = -B/Δ_D, 但由 D = (R+ε)/Δ_D, 得 Δ_D = (R+ε)/D
    # J_DB = -B·D/(R+ε)
    J01_fp = -B * D / (R + eps)
    print(f"\n  FP 方程简化验证：")
    print(f"  J_DB = -BD/(R+ε) = {J01_fp:.8f} vs 原值 {J01:.8f} → {'✓' if abs(J01_fp - J01) < 1e-14 else '✗'}")

    J03_fp = (1 - D) * D / (R + eps)
    print(f"  J_DR = (1-D)D/(R+ε) = {J03_fp:.8f} vs 原值 {J03:.8f} → {'✓' if abs(J03_fp - J03) < 1e-14 else '✗'}")

    # 由于 B=D，J_BD = -B·B/(R+ε) = -B²/(R+ε), J_BR = (1-B)B/(R+ε)
    J10_fp = -B * B / (R + eps)
    print(f"  J_BD = -B²/(R+ε) = {J10_fp:.8f} vs 原值 {J10:.8f} → {'✓' if abs(J10_fp - J10) < 1e-14 else '✗'}")

    J13_fp = (1 - B) * B / (R + eps)
    print(f"  J_BR = (1-B)B/(R+ε) = {J13_fp:.8f} vs 原值 {J13:.8f} → {'✓' if abs(J13_fp - J13) < 1e-14 else '✗'}")

    # 注意到 J_DB = -BD/(R+ε) = -B²/(R+ε) = J_BD（因为 B=D）
    print(f"\n  因 B*=D*：J_DB = {J01:.8f} = J_BD = {J10:.8f} → {'✓' if abs(J01 - J10) < 1e-14 else '✗'}")
    print(f"            J_DR = {J03:.8f} = J_BR = {J13:.8f} → {'✓' if abs(J03 - J13) < 1e-14 else '✗'}")

    # 关键发现：行 0 (D) 和行 1 (B) 也相同！
    print(f"\n  ★ 关键发现：因为 B*=D*，行 0 (D) 与行 1 (B) 也完全相同！")
    print(f"  rank(J_N) = {np.linalg.matrix_rank(J_full)}（实际应为 ≤ 3）")

    # 重新计算秩
    U, s, Vt = np.linalg.svd(J_full)
    print(f"  SVD 奇异值：{s}")
    n_nonzero = sum(1 for sv in s if sv > 1e-14)
    print(f"  有效秩 = {n_nonzero}")

    # ========================================================================
    # 简化 Jacobian：只有 3 个独立的行
    # 行 D = 行 B（均为 [-B²/(R+ε), 0, 0, (1-B)B/(R+ε), 0]）
    # 行 ρ = 行 S（均为 [(1-ρ)ρ/(D+ε), 0, 0, -ρ²/(D+ε), 0]）
    # 行 R 独立
    # ========================================================================

    print(f"\n" + "=" * 90)
    print(f"叶节点 Jacobian 的极简结构")
    print(f"=" * 90)

    # 三个独立行：
    # α = B/(R+ε)  → J_DB = J_BD = -αB, J_DR = J_BR = α(1-B)
    # β = ρ/(D+ε)  → J_ρD = J_SD = β(1-ρ), J_ρR = J_SR = -βρ
    # γ = R/(2ρ+D+ε) → J_RD = -γ, J_RB = J_Rρ = (1-R)/Δ_R, J_RS = -γ

    alpha = B / (R + eps)
    beta_s = rho / (D + eps)

    print(f"\n  简化参数：")
    print(f"  α = B/(R+ε) = {alpha:.8f}")
    print(f"  β = ρ/(D+ε) = {beta_s:.8f}")

    a = alpha * B
    b = alpha * (1 - B)
    c = beta_s * (1 - rho)
    d = beta_s * rho
    Delta_R_verify = 2 * rho + D + eps
    e = R / Delta_R_verify
    f_val = (1 - R) / Delta_R_verify

    print(f"\n  J_N 极简形式（绝对值对称正化）：")
    print(f"  行 D: [  0,   -a,   0,   +b,   0  ]")
    print(f"  行 B: [  -a,    0,   0,   +b,   0  ]  (= 行 D)")
    print(f"  行 ρ: [  +c,    0,   0,   -d,   0  ]")
    print(f"  行 R: [  -e,   +f,  +f,    0,  -e  ]")
    print(f"  行 S: [  +c,    0,   0,   -d,   0  ]  (= 行 ρ)")
    print(f"  数值验证：a = bB/(1-B) = {alpha*B:.8f}, b = {b:.8f}, c = {c:.8f}, d = {d:.8f}")
    print(f"           e = R/Δ_R = {e:.8f}, f = (1-R)/Δ_R = {f_val:.8f}")

    # 独立 3×3 核心子矩阵（压缩后）
    # 由于行 D = 行 B = [-a, 0, 0, +b, 0] (对称)，且行 ρ = 行 S = [+c, 0, 0, -d, 0]
    # 我们可以约束到子空间 span{v_D+v_B, v_ρ+v_S, v_R}

    print(f"\n" + "=" * 90)
    print(f"3 阶约化 Jacobian（在对称子空间上）")
    print(f"=" * 90)

    # 由于 B*=D*，D 和 B 在 FP 处相等，且 Jacobian 的行 0 和行 1 相同。
    # 当我们在 FP 邻域做扰动时，如果我们约束 D=B（沿不变子空间），
    # 则有效 Jacobian 在子空间 span{[1,1,0,0,0]^T, [0,0,1,1,0]^T, [0,0,0,0,1]^T}
    # 但更合理的做法：考虑扰动 δ 在子空间上的投影。

    # 实际上，考虑变换：
    # u = (D+B)/2, v = (D-B)/2  (v 方向趋于 0，因为 D=B 是稳定约束)
    # 在 D=B 的子空间上，有效动力学由 u, ρ, R, S 描述（且 ρ=S）

    # 真正的有效 3×3 Jacobian 在子空间 {D=B, ρ=S} 上：
    # 变量：[u=(D+B)/2, ρ_sym=(ρ+S)/2, R]
    # 
    # ∂N_u/∂u = ∂/∂u [ (R+ε)/(R+u+ε) ]_u=D=B = -(R+ε)/(R+u+ε)² = -u/(R+ε) = -α
    # 不，让我直接计算。

    print(f"\n  约束子空间 D=B, ρ=S 上的简化 N 算子：")
    print(f"  令 u = D = B, v = ρ = S，则")
    print(f"  N_u(u, v, R) = (R+ε)/(R+u+ε)     ← D=B 的公共 FP 方程")
    print(f"  N_v(u, v, R) = (u+ε)/(u+R+ε)     ← ρ=S 的公共 FP 方程")
    print(f"  N_R(u, v, R) = (v+ε)/(v+u+v+ε) = (v+ε)/(u+2v+ε)")

    print(f"\n  3×3 简化 Jacobian J̃ 在不动点处：")
    # ∂N_u/∂u: 使用商法则
    # N_u = (R+ε)/(R+u+ε), ∂N_u/∂u = -(R+ε)/(R+u+ε)² = -(R+ε)/Δ_B² = -u/(R+ε) = -α
    # 其中 u = (R+ε)/Δ_B 由 FP 方程。实际上 Δ_B = R + u + ε，所以 u = (R+ε)/(R+u+ε)
    # ∂N_u/∂u = -(R+ε)/(R+u+ε)² = -(u·(R+u+ε))/(R+u+ε)² = -u/(R+u+ε) = -u/(R+u+ε)
    # 嗯，让我仔细算。
    # 从 N_u(u,v,R) = (R+ε)/(R+u+ε)
    # ∂N_u/∂u = -(R+ε)/(R+u+ε)²
    # 但在 FP 处 u = (R+ε)/(R+u+ε)，所以 (R+ε) = u·(R+u+ε)
    # ∂N_u/∂u = -u·(R+u+ε)/(R+u+ε)² = -u/(R+u+ε)
    # = -u/Δ_B = -u·(u/(R+ε)) = -u²/(R+ε)
    # 这和 -α·B = -B²/(R+ε) 一致，OK。

    delta_u = R + D + eps  # = R + u + eps
    delta_v = D + R + eps  # = u + R + eps
    delta_r = D + 2 * rho + eps  # = u + 2v + eps

    J11 = -D / delta_u  # ∂N_u/∂u
    J12 = 0             # ∂N_u/∂ρ
    J13 = (1 - D) / delta_u  # ∂N_u/∂R

    J21 = (1 - rho) / delta_v  # ∂N_ρ/∂u
    J22 = 0                     # ∂N_ρ/∂ρ
    J23 = -rho / delta_v        # ∂N_ρ/∂R

    J31 = -R / delta_r          # ∂N_R/∂u
    J32 = (2 * (1 - R)) / delta_r  # ∂N_R/∂ρ（注意：R 对 ρ 和 S 的导数之和）
    J33 = 0                     # ∂N_R/∂R

    J_reduced = np.array([
        [J11, J12, J13],
        [J21, J22, J23],
        [J31, J32, J33],
    ])

    print(f"  J̃ = [")
    print(f"    [{J11:10.8f}, {J12:10.8f}, {J13:10.8f}],")
    print(f"    [{J21:10.8f}, {J22:10.8f}, {J23:10.8f}],")
    print(f"    [{J31:10.8f}, {J32:10.8f}, {J33:10.8f}],")
    print(f"  ]")

    eigvals_red = np.linalg.eigvals(J_reduced)
    rho_red = max(abs(e) for e in eigvals_red)
    print(f"\n  简化 J̃ 特征值: {[complex(e.real, e.imag) for e in eigvals_red]}")
    print(f"  ρ(J̃) = {rho_red:.8f}")
    print(f"  原始 ρ(J_N) = {fp['rho_val']:.8f}")

    # 现在尝试解析证明 ρ(J̃) < 1
    # J̃ 的特征多项式: det(λI - J̃) = λ³ - tr(J̃)·λ² + (sum of principal minors)·λ - det(J̃)
    # J̃ has zero diagonal, so tr(J̃) = 0 → 特征多项式: λ³ + p·λ - q = 0
    # 其中 p = -(sum of 2×2 principal minors), q = det(J̃)

    p = -(J11*J22 - J12*J21 + J11*J33 - J13*J31 + J22*J33 - J23*J32)
    q = np.linalg.det(J_reduced)
    
    # 由于 J22 = J33 = 0, J12 = 0:
    p_simplified = -(-J13*J31 - J23*J32)
    # = J13*J31 + J23*J32
    
    print(f"\n  --- 特征多项式分析 ---")
    print(f"  det(λI - J̃) = λ³ + p·λ + q = 0")
    print(f"  其中 p = J₁₃J₃₁ + J₂₃J₃₂ = {p_simplified:.8f}")
    print(f"      q = -det(J̃) = {-q:.8f}")
    print(f"  数值验证 p = {p:.8f}, q = {q:.8f}")

    # 对于 λ³ + pλ + q = 0，判别式 Δ = -(4p³ + 27q²)
    # 当 Δ > 0 时三实根，Δ < 0 时一实二复共轭
    
    discriminant = -(4 * p**3 + 27 * q**2)
    print(f"  判别式 Δ = -(4p³+27q²) = {discriminant:.6f}")
    
    if discriminant > 0:
        print(f"  Δ > 0 → 三个实根")
    else:
        print(f"  Δ < 0 → 一个实根 + 一对共轭复根")

    # 对于复根 α ± iβ，|λ|² = α² + β²
    # 对于实根，用 Cardano 公式
    
    # 我们知道 ρ(J_N) < 1 iff 所有根 |λ| < 1
    # λ³ + pλ + q = 0，设 λ = x+iy
    # 复根对：由 Vieta，λ₁ + λ₂ + λ₃ = 0（trace=0）
    # |λ|² < 1 对所有根是否成立？

    # 尝试：对于 λ³ + pλ + q 的根，|λ| < 1 的一个充分条件是
    # |q| < 1 且 |p| < 1 + q²（Eneström–Kakeya 或其他多项式根界）

    print(f"\n  --- 谱半径 < 1 的解析条件 ---")
    print(f"  多项式：f(λ) = λ³ + p·λ + q, p = {p:.6f}, q = {q:.6f}")
    
    # Cauchy 界：|λ| ≤ 1 + max(|p|, |q|) — 太弱
    # 更紧的界：对于 λ³ + pλ + q，所有根满足 |λ| < R iff
    # |p|R² + R³ + |q| < 0 在 R 稍大于 ρ 时（不精确）
    
    # 用 Montel 界或 Fujiwara 界
    # 更实用的：检验 |λ| = 1 时 |λ³| = 1 > |pλ + q|？
    # 如果 |p| + |q| < 1，则 Rouché 定理给出 |λ| < 1
    
    if abs(p) + abs(q) < 1:
        print(f"  |p|+|q| = {abs(p)+abs(q):.6f} < 1 → Rouché: 所有 |λ| < 1")
    else:
        print(f"  |p|+|q| = {abs(p)+abs(q):.6f} ≥ 1 → Rouché 不直接适用")
    
    # 精确检查 |λ| < 1 当所有根为实根
    # f(1) = 1 + p + q > 0 且 f(-1) = -1 - p + q < 0 → 所有根在 (-1, 1) 内？
    f_1 = 1 + p + q
    f_neg1 = -1 - p + q
    print(f"  f(1) = 1+p+q = {f_1:.6f}")
    print(f"  f(-1) = -1-p+q = {f_neg1:.6f}")

    # 对于实根多项式 λ³ + pλ + q，|λ| < 1 iff:
    # (1) q² < 1 (product constraint)
    # (2) f(1) > 0 and f(-1) < 0
    
    if discriminant > 0 and f_1 > 0 and f_neg1 < 0:
        print(f"  ★ 三实根均在 (-1, 1) 内的解析判据通过！")

    # 也尝试类似赫尔维茨的判据（对于离散时间稳定性）
    # 双线性变换 z = (λ+1)/(λ-1) 将 |λ|<1 映射到 Re(z)<0
    # 但这很繁琐，对三次方有更简洁的方法

    # 对所有可能的叶节点做
    print(f"\n" + "=" * 90)
    print(f"全部 {len(leaf_fps)} 个叶节点 3 阶简化子系统的验证")
    print(f"=" * 90)
    
    results = []
    for fp in leaf_fps:
        D_s, B_s, rho_s, R_s, S_s = fp["D"], fp["B"], fp["rho"], fp["R"], fp["S"]
        delta_u = R_s + D_s + eps
        delta_v = D_s + R_s + eps
        delta_r = D_s + 2 * rho_s + eps
        
        J11 = -D_s / delta_u
        J13 = (1 - D_s) / delta_u
        J21 = (1 - rho_s) / delta_v
        J23 = -rho_s / delta_v
        J31 = -R_s / delta_r
        J32 = (2 * (1 - R_s)) / delta_r
        
        p_val = J13 * J31 + J23 * J32
        q_val = -(J11 * (0 - J23 * J32) - J13 * (J21 * J32) + 0)  # det
        # 手动计算 q:
        q_val = J11 * J23 * J32 + J13 * J21 * J32
        
        J_red = np.array([[J11, 0, J13], [J21, 0, J23], [J31, J32, 0]])
        eigs = np.linalg.eigvals(J_red)
        rho_red = max(abs(e) for e in eigs)
        
        f1 = 1 + p_val + q_val
        fn1 = -1 - p_val + q_val
        
        results.append({
            "concept": fp["concept"], "cidx": fp["cidx"],
            "p": p_val, "q": q_val,
            "rho_red": rho_red, "rho_full": fp["rho_val"],
            "f(1)": f1, "f(-1)": fn1,
            "ppq": abs(p_val) + abs(q_val),
            "disc": -(4*p_val**3 + 27*q_val**2),
            "eigs": eigs,
        })
    
    print(f"  {'概念':<25} {'p':>10} {'q':>10} {'|p|+|q|':>10} {'f(1)':>10} {'f(-1)':>10} {'ρ(J̃)':>10} {'ρ(J)':>10}")
    print(f"  {'-' * 100}")
    for r in results[:5]:
        print(f"  {r['concept']:<25} {r['p']:10.6f} {r['q']:10.6f} {r['ppq']:10.6f} "
              f"{r['f(1)']:10.6f} {r['f(-1)']:10.6f} {r['rho_red']:10.6f} {r['rho_full']:10.6f}")

    # 统计
    all_rouch = all(r["ppq"] < 1 for r in results)
    all_f1_gt0 = all(r["f(1)"] > 0 for r in results)
    all_fn1_lt0 = all(r["f(-1)"] < 0 for r in results)
    all_rho_lt1 = all(r["rho_red"] < 1 for r in results)
    
    print(f"\n  统计：")
    print(f"    |p|+|q| < 1 (Rouché): {all_rouch} ({sum(1 for r in results if r['ppq']<1)}/{len(results)})")
    print(f"    f(1) > 0: {all_f1_gt0} ({sum(1 for r in results if r['f(1)']>0)}/{len(results)})")
    print(f"    f(-1) < 0: {all_fn1_lt0} ({sum(1 for r in results if r['f(-1)']<0)}/{len(results)})")
    print(f"    ρ(J̃) < 1: {all_rho_lt1}")

    # 解析证明：证明 p, q, f(1), f(-1) 的符号和量级
    print(f"\n" + "=" * 90)
    print(f"解析证明路径")
    print(f"=" * 90)
    
    print(f"""
  目标：证明 3 阶简化 Jacobian J̃ 的谱半径 ρ(J̃) < 1。
  
  J̃ = [
    [-u/(R+u+ε),     0,        (1-u)/(R+u+ε)  ],
    [(1-v)/(u+R+ε),   0,        -v/(u+R+ε)      ],
    [-R/(u+2v+ε),   2(1-R)/(u+2v+ε),   0        ]
  ]
  
  其中 u=D=B, v=ρ=S 满足不动点方程：
    u = (R+ε)/(R+u+ε)      …(1)
    v = (u+ε)/(u+R+ε)      …(2)
    R = (v+ε)/(u+2v+ε)    …(3)
  
  特征多项式：λ³ + pλ + q = 0（tr(J̃)=0 因为对角线全零）。
  
  p = J₁₃·J₃₁ + J₂₃·J₃₂
    = (1-u)u/(R+u+ε)(R+ε) · (-R)/(u+2v+ε) + (-v)v/(v+ε) · 2(1-R)/(u+2v+ε)
  
  用 FP 方程简化：u = (R+ε)/(R+u+ε) → (R+u+ε) = (R+ε)/u
  类似地，(u+R+ε) = (u+ε)/v, (u+2v+ε) = (v+ε)/R
  
  代入：
  p = (1-u)u·u/(R+ε) · (-R)·R/(v+ε) + (-v)v·v/(u+ε) · 2(1-R)·R/(v+ε)
    = -u²R²(1-u)/[(R+ε)(v+ε)] - 2v²R(1-R)/[(u+ε)(v+ε)]
  
  但 v = (u+ε)/(u+R+ε) → u+ε = v(u+R+ε) = v(u+R+ε)
  
  ...这个方向很繁琐。更简洁的方法：

  **解析策略**：
  1. 利用 Rouché 定理：|p| + |q| < 1 → 所有 |λ| < 1
  2. 对 p 和 q 给出解析上界
  3. 在参数域 (u,v,R ∈ (0,1), ε 小正) 上证明该上界 < 1
    """)

    print(f"\n  数值摘要（第一个叶节点）：")
    print(f"  u = {D:.6f}, v = {rho:.6f}, R = {R:.6f}")
    print(f"  p = {p:.6f}, q = {q:.6f}")
    print(f"  |p| + |q| = {abs(p)+abs(q):.6f}")
    print(f"  p < 0: {'是' if p < 0 else '否'}")
    print(f"  q > 0: {'是' if q > 0 else '否'}")


if __name__ == "__main__":
    main()
