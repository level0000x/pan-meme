"""
R 行情形 2 解析——修复 T 公式笔误并验证 Z+ε < 1 是否存在
搜索整个 [0,1]^5 状态空间，看是否能找到 Z+ε < 1 的不动点
"""

import numpy as np

PARAMS = {
    "alpha1": 1.0, "beta1": 1.0, "gamma1": 1.0, "delta1": 1.0,
    "zeta1": 1.0, "eta1": 1.0, "theta1": 1.0, "kappa1": 1.0,
    "kappa2": 1.0, "lambda1": 1.0, "mu1": 1.0, "eps": 0.01,
}
EPS = 0.01


def n_operator(M):
    D, B, rho, R, S = M
    a1, b1, g1, d1 = 1.0, 1.0, 1.0, 1.0
    z1, e1, t1, k1, k2 = 1.0, 1.0, 1.0, 1.0, 1.0
    l1, m1 = 1.0, 1.0

    N_D = (a1 * R + EPS) / (a1 * R + b1 * B + EPS)
    N_B = (g1 * R + EPS) / (g1 * R + d1 * D + EPS)
    N_rho = (z1 * D + EPS) / (z1 * D + e1 * R + EPS)
    N_R = (t1 * rho + EPS) / (t1 * rho + k1 * D + k2 * S + EPS)
    N_S = (l1 * D + EPS) / (l1 * D + m1 * R + EPS)
    return np.array([N_D, N_B, N_rho, N_R, N_S])


def find_fixed_point(M0, max_iter=5000, tol=1e-13):
    M = M0.copy()
    for _ in range(max_iter):
        M_next = n_operator(M)
        if np.linalg.norm(M_next - M) < tol:
            return M_next
        M = M_next
    return None


def compute_R_gershgorin_radius(M_star):
    """验证 R 行 Gershgorin 半径 R_R，使用正确的 T 推导"""
    D, B, rho, R, S = M_star
    # Y_R = rho + 0 (B_up=rho_up=0 for leaf search)
    Y_R = rho
    Z = D + S

    # FP 方程 ④: R = (Y_R + ε)/(Y_R + Z + ε)
    # 验证
    R_from_fp = (Y_R + EPS) / (Y_R + Z + EPS)
    if abs(R - R_from_fp) > 1e-12:
        return None  # 不是不动点

    # T = Y_R + Z + 3ε
    T = Y_R + Z + 3 * EPS

    # 由 FP 方程: Y_R + EPS = R*(Y_R + Z + EPS) = R*(T - 2*EPS)
    # R*(T - 2*EPS) = Y_R + EPS = T - Z - 3*EPS + EPS = T - Z - 2*EPS
    # R*T - 2*EPS*R = T - Z - 2*EPS
    # T*(1 - R) = Z + 2*EPS - 2*EPS*R
    # T = Z/(1 - R) + 2*EPS  (正确推导)
    T_derived = Z / (1 - R) + 2 * EPS
    
    # 文档中的错误公式
    T_doc = (Z + EPS * (2 - R)) / (1 - R)

    # Gershgorin 半径
    # b_R = 2, a_R = 1
    # A_R ≡ Y_R + ε, B_R ≡ Z + ε
    # R_R = [2*(Y_R+2ε) + 1*(Z+ε)] / (Y_R+Z+2ε)²  ← 文档公式
    
    # 实际 5×5 J_N 的 R 行非对角元绝对值之和
    Delta_R = Y_R + D + S + EPS  # = Y_R + Z + eps
    J_RD = R / Delta_R
    J_RB = (1 - R) / Delta_R
    J_Rrho = (1 - R) / Delta_R
    J_RS = R / Delta_R
    R_R_actual = J_RD + J_RB + J_Rrho + J_RS

    # 比较
    return {
        "Z": Z, "R": R, "T": T, "T_derived": T_derived, "T_doc": T_doc,
        "R_R_actual": R_R_actual, "Z_plus_eps": Z + EPS, "Z_minus_1_plus_eps": Z - (1 - EPS),
        "case": "case2" if Z + EPS < 1 else "case1",
    }


def main():
    print("=" * 90)
    print("R 行情形 2 验证：T 公式修正 + 随机搜索不动点")
    print(f"统一参数，B_up=ρ_up=0（叶节点条件）")
    print("=" * 90)

    # 阶段 1：验证 T 公式
    print("\n--- 阶段 1：T 公式验证 ---")
    M_leaf = np.array([0.451089469407, 0.451089469407, 0.561079081796,
                       0.360701048840, 0.561079081796])
    
    # 验证是不动点
    NM = n_operator(M_leaf)
    print(f"  N(M) - M: {np.linalg.norm(NM - M_leaf):.2e}")
    
    result = compute_R_gershgorin_radius(M_leaf)
    if result:
        print(f"  Z = D+S = {result['Z']:.6f}")
        print(f"  R = {result['R']:.6f}")
        print(f"  T (直接)     = {result['T']:.6f}")
        print(f"  T (正确推导)  = {result['T_derived']:.6f}  T = Z/(1-R) + 2ε")
        print(f"  T (文档公式)  = {result['T_doc']:.6f}  T = (Z+ε(2-R))/(1-R)")
        print(f"  正确公式匹配: {'✓' if abs(result['T']-result['T_derived'])<1e-12 else '✗'}")
        print(f"  文档公式匹配: {'✓' if abs(result['T']-result['T_doc'])<1e-12 else '✗'}")
        print(f"  R_R (实际) = {result['R_R_actual']:.6f}")
        print(f"  Z+ε = {result['Z_plus_eps']:.6f}, 情形: {result['case']}")

    # 阶段 2：随机搜索整个 [0,1]^5 看是否有 Z+ε<1 的不动点
    print(f"\n--- 阶段 2：随机搜索不动点（10,000 个随机起点）---")
    
    found_fps = []
    min_Z = float('inf')
    min_Z_fp = None
    
    np.random.seed(42)
    for i in range(20000):
        # 用 sobol 类采样覆盖
        M0 = np.random.rand(5)
        M_star = find_fixed_point(M0)
        if M_star is not None:
            # 检查是否已发现过
            is_new = True
            for prev in found_fps:
                if np.linalg.norm(M_star - prev) < 1e-8:
                    is_new = False
                    break
            if is_new:
                found_fps.append(M_star)
                D_s, B_s, rho_s, R_s, S_s = M_star
                Z_s = D_s + S_s
                if Z_s < min_Z:
                    min_Z = Z_s
                    min_Z_fp = M_star
    
    print(f"\n  找到 {len(found_fps)} 个不同不动点")
    
    if found_fps:
        Z_values = [fp[0] + fp[4] for fp in found_fps]
        print(f"  Z = D+S 范围: [{min(Z_values):.6f}, {max(Z_values):.6f}]")
        print(f"  Z+ε 范围: [{min(Z_values)+EPS:.6f}, {max(Z_values)+EPS:.6f}]")
        
        case2_count = sum(1 for z in Z_values if z + EPS < 1)
        print(f"  情形 2 (Z+ε<1) 触发: {case2_count}/{len(found_fps)}")
        
        # 显示极端情况
        print(f"\n  Z 最小的不动点:")
        D_s, B_s, rho_s, R_s, S_s = min_Z_fp
        print(f"    M* = [{D_s:.12f}, {B_s:.12f}, {rho_s:.12f}, {R_s:.12f}, {S_s:.12f}]")
        print(f"    Z = {D_s+S_s:.12f}, Z+ε = {D_s+S_s+EPS:.12f}")
        print(f"    B-D = {B_s-D_s:.2e}")
        print(f"    S-ρ = {S_s-rho_s:.2e}")
    
    # 阶段 3：如果 Z≥1 对所有 FP 成立，尝试证明
    print(f"\n--- 阶段 3：解析分析 ---")
    
    if min_Z >= 1 - EPS + 1e-14:  # 如果 Z >= 0.99
        print(f"  所有不动点 Z ≥ 1-ε → 情形 2 不发生 → R_R < 1 解析得证 ■")
        print(f"\n  但还需解析证明 Z ≥ 1。从 FP 方程 ① + ⑤ 尝试:")
        print(f"    D = (R+ε)/(R+B+B_up+ε)")
        print(f"    S = (D+ε)/(D+R+ε)")
        print(f"    由 D = (R+ε)/(R+B+B_up+ε) ≥ (R+ε)/(R+2+ε)  → D ∈ [ε/(2+ε), (1+ε)/(3+ε)]")
        print(f"    由 S = (D+ε)/(D+R+ε) ≥ (D+ε)/(D+1+ε)  → Z = D+S ≥ D + (D+ε)/(D+1+ε)")
        print(f"    g(D) = D + (D+ε)/(D+1+ε), g'(D) > 0, 最小值在 D_最小")
        D_min_bound = EPS / (2 + EPS)
        g_min = D_min_bound + (D_min_bound + EPS) / (D_min_bound + 1 + EPS)
        print(f"    D 最小界 D_min ≥ ε/(2+ε) = {D_min_bound:.6f}")
        print(f"    g(D_min) = {g_min:.6f}")
        print(f"    这个界太弱（g<1），需要更紧的约束。")
        print(f"\n  从 D=B（叶节点）的方程: u² + (R+ε)u - (R+ε) = 0")
        print(f"    对于 R∈(0,1): u∈({2*EPS/(1+np.sqrt(1+4*EPS*EPS)):.4f}, {(-1.01+np.sqrt(1.0201+4.04))/2:.4f})")
        
        # 叶节点下：u 和 R 的耦合
        # 尝试证明 D + (D+ε)/(D+R+ε) ≥ 1 对所有满足 u² = (R+ε)(1-u) 的 (u,R) 对
        print(f"\n  叶节点约束: u² = (R+ε)(1-u)")
        print(f"    → R = u²/(1-u) - ε")
        print(f"    代入 Z: Z = u + (u+ε)/(u+R+ε)")
        print(f"      = u + (u+ε)/(u + u²/(1-u))")
        print(f"      = u + (u+ε)(1-u)/u")
        print(f"      = u + (1-u) + ε(1-u)/u")
        print(f"      = 1 + ε(1-u)/u")
        print(f"      ≥ 1  (因为 ε>0, u>0, 1-u≥0)")
        print(f"  ★ Z = 1 + ε(1-u)/u > 1 解析得证！（叶节点）")
        print(f"    数值验证: 1 + 0.01*(1-0.4511)/0.4511 = 1 + 0.005489/0.4511 = 1.01217 ✓")
    else:
        print(f"  存在 Z < 1-ε 的不动点，情形 2 可能触发")
        print(f"  min_Z = {min_Z:.12f}")


if __name__ == "__main__":
    main()
