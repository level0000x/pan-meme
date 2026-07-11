"""路线4：两阶段论证
阶段1：N² 将任意点映射进 M* 的局部吸引盆
阶段2：在吸引盆内，局部收敛定理接管
"""
import numpy as np

e = 0.01

def N_vec(M, B_up, rho_up):
    D, B, rho, R, S = M
    ND = (R + e) / (R + B + B_up + e)
    NB = (R + B_up + e) / (R + B_up + D + e)
    Nrho = (D + rho_up + e) / (D + rho_up + R + e)
    NR = (rho + rho_up + B_up + e) / (rho + rho_up + B_up + D + S + e)
    NS = (D + e) / (D + R + e)
    return np.array([ND, NB, Nrho, NR, NS])

def find_fp(B_up, rho_up):
    M = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    for _ in range(30000):
        M_new = N_vec(M, B_up, rho_up)
        if np.max(np.abs(M_new - M)) < 1e-15:
            return M_new
        M = M_new
    return M

def J_at(M, B_up, rho_up):
    D, B, rho, R, S = M
    denD = R + B + B_up + e
    denB = R + B_up + D + e
    denRho = D + rho_up + R + e
    denR = rho + rho_up + B_up + D + S + e
    denS = D + R + e
    J = np.zeros((5,5))
    J[0,1] = -D/denD
    J[0,3] = (1-D)/denD
    J[1,0] = -B/denB
    J[1,3] = (1-B)/denB
    J[2,0] = (1-rho)/denRho
    J[2,3] = -rho/denRho
    J[3,0] = -R/denR
    J[3,2] = (1-R)/denR
    J[3,4] = -R/denR
    J[4,0] = (1-S)/denS
    J[4,3] = -S/denS
    return J

print("=" * 80)
print("两阶段论证：N²映射 + 局部盆地")
print("=" * 80)
print("关键问题：存在 r > 0 使 N²([0,1]^5 \\ B_r(M*)) ⊆ B_r(M*)?")
print("若成立：每个点最多2步进入盆地，局部收敛接管")
print()

test_params = [(0.0, 0.0), (0.0, 0.3), (0.3, 0.0), (0.5, 0.5)]

for B_up, rho_up in test_params:
    Mstar = find_fp(B_up, rho_up)
    rhoJ = max(abs(np.linalg.eigvals(J_at(Mstar, B_up, rho_up))))
    print(f"\n(B_up={B_up}, rho_up={rho_up}): M*={[f'{x:.4f}' for x in Mstar]}, ρ(J)={rhoJ:.4f}")
    
    basil_size = 0.5 * (1 - rhoJ)
    print(f"  局部盆地估计半径: {basil_size:.4f} (保守)")
    
    max_dist_after_N2 = 0
    worst = None
    np.random.seed(42)
    for _ in range(5000):
        M = np.random.uniform(0, 1, 5)
        if np.max(np.abs(M - Mstar)) < basil_size:
            continue
        M2 = N_vec(N_vec(M, B_up, rho_up), B_up, rho_up)
        d2 = np.max(np.abs(M2 - Mstar))
        if d2 > max_dist_after_N2:
            max_dist_after_N2 = d2
            worst = M
    
    print(f"  盆地外 N² 最大距离: {max_dist_after_N2:.4f}")
    print(f"  盆地半径: {basil_size:.4f}")
    print(f"  N²(外部) ⊆ 盆地? {'YES!' if max_dist_after_N2 < basil_size else 'NO'}")
    
    r_critical = max_dist_after_N2
    print(f"  临界半径 r* = {r_critical:.4f}")
    
    safe = 0
    n_test = 5000
    for _ in range(n_test):
        M = np.random.uniform(0, 1, 5)
        d = np.max(np.abs(M - Mstar))
        if d < r_critical:
            continue
        M2 = N_vec(N_vec(M, B_up, rho_up), B_up, rho_up)
        d2 = np.max(np.abs(M2 - Mstar))
        if d2 < r_critical:
            safe += 1
    print(f"  shell(r*<d≤1) → B_r*(M*) 概率: {safe}/{n_test} ({100*safe/max(1,n_test):.1f}%)")

print("\n" + "=" * 80)
print("结论：两阶段论证的关键瓶颈")
print("  阶段1需要证明 N² 的像集严格在某个半径 r 内")
print("  阶段2需要在 B_r 内证明 N 是压缩")
print("  若 r 能解析算出 → 全局收敛 ■")
print("  若 r 只能数值估计 → 全局收敛 ◆")
print("=" * 80)

print("\n" + "=" * 80)
print("解析 r 的构造尝试：利用 N 的单个迭代下界")
print("=" * 80)
print("N_D = (R+e)/(R+B+B_up+e) ≥ (a+e)/(1+b+e)")
print("其中 a = min component, b = max component")
print()
print("N² 后最小组件的下界可以通过两轮迭代的精细分析得到")
print("如果该下界 → 能推出 N² 的像集半径 → 闭合证明")
print("=" * 80)
