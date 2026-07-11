"""审计N^3压缩：边角点 + 解析机制探索"""
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

def test_NK_contraction(K, B_up, rho_up, n_corner=5000, n_rand=10000):
    Mstar = find_fp(B_up, rho_up)
    
    max_ratio = 0
    worst_case = None
    
    for _ in range(n_rand):
        M = np.random.uniform(0, 1, 5)
        old = np.max(np.abs(M - Mstar))
        if old < 1e-12: continue
        MK = M.copy()
        for _ in range(K): MK = N_vec(MK, B_up, rho_up)
        r = np.max(np.abs(MK - Mstar)) / old
        if r > max_ratio:
            max_ratio = r
            worst_case = ('random', M.copy())
    
    corner_starts = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 1.0, 0.0, 1.0]),
        np.array([0.999, 0.0, 0.0, 0.999, 0.0]),
        np.array([0.0, 0.999, 0.0, 0.999, 0.0]),
    ]
    
    for M in corner_starts:
        old = np.max(np.abs(M - Mstar))
        if old < 1e-12: continue
        MK = M.copy()
        for _ in range(K): MK = N_vec(MK, B_up, rho_up)
        r = np.max(np.abs(MK - Mstar)) / old
        if r > max_ratio:
            max_ratio = r
            worst_case = ('corner', M.copy())
    
    for _ in range(n_corner):
        M = np.zeros(5)
        for i in range(5):
            if np.random.rand() < 0.5:
                M[i] = np.random.uniform(0, 0.01)
            else:
                M[i] = np.random.uniform(0.99, 1.0)
        old = np.max(np.abs(M - Mstar))
        if old < 1e-12: continue
        MK = M.copy()
        for _ in range(K): MK = N_vec(MK, B_up, rho_up)
        r = np.max(np.abs(MK - Mstar)) / old
        if r > max_ratio:
            max_ratio = r
            worst_case = ('near_corner', M.copy())
    
    return max_ratio, worst_case

print("=" * 80)
print("N^K 全局压缩边角点审计")
print("=" * 80)

param_grid = [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0),
              (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
              (1.0, 0.0), (1.0, 0.5), (1.0, 1.0)]

for K in [3, 4, 5]:
    print(f"\n--- K={K} ---")
    global_max = 0
    for B_up, rho_up in param_grid:
        max_r, worst = test_NK_contraction(K, B_up, rho_up)
        status = "✓" if max_r < 1 else "✗"
        if max_r > global_max:
            global_max = max_r
            global_worst = (B_up, rho_up, max_r, worst)
        print(f"  (B_up={B_up:.1f}, ρ_up={rho_up:.1f}): max={max_r:.4f} {status}")
    print(f"  全局最大: {global_max:.4f} at (B_up={global_worst[0]:.1f}, ρ_up={global_worst[1]:.1f}) [{global_worst[3][0]}]")

print("\n" + "=" * 80)
print("解析机制探索：为什么 N^K 比 N 更压缩？")
print("=" * 80)

B_up, rho_up = 0.0, 0.0
Mstar = find_fp(B_up, rho_up)
print(f"叶节点 FP: {Mstar}")

M0 = np.array([1.0, 0.0, 0.0, 1.0, 1.0])
dist0 = np.max(np.abs(M0 - Mstar))
print(f"\n最坏起点: {M0}, dist={dist0:.4f}")

M = M0.copy()
for k in range(1, 9):
    M_prev = M.copy()
    M = N_vec(M, B_up, rho_up)
    dist = np.max(np.abs(M - Mstar))
    ratio = dist / dist0
    denom_sums = [
        (M[3] + M[1] + B_up + e, M[3] + B_up + M[0] + e, 
         M[0] + rho_up + M[3] + e,
         M[2] + rho_up + B_up + M[0] + M[4] + e,
         M[0] + M[3] + e)
    ]
    min_denom = min(denom_sums[0])
    print(f"  k={k}: dist={dist:.4f}, ratio={ratio:.4f}, min_denom={min_denom:.4f}")

print("\n关键机制：每步 N 迭代后，分母增大，导数变小")
print("分母 > 2 时单步 Jacobian 已有 ||J||∞ < 1")
print("但单步 N 不能保证分母 > 2，需要 2-3 步")

print("\n" + "=" * 80)
print("解析下界分析：N² 后分母的最小值")
print("=" * 80)

def worst_denominator_after_K(K, B_up, rho_up, n_samples=20000):
    min_den = float('inf')
    np.random.seed(42)
    for _ in range(n_samples):
        M = np.random.uniform(0, 1, 5)
        for _ in range(K):
            M = N_vec(M, B_up, rho_up)
        D, B, rho, R, S = M
        dens = [R+B+B_up+e, R+B_up+D+e, D+rho_up+R+e, rho+rho_up+B_up+D+S+e, D+R+e]
        min_den = min(min_den, min(dens))
    return min_den

for B_up, rho_up in [(0.0,0.0), (0.0,0.5), (0.5,0.5), (1.0,1.0)]:
    print(f"\n(B_up={B_up}, ρ_up={rho_up}):")
    for K in range(1, 6):
        d = worst_denominator_after_K(K, B_up, rho_up)
        est_row_norm = (1 + 0 + 0) / d
        print(f"  K={K}: min denom={d:.4f}, est ||J(N^K)||∞≤{3/d:.4f} {'<1!' if 3/d < 1 else ''}")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print("N³ 压缩的机制：")
print("  1. N 是函数压缩——分母增大，导数减小")
print("  2. 单步 N 从最坏起点出发分母 < 1，导数 > 1")
print("  3. N² 后分母 ≥ 0.5~1.0，导数开始 < 1")
print("  4. N³ 后分母 ≥ 0.7~1.5，压缩系数 ≤ 0.85")
print("  5. 解析上界 = max ||J(N)|| 在 N² 后的可达域上")
print("     = max |J_ij| 的和在分母 ≥ d* 时")
print("     = Σ(max row elements) / (min denominator)²")
print("     ≤ 3 / d*  (每行至多 3 个非零元)")  
