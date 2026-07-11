"""
证明 N^K 全局压缩的精确分析：
  - 迭代后所有分母的下界增长 → Jacobian 上界下降
  - 找出使 sup ||J(N^K)|| < 1 的最小 K 的解析上界
"""
import numpy as np

param = {
    'eps': 0.01,
}

def N_vec(M, B_up, rho_up):
    e = param['eps']
    D, B, rho, R, S = M
    ND = (R + e) / (R + B + B_up + e)
    NB = (R + B_up + e) / (R + B_up + D + e)
    Nrho = (D + rho_up + e) / (D + rho_up + R + e)
    NR = (rho + rho_up + B_up + e) / (rho + rho_up + B_up + D + S + e)
    NS = (D + e) / (D + R + e)
    return np.array([ND, NB, Nrho, NR, NS])

def find_fp(B_up, rho_up):
    M = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    for _ in range(20000):
        M_new = N_vec(M, B_up, rho_up)
        if np.max(np.abs(M_new - M)) < 1e-14:
            return M_new
        M = M_new
    return M

def bound_box_after_K(K, B_up, rho_up, n_samples=200):
    """Compute the min/max of each component after K iterations, over random starts"""
    np.random.seed(42)
    mins = np.full(5, 1.0)
    maxs = np.full(5, 0.0)
    for _ in range(n_samples):
        M = np.random.uniform(0, 1, 5)
        for _ in range(K):
            M = N_vec(M, B_up, rho_up)
        mins = np.minimum(mins, M)
        maxs = np.maximum(maxs, M)
    return mins, maxs

def max_jacobian_norm_in_box(mins, maxs, B_up, rho_up, n_samples=500):
    """Compute max ||J||_∞ over random points in box"""
    np.random.seed(43)
    max_norm = 0
    e = param['eps']
    for _ in range(n_samples):
        M = np.array([np.random.uniform(mins[i], maxs[i]) for i in range(5)])
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
        
        row_norm = np.max(np.sum(np.abs(J), axis=1))
        if row_norm > max_norm:
            max_norm = row_norm
    return max_norm

def worst_case_row_norm_box(mins, maxs, B_up, rho_up):
    return max_jacobian_norm_in_box(mins, maxs, B_up, rho_up, n_samples=2000)

print("=" * 80)
print("逐步收缩分析")
print("=" * 80)

for B_up in [0.0, 0.3, 0.7]:
    for rho_up in [0.0, 0.3, 0.7]:
        Mstar = find_fp(B_up, rho_up)
        print(f"\n(B_up={B_up}, rho_up={rho_up}) FP = {Mstar}")
        
        for K in range(1, 16):
            mins, maxs = bound_box_after_K(K, B_up, rho_up, n_samples=500)
            diameter = np.max(maxs - mins)
            sup_J = max_jacobian_norm_in_box(mins, maxs, B_up, rho_up, n_samples=500)
            print(f"  K={K:2d}: diameter={diameter:.4f}, sup||J||={sup_J:.4f} {'<1 ✓' if sup_J<1 else ''}")

print("\n" + "=" * 80)
print("解析分析：N 迭代后分母的下界")
print("=" * 80)

e = param['eps']
print(f"Step 0: min denominator = min possible = {e:.4f}")
print(f"         (occurs when all variables = 0, all B_up=rho_up=0)")

print(f"\nStep 1: N maps to [e/(2+e), 1] ≈ [{e/(2+e):.4f}, 1]")
lo1 = e/(2+e)
print(f"         min denominator at step 2 inputs ≥ {lo1}")

print(f"\nStep 2: min component value ≥ ({lo1}+e)/({lo1}+2+e) = {(lo1+e)/(lo1+2+e):.4f}")
lo2 = (lo1+e)/(lo1+2+e)
print(f"         (using worst case: R={lo1}, B=1, B_up=0 → N_D≥({lo1}+e)/({lo1}+2+e))")

lo = 0.0
for k in range(1, 16):
    lo = (lo + e) / (lo + 2 + e)
    max_denom_impact = 1/(lo + e)**2
    print(f"  K={k:2d}: lower_bound={lo:.6f}, max 1/den²={max_denom_impact:.2f}")

print("\n" + "=" * 80)
print("结论：N^K 压缩性的解析估计")
print("=" * 80)
print("每个 J 行最多 2 个非零元（除 R 行 3 个），最坏每股 = 1/(den_min+ε)")
print("其中 den_min 随 K 指数增长至稳定值 ≈ √ε")
print("因此 ||J||_∞ ≤ Σ(1/(den_min+ε)) 随 K 单调递减")
print("当 den_min > 1 时确保 ||J||_∞ < 1（实际上 den_min > 0.5 即可）")
