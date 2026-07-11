"""
探索全局收敛的三条路线：
  路线A：N 在 [0,1]^5 上是否整体压缩（sup ||J|| < 1）
  路线B：是否存在 Lyapunov 函数 V(M) 使 V(N(M)) < V(M)
  路线C：单调迭代（上解/下解 sandwich）
"""
import numpy as np

param = {
    'alpha1': 1.0, 'beta1': 1.0,
    'gamma1': 1.0, 'delta1': 1.0,
    'zeta1': 1.0, 'eta1': 1.0,
    'theta1': 1.0, 'kappa1': 1.0, 'kappa2': 1.0,
    'lambda1': 1.0, 'mu1': 1.0,
    'eps1': 0.01, 'eps2': 0.01, 'eps3': 0.01, 'eps4': 0.01, 'eps5': 0.01,
}

def N_vec(M, B_up, rho_up):
    D, B, rho, R, S = M
    ND = (R + param['eps1']) / (R + B + B_up + param['eps1'])
    NB = (R + B_up + param['eps2']) / (R + B_up + D + param['eps2'])
    Nrho = (D + rho_up + param['eps3']) / (D + rho_up + R + param['eps3'])
    NR = (rho + rho_up + B_up + param['eps4']) / (rho + rho_up + B_up + D + S + param['eps4'])
    NS = (D + param['eps5']) / (D + R + param['eps5'])
    return np.array([ND, NB, Nrho, NR, NS])

def J_num(M, B_up, rho_up, h=1e-8):
    J = np.zeros((5, 5))
    f0 = N_vec(M, B_up, rho_up)
    for i in range(5):
        Mh = M.copy()
        Mh[i] += h
        J[:, i] = (N_vec(Mh, B_up, rho_up) - f0) / h
    return J

def find_fp(B_up, rho_up):
    M = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    for _ in range(20000):
        M_new = N_vec(M, B_up, rho_up)
        if np.max(np.abs(M_new - M)) < 1e-14:
            return M_new
        M = M_new
    return M

print("=" * 80)
print("路线A：N 在 [0,1]^5 上是整体压缩？")
print("  (sup_{M∈[0,1]^5} ||J(M)||_∞ < 1 ?)")
print("=" * 80)

np.random.seed(42)
max_norm = 0
max_point = None
max_B_up = max_rho_up = None

for trial in range(10000):
    M = np.random.uniform(0, 1, 5)
    B_up = np.random.uniform(0, 1)
    rho_up = np.random.uniform(0, 1)
    J = J_num(M, B_up, rho_up)
    row_norm = np.max(np.sum(np.abs(J), axis=1))
    if row_norm > max_norm:
        max_norm = row_norm
        max_point = M
        max_B_up, max_rho_up = B_up, rho_up

print(f"\n10000 随机点结果:")
print(f"  sup ||J||_∞ = {max_norm:.4f}")
print(f"  位置: M={max_point}, B_up={max_B_up:.3f}, ρ_up={max_rho_up:.3f}")

sweep_max = 0
sweep_point = None
for D in np.linspace(0, 1, 6):
    for B in np.linspace(0, 1, 6):
        for rho in np.linspace(0, 1, 6):
            for R in np.linspace(0, 1, 6):
                for S in np.linspace(0, 1, 6):
                    for B_up in np.linspace(0, 1, 4):
                        for rho_up in np.linspace(0, 1, 4):
                            M = np.array([D, B, rho, R, S])
                            J = J_num(M, B_up, rho_up)
                            row_norm = np.max(np.sum(np.abs(J), axis=1))
                            if row_norm > sweep_max:
                                sweep_max = row_norm
                                sweep_point = (M, B_up, rho_up)

print(f"\n网格扫参 (6^5 × 4^2 = 124,416 点):")
print(f"  sup ||J||_∞ = {sweep_max:.4f}")
if sweep_max >= 1:
    print(f"  → 全局压缩失败！存在 ||J||≥1 的点")
else:
    print(f"  → 全局压缩可能成立？需要进一步探索")

print("\n" + "=" * 80)
print("路线B：Lyapunov 函数 V(M) = ||M - M*||_∞")
print("  检查 ∀M∈[0,1]^5: ||N(M)-M*||_∞ < ||M-M*||_∞ ?")
print("=" * 80)

for B_up in [0.0, 0.3, 0.7]:
    for rho_up in [0.0, 0.3, 0.7]:
        Mstar = find_fp(B_up, rho_up)
        
        max_ratio = 0
        violator = None
        
        for _ in range(5000):
            M = np.random.uniform(0, 1, 5)
            if np.max(np.abs(M - Mstar)) < 1e-8:
                continue
            
            M_next = N_vec(M, B_up, rho_up)
            new_dist = np.max(np.abs(M_next - Mstar))
            old_dist = np.max(np.abs(M - Mstar))
            ratio = new_dist / old_dist
            
            if ratio > max_ratio:
                max_ratio = ratio
                violator = M
        
        print(f"\n  (B_up={B_up}, ρ_up={rho_up})")
        print(f"    max ratio = {max_ratio:.4f}")
        if max_ratio >= 1.0:
            print(f"    ✗ Lyapunov 失败！存在点使距离不严格下降")
        else:
            print(f"    ✓ 距离单调递减")

print("\n" + "=" * 80)
print("路线B-2：Lyapunov V(M) = ||M - N(M)||_∞ （残差）")
print("  检查 ∀M∈[0,1]^5: ||N(M)-N(N(M))|| < ||M-N(M)|| ?")
print("=" * 80)

for B_up in [0.0, 0.3, 0.7]:
    for rho_up in [0.0, 0.3, 0.7]:
        max_ratio = 0
        for _ in range(2000):
            M = np.random.uniform(0, 1, 5)
            r0 = M - N_vec(M, B_up, rho_up)
            r0_norm = np.max(np.abs(r0))
            if r0_norm < 1e-10:
                continue
            
            M2 = N_vec(N_vec(M, B_up, rho_up), B_up, rho_up)
            r1 = M2 - N_vec(M, B_up, rho_up)
            r1_norm = np.max(np.abs(r1))
            ratio = r1_norm / r0_norm
            if ratio > max_ratio:
                max_ratio = ratio
        
        print(f"  (B_up={B_up}, ρ_up={rho_up}): max ||res(N(M))||/||res(M)|| = {max_ratio:.4f}")

print("\n" + "=" * 80)
print("路线C：单调迭代 sandwich")
print("  上解 M⁺ = (1,1,1,1,1) 和下解 M⁻ = (0,0,0,0,0) 的迭代序列")
print("=" * 80)

for B_up in [0.0, 0.3, 0.7]:
    for rho_up in [0.0, 0.3, 0.7]:
        Mstar = find_fp(B_up, rho_up)
        
        M_upper = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        M_lower = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        converges = True
        for _ in range(100):
            M_upper_new = N_vec(M_upper, B_up, rho_up)
            M_lower_new = N_vec(M_lower, B_up, rho_up)
            
            if np.any(M_upper_new > M_upper) or np.any(M_lower_new < M_lower):
                converges = False
            
            M_upper, M_lower = M_upper_new, M_lower_new
        
        Mstar_u = find_fp(B_up, rho_up)  # already have this
        # Actually let me iterate M_upper and M_lower many times
        Mu = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        Ml = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        for _ in range(5000):
            Mu = N_vec(Mu, B_up, rho_up)
            Ml = N_vec(Ml, B_up, rho_up)
        
        dist_upper = np.max(np.abs(Mu - Mstar))
        dist_lower = np.max(np.abs(Ml - Mstar))
        
        print(f"  (B_up={B_up}, ρ_up={rho_up}):")
        print(f"    上解收敛到 FP: dist={dist_upper:.2e}")
        print(f"    下解收敛到 FP: dist={dist_lower:.2e}")
        print(f"    上下解同目标: {'YES' if np.max(np.abs(Mu-Ml)) < 1e-10 else 'NO'}")

print("\n" + "=" * 80)
print("初步结论")
print("=" * 80)
print("路线A（全局压缩）：取决于 sup ||J||_∞")
print("路线B（Lyapunov ||M-M*||）：需要验证")
print("路线C（单调迭代 sandwich）：最可行——若上下解收敛到同一点")
