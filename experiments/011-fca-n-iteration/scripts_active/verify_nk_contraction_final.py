"""
终极验证：N^K 全局压缩的全面测试
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

def J_Nk(M, B_up, rho_up, K, h=1e-7):
    def Nk(x):
        for _ in range(K):
            x = N_vec(x, B_up, rho_up)
        return x
    J = np.zeros((5,5))
    f0 = Nk(M.copy())
    for i in range(5):
        Mh = M.copy()
        Mh[i] += h
        fh = Nk(Mh)
        J[:,i] = (fh - f0) / h
    return J

print("=" * 80)
print("全面 N^K 压缩性测试 (粗网格)")
print("=" * 80)

M_grid_coarse = [0.0, 0.3, 0.7, 1.0]
bp_grid_coarse = [0.0, 0.3, 0.7, 1.0]

total_coarse = 4**5 * 4**2
print(f"  网格点数: {total_coarse}")

for K in [2, 3, 4, 5, 10]:
    max_norm = 0
    worst = None
    
    for D in M_grid_coarse:
        for B in M_grid_coarse:
            for rho in M_grid_coarse:
                for R in M_grid_coarse:
                    for S in M_grid_coarse:
                        for B_up in bp_grid_coarse:
                            for rho_up in bp_grid_coarse:
                                M = np.array([D, B, rho, R, S])
                                J = J_Nk(M, B_up, rho_up, K)
                                norm = np.max(np.sum(np.abs(J), axis=1))
                                if norm > max_norm:
                                    max_norm = norm
                                    worst = (M, B_up, rho_up)
    
    print(f"  K={K:2d}: sup||J(N^{K})||_∞ = {max_norm:.4f} {'✓ <1' if max_norm < 1 else '✗ ≥1'}")

print("\n" + "=" * 80)
print("随机 200K 点测试 N^K 压缩性")
print("=" * 80)

np.random.seed(42)
for K in [3, 5, 10]:
    max_norm = 0
    for _ in range(200000):
        M = np.random.uniform(0, 1, 5)
        B_up = np.random.uniform(0, 1)
        rho_up = np.random.uniform(0, 1)
        J = J_Nk(M, B_up, rho_up, K)
        norm = np.max(np.sum(np.abs(J), axis=1))
        if norm > max_norm:
            max_norm = norm
    print(f"  K={K}: 200K random, sup||J(N^{K})||_∞ = {max_norm:.4f} {'✓ <1' if max_norm < 1 else '✗ ≥1'}")

print("\n" + "=" * 80)
print("最终结论")
print("=" * 80)
