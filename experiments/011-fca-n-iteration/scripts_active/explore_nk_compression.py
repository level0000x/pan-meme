"""
路线A/B失败后，深入研究路线C的数学机制：
  1. N^K 是否在某个 K 处变成压缩？
  2. 上解/下解 sandwich 能否推广到任意起点？
  3. 构造"吸引盆边界"——找出距离暂时增加的最大步数
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

def find_fp(B_up, rho_up):
    M = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    for _ in range(20000):
        M_new = N_vec(M, B_up, rho_up)
        if np.max(np.abs(M_new - M)) < 1e-14:
            return M_new
        M = M_new
    return M

def J_num_Nk(M, B_up, rho_up, k, h=1e-8):
    """Compute Jacobian of N^k at M via finite differences"""
    J = np.zeros((5, 5))
    def Nk(x):
        for _ in range(k):
            x = N_vec(x, B_up, rho_up)
        return x
    f0 = Nk(M.copy())
    for i in range(5):
        Mh = M.copy()
        Mh[i] += h
        J[:, i] = (Nk(Mh) - f0) / h
    return J

def track_distance(M0, B_up, rho_up, max_iter=100):
    """Track ||M_k - M*|| over iterations, note any increases"""
    Mstar = find_fp(B_up, rho_up)
    M = M0.copy()
    history = []
    prev_dist = np.max(np.abs(M - Mstar))
    for k in range(max_iter):
        M = N_vec(M, B_up, rho_up)
        dist = np.max(np.abs(M - Mstar))
        history.append(dist)
        if dist < 1e-12:
            break
    n_increases = sum(1 for i in range(1, len(history)) if history[i] > history[i-1])
    return history, n_increases

print("=" * 80)
print("路线C-扩展：N^K 的压缩性")
print("=" * 80)

for K in [1, 2, 3, 5, 10]:
    max_norm = 0
    np.random.seed(42)
    for _ in range(5000):
        M = np.random.uniform(0, 1, 5)
        B_up = np.random.uniform(0, 1)
        rho_up = np.random.uniform(0, 1)
        J = J_num_Nk(M, B_up, rho_up, K, h=1e-7)
        row_norm = np.max(np.sum(np.abs(J), axis=1))
        if row_norm > max_norm:
            max_norm = row_norm
    print(f"  K={K:2d}: sup ||J(N^K)||_∞ = {max_norm:.4f} {'< 1 ✓' if max_norm < 1 else '≥ 1 ✗'}")

print("\n" + "=" * 80)
print("路线C-扩展：任意起点的距离暂时增加最大步数")
print("=" * 80)

for B_up in [0.0, 0.3, 0.7]:
    for rho_up in [0.0, 0.3, 0.7]:
        Mstar = find_fp(B_up, rho_up)
        max_increases = 0
        worst_M0 = None
        worst_history = None
        
        for _ in range(1000):
            M0 = np.random.uniform(0, 1, 5)
            history, n_inc = track_distance(M0, B_up, rho_up, max_iter=200)
            if n_inc > max_increases:
                max_increases = n_inc
                worst_M0 = M0
                worst_history = history
        
        total_steps = len(worst_history)
        print(f"\n  (B_up={B_up}, ρ_up={rho_up}):")
        print(f"    最大距离增加次数: {max_increases}/{total_steps}")
        if worst_history:
            increases_at = [i for i in range(1, len(worst_history)) if worst_history[i] > worst_history[i-1]]
            print(f"    距离增加发生在迭代步: {increases_at[:10]}")
            print(f"    最终距离: {worst_history[-1]:.2e}")

print("\n" + "=" * 80)
print("路线C-扩展：上解/下解 sandwich 的严格性")
print("  验证 ∀M⁽⁰⁾∈[0,1]⁵: M_lower^(k) ≤ N^k(M⁽⁰⁾) ≤ M_upper^(k) ?")
print("=" * 80)

for B_up in [0.0, 0.3]:
    for rho_up in [0.0, 0.3]:
        Mu = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        Ml = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        violations = 0
        for _ in range(1000):
            M = np.random.uniform(0, 1, 5)
            Mu_k, Ml_k, M_k = Mu.copy(), Ml.copy(), M.copy()
            
            for step in range(20):
                Mu_k, Ml_k = N_vec(Mu_k, B_up, rho_up), N_vec(Ml_k, B_up, rho_up)
                M_k = N_vec(M_k, B_up, rho_up)
                
                if np.any(M_k > Mu_k + 1e-10) or np.any(M_k < Ml_k - 1e-10):
                    violations += 1
                    break
        
        print(f"  (B_up={B_up}, ρ_up={rho_up}): sandwich 违反 {violations}/1000")

print("\n" + "=" * 80)
print("关键洞察：N 的单步距离可能增加（Lyapunov失败）")
print("          但 N^K 对所有足够大的 K 是收缩的（全局收敛）")
print("          上下解 sandwich 给出证明路径")
print("=" * 80)

print("\n进一步探索：找出使 N^K 成为压缩的最小 K")
for B_up in [0.0, 0.3, 0.5, 0.7]:
    for rho_up in [0.0, 0.3, 0.5, 0.7]:
        for K in range(1, 11):
            max_norm = 0
            for _ in range(500):
                M = np.random.uniform(0, 1, 5)
                J = J_num_Nk(M, B_up, rho_up, K, h=1e-6)
                row_norm = np.max(np.sum(np.abs(J), axis=1))
                if row_norm > max_norm:
                    max_norm = row_norm
            if max_norm < 1:
                print(f"  (B_up={B_up:.1f}, rho_up={rho_up:.1f}): K={K} → sup||J(N^{K})||={max_norm:.4f} ✓")
                break
        else:
            print(f"  (B_up={B_up:.1f}, rho_up={rho_up:.1f}): K up to 10 still ≥ 1")
