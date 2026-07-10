"""
修复 D_low 违规: 用可证明的保守下界
=====================================
使用 m_low^(0) = a_k / (a_k+b_k+ε_k+Σ(w+v)) (首次迭代下界, 可证明正确)
而非迭代收敛的 m_low (可能稍有高估)
"""
import numpy as np

def n_operator(M, a, b, eps, W, V):
    num = a + W @ M
    den = num + b + V @ M + eps
    return num / den

def compute_fp(a,b,eps,W,V):
    M = np.full(5, 0.5)
    for _ in range(20000):
        Mn = n_operator(M, a, b, eps, W, V)
        if np.max(np.abs(Mn - M)) < 1e-15:
            return Mn
        M = Mn
    return M

def gen_FCA(seed):
    rs = np.random.RandomState(seed % (2**31))
    a = rs.uniform(0.01, 0.5, 5)
    b = rs.uniform(0.01, 0.5, 5)
    e = rs.uniform(0.001, 0.1, 5)
    W = rs.uniform(0.01, 0.3, (5, 5))
    V = rs.uniform(0.01, 0.3, (5, 5))
    np.fill_diagonal(W, 0.0)
    np.fill_diagonal(V, 0.0)
    t = a.sum() + b.sum() + W.sum() + V.sum()
    W *= 5.0 / t
    V *= 5.0 / t
    return a, b, e, W, V

def D_low_provable(a, b, e, W, V):
    """可证明正确的 D_low:
    m_low_k = a_k / D_max_k  (M* lower bound, provably correct)
    D_low_k = a_k+b_k+ε_k + Σ(w+v)·m_low"""
    D_max = a + b + e + np.sum(W + V, axis=1)
    m_low = a / D_max  # provable lower bound on M*
    D_low = a + b + e + (W + V) @ m_low
    return m_low, D_low

def iterative_Mstar_lower(a, b, e, W, V, max_iter=20):
    """迭代收敛下界 (可能稍有高估)"""
    D_max = a + b + e + np.sum(W + V, axis=1)
    m_low = np.clip(a / D_max, 0, 1)
    for it in range(max_iter):
        D_high = a + b + e + (W + V) @ np.ones(5)
        A_low = a + W @ m_low
        m_low_new = np.clip(A_low / D_high, 0, 1)
        if np.max(np.abs(m_low_new - m_low)) < 1e-14:
            break
        m_low = m_low_new
    D_low = a + b + e + (W + V) @ m_low
    return m_low, D_low

print("=" * 70)
print("D_low 可证明版本 vs 迭代版本")
print("=" * 70)

# Compare for seed 149 and seed 11
for seed_id in [149, 11, 21]:
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    
    m_provable, D_provable = D_low_provable(a, b, e, W, V)
    m_iter, D_iter = iterative_Mstar_lower(a, b, e, W, V)
    
    Dstar = a + b + e + (W+V) @ Mstar
    
    print(f"\nseed {seed_id}: M* = {Mstar}")
    print(f"  m_provable = [{', '.join(f'{x:.4f}' for x in m_provable)}]")
    print(f"  m_iter     = [{', '.join(f'{x:.4f}' for x in m_iter)}]")
    print(f"  D*/D_provable = [{', '.join(f'{x:.2f}x' for x in Dstar/D_provable)}]")
    print(f"  D*/D_iter     = [{', '.join(f'{x:.2f}x' for x in Dstar/D_iter)}]")
    
    # Orbit test
    M = np.full(5, 0.5)
    for t in range(5):
        D = a + b + e + (W+V) @ M
        ratio_prov = np.min(D / D_provable)
        ratio_iter = np.min(D / D_iter)
        show = "✗" if ratio_iter < 1 else "✓"
        print(f"  t={t}: min(D/D_prov)={ratio_prov:.2f}  min(D/D_iter)={ratio_iter:.4f} {show}")
        M = n_operator(M, a, b, e, W, V)

# ============================================================
# 批量: 可证明 D_low vs 迭代 D_low
# ============================================================
print(f"\n{'='*70}")
print("批量 200 种子: 两种 D_low 的轨道下限 + α_bound 对比")
print("=" * 70)

violations_provable = 0
violations_iter = 0
worst_alpha_prov = 0
worst_alpha_iter = 0
worst_seed_prov = 0
worst_seed_iter = 0

for seed_id in range(200):
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W+V) @ Mstar
    
    m_prov, D_prov = D_low_provable(a, b, e, W, V)
    m_iter, D_iter = iterative_Mstar_lower(a, b, e, W, V)
    
    J = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k != j:
                J[k,j] = (W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k]
    
    # α with provable D_low
    gamma_prov = Dstar / D_prov
    alpha_prov = max([sum(abs(J[:,j]) * gamma_prov) for j in range(5)])
    
    # α with iterative D_low
    gamma_iter = Dstar / D_iter
    alpha_iter = max([sum(abs(J[:,j]) * gamma_iter) for j in range(5)])
    
    if alpha_prov > worst_alpha_prov:
        worst_alpha_prov = alpha_prov
        worst_seed_prov = seed_id
    
    if alpha_iter > worst_alpha_iter:
        worst_alpha_iter = alpha_iter
        worst_seed_iter = seed_id
    
    # Orbit test
    M = np.full(5, 0.5)
    for t in range(100):
        D = a + b + e + (W+V) @ M
        if np.min(D / D_prov) < 1:
            violations_provable += 1
            break
        if np.min(D / D_iter) < 1:
            violations_iter += 1
            break
        M_next = n_operator(M, a, b, e, W, V)
        if np.max(np.abs(M_next - M)) < 1e-10:
            break
        M = M_next

print(f"  D_low可证明: α_max = {worst_alpha_prov:.4f} (seed {worst_seed_prov}), 轨道违规 {violations_provable}/200")
print(f"  D_low迭代:   α_max = {worst_alpha_iter:.4f} (seed {worst_seed_iter}), 轨道违规 {violations_iter}/200")
print(f"  α可证明 < 1: {'✓' if worst_alpha_prov < 1 else '✗'}")
print(f"  α迭代   < 1: {'✓' if worst_alpha_iter < 1 else '✗'}")

# ============================================================
# 详细展示最劣种子
# ============================================================
print(f"\n  --- 最劣可证明种子 (seed {worst_seed_prov}) ---")
a, b, e, W, V = gen_FCA(worst_seed_prov)
Mstar = compute_fp(a, b, e, W, V)
Dstar = a + b + e + (W+V) @ Mstar
m_prov, D_prov = D_low_provable(a, b, e, W, V)

J = np.zeros((5,5))
for k in range(5):
    for j in range(5):
        if k != j:
            J[k,j] = (W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k]

gamma = Dstar / D_prov
col_sums = [sum(abs(J[:,j]) * gamma) for j in range(5)]
print(f"  γ = D*/D_prov = {gamma}")
print(f"  列和 = {[f'{x:.4f}' for x in col_sums]}")
print(f"  α = {max(col_sums):.4f}")
