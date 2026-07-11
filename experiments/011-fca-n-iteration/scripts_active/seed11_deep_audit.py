"""
深层审计: 问题追踪
====================
发现 1: 随机起点有 5/200 D_low 违规
发现 2: B 测试中 D_low 在随机采样中有大量违规 (139/200)
根因分析 + 修正方案
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

# ============================================================
# 问题 1: m^(0) 是否对所有种子 ≤ 0.5?
# ============================================================
print("=" * 70)
print("问题 1: m^(0)_k 是否对所有种子 ≤ 0.5?")
print("=" * 70)

max_mlow = 0
for seed_id in range(200):
    a, b, e, W, V = gen_FCA(seed_id)
    D_max = a + b + e + np.sum(W + V, axis=1)
    m_low = a / D_max
    current_max = np.max(m_low)
    if current_max > max_mlow:
        max_mlow = current_max
        if current_max > 0.3:
            print(f"  seed {seed_id}: max m_low = {current_max:.4f}  m_low = {m_low}")

print(f"  全局 max m_low = {max_mlow:.4f}")
print(f"  结论: {'OK — 全部 ≤ 0.5' if max_mlow <= 0.5 else 'PROBLEM!'}")

# ============================================================
# 问题 2: D_low 对外部点 M 不成立，什么条件下成立?
# ============================================================
print(f"\n{'='*70}")
print("问题 2: D_low ≤ D(M) 的条件")
print("=" * 70)

# D_k(M) = a_k+b_k+ε_k + Σ_j(w+v)_kj M_j
# D_low = a_k+b_k+ε_k + Σ_j(w+v)_kj m_j^(0)
# So D_k(M) ≥ D_low 成立当 M_j ≥ m_j^(0) ∀j

print("  引理: 若 M_j ≥ m_j^(0) ∀j, 则 D_k(M) ≥ D_low,k ∀k")
print("  反例: 若存在 j 使 M_j < m_j^(0), 则 D_k(M) 可能 < D_low,k")
print()

# ============================================================
# 问题 3: M(0) = 0.5 是否 ≥ m^(0)?
# ============================================================
print("=" * 70)
print("问题 3: M_j(0) = 0.5 ≥ m_j^(0) ∀j?")
print("=" * 70)

for seed_id in range(200):
    a, b, e, W, V = gen_FCA(seed_id)
    D_max = a + b + e + np.sum(W + V, axis=1)
    m_low = a / D_max
    if np.any(m_low > 0.5):
        print(f"  ✗ seed {seed_id}: m_low = {m_low}")
        break
else:
    print(f"  ✓ 全部 200 种子 m^(0)_j ≤ 0.5")
    print(f"  即 M(0)=0.5 ≥ m^(0) component-wise, D_low 在 t=0 有效")

# ============================================================
# 问题 4: 归纳步是否对 t≥1 保持 M(t) ≥ m^(0)?
# ============================================================
print(f"\n{'='*70}")
print("问题 4: M(t)_k ≥ m_k^(0) 对所有 t≥1 是否始终成立?")
print("=" * 70)

violations = []
for seed_id in range(200):
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    D_max = a + b + e + np.sum(W + V, axis=1)
    m_low = a / D_max
    
    M = np.full(5, 0.5)
    for t in range(200):
        if not np.all(M >= m_low * (1 - 1e-12)):
            violations.append((seed_id, t, M.copy()))
            break
        M_next = n_operator(M, a, b, e, W, V)
        if np.max(np.abs(M_next - M)) < 1e-12:
            break
        M = M_next

if violations:
    print(f"  ✗ {len(violations)}/200 种子违规!")
    for s, t, M in violations[:10]:
        print(f"    seed {s}, t={t}: M_k < m_low")
else:
    print(f"  ✓ 0/200 种子违规 — M(t) ≥ m^(0) 始终成立")

# ============================================================
# 问题 5: 之前在随机起点有 5/200 轨道违规 — 为什么?
# ============================================================
print(f"\n{'='*70}")
print("问题 5: 随机起点违规的细节")
print("=" * 70)

violations_rand = []
for seed_id in range(200):
    a, b, e, W, V = gen_FCA(seed_id)
    D_max = a + b + e + np.sum(W + V, axis=1)
    m_low = a / D_max
    D_low = a + b + e + (W+V) @ m_low
    
    np.random.seed(12345)  # reproducible
    M = np.random.uniform(0.1, 0.9, 5)
    
    # Check at t=0
    D0 = a + b + e + (W+V) @ M
    if np.any(D0 < D_low * (1 - 1e-10)):
        violations_rand.append((seed_id, 0, M.copy(), D0.copy(), D_low.copy()))
    
    for t in range(100):
        D = a + b + e + (W+V) @ M
        if np.any(D < D_low * (1 - 1e-10)):
            violations_rand.append((seed_id, t, M.copy(), D.copy(), D_low.copy()))
            break
        M_next = n_operator(M, a, b, e, W, V)
        if np.max(np.abs(M_next - M)) < 1e-12:
            break
        M = M_next

print(f"  违规数: {len(violations_rand)}/200")
for s, t, M, D, Dl in violations_rand:
    below = [k for k in range(5) if D[k] < Dl[k] * (1 - 1e-10)]
    print(f"  seed {s} t={t}:")
    print(f"    D     = {D}")
    print(f"    D_low = {Dl}")
    print(f"    M     = {M}")
    for k in below:
        print(f"    行{k}: D_k<D_low_k ({D[k]:.4f} < {Dl[k]:.4f})")

# ============================================================
# 问题 6: 在 M(0) = 0.5 前提下, α 是否对 t≥0 都有效?
# ============================================================
print(f"\n{'='*70}")
print("问题 6: 从 M(0)=0.5 出发, α 沿轨道有效性")
print("=" * 70)

for seed_id in [11, 149, 0, 1]:
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W+V) @ Mstar
    D_max = a + b + e + np.sum(W + V, axis=1)
    m_low = a / D_max
    D_low = a + b + e + (W+V) @ m_low
    
    J = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k != j:
                J[k,j] = (W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k]
    gamma = Dstar / D_low
    alpha = max([sum(abs(J[:,j]) * gamma) for j in range(5)])
    
    M = np.full(5, 0.5)
    for t in range(10):
        D = a + b + e + (W+V) @ M
        D_valid = np.all(D >= D_low * (1 - 1e-10))
        
        actual_ratio = np.sum(np.abs(M - Mstar)) / max(1e-10, np.sum(np.abs(M - Mstar)))
        # actually we want: ||M_next - M*|| ≤ α · ||M - M*||
        M_next = n_operator(M, a, b, e, W, V)
        contraction = np.sum(np.abs(M_next - Mstar)) / max(1e-10, np.sum(np.abs(M - Mstar)))
        ok = contraction <= alpha * (1 + 1e-10)
        
        print(f"  seed {seed_id} t={t}: D≥D_low={D_valid}  actual_ratio={contraction:.4f}  α={alpha:.4f}  {'✓' if ok else '✗'}")
        
        M = M_next
        if np.max(np.abs(M - Mstar)) < 1e-8:
            print(f"    converged at t={t+1}")
            break

# ============================================================
# 总结
# ============================================================
print(f"\n{'='*70}")
print("最终审计结论")
print("=" * 70)
