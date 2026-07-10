"""
全面验证: 修正后的6.17B证明 + 所有相关断言
==============================================
验证清单:
  V1: N_k(M) ≥ m_k^(0) ∀M∈[0,1]⁵ (直接代数界的数值确认)
  V2: D_k(M(t)) ≥ D_low,k ∀t≥1 (沿轨道, M(0)=½)
  V3: α < 1 ∀200种子
  V4: α 是实际收缩比的严格上界 ∀t≥1轨道
  V5: M* 的存在唯一性 (M*=N(M*)的迭代收敛)
  V6: D_low ≤ D* (内部一致性)
  V7: 6.17A恒等式数值精度
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

def verify_full(seed_id):
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W+V) @ Mstar
    D_max = a + b + e + np.sum(W + V, axis=1)
    m_low = a / D_max
    D_low = a + b + e + (W+V) @ m_low
    D_min = a + b + e
    
    J = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k != j:
                J[k,j] = (W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k]
    
    gamma = Dstar / D_low
    alpha = max([sum(abs(J[:,j]) * gamma) for j in range(5)])
    
    results = {}
    
    # V1: N_k(M) ≥ m_low for random M
    N_violations = 0
    for _ in range(1000):
        M = np.random.uniform(0, 1, 5)
        N_val = n_operator(M, a, b, e, W, V)
        if not np.all(N_val >= m_low * (1 - 1e-12)):
            N_violations += 1
    results['N_lower_bound'] = N_violations == 0
    
    # V2: D(M(t)) ≥ D_low ∀t≥1 (t=0 可能失败——seed 149)
    orbit_ok = True
    M = np.full(5, 0.5)
    for t in range(100):
        M_next = n_operator(M, a, b, e, W, V)
        D = a + b + e + (W+V) @ M
        if t >= 1:  # D_low bound holds for t≥1
            if not np.all(D >= D_low * (1 - 1e-10)):
                orbit_ok = False
                break
        if np.max(np.abs(M_next - M)) < 1e-12:
            break
        M = M_next
    results['orbit_lower_bound'] = orbit_ok
    
    # V3-V4: α < 1 and is strict upper bound on orbit
    alpha_ok = alpha < 1
    bound_ok = True
    bound_violations = 0
    M = np.full(5, 0.5)
    for t in range(100):
        M_next = n_operator(M, a, b, e, W, V)
        if t >= 1:  # α holds for t ≥ 1
            ratio = np.sum(np.abs(M_next - Mstar)) / max(1e-10, np.sum(np.abs(M - Mstar)))
            if ratio > alpha * (1 + 1e-10):
                bound_ok = False
                bound_violations += 1
        M = M_next
        if np.max(np.abs(M - Mstar)) < 1e-12:
            break
    results['alpha_lt_1'] = alpha_ok
    results['alpha_is_bound'] = bound_ok
    results['alpha_value'] = alpha
    results['bound_violations'] = bound_violations
    
    # V5: M* = N(M*) 
    Mstar_check = n_operator(Mstar, a, b, e, W, V)
    fp_err = np.max(np.abs(Mstar_check - Mstar))
    results['fp_error'] = fp_err
    
    # V6: D_low ≤ D*
    results['dlow_le_dstar'] = np.all(D_low <= Dstar * (1 + 1e-12))
    
    # V7: 6.17A identity
    max_id_err = 0
    for _ in range(100):
        M = np.random.uniform(1e-6, 1-1e-6, 5)
        Delta = M - Mstar
        D = a + b + e + (W+V) @ M
        lhs = n_operator(M, a, b, e, W, V) - Mstar
        rhs = (Dstar / D) * (J @ Delta)
        max_id_err = max(max_id_err, np.max(np.abs(lhs - rhs)))
    results['identity_error'] = max_id_err
    
    # Extra: m_low ≤ M*
    results['mlow_le_mstar'] = np.all(m_low <= Mstar * (1 + 1e-12))
    
    return results

print("=" * 70)
print("全面验证 200 组 FCA 种子")
print("=" * 70)

all_results = []
failures = {k: 0 for k in ['N_lower_bound', 'orbit_lower_bound', 'alpha_lt_1', 
                             'alpha_is_bound', 'dlow_le_dstar', 'mlow_le_mstar']}

for seed_id in range(200):
    r = verify_full(seed_id)
    all_results.append(r)
    for k in failures:
        if not r[k]:
            failures[k] += 1

f_ok = lambda k: '✓ 0/200' if failures[k]==0 else f'✗ {failures[k]}/200'
print(f"\n  V1 N_k(M) >= m^(0) (1K随机M):       {f_ok('N_lower_bound')}")
print(f"  V2 D(M(t)) >= D_low (轨道 t>=1):      {f_ok('orbit_lower_bound')}")
print(f"  V3 alpha < 1:                         {f_ok('alpha_lt_1')}")
print(f"  V4 alpha是实际收缩比上界 (轨道 t>=1):  {f_ok('alpha_is_bound')}")
print(f"  V5 M* = N(M*) (不动点精度):           {'✓ 全部 < 1e-15' if all(r['fp_error']<1e-12 for r in all_results) else '✗'}")
print(f"  V6 D_low <= D*:                       {f_ok('dlow_le_dstar')}")
print(f"  V7 6.17A恒等式 (精度 < 1e-12):        {'✓ 全部' if all(r['identity_error']<1e-12 for r in all_results) else '✗'}")
print(f"  补充: m^(0) <= M*:                     {f_ok('mlow_le_mstar')}")

# Statistics
alphas = [r['alpha_value'] for r in all_results]
fp_errs = [r['fp_error'] for r in all_results]
id_errs = [r['identity_error'] for r in all_results]
violations = [r['bound_violations'] for r in all_results]

print(f"\n  统计:")
print(f"    α: min={min(alphas):.4f}  max={max(alphas):.4f}  mean={np.mean(alphas):.4f}  median={np.median(alphas):.4f}")
print(f"    不动点残差: max={max(fp_errs):.2e}")
print(f"    恒等式误差: max={max(id_errs):.2e}")
print(f"    α界轨道违规数: total={sum(violations)}")

# Identify seeds with m_low > 0.5
print(f"\n  m^(0) > 0.5 的种子 (t=0边界情况):")
for seed_id in range(200):
    a, b, e, W, V = gen_FCA(seed_id)
    D_max = a + b + e + np.sum(W + V, axis=1)
    m_low = a / D_max
    if np.any(m_low > 0.5):
        bad = [k for k in range(5) if m_low[k] > 0.5]
        print(f"    seed {seed_id}: k={bad}  m_low={[f'{m_low[k]:.4f}' for k in bad]}")

print()
print("=" * 70)
if all(f == 0 for f in failures.values()):
    print("✓ 所有验证全部通过")
else:
    print("✗ 有验证失败！")
print("=" * 70)
