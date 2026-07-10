"""
6.17B D_low 收紧证明 — 全面审计
====================================
审计清单:
  A. 6.17A 恒等式数值验证 (随机 M, 大量测试)
  B. J_kj(M*) vs J_kj(M) — 恒等式用的是哪个 J?
  C. D_low 对所有 t≥0 的 D(M(t)) 是否成立
  D. α 界对随机 Δ 是否严格上界
  E. 边界条件: m^(0) 是否 < M*_k? D_low_b 是否 < D*_k?
  F. 从非均匀起点 (非 0.5) 的轨道
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
# A. 6.17A 恒等式数值验证
# ============================================================
print("=" * 70)
print("A. 6.17A 恒等式数值验证")
print("=" * 70)

for seed_id in [11, 21, 126, 149, 0, 1, 2, 5, 10]:
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W+V) @ Mstar
    
    # Compute J(M*)
    J = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k != j:
                J[k,j] = (W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k]
    
    max_err = 0
    for _ in range(1000):
        M = np.random.uniform(1e-6, 1-1e-6, 5)
        Delta = M - Mstar
        D = a + b + e + (W+V) @ M
        
        # LHS: N(M) - M*
        N_val = n_operator(M, a, b, e, W, V)
        lhs = N_val - Mstar
        
        # RHS: diag(D*/D) · J · Δ
        rhs = (Dstar / D) * (J @ Delta)
        
        err = np.max(np.abs(lhs - rhs))
        max_err = max(max_err, err)
    
    print(f"  seed {seed_id}: 6.17A 恒等式最大误差 = {max_err:.2e}  {'✓' if max_err < 1e-12 else '✗'}")

# ============================================================
# B. D_low 定义及性质检查
# ============================================================
print(f"\n{'='*70}")
print("B. D_low/k 性质检查 (全部 200 种子)")
print("=" * 70)

violations_dlow_vs_dstar = 0
violations_mlow_vs_mstar = 0
violations_dlow_global = 0  # D_low ≤ D(M) for random M

for seed_id in range(200):
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W+V) @ Mstar
    
    D_max = a + b + e + np.sum(W + V, axis=1)
    m_low = a / D_max
    D_low = a + b + e + (W+V) @ m_low
    
    # Check 1: m_low ≤ M* (component-wise)
    if not np.all(m_low <= Mstar * (1 + 1e-12)):
        violations_mlow_vs_mstar += 1
    
    # Check 2: D_low ≤ D*
    if not np.all(D_low <= Dstar * (1 + 1e-12)):
        violations_dlow_vs_dstar += 1
    
    # Check 3: D_low ≤ D(M) for random M
    for _ in range(20):
        M = np.random.uniform(0, 1, 5)
        D = a + b + e + (W+V) @ M
        if np.any(D < D_low * (1 - 1e-10)):
            violations_dlow_global += 1
            break

print(f"  m_low ≤ M*: {'✓ 0/200' if violations_mlow_vs_mstar == 0 else f'✗ {violations_mlow_vs_mstar}/200'}")
print(f"  D_low ≤ D*: {'✓ 0/200' if violations_dlow_vs_dstar == 0 else f'✗ {violations_dlow_vs_dstar}/200'}")
print(f"  D_low ≤ D(M) (随机采样): {'✓ 0/200' if violations_dlow_global == 0 else f'✗ {violations_dlow_global}/200'}")

# ============================================================
# C. 沿轨道 D(M(t)) ≥ D_low 验证
# ============================================================
print(f"\n{'='*70}")
print("C. 沿迭代轨道 D(M(t)) ≥ D_low (从 0.5 出发 + 从随机点出发)")
print("=" * 70)

orbit_violations_05 = 0
orbit_violations_rand = 0

for seed_id in range(200):
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    
    D_max = a + b + e + np.sum(W + V, axis=1)
    m_low = a / D_max
    D_low = a + b + e + (W+V) @ m_low
    
    # From M(0) = 0.5
    M = np.full(5, 0.5)
    violated = False
    for t in range(100):
        D = a + b + e + (W+V) @ M
        if np.any(D < D_low * (1 - 1e-10)):
            orbit_violations_05 += 1
            violated = True
            break
        M_next = n_operator(M, a, b, e, W, V)
        if np.max(np.abs(M_next - M)) < 1e-12:
            break
        M = M_next
    
    # From random starting points
    M = np.random.uniform(0.1, 0.9, 5)
    for t in range(100):
        D = a + b + e + (W+V) @ M
        if np.any(D < D_low * (1 - 1e-10)):
            orbit_violations_rand += 1
            break
        M_next = n_operator(M, a, b, e, W, V)
        if np.max(np.abs(M_next - M)) < 1e-12:
            break
        M = M_next

print(f"  D_low ≤ D(M(t)) 沿轨道 (0.5起点): {'✓ 0/200' if orbit_violations_05 == 0 else f'✗ {orbit_violations_05}/200'}")
print(f"  D_low ≤ D(M(t)) 沿轨道 (随机起点): {'✓ 0/200' if orbit_violations_rand == 0 else f'✗ {orbit_violations_rand}/200'}")

# ============================================================
# D. α 界 vs 实际收缩比 (大规模直接测试)
# ============================================================
print(f"\n{'='*70}")
print("D. α 界 vs 实际收缩比 (10000 随机 Δ × 200 种子)")
print("=" * 70)

worst_ratio_global = 0
worst_seed_global = 0
total_violations = 0

for seed_id in range(200):
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
    
    local_worst = 0
    local_violations = 0
    for _ in range(10000):
        Delta = np.random.uniform(-1, 1, 5)
        M = Mstar + Delta
        M = np.clip(M, 1e-6, 1-1e-6)
        N_val = n_operator(M, a, b, e, W, V)
        ratio = np.sum(np.abs(N_val - Mstar)) / max(np.sum(np.abs(Delta)), 1e-10)
        local_worst = max(local_worst, ratio)
        if ratio > alpha:
            local_violations += 1
    
    if local_violations > 0:
        total_violations += 1
        worst_ratio_global = max(worst_ratio_global, local_worst)
        worst_seed_global = seed_id
        if local_violations > 10:  # 只打印严重的
            print(f"  ✗ seed {seed_id}: α={alpha:.4f}  实际max={local_worst:.4f}  违规{local_violations}/10000")

if total_violations == 0:
    print(f"  ✓ 0/200 种子有违规 — α 是严格上界")
else:
    print(f"  ✗ {total_violations}/200 种子有违规")
    print(f"  最劣违规: α={alpha:.4f} vs 实际={worst_ratio_global:.4f} (seed {worst_seed_global})")

# ============================================================
# E. 详细检查最劣种子
# ============================================================
print(f"\n{'='*70}")
print("E. 最劣 (最高α) 种子详细分析")
print("=" * 70)

# Find seed with highest alpha
worst_alpha = 0
worst_seed_alpha = 0
for seed_id in range(200):
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
    if alpha > worst_alpha:
        worst_alpha = alpha
        worst_seed_alpha = seed_id

print(f"  最高α种子: seed {worst_seed_alpha}, α = {worst_alpha:.4f}")

a, b, e, W, V = gen_FCA(worst_seed_alpha)
Mstar = compute_fp(a, b, e, W, V)
Dstar = a + b + e + (W+V) @ Mstar
D_max = a + b + e + np.sum(W + V, axis=1)
m_low = a / D_max
D_low = a + b + e + (W+V) @ m_low
gamma = Dstar / D_low

J = np.zeros((5,5))
for k in range(5):
    for j in range(5):
        if k != j:
            J[k,j] = (W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k]

print(f"\n  M* = {Mstar}")
print(f"  m_low = {m_low}")
print(f"  D_min = {a+b+e}")
print(f"  D_low = {D_low}")
print(f"  D* = {Dstar}")
print(f"  γ = D*/D_low = {gamma}")

print(f"\n  |J| 矩阵:")
for k in range(5):
    row = []
    for j in range(5):
        if k == j:
            row.append("  ·  ")
        else:
            row.append(f"{abs(J[k,j]):.4f}")
    print(f"  行{k}: " + " ".join(row))

col_sums = [sum(abs(J[:,j]) * gamma) for j in range(5)]
print(f"\n  列式 α 分量 = {[f'{x:.4f}' for x in col_sums]}")
print(f"  α = max = {max(col_sums):.4f}")

# ============================================================
# F. 边界条件检查: m_low 是否太小导致 D_low 过于保守?
# ============================================================
print(f"\n{'='*70}")
print("F. D_low 保守性检查")
print("=" * 70)

for seed_id in [worst_seed_alpha, 11, 149]:
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W+V) @ Mstar
    D_max = a + b + e + np.sum(W + V, axis=1)
    m_low = a / D_max
    D_low = a + b + e + (W+V) @ m_low
    
    # Orbit minimum of D_k
    M = np.full(5, 0.5)
    min_D = np.inf * np.ones(5)
    for t in range(100):
        D = a + b + e + (W+V) @ M
        for k in range(5):
            min_D[k] = min(min_D[k], D[k])
        M_next = n_operator(M, a, b, e, W, V)
        if np.max(np.abs(M_next - M)) < 1e-12:
            break
        M = M_next
    
    print(f"\n  seed {seed_id}:")
    print(f"    D_low = [{', '.join(f'{x:.4f}' for x in D_low)}]")
    print(f"    D*    = [{', '.join(f'{x:.4f}' for x in Dstar)}]")
    print(f"    D_min = [{', '.join(f'{x:.4f}' for x in a+b+e)}]")
    print(f"    轨道 min D = [{', '.join(f'{x:.4f}' for x in min_D)}]")
    print(f"    保守比 D_low/D*: [{', '.join(f'{x:.2f}x' for x in D_low/Dstar)}]")
    print(f"    轨道min/D_low: [{', '.join(f'{x:.2f}x' for x in min_D/D_low)}]")

# ============================================================
# G. 交叉验证: 用 D_min 的 α 和 D_low 的 α 对比
# ============================================================
print(f"\n{'='*70}")
print("G. α(D_min) vs α(D_low) 对比 + 实际收缩比")
print("=" * 70)

for seed_id in range(15):
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
    
    alpha_dmin = max([sum(abs(J[:,j]) * Dstar/D_min) for j in range(5)])
    alpha_dlow = max([sum(abs(J[:,j]) * Dstar/D_low) for j in range(5)])
    
    # Actual contraction ratio
    actual_max = 0
    for _ in range(1000):
        Delta = np.random.uniform(-1, 1, 5)
        M = Mstar + Delta
        M = np.clip(M, 1e-6, 1-1e-6)
        N_val = n_operator(M, a, b, e, W, V)
        ratio = np.sum(np.abs(N_val - Mstar)) / max(np.sum(np.abs(Delta)), 1e-10)
        actual_max = max(actual_max, ratio)
    
    flag = "✓" if alpha_dlow < 1 else "✗"
    print(f"  seed {seed_id:2d}: α(D_min)={alpha_dmin:.4f}  α(D_low)={alpha_dlow:.4f}  actualmax={actual_max:.4f}  {flag}")

# ============================================================
# H. 检查: 6.17A 恒等式中的 J 是否应该用 J(M) 而不是 J(M*)?
# ============================================================
print(f"\n{'='*70}")
print("H. J(M) vs J(M*) 在恒等式中的测试")
print("=" * 70)

for seed_id in [11, 0]:
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W+V) @ Mstar
    
    # J(M*)
    J_star = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k != j:
                J_star[k,j] = (W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k]
    
    errors_star = []
    errors_M = []
    
    for _ in range(1000):
        M = np.random.uniform(1e-6, 1-1e-6, 5)
        Delta = M - Mstar
        D = a + b + e + (W+V) @ M
        
        # 6.17A with J(M*)
        lhs = n_operator(M, a, b, e, W, V) - Mstar
        rhs_star = (Dstar / D) * (J_star @ Delta)
        errors_star.append(np.max(np.abs(lhs - rhs_star)))
        
        # What if we used J(M)?
        J_M = np.zeros((5,5))
        for k in range(5):
            for j in range(5):
                if k != j:
                    J_M[k,j] = (W[k,j]*(1-M[k]) - V[k,j]*M[k]) / ((a+W@M)[k] + (b+V@M)[k] + e[k])
        rhs_M = (Dstar / D) * (J_M @ Delta)
        errors_M.append(np.max(np.abs(lhs - rhs_M)))
    
    print(f"  seed {seed_id}:")
    print(f"    6.17A with J(M*): max err = {max(errors_star):.2e}  {'✓' if max(errors_star) < 1e-12 else '✗'}")
    print(f"    6.17A with J(M):  max err = {max(errors_M):.2e}  {'✓' if max(errors_M) < 1e-12 else '✗'}")

print(f"\n{'='*70}")
print("审计总结")
print("=" * 70)
