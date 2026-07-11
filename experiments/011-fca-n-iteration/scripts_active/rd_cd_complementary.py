"""
互补上界迭代 —— 用 M* = 1 - (b+ε+V·M*)/D* 构建上界
========================================================
关键: 上界迭代 1 - (b+ε) / D_max 不依赖 m_low, 可能收敛到 < 1
"""
import numpy as np

def n_operator(M, a, b, eps, W, V):
    num = a + W @ M
    den = num + b + V @ M + eps
    return num / den

def compute_fp(a, b, eps, W, V):
    M = np.full(5, 0.5)
    for _ in range(20000):
        M_new = n_operator(M, a, b, eps, W, V)
        if np.max(np.abs(M_new - M)) < 1e-15:
            return M_new
        M = M_new
    return M

def gen_FCA(seed):
    rs = np.random.RandomState(seed)
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

def gen_extended(seed):
    rs = np.random.RandomState(seed + 10000)
    a = rs.uniform(0.005, 0.5, 5)
    b = rs.uniform(0.005, 0.5, 5)
    e = rs.uniform(0.0005, 0.5, 5)
    W = rs.uniform(0.005, 0.3, (5, 5))
    V = rs.uniform(0.005, 0.3, (5, 5))
    np.fill_diagonal(W, 0.0)
    np.fill_diagonal(V, 0.0)
    t = a.sum() + b.sum() + W.sum() + V.sum()
    W *= 5.0 / t
    V *= 5.0 / t
    return a, b, e, W, V

def comp_upper_bound_iter(a, b, e, W, V, max_iter=50):
    """互补上界迭代"""
    m_high = np.ones(5)
    for it in range(max_iter):
        D_high = a + b + e + (W + V) @ m_high
        m_high_new = 1 - (b + e) / D_high
        m_high_new = np.clip(m_high_new, 0, 1)
        if np.max(np.abs(m_high_new - m_high)) < 1e-12:
            break
        m_high = m_high_new
    return m_high, it+1

def complementary_bounds(a, b, e, W, V, max_iter=50):
    """互补界: 同时收紧上界和下界"""
    m_high = np.ones(5)
    m_low = np.zeros(5)
    for it in range(max_iter):
        D_high = a + b + e + (W + V) @ m_high
        D_low = a + b + e + (W + V) @ m_low
        
        m_high_new = 1 - (b + e + V @ m_low) / D_high
        m_low_new = a / (D_high + np.finfo(float).eps)  # 用 D_high 作为分母上界
        
        m_high_new = np.clip(m_high_new, 0, 1)
        m_low_new = np.clip(m_low_new, 0, 1)
        
        if (np.max(np.abs(m_high_new - m_high)) < 1e-12 and 
            np.max(np.abs(m_low_new - m_low)) < 1e-12):
            break
        
        m_high = m_high_new
        m_low = m_low_new
    
    return m_low, m_high, D_low, it+1

print("=" * 70)
print("互补上界迭代测试 (不依赖 m_low)")
print("=" * 70)

# Test on hard seeds
hard_seeds = [21, 126, 140, 9, 84, 11]
for seed in hard_seeds:
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    
    m_high_comp, nit = comp_upper_bound_iter(a, b, e, W, V)
    print(f"\nseed {seed}:")
    print(f"  M* = [{', '.join(f'{x:.4f}' for x in Mstar)}]")
    print(f"  互补上界 = [{', '.join(f'{x:.4f}' for x in m_high_comp)}] ({nit} iters)")
    print(f"  max 互补上界 = {max(m_high_comp):.4f}")
    
    # Check if any < 1
    if all(m_high_comp < 1):
        print(f"  ✓ 所有分量 < 1! min slack = {1 - max(m_high_comp):.4f}")

# Now try complementary bounds
print(f"\n{'='*70}")
print("互补界迭代 (同时收紧上下界)")
print(f"{'='*70}")

for seed in hard_seeds:
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    
    m_low, m_high, D_low, nit = complementary_bounds(a, b, e, W, V)
    print(f"\nseed {seed}:")
    print(f"  M* = [{', '.join(f'{x:.4f}' for x in Mstar)}]")
    print(f"  m_low  = [{', '.join(f'{x:.4f}' for x in m_low)}]")
    print(f"  m_high = [{', '.join(f'{x:.4f}' for x in m_high)}]")
    print(f"  D_low  = [{', '.join(f'{x:.4f}' for x in D_low)}]")
    
    # Row-sum bound with these improved M* bounds
    rd_bound = np.zeros(5)
    for k in range(5):
        num_low = sum([abs(W[k,j]*(1-m_low[k]) - V[k,j]*m_low[k]) for j in range(5) if j != k])
        num_high = sum([abs(W[k,j]*(1-m_high[k]) - V[k,j]*m_high[k]) for j in range(5) if j != k])
        num_bound = max(num_low, num_high)
        rd_bound[k] = num_bound / D_low[k]
    print(f"  RD bound = [{', '.join(f'{x:.3f}' for x in rd_bound)}]  max={max(rd_bound):.3f}")
    
    # Actual RD
    Dstar = (a + W @ Mstar) + (b + V @ Mstar) + e
    rd_act = np.zeros(5)
    for k in range(5):
        rd_act[k] = sum([abs((W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k]) for j in range(5) if j != k])
    print(f"  Actual RD = [{', '.join(f'{x:.3f}' for x in rd_act)}]  max={max(rd_act):.3f}")

# ============================================================
# 批量测试
# ============================================================
print(f"\n{'='*70}")
print("批量测试: 互补界 + 行和界 覆盖率")
print(f"{'='*70}")

for gen_func, domain_name, n_seeds in [(gen_FCA, "FCA", 200), (gen_extended, "扩展域", 500)]:
    count = 0
    worst_bound = 0
    worst_seed = 0
    for seed in range(n_seeds):
        a, b, e, W, V = gen_func(seed)
        m_low, m_high, D_low, _ = complementary_bounds(a, b, e, W, V)
        
        rd_bound = np.zeros(5)
        for k in range(5):
            num_low = sum([abs(W[k,j]*(1-m_low[k]) - V[k,j]*m_low[k]) for j in range(5) if j != k])
            num_high = sum([abs(W[k,j]*(1-m_high[k]) - V[k,j]*m_high[k]) for j in range(5) if j != k])
            num_bound = max(num_low, num_high)
            rd_bound[k] = num_bound / D_low[k]
        
        if max(rd_bound) < 1:
            count += 1
        if max(rd_bound) > worst_bound:
            worst_bound = max(rd_bound)
            worst_seed = seed
    
    pct = 100*count/n_seeds
    print(f"  {domain_name}: {count}/{n_seeds} ({pct:.0f}%)  最劣界={worst_bound:.3f} (seed {worst_seed})")
