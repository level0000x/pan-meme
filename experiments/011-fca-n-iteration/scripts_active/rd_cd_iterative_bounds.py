"""
诊断: 行和界的剩余 gap 来源
============================
两种过估来源:
  1. 分子: max(Σw, Σv) 是 M*_k=0或1 的极端值, 实际 M*_k 远离端点
  2. 分母: D_low_v2 是 D*_k 的下界, 可能偏小

策略:
  - 迭代收紧 M* 上下界 (通过 FP 方程)
  - 对 M*_k 的收紧等效于对分子界的收紧
  - 迭代地同时收紧分母
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

def iterative_Mstar_bounds(a, b, e, W, V, max_iter=20):
    """迭代收紧 M* 的上下界, 使用 FP 方程 M*_k = (a_k + ΣwM*_j) / (a_k+b_k+ε_k+Σ(w+v)M*_j)"""
    D_min = a + b + e
    D_max_v0 = D_min + np.sum(W + V, axis=1)
    
    m_low = np.clip(a / D_max_v0, 0, 1)
    m_high = np.ones(5)
    
    history = []
    
    for it in range(max_iter):
        A_low = a + W @ m_low
        A_high = a + W @ m_high
        D_low = a + b + e + (W + V) @ m_low
        D_high = a + b + e + (W + V) @ m_high
        
        m_low_new = np.clip(A_low / D_high, 0, 1)
        m_high_new = np.clip(A_high / D_low, 0, 1)
        
        history.append({
            'm_low': m_low.copy(),
            'm_high': m_high.copy(),
            'D_low': D_low.copy(),
        })
        
        if np.max(np.abs(m_low_new - m_low)) < 1e-10 and np.max(np.abs(m_high_new - m_high)) < 1e-10:
            break
        
        m_low = m_low_new
        m_high = m_high_new
    
    return history

print("=" * 70)
print("迭代 M* 界收紧 - 诊断行和界的剩余 gap")
print("=" * 70)

# 先找一个 FCA 硬种子分析
print("\n--- 硬种子诊断 (FCA) ---")
hard_fca = [11, 12, 126, 54, 16]
for seed_id in hard_fca[:3]:
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = (a + W @ Mstar) + (b + V @ Mstar) + e
    
    history = iterative_Mstar_bounds(a, b, e, W, V)
    
    print(f"\n  seed {seed_id}:")
    print(f"    M* = [{', '.join(f'{x:.4f}' for x in Mstar)}]")
    
    for it in [0, 1, 2, -1]:
        h = history[it]
        it_name = f"iter {it}" if it >= 0 else f"iter {len(history)-1} (final)"
        print(f"    {it_name}:")
        print(f"      M* ∈ [{', '.join(f'{x:.4f}' for x in h['m_low'])}] × [{', '.join(f'{x:.4f}' for x in h['m_high'])}]")
        print(f"      D_low = [{', '.join(f'{x:.4f}' for x in h['D_low'])}]")
        
        # 行和 bound 在这个迭代
        rd_bound = np.zeros(5)
        for k in range(5):
            num = max(W[k].sum(), V[k].sum())
            rd_bound[k] = num / h['D_low'][k]
        print(f"      RD行和界 = [{', '.join(f'{x:.3f}' for x in rd_bound)}]  max={max(rd_bound):.3f}")
    
    # Actual
    J_actual = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k != j:
                J_actual[k,j] = abs((W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k])
    rd_actual = J_actual.sum(axis=1)
    
    # 也计算"用真 M*_k"的分子界（相当于已知 M*_k 的准确值但不知道 D*）
    rd_with_exact_Mk = np.zeros(5)
    for k in range(5):
        num_exact = sum([abs(W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) for j in range(5) if j != k])
        rd_with_exact_Mk[k] = num_exact / history[1]['D_low'][k]
    
    print(f"    实际RD = [{', '.join(f'{x:.3f}' for x in rd_actual)}]  max={max(rd_actual):.3f}")
    print(f"    真分子+Dv2分母RD = [{', '.join(f'{x:.3f}' for x in rd_with_exact_Mk)}]  max={max(rd_with_exact_Mk):.3f}")
    
    # Gap decomposition
    new_rd_bound_v2 = np.zeros(5)
    for k in range(5):
        new_rd_bound_v2[k] = max(W[k].sum(), V[k].sum()) / history[1]['D_low'][k]
    
    num_gap = abs(max(new_rd_bound_v2) - max(rd_with_exact_Mk))
    den_gap = abs(max(rd_with_exact_Mk) - max(rd_actual))
    print(f"    Gap分解: 分子过估={num_gap:.3f}, 分母过估={den_gap:.3f}")


# ============================================================
# 批量迭代
# ============================================================
print(f"\n{'='*70}")
print("批量: 迭代 M* 界 对覆盖率的影响")
print(f"{'='*70}")

for gen_func, domain_name, n_seeds in [(gen_FCA, "FCA", 200), (gen_extended, "扩展域", 500)]:
    print(f"\n  {domain_name}:")
    
    for max_iter in [1, 2, 5, 10]:
        count = 0
        worst_bound = 0
        for seed in range(n_seeds):
            a, b, e, W, V = gen_func(seed)
            history = iterative_Mstar_bounds(a, b, e, W, V, max_iter=min(max_iter, 20))
            h = history[-1]  # last iteration
            
            rd_bound = np.zeros(5)
            for k in range(5):
                num = max(W[k].sum(), V[k].sum())
                rd_bound[k] = num / h['D_low'][k]
            
            if max(rd_bound) < 1:
                count += 1
            worst_bound = max(worst_bound, max(rd_bound))
        
        pct = 100*count/n_seeds
        print(f"    迭代{max_iter:2d}: {count}/{n_seeds} ({pct:.0f}%)  最劣界={worst_bound:.3f}")


# ============================================================
# 如果只用D_low_v2分母 + 收紧的M*区间算分子
# ============================================================
print(f"\n{'='*70}")
print("终极测试: 行和界 + 迭代收紧 M* + D* 界")
print(f"{'='*70}")

for gen_func, domain_name, n_seeds in [(gen_FCA, "FCA", 200)]:
    print(f"\n  {domain_name}:")
    
    for max_iter in [1, 2, 5, 10]:
        count = 0
        for seed in range(n_seeds):
            a, b, e, W, V = gen_func(seed)
            history = iterative_Mstar_bounds(a, b, e, W, V, max_iter=min(max_iter, 20))
            h = history[-1]
            
            # 使用收紧后的 M* 区间来收紧分子界
            rd_bound = np.zeros(5)
            for k in range(5):
                # 用 M*_k 的界来缩小分子
                # Σ_j |w - (w+v)M*_k| 的最大值在 M*_k 区间的哪个端点?
                # 函数在 [m_low, m_high] 上凸 ⇒ 最大值在端点
                m_low = h['m_low'][k]
                m_high = h['m_high'][k]
                num_low = sum([abs(W[k,j]*(1-m_low) - V[k,j]*m_low) for j in range(5) if j != k])
                num_high = sum([abs(W[k,j]*(1-m_high) - V[k,j]*m_high) for j in range(5) if j != k])
                num_bound = max(num_low, num_high)
                rd_bound[k] = num_bound / h['D_low'][k]
            
            if max(rd_bound) < 1:
                count += 1
        
        pct = 100*count/n_seeds
        print(f"    迭代{max_iter:2d} (收紧分子+分母): {count}/{n_seeds} ({pct:.0f}%)")
