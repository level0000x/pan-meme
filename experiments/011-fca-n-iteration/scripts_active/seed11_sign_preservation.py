"""
6.17B seed 11 闭合策略
======================
三条路径并行探索:
  A. 符号保持 — J 沿轨道符号不翻转的解析条件
  B. D-bound 收紧 — 利用迭代 M* 界改善 D_min → D_low
  C. 行聚合 — 行和替代列式的保守界

关键洞察: 如果 D_1 的下界从 D_min,1=0.048 提高到 D_low,1≈0.12
(利用 m_low 的迭代收紧), 则 α_bound 从 1.67 降至 0.67 □
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

def iterative_Mstar_bounds(a, b, e, W, V, max_iter=20):
    """迭代收紧 M* 上下界"""
    D_min = a + b + e
    D_max_v0 = D_min + np.sum(W + V, axis=1)
    m_low = np.clip(a / D_max_v0, 0, 1)
    m_high = np.ones(5)
    for it in range(max_iter):
        A_low = a + W @ m_low
        A_high = a + W @ m_high
        D_low = a + b + e + (W + V) @ m_low
        D_high = a + b + e + (W + V) @ m_high
        m_low_new = np.clip(A_low / D_high, 0, 1)
        m_high_new = np.clip(A_high / D_low, 0, 1)
        if np.max(np.abs(m_low_new - m_low)) < 1e-12:
            break
        m_low = m_low_new
        m_high = m_high_new
    D_low_final = a + b + e + (W + V) @ m_low
    return m_low, m_high, D_low_final

print("=" * 70)
print("路径 B: D-bound 收紧对 α_bound 的影响")
print("=" * 70)

for seed_id in [11] + list(range(200)):
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W + V) @ Mstar
    D_min = a + b + e
    m_low, m_high, D_low = iterative_Mstar_bounds(a, b, e, W, V)
    
    # Old α_bound: 列式 with D_min
    J_abs = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k != j:
                J_abs[k,j] = abs((W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k])
    
    alpha_old = max([sum(J_abs[:,j] * Dstar / D_min) for j in range(5)])
    
    # New: 行式 with D_low
    alpha_row_Dlow = 0
    for k in range(5):
        row_sum = sum(J_abs[k,:])  # sum_j |J_kj|
        alpha_row_Dlow = max(alpha_row_Dlow, row_sum * Dstar[k] / D_low[k])
    
    # New: 行式 with D_min (for comparison)
    alpha_row_Dmin = 0
    for k in range(5):
        row_sum = sum(J_abs[k,:])
        alpha_row_Dmin = max(alpha_row_Dmin, row_sum * Dstar[k] / D_min[k])
    
    if seed_id == 11 or alpha_old >= 1:
        tag = "✗" if alpha_old >= 1 else ""
        print(f"  seed {seed_id:3d}: α_old(列+Dmin)={alpha_old:.3f}  "
              f"α_row_Dlow={alpha_row_Dlow:.3f}  "
              f"Dlow/Dmin = [{', '.join(f'{x:.2f}x' for x in D_low/D_min)}]  {tag}")
    
    if seed_id > 11 and len([x for x in [alpha_old] if x >= 1]) == 0:
        break
    if seed_id > 30:
        break

# ============================================================
# 路径 A: 符号保持分析
# ============================================================
print(f"\n{'='*70}")
print("路径 A: 符号保持 — N_k 穿越 θ_kj 的条件")
print("=" * 70)

for seed_id in [11, 21, 126]:
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    
    print(f"\n--- seed {seed_id} ---")
    print(f"M* = {Mstar}")
    
    # For each (k,j), check threshold θ_kj and distance from M*_k
    for k in range(5):
        row_info = []
        for j in range(5):
            if k == j: continue
            w, v = W[k,j], V[k,j]
            theta = w / (w + v)
            dist = abs(Mstar[k] - theta)
            row_info.append((j, theta, dist))
        
        # Show the most dangerous (k,j) pair — smallest distance
        row_info.sort(key=lambda x: x[2])
        print(f"  行{k} (M*_k={Mstar[k]:.4f}): 最近阈值 θ=({row_info[0][0]},{row_info[0][1]:.4f}) "
              f"距={row_info[0][2]:.4f}")
        
        # Check if there's a danger zone: if w/(w+v) is close to M*_k
        dangerous = [(j, th, d) for j, th, d in row_info if d < 0.1]
        if dangerous:
            print(f"    ⚠ 危险: {dangerous}")

# ============================================================
# 沿实际迭代轨道测试符号保持
# ============================================================
print(f"\n{'='*70}")
print("沿 N 迭代轨道测试 J 符号翻转")
print("=" * 70)

for seed_id in [11, 21, 126]:
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W + V) @ Mstar
    
    # Reference signs at M*
    ref_signs = {}
    for k in range(5):
        for j in range(5):
            if k == j: continue
            val = W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]
            if abs(val) > 1e-12:
                ref_signs[(k,j)] = np.sign(val)
    
    # Multiple starting points
    total_flips = 0
    total_checks = 0
    n_starts = 20
    
    for _ in range(n_starts):
        M = np.random.uniform(0.1, 0.9, 5)
        for step in range(100):
            M_next = n_operator(M, a, b, e, W, V)
            N_vals = M_next  # N(M)
            
            for (k,j), s0 in ref_signs.items():
                w, v = W[k,j], V[k,j]
                val = w*(1-N_vals[k]) - v*N_vals[k]
                if abs(val) > 1e-12:
                    if np.sign(val) != s0:
                        total_flips += 1
                    total_checks += 1
            
            M = M_next
            if np.max(np.abs(M - Mstar)) < 1e-8:
                break
    
    print(f"  seed {seed_id}: {total_flips}/{total_checks} 翻转 ({100*total_flips/total_checks:.3f}%)  "
          f"{n_starts}起点")

# ============================================================
# 路径 C: 列式 → 行聚合的 α_bound 收紧
# ============================================================ 
print(f"\n{'='*70}")
print("路径 B+C 综合: 行式+迭代D_low 对 α_bound 的影响")
print("=" * 70)
print(f"{'seed':>5} {'α_old':>8} {'α_row_Dmin':>12} {'α_row_Dlow':>12} {'闭合?':>8}")
print(f"{'-'*50}")

total_closed_old = 0
total_closed_rowDmin = 0
total_closed_rowDlow = 0

for seed_id in range(200):
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W + V) @ Mstar
    D_min = a + b + e
    m_low, m_high, D_low = iterative_Mstar_bounds(a, b, e, W, V)
    
    J_abs = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k != j:
                J_abs[k,j] = abs((W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k])
    
    # Old: 列式 + D_min
    alpha_old = max([sum(J_abs[:,j] * Dstar / D_min) for j in range(5)])
    
    # New row-wise: max_k (Σ_j |J_kj|) * (D*_k / D_min,k)
    alpha_row_Dmin = max([sum(J_abs[k,:]) * Dstar[k] / D_min[k] for k in range(5)])
    
    # New row-wise with D_low
    alpha_row_Dlow = max([sum(J_abs[k,:]) * Dstar[k] / D_low[k] for k in range(5)])
    
    if alpha_old < 1:
        total_closed_old += 1
    if alpha_row_Dmin < 1:
        total_closed_rowDmin += 1
    if alpha_row_Dlow < 1:
        total_closed_rowDlow += 1
    
    if seed_id < 15 or alpha_old >= 1:
        status = "✓" if alpha_row_Dlow < 1 else "✗"
        print(f"  {seed_id:3d}  {alpha_old:8.3f}  {alpha_row_Dmin:12.3f}  {alpha_row_Dlow:12.3f}  {status:>6}")

print(f"\n  总计闭合:")
print(f"    列式+D_min:  {total_closed_old}/200 ({100*total_closed_old/200:.0f}%)")
print(f"    行式+D_min:  {total_closed_rowDmin}/200 ({100*total_closed_rowDmin/200:.0f}%)")
print(f"    行式+D_low:  {total_closed_rowDlow}/200 ({100*total_closed_rowDlow/200:.0f}%)")


# ============================================================
# 种子级诊断: 找出所有列式 α_old ≥ 1 的种子
# ============================================================
print(f"\n{'='*70}")
print("行式+D_low 不闭合的种子详细诊断")
print("=" * 70)

hard_seeds = []
for seed_id in range(200):
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W + V) @ Mstar
    D_min = a + b + e
    m_low, m_high, D_low = iterative_Mstar_bounds(a, b, e, W, V)
    
    J_abs = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k != j:
                J_abs[k,j] = abs((W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k])
    
    alpha_old = max([sum(J_abs[:,j] * Dstar / D_min) for j in range(5)])
    alpha_row_Dlow = max([sum(J_abs[k,:]) * Dstar[k] / D_low[k] for k in range(5)])
    
    if alpha_row_Dlow >= 1:
        worst_k = np.argmax([sum(J_abs[k,:]) * Dstar[k] / D_low[k] for k in range(5)])
        hard_seeds.append((seed_id, alpha_old, alpha_row_Dlow, worst_k,
                          Dstar[worst_k], D_low[worst_k], D_min[worst_k],
                          sum(J_abs[worst_k,:]), Mstar[worst_k]))

hard_seeds.sort(key=lambda x: -x[2])

print(f"  共 {len(hard_seeds)} 个种子")
print(f"  {'seed':>5} {'α_old':>8} {'α_rowDlow':>11} {'行':>3} "
      f"{'D*':>7} {'D_low':>7} {'D_min':>7} {'Σ|J|':>7} {'M*_k':>7}")

for s in hard_seeds:
    print(f"  {s[0]:5d} {s[1]:8.3f} {s[2]:11.3f} {s[3]:3d} "
          f"{s[4]:7.4f} {s[5]:7.4f} {s[6]:7.4f} {s[7]:7.4f} {s[8]:7.4f}")

# ============================================================
# 关键: 对 hard seeds, 实际 l₁ 行范数 vs 三角不等式上界
# ============================================================
print(f"\n{'='*70}")
print("Hard seed 的 实际 l₁行范数 vs 界")
print("=" * 70)

for seed_id, alpha_old, alpha_row_Dlow, worst_k, _, _, _, _, _ in hard_seeds:
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W + V) @ Mstar
    m_low, m_high, D_low = iterative_Mstar_bounds(a, b, e, W, V)
    
    true_l1 = np.zeros(5)
    tri_l1 = np.zeros(5)
    for k in range(5):
        true_l1[k] = sum(abs((W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k]) for j in range(5) if j != k)
        tri_l1[k] = sum((W[k,j]*(1-Mstar[k]) + V[k,j]*Mstar[k]) / (a[k]+b[k]+e[k]) for j in range(5) if j != k)
    
    printed = set()
    for k in range(5):
        ratio = tri_l1[k] / max(true_l1[k], 1e-10)
        if ratio > 2 or k == worst_k:
            if k not in printed:
                print(f"  seed{seed_id} 行{k}: 真实={true_l1[k]:.3f} 三角={tri_l1[k]:.3f} 过估={ratio:.1f}x  D*/D_low={Dstar[k]/D_low[k]:.1f}")
                printed.add(k)
