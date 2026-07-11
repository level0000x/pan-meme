"""
6.17A2 RD行和界闭合：迭代收紧 M* 区间
======================================
核心思路:
  凸函数 g_k(x) = Σ|w-(w+v)x| 在 [0,1] 取端点最大值
  → 若能收紧 M*_k 的区间到 [m_k, u_k] ⊂ [0,1]
  → 则 g_k(M*_k) ≤ max(g_k(m_k), g_k(u_k)) < max(g_k(0), g_k(1))

路线:
  R1: 纯参数下界 m_k^(0) = a_k/D_max,k (已证)
  R2: 纯参数上界 u_k^(0) = (a_k+Σw_kj) / (a_k+Σw_kj+b_k+ε_k) 
       ← M* ≤ N_k(M with M_j=1∀W, M_j=0∀V)? 不行。
       M*_k ≤ 1 - (b_k+ε_k)/(a_k+b_k+ε_k+Σw_kj) 是更紧的界
  R3: 利用 m_k^(0) 作为所有M*_j的下界可紧缩D*_k下界进而收紧M*_k上界
  R4: 若还不够 → 迭代收紧 m_k → u_k → m_k ...
  
注意: N 非单调, 不能简单地说 N(m) ≤ M* ≤ N(u)
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
    np.fill_diagonal(W, 0.0); np.fill_diagonal(V, 0.0)
    t = a.sum() + b.sum() + W.sum() + V.sum()
    W *= 5.0 / t; V *= 5.0 / t
    return a, b, e, W, V

def gen_extended(seed):
    """扩展参数域 (更大变异)"""
    rs = np.random.RandomState(seed % (2**31))
    a = rs.uniform(0.005, 0.8, 5)
    b = rs.uniform(0.005, 0.8, 5)
    e = rs.uniform(0.0005, 0.2, 5)
    W = rs.uniform(0.005, 0.5, (5, 5))
    V = rs.uniform(0.005, 0.5, (5, 5))
    np.fill_diagonal(W, 0.0); np.fill_diagonal(V, 0.0)
    return a, b, e, W, V

# ============================================================
# 区间收紧理论
# ============================================================
print("=" * 70)
print("区间收紧策略")
print("=" * 70)
print("""
下界 m_k^(0) = a_k/D_max,k (已证 N_k(M)≥a_k/D_max ∀M∈[0,1]⁵)

上界 u_k^(0) 推导:
  M*_k = (a_k+Σw_kj·M*_j) / (a_k+b_k+ε_k+Σ(w_kj+vkj)·M*_j)
  
  分子 ≤ a_k+Σw_kj  (M*_j≤1)
  分母 ≥ a_k+b_k+ε_k+Σw_kj·M*_j  (扔掉正的v_kj·M*_j项)
  
  u_k^(0) = (a_k+Σw_kj) / (a_k+b_k+ε_k+Σw_kj)
          = 1 - (b_k+ε_k)/(a_k+b_k+ε_k+Σw_kj)  < 1
  
  证明: 令 f(z) = (A+z)/(A+z+B) 其中 A=a_k, z=Σw_kj·M*_j, B=b_k+ε_k+Σv_kj·M*_j.
  f 在 z 上递增 (f' = B/(A+z+B)² > 0), 在 B 上递减.
  z_max = Σw_kj, B_min = b_k+ε_k (取 M*_j=0), 但 z 和 B 不能同时取极端!
  
  保守上界 (允许z和B独立取极端):
  u_k^(0) = (a_k+Σw_kj) / (a_k+Σw_kj+b_k+ε_k)  ← 最紧纯参数上界
           ≥ 真实 M*_k
  
  检验: 对所有200FCA种子, u_k^(0) ≥ M*_k ?
""")

# Quick verification
all_ok = True
max_viol = 0
for s in range(200):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    sum_w = np.sum(W, axis=1)
    u0 = (a + sum_w) / (a + sum_w + b + e)
    
    if np.any(u0 < Mstar * (1 - 1e-12)):
        all_ok = False
        viol_idx = np.where(u0 < Mstar * (1 - 1e-12))[0]
        viol_val = max((Mstar - u0)[viol_idx])
        max_viol = max(max_viol, viol_val)

print(f"  u^(0) ≥ M* 验证 (200 FCA): {'✓' if all_ok else f'✗ 最大违规={max_viol:.6f}'}")

# ============================================================
# 行和界覆盖率 — 区间收紧后的效果
# ============================================================
print(f"\n{'='*70}")
print("覆盖率实验: 行和界 (200 FCA + 500 扩展)")
print("=" * 70)

# Compute g_k(x) = Σ|w-(w+v)x| and evaluate at interval endpoints
def g_k_val(k, x, W, V):
    return sum(abs(W[k,j] - (W[k,j]+V[k,j])*x) for j in range(5) if j != k)

# Level 0: [0, 1] (原始行和界)
# Level 1: [m, 1] (只用下界)
# Level 2: [m, u] (用上下界)
# Level 3: 迭代收紧

for label, gen_fn, n_seeds in [("FCA", gen_FCA, 200), ("扩展", gen_extended, 500)]:
    print(f"\n  --- {label} 域 ({n_seeds}种子) ---")
    
    cov_L0 = 0  # [0,1]
    cov_L1 = 0  # [m,1]
    cov_L2 = 0  # [m,u]
    cov_L3 = 0  # 迭代收紧
    
    for s in range(n_seeds):
        a,b,e,W,V = gen_fn(s)
        Mstar = compute_fp(a,b,e,W,V)
        Dstar = a + b + e + (W+V) @ Mstar
        
        D_max = a + b + e + np.sum(W + V, axis=1)
        sum_w = np.sum(W, axis=1)
        sum_v = np.sum(V, axis=1)
        
        m0 = a / D_max
        u0 = (a + sum_w) / (a + sum_w + b + e)
        
        # L0: [0,1] — 端点max
        ok_L0 = True
        for k in range(5):
            bound_L0 = max(sum_w[k], sum_v[k])  # g_k(0)=Σw, g_k(1)=Σv
            if bound_L0 >= Dstar[k] * (1 - 1e-12):
                ok_L0 = False
        if ok_L0: cov_L0 += 1
        
        # L1: [m,1] — max(g(m), g(1))
        ok_L1 = True
        for k in range(5):
            g_m = g_k_val(k, m0[k], W, V)
            g_1 = g_k_val(k, 1.0, W, V)
            bound_L1 = max(g_m, g_1)
            if bound_L1 >= Dstar[k] * (1 - 1e-12):
                ok_L1 = False
        if ok_L1: cov_L1 += 1
        
        # L2: [m,u] — max(g(m), g(u))
        ok_L2 = True
        for k in range(5):
            g_m = g_k_val(k, m0[k], W, V)
            g_u = g_k_val(k, u0[k], W, V)
            bound_L2 = max(g_m, g_u)
            if bound_L2 >= Dstar[k] * (1 - 1e-12):
                ok_L2 = False
        if ok_L2: cov_L2 += 1
        
        # L3: [m,u] using tightened D* lower bound
        D_low = a + b + e + (W+V) @ m0
        ok_L3 = True
        for k in range(5):
            g_m = g_k_val(k, m0[k], W, V)
            g_u = g_k_val(k, u0[k], W, V)
            bound_L3 = max(g_m, g_u)
            if bound_L3 >= D_low[k] * (1 - 1e-12):
                ok_L3 = False
        if ok_L3: cov_L3 += 1
    
    print(f"    L0 [0,1]:  {cov_L0}/{n_seeds} ({100*cov_L0/n_seeds:.1f}%)")
    print(f"    L1 [m,1]:  {cov_L1}/{n_seeds} ({100*cov_L1/n_seeds:.1f}%)")
    print(f"    L2 [m,u]:  {cov_L2}/{n_seeds} ({100*cov_L2/n_seeds:.1f}%)")
    print(f"    L3 [m,u]+Dlow: {cov_L3}/{n_seeds} ({100*cov_L3/n_seeds:.1f}%)")

# ============================================================
# 分析剩余未覆盖种子
# ============================================================
print(f"\n{'='*70}")
print("深度分析: L2 [m,u] 未覆盖种子的特征")
print("=" * 70)

fail_seeds = []
for s in range(200):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    Dstar = a + b + e + (W+V) @ Mstar
    D_max = a + b + e + np.sum(W+V, axis=1)
    sum_w = np.sum(W, axis=1)
    sum_v = np.sum(V, axis=1)
    m0 = a / D_max
    u0 = (a + sum_w) / (a + sum_w + b + e)
    
    for k in range(5):
        g_m = g_k_val(k, m0[k], W, V)
        g_u = g_k_val(k, u0[k], W, V)
        bound_L2 = max(g_m, g_u)
        if bound_L2 >= Dstar[k] * (1 - 1e-10):
            fail_seeds.append((s,k,bound_L2/Dstar[k], m0[k], u0[k], Mstar[k],
                              g_k_val(k, Mstar[k], W, V)/Dstar[k]))
            break

if fail_seeds:
    print(f"\n  {'seed':>6} {'k':>3} {'ratio':>8} {'m^(0)':>8} {'u^(0)':>8} {'M*':>8} {'true':>8}")
    print(f"  {'-'*60}")
    for s,k,ratio,m,u,ms,tr in fail_seeds[:20]:
        print(f"  {s:>6} {k:>3} {ratio:>8.4f} {m:>8.4f} {u:>8.4f} {ms:>8.4f} {tr:>8.4f}")
else:
    print("  ✓ 所有种子均被 L2 覆盖!")

# ============================================================
# 对失败种子尝试迭代收紧
# ============================================================
print(f"\n{'='*70}")
print("迭代收紧试验 (对失败种子)")
print("=" * 70)

# 利用 M*_j ∈ [m_j, u_j] 收紧 D*_k 的下界
# D*_k = a_k+b_k+ε_k + Σ(w+v)·M*_j ≥ a_k+b_k+ε_k + Σ(w+v)·m_j = D_low_k
# 
# 同时 X_k = Σ(w-(w+v)M*_k)M*_j 的界也可收紧:
#   对每个 j, (w_kj - (w_kj+v_kj)M*_k)M*_j
#   该乘积在 M*_j∈[m_j,u_j] 上线性 (M*_k固定), 取端点max
#   但这本身又依赖 M*_k... 

# 更简单: 迭代收紧 M* 的上下界
#   m_k^{(t+1)} = min_{M∈[m^{(t)},u^{(t)}]∩[0,1]⁵} N_k(M)
#   u_k^{(t+1)} = max_{M∈[m^{(t)},u^{(t)}]∩[0,1]⁵} N_k(M)
# 
# 由于N不单调, 需要搜索...

# 尝试: 对失败种子, 迭代3轮
if fail_seeds:
    test_s = fail_seeds[0][0]
    a,b,e,W,V = gen_FCA(test_s)
    Mstar = compute_fp(a,b,e,W,V)
    D_max = a+b+e+np.sum(W+V, axis=1)
    D_low_init = a+b+e+(W+V)@(a/D_max)
    
    print(f"\n  种子 {test_s}:")
    print(f"    M* = {[f'{x:.4f}' for x in Mstar]}")
    print(f"    m^(0) = {[f'{x:.4f}' for x in (a/D_max)]}")
    
    # 尝试: 用 M*_j ≥ m_j 收紧每个分子项
    # Σ_j |w_kj - (w_kj+v_kj)M*_k| · M*_j 的界
    # 不对, g_k(M*_k) 只依赖于 M*_k, 不依赖于其他 M*_j
    #
    # 问题在于: g_k(x) = Σ_j |w_kj - (w_kj+v_kj)x| 
    # 若我们知道 x ∈ [m_k, u_k], 则最大在 endpoint
    # 问题在于 m_k 和 u_k 还不够紧!
    
    # 试试迭代: 
    # 已知 m_j^(0) ≤ M*_j ≤ u_j^(0) ∀j
    # D*_k 的下界可以收紧:
    #   D*_k ≥ a_k+b_k+ε_k + Σ_j (w_kj+v_kj)·m_j^(0)
    # 
    # M*_k 的上界可以收紧:
    #   M*_k ≤ (a_k+Σ_j w_kj·u_j^(0)) / (a_k+b_k+ε_k+Σ_j w_kj·m_j^(0))
    # 因为分子被 M*_j≤u_j 收紧, 分母被 M*_j≥m_j 收紧
    
    # 类似地下界:
    #   M*_k ≥ (a_k+Σ_j w_kj·m_j^(0)) / (a_k+b_k+ε_k+Σ_j (w_kj+v_kj)·u_j^(0))
    
    # 迭代一轮
    m = a / D_max
    u = (a + np.sum(W, axis=1)) / (a + np.sum(W, axis=1) + b + e)
    
    for it in range(5):
        m_next = (a + W @ m) / (a + b + e + (W+V) @ u)
        u_next = (a + W @ u) / (a + b + e + (W+V) @ m)
        
        # Clamp
        m_next = np.maximum(m_next, 0)
        u_next = np.minimum(u_next, 1)
        
        print(f"    iter {it}: m=[{', '.join(f'{x:.4f}' for x in m)}]  "
              f"u=[{', '.join(f'{x:.4f}' for x in u)}]")
        
        if np.max(np.abs(m_next - m)) < 1e-6 and np.max(np.abs(u_next - u)) < 1e-6:
            break
        m, u = m_next, u_next
    
    # 用收紧的区间重算 g_k bound
    print(f"\n    区间收紧后的行和界 vs D*:")
    Dstar = a+b+e+(W+V)@Mstar
    for k in range(5):
        g_m = g_k_val(k, m[k], W, V)
        g_u = g_k_val(k, u[k], W, V)
        g_bound = max(g_m, g_u)
        g_true = g_k_val(k, Mstar[k], W, V)
        print(f"      k={k}: bound={g_bound:.4f} / D*={Dstar[k]:.4f} = {g_bound/Dstar[k]:.4f}  "
              f"(true={g_true/Dstar[k]:.4f}, M*∈[{m[k]:.4f},{u[k]:.4f}]) {'✓' if g_bound<Dstar[k] else '✗' if g_bound/Dstar[k]>1.001 else '~'}")

# ============================================================
# 全扫描：迭代收紧后的覆盖率
# ============================================================
print(f"\n{'='*70}")
print("全扫描: 迭代收紧 (2轮) 后的行和界覆盖率")
print("=" * 70)

for label, gen_fn, n_seeds in [("FCA", gen_FCA, 200), ("扩展", gen_extended, 500)]:
    cov_iterated = 0
    cov_Dlow = 0
    max_ratio = 0
    
    for s in range(n_seeds):
        a,b,e,W,V = gen_fn(s)
        Mstar = compute_fp(a,b,e,W,V)
        Dstar = a+b+e+(W+V)@Mstar
        D_max = a+b+e+np.sum(W+V, axis=1)
        
        m = a / D_max
        u = (a + np.sum(W, axis=1)) / (a + np.sum(W, axis=1) + b + e)
        
        # 迭代 2 轮
        for _ in range(2):
            m = (a + W @ m) / (a + b + e + (W+V) @ u)
            u = (a + W @ u) / (a + b + e + (W+V) @ m)
            m = np.maximum(m, 0)
            u = np.minimum(u, 1)
        
        ok = True
        for k in range(5):
            g_m = g_k_val(k, m[k], W, V)
            g_u = g_k_val(k, u[k], W, V)
            bound = max(g_m, g_u)
            ratio = bound / Dstar[k]
            max_ratio = max(max_ratio, ratio)
            if ratio >= 1 - 1e-10:
                ok = False
        
        if ok: cov_iterated += 1
    
    print(f"  {label} 域 ({n_seeds}): 迭代后覆盖={cov_iterated}/{n_seeds} ({100*cov_iterated/n_seeds:.1f}%)  "
          f"max ratio={max_ratio:.4f}")

print(f"\n{'='*70}")
print("结论")
print("=" * 70)
