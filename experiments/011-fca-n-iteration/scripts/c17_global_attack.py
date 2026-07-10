"""
6.17C 全局方向单调性：深度结构分析
====================================
核心问题: 凹凸性引理(∂²N_k/∂M_j² 符号固定)能否推出全局 F(M)>0?
F(M) = (N(M)-M)·(M*-M)

策略 A: 坐标切片 — 固定其他坐标, F(M_j)的极值位置
策略 B: 正交分解 — F = ||Δ||² - Δ^T diag(D*/D) J Δ
策略 C: 逐分量下界 — 证明每个(N_k-M_k)(M*_k-M_k)可被边界包围
策略 D: 连接到 l₁ 收缩 — 用 6.17B 导出方向单调性
"""
import numpy as np

def n_operator(M,a,b,eps,W,V):
    num=a+W@M; return num/(num+b+V@M+eps)
def compute_fp(a,b,eps,W,V):
    M=np.full(5,.5)
    for _ in range(20000):
        Mn=n_operator(M,a,b,eps,W,V)
        if np.max(np.abs(Mn-M))<1e-15: return Mn
        M=Mn
    return M
def gen_FCA(seed):
    rs=np.random.RandomState(seed%(2**31))
    a=rs.uniform(.01,.5,5);b=rs.uniform(.01,.5,5);e=rs.uniform(.001,.1,5)
    W=rs.uniform(.01,.3,(5,5));V=rs.uniform(.01,.3,(5,5))
    np.fill_diagonal(W,0);np.fill_diagonal(V,0)
    t=a.sum()+b.sum()+W.sum()+V.sum();W*=5./t;V*=5./t
    return a,b,e,W,V

# ============================================================
print("="*70)
print("策略 A: 坐标切片分析 — F(M_j)的极值位置")
print("="*70)

# 对每个种子, 固定其他3个坐标, 扫描M_j ∈ [0,1]
# 检查F(M_j)的最小值是否总在端点

for seed_id in [0, 11, 21, 67, 149]:
    a,b,e,W,V = gen_FCA(seed_id)
    Mstar = compute_fp(a,b,e,W,V)
    
    interior_violations = 0
    total_slices = 0
    
    for _ in range(500):
        # Randomly select j and other coordinates
        j = np.random.randint(0,5)
        M_others = np.random.uniform(0.05, 0.95, 5)
        M_others[j] = np.nan  # placeholder
        
        # Sample M_j at 50 points
        M_j_vals = np.linspace(0.01, 0.99, 50)
        F_vals = []
        for mj in M_j_vals:
            M = M_others.copy()
            M[j] = mj
            N_val = n_operator(M, a, b, e, W, V)
            F_vals.append(np.sum((N_val - M) * (Mstar - M)))
        
        F_vals = np.array(F_vals)
        F_min = np.min(F_vals)
        F_endpoints = min(F_vals[0], F_vals[-1])
        F_interior_min = np.min(F_vals[1:-1])
        
        total_slices += 1
        # Interior minimum significantly lower than endpoints
        if F_interior_min < F_endpoints * 0.99:  # 1% tolerance
            interior_violations += 1
    
    print(f"  seed {seed_id}: 内部极值低于端点的切片比例 = "
          f"{interior_violations}/{total_slices} ({100*interior_violations/total_slices:.1f}%)")

# ============================================================
print(f"\n{'='*70}")
print("策略 A2: 更精确的切片扫描 — 稠密采样 + 二阶分析")
print("="*70)

for seed_id in [0, 11]:
    a,b,e,W,V = gen_FCA(seed_id)
    Mstar = compute_fp(a,b,e,W,V)
    
    print(f"\n  种子 {seed_id}: M*={[f'{x:.4f}' for x in Mstar]}")
    
    for j in range(5):
        # Fix ALL other coordinates at their true M* values
        M_ref = Mstar.copy()
        M_ref[j] = np.nan
        
        M_j = np.linspace(0.001, 0.999, 200)
        F_vals = np.zeros(200)
        
        for i, mj in enumerate(M_j):
            M = M_ref.copy()
            M[j] = mj
            Nv = n_operator(M, a, b, e, W, V)
            F_vals[i] = np.sum((Nv - M) * (Mstar - M))
        
        # Find interior minimum
        interior_idx = slice(10, -10)
        F_min_idx = np.argmin(F_vals)
        F_min_val = F_vals[F_min_idx]
        F_min_mj = M_j[F_min_idx]
        
        endpoint_min = min(F_vals[0], F_vals[-1])
        
        print(f"    M_{j}: F端点min={endpoint_min:.6f}  "
              f"F全域min={F_min_val:.6f} @ M_{j}={F_min_mj:.3f}  "
              f"{'✓(端点是min)' if F_min_val >= endpoint_min*0.999 else '✗(内部更低!)'}")

# ============================================================
print(f"\n{'='*70}")
print("策略 B: F 的正交分解 — Δ^T(I - diag(D*/D) J)Δ")
print("="*70)

for seed_id in [0, 11, 21]:
    a,b,e,W,V = gen_FCA(seed_id)
    Mstar = compute_fp(a,b,e,W,V)
    Dstar = a+b+e+(W+V)@Mstar
    
    J = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k!=j: J[k,j] = (W[k,j]*(1-Mstar[k])-V[k,j]*Mstar[k])/Dstar[k]
    
    Sym = (np.eye(5)-J + (np.eye(5)-J).T)/2
    lam_min = np.min(np.linalg.eigvalsh(Sym))
    
    # For many random M, compute the gap
    min_F_norm = np.inf
    min_gap = np.inf
    
    for _ in range(5000):
        M = np.random.uniform(0.05, 0.95, 5)
        D = a+b+e+(W+V)@M
        delta = M - Mstar
        norm_delta = np.linalg.norm(delta)
        if norm_delta < 1e-10: continue
        
        Nv = n_operator(M, a, b, e, W, V)
        raw_F = np.sum((Nv - M) * (Mstar - M))
        
        # B1: Lower bound using quadratic form
        quad = delta @ Sym @ delta
        F_norm = raw_F / norm_delta**2
        min_F_norm = min(min_F_norm, F_norm)
        
        # B2: Tightness gap
        D_ratio = Dstar / D
        scale_J = np.diag(D_ratio) @ J
        exact_quad = delta @ (np.eye(5) - scale_J) @ delta
        gap = raw_F - exact_quad
        min_gap = min(min_gap, gap / norm_delta**2)
    
    print(f"  seed {seed_id}: λ_min(sym(I-J))={lam_min:.4f}  "
          f"inf F/||Δ||²={min_F_norm:.4f}  "
          f"min gap/||Δ||²={min_gap:.6f}")

# ============================================================
print(f"\n{'='*70}")
print("策略 C: (N_k-M_k)(M*_k-M_k)的逐分量下界")
print("="*70)

for seed_id in [0, 11]:
    a,b,e,W,V = gen_FCA(seed_id)
    Mstar = compute_fp(a,b,e,W,V)
    
    # With the iteration bounds m,u, can we bound each component?
    D_max = a+b+e+np.sum(W+V, axis=1); sum_w=np.sum(W,axis=1)
    m = a/D_max; u = (a+sum_w)/(a+sum_w+b+e)
    for _ in range(2):
        mn=(a+W@m)/(a+b+e+(W+V)@u); un=(a+W@u)/(a+b+e+(W+V)@m)
        m=np.maximum(mn,0); u=np.minimum(un,1)
    
    print(f"  seed {seed_id}: M*={[f'{x:.4f}' for x in Mstar]}")
    print(f"    m_(2)={[f'{x:.4f}' for x in m]}  u_(2)={[f'{x:.4f}' for x in u]}")
    print(f"    gap: M*-m={[f'{x:.4f}' for x in (Mstar-m)]}  u-M*={[f'{x:.4f}' for x in (u-Mstar)]}")
    
    # 关键: M*_k 在[m_k,u_k]内, 所以 (M_k-M*_k) 的符号:
    # 若 M_k < m_k ≤ M*_k, 则 M_k < M*_k
    # 若 M_k > u_k ≥ M*_k, 则 M_k > M*_k
    # 若 M_k ∈ [m_k, u_k], 符号不确定
    
    # Bounds on N_k:
    # N_k ≥ (a_k+Σw_kj m_j) / (a_k+b_k+ε_k+Σ(w+v)_kj u_j) = m_k^(1)
    # N_k ≤ (a_k+Σw_kj u_j) / (a_k+b_k+ε_k+Σ(w+v)_kj m_j) = u_k^(1)
    
    m1 = (a+W@m)/(a+b+e+(W+V)@u)
    u1 = (a+W@u)/(a+b+e+(W+V)@m)
    
    print(f"\n    当 M_k < m_k 时: N_k ∈ [{min(m1):.4f},{max(u1):.4f}]")
    print(f"    若 N_k-min > M_k 则 (N_k-M_k)(M*_k-M_k) > 0")
    print(f"    区域外分量自动 > 0")

# ============================================================
print(f"\n{'='*70}")
print("策略 D: l₁ 收缩 → 方向单调性 的逻辑连通")
print("="*70)

print("""
l₁ 收缩 (6.17B): ||N(M)-M*||₁ ≤ α||M-M*||₁
方向单调性 (6.17C): (N-M)·(M*-M) > 0

关系: 若 α<1 且 M坐标wise在M*同侧, 可导出方向单调性.
但一般情况 (混合符号) 需要更精细论证.

关键不等式:
  (N-M)·(M*-M) = Σ(N_k-M_k)(M*_k-M_k)
  = Σ[(N_k-M*_k)+(M*_k-M_k)](M*_k-M_k)
  = Σ[(N_k-M*_k)(M*_k-M_k) - (M_k-M*_k)²]
  = Σ(N_k-M*_k)(M*_k-M_k) - ||M-M*||²

令 p_k = N_k-M*_k, d_k = M_k-M*_k
则 F = -||d||² + p·d

用 Cauchy-Schwarz: |p·d| ≤ ||p||₁·||d||∞ (不对...)
用 Hölder: |p·d| ≤ ||p||₁·||d||∞

由 6.17B: ||p||₁ ≤ α||d||₁ ≤ 5α||d||∞
所以 |p·d| ≤ 5α||d||∞² ≤ 5α||d||²

因此 F ≥ ||d||² - 5α||d||² = (1-5α)||d||²

当 α < 0.2 时 F > 0. 实证: α ∈ [0.15, 0.545]
对于 α ≥ 0.2 的种子, 此界不够.

但此界过于保守 (用了 ||d||₁ ≤ 5||d||∞ 的松估计).
""")

# Better bound attempt
print("更紧的界:")
print("  F = ||d||² - d^T diag(D*/D) J d  (来自6.17A)")
print("  设 S = diag(D*/D)J, 则 F = d^T(I-S)d ")
print("  = d^T(I-J)d + d^T(J-S)d")
print("  ≥ λ_min(sym(I-J))||d||² + d^T(J-S)d")
print("")
print("  J-S = J - diag(D*/D)J = (I-diag(D*/D))J")
print("  (I-diag(D*/D))_k = 1-D*/D_k = (D_k-D*_k)/D_k = Σ(w+v)_kj d_j / D_k")
print("")
print("  因此 d^T(J-S)d = Σ_k Σ_j J_kj d_k d_j (D_k-D*_k)/D_k")
print("  = Σ_k (Σ_j J_kj d_j) · d_k · (D_k-D*_k)/D_k")
print("  = Σ_k (N_k-M*_k)·(D_k/D*_k)·d_k·(D_k-D*_k)/D_k")
print("  = Σ_k (N_k-M*_k)·d_k·(D_k-D*_k)/D*_k")
print("")
print("  Note: D_k-D*_k = Σ(w+v)_kj(M_j-M*_j) = Σ(w+v)_kj d_j")
print("  所以此项 = Σ_k (N_k-M*_k)·d_k·(Σ_j(w+v)_kj d_j)/D*_k")
print("")
print("  关键: 符号分析. N_k-M*_k 和 d_k 符号:")
print("  当 M_k < M*_k 时, N_k? M*_k → 若 N_k > M*_k (超调), 则(N_k-M*_k)>0")
print("  但若 N 是收缩的, 超调幅度有限...")

# ============================================================
print(f"\n{'='*70}")
print("策略 B2: J 行和界 + 对角D*/D 的联合界")
print("="*70)

# F = Σ_k d_k² - Σ_k D*/D_k Σ_j J_kj d_k d_j
# = Σ_k d_k² - Σ_k Σ_j J_kj d_k d_j - Σ_k (D*/D_k - 1) Σ_j J_kj d_k d_j
# = d^T(I-J)d - Σ_k (D*/D_k - 1) Σ_j J_kj d_k d_j

# The "extra" term beyond the quadratic form:
# Extra = Σ_k (1 - D*/D_k) Σ_j J_kj d_k d_j
#       = Σ_k Σ_j J_kj d_k d_j (D_k-D*_k)/D_k

# But (D_k-D*_k)/D_k = Σ_l (w+v)_kl d_l / D_k
# And D_k ≥ D_min,k

# 上界: |Extra| ≤ Σ_k Σ_j |J_kj| |d_k| |d_j| · Σ_l (w+v)_kl |d_l|/D_min,k
# 当所有 |d_l| 小 → Extra = O(||d||³)
# 当有 d_j → 边界 → ||d|| 大

print("在座标切片上的行为分析...")
for seed_id in [0, 11]:
    a,b,e,W,V = gen_FCA(seed_id)
    Mstar = compute_fp(a,b,e,W,V)
    Dstar = a+b+e+(W+V)@Mstar
    D_min = a+b+e
    
    # Along random directions, compute F vs its quadratic approx
    D_max = a+b+e+np.sum(W+V, axis=1)
    J=np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k!=j: J[k,j]=(W[k,j]*(1-Mstar[k])-V[k,j]*Mstar[k])/Dstar[k]
    
    print(f"\n  种子 {seed_id}: F vs 二次近似 vs ||Δ||")
    print(f"  {'||Δ||':>8}  {'F':>10}  {'Δ^T(I-J)Δ':>12}  {'Extra':>10}  {'F/||Δ||²':>10}")
    print(f"  {'-'*55}")
    
    for _ in range(10):
        u = np.random.randn(5); u /= np.linalg.norm(u)
        for r in [0.05, 0.2, 0.5, 1.0, 2.0, 3.0]:
            M = Mstar + r*u
            if np.any(M < 1e-6) or np.any(M > 1-1e-6): continue
            M = np.clip(M, 1e-6, 1-1e-6)
            D = a+b+e+(W+V)@M
            delta = M-Mstar
            nd = np.linalg.norm(delta)
            
            F_raw = np.sum((n_operator(M,a,b,e,W,V)-M)*(Mstar-M))
            quad = delta @ (np.eye(5)-J) @ delta
            extra = F_raw - quad
            
            print(f"  {nd:>8.3f}  {F_raw:>10.4f}  {quad:>12.4f}  {extra:>10.4f}  {F_raw/nd**2:>10.4f}")

print(f"\n{'='*70}")
print("综合结论")
print("="*70)
