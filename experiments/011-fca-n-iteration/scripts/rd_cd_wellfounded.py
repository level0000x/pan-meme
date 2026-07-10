"""
攻坚 RD/CD 纯良基充分条件
===========================
路径1: 利用 |w(1-N)-vN| ≤ max(w,v) 替换三角不等式
得到纯良基充分条件: Σ max(w,v) < D_min

路径2: M*_k 的界 → 符号模式压缩 → 条件类检验
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
    """扩展参数域，a,b∈[0.005,0.5], e∈[0.0005,0.5]"""
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

# ============================================================
# 路径1: Σ max(w,v) < D_min 纯良基充分条件
# ============================================================
print("=" * 70)
print("路径1: 纯良基 RD/CD 充分条件")
print()
print("关键引理: |w_kj(1-M*_k) - v_kj M*_k| ≤ max(w_kj, v_kj)")
print("(因为 w,v≥0, M*_k∈[0,1], 两项皆正，差≤max)")
print()
print("由此: |J_kj(M*)| ≤ max(w_kj, v_kj) / D*_k ≤ max(w_kj, v_kj) / D_min,k")
print()
print("纯良基充分条件:")
print("  RD: Σ_{j≠k} max(w_kj, v_kj) < a_k + b_k + ε_k")
print("  CD: Σ_{k≠j} max(w_kj, v_kj) < a_j + b_j + ε_j")
print("  (仅依赖参数，无需计算M*)")
print()

# 验证 200 FCA + 500 扩展域
results = []
for gen_func, domain_name, n_seeds in [(gen_FCA, "FCA", 200), (gen_extended, "扩展域", 500)]:
    rd_ok = 0; cd_ok = 0; both_ok = 0
    
    for seed in range(n_seeds):
        a, b, e, W, V = gen_func(seed)
        Mstar = compute_fp(a, b, e, W, V)
        Dstar = (a + W @ Mstar) + (b + V @ Mstar) + e
        D_min = a + b + e
        
        # 实际 J(M*)
        J = np.zeros((5, 5))
        for k in range(5):
            for j in range(5):
                J[k, j] = (W[k, j] * (1.0 - Mstar[k]) - Mstar[k] * V[k, j]) / Dstar[k]
        
        # 实际 RD/CD
        actual_RD = np.array([sum(abs(J[k, j]) for j in range(5) if j != k) for k in range(5)])
        actual_CD = np.array([sum(abs(J[k, j]) for k in range(5) if k != j) for j in range(5)])
        
        # 纯良基充分条件
        max_WV = np.maximum(W, V)
        bound_RD = np.array([sum(max_WV[k, j] for j in range(5) if j != k) / D_min[k] for k in range(5)])
        bound_CD = np.array([sum(max_WV[k, j] for k in range(5) if k != j) / D_min[j] for j in range(5)])
        
        rd_pass = bool(np.all(bound_RD < 1.0))
        cd_pass = bool(np.all(bound_CD < 1.0))
        both_pass = rd_pass and cd_pass
        
        if rd_pass: rd_ok += 1
        if cd_pass: cd_ok += 1
        if both_pass: both_ok += 1
        
        if seed < 30 or both_pass:
            results.append({
                'domain': domain_name, 'seed': seed,
                'bound_RD': np.max(bound_RD), 'bound_CD': np.max(bound_CD),
                'actual_RD': np.max(actual_RD), 'actual_CD': np.max(actual_CD),
                'rd_pass': rd_pass, 'cd_pass': cd_pass, 'both_pass': both_pass,
                'D_min': D_min, 'Mstar': Mstar,
            })
    
    print(f"  {domain_name} ({n_seeds} 种子):")
    print(f"    纯良基 RD 通过: {rd_ok}/{n_seeds} ({100*rd_ok/n_seeds:.1f}%)")
    print(f"    纯良基 CD 通过: {cd_ok}/{n_seeds} ({100*cd_ok/n_seeds:.1f}%)")
    print(f"    两者同时通过:   {both_ok}/{n_seeds} ({100*both_ok/n_seeds:.1f}%)")

# 看几个失败案例
print()
print("=== 失败案例分析 (FCA) ===")
fca_fails = [r for r in results if r['domain'] == 'FCA' and not r['both_pass']]
for r in fca_fails[:3]:
    print(f"  seed {r['seed']}: bound_RD={r['bound_RD']:.3f}, actual_RD={r['actual_RD']:.3f}, "
          f"bound_CD={r['bound_CD']:.3f}, actual_CD={r['actual_CD']:.3f}")
    print(f"    M* = {r['Mstar']}")

print()

# ============================================================
# 路径2: M* 的界 → J 符号模式压缩
# ============================================================
print("=" * 70)
print("路径2: M* 的界 → J 符号模式压缩")
print()
print("M*_k = A*_k / D*_k")
print("A*_k = a_k + (WM*)_k ≥ a_k ≥ 0.01")
print("D*_k = A*_k + B*_k + ε_k ≤ total_mass / something")
print()
print("由于 FCA 归一化: Σ(all params) = 5, 且 a_k,b_k,ε_k 各自有下界")
print("M*_k 被约束在 [m_low, m_high] ⊆ [0,1] 内, m_low > 0, m_high < 1")
print()

# 实际 M* 范围
Mstar_all_fca = []
Mstar_all_ext = []
for seed in range(200):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    Mstar_all_fca.append(Mstar)
for seed in range(500):
    a, b, e, W, V = gen_extended(seed)
    Mstar = compute_fp(a, b, e, W, V)
    Mstar_all_ext.append(Mstar)

Mstar_fca = np.array(Mstar_all_fca)
Mstar_ext = np.array(Mstar_all_ext)

print("  实测 M* 范围:")
print(f"    FCA:   min={Mstar_fca.min():.4f}, max={Mstar_fca.max():.4f}")
print(f"    扩展域: min={Mstar_ext.min():.4f}, max={Mstar_ext.max():.4f}")
print()

# 理论下界 (conservative):
# M*_k ≥ a_k / (max possible D_k)
# D_k ≤ a_k + b_k + ε_k + Σ(w_kj+v_kj) = total_row_sum
# total_row_sum ≤ max_row_total
# 由于归一化 Σ(all params)=5, Σ_{row k} params ≤ 5
# 但单个 D*_k 可能很大——取决于耦合配置

# 更好的下界: M*_k ≥ a_k / D*_k
# 且 D*_k ≤ a_k + b_k + ε_k + Σ(w+v)_kj ≤ total_assigned_to_k
# 在最紧的 FCA 归一化下: Σ_k D*_k = Σ_k (A*_k+B*_k+ε_k) 
#   = Σ_k(a_k+b_k+ε_k) + Σ_k (WM*) + Σ_k (VM*)
#   = Σ_k(a_k+b_k+ε_k) + Σ_{i,j} (w_ij+v_ij)M*_j
#   ≤ Σ_k(a_k+b_k+ε_k) + Σ_{i,j} (w_ij+v_ij)  (M*_j ≤ 1)

print("=== 解析 M* 界 ===")
# Conservative: M*_k ≥ a_k / (a_k + b_k + ε_k + Σ(w+v)_kj)
# 因为 N_k = A_k/(A_k+B_k+ε_k) 且 A_k ≥ a_k, D_k ≤ max possible
# Actually D_k = A_k + B_k + ε_k where A_k = a_k + (WM*)_k, B_k = b_k + (VM*)_k
# So D*_k = a_k + b_k + ε_k + ((W+V)M*)_k
# D*_k ≤ a_k + b_k + ε_k + Σ_j (w_kj+v_kj) * 1 = D_max,k

print("  D*_k ≤ D_max,k ≡ a_k + b_k + ε_k + Σ_{j≠k} (w_kj+v_kj)")
print("  D*_k ≥ D_min,k ≡ a_k + b_k + ε_k")
print("  M*_k ≥ a_k / D_max,k (conservative)")
print("  M*_k ≤ (a_k + Σ w_kj) / D_min,k (conservative)")

# Test conservativeness
for seed in [0, 11, 42]:
    a,b,e,W,V = gen_FCA(seed)
    Mstar = compute_fp(a,b,e,W,V)
    D_min = a + b + e
    D_max = a + b + e + np.sum(W + V, axis=1)
    M_low = a / D_max
    M_high = (a + np.sum(W, axis=1)) / D_min
    
    print(f"  seed {seed}: M*={Mstar}")
    print(f"    M_low={M_low}, M_high={M_high}")
    print(f"    within bounds: {np.all(M_low <= Mstar):s} / {np.all(Mstar <= M_high):s}")
    print(f"    tightness: {(Mstar-M_low).mean():.4f} / {(M_high-Mstar).mean():.4f}")

print()

# ============================================================
# 路径3: 直接用 M* 的界优化 RD/CD bound
# ============================================================
print("=" * 70)
print("路径3: 用 M* 界优化 J 的 RD/CD bound")
print()
print("|J_kj| = |w_kj(1-M*_k) - v_kj M*_k| / D*_k")
print("在已知 M*_k ∈ [m_low, m_high] 时:")
print("  w_kj(1-M*_k) - v_kj M*_k = w_kj - (w_kj+v_kj)M*_k")
print("  在 [m_low, m_high] 上线性, 最大值在端点")
print("  max = max(|w_kj - (w+v)m_low|, |w_kj - (w+v)m_high|)")
print()

# 实际 max|J| 的分布
print("=== 比较不同的 |J| 上界 ===")
for seed in [0, 11, 42]:
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = (a + W @ Mstar) + (b + V @ Mstar) + e
    D_min = a + b + e
    
    J = np.zeros((5, 5))
    for k in range(5):
        for j in range(5):
            J[k, j] = (W[k, j] * (1.0 - Mstar[k]) - Mstar[k] * V[k, j]) / Dstar[k]
    
    # Bound 1: 三角不等式 |J| ≤ (w+v)/D_min
    b1 = np.array([[(W[k,j]+V[k,j])/D_min[k] for j in range(5)] for k in range(5)])
    np.fill_diagonal(b1, 0)
    
    # Bound 2: max(w,v) bound |J| ≤ max(w,v)/D_min
    b2 = np.array([[max(W[k,j], V[k,j])/D_min[k] for j in range(5)] for k in range(5)])
    np.fill_diagonal(b2, 0)
    
    # Bound 3: 用实际 M* 的界 (知道 M*_k 后)
    # 但这是逐实例的, 不算纯良基
    b3 = np.zeros((5, 5))
    for k in range(5):
        for j in range(5):
            if j == k: continue
            # |w(1-M_k) - v M_k| / D*_k
            val_low = abs(W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k]
            b3[k, j] = val_low
    
    actual_RD = np.array([sum(abs(J[k, j]) for j in range(5) if j != k) for k in range(5)])
    b1_RD = np.sum(b1, axis=1)
    b2_RD = np.sum(b2, axis=1)
    
    print(f"  seed {seed}:")
    for k in range(5):
        print(f"    k={k}: actual={actual_RD[k]:.4f}, "
              f"b1=(w+v)/D_min={b1_RD[k]:.4f}, "
              f"b2=max(w,v)/D_min={b2_RD[k]:.4f}")
    rd_pass_b1 = np.all(b1_RD < 1)
    rd_pass_b2 = np.all(b2_RD < 1)
    print(f"    b1 (三角不等式) 能证明RD: {rd_pass_b1}, b2 (max bound) 能证明RD: {rd_pass_b2}")

print()
print("=" * 70)
print("总结")
print()
print("路径1 (纯良基 Σ max(w,v) < D_min):")
print("  提供了不依赖 M* 的充分条件，但覆盖率有限")
print("  D_min = D*_k 的最小可能值 → 过于保守")
print()
print("路径2 (M* 的界):")
print("  M* 有解析上下界 [a/D_max, (a+Σw)/D_min]")
print("  但区间较宽 → 符号模式可能摇摆")
print()
print("最优策略: 结合路径1+2")
print("  用 D_max (而非 D_min) 做分母 + max(w,v) bound")
print("  得到: |J_kj| ≤ max(w_kj, v_kj) / (a_k + b_k + ε_k)")
print("  充分条件: Σ max(w,v) < D_min (已检验)")
print("  更进一步: 用 D_max 替代部分 bound 提高覆盖率")
