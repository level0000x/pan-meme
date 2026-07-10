"""
审计: N 的单调性 + 直接下界不等式
===================================
关键问题:
  Q1: N 是否单调? (即 M ≥ M' ⇒ N(M) ≥ N(M'))
  Q2: 如果不单调, N_k(M) ≥ m_k^(0) 是否仍然成立?
  Q3: 6.17B 的引理证明是否可以不依赖单调性?

理论分析:
  m_k^(0) = a_k / D_max, 其中 D_max = a_k + b_k + ε_k + Σ_j (w_kj+v_kj)
  
  对任意 M ∈ [0,1]^5:
  N_k(M) = (a_k + Σ w M_j) / (a_k + b_k + ε_k + Σ (w+v) M_j)
         ≥ (a_k + Σ w M_j) / D_max              (分母 ≤ D_max)
         ≥ a_k / D_max                          (分子 ≥ a_k)
         = m_k^(0)
  
  这不需要 N 的单调性! 这是纯代数下界.
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
# Q1: N 的单调性测试
# ============================================================
print("=" * 70)
print("Q1: N 是否单调?")
print("=" * 70)

for seed_id in range(10):
    a, b, e, W, V = gen_FCA(seed_id)
    
    violations = 0
    total = 0
    for _ in range(5000):
        M1 = np.random.uniform(0.01, 0.99, 5)
        M2 = np.random.uniform(0.01, 0.99, 5)
        # Ensure M1 ≥ M2 component-wise
        M_big = np.maximum(M1, M2)
        M_sml = np.minimum(M1, M2)
        
        N_big = n_operator(M_big, a, b, e, W, V)
        N_sml = n_operator(M_sml, a, b, e, W, V)
        
        total += 5
        for k in range(5):
            if N_big[k] < N_sml[k] * (1 - 1e-12):
                violations += 1
    
    if violations > 0:
        print(f"  ✗ seed {seed_id}: {violations}/{total} 单调性违规 ({(violations/total)*100:.1f}%)")
    else:
        print(f"  seed {seed_id}: 0/{total} 违规")

# ============================================================
# Q1b: 检查 ∂N_k/∂M_j 的符号
# ============================================================
print(f"\n{'='*70}")
print("Q1b: ∂N_k/∂M_j 符号分析 (能否为负?)")
print("=" * 70)
print("  ∂N_k/∂M_j = (w_kj·D_k - A_k·(w_kj+v_kj)) / D_k²")
print("  负号条件: w_kj·D_k < A_k·(w_kj+v_kj)")
print()

for seed_id in range(3):
    a, b, e, W, V = gen_FCA(seed_id)
    n_neg = 0
    total = 0
    for _ in range(10000):
        M = np.random.uniform(0.05, 0.95, 5)
        A = a + W @ M
        D = A + b + V @ M + e
        for k in range(5):
            for j in range(5):
                if k == j:
                    continue
                grad = W[k,j]*D[k] - A[k]*(W[k,j]+V[k,j])
                total += 1
                if grad < 0:
                    n_neg += 1
    print(f"  seed {seed_id}: {n_neg}/{total} ({n_neg/total*100:.1f}%) 负梯度")

# ============================================================
# Q2: 直接下界不等式 N_k(M) ≥ m_k^(0) 
# ============================================================
print(f"\n{'='*70}")
print("Q2: N_k(M) ≥ m_k^(0) 对随机 M ∈ [0,1]^5 是否成立?")
print("=" * 70)

all_ok = True
for seed_id in range(200):
    a, b, e, W, V = gen_FCA(seed_id)
    D_max = a + b + e + np.sum(W + V, axis=1)
    m_low = a / D_max
    
    max_viol = 0
    for _ in range(10000):
        M = np.random.uniform(0, 1, 5)
        N_val = n_operator(M, a, b, e, W, V)
        for k in range(5):
            if N_val[k] < m_low[k] * (1 - 1e-12):
                max_viol = max(max_viol, m_low[k] - N_val[k])
    
    if max_viol > 1e-10:
        print(f"  ✗ seed {seed_id}: max违反量 = {max_viol:.2e}")
        all_ok = False

if all_ok:
    print(f"  ✓ 全部 200 种子 200000 K 随机测试: N_k(M) ≥ m_k^(0) 零违规")
else:
    print(f"  ✗ 有违反!")

# ============================================================
# Q3: 代数证明验证
# ============================================================
print(f"\n{'='*70}")
print("Q3: 直接下界不等式的代数结构验证")
print("=" * 70)

for seed_id in [0, 11, 149]:
    a, b, e, W, V = gen_FCA(seed_id)
    D_max = a + b + e + np.sum(W + V, axis=1)
    m_low = a / D_max
    
    Mstar = compute_fp(a, b, e, W, V)
    
    # 验证: 对10000个随机M
    max_err = 0
    for _ in range(10000):
        M = np.random.uniform(0, 1, 5)
        D = a + b + e + (W+V) @ M
        
        # 方法1: 直接 N_k(M)
        N_val = n_operator(M, a, b, e, W, V)
        
        # 方法2: 下界 bound_k = (a_k + Σ w M_j) / D_max ≥ a_k / D_max
        bound1 = (a + W @ M) / D_max
        
        # 方法3: m_low
        for k in range(5):
            # Check: N_k >= bound1_k >= m_k^(0)
            ok1 = N_val[k] >= bound1[k] * (1 - 1e-12)
            ok2 = bound1[k] >= m_low[k] * (1 - 1e-12)
            if not (ok1 and ok2):
                max_err = max(max_err, -min(N_val[k]-bound1[k], bound1[k]-m_low[k]))
    
    print(f"  seed {seed_id}: M*={[f'{x:.4f}' for x in Mstar]}, m_low={[f'{x:.4f}' for x in m_low]}")
    print(f"    不等式链: N_k ≥ (a+ΣwM)/D_max ≥ a/D_max = m^(0)  一致 {'✓' if max_err < 1e-12 else '✗'}")

# ============================================================
# Q4: 归纳法是否必要?
# ============================================================
print(f"\n{'='*70}")
print("Q4: 6.17B 引理 \"M(t) ≥ m^(0) ∀t≥1\" 的简化证明")
print("=" * 70)
print("""
原证明 (依赖 N 单调性):
  t=1: N(½) ≥ N(0) > m^(0)  ← 需要单调性 N(½) ≥ N(0)
  归纳: M(t+1) = N(M(t)) ≥ N(m^(0)) ≥ m^(0)  ← 需要单调性

简化证明 (纯代数, 不依赖单调性):
  对任意 M ∈ [0,1]^5:
    N_k(M) = (a_k + ΣwM_j) / D_k(M)
          ≥ (a_k + ΣwM_j) / D_max,k    (分母: D_k ≤ D_max,k)
          ≥ a_k / D_max,k               (分子: ΣwM_j ≥ 0)
          = m_k^(0)
  
  故 M(t+1)_k = N_k(M(t)) ≥ m_k^(0) ∀t≥0, k
  无需归纳, 无需单调性.
  
  注: t=0 时 M(0) = ½, 对 seed 149 有 m_4^(0)=0.5035 > M_4(0)=0.5,
  但这与结论 M(t)_k ≥ m_k^(0) 对 t≥1 不矛盾.
""")

# ============================================================
# Q5: 边角情况 — 极值端点处的 N_k(M)
# ============================================================
print("=" * 70)
print("Q5: 极值端点的保守性检查")
print("=" * 70)

for seed_id in [0, 11, 149]:
    a, b, e, W, V = gen_FCA(seed_id)
    D_max = a + b + e + np.sum(W + V, axis=1)
    m_low = a / D_max
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W+V) @ Mstar
    D_low_val = a + b + e + (W+V) @ m_low
    
    # 所有 32 个角点
    corners_ok = True
    for bits in range(32):
        M = np.array([(bits >> i) & 1 for i in range(5)], dtype=float)
        N_val = n_operator(M, a, b, e, W, V)
        if not np.all(N_val >= m_low * (1 - 1e-12)):
            corners_ok = False
    
    # 轨道验证
    orbit_ok = True
    M = np.full(5, 0.5)
    for t in range(200):
        M_next = n_operator(M, a, b, e, W, V)
        if not np.all(M_next >= m_low * (1 - 1e-12)):
            orbit_ok = False
            break
        if np.max(np.abs(M_next - M)) < 1e-12:
            break
        M = M_next
    
    print(f"  seed {seed_id}: M*={[f'{x:.3f}' for x in Mstar]}")
    print(f"    32角点全≥m^(0): {'✓' if corners_ok else '✗'}")
    print(f"    轨道M(t)≥m^(0) ∀t≥1: {'✓' if orbit_ok else '✗'}")
    print(f"    D_low/D* conservatism: {[f'{l/s:.2f}x' for l,s in zip(D_low_val, Dstar)]}")

print()
print("=" * 70)
print("审计结论")
print("=" * 70)
print("""
   主要发现:
   1. N 不是单调的 —— ∂N_k/∂M_j 可以为负
   2. 但 N_k(M) ≥ m_k^(0) 不需要单调性!
      → 纯代数证明: 分母 ≤ D_max, 分子 ≥ a_k
   3. 6.17B 的引理 "M(t)_k ≥ m_k^(0)" 可用直接下界替代单调性+归纳
   4. 原证明中的 "N 单调" 语句是多余的且理论基础不充分
   5. 简化后证明更干净、更严谨
   
   建议修正:
   - 删除 "N 单调" 推理
   - 用直接代数下界 M(t+1)_k = N_k(M(t)) ≥ m_k^(0) 替代
   - 注明该下界对 ∀M ∈ [0,1]^5 成立, 无需归纳
""")
