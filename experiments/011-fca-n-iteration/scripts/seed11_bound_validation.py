"""
验证: 行式 α_bound = max_k (r_k · D*_k/D_low,k) 的有效性
=====================================================
理论问题: 界 max_k r_k γ_k 是否 ≥ ||N(M)-M*||₁ / ||Δ||₁ 对所有 Δ?
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
print("测试 1: 对 seed 11, 随机 Δ 验证行式界")
print("=" * 70)

seed_id = 11
a, b, e, W, V = gen_FCA(seed_id)
Mstar = compute_fp(a, b, e, W, V)
Dstar = a + b + e + (W + V) @ Mstar
m_low, m_high, D_low = iterative_Mstar_bounds(a, b, e, W, V)

# True operator norm via perturbation
J = np.zeros((5,5))
for k in range(5):
    for j in range(5):
        if k != j:
            J[k,j] = (W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k]

r_k = np.sum(np.abs(J), axis=1)  # row sums
gamma_k = Dstar / D_low

alpha_row = max(r_k * gamma_k)
alpha_col = max([sum(abs(J[:,j]) * gamma_k) for j in range(5)])

print(f"Seed {seed_id}:")
print(f"  D*/D_low = {gamma_k}")
print(f"  r_k (row sums of |J|) = {r_k}")
print(f"  r_k * γ_k = {r_k * gamma_k}")
print(f"  α_row = {alpha_row:.4f}")
print(f"  α_col (weighted) = {alpha_col:.4f}")

# DIRECT TEST: compute ||N(M)-M*||₁ / ||Δ||₁ for random Δ
print(f"\n  直接测试 (10000 随机 Δ):")
worst_ratio = 0
worst_D = None
n_violations = 0
for _ in range(10000):
    Delta = np.random.uniform(-1, 1, 5)
    M = Mstar + Delta
    M = np.clip(M, 1e-6, 1 - 1e-6)
    
    N_val = n_operator(M, a, b, e, W, V)
    ratio = np.sum(np.abs(N_val - Mstar)) / max(np.sum(np.abs(Delta)), 1e-10)
    
    if ratio > alpha_row:
        n_violations += 1
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_D = Delta

print(f"  违反 α_row 的次数: {n_violations}/10000")
print(f"  最劣实际 ratio: {worst_ratio:.6f}")

if worst_D is not None:
    print(f"  最劣 Δ 方向: {worst_D}")
    # Check at that Δ: what are the D_k values?
    M_test = Mstar + worst_D
    M_test = np.clip(M_test, 1e-6, 1-1e-6)
    D_test = a + b + e + (W+V) @ M_test
    gamma_test = Dstar / D_test
    print(f"  D*/D at worst point = {gamma_test}")


# ============================================================
# 导出行式界的正确推导
# ============================================================
print(f"\n{'='*70}")
print("行式界的推导验证")
print("=" * 70)

"""
我们需要证明:
  ||N(M)-M*||₁ ≤ α_row · ||Δ||₁
其中 α_row ≤ max_k (r_k · D*_k/D_low,k)

推导:
  ||N(M)-M*||₁ = Σ_k (D*_k/D_k) |Σ_j J_kj Δ_j|
  
设 γ_k = D*_k/D_k ≤ D*_k/D_low,k (因为 D_k ≥ D_low,k ∈ [0,1])
设 x_k = |Σ_j J_kj Δ_j| ≥ 0
  
  = Σ_k γ_k x_k
  
不等式: Σ_k γ_k x_k ≤ (max_k γ_k) · Σ_k x_k    (因为 x_k ≥ 0)
  
  ≤ max_k γ_k · Σ_k Σ_j |J_kj| |Δ_j|
  = max_k γ_k · Σ_j (Σ_k |J_kj|) |Δ_j|

设 c_j = Σ_k |J_kj| (列和)
  = max_k γ_k · Σ_j c_j |Δ_j|
  ≤ max_k γ_k · max_j c_j · ||Δ||₁

对 seed 11: max_k γ_k ≈ 7.4, max_j c_j ≈ 0.36 → α ≈ 2.7 (过松!)

问题: max_k γ_k 是全局因子, 乘以所有项.
"""

# 重新推导 — 不做 max_k 分离
print("\n  --- 更紧的推导 (不做 max_k 分离) ---")
print()

"""
直接:
  ||N(M)-M*||₁ ≤ Σ_k γ_k Σ_j |J_kj| |Δ_j|
                = Σ_j (Σ_k γ_k |J_kj|) |Δ_j|
                ≤ max_j (Σ_k γ_k |J_kj|) · ||Δ||₁

这就是列式加权界。对 seed 11 我们要算一下。
"""

# 列式 γ-bound
col_gamma_bound = np.array([sum(gamma_k * abs(J[:,j])) for j in range(5)])
print(f"列式 γ-bound = {col_gamma_bound}")
print(f"max = {max(col_gamma_bound):.4f}")

"""
这个是正确有效的界。但这和旧的 α_old 有什么区别?
旧的: α_old = max_j Σ_k |J_kj| · D*_k/D_min,k
新的: α_col = max_j Σ_k |J_kj| · D*_k/D_low,k

区别在于分母: D_min → D_low
对 seed 11 row 1: D_min = 0.048, D_low = 0.346 (7.2x 改进!)
"""

# 验证
col_bound = max([sum(abs(J[:,j]) * gamma_k) for j in range(5)])
print(f"\n列式 D_low-bound = {col_bound:.4f}")

# 直接大规模验证这个列式界
print(f"\n  直接验证列式 D_low-bound ({col_bound:.4f}):")
n_violations_col = 0
worst_ratio_col = 0
for _ in range(10000):
    Delta = np.random.uniform(-1, 1, 5)
    M = Mstar + Delta
    M = np.clip(M, 1e-6, 1 - 1e-6)
    N_val = n_operator(M, a, b, e, W, V)
    ratio = np.sum(np.abs(N_val - Mstar)) / max(np.sum(np.abs(Delta)), 1e-10)
    if ratio > worst_ratio_col:
        worst_ratio_col = ratio
    if ratio > col_bound:
        n_violations_col += 1

print(f"  违反次数: {n_violations_col}/10000")
print(f"  最劣实际 ratio: {worst_ratio_col:.6f}")


# ============================================================
# 行式 α_row 的真正推导
# ============================================================
print(f"\n{'='*70}")
print("行式界的正确推导")
print("=" * 70)

"""
我们需要导出 max_k (r_k · γ_k) 形式的界。

一种方式: 使用 Hölder 不等式
  |Σ_j J_kj Δ_j| ≤ Σ_j |J_kj| · |Δ_j| ≤ r_k · ||Δ||_∞

则:
  ||N(M)-M*||₁ = Σ_k γ_k |Σ_j J_kj Δ_j|
                ≤ Σ_k γ_k · r_k · ||Δ||_∞
                = (Σ_k γ_k r_k) · ||Δ||_∞
                ≤ (Σ_k γ_k r_k) · ||Δ||₁

但这给出 α = Σ_k γ_k r_k (求和!), 而不是 max_k γ_k r_k.

对于 seed 11:
  Σ_k γ_k r_k = ?
"""

alpha_row_sum = np.sum(gamma_k * r_k)
print(f"  Σ_k γ_k r_k = {alpha_row_sum:.4f}")
print(f"  max_k γ_k r_k = {alpha_row:.4f}")

"""
Σ_k γ_k r_k ≈ 1.16 对于 seed 11 — 虽然 > 1 但接近 1!
(之前我以为会很大, 其实这已经在 1 附近了)

等等——这个界过估了多少? 让我验证.
"""

# 验证求和界
print(f"\n  直接验证求和界 (Σ_k γ_k r_k = {alpha_row_sum:.4f}):")
n_violations_sum = 0
worst_ratio_sum = 0
for _ in range(10000):
    Delta = np.random.uniform(-1, 1, 5)
    M = Mstar + Delta
    M = np.clip(M, 1e-6, 1 - 1e-6)
    N_val = n_operator(M, a, b, e, W, V)
    ratio = np.sum(np.abs(N_val - Mstar)) / max(np.sum(np.abs(Delta)), 1e-10)
    if ratio > worst_ratio_sum:
        worst_ratio_sum = ratio

print(f"  最劣实际 ratio: {worst_ratio_sum:.6f}")

# ============================================================
# 关键测试: 列式 D_low bound 对所有 200 种子
# ============================================================
print(f"\n{'='*70}")
print("列式 D_low-bound 批量测试 (200 种子)")
print("=" * 70)

total_ok = 0
worst_bounds = []
for seed_id in range(200):
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W + V) @ Mstar
    m_low, m_high, D_low = iterative_Mstar_bounds(a, b, e, W, V)
    
    J = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k != j:
                J[k,j] = (W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k]
    
    gamma_k = Dstar / D_low
    
    # 列式 D_low-bound
    col_bound = max([sum(abs(J[:,j]) * gamma_k) for j in range(5)])
    
    if col_bound < 1:
        total_ok += 1
    if col_bound >= 0.9:
        worst_bounds.append((seed_id, col_bound))

print(f"  闭合: {total_ok}/200 ({100*total_ok/200:.0f}%)")
if worst_bounds:
    print(f"  接近 1 的种子:")
    for s, b in sorted(worst_bounds, key=lambda x: -x[1])[:10]:
        print(f"    seed {s:3d}: {b:.4f}")

# ============================================================
# 最佳策略: 混合界
# ============================================================
print(f"\n{'='*70}")
print("混合界: 对 '硬行' 用列式, 其余用行式")
print("=" * 70)

# 思路: 对每个特定 Δ, 选择更紧的界
# 但这不可操作——我们需要统一对所有 Δ 的界

# 替代: 使用 min(max_k r_k, max_j c_j) 形式的混合
# 或者: 寻找更好的推导使得 max_k r_k γ_k 成为有效界

# 关键洞察: 也许可以用矩阵范数等价性
# ||J||₁,op / min_k D_low,k ≤ 界的某些形式...

print()

# ============================================================
# 根本性突破尝试
# ============================================================
print("=" * 70)
print("根本性重新推导")
print("=" * 70)

"""
恒等式 6.17A:
  N_k(M) - M*_k = (D*_k/D_k) · Σ_j J_kj · (M_j - M*_j)

设 Δ = M - M*
设 d_k = D*_k/D_k, 注意 0 < d_k ≤ D*_k/D_low,k

设 v_k = Σ_j J_kj Δ_j

则 ||N(M)-M*||₁ = Σ_k d_k |v_k|

注意: d_k 和 v_k 都是 Δ 的函数! (通过 D_k(M))
但 D_k(M) = D*_k + Σ_j (w_kj+v_kj) Δ_j

关键简化: 如果 Δ_j ≥ 0 ∀ j, 则 D_k(M) ≥ D*_k, 所以 d_k ≤ 1
           如果 Δ_j ≤ 0 ∀ j, 则 D_k(M) ≤ D*_k, 所以 d_k ≥ 1

但对于一般 Δ (有正有负), d_k 可能 >1 或 <1.

三角不等式在符号混合的 Δ 上导致巨大过估。

更好的策略: 将 Δ 分解为正负部分
  Δ = Δ⁺ - Δ⁻  (Δ⁺_j = max(Δ_j, 0), Δ⁻_j = max(-Δ_j, 0))

则 v_k = Σ_j J_kj Δ⁺_j - Σ_j J_kj Δ⁻_j

|v_k| = |Σ_j J_kj Δ⁺_j + Σ_j (-J_kj) Δ⁻_j|

对某一行 k, J_kj 和 -J_kj 可能有不同符号。
但所有项 Σ_j |J_kj| Δ⁺_j + Σ_j |J_kj| Δ⁻_j = Σ_j |J_kj| |Δ_j|

这就是三角不等式——我们在保留 J 符号的时候又丢了。

问题的根源: 对每一行 k, 我们不知道 Δ_j 的符号和 J_kj 的符号是否对齐。
对齐时, |Σ J Δ| ≈ Σ |J| |Δ| (等号成立当所有 J_kj Δ_j 同号)
不对齐时, |Σ J Δ| ≪ Σ |J| |Δ| (大量抵消)

这本质上是 符号模式 问题——又回到了 RD/CD 和符号涌现。
"""

print("--- 新思路: 正负分解 + Hölder ---")
print()
"""
将 Δ 分解:
  P = {j: Δ_j ≥ 0}, N = {j: Δ_j < 0}

v_k = Σ_{j∈P} J_kj Δ_j + Σ_{j∈N} J_kj Δ_j
    = Σ_{j∈P} J_kj Δ⁺_j + Σ_{j∈N} J_kj (-Δ⁻_j)
    = Σ_{j∈P} J_kj Δ⁺_j - Σ_{j∈N} J_kj Δ⁻_j

|v_k| ≤ |Σ_{j∈P} J_kj Δ⁺_j| + |Σ_{j∈N} J_kj Δ⁻_j|
     ≤ Σ_{j∈P} |J_kj| Δ⁺_j + Σ_{j∈N} |J_kj| Δ⁻_j
     = Σ_j |J_kj| |Δ_j|

又回到了三角不等式。

但这暗示: 如果我们能证明 J_kj 对 j∈P 同号 且对 j∈N 同号,
则:
  |Σ_{j∈P} J_kj Δ⁺_j| = Σ_{j∈P} |J_kj| Δ⁺_j  (如果所有 J_kj 在 P 上有相同符号)
  
在这种情况下, 三角不等式没有损失!

所以问题的核心是: J 行的符号模式.
对于行 k, 如果 J_kj 全为同号 (或零), 则三角不等式不丢信息。
如果 J_kj 符号混合, 则三角不等式丢掉的正是正负项抵消。

seed 11 的行 1: 正负项混合 → 三角不等式损失 7.9×
seed 11 的其他行: 可能更好?

快看 seed 11 的 J 各行符号:
"""
# Print seed 11 J matrix
J11 = np.zeros((5,5))
for k in range(5):
    for j in range(5):
        if k != j:
            J11[k,j] = (W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k]

print("J(M*) 矩阵 (seed 11):")
for k in range(5):
    row = []
    for j in range(5):
        if k == j:
            row.append("  ·  ")
        else:
            row.append(f"{J11[k,j]:+.4f}")
    print(f"  行{k}: " + " ".join(row))

print()
print("各行符号模式:")
for k in range(5):
    signs = []
    vals = []
    for j in range(5):
        if k == j: continue
        s = "+" if J11[k,j] > 1e-10 else "-"
        signs.append(f"{s}({j})")
        vals.append(J11[k,j])
    print(f"  行{k}: {' '.join(signs)}")
    pos_sum = sum(v for v in vals if v > 0)
    neg_sum = sum(v for v in vals if v < 0)
    print(f"        正和={pos_sum:.4f} 负和={neg_sum:.4f}  总和={pos_sum+neg_sum:.4f}")
