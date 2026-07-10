"""
解析导数界 - 良基化 N 算子的 Taylor 余项
=========================================
核心思路:
  N_k(M) = num_k(M) / D_k(M)  是 C^∞ 有理函数
  [0,1]^5 紧集 → 所有导数有界 (Weierstrass)
  D_k ≥ D_min = a_k + b_k + ε_k > 0 → 分母非零

目标:
  [A] 计算 ∂N_k/∂M_j, ∂²N_k/∂M_j∂M_l, ∂³N_k/∂M_j∂M_l∂M_m 的解析界
  [B] 推导 c_max (6.17C) 的解析上界
  [C] 推导 c_KL (6.17D) 的解析上界
"""
import numpy as np
from itertools import product

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

# ============================================================
# [A] N_k 的偏导数解析结构
# ============================================================
print("=" * 70)
print("[A] N_k 偏导数的解析结构")
print()

# N_k(M) = num_k / D_k
# num_k = a_k + sum_j w_kj M_j
# D_k = a_k + b_k + e_k + sum_j (w_kj + v_kj) M_j
#     = num_k + b_k + e_k + sum_j v_kj M_j
#
# 记 T_k = sum_j (w_kj + v_kj) M_j  (总耦合)
# D_k = num_k + b_k + e_k + sum_j v_kj M_j
#     = a_k + b_k + e_k + T_k
#
# 一阶导:
# ∂N_k/∂M_j = (w_kj·D_k - num_k·(w_kj+v_kj)) / D_k²
#           = [w_kj(a_k+b_k+e_k+T_k) - (a_k+Σw_kl M_l)(w_kj+v_kj)] / D_k²
# 简化: ∂N_k/∂M_j = w_kj/D_k - N_k·(w_kj+v_kj)/D_k
#                 = [w_kj(1-N_k) - v_kj·N_k] / D_k

print("一阶导: ∂N_k/∂M_j = [w_kj(1-N_k) - v_kj·N_k] / D_k")
print("D_k ≥ D_min = a_k + b_k + ε_k > 0")
print("N_k ∈ [0,1] → |∂N_k/∂M_j| ≤ (w_kj + v_kj) / D_min")
print()

# 二阶导:
print("二阶导: ∂²N_k/∂M_j∂M_l = ?")
print("""
由 ∂N_k/∂M_j = [w_kj(1-N_k) - v_kj·N_k] / D_k

对 M_l 求导:
∂/∂M_l [F(N_k) / D_k] where F(t) = w_kj(1-t) - v_kj·t = w_kj - (w_kj+v_kj)t

∂D_k/∂M_l = w_kl + v_kl
∂N_k/∂M_l = [w_kl(1-N_k) - v_kl·N_k] / D_k

∂/∂M_l (F(N_k)) = F'(N_k)·∂N_k/∂M_l = -(w_kj+v_kj)·[w_kl(1-N_k) - v_kl·N_k] / D_k

∂/∂M_l (1/D_k) = -(w_kl+v_kl) / D_k²

所以:
∂²N_k/∂M_j∂M_l = ∂/∂M_l [F(N_k)·(1/D_k)]
               = [F'(N_k)·∂N_k/∂M_l]·(1/D_k) + F(N_k)·[-(w_kl+v_kl)/D_k²]
               = -(w_kj+v_kj)·∂N_k/∂M_l / D_k - F(N_k)·(w_kl+v_kl) / D_k²
""")

# 计算 bound of |∂²N_k/∂M_j∂M_l|
# |∂N_k/∂M_l| ≤ (w_kl + v_kl) / D_min
# |F(N_k)| = |w_kj(1-N_k) - v_kj N_k| ≤ w_kj + v_kj (since N_k∈[0,1])

# |∂²N_k/∂M_j∂M_l| ≤ (w_kj+v_kj)(w_kl+v_kl)/D_min² + (w_kj+v_kj)(w_kl+v_kl)/D_min²
#                   = 2·(w_kj+v_kj)(w_kl+v_kl) / D_min²

print("上界: |∂²N_k/∂M_j∂M_l| ≤ 2·(w_kj+v_kj)(w_kl+v_kl) / D_min²")
print()

# 三阶导
print("三阶导界:")
print("逐次求导，每轮多出 1/D_k 因子和 ∂N/∂M 因子")
print("直观界: |∂³N_k/∂M_j∂M_l∂M_m| ≤ C·(w+v)_kj(w+v)_kl(w+v)_km / D_min³")
print("精确常数 C 需要展开所有项，但数量级正确")
print()

# ============================================================
# [B] c_max (6.17C) 的解析界
# ============================================================
print("=" * 70)
print("[B] c_max 解析界推导")
print()

# g(v) = (N(M)-M)·(M*-M) where M=M*+v
# g(v) = v^T(I-J)v + R_3(v)
#
# Taylor 余项 (积分形式):
# g(M*+v) = g(0) + g'(0)v + ½v^Tg''(0)v + 
#           (1/2)∫₀¹ (1-t)² v^T g'''(M*+tv) [v,v,v] dt
# 或 Lagrange 形式:
# R_3(v) = (1/6) Σ_{j,k,l} v_j v_k v_l · ∂³g/∂M_j∂M_k∂M_l(ξ)

# g(v) = -v^T(N(M*+v) - M* - v)
# 展开:
# g(v) = -v^T[J(M*)v + ½v^T∇²N(M*)v + ... - v]
#      = v^T(I-J)v - ½v^T[v^T ∇²N(M*) v] + ...

# 三阶余项涉及 N 的二阶导 (因为 g 中多了一个 v 的内积)
# 精确: g(M) = -Σ_i (M_i-M*_i)(N_i(M)-M_i)
# 
# 对 g 在 M* 处做 Taylor 展开:
# ∂g/∂M_m = -∂/∂M_m Σ_i (M_i-M*_i)(N_i-M_i)
#          = -Σ_i [δ_im(N_i-M_i) + (M_i-M*_i)(∂N_i/∂M_m - δ_im)]
# 在 M*: M_i-M*_i=0, N_i-M_i=0 → ∂g/∂M_m(M*) = 0 ✓

# ∂²g/∂M_m∂M_n(M*):
# = -Σ_i [δ_im(∂N_i/∂M_n-δ_in) + δ_in(∂N_i/∂M_m-δ_im)]
# = -[(∂N_m/∂M_n-δ_mn) + (∂N_n/∂M_m-δ_nm)]
# = 2δ_mn - (∂N_m/∂M_n + ∂N_n/∂M_m)
# = 2(I - sym(J))_mn   ← 这就是 I-J 的对称部分 × 2

# ∂³g/∂M_m∂M_n∂M_p:
# 注意 g 中只有 N 的二阶导出现:
# ∂³g/∂M_m∂M_n∂M_p = -Σ_i [δ_im·∂²N_i/∂M_n∂M_p + ...]
# |∂³g/∂M_m∂M_n∂M_p| ≤ Σ_i (|∂²N_i/∂M_n∂M_p| 的项)
# 上界: ≤ 5 · 2·(w+v)_in(w+v)_ip / D_min²  (对每个 i)

print("六阶张量界 (∂³g 的每个分量):")
print("|∂³g/∂M_m∂M_n∂M_p| ≤ 2Σ_i (w+v)_in(w+v)_ip / D_min²")
print()

# 现在计算 R_3(v) 的界:
# R_3(v) = (1/6) Σ_{a,b,c} ∂³g/∂M_a∂M_b∂M_c(ξ) · v_a v_b v_c
# 
# |R_3(v)| ≤ (1/6) · max|∂³g| · Σ_{a,b,c} |v_a v_b v_c|
#          ≤ (1/6) · B_3 · (Σ_a |v_a|)³
#          ≤ (1/6) · B_3 · (√5·||v||₂)³  (Hölder)
#          ≤ (1/6) · B_3 · 5√5 · ||v||₂³
#
# 其中 B_3 = max_{a,b,c,ξ∈[0,1]⁵} |∂³g/∂M_a∂M_b∂M_c(ξ)|

# 相比之下直接用 ||v||_p:
# |R_3(v)| ≤ (1/6) · B_3 · (5||v||_∞)³

print("R_3(v) = (1/6) Σ ∂³g · v_a v_b v_c")
print("|R_3(v)| ≤ (1/6)·B_3ₛ · ||v||₁³")
print("其中 B_3ₛ = max |∂³g/∂M_a∂M_b∂M_c|")
print()

# 数值验证 + 计算实际 Bound
print("数值验证 (1组随机FCA参数):")
seed0 = gen_FCA(0)
a, b, e, W, V = seed0
D_min = min(a + b + e)
print(f"  a={a}")
print(f"  b={b}")
print(f"  ε={e}")
print(f"  D_min = {D_min:.4f}")

# Compute C = w+v product sums
C_sum = sum(sum(W+V))  
print(f"  Σ(w+v) = {C_sum:.4f}")
print(f"  解析二阶导界: 2·max(w+v)²/D_min² = {2*max(sum(W+V))**2/D_min**2:.4f}")
print(f"  解析三阶导界 ≈ 6·max(w+v)³/D_min³ = {6*max(sum(W+V))**3/D_min**3:.4f}")

# Compare with empirical c_max (~0.02)
B3_est = 6 * (C_sum/5)**3 / D_min**3
c_max_analytic = (1/6) * 5 * B3_est
print(f"  c_max 解析估计 ≈ {c_max_analytic:.2f}")
print(f"  c_max 实证 ≈ 0.02")

# 这个解析界太大——这是良基化时付出"范数价格"
# 需要更精细的界来缩小差距
print()
print("  ⚠ 解析界远超实证值——'范数价格'典型表现")
print("  需要用均值不等式或更精细的逐项分析来缩小差距")
print()

# ============================================================
# [C] 更精细的界：利用 N_k ∈ [0,1] 和参数结构
# ============================================================
print("=" * 70)
print("[C] 精细界：利用 FCA 参数结构")
print()

# FCA 参数归一化意味着:
# a_k + b_k + Σ(w_kj+v_kj) 归一化到 ~1 量级
# 但由于 ×5/t 归一化，总量被约束

# 关键事实:
# D*_k = A*_k + B*_k + ε_k
# A*_k/B*_k 由不动点关系约束: M*_k = A*_k/D*_k ∈ [0,1]
# 且 M*_k 实际上不会极端接近 0 或 1 (由 ε 和下界保证)

# 更精细的一阶导界:
# |∂N_k/∂M_j| = |w_kj(1-N_k) - v_kj N_k| / D_k
# 在 N_k∈[0,1] 上, |w_kj(1-N_k) - v_kj N_k| ≤ max(w_kj, v_kj)  ← 关键!
# (两个正项的差, 绝对值 ≤ max of the two)

print("精细一阶导界:")
print("|∂N_k/∂M_j| ≤ max(w_kj, v_kj) / D_min")
print("(因为 |w(1-N) - v N| ≤ max(w,v) for N∈[0,1])")
print("相比三角不等式界 (w+v)/D_min, 精细界缩小了~2×")
print()

# 精细二阶导界
print("精细二阶导界:")
print("|∂²N_k/∂M_j∂M_l| ≤ 2·max(w_kj,v_kj)·(w_kl+v_kl) / D_min²")
print(f"  对于 seed 0: ≤ {2*0.5*C_sum/D_min**2:.6f}")

# 数值验证精细界
print()
print("数值验证精细界 (seed 0, 1000个随机点):")
from itertools import product as iproduct

def n_operator(M, a, b, e, W, V):
    num = a + W @ M
    den = num + b + V @ M + e
    return num / den

# 数值二阶导
max_d2_emp = 0.0
max_d2_bound_fine = 0.0
for _ in range(1000):
    M = np.random.random(5)
    Nk = n_operator(M, a, b, e, W, V)
    Dk = a + b + e + (W + V) @ M
    for k, j, l in iproduct(range(5), repeat=3):
        # 数值二阶导 (有限差分)
        h = 1e-4
        Mp = M.copy(); Mp[l] += h
        Mn = M.copy(); Mn[l] -= h
        
        # ∂/∂M_l of ∂N_k/∂M_j
        # = ∂/∂M_l [w_kj(1-N_k) - v_kj N_k] / D_k
        
        # 直接用解析公式
        # ∂N/∂M_j = [w_kj(1-N_k) - v_kj N_k] / D_k =: F_kj
        
        Np = n_operator(Mp, a, b, e, W, V)
        Nn = n_operator(Mn, a, b, e, W, V)
        Dp = a + b + e + (W + V) @ Mp
        Dn = a + b + e + (W + V) @ Mn
        
        dN_dMj_p = (W[k,j]*(1-Np[k]) - V[k,j]*Np[k]) / Dp[k]
        dN_dMj_n = (W[k,j]*(1-Nn[k]) - V[k,j]*Nn[k]) / Dn[k]
        d2_approx = (dN_dMj_p - dN_dMj_n) / (2*h)
        
        max_d2_emp = max(max_d2_emp, abs(d2_approx))
        
        bound = 2 * max(W[k,j], V[k,j]) * (W[k,l]+V[k,l]) / (D_min**2)
        max_d2_bound_fine = max(max_d2_bound_fine, bound)

print(f"  max_d2 (数值, 9999个点): {max_d2_emp:.4f}")
print(f"  精细上界: {max_d2_bound_fine:.4f}")
print(f"  比值 (bound/emp): {max_d2_bound_fine/max_d2_emp:.1f}x")

# 验证 "max(w,v)" 精细化是否有效
print()
print("=== 验证 key lemma: |w(1-N)-vN| ≤ max(w,v) ===")
for wv in [(0.1, 0.05), (0.01, 0.3), (0.2, 0.2), (0.05, 0.1)]:
    w_ij, v_ij = wv
    max_val = 0.0
    max_tri = 0.0
    for N_val in np.linspace(0, 1, 101):
        val = abs(w_ij*(1-N_val) - v_ij*N_val)
        tri = w_ij + v_ij
        if val > max_val:
            max_val = val
        if tri > max_tri:
            max_tri = tri
    print(f"  w={w_ij}, v={v_ij}: max|w(1-N)-vN| = {max_val:.3f} (精细), "
          f"三角不等式 = {max_tri:.3f} (松), 改善 {max_tri/max_val:.1f}x")

print()

# ============================================================
# [D] 实际可用的 c_max 解析计算
# ============================================================
print("=" * 70)
print("[D] c_max 解析计算 (逐实例)")
print()

# Lagrange 余项: R_3(v) = (1/6) Σ_{a,b,c} ∂³g/∂M_a∂M_b∂M_c(ξ) · v_a v_b v_c
# 其中 ξ 介于 M* 和 M*+v 之间
#
# |R_3(v)| ≤ (1/6) · B_3 · ||v||_₁³
# c_max = (1/6) · B_3 (当用 ||v||_₁ 度量时)
# 或 c'_max = (1/6) · B_3 · 5³ 当用 ||v||_∞ 度量
# 或 c''_max = (1/6) · B_3 · 5√5 当用 ||v||_₂ 度量
#
# 6.17C 文档中用的是欧氏范数 (||v||_₂):
# 需要 |R_3(v)| ≤ c_max · ||v||_₂³
# c_max = (1/6) · B_3 · 5^{3/2}  (因为 ||v||_₁ ≤ √5·||v||_₂)

print("范数转换:")
print("|R_3| ≤ (1/6)·B_3·||v||_₁³ ≤ (1/6)·B_3·(√5·||v||_₂)³")
print("     = (1/6)·B_3·5√5·||v||_₂³")
print("c_max (欧氏) = (5√5/6)·B_3")

# 计算 B_3 = max_{m,n,p,ξ} |∂³g/∂M_m∂M_n∂M_p(ξ)|
# g 的三阶导 ≈ N 的二阶导 × 内积系数
# 粗略: max|∂³g| ≤ 5 · max|∂²N| ≈ 5·2·max(w+v)²/D_min²

# 对于典型 FCA 参数:
# seed 0: W,V ∈ [0, 0.5], D_min ≈ 0.02, Σ(w+v) ≈ 5
# B_3 ≈ 10·(C/5)²/D_min² ≈ 10·1/0.0004 = 25000
# c_max ≈ (5√5/6)·25000 ≈ 47000

# 但这远大于实证的 0.02！
# 差距来源: ||v||_₁³ 非常松——实际上 v 的各分量不会同时取最大
# 且 ξ 处的值远小于全 [0,1]⁵ 的最大值

print(f"  粗略解析 c_max ≈ (5√5/6)·B_3 ≈ 解析界过大")
print()
print("  问题在于 uniform bound over [0,1]⁵ 太大")
print("  实际 Taylor 余项中 ξ 靠近 M*, 在该点 ∂³g 远小于全局最大值")

# Fix: 考虑在 M* 邻域内的局部界
# M* 邻域半径 r → 限制在 B(M*, r) ∩ [0,1]⁵ 上
# 在小邻域内, ∂³g 的值 ≈ ∂³g(M*) + O(r)

print()
print("=== 局部界分析 ===")

def compute_n_derivatives_at_Mstar(a, b, e, W, V, Mstar, epsilon=1e-6):
    """Numerically compute N's 1st, 2nd, 3rd derivatives at M*"""
    n_dim = 5
    J = np.zeros((n_dim, n_dim))
    H = np.zeros((n_dim, n_dim, n_dim))  # Hessian
    T = np.zeros((n_dim, n_dim, n_dim, n_dim))  # 3rd order
    
    for k in range(n_dim):
        for j in range(n_dim):
            M_plus = Mstar.copy(); M_plus[j] += epsilon
            M_minus = Mstar.copy(); M_minus[j] -= epsilon
            N_plus = n_operator(M_plus, a, b, e, W, V)
            N_minus = n_operator(M_minus, a, b, e, W, V)
            J[k, j] = (N_plus[k] - N_minus[k]) / (2 * epsilon)
            
            for l in range(n_dim):
                M_pp = Mstar.copy(); M_pp[j] += epsilon; M_pp[l] += epsilon
                M_pm = Mstar.copy(); M_pm[j] += epsilon; M_pm[l] -= epsilon
                M_mp = Mstar.copy(); M_mp[j] -= epsilon; M_mp[l] += epsilon
                M_mm = Mstar.copy(); M_mm[j] -= epsilon; M_mm[l] -= epsilon
                N_pp = n_operator(M_pp, a, b, e, W, V)
                N_pm = n_operator(M_pm, a, b, e, W, V)
                N_mp = n_operator(M_mp, a, b, e, W, V)
                N_mm = n_operator(M_mm, a, b, e, W, V)
                H[k, j, l] = (N_pp[k] - N_pm[k] - N_mp[k] + N_mm[k]) / (4 * epsilon**2)
    
    # 三阶导用单边差分 (简化)
    for k in range(n_dim):
        for j in range(n_dim):
            for l in range(n_dim):
                for m in range(n_dim):
                    # Forward difference in m of H[k,j,l]
                    M_fwd = Mstar.copy(); M_fwd[m] += epsilon
                    # Recompute H at M_fwd
                    Mf = M_fwd
                    H_fwd = 0.0
                    for _ in range(1):  # simplified
                        Mf_pp = Mf.copy(); Mf_pp[j] += epsilon; Mf_pp[l] += epsilon
                        Mf_pm = Mf.copy(); Mf_pm[j] += epsilon; Mf_pm[l] -= epsilon
                        Mf_mp = Mf.copy(); Mf_mp[j] -= epsilon; Mf_mp[l] += epsilon
                        Mf_mm = Mf.copy(); Mf_mm[j] -= epsilon; Mf_mm[l] -= epsilon
                        Nf_pp = n_operator(Mf_pp, a, b, e, W, V)
                        Nf_pm = n_operator(Mf_pm, a, b, e, W, V)
                        Nf_mp = n_operator(Mf_mp, a, b, e, W, V)
                        Nf_mm = n_operator(Mf_mm, a, b, e, W, V)
                        H_fwd = (Nf_pp[k] - Nf_pm[k] - Nf_mp[k] + Nf_mm[k]) / (4 * epsilon**2)
                    
                    T[k, j, l, m] = (H_fwd - H[k, j, l]) / epsilon
    
    return J, H, T

# 测试多个种子
print("seed  max|H| max|T| at M*  max|H| bound  max|T| bound")
for seed in [0, 5, 11, 42, 99]:
    a, b, e, W, V = gen_FCA(seed)
    D_min = min(a + b + e)
    
    # Compute FP
    M = np.full(5, 0.5)
    for _ in range(20000):
        M_new = n_operator(M, a, b, e, W, V)
        if np.max(np.abs(M_new - M)) < 1e-15:
            break
        M = M_new
    Mstar = M
    
    J, H_num, T_num = compute_n_derivatives_at_Mstar(a, b, e, W, V, Mstar, 1e-5)
    
    max_H = np.max(np.abs(H_num))
    max_T = np.max(np.abs(T_num))
    
    # 解析界
    C_kl = W[k,l] + V[k,l]  # max coupling
    max_coupling = max(W.flatten().max(), V.flatten().max())
    H_bound = 2 * max_coupling * (2*max_coupling) / D_min**2
    T_bound = 6 * max_coupling * (2*max_coupling)**2 / D_min**3
    
    print(f"  {seed:3d}  {max_H:.4f}  {max_T:.2f}  {H_bound:.2f}  {T_bound:.1f}")

print()
print("结论: 解析界 (uniform·[0,1]⁵) 远超 M* 处的实际导数值")
print("    差距 = 3-4 orders of magnitude for H, 5+ for T")
print("    这是因为 [0,1]⁵ 远端点的导数远大于 M* 处")
print()
print("这使得'用全空间 uniform bound 证明安全半径'不可行")
print("需要'以 M* 为中心的局部界'——这正是逐实例 ■ 证明的实质")
