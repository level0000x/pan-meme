"""
路径3: 凹凸性引理 - ∂²N_k/∂M_j² 的符号分析
==========================================
关键推导:
  ∂N_k/∂M_j = F(N_k) / D_k, F(t) = w_kj(1-t) - v_kj·t
  ∂²N_k/∂M_j² = -2(w_kj+v_kj)·F(N_k) / D_k²

  sign(∂²N_k/∂M_j²) = -sign(F(N_k))
                     = -sign(w_kj(1-N_k) - v_kj·N_k)

  F(N_k) = 0 ⇔ N_k = w_kj/(w_kj+v_kj) ≡ θ_kj

  问题: 当 M_j 遍历 [0,1] (其他分量固定) 时, N_k 是否总是
        在 θ_kj 的同一侧? 即: 要么 N_k ≤ θ_kj 对所有 M_j,
        要么 N_k ≥ θ_kj 对所有 M_j.

  如果能证明这一点, 则 ∂²N_k/∂M_j² 的符号在整个 [0,1] 切片上不变!
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

# ============================================================
# 理论基础
# ============================================================
print("=" * 70)
print("理论: ∂²N_k/∂M_j² = -2(w_kj+v_kj)·[w_kj(1-N_k)-v_kj·N_k] / D_k²")
print()
print("F(t) = w_kj(1-t) - v_kj·t = w_kj - (w_kj+v_kj)t")
print("F(t) > 0 ⇔ t < w_kj/(w_kj+v_kj)")
print("F(t) < 0 ⇔ t > w_kj/(w_kj+v_kj)")
print()
print("∂²N_k/∂M_j² > 0 ⇔ F(N_k) < 0 ⇔ N_k > w_kj/(w_kj+v_kj) (凸)")
print("∂²N_k/∂M_j² < 0 ⇔ F(N_k) > 0 ⇔ N_k < w_kj/(w_kj+v_kj) (凹)")
print()
print("N_k = (a_k + Σ w_kl M_l) / (a_k + Σ w_kl M_l + b_k + Σ v_kl M_l + ε_k)")
print("   = A_k / (A_k + B_k + ε_k)")
print()
print("N_k 作为 M_j 的函数是单调的")
print("  dN_k/dM_j = F(N_k)/D_k 符号取决于 N_k; 但 N_k 本身依赖 M_j")
print()
print("关键: N_k 在 M_j ∈ [0,1] 上的值域能否跨越 θ_kj?")
print()

# ============================================================
# 数值扫描
# ============================================================
print("=" * 70)
print("数值扫描: 25条(k,j)曲线×200种子×1000个M_j值")

total_flips = 0
flip_examples = []

for seed in range(200):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    
    for k in range(5):
        for j in range(5):
            if j == k:
                continue
            theta = W[k,j] / (W[k,j] + V[k,j])  # critical N_k value
            
            # Pick 3 random fixed configurations for other components
            for _ in range(3):
                M_other = np.random.uniform(0.02, 0.98, 5)
                
                n_vals = []
                d2_signs = []
                
                for mj in np.linspace(0.01, 0.99, 200):
                    M = M_other.copy()
                    M[j] = mj
                    Nk = n_operator(M, a, b, e, W, V)[k]
                    F_val = W[k,j]*(1-Nk) - V[k,j]*Nk
                    d2_sign = -np.sign(F_val)
                    n_vals.append(Nk)
                    d2_signs.append(d2_sign)
                
                n_arr = np.array(n_vals)
                signs = np.array(d2_signs)
                
                # Check if signs change
                unique_signs = np.unique(signs[signs != 0])
                if len(unique_signs) >= 2:
                    total_flips += 1
                    if len(flip_examples) < 5:
                        flip_examples.append({
                            'seed': seed, 'k': k, 'j': j, 'theta': theta,
                            'Nk_range': [n_arr.min(), n_arr.max()],
                            'signs': unique_signs,
                            'W_kj': W[k,j], 'V_kj': V[k,j],
                            'Mstar': Mstar,
                        })

print(f"  总符号翻转次数: {total_flips}")
print()

if total_flips > 0:
    print("  ⚠ 符号翻转确实发生!")
    print()
    print("  翻转案例:")
    for ex in flip_examples[:3]:
        print(f"    seed={ex['seed']}, k={ex['k']}, j={ex['j']}")
        print(f"    θ = {ex['theta']:.4f}, N_k range = [{ex['Nk_range'][0]:.4f}, {ex['Nk_range'][1]:.4f}]")
        print(f"    W={ex['W_kj']:.4f}, V={ex['V_kj']:.4f}")
        print(f"    signs observed: {ex['signs']}")
        print()
else:
    print("  零翻转! 文档声称正确 ✓")
    print()
    print("  需要证明: N_k(M_j) 在 M_j∈[0,1] 上不会跨越 θ_kj")

print()

# ============================================================
# 分析为何 (不) 翻转
# ============================================================
print("=" * 70)
print("解析: N_k 何时跨越 θ_kj?")
print()

# N_k(M_j) 是 M_j 的单调函数吗?
# ∂N_k/∂M_j = [w_kj(1-N_k) - v_kj·N_k]/D_k
# 
# 在 N_k = θ_kj = w_kj/(w_kj+v_kj) 处, F(N_k)=0, ∂N_k/∂M_j=0
# 所以 N_k(M_j) 在 θ_kj 处有停顿点 (导数变号)
#
# ∵ ∂N_k/∂M_j 的符号 = sign(F(N_k)) = sign(θ_kj - N_k)
# 所以 N_k(M_j) 总是向 θ_kj 单调趋近:
#   - 如果 N_k < θ_kj, 则 ∂N_k/∂M_j > 0, N_k 随 M_j 增加
#   - 如果 N_k > θ_kj, 则 ∂N_k/∂M_j < 0, N_k 随 M_j 减少
# 
# 这意味着 θ_kj 是 N_k 作为 M_j 函数的一个 "吸引子"
# N_k 趋近于 θ_kj 但可能跨越它!

# 在 M_j 的端点:
# M_j = 0: N_k = A_k(-j) / D_k(-j) where A_k(-j) = a_k + Σ_{l≠j} w_kl M_l
# M_j = 1: N_k = (A_k(-j) + w_kj) / (D_k(-j) + w_kj + v_kj)

# N_k 跨越 θ_kj 的条件:
# 端点值分别在 θ_kj 两侧

print("N_k(M_j) 的单调方向始终指向 θ_kj (因为 dN/dM_j ∝ (θ-N))")
print("所以只要 N_k 在某个 M_j 处不等于 θ_kj, N_k 就单调趋近 θ_kj")
print()
print("问题转为: N_k(0) 和 N_k(1) 是否在 θ_kj 的同侧?")
print()

# 详尽测试
print("详尽测试 (200种子 × 25 (k,j)对 × 100 个固定配置):")
cross_count = 0
no_cross_count = 0

for seed in range(200):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    
    for k in range(5):
        for j in range(5):
            if j == k:
                continue
            theta = W[k,j] / (W[k,j] + V[k,j])
            
            for _ in range(100):
                M_other = np.random.uniform(0.02, 0.98, 5)
                
                # Endpoints
                M0 = M_other.copy(); M0[j] = 0.01
                M1 = M_other.copy(); M1[j] = 0.99
                N0 = n_operator(M0, a, b, e, W, V)[k]
                N1 = n_operator(M1, a, b, e, W, V)[k]
                
                if (N0 - theta) * (N1 - theta) < 0:
                    cross_count += 1
                else:
                    no_cross_count += 1

total = cross_count + no_cross_count
print(f"  跨越: {cross_count}/{total} ({100*cross_count/total:.4f}%)")
print(f"  不跨越: {no_cross_count}/{total} ({100*no_cross_count/total:.4f}%)")

# ============================================================
# 如果确实跨越了, 那文档的 "无符号翻转" 指的是什么?
# ============================================================
print()
print("=" * 70)
print("辨析: 文档可能测试的是 ∂²N_k/∂M_j²(M*) (在不动点处)")
print("而不是沿整条曲线")
print()

# 不动点处:
flips_at_fp = 0
for seed in range(200):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    
    for k in range(5):
        for j in range(5):
            if j == k:
                continue
            N_k_fp = Mstar[k]  # at FP, N_k = M*_k
            theta = W[k,j] / (W[k,j] + V[k,j])
            F_at_fp = W[k,j]*(1-N_k_fp) - V[k,j]*N_k_fp
            
            # 在 M* 邻域内沿 M_j 方向
            h = 1e-4
            M_plus = Mstar.copy(); M_plus[j] += h
            M_minus = Mstar.copy(); M_minus[j] -= h
            N_plus = n_operator(M_plus, a, b, e, W, V)[k]
            N_minus = n_operator(M_minus, a, b, e, W, V)[k]
            # Second derivative numerical
            d2 = (N_plus - 2*N_k_fp + N_minus) / (h*h)
            
            # 在FP处, N_k = M*_k, 所以 sign(d2) = -sign(F(M*_k))
            # 这个符号在FP处是固定的 (对给定参数)
            # 但沿整条曲线可能变化!

print(f"  文档测试的是沿 M_j 切片 (holding other M's fixed)")
print(f"  但 '25条偏导曲线' 可能指不同的东西")
print()

# 再确认: N_k(M_j) 是否确实可以跨越 θ_kj
# Analytic proof:
# N_k(0) = A(-j) / D(-j)
# N_k(1) = (A(-j)+w) / (D(-j)+w+v)
#
# N_k(0) < θ ⇔ A/D < w/(w+v) ⇔ Av < Dw-Aw ⇔ A(w+v) < D·w
# N_k(1) > θ ⇔ (A+w)/(D+w+v) > w/(w+v) ⇔ (A+w)(w+v) > w(D+w+v)
#
# The question is: can both hold simultaneously for some A(-j), D(-j)?
#
# This is an algebraic question about the rational birth-death operator.

print("=" * 70)
print("解析证明: N_k(M_j) 跨越 θ_kj 的条件")
print()

# Let A = A_k(-j) ≥ a_k, D = D_k(-j) ≥ a_k + b_k + ε_k
# Condition for crossing: N_k(0) ≤ θ ≤ N_k(1) or N_k(1) ≤ θ ≤ N_k(0)
#
# Case 1: N_k(0) ≤ θ ≤ N_k(1)
#   A/D ≤ w/(w+v) and (A+w)/(D+w+v) ≥ w/(w+v)
#
# A/D ≤ w/(w+v) ⇒ A(w+v) ≤ D·w
# (A+w)/(D+w+v) ≥ w/(w+v) ⇒ (A+w)(w+v) ≥ w(D+w+v)
#                           ⇒ A(w+v) + w(w+v) ≥ wD + w² + wv
#                           ⇒ A(w+v) ≥ wD + w² + wv - w² - wv = wD
#
# So: A(w+v) ≤ wD and A(w+v) ≥ wD
# Both ⇒ A(w+v) = wD ⇒ A/D = w/(w+v) = θ
# But this means N_k(0) = θ! (not strict crossing)
#
# Let me check with actual inequalities... 

# Actually, for monotonic function to cross θ:
# N_k is monotonic in M_j (but direction may change!)
# Since dN_k/dM_j changes sign at θ:
# - If N_k < θ: N_k ↑ as M_j ↑
# - If N_k > θ: N_k ↓ as M_j ↑
# This means N_k can only CROSS θ if the monotonic direction doesn't prevent it.
#
# If N_k(0) < θ: N_k increases, possibly crossing θ
# If N_k(1) > θ: N_k decreases as M_j ↑ from some point... but wait
#
# The monotonic direction CHANGES at θ! So:
# - Before crossing: N_k < θ, dN/dM_j > 0, N_k ↑
# - At θ: dN/dM_j = 0
# - After crossing: N_k > θ, dN/dM_j < 0, N_k ↓
#
# This means θ is actually a POINT OF MAXIMUM of N_k as a function of M_j!
# N_k approaches θ from below, reaches it, then CANNOT cross it because
# dN/dM_j becomes negative after crossing.
#
# Wait, that's wrong. Let me think again...
# 
# dN_k/dM_j = [w_kj(1-N_k)-v_kj·N_k] / D_k
# θ = w/(w+v)
#
# If N_k < θ: dN/dM_j > 0 (N_k increases with M_j)
# If N_k = θ: dN/dM_j = 0
# If N_k > θ: dN/dM_j < 0 (N_k decreases with M_j)
#
# This is a "pulling toward θ" behavior. If N_k starts below θ and increases,
# it approaches θ. As it approaches, the rate of increase slows down.
# But can it cross θ?
#
# For N_k to cross θ from below, it needs to go PAST θ. But just after crossing,
# dN/dM_j would be negative, pulling it BACK toward θ. 
#
# So the question is: is θ an asymptote or can N_k cross through it?
#
# N_k(M_j) is continuous and differentiable. At θ, dN/dM_j = 0. 
# This is a critical point. The second derivative:
#
# d²N_k/dM_j² at θ: F(N_k) = 0, so 
# d²N_k/dM_j² = -2(w+v)·F(N_k)/D² = 0 at θ
#
# We need the THIRD derivative to determine if θ is a stationary point, inflection, etc.
#
# Actually, I realize the issue. Let me think about this differently.
# 
# N_k is NOT an independent variable - it's a function of ALL M components.
# When we vary M_j, N_k changes. The equation for equilibrium:
#
# dN_k/dM_j = 0 ⇔ N_k = θ
#
# But N_k also depends on M_j. So the condition is actually:
# (A(-j) + w_kj·M_j) / (D(-j) + (w_kj+v_kj)·M_j) = w_kj/(w_kj+v_kj)
#
# Cross-multiplying:
# (A(-j) + w·M_j)(w+v) = w·(D(-j) + (w+v)·M_j)
# A(w+v) + w(w+v)M_j = wD + w(w+v)M_j
# A(w+v) = wD
#
# This is INDEPENDENT of M_j! So either A(w+v) = wD for all M_j (degenerate case),
# or A(w+v) ≠ wD and N_k NEVER equals θ for any M_j ∈ ℝ!
#
# WOW! This is a remarkable result.

print("重要推导:")
print("  N_k = (A(-j) + w·M_j) / (D(-j) + (w+v)·M_j)")
print("  dN_k/dM_j = 0 ⇔ N_k = θ = w/(w+v)")
print("")
print("  代入 N_k = θ:")
print("  (A + w·M_j)/(D + (w+v)·M_j) = w/(w+v)")
print("  (A + w·M_j)(w+v) = w(D + (w+v)M_j)")
print("  A(w+v) + w(w+v)M_j = wD + w(w+v)M_j")
print("  A(w+v) = wD")
print("")
print("  M_j 消掉了! 所以 N_k 永不等于 θ (除非 A(w+v)=wD 退化情况)")
print()

# Verify numerically
print("数值验证 (100种子 × 25 (k,j)对 × 1000个M_j值):")
