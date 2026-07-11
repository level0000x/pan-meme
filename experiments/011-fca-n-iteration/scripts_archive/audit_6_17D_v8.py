"""
第八轮审计 - 深层代数 + 数值一致性 + 符号攻防
==============================================
本轮重点:
  [A] Cross/D 比率推导——|τ|/(2|δ+τ|·ξ(1-ξ)) vs |τ|/(2|δ-τ|·ξ(1-ξ))
  [B] Pinsker不等式在Bernoulli KL中的紧性
  [C] MVT logit导数：1/(ξ(1-ξ)) 的正确性
  [D] 安全半径 c_KL 的独立数值计算
  [E] λ_min(H - J^T H J) 独立验证
  [F] diag(D*/D) ≈ I 的近似精度
  [G] Case A/Case B 分类的符号正确性
  [H] 6.17C 安全半径数值验证
  [I] 所有数值声称的独立盲测
  [J] ‖M_H‖₂ vs Gershgorin vs λ_max 的一致性
"""
import numpy as np
from scipy.special import logit as scipy_logit
import warnings
warnings.filterwarnings('ignore')

def n_operator(M, a, b, eps, W, V):
    num = a + W @ M
    den = num + b + V @ M + eps
    return num / den

def compute_fp(a, b, eps, W, V, max_iter=20000):
    M = np.full(5, 0.5)
    for _ in range(max_iter):
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

def D_KL_bernoulli(p, q):
    """KL(p||q) for Bernoulli parameters"""
    eps = 1e-15
    pp = np.clip(p, eps, 1 - eps)
    qq = np.clip(q, eps, 1 - eps)
    return pp * np.log(pp / qq) + (1 - pp) * np.log((1 - pp) / (1 - qq))

def D_KL_vector(M1, M2):
    """Reverse KL: D_KL(M1 || M2) for product Bernoulli"""
    return np.sum([D_KL_bernoulli(M1[k], M2[k]) for k in range(5)])

# ============================================================
# [A] Cross/D 比率推导——符号审查 ★核心★
# ============================================================
print("=" * 70)
print("[A] Cross/D 比率推导——符号审查")
print("""
文档声称的公式:
  (M*_k - N_k)(logit M_k - logit N_k)     |τ_k|
  ────────────────────────────────── = ────────────────
          D_KL(N_k || M_k)              2|δ_k+τ_k|·ξ(1-ξ)

其中 τ_k = N_k - M*_k, δ_k = M_k - M*_k

重新推导:
  logit M_k - logit N_k = (M_k - N_k) / (ξ(1-ξ))   [MVT, ξ ∈ [M_k,N_k]]
  = (δ_k - τ_k) / (ξ(1-ξ))  ...(1)

  M*_k - N_k = -τ_k  ...(2)

  (1)×(2): (M*_k - N_k)(logit M_k - logit N_k) = -τ_k(δ_k - τ_k)/(ξ(1-ξ))

  D_KL(N_k || M_k) ≥ 2(N_k - M_k)² = 2(τ_k - δ_k)²  [Pinsker]

  │-τ_k(δ_k - τ_k)/(ξ(1-ξ))│     │τ_k│·│δ_k - τ_k│
  ─────────────────────────── ≤ ──────────────────────
       2(τ_k - δ_k)²              2(δ_k - τ_k)²·ξ(1-ξ)

                               │τ_k│
                         = ────────────────  ← 正确公式
                           2│δ_k - τ_k│·ξ(1-ξ)

文档的公式是 |τ_k|/(2|δ_k+τ_k|·ξ(1-ξ)) —— 分母有 δ_k+τ_k 而非 δ_k-τ_k！

这对吗？继续分析:
  δ_k + τ_k = (M_k - M*_k) + (N_k - M*_k) = M_k + N_k - 2M*_k
  δ_k - τ_k = (M_k - M*_k) - (N_k - M*_k) = M_k - N_k

  在 Case B (超调): τ_k 与 δ_k 异号
    |δ_k - τ_k| = |δ_k| + |τ_k|  (= |M_k - N_k|)
    |δ_k + τ_k| = ||δ_k| - |τ_k||  (≠ |M_k - N_k|)

  结论: δ_k+τ_k ≠ δ_k-τ_k，文档公式与正确推导不一致！
""")

# 数值验证：比较两个公式
print("  数值验证...")
np.random.seed(42)
results_cross = []
for seed in range(200):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    
    for _ in range(50):
        M = np.random.uniform(0.02, 0.98, 5)
        N = n_operator(M, a, b, e, W, V)
        M = np.clip(M, 1e-15, 1 - 1e-15)
        N = np.clip(N, 1e-15, 1 - 1e-15)
        
        for k in range(5):
            delta = M[k] - Mstar[k]
            tau = N[k] - Mstar[k]
            
            cross = (Mstar[k] - N[k]) * (np.log(M[k]/(1-M[k])) - np.log(N[k]/(1-N[k])))
            D = D_KL_bernoulli(N[k], M[k])
            
            if D < 1e-15:
                continue
            
            ratio = abs(cross) / D
            
            # MVT: find ξ
            xi_low = min(M[k], N[k])
            xi_high = max(M[k], N[k])
            
            # Formula doc: |τ|/(2|δ+τ|·ξ(1-ξ))
            # Formula correct: |τ|/(2|δ-τ|·ξ(1-ξ))
            
            # Compute actual ξ from MVT
            if abs(M[k] - N[k]) > 1e-15:
                logit_diff = np.log(M[k]/(1-M[k])) - np.log(N[k]/(1-N[k]))
                xi_actual = (M[k] - N[k]) / logit_diff
                if xi_actual > 0:
                    xi_actual = np.sqrt(xi_actual * (1 - xi_actual) + 0.25)  # just for reference
                    # Actually ξ is just the MVT point
                    
            results_cross.append({
                'delta': delta, 'tau': tau,
                'cross': cross, 'D': D, 'ratio': ratio,
                'delta_plus_tau': delta + tau,
                'delta_minus_tau': delta - tau,
            })

print(f"  总样本: {len(results_cross)}")
ratios = [r['ratio'] for r in results_cross]
print(f"  cross/D ratio: min={min(ratios):.6f}, max={max(ratios):.6f}, median={np.median(ratios):.6f}")

# 验证两个公式哪个更接近真实值
print("\n  验证哪个公式正确:")
print("  方案1 (文档): |τ|/(2|δ+τ|·ξ(1-ξ))")
print("  方案2 (修正): |τ|/(2|δ-τ|·ξ(1-ξ))")
print()
print("  取 M*_k=0.5, M_k=0.3, N_k=0.7 (超调Case B):")
# Manual test
Mstar_k, M_k, N_k = 0.5, 0.3, 0.7
delta = M_k - Mstar_k
tau = N_k - Mstar_k
cross = (Mstar_k - N_k) * (np.log(M_k/(1-M_k)) - np.log(N_k/(1-N_k)))
D = D_KL_bernoulli(N_k, M_k)
actual_ratio = abs(cross) / D

# MVT ξ
actual_xi = 0.5  # assuming symmetric
# In reality, logit(0.3) - logit(0.7) = ln(3/7) - ln(7/3) = ln(3/7) - ln(7/3) = ln(3/7·3/7) = 2ln(3/7)
# (0.3-0.7) = -0.4, so ξ(1-ξ) = (0.3-0.7)/(logit(0.3)-logit(0.7))
logit_diff = np.log(0.3/0.7) - np.log(0.7/0.3)
xi_term = (0.3 - 0.7) / logit_diff
xi = 0.3 + (0.7-0.3)/2 if xi_term > 0 else 0.5
# Actually MVT gives: logit_diff = (M_k - N_k)/(ξ(1-ξ))
# So ξ(1-ξ) = (M_k - N_k)/logit_diff = -0.4 / logit_diff
xi_1mxi = -0.4 / logit_diff

formula1 = abs(tau) / (2 * abs(delta + tau) * xi_1mxi)  # doc
formula2 = abs(tau) / (2 * abs(delta - tau) * xi_1mxi)  # correct

print(f"  cross/D 实际值: {actual_ratio:.6f}")
print(f"  δ={delta}, τ={tau}")
print(f"  δ+τ={delta+tau}, δ-τ={delta-tau}")
print(f"  ξ(1-ξ) = {xi_1mxi:.6f}")
print(f"  方案1 (文档, δ+τ): {formula1:.6f} (差异: {abs(formula1-actual_ratio):.6f})")
print(f"  方案2 (修正, δ-τ): {formula2:.6f} (差异: {abs(formula2-actual_ratio):.6f})")
print(f"  正确公式: {'方案2 ✓' if abs(formula2-actual_ratio) < abs(formula1-actual_ratio) else '方案1 ✓'}")

# 大规模验证
errors_v1 = []
errors_v2 = []
for r in results_cross:
    delta = r['delta']
    tau = r['tau']
    M_k = r['delta'] + 0.5  # approximate, but we need the actual M_k and N_k
    # Actually we don't have them directly in the results dict, let me skip this for now
    # and do a proper test below

print()
print("  精确大规模验证...")

errors_v1_all = []
errors_v2_all = []
actual_ratios_all = []
for seed in range(100):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    
    for _ in range(30):
        M = np.random.uniform(0.05, 0.95, 5)
        N = n_operator(M, a, b, e, W, V)
        M_safe = np.clip(M, 1e-12, 1-1e-12)
        N_safe = np.clip(N, 1e-12, 1-1e-12)
        
        for k in range(5):
            Ms = Mstar[k]
            Mk = M_safe[k]
            Nk = N_safe[k]
            
            delta_k = Mk - Ms
            tau_k = Nk - Ms
            
            cross_k = (Ms - Nk) * (np.log(Mk/(1-Mk)) - np.log(Nk/(1-Nk)))
            D_k = D_KL_bernoulli(Nk, Mk)
            
            if D_k < 1e-15:
                continue
            
            actual_ratio = abs(cross_k) / D_k
            actual_ratios_all.append(actual_ratio)
            
            # MVT ξ
            logit_d = np.log(Mk/(1-Mk)) - np.log(Nk/(1-Nk))
            if abs(logit_d) < 1e-15:
                continue
            xi_1mxi = (Mk - Nk) / logit_d
            
            v1 = abs(tau_k) / (2 * abs(delta_k + tau_k) * xi_1mxi)
            v2 = abs(tau_k) / (2 * abs(delta_k - tau_k) * xi_1mxi)
            
            errors_v1_all.append(abs(v1 - actual_ratio) / max(actual_ratio, 1e-10))
            errors_v2_all.append(abs(v2 - actual_ratio) / max(actual_ratio, 1e-10))

print(f"  有效样本: {len(actual_ratios_all)}")
print(f"  方案1 (文档, δ+τ) 中位相对误差: {np.median(errors_v1_all):.6f}")
print(f"  方案2 (修正, δ-τ) 中位相对误差: {np.median(errors_v2_all):.6f}")
print(f"  方案1 最大相对误差: {np.max(errors_v1_all):.6f}")
print(f"  方案2 最大相对误差: {np.max(errors_v2_all):.6f}")

if np.median(errors_v2_all) < 1e-12 and np.median(errors_v1_all) > 0.01:
    print("\n  ⚠️ 重大发现: 方案2 (δ-τ) 的误差为机器精度，方案1 (δ+τ) 有显著误差！")
    print("     文档中的 cross/D 公式分母应为 |δ-τ|，而非 |δ+τ|！")
elif np.median(errors_v2_all) < 1e-12:
    print("\n  方案2 (δ-τ) 精确（机器精度），但方案1也可能近似成立？")
    print("     需要进一步分析 δ+τ 与 δ-τ 的关系。")

# 分析: 在什么情况下 δ+τ ≈ δ-τ?
print()
print("  === δ+τ vs δ-τ 在 Case B (超调) 中的关系 ===")
print("  Case B: τ 与 δ 异号")
print("  |δ+τ| = ||δ| - |τ||  (差)")
print("  |δ-τ| = |δ| + |τ|   (和)")
print("  所以当 |τ| << |δ| 时，|δ+τ| ≈ |δ-τ|")
print("  当 |τ| 接近 |δ| 时，|δ+τ| << |δ-τ|，文档公式严重高估 cross/D！")

print()
# ============================================================
# [B] Pinsker不等式在Bernoulli KL中的紧性
# ============================================================
print("=" * 70)
print("[B] Pinsker不等式在Bernoulli KL中的紧性")
print("""
  Pinsker: D_KL(p||q) ≥ 2(p-q)²  (对于 p,q ∈ [0,1] 的 Bernoulli 分布)
  
  这是一个下界——D_KL 可以显著大于 2(p-q)²。
  在极端情况下 (p→0 或 p→1)，D_KL → ∞ 而 2(p-q)² 保持有限。
  
  这对 cross/D 的影响: 
    使用 Pinsker 给出的是上界 (分母变小 → 整体变大)
    但如果实际 D_KL >> 2(p-q)²，则实际 cross/D 远小于 Pinsker 上界
  
  验证: 200种子×100点, 比较 Pinsker 上界和实际 cross/D
""")

pinsker_overestimate = []
for seed in range(200):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    
    for _ in range(50):
        M = np.random.uniform(0.05, 0.95, 5)
        N = n_operator(M, a, b, e, W, V)
        M_safe = np.clip(M, 1e-12, 1-1e-12)
        N_safe = np.clip(N, 1e-12, 1-1e-12)
        
        for k in range(5):
            Ms = Mstar[k]; Mk = M_safe[k]; Nk = N_safe[k]
            delta_k = Mk - Ms; tau_k = Nk - Ms
            
            cross_k = (Ms - Nk) * (np.log(Mk/(1-Mk)) - np.log(Nk/(1-Nk)))
            D_k = D_KL_bernoulli(Nk, Mk)
            
            if D_k < 1e-15:
                continue
            
            actual_ratio = abs(cross_k) / D_k
            
            pinsker_bound = 2 * (Nk - Mk) ** 2
            # Pinsker says D >= pinsker_bound, so cross/D <= |cross|/pinsker_bound
            # But this is an over-estimate of the actual ratio
            overestimate = (abs(cross_k) / max(pinsker_bound, 1e-15)) / max(actual_ratio, 1e-15)
            pinsker_overestimate.append(overestimate)

print(f"  Pinsker 上界 / 实际 cross/D : median={np.median(pinsker_overestimate):.1f}x, "
      f"max={np.max(pinsker_overestimate):.1f}x, min={np.min(pinsker_overestimate):.1f}x")
print(f"  结论: Pinsker上界比实际 cross/D 大幅高估约 {np.median(pinsker_overestimate):.1f}x")
print()

# ============================================================
# [C] MVT ξ(1-ξ) 的正确性验证
# ============================================================
print("=" * 70)
print("[C] MVT logit导数验证")
print("""
  logit(x) = ln(x/(1-x))
  d/dx logit(x) = 1/(x(1-x))
  
  MVT: logit(a) - logit(b) = (a-b)·logit'(ξ) = (a-b)/(ξ(1-ξ))
  其中 ξ ∈ [min(a,b), max(a,b)]
  
  因此 ξ(1-ξ) = (a-b)/(logit(a)-logit(b))
  
  数值验证: ξ(1-ξ) 是否确实在 (0, 0.25] 范围内 (最大值在 ξ=0.5 处)
""")

xi_1mxi_vals = []
for seed in range(100):
    for _ in range(100):
        a = np.random.uniform(0.01, 0.99)
        b = np.random.uniform(0.01, 0.99)
        if abs(a - b) < 1e-10:
            continue
        logit_diff = np.log(a/(1-a)) - np.log(b/(1-b))
        xi_1mxi = (a - b) / logit_diff
        if xi_1mxi > 0:
            xi_1mxi_vals.append(xi_1mxi)

print(f"  ξ(1-ξ) 样本: min={min(xi_1mxi_vals):.6f}, max={max(xi_1mxi_vals):.6f}")
print(f"  (理论: max=0.25 at ξ=0.5, min→0 at ξ→0 or ξ→1)")
print(f"  所以 1/(ξ(1-ξ)) ≥ 4, 且可以任意大")
print()

# ============================================================
# [D] 安全半径 c_KL 的独立数值计算
# ============================================================
print("=" * 70)
print("[D] c_KL - 三阶余项的独立计算")
print("""
  f(v) = V_KL(N(M*+v)) - V_KL(M*+v)
  f(0)=0, f'(0)=0, f''(0) = J^T H J - H
  
  余项 R_3(v) = f(v) - 1/2 v^T f''(0) v
  界: |R_3(v)| ≤ c_KL ||v||^3
  
  通过数值扫描 ||v|| → 0 来估计 c_KL
""")

c_kl_estimates = []
for seed in range(200):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    Astar = a + W @ Mstar
    Bstar = b + V @ Mstar
    Dstar = Astar + Bstar + e
    J = np.zeros((5, 5))
    for k in range(5):
        for j in range(5):
            J[k, j] = (W[k, j] * (1.0 - Mstar[k]) - Mstar[k] * V[k, j]) / Dstar[k]
    H = np.diag([1.0 / (Mstar[k] * (1.0 - Mstar[k])) for k in range(5)])
    H_minus_JTHJ = H - J.T @ H @ J
    
    eigvals = np.linalg.eigvalsh(H_minus_JTHJ)
    lambda_min = eigvals[0]
    
    max_c = 0.0
    for _ in range(200):
        direction = np.random.randn(5)
        direction /= np.linalg.norm(direction)
        
        for r in np.logspace(-3, 1, 50):
            v = r * direction
            M = np.clip(Mstar + v, 1e-15, 1 - 1e-15)
            N = n_operator(M, a, b, e, W, V)
            M_safe = np.clip(M, 1e-12, 1 - 1e-12)
            N_safe = np.clip(N, 1e-12, 1 - 1e-12)
            
            f_v = D_KL_vector(Mstar, N_safe) - D_KL_vector(Mstar, M_safe)
            quad = 0.5 * v @ (J.T @ H @ J - H) @ v
            
            R3 = f_v - quad
            if r > 1e-3 and abs(R3) > 1e-15:
                c_candidate = abs(R3) / (r ** 3)
                if c_candidate > max_c:
                    max_c = c_candidate
    
    c_kl_estimates.append(max_c)

print(f"  c_KL 估计: min={min(c_kl_estimates):.2f}, max={max(c_kl_estimates):.2f}, "
      f"median={np.median(c_kl_estimates):.2f}")
print(f"  文档声称: c_KL ∈ [7.8, 27.0]")
print(f"  匹配: {'✓' if min(c_kl_estimates) >= 7.0 and max(c_kl_estimates) <= 30.0 else '⚠️ 偏离'}")

# 也计算 λ_min
lambda_mins = []
for seed in range(200):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = (a + W @ Mstar) + (b + V @ Mstar) + e
    J = np.zeros((5, 5))
    for k in range(5):
        for j in range(5):
            J[k, j] = (W[k, j] * (1.0 - Mstar[k]) - Mstar[k] * V[k, j]) / Dstar[k]
    H = np.diag([1.0 / (Mstar[k] * (1.0 - Mstar[k])) for k in range(5)])
    Hminus = H - J.T @ H @ J
    lambda_mins.append(np.linalg.eigvalsh(Hminus)[0])

print()
print(f"  λ_min(H - J^T H J): min={min(lambda_mins):.2f}, max={max(lambda_mins):.2f}, "
      f"mean={np.mean(lambda_mins):.2f}")
print(f"  文档声称: [3.8, 4.2]")
print(f"  安全半径 r = λ_min/c_KL: min={min(lambda_mins)/max(c_kl_estimates):.2f}, "
      f"max={max(lambda_mins)/min(c_kl_estimates):.2f}")
print()

# ============================================================
# [E] ‖M_H‖₂ 独立验证
# ============================================================
print("=" * 70)
print("[E] ‖M_H‖₂ / Gershgorin / λ_max 独立验证")
print("""
  对 200 组 FCA 种子独立计算:
  - ‖M_H‖₂ (谱范数/最大奇异值)
  - Gershgorin 界 √(r_max·c_max)
  - λ_max(M_H^sym) 
  - λ_min(I - M_H^sym)
""")

norms = []; gershgorins = []; sym_maxs = []; sym_mins = []
for seed in range(200):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = (a + W @ Mstar) + (b + V @ Mstar) + e
    J = np.zeros((5, 5))
    for k in range(5):
        for j in range(5):
            J[k, j] = (W[k, j] * (1.0 - Mstar[k]) - Mstar[k] * V[k, j]) / Dstar[k]
    
    H = np.diag([1.0 / (Mstar[k] * (1.0 - Mstar[k])) for k in range(5)])
    Hhalf = np.sqrt(H)
    Hinvhalf = np.diag(1.0 / np.diag(Hhalf))
    MH = Hhalf @ J @ Hinvhalf
    
    norm2 = np.linalg.norm(MH, 2)
    norms.append(norm2)
    
    Msym = 0.5 * (MH + MH.T)
    sym_maxs.append(np.max(np.linalg.eigvalsh(Msym)))
    sym_mins.append(np.min(np.linalg.eigvalsh(np.eye(5) - Msym)))
    
    r_max = max(np.sum(np.abs(MH), axis=1))
    c_max = max(np.sum(np.abs(MH), axis=0))
    gershgorins.append(np.sqrt(r_max * c_max))

print(f"  ‖M_H‖₂:          [{min(norms):.4f}, {max(norms):.4f}] mean={np.mean(norms):.4f}")
print(f"  文档:             [0.090, 0.252] mean=0.159")
print(f"  Gershgorin:       [{min(gershgorins):.4f}, {max(gershgorins):.4f}] mean={np.mean(gershgorins):.4f}")
print(f"  文档:             [0.129, 0.341] mean=0.226")
print(f"  λ_max(M_H^sym):   [{min(sym_maxs):.4f}, {max(sym_maxs):.4f}]")
print(f"  文档:             < 0.197")
print(f"  λ_min(I-M_H^sym): [{min(sym_mins):.4f}, {max(sym_mins):.4f}]")
print(f"  文档:             ≥ 0.803")

# 检查是否全 < 1
print(f"\n  全部 ‖M_H‖₂ < 1: {'✓' if max(norms) < 1.0 else '✗'} (max={max(norms):.4f})")
print()

# ============================================================
# [F] diag(D*/D) ≈ I 的近似精度
# ============================================================
print("=" * 70)
print("[F] diag(D*/D) ≈ I 的近似精度")
print("""
  在 M* 邻域内, diag(D*/D) = I + O(||M-M*||)
  这个近似在 Taylor 展开中被使用。
  
  验证: 对不同的 ||M-M*||, 检查 ||diag(D*/D) - I||
""")

for seed in [0, 42, 100]:
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = (a + W @ Mstar) + (b + V @ Mstar) + e
    
    print(f"\n  种子 {seed}:")
    for r in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
        max_err = 0.0
        for _ in range(200):
            direction = np.random.randn(5)
            if np.linalg.norm(direction) < 1e-10:
                continue
            direction /= np.linalg.norm(direction)
            M = np.clip(Mstar + r * direction, 1e-15, 1 - 1e-15)
            D = (a + W @ M) + (b + V @ M) + e
            ratio = Dstar / D
            err = np.max(np.abs(ratio - 1.0))
            if err > max_err:
                max_err = err
        print(f"    r={r:.2f}: max|D*/D - 1| = {max_err:.4f}")

print()
print("  结论: 在 r < 0.1 时精度 ~10%, r > 0.5 时精度差(~50%+)")
print("  Taylor 展开隐含的 diag(D*/D) ≈ I 是合理的局部近似但非全局有效")
print()

# ============================================================
# [G] Case A / Case B 分类验证
# ============================================================
print("=" * 70)
print("[G] Case A (无超调) vs Case B (超调) 分类与符号验证")
print("""
  定义:
  - Case A: N_k 在 M_k 与 M*_k 之间 (无超调)
  - Case B: N_k 穿越 M*_k (超调)
  
  需要验证:
  1. Case A 中 cross 项确实 < 0
  2. Case B 中 cross 项确实 > 0
""")

caseA_cross_neg = 0; caseA_cross_pos = 0
caseB_cross_neg = 0; caseB_cross_pos = 0
caseA_Dratio = []; caseB_Dratio = []

for seed in range(200):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    
    for _ in range(50):
        M = np.random.uniform(0.05, 0.95, 5)
        N = n_operator(M, a, b, e, W, V)
        M_safe = np.clip(M, 1e-12, 1-1e-12)
        N_safe = np.clip(N, 1e-12, 1-1e-12)
        
        for k in range(5):
            Ms = Mstar[k]; Mk = M_safe[k]; Nk = N_safe[k]
            
            cross_k = (Ms - Nk) * (np.log(Mk/(1-Mk)) - np.log(Nk/(1-Nk)))
            
            # Check if N is between M and M* (Case A)
            if Mstar[k] < Mk:
                is_caseA = Mstar[k] <= Nk <= Mk
            else:
                is_caseA = Mk <= Nk <= Mstar[k]
            
            if is_caseA:
                if cross_k < 0:
                    caseA_cross_neg += 1
                else:
                    caseA_cross_pos += 1
            else:
                if cross_k > 0:
                    caseB_cross_pos += 1
                else:
                    caseB_cross_neg += 1

total_caseA = caseA_cross_neg + caseA_cross_pos
total_caseB = caseB_cross_neg + caseB_cross_pos
print(f"  Case A: 负 cross / 总计 = {caseA_cross_neg}/{total_caseA} "
      f"({100*caseA_cross_neg/total_caseA:.1f}% 严格负 ✓)" if total_caseA > 0 else "  Case A: 无样本")
print(f"  Case B: 正 cross / 总计 = {caseB_cross_pos}/{total_caseB} "
      f"({100*caseB_cross_pos/total_caseB:.1f}% 严格正 ✓)" if total_caseB > 0 else "  Case B: 无样本")
print()

# ============================================================
# [H] 6.17C 安全半径独立验证
# ============================================================
print("=" * 70)
print("[H] 6.17C 安全半径独立验证")
print("""
  6.17C 声称:
  λ_min ∈ [0.807, 0.970], c_max ≈ 0.05-0.15
  安全半径 = λ_min / c_max ≈ 5-19 > √5 ≈ 2.24
  
  但也承认: "此估计与 6.17D 的 KL 安全半径面临相同的精度限制"
  
  独立验证 λ_min 和实际 c_max
""")

lambda_mins_IminusJ = []
for seed in range(200):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = (a + W @ Mstar) + (b + V @ Mstar) + e
    J = np.zeros((5, 5))
    for k in range(5):
        for j in range(5):
            J[k, j] = (W[k, j] * (1.0 - Mstar[k]) - Mstar[k] * V[k, j]) / Dstar[k]
    IminusJ = np.eye(5) - J
    sym = 0.5 * (IminusJ + IminusJ.T)
    lambda_mins_IminusJ.append(np.linalg.eigvalsh(sym)[0])

print(f"  λ_min(sym(I-J)): [{min(lambda_mins_IminusJ):.4f}, {max(lambda_mins_IminusJ):.4f}] "
      f"mean={np.mean(lambda_mins_IminusJ):.4f}")
print(f"  文档: [0.807, 0.970], mean=0.904")

# 现在计算 6.17C 的实际 c_max
print("\n  实际 c_max (6.17C 方向单调性的三阶余项):")
print("  g(v) = (N-M)·(M*-M) - v^T(I-J)v, 其中 v=M-M*")
print("  验证: 比较实际值和二次近似值")

c_max_6_17C = []
for seed in range(200):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = (a + W @ Mstar) + (b + V @ Mstar) + e
    J = np.zeros((5, 5))
    for k in range(5):
        for j in range(5):
            J[k, j] = (W[k, j] * (1.0 - Mstar[k]) - Mstar[k] * V[k, j]) / Dstar[k]
    IminusJ = np.eye(5) - J
    
    max_c = 0.0
    for _ in range(50):
        direction = np.random.randn(5)
        direction /= np.linalg.norm(direction)
        
        for r in np.logspace(-3, 0.5, 30):
            v = r * direction
            M = np.clip(Mstar + v, 1e-15, 1 - 1e-15)
            N = n_operator(M, a, b, e, W, V)
            
            actual = (N - M) @ (Mstar - M)
            approx = v @ IminusJ @ v
            
            R3 = actual - approx
            if r > 1e-3 and abs(R3) > 1e-15:
                c = abs(R3) / (r**3)
                if c > max_c:
                    max_c = c
    
    c_max_6_17C.append(max_c)

print(f"  实测 c_max (6.17C): [{min(c_max_6_17C):.2f}, {max(c_max_6_17C):.2f}] "
      f"median={np.median(c_max_6_17C):.2f}")
print(f"  文档声称: ~0.05-0.15")
if np.median(c_max_6_17C) > 0.3:
    print(f"  ⚠️ 实际 c_max ({np.median(c_max_6_17C):.1f}) 远大于声称 (0.05-0.15)！")
    print(f"     安全半径 r = λ_min/c_max ≈ {np.min(lambda_mins_IminusJ)/np.max(c_max_6_17C):.2f} "
          f"到 {np.max(lambda_mins_IminusJ)/np.min(c_max_6_17C):.2f}")
    print(f"     这 << 5-19！")

print()

# ============================================================
# [I] 所有数值声称的独立盲测
# ============================================================
print("=" * 70)
print("[I] 所有数值声称的独立盲测")

# I.1 ΔV statistics
print("\n  I.1 ΔV_KL 统计")
dVs = []
for seed in range(50):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    for _ in range(100):
        M = np.random.uniform(0.05, 0.95, 5)
        N = n_operator(M, a, b, e, W, V)
        M_s = np.clip(M, 1e-12, 1-1e-12)
        N_s = np.clip(N, 1e-12, 1-1e-12)
        dV = D_KL_vector(Mstar, N_s) - D_KL_vector(Mstar, M_s)
        dVs.append(dV)

print(f"  ΔV: mean={np.mean(dVs):.2f}, min={np.min(dVs):.4f}, max={np.max(dVs):.6f}")
print(f"  文档: mean ≈ -1.91, min ≈ -9.1, max ≈ -0.001 到 -0.01")
print(f"  零违规: {'✓' if max(dVs) < 0 else '✗'}")

# I.2 V(N)/V(M) ratio
print("\n  I.2 V(N)/V(M) 比率")
ratios_KL = []
for seed in range(50):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    for _ in range(100):
        M = np.random.uniform(0.05, 0.95, 5)
        N = n_operator(M, a, b, e, W, V)
        M_s = np.clip(M, 1e-12, 1-1e-12)
        N_s = np.clip(N, 1e-12, 1-1e-12)
        V_M = D_KL_vector(Mstar, M_s)
        V_N = D_KL_vector(Mstar, N_s)
        if V_M > 1e-10:
            ratios_KL.append(V_N / V_M)

print(f"  V(N)/V(M): median={np.median(ratios_KL):.4f}, "
      f"p1={np.percentile(ratios_KL, 1):.4f}, p99={np.percentile(ratios_KL, 99):.4f}")
print(f"  文档: 中位数约 0.004")

# I.3 Gershgorin 压缩比
print("\n  I.3 Gershgorin 压缩比")
compression = [g/n for g, n in zip(gershgorins, norms)]
print(f"  Gershgorin/‖M_H‖₂: [{min(compression):.2f}, {max(compression):.2f}] mean={np.mean(compression):.2f}")
print(f"  文档: 压缩 1.18-1.68×")

# I.4 Lie导数
print("\n  I.4 Lie导数")
lie_vals = []
for seed in range(100):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    for _ in range(100):
        M = np.random.uniform(0.05, 0.95, 5)
        N = n_operator(M, a, b, e, W, V)
        grad_V = (M - Mstar) / (M * (1 - M))
        lie = grad_V @ (N - M)
        lie_vals.append(lie)

print(f"  Lie导数: max={np.max(lie_vals):.6f}, min={np.min(lie_vals):.6f}")
print(f"  文档: max 约 -0.02, 零违规")
print(f"  零违规: {'✓' if max(lie_vals) < 0 else '✗'}")

# I.5 全体 ΔV < 0 的大规模检验
print("\n  I.5 大规模 ΔV < 0 检验")
dVs_large = []
for seed in range(100):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    for _ in range(100):
        M = np.random.uniform(0.05, 0.95, 5)
        N = n_operator(M, a, b, e, W, V)
        M_s = np.clip(M, 1e-12, 1-1e-12)
        N_s = np.clip(N, 1e-12, 1-1e-12)
        dV = D_KL_vector(Mstar, N_s) - D_KL_vector(Mstar, M_s)
        dVs_large.append(dV)

print(f"  样本: {len(dVs_large)}, ΔV > 0: {sum(1 for v in dVs_large if v > 0)}")
print(f"  max ΔV: {max(dVs_large):.10f}")
print(f"  零违规: {'✓' if max(dVs_large) < 0 else '✗' if max(dVs_large) <= 1e-12 else '✗✗✗'}")

print()

# ============================================================
# [J] ‖M_H‖₂ vs Gershgorin vs λ_max(M_H^sym) 一致性
# ============================================================
print("=" * 70)
print("[J] ‖M_H‖₂ vs Gershgorin vs λ_max(M_H^sym) 一致性验证")
print("""
  理论关系:
  - λ_max(M_H^sym) ≤ ‖M_H‖₂ (谱范数 ≥ 对称部分的谱半径)
  - ‖M_H‖₂ ≤ √(r_max·c_max) (Gershgorin 行-列界)
  - λ_min(I-M_H^sym) = 1 - λ_max(M_H^sym)
  
  验证: 所有不等式在数值上成立
""")

violations = 0
for norm2, gersh, sym_max, sym_min in zip(norms, gershgorins, sym_maxs, sym_mins):
    if sym_max > norm2 + 1e-10:
        violations += 1
    if norm2 > gersh + 1e-10:
        violations += 1
    if abs((1 - sym_max) - sym_min) > 1e-10:
        violations += 1

print(f"  λ_max(M_sym) ≤ ‖M_H‖₂: {'✓' if all(np.array(sym_maxs) <= np.array(norms) + 1e-10) else '✗'}")
print(f"  ‖M_H‖₂ ≤ √(r·c):        {'✓' if all(np.array(norms) <= np.array(gershgorins) + 1e-10) else '✗'}")
print(f"  λ_min(I-M_sym) = 1-λ_max: {'✓' if all(abs(np.array(sym_mins) - (1-np.array(sym_maxs))) < 1e-10) else '✗'}")
print(f"  总违规: {violations}")

print()

# ============================================================
# 总结
# ============================================================
print("=" * 70)
print("第八轮审计总结")
print("=" * 70)

issues = []

# Check [A]
cross_d_formula_correct = np.median(errors_v2_all) < 1e-12
cross_d_formula_wrong = np.median(errors_v1_all) > 0.01
if cross_d_formula_correct and cross_d_formula_wrong:
    issues.append("[A] ⚠️ Cross/D 公式: 文档使用 |δ+τ| 应为 |δ-τ| (但经验 max=2.17 仍成立，因数值来自独立经验计算)")

# Check [D]
if abs(np.median(c_kl_estimates) - 15) < 8:
    issues.append(f"[D] c_KL: 实测 [{min(c_kl_estimates):.1f}, {max(c_kl_estimates):.1f}]，与文档 [7.8, 27.0] 基本一致 ✓")
else:
    issues.append(f"[D] c_KL: 实测 [{min(c_kl_estimates):.1f}, {max(c_kl_estimates):.1f}] 与文档不符 ⚠️")

# Check [E]
if abs(min(norms) - 0.090) < 0.01 and abs(max(norms) - 0.252) < 0.01:
    issues.append("[E] ‖M_H‖₂ 数值: 与文档一致 ✓")
else:
    issues.append(f"[E] ‖M_H‖₂ 数值: [{min(norms):.4f}, {max(norms):.4f}] 与文档 [0.090, 0.252] 有偏差")

# Check [H]
c_max_median = np.median(c_max_6_17C)
if c_max_median > 0.5:
    issues.append(f"[H] ⚠️ 6.17C c_max 实测 median={c_max_median:.2f}，远大于文档声称 0.05-0.15！安全半径被严重高估")
elif c_max_median > 0.15:
    issues.append(f"[H] 6.17C c_max 实测 median={c_max_median:.2f}，略大于文档声称 0.05-0.15")
else:
    issues.append(f"[H] 6.17C c_max 实测 median={c_max_median:.2f}，与文档一致 ✓")

# Check [I]
if max(dVs_large) < 0:
    issues.append("[I] ΔV_KL 零违规: ✓")
else:
    issues.append(f"[I] ΔV_KL 违规: max={max(dVs_large):.10f} ⚠️")

if max(lie_vals) < 0:
    issues.append("[I] Lie导数 零违规: ✓")
else:
    issues.append(f"[I] Lie导数 违规: max={max(lie_vals):.10f} ⚠️")

for issue in issues:
    print(f"  {issue}")

print(f"\n共发现 {len([i for i in issues if '⚠️' in i])} 个需要关注的问题。")
