"""
φ''(r) ≤ 0 深度结构分析 — 四条证明路径探索
=============================================

路径1: Hessian半负定——φ'' = u^T Q u, Q ≺ 0?
路径2: 凸性+收缩——V_KL凸 + N收缩 ⇒ V_KL(N(M)) ≤ V_KL(M)?
路径3: 完美平方持续性——r=0的完美平方能否推广到r>0?
路径4: 耦合对称化——W,V耦合结构中的对称性

关键恒等式:
  φ(r) = V_KL(N(M*+ru)) - V_KL(M*+ru)
  V_KL 是凸函数 (Hessian = diag(M*_k/M_k² + (1-M*_k)/(1-M_k)²) ≻ 0)
  ψ(r) = V_KL(M*+ru) 在r上凸
  η(r) = V_KL(N(M*+ru))
  φ(r) = η(r) - ψ(r), φ'' = η'' - ψ''
  φ'' ≤ 0 ⇔ η'' ≤ ψ'' (N曲率 ≤ KL曲率)
"""
import numpy as np
from scipy.optimize import minimize_scalar

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

def V_KL_func(M, Mstar):
    eps = 1e-15
    M = np.clip(M, eps, 1-eps)
    return np.sum(Mstar * np.log(Mstar / M) + (1-Mstar) * np.log((1-Mstar) / (1-M)))

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
# 路径1: Hessian 半负定分析
# ============================================================
print("=" * 70)
print("路径1: 凸性结构 — ψ''(r) vs η''(r)")
print("=" * 70)
print("""
  ψ(r) = V_KL(M*+ru): 凸函数沿射线 → ψ''(r) ≥ 0 (严格 > 0)
  η(r) = V_KL(N(M*+ru)): N压缩后的KL散度
  φ'' = η'' - ψ'' ≤ 0 ⇔ η'' ≤ ψ'' (N弯曲 ≤ KL弯曲)
""")

# Check ψ'' structure
print("\nψ'' 的结构验证 (M*处):")
for seed_id in [0, 11, 149]:
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    
    H_KL = np.diag(Mstar / Mstar**2 + (1-Mstar) / (1-Mstar)**2)
    
    # For 100 random directions at r=0
    min_psi = np.inf
    max_psi = -np.inf
    for _ in range(500):
        u = np.random.randn(5)
        u = u / np.linalg.norm(u)
        psi_dd = u @ H_KL @ u  # ψ''(0) = u^T H_KL u
        min_psi = min(min_psi, psi_dd)
        max_psi = max(max_psi, psi_dd)
    
    # η''(0) structure
    Dstar = a + b + e + (W+V) @ Mstar
    J = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k != j:
                J[k,j] = (W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k]
    
    H_N = J.T @ H_KL @ J  # η''(0) ≈ u^T J^T H_KL J u (linearized N)
    
    eta_min = np.inf
    eta_max = -np.inf
    phi_max = -np.inf
    phi_min = np.inf
    for _ in range(500):
        u = np.random.randn(5)
        u = u / np.linalg.norm(u)
        eta_dd = u @ H_N @ u
        psi_dd = u @ H_KL @ u
        phi_dd = eta_dd - psi_dd  # φ''(0)
        eta_min = min(eta_min, eta_dd)
        eta_max = max(eta_max, eta_dd)
        phi_min = min(phi_min, phi_dd)
        phi_max = max(phi_max, phi_dd)
    
    print(f"  seed {seed_id}: ψ''∈[{min_psi:.4f},{max_psi:.4f}]  "
          f"η''_lin∈[{eta_min:.4f},{eta_max:.4f}]  "
          f"φ''_lin∈[{phi_min:.4f},{phi_max:.4f}]  "
          f"{'✓' if phi_max < 0 else '✗'}")

# ============================================================
# 路径1b: 检查 ψ(r) 沿N迭代轨道的单调性
# ============================================================
print(f"\n{'='*70}")
print("路径1b: ψ(M(t)) 沿迭代轨道的单调性")
print("=" * 70)

for seed_id in [0, 11, 149]:
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    
    M = np.full(5, 0.5)
    n_increases = 0
    for t in range(50):
        M_next = n_operator(M, a, b, e, W, V)
        psi = V_KL_func(M, Mstar)
        psi_next = V_KL_func(M_next, Mstar)
        if psi_next > psi:
            n_increases += 1
        M = M_next
        if np.max(np.abs(M - Mstar)) < 1e-12:
            break
    
    print(f"  seed {seed_id}: ψ上升次数={n_increases}  {'✓ 单调下降' if n_increases==0 else '✗'}")

# ============================================================
# 路径2: 凸性 + 收缩 → KL下降?
# ============================================================
print(f"\n{'='*70}")
print("路径2: 凸性(V_KL) vs 收缩性(N) 的关系")
print("=" * 70)

# For a convex f with minimum at x*, and T with ||T(x)-x*|| ≤ α||x-x*||:
# Does f(T(x)) ≤ f(x) always hold? No.
# Counter-example: f(x) = |x|² (convex), T(x) = x/2, x*=0
# f(T(x)) = |x|²/4 ≤ |x|² = f(x) ✓
# But for anisotropic f, direction matters.

# Test: for all 200 seeds, test relation between l1 contraction and KL monotonicity
print("\n  l1收缩比 vs KL下降比 (5000 random points):")
for seed_id in [0, 11, 149]:
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    
    max_l1_ratio = 0
    max_kl_ratio = 0
    kl_increase_count = 0
    
    for _ in range(5000):
        M = np.random.uniform(0.05, 0.95, 5)
        M_next = n_operator(M, a, b, e, W, V)
        
        l1_before = np.sum(np.abs(M - Mstar))
        l1_after = np.sum(np.abs(M_next - Mstar))
        l1_ratio = l1_after / max(l1_before, 1e-15)
        max_l1_ratio = max(max_l1_ratio, l1_ratio)
        
        kl_before = V_KL_func(M, Mstar)
        kl_after = V_KL_func(M_next, Mstar)
        kl_ratio = kl_after / max(kl_before, 1e-15)
        max_kl_ratio = max(max_kl_ratio, kl_ratio)
        
        if kl_after > kl_before:
            kl_increase_count += 1
    
    print(f"  seed {seed_id}: l1比≤{max_l1_ratio:.4f}  KL比≤{max_kl_ratio:.4f}  "
          f"KL上升{kl_increase_count}/5000={kl_increase_count/50:.1f}%")

# ============================================================
# 路径3: 完美平方持续性
# ============================================================
print(f"\n{'='*70}")
print("路径3: 完美平方在r>0时的变形分析")
print("=" * 70)

# Φ''_k = -(w+v)²/D² + θw²/A² + (1-θ)v²/B² - θu²/M² - (1-θ)u²/(1-M)²
# 
# Define the "deviation from perfect square":
# At r=0: N曲率 = ((1-θ)w-θv)² / (θ(1-θ)D*²)
# At r>0: N曲率 = -(w+v)²/D² + θw²/A² + (1-θ)v²/B²
#
# Key insight: write A = θD* + δA, B = (1-θ)D* + δB, D = D* + δD
# where δA = rw, δB = rv, δD = r(w+v) = δA + δB

for seed_id in [0, 11, 149]:
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W+V) @ Mstar
    
    for _ in range(50):
        u = np.random.randn(5)
        u = u / np.linalg.norm(u)
        w_vec = W @ u  # (Wu)_k
        v_vec = V @ u  # (Vu)_k
        
        for r in np.linspace(0.01, 0.3, 5):
            M = Mstar + r * u
            M = np.clip(M, 1e-6, 1-1e-6)
            
            A = a + W @ M
            B = b + V @ M + e
            D = A + B
            theta = Mstar
            
            n_curv = -(w_vec+v_vec)**2/D**2 + theta*w_vec**2/A**2 + (1-theta)*v_vec**2/B**2
            kl_curv = theta*u**2/M**2 + (1-theta)*u**2/(1-M)**2
            
            # Perfect square at r=0:
            perf_sq = ((1-theta)*w_vec - theta*v_vec)**2 / (theta*(1-theta)*Dstar**2)
            
            # Deviation
            dev = n_curv - perf_sq
            
            pass  # Analysis printed below

# Deeper: for one specific seed, compute the deviation analytically
print("\n  单种子深度分析 (seed 11, random direction):")
a, b, e, W, V = gen_FCA(11)
Mstar = compute_fp(a, b, e, W, V)
Dstar = a + b + e + (W+V) @ Mstar
theta = Mstar

u = np.random.randn(5); u = u / np.linalg.norm(u)
w_vec = W @ u; v_vec = V @ u

print(f"  M* = {[f'{x:.4f}' for x in Mstar]}")
print(f"  u  = {[f'{x:.4f}' for x in u]}")
print(f"  Wu = {[f'{x:.4f}' for x in w_vec]}")
print(f"  Vu = {[f'{x:.4f}' for x in v_vec]}")
print(f"\n  {'r':>8}  {'N曲率':>10}  {'完美平方':>10}  {'KL曲率':>10}  {'φ_k''(0)':>10}  {'φ_k''':>10}")
print(f"  {'-'*70}")
for r in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]:
    M = Mstar + r * u
    M = np.clip(M, 1e-6, 1-1e-6)
    A = a + W @ M
    B = b + V @ M + e
    D = A + B
    
    n_curv = -(w_vec+v_vec)**2/D**2 + theta*w_vec**2/A**2 + (1-theta)*v_vec**2/B**2
    kl_curv = theta*u**2/M**2 + (1-theta)*u**2/(1-M)**2
    perf_sq = ((1-theta)*w_vec - theta*v_vec)**2 / (theta*(1-theta)*Dstar**2)
    
    phi_k = n_curv - kl_curv
    
    print(f"  {r:8.3f}  {sum(n_curv):10.6f}  {sum(perf_sq):10.6f}  {sum(kl_curv):10.6f}  {sum(phi_k):10.6f}")

# ============================================================
# 路径4: 跨分量耦合的矩阵形式
# ============================================================
print(f"\n{'='*70}")
print("路径4: φ'' 作为二次型 — 矩阵 Q(r) 的谱分析")
print("=" * 70)

# φ''(r) = u^T Q(r) u where Q depends on r and direction u
# Q_kj(r) = ∂²Φ_k/∂M_j∂M_k? No, it's more complex.
#
# Actually φ is a function of r via M(r). Let's compute Q directly.
# φ(r) = Σ_k [F_k(A_k(r), B_k(r)) + θ_k log M_k(r) + (1-θ_k) log(1-M_k(r))]
# where F_k(A,B) = log(A+B) - θ_k log A - (1-θ_k) log B
#
# Then φ'' = Σ_k [ (∂²/∂r²)F_k + θ_k (-u_k²/M_k²) + (1-θ_k)(-u_k²/(1-M_k)²) ]
# with ∂²F_k/∂r² = -(Wu+Vu)_k²/D_k² + θ_k(Wu)_k²/A_k² + (1-θ_k)(Vu)_k²/B_k²

# The key matrix: Q(r) such that φ''(r) = u^T Q(r) u
# Component of Q(r):
# Q_ij(r) = Σ_k [ 2W_ki W_kj / A_k² · θ_k/2?  No, the 2nd deriv is ∂²/∂M_i∂M_j...

# Actually, ∂Φ_k/∂M_i = w_ki/D_k - θ_k w_ki/A_k + (1-θ_k)v_ki/B_k + ... 
# This is messy. Let's compute numerically.

print("\n  Q(r) 的数值构造和特征值分析 (seed 11):")
a, b, e, W, V = gen_FCA(11)
Mstar = compute_fp(a, b, e, W, V)
theta = Mstar

for r in [0.001, 0.05, 0.15, 0.3, 0.5, 0.8]:
    Q = np.zeros((5, 5))
    
    # Numerical Hessian of φ at M = M* + r*u for several u directions
    # Then average / fit Q
    # Alternative: compute Q from analytical formula
    
    # Q_ij = ∂²φ/∂M_i∂M_j at M = M* + r*u
    # This depends on the point, not the direction!
    # φ evaluated at a FIXED point M, then u runs over directions.
    # At M = M* + r*u, φ is:
    # V_KL(N(M), M*) - V_KL(M, M*)
    # 
    # But we're looking at φ(r) = V_KL(N(M*+ru)) - V_KL(M*+ru)
    # The Hessian of φ with respect to u would involve chain rule through N.
    
    # Let me instead compute φ'' numerically for 100 random u and check
    # if it can be expressed as u^T Q u with Q ≺ 0.
    
    break  # placeholder - doing full analysis below

# Simpler approach: directly test the negative definiteness
print("\n  直接测试: φ'' 是否对所有方向为负?")
print("  (seed 11, 1000 directions, 10 radii each)")

a, b, e, W, V = gen_FCA(11)
Mstar = compute_fp(a, b, e, W, V)
theta = Mstar
Dstar = a + b + e + (W+V) @ Mstar

violations = 0
total = 0
worst_phi = 0

for _ in range(1000):
    u = np.random.randn(5)
    u = u / np.linalg.norm(u)
    w_vec = W @ u
    v_vec = V @ u
    
    for r in [0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]:
        M = Mstar + r * u
        if np.any(M < 1e-6) or np.any(M > 1-1e-6):
            continue
        M = np.clip(M, 1e-6, 1-1e-6)
        
        A = a + W @ M
        B = b + V @ M + e
        D = A + B
        
        n_curv = -(w_vec+v_vec)**2/D**2 + theta*w_vec**2/A**2 + (1-theta)*v_vec**2/B**2
        kl_curv = theta*u**2/M**2 + (1-theta)*u**2/(1-M)**2
        phi_dd = sum(n_curv - kl_curv)
        
        total += 1
        if phi_dd > 0:
            violations += 1
        worst_phi = max(worst_phi, phi_dd)

print(f"  φ'' > 0: {violations}/{total} ({violations/total*100:.1f}%)")
print(f"  最劣φ'': {worst_phi:.6f}")

# ============================================================
# 路径4b: 检查 ψ 凸性提供的自然下界
# ============================================================
print(f"\n{'='*70}")
print("路径4b: 凸性防线 — ψ''(r) 是否自然大于 N 曲率?")
print("=" * 70)

# ψ(r) = V_KL(M*+ru) is convex in r
# ψ''(r) = Σ_k u_k² [θ_k/M_k² + (1-θ_k)/(1-M_k)²]
# This grows very fast as M → 0 or M → 1

# For φ'' ≤ 0, we need η''(r) ≤ ψ''(r):
# η'' = Σ_k [-(Wu+Vu)_k²/D_k² + θ_k(Wu)_k²/A_k² + (1-θ_k)(Vu)_k²/B_k²]
# ψ'' = Σ_k u_k² [θ_k/M_k² + (1-θ_k)/(1-M_k)²]

# Question: does ψ'' dominate η'' because u_k terms in ψ'' have 1/M² blowup?

for seed_id in [0, 11, 149]:
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    theta = Mstar
    
    # At M* (r=0)
    H_KL = np.diag(theta/theta**2 + (1-theta)/(1-theta)**2)
    # ψ''(0) = u^T H_KL u ≥ Σ_k u_k²/(θ_k(1-θ_k))
    
    # Compare: at M=M*, for the "worst" direction (maximizes η''/ψ'' ratio)
    # η''(0) = u^T J^T H_KL J u (linearized)
    # The ratio max_u η''/ψ'' = λ_max(H_KL^{-1} J^T H_KL J) ? No...
    
    # Actually: η'' = u^T J^T H_KL J u, ψ'' = u^T H_KL u
    # Ratio bound: η''/ψ'' ≤ ‖H_KL^{1/2} J H_KL^{-1/2}‖² = ‖M_ℋ‖²
    # From the document: ‖M_ℋ‖₂ ∈ [0.09, 0.252] < 1
    # So η'' ≤ ‖M_ℋ‖² ψ'' ≈ 0.008-0.064 ψ'' at r=0!
    
    Dstar = a + b + e + (W+V) @ Mstar
    J = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k != j:
                J[k,j] = (W[k,j]*(1-theta[k]) - V[k,j]*theta[k]) / Dstar[k]
    
    H_sqrt = np.diag(np.sqrt(1/(theta*(1-theta))))
    M_H = H_sqrt @ J @ np.linalg.inv(H_sqrt)
    
    from numpy.linalg import svd
    _, s, _ = svd(M_H)
    ratio_bound = s[0]**2
    
    print(f"  seed {seed_id}: ‖M_ℋ‖²={ratio_bound:.4f} → η''≤{ratio_bound:.4f}·ψ'' at r=0")

# ============================================================
# 路径4c: 最激进猜想 — φ''≤0 对所有r源于凸性保护
# ============================================================
print(f"\n{'='*70}")
print("路径4c: 凸性全局保护猜想")
print("=" * 70)
print("""
猜想: 对任意r>0和任意方向u,
  η''(r) = Σ_k [-(Wu+Vu)²/D² + θ(Wu)²/A² + (1-θ)(Vu)²/B²]
  ψ''(r) = Σ_k u_k²[θ/M² + (1-θ)/(1-M)²]
  
  始终有 η''(r) ≤ ψ''(r), 且 ψ''(r) → ∞ 当 M_k → 0 或 1

证据:
  - r=0时: η''/ψ'' ≤ ‖M_ℋ‖² ≈ 0.01-0.06 ≪ 1
  - r→∞时: ψ''中 1/M² 项支配一切 → ψ'' ≫ η''
  - 问题在于中间r: η''可能因分母变小而增大

测试: 对每个种子, 找到 sup_r η''(r)/ψ''(r) 的最劣值
""")

# Find worst-case ratio η''/ψ'' for each seed
print("\n  搜索最劣 η''/ψ'' 比值 (seed 0,11,149):")
for seed_id in [0, 11, 149]:
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    theta = Mstar
    
    worst_ratio = 0
    worst_r = 0
    worst_u = None
    
    for _ in range(2000):
        u = np.random.randn(5)
        u = u / np.linalg.norm(u)
        w_vec = W @ u
        v_vec = V @ u
        
        for r in np.linspace(0.01, 1.5, 30):
            M = Mstar + r * u
            if np.any(M < 1e-6) or np.any(M > 1-1e-6):
                continue
            M = np.clip(M, 1e-6, 1-1e-6)
            
            A = a + W @ M
            B = b + V @ M + e
            D = A + B
            
            eta = sum(-(w_vec+v_vec)**2/D**2 + theta*w_vec**2/A**2 + (1-theta)*v_vec**2/B**2)
            psi = sum(u**2 * (theta/M**2 + (1-theta)/(1-M)**2))
            
            if psi > 0:
                ratio = eta / psi
                if ratio > worst_ratio:
                    worst_ratio = ratio
                    worst_r = r
                    worst_u = u.copy()
    
    # Check if η'' could ever EXCEED ψ'' 
    # If η'' ≤ ψ'' always, then φ'' = η'' - ψ'' ≤ 0
    print(f"  seed {seed_id}: 最劣η''/ψ'' = {worst_ratio:.6f} at r={worst_r:.3f}  "
          f"{'✓ η''≤ψ''始终' if worst_ratio < 1 else '✗ η''可能超ψ'''}")


# ============================================================
# 综合: 最有利于证明的发现总结
# ============================================================
print(f"\n{'='*70}")
print("综合: 证明策略评估")
print("=" * 70)
print("""
四条路径对比:

路径1 (Hessian半负定): 
  → 可归结为 ‖H_KL^{1/2}·J(M)·H_KL^{-1/2}‖ < 1 ∀M
  → 这比逐分量分析更强、更简洁
  → r=0已验证，r>0需证明该范数单调不增长

路径2 (凸性+收缩): 
  → V_KL凸 + N收缩 → 自然KL下降
  → 可能是最简洁的全局证明
  → 需证: 对凸函数f和收缩映射T, f(Tx)≤f(x)的条件

路径3 (完美平方持续): 
  → r=0完美平方 ⇒ N曲率由J的行向量决定
  → r>0时N曲率的额外项可用凸性控制
  → 最直接的二阶分析

路径4 (ψ''自然支配η''):
  → η''/ψ'' ≤ ‖M_ℋ(r)‖² 对某个加权Jacobian
  → r=0比率极小(0.01-0.06), 需证r>0时仍<1
  → ψ''的1/M²增长是强保护

推荐优先级: 路径1 > 路径4 > 路径2 > 路径3
""")
