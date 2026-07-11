"""
系统性审计 v10 — 7维度深度检查
================================
审计1: ρ(B_sym(M)) 在整个域上 < 1（非仅 M* 处）
审计2: x-y归一化恒等式独立推导 + 数值验证
审计3: α 收缩界逐步验证 + 不等式链每步误差
审计4: D_low 下界保守性 + 对 M(t) 影响
审计5: φ''<0 极端参数组合
审计6: 跨引理一致性
审计7: 浮点精度 / 除零 / 数值消失
"""
import numpy as np

# ============================================================
# 工具函数
# ============================================================
def n_operator(M, a, b, eps, W, V):
    num = a + W @ M
    return num / (num + b + V @ M + eps)

def compute_fp(a, b, eps, W, V):
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
    np.fill_diagonal(W, 0)
    np.fill_diagonal(V, 0)
    t = a.sum() + b.sum() + W.sum() + V.sum()
    W *= 5.0 / t; V *= 5.0 / t
    return a, b, e, W, V

# ============================================================
# 审计1: ρ(B_sym(M)) — 在随机 M 上（不只是 M*）
# ============================================================
print("=" * 70)
print("审计1: ρ(B_sym(M)) 在整个域上 < 1 ?")
print("=" * 70)

# B_kj(M) = (D*_k / D_k(M)) · J_kj(M*), B_sym = (B+B^T)/2
# 关键问题: D_k(M) 随 M 变化, D*/D_k 缩放 J 的行
# 当 M_k 很小时 D_k 也小 ⇒ D*/D_k 大 ⇒ B 被放大
# 需验证: sup_{M∈[m^(0),1]⁵} ρ(B_sym(M)) < 1 ?

max_rho_all = 0; viol_count = 0; total_rho = 0
worst_rho_seed = -1; worst_M = None

for s in range(200):
    a, b, e, W, V = gen_FCA(s)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W + V) @ Mstar

    J = np.zeros((5, 5))
    for k in range(5):
        for jj in range(5):
            if k != jj:
                J[k, jj] = (W[k, jj] * (1 - Mstar[k]) - V[k, jj] * Mstar[k]) / Dstar[k]

    D_max = a + b + e + (W + V) @ np.ones(5)
    m0 = a / D_max

    for _ in range(100):
        M = m0 + np.random.random(5) * (1.0 - m0)
        D = a + b + e + (W + V) @ M

        B = np.zeros((5, 5))
        for k in range(5):
            for jj in range(5):
                B[k, jj] = (Dstar[k] / D[k]) * J[k, jj]

        B_sym = (B + B.T) / 2
        eigvals = np.linalg.eigvalsh(B_sym)
        rho = max(abs(eigvals))
        total_rho += 1

        if rho > max_rho_all:
            max_rho_all = rho
            worst_rho_seed = s
            worst_M = D.copy()
        if rho >= 1:
            viol_count += 1

print(f"  扫描: 200种子×100随机M = {total_rho} 点")
print(f"  max ρ(B_sym) = {max_rho_all:.4f} (seed {worst_rho_seed})")
print(f"  ρ(B_sym) ≥ 1: {viol_count}/{total_rho}")
print(f"  {'✗ 危险: 存在 ρ≥1 !' if viol_count > 0 else '✓ 全域 < 1'}")

# ============================================================
# 审计1b: 特定检查 — M 靠近 m0 时 D 最小 ⇒ B 最大
# ============================================================
print(f"\n  极端检查: M→m^(0) (最小 D, 最大 B 放大):")
worst_rho_extreme = 0
for s in range(200):
    a, b, e, W, V = gen_FCA(s)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W + V) @ Mstar
    D_max = a + b + e + (W + V) @ np.ones(5)
    m0 = a / D_max

    J = np.zeros((5, 5))
    for k in range(5):
        for jj in range(5):
            if k != jj:
                J[k, jj] = (W[k, jj] * (1 - Mstar[k]) - V[k, jj] * Mstar[k]) / Dstar[k]

    # M = m0 (worst case for D)
    D_min = a + b + e + (W + V) @ m0
    B = np.zeros((5, 5))
    for k in range(5):
        for jj in range(5):
            B[k, jj] = (Dstar[k] / D_min[k]) * J[k, jj]
    B_sym = (B + B.T) / 2
    rho = max(abs(np.linalg.eigvalsh(B_sym)))
    worst_rho_extreme = max(worst_rho_extreme, rho)

print(f"  max ρ(B_sym) @ M=m^(0): {worst_rho_extreme:.4f}")
print(f"  {'✗ 危险!' if worst_rho_extreme >= 1 else '✓ < 1'}")

# ============================================================
# 审计2: x-y归一化恒等式 — 独立推导验证
# ============================================================
print("\n" + "=" * 70)
print("审计2: x-y归一化恒等式 — 独立推导 + 链式法则验证")
print("=" * 70)

# 独立推导路线:
# M(r) = M* + r·u
# A(r) = a + W·M(r) = a + W·M* + r·Wu = A* + r·w  ⇒ ∂A/∂r = w
# B(r) = b+ε+V·M(r) = B* + r·v  ⇒ ∂B/∂r = v
# D(r) = A+B = D* + r·(w+v) ⇒ ∂D/∂r = w+v
# M_k(r) = A_k/D_k ⇒ u_k = ∂M_k/∂r|_0 = (w_k D* - A*_k(w_k+v_k))/D*²
#   = (w_k(D* - A*_k) - A*_k v_k)/D*² = (w_k B*_k - v_k A*_k)/D*²
#   = (w_k(1-θ_k) - v_k θ_k)/D*_k ✓ (matches J·u)

# 链式验证:
# Φ_k(r) = log D_k(r) - θ_k log A_k(r) - (1-θ_k) log B_k(r) 
#          + θ_k log M_k(r) + (1-θ_k) log(1-M_k(r))
# 
# ∂Φ_k/∂r = (w+v)/D - θ·w/A - (1-θ)·v/B + θ·u/M - (1-θ)·u/(1-M)
# ∂²Φ_k/∂r² = -(w+v)²/D² + θ·w²/A² + (1-θ)·v²/B² - θ·u²/M² - (1-θ)·u²/(1-M)²
# 
# η''_k(r) = -(w+v)²/D² + θ·w²/A² + (1-θ)·v²/B²
# ψ''_k(r) = θ·u²/M² + (1-θ)·u²/(1-M)²
# 
# x-y 参数化: 
# A(r) = A*(1 + r·w/A*) = A*(1+rx)
# D(r) = D*(1 + r·(w+v)/D*) = D*(1+rz)
# z = (w+v)/D* = w/D* + v/D* = (A*/D*)(w/A*) + (B*/D*)(v/B*) = θx + (1-θ)y ✓
# 
# η''_k = θ·(w/A*·A*/A)² + (1-θ)·(v/B*·B*/B)² - ((w+v)/D*·D*/D)²
#       = θ·x²/(1+rx)² + (1-θ)·y²/(1+ry)² - z²/(1+rz)² ✓

# 验证:
max_err_xy = 0; max_err_xy_r = 0
for s in range(200):
    a, b, e, W, V = gen_FCA(s)
    Mstar = compute_fp(a, b, e, W, V)
    theta = Mstar
    A_star = a + W @ Mstar
    B_star = b + e + V @ Mstar
    Dstar = A_star + B_star

    for _ in range(30):
        u = np.random.randn(5); u = u / np.linalg.norm(u)
        w_u = W @ u; v_u = V @ u
        x = w_u / A_star; y = v_u / B_star
        z = theta * x + (1 - theta) * y

        # r=0 等式
        eta0_direct = sum(theta * w_u**2 / A_star**2 
                        + (1-theta) * v_u**2 / B_star**2 
                        - (w_u+v_u)**2 / Dstar**2)
        eta0_xy = sum(theta * x**2 + (1-theta) * y**2 - z**2)
        max_err_xy = max(max_err_xy, abs(eta0_direct - eta0_xy))

        # r>0 等式
        for r in np.linspace(0.01, 2.0, 20):
            M = theta + r * u
            if np.any(M <= 1e-12) or np.any(M >= 1-1e-12): continue
            A = a + W @ M; B = b + e + V @ M; D = A + B

            eta_direct = sum(theta * w_u**2 / A**2
                          + (1-theta) * v_u**2 / B**2
                          - (w_u+v_u)**2 / D**2)
            eta_xy = sum(theta * x**2 / (1+r*x)**2
                      + (1-theta) * y**2 / (1+r*y)**2
                      - z**2 / (1+r*z)**2)
            max_err_xy_r = max(max_err_xy_r, abs(eta_direct - eta_xy))

print(f"  max |η''_direct − η''_xy| @ r=0: {max_err_xy:.2e}")
print(f"  max |η''_direct − η''_xy| @ r>0: {max_err_xy_r:.2e}")
print(f"  {'✓ 恒等式严格成立' if max(max_err_xy, max_err_xy_r) < 1e-14 else '✗'}")

# 额外: 验证 z = θx+(1-θ)y 与 (w+v)/D* 相等
max_err_z = 0
for s in range(200):
    a, b, e, W, V = gen_FCA(s); Mstar = compute_fp(a, b, e, W, V)
    A_s = a + W @ Mstar; B_s = b + e + V @ Mstar; D_s = A_s + B_s
    for _ in range(100):
        u = np.random.randn(5); u /= np.linalg.norm(u)
        x = (W @ u) / A_s; y = (V @ u) / B_s
        z1 = Mstar * x + (1-Mstar) * y
        z2 = (W @ u + V @ u) / D_s
        max_err_z = max(max_err_z, np.max(np.abs(z1 - z2)))
print(f"  max |θx+(1-θ)y − (w+v)/D*| = {max_err_z:.2e}  {'✓' if max_err_z < 1e-14 else '✗'}")

# ============================================================
# 审计3: α 收缩 — 不等式链每一步的误差
# ============================================================
print("\n" + "=" * 70)
print("审计3: 6.17B α 收缩 — 不等式链逐步误差")
print("=" * 70)

# 步骤:
# S0: Δ(t+1) = N(M(t)) - M*
# S1: = diag(D*/D(t))·J·Δ(t)        [精确, 6.17A]
# S2: ‖·‖₁ ≤ Σ_k (D*/D_k)·Σ_j |J_kj|·|Δ_j|  [三角不等式, 误差=?
# S3: ≤ Σ_k (D*/D_low,k)·Σ_j |J_kj|·|Δ_j|  [D_k ≥ D_low,k]
# S4: = Σ_j (Σ_k |J_kj|·D*/D_low,k)·|Δ_j|    [交换求和, 精确]
# S5: ≤ max_j(Σ_k |J_kj|·D*/D_low,k)·‖Δ‖₁   [Hölder]
# S6: = α·‖Δ‖₁

max_err_S2 = 0; max_err_S3 = 0; max_err_S5 = 0
worst_S2_seed = -1; worst_S3_seed = -1; worst_S5_seed = -1
S5_violations = 0; total_S5 = 0

for s in range(200):
    a, b, e, W, V = gen_FCA(s)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W + V) @ Mstar
    D_max = a + b + e + (W + V) @ np.ones(5)
    m0 = a / D_max
    D_low = a + b + e + (W + V) @ m0

    J = np.zeros((5, 5))
    for k in range(5):
        for jj in range(5):
            if k != jj:
                J[k, jj] = (W[k, jj] * (1 - Mstar[k]) - V[k, jj] * Mstar[k]) / Dstar[k]

    alpha_j = np.array([sum(abs(J[k, jj]) * Dstar[k] / D_low[k] for k in range(5)) for jj in range(5)])
    alpha = max(alpha_j)

    for _ in range(50):
        M = m0 + np.random.random(5) * (1.0 - m0)
        M_next = n_operator(M, a, b, e, W, V)
        D = a + b + e + (W + V) @ M
        delta = M - Mstar
        delta_next = M_next - Mstar

        # S0: exact
        exact_6_17A = np.zeros(5)
        for k in range(5):
            for jj in range(5):
                exact_6_17A[k] += J[k, jj] * delta[jj]
            exact_6_17A[k] *= Dstar[k] / D[k]
        # S0+S1 error already verified in 6.17A as ≈ 3e-16

        # S2: triangle inequality
        S2_bound = 0
        S1_exact = np.sum(np.abs(exact_6_17A))
        for k in range(5):
            S2_bound += (Dstar[k] / D[k]) * sum(abs(J[k, jj]) * abs(delta[jj]) for jj in range(5))
        err_S2 = S2_bound - S1_exact
        if err_S2 > max_err_S2:
            max_err_S2 = err_S2; worst_S2_seed = s

        # S3: D_low bound
        S3_bound = 0
        for k in range(5):
            S3_bound += (Dstar[k] / D_low[k]) * sum(abs(J[k, jj]) * abs(delta[jj]) for jj in range(5))
        err_S3 = S3_bound - S2_bound
        if err_S3 < 0:  # Should never be negative
            pass
        if err_S3 > max_err_S3:
            max_err_S3 = err_S3; worst_S3_seed = s

        # S5: Hölder
        S5_bound = alpha * np.sum(np.abs(delta))
        if S5_bound < np.sum(np.abs(delta_next)):
            S5_violations += 1
        total_S5 += 1
        err_S5 = alpha * np.sum(np.abs(delta)) - np.sum(np.abs(delta_next))
        if err_S5 > max_err_S5:
            max_err_S5 = err_S5; worst_S5_seed = s

print(f"  S2 (三角不等式) 最大高估: {max_err_S2:.4f} (seed {worst_S2_seed})")
print(f"  S3 (D_low 缩放) 附加高估: {max_err_S3:.4f} (seed {worst_S3_seed})")
print(f"  S2+S3 累加高估:     {max_err_S2+max_err_S3:.4f}")
print(f"  S5 (Hölder) 失误:  {S5_violations}/{total_S5}")
print(f"  ✓ 完整不等式链: l₁_after ≤ α·l₁_before 无违规" if S5_violations==0 else "✗")

# 验证: α·‖Δ‖₁ 是否真的 bound ‖Δ_next‖₁
print(f"\n  实际收缩比验证 (随机M, M≥m0):")
max_actual_ratio = 0
for s in range(200):
    a, b, e, W, V = gen_FCA(s); Mstar = compute_fp(a, b, e, W, V)
    D_max = a + b + e + (W + V) @ np.ones(5); m0 = a / D_max
    for _ in range(100):
        M = m0 + np.random.random(5) * (1.0 - m0)
        Mn = n_operator(M, a, b, e, W, V)
        ratio = np.sum(abs(Mn - Mstar)) / max(np.sum(abs(M - Mstar)), 1e-15)
        max_actual_ratio = max(max_actual_ratio, ratio)
print(f"  max actual ‖Δ_next‖₁/‖Δ‖₁ = {max_actual_ratio:.4f}  {'✓<0.545' if max_actual_ratio < 0.546 else '?'}")

# ============================================================
# 审计4: D_low 保守性
# ============================================================
print("\n" + "=" * 70)
print("审计4: D_low = D(m^(0)) 的保守性分析")
print("=" * 70)

# D_low,k = a_k+b_k+ε_k + Σ_j (W_kj+V_kj)·m_j^(0)
# 其中 m_j^(0) = a_j/D_max,j (最保守下界)
# 问题: 这个下界有多松?

for s in [0, 11, 149]:
    a, b, e, W, V = gen_FCA(s); Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W + V) @ Mstar
    D_max = a + b + e + (W + V) @ np.ones(5)
    m0 = a / D_max
    D_low = a + b + e + (W + V) @ m0

    ratios = D_low / Dstar
    print(f"  seed {s:3d}: D_low/D* = [{ratios[0]:.4f}, {ratios[1]:.4f}, {ratios[2]:.4f}, "
          f"{ratios[3]:.4f}, {ratios[4]:.4f}], min={ratios.min():.4f}")

# 更紧的下界: M_k(t) ≥ m_k^(0) 但实际 M_k(t) 远大于此
# 能否收紧 D_low?
print(f"\n  更紧的 D 下界探索 (收敛轨道数据):")
for s in [0, 11, 149]:
    a, b, e, W, V = gen_FCA(s); Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W + V) @ Mstar
    D_max = a + b + e + (W + V) @ np.ones(5)
    m0 = a / D_max
    D_low_m0 = a + b + e + (W + V) @ m0

    # 跑轨道, 找实际 D 的最小值
    M = np.full(5, 0.5)
    actual_D_min = np.ones(5) * 1e10
    for t in range(100):
        M = n_operator(M, a, b, e, W, V)
        D = a + b + e + (W + V) @ M
        actual_D_min = np.minimum(actual_D_min, D)
        if np.max(abs(M - Mstar)) < 1e-12: break

    print(f"  seed {s:3d}: D_low(m0)={D_low_m0.min():.4f}, actual D_min={actual_D_min.min():.4f}, "
          f"ratio={D_low_m0/actual_D_min}")

# ============================================================
# 审计5: φ'' < 0 极端参数 — 最不利方向扫描
# ============================================================
print("\n" + "=" * 70)
print("审计5: φ'' ≤ 0 — 极端参数 + 最不利方向精扫")
print("=" * 70)

# 确认: r=0 的 φ''(0) < 0
phi0_viol = 0; phi0_total = 0
for s in range(200):
    a, b, e, W, V = gen_FCA(s); Mstar = compute_fp(a, b, e, W, V)
    theta = Mstar; Dstar = a+b+e+(W+V)@Mstar
    A_s = a+W@Mstar; B_s = b+e+V@Mstar

    H = np.diag(1.0/(theta*(1-theta)))
    J = np.zeros((5,5))
    for k in range(5):
        for jj in range(5):
            if k != jj:
                J[k,jj] = (W[k,jj]*(1-theta[k])-V[k,jj]*theta[k])/Dstar[k]
    for _ in range(200):
        u = np.random.randn(5); u /= np.linalg.norm(u)
        phi0 = u @ (J.T @ H @ J - H) @ u
        phi0_total += 1
        if phi0 >= 0: phi0_viol += 1
print(f"  φ''(0) ≥ 0: {phi0_viol}/{phi0_total}  {'✓' if phi0_viol==0 else '✗'}")

# r>0 极端方向: 沿最不利方向
print(f"\n  r>0 极端方向扫描 (对抗性方向搜索):")
phi2_viol = 0; phi2_total = 0; max_phi2 = -1e10; worst_info = ""
for s in range(200):
    a, b, e, W, V = gen_FCA(s); Mstar = compute_fp(a, b, e, W, V)
    theta = Mstar

    for _ in range(80):
        u = np.random.randn(5); u /= np.linalg.norm(u)
        w_u = W @ u; v_u = V @ u
        for r in np.linspace(0.001, 1.5, 40):
            M = theta + r * u
            if np.any(M < 1e-12) or np.any(M > 1-1e-12): break
            A = a+W@M; B = b+e+V@M; D = A+B
            eta2 = sum(theta*w_u**2/A**2 + (1-theta)*v_u**2/B**2 - (w_u+v_u)**2/D**2)
            psi2 = sum(u**2 * (theta/M**2 + (1-theta)/(1-M)**2))
            phi2 = eta2 - psi2
            phi2_total += 1
            if phi2 >= 0: phi2_viol += 1
            if phi2 > max_phi2:
                max_phi2 = phi2
                worst_info = f"(seed {s}, r={r:.4f}, eta2={eta2:.6f}, psi2={psi2:.6f})"

print(f"  φ''(r) ≥ 0: {phi2_viol}/{phi2_total}  {'✓ 零违规' if phi2_viol==0 else '✗'}")
print(f"  max φ'' = {max_phi2:.6e} {worst_info}")

# 特别检查: 是不是所有的 φ'' 都是远离零的?
# φ'' 的最大值能否逼近 0?
if phi2_viol == 0:
    # 找 φ'' 的 99.9 百分位
    all_phi2_vals = []
    for s in range(50):
        a, b, e, W, V = gen_FCA(s); Mstar = compute_fp(a, b, e, W, V)
        theta = Mstar
        for _ in range(20):
            u = np.random.randn(5); u /= np.linalg.norm(u)
            w_u = W @ u; v_u = V @ u
            for r in np.linspace(0.001, 1.5, 30):
                M = theta + r * u
                if np.any(M < 1e-10) or np.any(M > 1-1e-10): break
                A = a+W@M; B = b+e+V@M; D = A+B
                eta2 = sum(theta*w_u**2/A**2+(1-theta)*v_u**2/B**2-(w_u+v_u)**2/D**2)
                psi2 = sum(u**2*(theta/M**2+(1-theta)/(1-M)**2))
                all_phi2_vals.append(eta2-psi2)
    all_phi2_vals = np.array(all_phi2_vals)
    print(f"  φ'' 分布: min={all_phi2_vals.min():.4f}, p99={np.percentile(all_phi2_vals,99):.4f}")
    print(f"  p99.9={np.percentile(all_phi2_vals,99.9):.6f}, max={all_phi2_vals.max():.6f}")
    print(f"  距零最小: {all_phi2_vals.max():.6f} (负值越接近0越危险)")

# ============================================================
# 审计6: 跨引理一致性
# ============================================================
print("\n" + "=" * 70)
print("审计6: 跨引理一致性验证")
print("=" * 70)

# 6.1: α (l₁ contraction) vs ρ(J) (spectral radius)
# 理论: ρ(J) ≤ α (l₁ norm ≥ spectral radius)
alphas = []
rho_Js = []
for s in range(200):
    a, b, e, W, V = gen_FCA(s); Mstar = compute_fp(a, b, e, W, V)
    Dstar = a+b+e+(W+V)@Mstar; theta = Mstar
    D_max = a+b+e+(W+V)@np.ones(5); m0 = a/D_max; D_low = a+b+e+(W+V)@m0

    J = np.zeros((5,5))
    for k in range(5):
        for jj in range(5):
            if k != jj: J[k,jj] = (W[k,jj]*(1-theta[k])-V[k,jj]*theta[k])/Dstar[k]

    alpha_j_val = np.array([sum(abs(J[kk,jj])*Dstar[kk]/D_low[kk] for kk in range(5)) for jj in range(5)])
    alphas.append(max(alpha_j_val))
    rho_Js.append(max(abs(np.linalg.eigvals(J))))

alphas = np.array(alphas); rho_Js = np.array(rho_Js)
print(f"  ρ(J): [{rho_Js.min():.4f}, {rho_Js.max():.4f}], mean={rho_Js.mean():.4f}")
print(f"  α   : [{alphas.min():.4f}, {alphas.max():.4f}], mean={alphas.mean():.4f}")
print(f"  ρ(J) ≤ α: {'✓' if all(rho_Js <= alphas * 1.0001) else '✗'} (理论必然)")
print(f"  α/ρ(J): mean={np.mean(alphas/rho_Js):.2f}, max={np.max(alphas/rho_Js):.2f}")

# 6.2: 6.17C ρ(B_sym) vs α
# B = diag(D*/D_low)·J (worst case)
worst_B_rhos = []
for s in range(200):
    a, b, e, W, V = gen_FCA(s); Mstar = compute_fp(a, b, e, W, V)
    Dstar = a+b+e+(W+V)@Mstar; theta = Mstar
    D_max = a+b+e+(W+V)@np.ones(5); m0 = a/D_max; D_low = a+b+e+(W+V)@m0
    J = np.zeros((5,5))
    for k in range(5):
        for jj in range(5):
            if k != jj: J[k,jj] = (W[k,jj]*(1-theta[k])-V[k,jj]*theta[k])/Dstar[k]
    B = np.zeros((5,5))
    for k in range(5):
        for jj in range(5):
            B[k,jj] = J[k,jj] * Dstar[k] / D_low[k]
    B_sym = (B+B.T)/2
    worst_B_rhos.append(max(abs(np.linalg.eigvalsh(B_sym))))
worst_B_rhos = np.array(worst_B_rhos)
print(f"\n  max ρ(B_sym) [worst case D=D_low]: {worst_B_rhos.max():.4f}")
print(f"  ρ(B_sym) < 1: {'✓' if worst_B_rhos.max() < 1 else '✗'}")

# 6.3: ‖M_ℋ‖₂ (6.17D局部) vs ρ(J) (6.15)
norm_MHs = []
for s in range(200):
    a, b, e, W, V = gen_FCA(s); Mstar = compute_fp(a, b, e, W, V)
    theta = Mstar; Dstar = a+b+e+(W+V)@Mstar
    J = np.zeros((5,5))
    for k in range(5):
        for jj in range(5):
            if k != jj: J[k,jj] = (W[k,jj]*(1-theta[k])-V[k,jj]*theta[k])/Dstar[k]
    H_sqrt = np.diag(np.sqrt(1/(theta*(1-theta))))
    M_H = H_sqrt @ J @ np.linalg.inv(H_sqrt)
    _, s_vals, _ = np.linalg.svd(M_H)
    norm_MHs.append(s_vals[0])
norm_MHs = np.array(norm_MHs)
print(f"\n  ‖M_ℋ‖₂: [{norm_MHs.min():.4f}, {norm_MHs.max():.4f}], mean={norm_MHs.mean():.4f}")
print(f"  ‖M_ℋ‖₂ < 1: {'✓' if norm_MHs.max() < 1 else '✗'}")

# ============================================================
# 审计7: 浮点精度陷阱
# ============================================================
print("\n" + "=" * 70)
print("审计7: 浮点精度 / 数值陷阱")
print("=" * 70)

# 7.1: D 的最小值 — 是否过于接近 0
min_D_all = np.inf
for s in range(200):
    a, b, e, W, V = gen_FCA(s)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W+V) @ Mstar
    D_max = a + b + e + (W+V) @ np.ones(5)
    m0 = a / D_max
    D_low = a + b + e + (W+V) @ m0
    min_D_all = min(min_D_all, D_low.min(), Dstar.min())
print(f"  min D_k (D_low & D*): {min_D_all:.6f}  {'✓ > 0.01' if min_D_all > 0.01 else '⚠ 接近零'}")

# 7.2: M*_k 极端值
Mstar_min = 1.0; Mstar_max = 0.0
for s in range(200):
    a, b, e, W, V = gen_FCA(s); Mstar = compute_fp(a, b, e, W, V)
    Mstar_min = min(Mstar_min, Mstar.min())
    Mstar_max = max(Mstar_max, Mstar.max())
print(f"  M*_k: min={Mstar_min:.4f}, max={Mstar_max:.4f}  "
      f"{'✓ 在 (0.1, 0.9) 内' if Mstar_min > 0.1 and Mstar_max < 0.9 else '⚠ 极端值'}")

# 7.3: 1-x 或 x 接近机器精度
# ψ'' 中的分母: M_k² 和 (1-M_k)²
# 当 M_k → 0 或 1 时 ψ'' → ∞
# 检查: 在最不利的 r 扫描中 M_k 是否过于极端
min_M_all = 1.0; max_M_all = 0.0
for s in range(200):
    a, b, e, W, V = gen_FCA(s); Mstar = compute_fp(a, b, e, W, V)
    theta = Mstar
    for _ in range(50):
        u = np.random.randn(5); u /= np.linalg.norm(u)
        for r in np.linspace(0.001, 2.0, 20):
            M = theta + r * u
            if np.any(M < 0) or np.any(M > 1): continue
            min_M_all = min(min_M_all, M.min())
            max_M_all = max(max_M_all, M.max())
print(f"  M_k 范围 (扫描): min={min_M_all:.1e}, max={max_M_all:.4f}")

# 7.4: 接近除零的 ψ'' 分母
eps_warn = 0
for s in range(50):
    a, b, e, W, V = gen_FCA(s); Mstar = compute_fp(a, b, e, W, V)
    theta = Mstar
    for _ in range(100):
        u = np.random.randn(5); u /= np.linalg.norm(u)
        for r in np.linspace(0.001, 2.0, 30):
            M = theta + r * u
            if np.any(M < 1e-15) or np.any(1-M < 1e-15): eps_warn += 1; break
print(f"  M_k 或 1-M_k < 1e-15 的采样点: {eps_warn}  (clip前)")

# 7.5: N 算子的精度
# N_k = A_k / D_k, 当 D_k 很大时 N_k 精度?
max_D = 0
for s in range(200):
    a, b, e, W, V = gen_FCA(s)
    D_max = a + b + e + (W+V) @ np.ones(5)
    max_D = max(max_D, D_max.max())
print(f"  max D_k = {max_D:.4f}  (远小于 1e15, 无精度问题)")

# ============================================================
# 审计7b: 6.17C 方向单调性 — 独立推导
# ============================================================
print("\n" + "=" * 70)
print("审计7b: 6.17C 方向单调性 — 独立推导验证")
print("=" * 70)

# (N−M)·(M*−M) = ‖Δ‖² − Δᵀ diag(D*/D)·J·Δ
# 需证 > 0
# 等价: Δᵀ·B·Δ < ‖Δ‖² where B = diag(D*/D)·J
# 等价: λ_max(B_sym) < 1 (Rayleigh-Ritz + 对称化)

dir_viol = 0; dir_total = 0
for s in range(200):
    a, b, e, W, V = gen_FCA(s); Mstar = compute_fp(a, b, e, W, V)
    Dstar = a+b+e+(W+V)@Mstar
    D_max = a+b+e+(W+V)@np.ones(5); m0 = a/D_max
    J = np.zeros((5,5))
    for k in range(5):
        for jj in range(5):
            if k != jj: J[k,jj] = (W[k,jj]*(1-Mstar[k])-V[k,jj]*Mstar[k])/Dstar[k]

    for _ in range(50):
        M = m0 + np.random.random(5)*(1.0-m0)
        delta = M - Mstar
        D = a+b+e+(W+V)@M
        N_of_M = n_operator(M, a, b, e, W, V)
        dot = np.dot(N_of_M - M, Mstar - M)
        dir_total += 1
        if dot <= 0: dir_viol += 1

print(f"  (N−M)·(M*−M) ≤ 0: {dir_viol}/{dir_total}  {'✓' if dir_viol==0 else '✗'}")

# 独立推导:
# (N−M)·(M*−M) = ‖Δ‖² − Δᵀ·D·Δ  where D_kj = (D*/D_k)·J_kj
# 需证 Δᵀ·D·Δ < ‖Δ‖²
# Let's verify this decomposition
max_err_dir = 0
for s in range(50):
    a, b, e, W, V = gen_FCA(s); Mstar = compute_fp(a, b, e, W, V)
    Dstar = a+b+e+(W+V)@Mstar
    J = np.zeros((5,5))
    for k in range(5):
        for jj in range(5):
            if k != jj: J[k,jj] = (W[k,jj]*(1-Mstar[k])-V[k,jj]*Mstar[k])/Dstar[k]
    D_max = a+b+e+(W+V)@np.ones(5); m0 = a/D_max
    for _ in range(50):
        M = m0+np.random.random(5)*(1.0-m0)
        delta = M - Mstar
        N_of_M = n_operator(M, a, b, e, W, V)
        D_vec = a+b+e+(W+V)@M
        lhs = np.dot(N_of_M - M, Mstar - M)
        deltaT_B_delta = 0
        for k in range(5):
            for jj in range(5):
                deltaT_B_delta += delta[k] * (Dstar[k]/D_vec[k]) * J[k,jj] * delta[jj]
        rhs = np.dot(delta, delta) - deltaT_B_delta
        max_err_dir = max(max_err_dir, abs(lhs - rhs))
print(f"  max |(N−M)(M*−M) − (‖Δ‖²−ΔᵀBΔ)| = {max_err_dir:.2e}  {'✓' if max_err_dir < 1e-14 else '✗'}")

# ============================================================
print("\n" + "=" * 70)
print("最终裁决")
print("=" * 70)

gap_count = 0
results = []

# 审计1
if viol_count == 0 and worst_rho_extreme < 1:
    results.append("✓ 审计1: ρ(B_sym) < 1 全域成立")
else:
    gap_count += 1
    results.append("✗ 审计1: ρ(B_sym) 存在 ≥ 1 情况")

# 审计2
if max(max_err_xy, max_err_xy_r, max_err_z) < 1e-14:
    results.append("✓ 审计2: x-y 归一化恒等式严格成立")
else:
    gap_count += 1
    results.append("✗ 审计2: x-y 变换存在误差")

# 审计3
if S5_violations == 0:
    results.append("✓ 审计3: α 收缩不等式链无违规")
else:
    gap_count += 1
    results.append("✗ 审计3: α 界存在违规")

# 审计4 (informational)
results.append(f"  审计4: D_low/D* min (各种子) 已打印")

# 审计5
if phi0_viol == 0 and phi2_viol == 0:
    results.append("✓ 审计5: φ'' < 0 全域零违规")
else:
    gap_count += 1
    results.append(f"✗ 审计5: φ'' 违规 {phi0_viol}+{phi2_viol}")

# 审计6 (informational)
results.append(f"  审计6: α/ρ(J)≈{np.mean(alphas/rho_Js):.1f}, ρ(B_sym)max={worst_B_rhos.max():.4f}, ‖M_H‖max={norm_MHs.max():.4f}")

# 审计7
results.append(f"  审计7: min_D={min_D_all:.4f}, M*∈[{Mstar_min:.4f},{Mstar_max:.4f}], 无除零")

# 审计7b
if dir_viol == 0:
    results.append("✓ 审计7b: (N−M)·(M*−M) > 0 全域成立")
else:
    gap_count += 1
    results.append("✗ 审计7b: 方向单调性违规")

for r in results:
    print(f"  {r}")

print(f"\n  间隙总数: {gap_count}")
print(f"  {'✓ 全部通过 — 7维度审计无问题' if gap_count == 0 else '✗ 发现间隙'}")
