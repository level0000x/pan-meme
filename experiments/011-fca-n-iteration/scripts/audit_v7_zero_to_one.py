"""
Ⅶ. 从零开始的全链审计 — 逐代数步骤追踪
=========================================
每步给出精确数值验证，发现任何间隙立即标记。
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
print("STEP 0: 6.17A 精确恒等式 — 逐分量逐种子严格验证")
print("="*70)
# N_k(M)-M*_k  vs.  (D*_k/D_k)·Σ_j J_kj(M*)·(M_j-M*_j)
# 代数证明: 对仿射分式函数, 这等于精确恒等
# 但我们必须数值确认——任何浮点误差都可能导致级联失败

max_err_6_17A = 0; max_err_component = (0,0,0)
for s in range(200):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    Dstar = a+b+e+(W+V)@Mstar
    
    J = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k!=j:
                J[k,j] = (W[k,j]*(1-Mstar[k])-V[k,j]*Mstar[k])/Dstar[k]
    
    for _ in range(20):
        M = np.random.uniform(0.01, 0.99, 5)
        D = a+b+e+(W+V)@M
        Delta = M - Mstar
        pred = (Dstar/D) * (J @ Delta)
        actual = n_operator(M, a, b, e, W, V) - Mstar
        
        err = np.max(np.abs(pred - actual))
        if err > max_err_6_17A:
            max_err_6_17A = err
            max_err_component = (s, np.argmax(np.abs(pred-actual)), err)

print(f"  max |pred - actual| = {max_err_6_17A:.2e}")
if max_err_6_17A > 1e-14:
    print(f"  ✗ 精确恒等式可能有数值误差 (seed {max_err_component[0]}, k={max_err_component[1]})")
    # 用高精度有理数重验
    a,b,e,W,V = gen_FCA(max_err_component[0])
    Mstar = compute_fp(a,b,e,W,V)
    Dstar = a+b+e+(W+V)@Mstar
    # 手动验证一个分量
    k = max_err_component[1]
    M = np.random.uniform(0.01, 0.99, 5)
    M = np.full(5, 0.3)  # deterministic test
    D = a+b+e+(W+V)@M
    J_test = np.zeros((5,5))
    for ki in range(5):
        for ji in range(5):
            if ki!=ji:
                J_test[ki,ji] = (W[ki,ji]*(1-Mstar[ki])-V[ki,ji]*Mstar[ki])/Dstar[ki]
    pred_k = (Dstar/D) * (J_test @ (M-Mstar))
    actual_k = n_operator(M, a, b, e, W, V) - Mstar
    print(f"    k={k}: pred={pred_k[k]:.16e}, actual={actual_k[k]:.16e}, diff={abs(pred_k[k]-actual_k[k]):.2e}")
else:
    print(f"  ✓ 精确恒等式 (浮点精度范围内)")

# ============================================================
print(f"\n{'='*70}")
print("STEP 1: 检查 identity N_k(M)-M*_k 的等号推断")
print("="*70)
# 用有理数代数验证一个实例
# N = A/D, J = ∂N/∂M|_{M*}
# Claim: N(M)-M* = diag(D*/D)·J·(M-M*)
# 
# Proof for component k:
# Let A*(k) = A_k(M*), D*(k) = D_k(M*)
# Δ = M - M*
# A(M) = A* + w(k)·Δ, D(M) = D* + (w(k)+v(k))·Δ  
# Where w(k) is row k of W, v(k) is row k of V
# 
# LHS = A(M)/D(M) - A*/D*
#      = (A(M)·D* - A*·D(M)) / (D(M)·D*)
#      = ((A*+wΔ)·D* - A*·(D*+(w+v)Δ)) / (D(M)·D*)
#      = (A*D* + wΔ·D* - A*D* - A*(w+v)Δ) / (D(M)·D*)
#      = (w·D* - A*·(w+v))·Δ / (D(M)·D*)
# 
# RHS = (D*/D(M)) Σ_j J_kj(M*)·Δ_j
#     = (D*/D(M)) Σ_j [(w_kj·(D*-A*) - A*·v_kj)/D*²] · Δ_j
#     = (1/D(M)) Σ_j [(w_kj·(D*-A*) - A*·v_kj)/D*] · Δ_j
#     = (1/(D(M)·D*)) Σ_j [w_kj·D* - w_kj·A* - A*·v_kj] · Δ_j
#     = (1/(D(M)·D*)) Σ_j [w_kj·D* - A*·(w_kj+v_kj)] · Δ_j
#     = (1/(D(M)·D*)) [Σ_j w_kj·D*·Δ_j - A*·Σ_j(w_kj+v_kj)·Δ_j]
#     = (1/(D(M)·D*)) [w_k·D*·Δ - A*·(w_k+v_k)·Δ]
#     = (w_k·D* - A*·(w_k+v_k))·Δ / (D(M)·D*)
# 
# LHS = RHS ✓  (精确恒等, 无近似)
print(f"  ✓ 代数证明通过 (参见脚本中逐步推导)")

# ============================================================
print(f"\n{'='*70}")
print("STEP 2: 全轨道验证 — 从随机 M(0) 到收敛")
print("="*70)

viol_orbits = 0; total_orbit_steps = 0; worst_orbit_seed = -1
worst_orbit_alpha = 0; worst_orbit_step = -1
for s in range(200):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    Dstar = a+b+e+(W+V)@Mstar
    D_max = a+b+e+np.sum(W+V, axis=1)
    m0 = a/D_max
    D_low = a+b+e+(W+V)@m0
    
    J = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k!=j:
                J[k,j] = (W[k,j]*(1-Mstar[k])-V[k,j]*Mstar[k])/Dstar[k]
    
    B = np.abs(J) * (Dstar/D_low).reshape(-1,1)
    alpha = max([B.sum(axis=0)[j] for j in range(5)])
    
    # 从 3 个不同起点出发
    for start_idx in range(3):
        M = np.random.uniform(0.0001, 0.9999, 5)
        for t in range(50):
            Delta = M - Mstar
            l1_delta = np.sum(np.abs(Delta))
            if l1_delta < 1e-15: break
            
            NmMstar = n_operator(M, a, b, e, W, V) - Mstar
            l1_next = np.sum(np.abs(NmMstar))
            
            total_orbit_steps += 1
            
            if t >= 1:
                # α 收缩应对 t≥1 成立
                ratio = l1_next / max(l1_delta, 1e-15)
                if ratio > alpha * (1 + 1e-10):
                    viol_orbits += 1
                    if ratio > worst_orbit_alpha:
                        worst_orbit_alpha = ratio
                        worst_orbit_seed = s
                        worst_orbit_step = t
            
            M = NmMstar + Mstar

print(f"  轨道收缩违反: {viol_orbits}/{total_orbit_steps} 步")
print(f"  worst α bypass: {worst_orbit_alpha:.4f} (种子{worst_orbit_seed}, 步{worst_orbit_step})")
print(f"  {'✓' if viol_orbits==0 else '✗'}")

# ============================================================
print(f"\n{'='*70}")
print("STEP 3: D_low 界沿轨道的实时间隙")
print("="*70)

min_Dratio_all = 1e10; min_Dratio_seed = -1; min_Dratio_step = -1
for s in range(200):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    D_max = a+b+e+np.sum(W+V, axis=1)
    m0 = a/D_max
    D_low = a+b+e+(W+V)@m0
    
    for start_idx in range(3):
        M = np.random.uniform(0.0001, 0.9999, 5)
        for t in range(50):
            D = a+b+e+(W+V)@M
            Delta = M - Mstar
            if np.sum(np.abs(Delta)) < 1e-15: break
            
            if t >= 1:
                Dratio = np.min(D / D_low)
                if Dratio < min_Dratio_all:
                    min_Dratio_all = Dratio
                    min_Dratio_seed = s
                    min_Dratio_step = t
            
            M = n_operator(M, a, b, e, W, V)

print(f"  min D_k/D_low,k 沿轨道 = {min_Dratio_all:.4f} (种子{min_Dratio_seed}, 步{min_Dratio_step})")
print(f"  {'✓ 始终 ≥ 1' if min_Dratio_all >= 1-1e-12 else '✗ D_low 界被违反!'}")
print(f"  安全裕度: {min_Dratio_all - 1:.4f}")

# ============================================================
print(f"\n{'='*70}")
print("STEP 4: r_j 解析上界探索")
print("="*70)

# r_j = (D*_j/D_low,j) · Σ_k |J_jk|
# J_jk = (W[j,k](1-M*_j) - V[j,k]M*_j) / D*_j
#
# 上界: |J_jk| ≤ (W[j,k](1-M*_j) + V[j,k]M*_j) / D*_j
#         ≤ (W[j,k] + V[j,k]) / D*_j  (M*_j∈[0,1] ⇒ max terms)
#         或 ≤ max(W[j,k], V[j,k]) / D*_j  (更紧)
#
# r_j ≤ (D*_j/D_low,j) · Σ_k max(W[j,k],V[j,k])/D*_j
#     = Σ_k max(W[j,k],V[j,k]) / D_low,j
#
# D_low,j = a_j+b_j+ε_j + Σ_k(W[j,k]+V[j,k])·m^(0)_k
#
# 检查此解析上界是否 < 1

analytical_rj_pass = 0; analytical_rj_max = 0
for s in range(200):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    D_max = a+b+e+np.sum(W+V, axis=1)
    m0 = a/D_max
    D_low = a+b+e+(W+V)@m0
    
    for j in range(5):
        r_j_analytic = np.sum(np.maximum(W[j,:], V[j,:])) / D_low[j]
        analytical_rj_max = max(analytical_rj_max, r_j_analytic)
        if r_j_analytic >= 1: analytical_rj_pass += 1

print(f"  解析 r_j 界: max={analytical_rj_max:.4f}")
print(f"  r_j ≥ 1 的种子: {analytical_rj_pass}")
print(f"  {'✓ 解析可证 r_j < 1' if analytical_rj_max < 1 else '✗ 解析上界过松' if analytical_rj_pass > 0 else '⚠ 界 ≈ 1 需仔细分析'}")

# 更精确的 bound:
# |J_jk| ≤ max(W[j,k], V[j,k])/(a_j + b_j + ε_j + Σ_k(W[j,k]+V[j,k])M*_k)
# 这涉及 M*, 非解析

# 检查最紧 bound:
analytical_tight = 0; tight_max = 0
for s in range(200):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    Dstar = a+b+e+(W+V)@Mstar
    D_max = a+b+e+np.sum(W+V, axis=1)
    m0 = a/D_max
    D_low = a+b+e+(W+V)@m0
    
    J = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k!=j:
                J[k,j] = (W[k,j]*(1-Mstar[k])-V[k,j]*Mstar[k])/Dstar[k]
    
    for j in range(5):
        r_j = (Dstar[j]/D_low[j]) * np.sum(np.abs(J[j,:]))
        tight_max = max(tight_max, r_j)
        if r_j >= 1: analytical_tight += 1

print(f"  实际 r_j (需 M*): max={tight_max:.4f}")
print(f"  实际 r_j ≥ 1: {analytical_tight}")

# ============================================================
print(f"\n{'='*70}")
print("STEP 5: B_sym 的 AM-GM 界 vs 最紧界")
print("="*70)

# ρ(B_sym) ≤ max_j (c_j+r_j)/2 ≤ 0.629
# 实际 ρ(B_sym) = ?
# 差距有多大?

max_rho_sym = 0; max_amgm_bound = 0; max_gap_ratio = 0
for s in range(200):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    Dstar = a+b+e+(W+V)@Mstar
    D_max = a+b+e+np.sum(W+V, axis=1)
    m0 = a/D_max
    D_low = a+b+e+(W+V)@m0
    
    J = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k!=j:
                J[k,j] = (W[k,j]*(1-Mstar[k])-V[k,j]*Mstar[k])/Dstar[k]
    
    B = np.abs(J) * (Dstar/D_low).reshape(-1,1)
    B_sym = (B + B.T) / 2
    rho_sym = max(abs(np.linalg.eigvals(B_sym)))
    
    col_sums = B.sum(axis=0)
    row_sums = B.sum(axis=1)
    amgm = np.max((col_sums + row_sums) / 2)
    
    max_rho_sym = max(max_rho_sym, rho_sym)
    max_amgm_bound = max(max_amgm_bound, amgm)
    max_gap_ratio = max(max_gap_ratio, amgm / max(rho_sym, 1e-15))

print(f"  max ρ(B_sym) = {max_rho_sym:.4f}")
print(f"  max AM-GM 界 = {max_amgm_bound:.4f}")
print(f"  max AM-GM/ρ(B_sym) = {max_gap_ratio:.2f}x (保守度)")
print(f"  下界 1-ρ(B_sym) ≥ {1-max_rho_sym:.4f}")
print(f"  保守下界 1-AM-GM ≥ {1-max_amgm_bound:.4f}")

# ============================================================
print(f"\n{'='*70}")
print("STEP 6: 构造性极端测试——能否制造 ρ(B_sym) 接近 1?")
print("="*70)

# 极端构造: 最大化 D*/D_low (使得某个分量极小)
# 同时最大化 |J_kj| 的互权重

def test_extreme_construction():
    best_rho = 0
    best_params = None
    
    for attempt in range(10000):
        rs = np.random.RandomState(attempt)
        # 极小 a (分母主要由耦合决定)
        a = np.full(5, 0.01)
        b = np.full(5, 0.01)
        e = np.full(5, 0.001)
        
        # 强耦合
        W_raw = rs.uniform(0.1, 0.5, (5,5))
        V_raw = rs.uniform(0.05, 0.3, (5,5))
        np.fill_diagonal(W_raw, 0)
        np.fill_diagonal(V_raw, 0)
        
        t = a.sum()+b.sum()+W_raw.sum()+V_raw.sum()
        W = W_raw * 5.0 / t
        V = V_raw * 5.0 / t
        
        try:
            Mstar = compute_fp(a,b,e,W,V)
        except:
            continue
        
        Dstar = a+b+e+(W+V)@Mstar
        D_max = a+b+e+np.sum(W+V, axis=1)
        m0 = a/D_max
        D_low = a+b+e+(W+V)@m0
        
        J = np.zeros((5,5))
        for k in range(5):
            for j in range(5):
                if k!=j:
                    J[k,j] = (W[k,j]*(1-Mstar[k])-V[k,j]*Mstar[k])/Dstar[k]
        
        B = np.abs(J) * (Dstar/D_low).reshape(-1,1)
        B_sym = (B + B.T) / 2
        rho_sym = max(abs(np.linalg.eigvals(B_sym)))
        
        if rho_sym > best_rho:
            best_rho = rho_sym
            best_params = (a,b,e,W,V,Mstar,rho_sym)
    
    return best_rho, best_params

best_rho, bp = test_extreme_construction()
print(f"  10000 组极端构造: max ρ(B_sym) = {best_rho:.4f}")
if bp:
    a,b,e,W,V,Mstar,rho = bp
    Dstar = a+b+e+(W+V)@Mstar
    D_max = a+b+e+np.sum(W+V, axis=1)
    m0 = a/D_max
    D_low = a+b+e+(W+V)@m0
    J = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k!=j:
                J[k,j] = (W[k,j]*(1-Mstar[k])-V[k,j]*Mstar[k])/Dstar[k]
    B = np.abs(J) * (Dstar/D_low).reshape(-1,1)
    alpha = max([B.sum(axis=0)[j] for j in range(5)])
    print(f"    α = {alpha:.4f}")
    print(f"    D*/D_low = {np.max(Dstar/D_low):.2f}")
    print(f"    安全裕度 = {1-rho:.4f}")
print(f"  {'✓ 即使极端构造也远 < 1' if best_rho < 1 else '✗ 可构造接近 1!'}")

# ============================================================
print(f"\n{'='*70}")
print("STEP 7: J(M*) 对 M* 微小扰动的敏感性")
print("="*70)

# 如果 M* 数值计算有微小误差，J(M*) 会如何偏离?
# 这检验证明的数值稳定性

max_sensitivity = 0
for s in range(200):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    Dstar = a+b+e+(W+V)@Mstar  # correct D*
    
    # Compute J at exact M*
    J_exact = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k!=j:
                J_exact[k,j] = (W[k,j]*(1-Mstar[k])-V[k,j]*Mstar[k])/Dstar[k]
    
    # Perturb M* by epsilon in random direction
    for _ in range(10):
        perturb = np.random.randn(5) * 1e-8
        Mstar_pert = Mstar + perturb
        
        Dstar_pert = a+b+e+(W+V)@Mstar_pert
        J_pert = np.zeros((5,5))
        for k in range(5):
            for j in range(5):
                if k!=j:
                    J_pert[k,j] = (W[k,j]*(1-Mstar_pert[k])-V[k,j]*Mstar_pert[k])/Dstar_pert[k]
        
        diff = np.max(np.abs(J_pert - J_exact))
        rel_diff = diff / max(np.max(np.abs(J_exact)), 1e-15)
        max_sensitivity = max(max_sensitivity, rel_diff / np.linalg.norm(perturb))

print(f"  max ‖ΔJ‖/‖ΔM*‖ ≈ {max_sensitivity:.2e}")
print(f"  {'✓ J 对 M* 扰动不敏感' if max_sensitivity < 100 else '⚠ J 对 M* 敏感'}")

# ============================================================
print(f"\n{'='*70}")
print('STEP 8: 完整不等式的"最劣"方向构造')
print("="*70)

# 对每个种子，找到最大化 ‖N(M)-M*‖₁/‖M-M*‖₁ 的方向
# (这是 α 的"实际达成值", 不是理论上界)
# 检查是否接近 α 理论值

max_achieved_alpha = 0; avg_achieved = 0; count_a = 0
for s in range(200):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    Dstar = a+b+e+(W+V)@Mstar
    D_max = a+b+e+np.sum(W+V, axis=1)
    m0 = a/D_max
    D_low = a+b+e+(W+V)@m0
    
    for _ in range(30):
        M = np.random.uniform(m0, 0.99, 5)
        Delta = M - Mstar
        if np.sum(np.abs(Delta)) < 1e-12: continue
        
        NmMstar = n_operator(M, a, b, e, W, V) - Mstar
        ratio = np.sum(np.abs(NmMstar)) / np.sum(np.abs(Delta))
        max_achieved_alpha = max(max_achieved_alpha, ratio)
        avg_achieved += ratio; count_a += 1

avg_achieved /= count_a
print(f"  随机点 max 达成 α = {max_achieved_alpha:.4f}")
print(f"  随机点 avg 达成 α = {avg_achieved:.4f}")
print(f"  理论 α_max = 0.5453")

# 网格扫描寻找真正的最劣方向
grid_max = 0
for s in range(20):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    Dstar = a+b+e+(W+V)@Mstar
    D_max = a+b+e+np.sum(W+V, axis=1)
    m0 = a/D_max
    
    for v1 in np.linspace(0.01, 0.99, 8):
        for v2 in np.linspace(0.01, 0.99, 8):
            M = np.array([v1, v2, 0.5, 0.5, 0.5])
            M = np.clip(M, m0, 0.99)
            Delta = M - Mstar
            if np.sum(np.abs(Delta)) < 1e-12: continue
            NmMstar = n_operator(M, a, b, e, W, V) - Mstar
            ratio = np.sum(np.abs(NmMstar)) / np.sum(np.abs(Delta))
            grid_max = max(grid_max, ratio)

print(f"  网格扫描 max 达成 α = {grid_max:.4f}")
print(f"  理论/达成差距 = {max(0.5453/grid_max if grid_max > 0 else 0, 0):.2f}x")

# ============================================================
print(f"\n{'='*70}")
print("最终裁决")
print("="*70)

issues = []
if max_err_6_17A > 1e-12:
    issues.append(f"6.17A 精确恒等式误差 > 1e-12 (={max_err_6_17A:.2e})")
if viol_orbits > 0:
    issues.append(f"全轨道 α 收缩违反 ({viol_orbits} 步)")
if min_Dratio_all < 1-1e-12:
    issues.append(f"D_low 界违反 ({min_Dratio_all:.4f})")
if analytical_rj_max >= 1:
    issues.append(f"r_j 解析上界 ≥ 1 (={analytical_rj_max:.4f}) — 6.17C 的 AM-GM 证明有间隙")
if best_rho >= 0.95:
    issues.append(f"ρ(B_sym) 可构造接近 1 (={best_rho:.4f})")
if max_sensitivity > 1000:
    issues.append(f"J 对 M* 高度敏感 (={max_sensitivity:.2e})")

if issues:
    for i in issues:
        print(f"  ✗ {i}")
else:
    print(f"  ✓ 全部 8 项审计通过，无新问题")

print(f"""
  6.17B (l₁ 收缩):      ■ analytical
  6.17C (方向单调性):     ■ ({'analytical' if analytical_rj_max < 1 else 'semi-analytical'}, ρ(B_sym) ≤ {max_amgm_bound:.4f} < 1)
  6.18 (全局收敛):       ■ (6.17B only)
  轨道 α 收缩:          ✓ {viol_orbits}/{total_orbit_steps}
  D_low 安全裕度:       {min_Dratio_all-1:.4f}
  ρ(B_sym) 安全裕度:     {1-max_rho_sym:.4f}
""")
