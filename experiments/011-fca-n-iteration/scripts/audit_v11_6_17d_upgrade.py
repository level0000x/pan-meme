"""
审计 v11 — 6.17D ■† 升级框架全面验证
========================================
审计A: η'' Cauchy-Schwarz 解析上界是否真 ≤ 2.7
审计B: ψ'' 在球面 r=0.049 的下界是否真 ≥ 4.0  
审计C: 对抗性方向搜索 (max φ'')
审计D: l₁→l₂ R_global 范数转换一致性
审计E: T=9 逐实例 V_KL 下降验证
"""
import numpy as np

def n_operator(M, a, b, eps, W, V):
    num = a + W @ M
    return num / (num + b + V @ M + eps)

def compute_fp(a,b,eps,W,V):
    M=np.full(5,0.5)
    for _ in range(20000):
        Mn=n_operator(M,a,b,eps,W,V)
        if np.max(np.abs(Mn-M))<1e-15: return Mn
        M=Mn
    return M

def gen_FCA(seed):
    rs=np.random.RandomState(seed%(2**31))
    a=rs.uniform(0.01,0.5,5); b=rs.uniform(0.01,0.5,5); e=rs.uniform(0.001,0.1,5)
    W=rs.uniform(0.01,0.3,(5,5)); V=rs.uniform(0.01,0.3,(5,5))
    np.fill_diagonal(W,0); np.fill_diagonal(V,0)
    t=a.sum()+b.sum()+W.sum()+V.sum(); W*=5.0/t; V*=5.0/t
    return a,b,e,W,V

# ============================================================
print("="*65)
print("审计A: η'' Cauchy-Schwarz 解析上界")
print("="*65)

# η''(r) = Σ_k [θ_k·(w_k/A_k)² + (1-θ_k)·(v_k/B_k)² − z²/(1+rz)²]
#        ≤ Σ_k [θ_k·(w_k/A_k)² + (1-θ_k)·(v_k/B_k)²]
#        ≤ Σ_k [θ_k·(Σ_j w_kj u_j)²/A*²_k + (1-θ_k)·(Σ_j v_kj u_j)²/B*²_k]
# Cauchy-Schwarz: (Σ_j w_kj u_j)² ≤ Σ_j w_kj² · Σ_j u_j² = ‖W_row_k‖²₂·‖u‖² = ‖W_row_k‖²₂
# ∴ η'' ≤ Σ_k [θ_k·‖W_row_k‖²₂/A*²_k + (1-θ_k)·‖V_row_k‖²₂/B*²_k]

# Verify this bound holds for ALL directions u
max_eta_cauchy = 0; max_over_bound = 0
for s in range(200):
    a,b,e,W,V=gen_FCA(s); theta=compute_fp(a,b,e,W,V)
    A_s=a+W@theta; B_s=b+e+V@theta

    # Analytic bound per seed
    cauchy_bound = 0
    for k in range(5):
        w_row_norm2 = np.sum(W[k,:]**2)
        v_row_norm2 = np.sum(V[k,:]**2)
        cauchy_bound += theta[k]*w_row_norm2/A_s[k]**2 + (1-theta[k])*v_row_norm2/B_s[k]**2
    max_eta_cauchy = max(max_eta_cauchy, cauchy_bound)

    # Verify against random u
    for _ in range(200):
        u=np.random.randn(5); u/=np.linalg.norm(u)
        w_u=W@u; v_u=V@u
        eta_ub_u = sum(theta*w_u**2/A_s**2 + (1-theta)*v_u**2/B_s**2)
        if eta_ub_u > cauchy_bound + 1e-12:
            max_over_bound = max(max_over_bound, eta_ub_u - cauchy_bound)

print(f"  Cauchy-Schwarz 上界 (200 seeds): max = {max_eta_cauchy:.4f}")
print(f"  方向 u 超出 Cauchy-Schwarz 界的最大超量: {max_over_bound:.2e}")
print(f"  {'✓ Cauchy-Schwarz 界对所有 u 成立' if max_over_bound < 1e-10 else '✗'}")

# But wait — this bound is for ALL r. 
# For r=0 (B(M*,R) sphere), η'' ≤ this bound always.
# However: CAUTION — the upper bound we use is:
#   η''(r) ≤ Σ θ·x² + Σ (1-θ)·y²  (dropping −z² term)
# But at r>0, the third term is −z²/(1+rz)², which is NEGATIVE
# So η''(r) ≤ Σ θ·x²+(1-θ)·y² holds as a VALID upper bound
# (dropping a non-positive term only increases the bound)

# ALSO: for r>0, the A_k, B_k change:
# η''_k(r) = θ·w²/A(r)² + (1-θ)·v²/B(r)² − (w+v)²/D(r)²
# ≤ θ·w²/A(r)² + (1-θ)·v²/B(r)²
# But A(r) ≥ A* − r·|w|, so 1/A(r)² ≥ 1/(A* − r·w)²
# For the UPPER bound we need an UPPER bound on η'', so we need
# 1/A(r) to be as LARGE as possible, i.e. A(r) as SMALL as possible.
# A(r) = A* + r·w, where w can be negative.
# min A(r) = A* − r·max_k(|w_k|) [at extreme w<0]
# 
# This is the CRITICAL CHECK: our Cauchy-Schwarz bound used A*, not min A(r)
# For the ball B(M*,R), need to verify 1/A(r) ≤ 1/(A* − R·‖W_row‖)
# i.e., the denominator shrinkage is accounted for.

print(f"\n  关键检查: Cauchy-Schwarz界用了 A* 而非 min A(r)")
print(f"  R = 0.049, 在最不利方向 w_k < 0 且 |w_k| 最大时:")

max_eta_with_denom = 0
for s in range(200):
    a,b,e,W,V=gen_FCA(s); theta=compute_fp(a,b,e,W,V)
    A_s=a+W@theta; B_s=b+e+V@theta
    R = 0.049

    for _ in range(500):
        u=np.random.randn(5); u/=np.linalg.norm(u)
        w_u=W@u; v_u=V@u; M=theta+R*u

        # Actual η'' on sphere (with correct denominators)
        if np.any(M<1e-8) or np.any(M>1-1e-8): continue
        A=a+W@M; B=b+e+V@M; D=A+B
        eta2_actual = sum(theta*w_u**2/A**2 + (1-theta)*v_u**2/B**2 - (w_u+v_u)**2/D**2)
        max_eta_with_denom = max(max_eta_with_denom, eta2_actual)

print(f"  η''_actual on sphere r=0.049: max = {max_eta_with_denom:.6f}")
print(f"  vs Cauchy-Schwarz bound:        max = {max_eta_cauchy:.4f}")
print(f"  {'✓ Cauchy-Schwarz bound holds (updated denominators safe)' if max_eta_with_denom <= max_eta_cauchy + 1e-10 else '✗ DENOMINATOR SHRINKAGE ISSUE'}")

# ============================================================
print(f"\n{'='*65}")
print("审计B: ψ'' 在球面 r=0.049 的全局下界")
print("="*65)

psi_min_sphere = np.inf; psi_min_seed = -1; psi_min_u = None
for s in range(200):
    a,b,e,W,V=gen_FCA(s); theta=compute_fp(a,b,e,W,V)
    for _ in range(1000):
        u=np.random.randn(5); u/=np.linalg.norm(u)
        M=theta+0.049*u
        if np.any(M<1e-8) or np.any(M>1-1e-8): continue
        psi_r=sum(u**2*(theta/M**2+(1-theta)/(1-M)**2))
        if psi_r < psi_min_sphere:
            psi_min_sphere = psi_r
            psi_min_seed = s
            psi_min_u = u.copy()

print(f"  ψ'' 球面下界 (200x1000=200K checks): min = {psi_min_sphere:.4f} (seed {psi_min_seed})")
print(f"  最劣方向 u: [{psi_min_u[0]:.4f}, {psi_min_u[1]:.4f}, {psi_min_u[2]:.4f}, {psi_min_u[3]:.4f}, {psi_min_u[4]:.4f}]")
print(f"  u 的组成: max|u|={max(abs(psi_min_u)):.4f}, min|u|={min(abs(psi_min_u)):.4f}")

# Check: is this a degenerate direction (u concentrated on one component)?
# If ψ'' is small because u is concentrated on components with large θ(1-θ), 
# that's fine — it's still ≥ something.
# But we need ψ'' ≥ 4.0 to make the gap work.

print(f"\n  φ'' 上界: {max_eta_cauchy:.4f} − {psi_min_sphere:.4f} = {max_eta_cauchy - psi_min_sphere:.4f}")
print(f"  {'✓ φ'' < 0 on sphere (via Cauchy-Schwarz + ψ''_min)' if max_eta_cauchy < psi_min_sphere else '✗ GAP MAY EXIST'}")

# ============================================================
print(f"\n{'='*65}")
print("审计C: 对抗性方向搜索 — 找 max φ''")
print("="*65)

max_phi2_adversarial = -1e10
worst_ad_seed = -1; worst_ad_r = 0; worst_ad_u = None
for s in range(200):
    a,b,e,W,V=gen_FCA(s); theta=compute_fp(a,b,e,W,V)
    A_s=a+W@theta; B_s=b+e+V@theta
    for _ in range(500):
        u=np.random.randn(5); u/=np.linalg.norm(u)
        w_u=W@u; v_u=V@u
        for r in [0.01, 0.02, 0.03, 0.04, 0.049, 0.06, 0.08, 0.1]:
            M=theta+r*u
            if np.any(M<1e-8) or np.any(M>1-1e-8): continue
            A=a+W@M; B=b+e+V@M; D=A+B
            e2=sum(theta*w_u**2/A**2+(1-theta)*v_u**2/B**2-(w_u+v_u)**2/D**2)
            p2=sum(u**2*(theta/M**2+(1-theta)/(1-M)**2))
            phi2=e2-p2
            if phi2 > max_phi2_adversarial:
                max_phi2_adversarial = phi2
                worst_ad_seed = s; worst_ad_r = r; worst_ad_u = u.copy()

print(f"  max φ'' (对抗搜索): {max_phi2_adversarial:.6f} (seed {worst_ad_seed}, r={worst_ad_r:.4f})")
print(f"  最劣 u: [{worst_ad_u[0]:.4f}, {worst_ad_u[1]:.4f}, {worst_ad_u[2]:.4f}, {worst_ad_u[3]:.4f}, {worst_ad_u[4]:.4f}]")
print(f"  {'✓ 对抗搜索零违规 (φ''始终<0)' if max_phi2_adversarial < -1e-10 else '✗'}")

# ============================================================
print(f"\n{'='*65}")
print("审计D: l₁ 收缩 → R_global 球域一致性")
print("="*65)

# R_global = 0.049 is in l₂ norm (since ‖u‖=1)  
# But l₁ contraction says: ‖Δ(t+1)‖₁ ≤ α·‖Δ(t)‖₁
# We need to convert l₁ distance to l₂ radius.
# Relationship: ‖x‖₂ ≤ ‖x‖₁ ≤ √n·‖x‖₂ with n=5
# So: ‖Δ‖₂ ≤ ‖Δ‖₁ ≤ √5·‖Δ‖₂

# Our criterion: ‖Δ‖₂ ≤ R_global = 0.049
# This means ‖Δ‖₁ ≤ √5·0.049 ≈ 0.110
# So we need: α^{T-1}·D0 ≤ 0.110

alpha_max = 0; D0_max = 0
for s in range(200):
    a,b,e,W,V=gen_FCA(s); theta=compute_fp(a,b,e,W,V)
    Dstar=a+b+e+(W+V)@theta
    D_max=a+b+e+(W+V)@np.ones(5); m0=a/D_max; D_low=a+b+e+(W+V)@m0
    J=np.zeros((5,5))
    for k in range(5):
        for jj in range(5):
            if k!=jj: J[k,jj]=(W[k,jj]*(1-theta[k])-V[k,jj]*theta[k])/Dstar[k]
    alpha_j=np.array([sum(abs(J[kk,jj])*Dstar[kk]/D_low[kk] for kk in range(5)) for jj in range(5)])
    alpha_max=max(alpha_max, max(alpha_j))
    D0_max=max(D0_max, sum(1-m0))

print(f"  α_max = {alpha_max:.4f}")
print(f"  D0_max (l₁) = {D0_max:.4f}")
print(f"  R_global (l₂) = 0.049 → R_global (l₁) ≤ {np.sqrt(5)*0.049:.4f}")

T_l2 = 1 + int(np.ceil(np.log(np.sqrt(5)*0.049/D0_max)/np.log(alpha_max)))
print(f"  T (‖·‖₂ ≤ 0.049 after l₁ contraction): {T_l2}")

# BUT: l₁ norm is in 5-dim. Does R_global need to be in l₁ or l₂?
# Our Lemma 2 says: φ'' < 0 in B(M*, R_global) in l₂ (Euclidean) norm
# Lemma 3 says: ‖M(t)−M*‖₂ ≤ R_global after T steps
# ‖M(t)−M*‖₂ ≤ ‖M(t)−M*‖₁ ≤ α^{t-1}·D0
# So if α^{t-1}·D0 ≤ R_global, then ‖M(t)−M*‖₂ ≤ R_global
# Wait — ‖M−M*‖₂ ≤ ‖M−M*‖₁ is false in general!
# ‖x‖₂ ≤ ‖x‖₁ is always true. Yes!

# So if ‖Δ‖₁ ≤ R_l1, then ‖Δ‖₂ ≤ R_l1 (since ‖·‖₂ ≤ ‖·‖₁)
# We need ‖Δ‖₁ ≤ R_global(l₂) = 0.049
# So condition: α^{T-1}·D0 ≤ 0.049
T_correct = 1 + int(np.ceil(np.log(0.049/D0_max)/np.log(alpha_max)))
print(f"  T_correct (||Delta||_2 <= 0.049 via l2 <= l1 <= alpha^(t-1) * D0): {T_correct}")
print(f"  Note: ‖x‖₂ ≤ ‖x‖₁ always, so l₁ ≤ R ⇒ l₂ ≤ R")

# Verify this actually holds
l1_to_l2_ok = True
for s in range(200):
    a,b,e,W,V=gen_FCA(s); theta=compute_fp(a,b,e,W,V)
    M=np.random.random(5)
    for t in range(T_correct):
        M=n_operator(M,a,b,e,W,V)
    if np.sum(abs(M-theta)) > 0.049 + 1e-10:
        l1_to_l2_ok = False
        print(f"  ✗ Seed {s}: ‖M({T_correct})−M*‖₂ = {np.sqrt(np.sum((M-theta)**2)):.4f} > 0.049")
        break

if l1_to_l2_ok:
    print(f"  ✓ ‖M({T_correct})−M*‖₂ ≤ 0.049 verified (200 seeds)")

# ============================================================
print(f"\n{'='*65}")
print("审计E: T-correct 逐实例 V_KL 下降验证")
print("="*65)

viol_T=0; total_T=0
T_verify = T_correct
for s in range(200):
    a,b,e,W,V=gen_FCA(s); Mstar=compute_fp(a,b,e,W,V)
    for _ in range(10):
        M0=np.random.random(5); M=M0.copy()
        for t in range(T_verify):
            Mn=n_operator(M,a,b,e,W,V)
            kl_before=sum(Mstar*np.log(Mstar/np.clip(M,1e-15,None))+(1-Mstar)*np.log((1-Mstar)/np.clip(1-M,1e-15,None)))
            kl_after=sum(Mstar*np.log(Mstar/np.clip(Mn,1e-15,None))+(1-Mstar)*np.log((1-Mstar)/np.clip(1-Mn,1e-15,None)))
            total_T+=1
            if kl_after>=kl_before-1e-14: viol_T+=1
            M=Mn

print(f"  T = {T_verify}")
print(f"  V_KL decrease violations: {viol_T}/{total_T}")
print(f"  {'✓ 无 V_KL 不降' if viol_T==0 else '✗ 存在 V_KL 不降或上升'}")

# ============================================================
print(f"\n{'='*65}")
print("审计F: 内核漏洞 — 解析 η'' 界在 R=0.049 处的分母修正")
print("="*65)

# THE CRITICAL FLAW CHECK:
# Our Cauchy-Schwarz bound for η'' uses A* and B* as denominators.
# But at r>0, the actual denominator is A(r) = A* + r·w
# When w is NEGATIVE, A(r) < A*, so 1/A(r)² > 1/A*²
# This means η''_actual(r) > η''_cauchy for certain directions!
#
# We need to bound |w_k| to account for denominator shrinkage.
# w_k = (Wu)_k, |w_k| ≤ ‖W_row_k‖₂ (Cauchy-Schwarz)
# ∴ A(r) = A* + r·w ≥ A* − r·‖W_row‖₂
# ∴ 1/A(r)² ≤ 1/(A* − r·‖W_row‖)²
#
# The corrected bound:
# η''_cauchy_corrected = Σ [θ·w²/(A*−R·‖W_row‖)² + (1-θ)·v²/(B*−R·‖V_row‖)²]
# But we still need to bound w², v²...

# Check: at R=0.049, how much can A_k shrink?
# w_k = Σ_j W_kj u_j, min w_k = −‖W_row_k‖₂
# A_k = A*_k + r·w_k ≥ A*_k − r·‖W_row_k‖₂

min_A_ratio=1.0; min_B_ratio=1.0
for s in range(200):
    a,b,e,W,V=gen_FCA(s); theta=compute_fp(a,b,e,W,V)
    A_s=a+W@theta; B_s=b+e+V@theta
    R=0.049
    for k in range(5):
        w_max=np.sqrt(np.sum(W[k,:]**2))
        v_max=np.sqrt(np.sum(V[k,:]**2))
        min_A_k=A_s[k]-R*w_max
        min_B_k=B_s[k]-R*v_max
        min_A_ratio=min(min_A_ratio, min_A_k/A_s[k])
        min_B_ratio=min(min_B_ratio, min_B_k/B_s[k])

print(f"  min A(r)/A* = {min_A_ratio:.4f} (across all seeds)")
print(f"  min B(r)/B* = {min_B_ratio:.4f} (across all seeds)")
print(f"  Denominator shrinkage: A can drop to {min_A_ratio*100:.1f}% of A*")
print(f"  → 1/A² can increase by factor {1/min_A_ratio**2:.2f}x")

# Corrected bound:
corrected_max = 0
for s in range(200):
    a,b,e,W,V=gen_FCA(s); theta=compute_fp(a,b,e,W,V)
    A_s=a+W@theta; B_s=b+e+V@theta
    R=0.049
    cb=0
    for k in range(5):
        w_norm=np.sqrt(np.sum(W[k,:]**2))
        v_norm=np.sqrt(np.sum(V[k,:]**2))
        A_min=max(A_s[k]-R*w_norm, 1e-15)
        B_min=max(B_s[k]-R*v_norm, 1e-15)
        cb+=theta[k]*w_norm**2/A_min**2+(1-theta[k])*v_norm**2/B_min**2
    corrected_max=max(corrected_max, cb)

print(f"\n  原 Cauchy-Schwarz 界:  {max_eta_cauchy:.4f}")
print(f"  修正后 Cauchy-Schwarz 界: {corrected_max:.4f}")
print(f"  增幅因子: {corrected_max/max_eta_cauchy:.2f}x")
print(f"  φ'' 上界 (修正): {corrected_max:.4f} − {psi_min_sphere:.4f} = {corrected_max-psi_min_sphere:.4f}")
print(f"  {'✓ 修正后仍 < 0' if corrected_max < psi_min_sphere else '✗ 修正后 φ'' 可正!'}")

# ============================================================
print(f"\n{'='*65}")
print("审计G: 全链重新扫描 — 球内逐点 φ'' < 0")
print("="*65)

viol_ball=0; total_ball=0; max_phi_ball=-1e10
for s in range(200):
    a,b,e,W,V=gen_FCA(s); theta=compute_fp(a,b,e,W,V); R=0.049
    for _ in range(50):
        v=np.random.randn(5); v=v/np.linalg.norm(v)*R*np.random.random()
        M=theta+v
        if np.any(M<1e-8) or np.any(M>1-1e-8): continue
        for __ in range(20):
            u=np.random.randn(5); u/=np.linalg.norm(u)
            A=a+W@M; B=b+e+V@M; D=A+B
            e2=sum(theta*(W@u)**2/A**2+(1-theta)*(V@u)**2/B**2-((W+V)@u)**2/D**2)
            p2=sum(u**2*(theta/M**2+(1-theta)/(1-M)**2))
            phi2=e2-p2
            total_ball+=1
            if phi2>=0: viol_ball+=1
            if phi2>max_phi_ball: max_phi_ball=phi2

print(f"  B(0.049) 球内 随机点×随机方向: {viol_ball}/{total_ball}")
print(f"  max φ'' = {max_phi_ball:.6e}")
if viol_ball==0:
    print(f"  ✓ 球内全负 — 紧致性成立")
else:
    gap_ratio = viol_ball/total_ball*100
    print(f"  ✗ {viol_ball} 个违规 ({gap_ratio:.2f}%)")

# ============================================================
print(f"\n{'='*65}")
print("最终裁决")
print("="*65)

issues = []
if max_over_bound > 1e-10:
    issues.append(f"A: Cauchy-Schwarz 方向检查有超量")
if max_eta_cauchy >= psi_min_sphere:
    issues.append(f"B: Cauchy-Schwarz 界 ({max_eta_cauchy:.2f}) ≥ ψ''_min ({psi_min_sphere:.2f})")
if max_eta_with_denom >= psi_min_sphere:
    issues.append(f"A': 分母修正后 η''_actual ({max_eta_with_denom:.4f}) ≥ ψ''_min")
if corrected_max >= psi_min_sphere:
    issues.append(f"F: 全修正 Cauchy-Schwarz ({corrected_max:.2f}) ≥ ψ''_min ({psi_min_sphere:.2f})")
if viol_ball > 0:
    issues.append(f"G: 球内 {viol_ball} 个 φ'' ≥ 0")
if max_phi2_adversarial > -1e-10:
    issues.append(f"C: 对抗搜索 max φ'' = {max_phi2_adversarial:.6e}")
if viol_T > 0:
    issues.append(f"E: T-step V_KL 不降 {viol_T} 次")
if not l1_to_l2_ok:
    issues.append(f"D: l₁→l₂ 范数转换失败")

if issues:
    print("  发现以下问题:")
    for i in issues:
        print(f"    - {i}")
    print(f"\n  需要修正!")
else:
    print("  ✓ 全部审计通过")
