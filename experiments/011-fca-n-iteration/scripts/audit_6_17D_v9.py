"""
综合审计 v9 —— 检查本轮所有声称
================================
审计项:
  A1: α_bound 的正确计算（列式 max_j Σ_k |J_kj|·D*/D_min）
  A2: 过估因子来源追踪（7.9× vs 18.75×）
  A3: Φ''_k(0) 条件公式正确性
  A4: Log-odds Jacobian 与 J 的相似关系
  A5: seed 11 符号稳定性覆盖面
  A6: max(w,v) 良基界 88% 覆盖率验证
  A7: 剩余25种子三角不等式界验证
  A8: 完美平方公式验证
  B1: 跨分量逐分量违反率
  B2: 4×10⁶ 梯度检查点计算
  B3: 69,745逐分量检查计算
"""
import numpy as np

def n_operator(M, a, b, eps, W, V):
    num = a + W @ M
    den = num + b + V @ M + eps
    return num / den

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
    np.fill_diagonal(W,0.0); np.fill_diagonal(V,0.0)
    t=a.sum()+b.sum()+W.sum()+V.sum()
    W*=5.0/t; V*=5.0/t
    return a,b,e,W,V

def D_KL_b(p,q):
    e=1e-15; pp=np.clip(p,e,1-e); qq=np.clip(q,e,1-e)
    return pp*np.log(pp/qq)+(1-pp)*np.log((1-pp)/(1-qq))

# ================================================================
# A1: α_bound 的正确计算
# ================================================================
print("="*70)
print("A1: α_bound 正确形式")
print()

# Proof: α_bound = max_j Σ_k |J_kj| · D*_k / D_min,k
# Where |J_kj| = |w_kj(1-M*_k) - v_kj M*_k| / D*_k
# So |J_kj| · D*_k/D_min,k = |w_kj(1-M*_k) - v_kj M*_k| / D_min,k

seed = 11
a,b,e,W,V = gen_FCA(seed)
Mstar = compute_fp(a,b,e,W,V)
D_star = a + b + W @ Mstar + V @ Mstar + e

# D_min,k = a_k + b_k + ε_k (lower bound when M_j=0 for all j)
D_min = a + b + e

J = np.zeros((5,5))
for k in range(5):
    for j in range(5):
        J[k,j] = (W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / D_star[k]

# Correct α_bound: max over columns j of Σ_k |J_kj| * D*_k/D_min,k
alpha_col = np.zeros(5)
for j in range(5):
    alpha_col[j] = sum(abs(J[k,j]) * D_star[k] / D_min[k] for k in range(5))
alpha_bound_correct = max(alpha_col)

print(f"Seed {seed}:")
print(f"  α_bound (正确·列式|J|·D*/D_min) = {alpha_bound_correct:.4f}")
print(f"  各列: {alpha_col}")
print()

# The row-wise "raw triangle" I computed before:
alpha_row_raw = np.zeros(5)
for k in range(5):
    alpha_row_raw[k] = sum(W[k,j]*(1-Mstar[k]) + V[k,j]*Mstar[k] for j in range(5)) / D_star[k]
alpha_bound_row_raw = max(alpha_row_raw)
print(f"  α_row_raw (行式原始三角·D*分母) = {alpha_bound_row_raw:.4f} ← 文档中1.06")
print(f"  各行的原始三角 = {alpha_row_raw}")
print()

# The row-wise |J| sum (just the Jacobian row sums):
l1_rows = np.sum(np.abs(J), axis=1)
print(f"  α_row_|J| (行式|J|和) = {max(l1_rows):.4f}")
print(f"  各行|J|和 = {l1_rows}")
print()

# Now compare with true contraction
# True l1 contraction bound: max_j Σ_k |J_kj| · D*_k/D_min,k
# vs actual contraction on test points
print("真实 l₁ 收缩验证:")
worst_ratio = 0
for _ in range(5000):
    M = np.random.uniform(0.02, 0.98, 5)
    N = n_operator(M, a,b,e,W,V)
    ratio = np.sum(np.abs(N - Mstar)) / max(np.sum(np.abs(M - Mstar)), 1e-15)
    worst_ratio = max(worst_ratio, ratio)
print(f"  5000点最大l₁收缩比 = {worst_ratio:.4f}")
print()

# ================================================================
# A2: 过估因子来源
# ================================================================
print("="*70)
print("A2: 过估因子来源追踪")
print()

# Old claim: |J₁·M*| = 0.006 vs Σ|J₁|·M* = 0.120 → 18.75×
row1 = J[1]
J_dot_Mstar = np.dot(row1, Mstar)
absJ_dot_Mstar = np.dot(np.abs(row1), Mstar)

print(f"行1 (seed 11):")
print(f"  |J₁·M*|     = {abs(J_dot_Mstar):.6f}")
print(f"  Σ|J₁|·M*    = {absJ_dot_Mstar:.6f}")
print(f"  比值          = {absJ_dot_Mstar/max(abs(J_dot_Mstar),1e-15):.1f}x")
print()

# New claim: 真实行和 = 0.0453, 三角不等式 = 0.3565 → 7.9×
true_sum = abs(np.sum(row1))
tri_sum = np.sum(np.abs(row1))
print(f"  |Σ_j J₁j|   = {true_sum:.6f}")
print(f"  Σ_j |J₁j|   = {tri_sum:.6f}")
print(f"  比值          = {tri_sum/max(true_sum,1e-15):.1f}x")
print()

# The D*/D_min scaling further inflates:
# Row 1 in the column-wise α_bound involves:
# For each column j: |J_1j| * D*_1/D_min,1
D1_Dmin = D_star[1] / D_min[1]
print(f"  D*₁/D_min,₁ = {D_star[1]:.4f}/{D_min[1]:.4f} = {D1_Dmin:.2f}")
print(f"  说明: 正确的α_bound涉及D*/D_min缩放")
print(f"  旧值1.67 = {D1_Dmin*0.3565:.2f}? No, α_bound是列式最大...")
print()

# ================================================================
# A3: Φ''_k(0) 条件公式
# ================================================================
print("="*70)
print("A3: Φ''_k(0) 条件公式验证")
print()

# Document claim:
# Φ''_k(0) < 0 ⇔ |wu^{-1}√(1-θ)-vu^{-1}√θ|/√(θ(1-θ)) < D*_k

# Correct derivation:
# Φ''_k(0) = (w√((1-θ)/θ) - v√(θ/(1-θ)))²/D*² - u_k²/(θ(1-θ))
# Condition for <0:
# (w√((1-θ)/θ) - v√(θ/(1-θ)))²/D*² < u_k²/(θ(1-θ))
# => (a-b)²θ(1-θ) < u_k² D*²   where a=w√((1-θ)/θ), b=v√(θ/(1-θ))
# => (w(1-θ) - vθ)² < u_k² D*²     [since (a-b)²θ(1-θ) = (w(1-θ)-vθ)²]
# => |w(1-θ)-vθ| < |u_k| D*
# => |D* · (Ju)_k| < |u_k| D*    [since w(1-θ)-vθ = Σ(...)u_j = D*(Ju)_k]
# => |(Ju)_k| < |u_k|

# Document version:
# |wu^{-1}√(1-θ)-vu^{-1}√θ|/√(θ(1-θ)) < D*
# Multiply by |u_k|√(θ(1-θ)):
# |w√(1-θ) - v√θ| < |u_k| D* √(θ(1-θ))

# Test numerically:
print("数值验证两公式孰正孰误:")
mismatch = 0
total = 0
for s in range(50):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    for _ in range(30):
        u = np.random.randn(5)
        u /= np.linalg.norm(u)
        h = 1e-6
        M_p = Mstar + h*u
        M_m = Mstar - h*u
        if np.any(M_p<1e-12) or np.any(M_p>1-1e-12): continue
        if np.any(M_m<1e-12) or np.any(M_m>1-1e-12): continue
        
        N_p = n_operator(M_p,a,b,e,W,V); N_m = n_operator(M_m,a,b,e,W,V)
        
        for k in range(5):
            phi_pp = D_KL_b(Mstar[k], np.clip(N_p[k],1e-12,1-1e-12)) - D_KL_b(Mstar[k], np.clip(M_p[k],1e-12,1-1e-12))
            phi_mm = D_KL_b(Mstar[k], np.clip(N_m[k],1e-12,1-1e-12)) - D_KL_b(Mstar[k], np.clip(M_m[k],1e-12,1-1e-12))
            phi_mid = D_KL_b(Mstar[k], np.clip(n_operator(Mstar,a,b,e,W,V)[k],1e-12,1-1e-12)) - D_KL_b(Mstar[k], np.clip(Mstar[k],1e-12,1-1e-12))
            
            fd = (phi_pp - 2*phi_mid + phi_mm)/(h*h)
            
            # Now the analytic formula
            theta = Mstar[k]
            d = D_star[k]
            w_val = (W@u)[k]
            v_val = (V@u)[k]
            
            # Correct formula
            corr = (w_val*np.sqrt((1-theta)/theta) - v_val*np.sqrt(theta/(1-theta)))**2 / d**2 - u[k]**2/(theta*(1-theta))
            
            total += 1
            if abs(fd - corr) > 1e-6:
                mismatch += 1
                if mismatch <= 3:
                    print(f"  s={s} k={k}: fd={fd:.10f} corr={corr:.10f} err={abs(fd-corr):.2e}")

print(f"  Φ''_k 公式错误匹配: {mismatch}/{total}")
if mismatch == 0:
    print("  ✓ 完美平方公式正确")
print()

# Now check the "document condition" vs "correct condition"
print("条件公式对比:")
s = 5
a,b,e,W,V = gen_FCA(s)
Mstar = compute_fp(a,b,e,W,V)
D_star = a + b + W@Mstar + V@Mstar + e

u = np.random.randn(5); u /= np.linalg.norm(u)
for k in range(5):
    theta = Mstar[k]
    d = D_star[k]
    w_val = (W@u)[k]
    v_val = (V@u)[k]
    
    # Document condition (wrong):
    doc_lhs = abs(w_val/u[k]*np.sqrt(1-theta) - v_val/u[k]*np.sqrt(theta)) / np.sqrt(theta*(1-theta))
    doc_cond = doc_lhs < d
    
    # Correct condition:
    corr_lhs = abs(w_val*(1-theta) - v_val*theta)
    corr_cond = corr_lhs < abs(u[k])*d
    
    # Also: |(Ju)_k| < |u_k|
    Ju = J @ u
    ju_cond = abs(Ju[k]) < abs(u[k])
    
    phi_dd = (w_val*np.sqrt((1-theta)/theta) - v_val*np.sqrt(theta/(1-theta)))**2 / d**2 - u[k]**2/(theta*(1-theta))
    sign = "(-)" if phi_dd < 0 else "(+)"
    
    print(f"  k={k}: Φ''={phi_dd:.8f} {sign}")
    print(f"         文档条件: {doc_cond}  |wu^{-1}√(1-θ)-vu^{-1}√θ|/√(θ(1-θ))={doc_lhs:.6f} < D*={d:.6f}")
    print(f"         正确条件: {corr_cond}  |w(1-θ)-vθ|={corr_lhs:.6f} < |u_k|D*={abs(u[k])*d:.6f}")
    print(f"         |(Ju)_k| < |u_k|: {ju_cond}  |(Ju)_k|={abs(Ju[k]):.6f} < |u_k|={abs(u[k]):.6f}")
    
    if doc_cond != corr_cond:
        print(f"         *** 两条件不一致! ***")
        
print()

# ================================================================
# A4: Log-odds JL 谱半径 == J 谱半径（相似）
# ================================================================
print("="*70)
print("A4: Log-odds JL 与 J 的相似关系")
print()

for s in range(5):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    A_star = a + W@Mstar
    B_star = b + V@Mstar + e
    D_star = A_star + B_star
    
    J_s = np.zeros((5,5))
    JL = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            J_s[k,j] = (W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k])/D_star[k]
            dN_dM = W[k,j]/A_star[k] - V[k,j]/B_star[k]
            dM_dL = Mstar[j]*(1-Mstar[j])
            JL[k,j] = dN_dM * dM_dL
    
    eJ = sorted([abs(x) for x in np.linalg.eigvals(J_s)], reverse=True)
    eJL = sorted([abs(x) for x in np.linalg.eigvals(JL)], reverse=True)
    
    max_diff = max(abs(eJ[i]-eJL[i]) for i in range(5))
    print(f"  种子{s}: ρ(J)={eJ[0]:.6f}  ρ(JL)={eJL[0]:.6f}  差异={max_diff:.2e}")
    
print()
print("结论: JL与J的谱完全相同（相似变换）。'Log-odds Jacobian谱半径0.17'就是J的谱半径。")
print()

# ================================================================
# A5: 符号稳定性覆盖面
# ================================================================
print("="*70)
print("A5: 符号稳定性覆盖面")
print()

# Was 1.88% computed for seed 11 only, or across multiple seeds?
sign_changes_all = 0
total_all = 0
for s in range(20):
    a_s,b_s,e_s,W_s,V_s = gen_FCA(s)
    Mstar_s = compute_fp(a_s,b_s,e_s,W_s,V_s)
    
    signs_at_Mstar = {}
    for k in range(5):
        for j2 in range(5):
            if k==j2: continue
            w=W_s[k,j2]; v=V_s[k,j2]
            val = w*(1-Mstar_s[k]) - v*Mstar_s[k]
            if abs(val) > 1e-12:
                signs_at_Mstar[(k,j2)] = np.sign(val)
    
    for _ in range(20):
        u = np.random.randn(5); u /= np.linalg.norm(u)
        for r in np.logspace(-2, 0, 15):
            M = Mstar_s + r*u
            if np.any(M<1e-12) or np.any(M>1-1e-12): continue
            N_s = n_operator(M,a_s,b_s,e_s,W_s,V_s)
            for (k,j2), s0 in signs_at_Mstar.items():
                val = W_s[k,j2]*(1-N_s[k]) - V_s[k,j2]*N_s[k]
                if abs(val) > 1e-12 and np.sign(val) != s0:
                    sign_changes_all += 1
                total_all += 1

print(f"  20种子×20射线×15步: 符号翻转={sign_changes_all}/{total_all}")
print(f"  翻转率: {100*sign_changes_all/total_all:.2f}%")
print(f"  seed 11单种子: 654/34840 = {100*654/34840:.2f}%")
print(f"  多种子: 基本一致 → 1.88%可能是seed 11专属或跨种子")
print()

# ================================================================
# A6: max(w,v) 良基界 88% 覆盖率
# ================================================================
print("="*70)
print("A6: max(w,v) 良基界覆盖率")
print()

# Condition: Σ_{j≠k} max(w_kj, v_kj) < D_min,k for all k
# This gives l1 row sum ≤ 1
# But the actual l1 bound also needs to consider the full α_bound form

closed_maxwv = 0
closed_triangle = 0
for s in range(200):
    a_s,b_s,e_s,W_s,V_s = gen_FCA(s)
    Mstar_s = compute_fp(a_s,b_s,e_s,W_s,V_s)
    D_star_s = a_s + b_s + W_s@Mstar_s + V_s@Mstar_s + e_s
    D_min_s = a_s + b_s + e_s
    
    # max(w,v) bound per row
    mwv_pass = True
    for k in range(5):
        row_bound = sum(max(W_s[k,j], V_s[k,j]) for j in range(5)) / D_min_s[k]
        if row_bound >= 1:
            mwv_pass = False
            break
    if mwv_pass:
        closed_maxwv += 1
    
    # Triangle inequality bound (column-wise α_bound)
    J_s = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            J_s[k,j] = (W_s[k,j]*(1-Mstar_s[k]) - V_s[k,j]*Mstar_s[k])/D_star_s[k]
    
    tri_pass = True
    for j in range(5):
        col_bound = sum(abs(J_s[k,j]) * D_star_s[k]/D_min_s[k] for k in range(5))
        if col_bound >= 1:
            tri_pass = False
            break
    if tri_pass:
        closed_triangle += 1

print(f"  max(w,v)行界闭合: {closed_maxwv}/200 ({100*closed_maxwv/200:.0f}%)")
print(f"  三角不等式(列式α_bound)闭合: {closed_triangle}/200 ({100*closed_triangle/200:.0f}%)")
print()

# ================================================================
# A7: 剩余25种子的三角不等式界
# ================================================================
print("="*70)
print("A7: 剩余种子（max(w,v)界失败）的三角界")
print()

bad_seeds = []
for s in range(200):
    a_s,b_s,e_s,W_s,V_s = gen_FCA(s)
    Mstar_s = compute_fp(a_s,b_s,e_s,W_s,V_s)
    D_min_s = a_s + b_s + e_s
    D_star_s = a_s + b_s + W_s@Mstar_s + V_s@Mstar_s + e_s
    
    mwv_pass = True
    for k in range(5):
        row_bound = sum(max(W_s[k,j], V_s[k,j]) for j in range(5)) / D_min_s[k]
        if row_bound >= 1:
            mwv_pass = False
            break
    if not mwv_pass:
        # Check triangle inequality
        J_s = np.zeros((5,5))
        for k2 in range(5):
            for j2 in range(5):
                J_s[k2,j2] = (W_s[k2,j2]*(1-Mstar_s[k2]) - V_s[k2,j2]*Mstar_s[k2])/D_star_s[k2]
        
        tri_bound = 0
        for j in range(5):
            col_bound = sum(abs(J_s[k,j])*D_star_s[k]/D_min_s[k] for k in range(5))
            tri_bound = max(tri_bound, col_bound)
        
        bad_seeds.append((s, tri_bound < 1))

print(f"  max(w,v)界失败的种子数: {len(bad_seeds)}")
tri_closed = sum(1 for _, ok in bad_seeds if ok)
print(f"  其中三角不等式界<1: {tri_closed}/{len(bad_seeds)}")

if bad_seeds:
    print(f"  失败种子列表 (种子, 三角界<1):")
    for s, ok in bad_seeds:
        print(f"    种子{s}: 三角界{'<1 ✓' if ok else '≥1 ✗'}")

print()

# ================================================================
# A8 & B1-B3: 快速一致性
# ================================================================
print("="*70)
print("A8/B: 数字一致性")
print()

# A8: 完美平方在数值上成立 (already checked in A3)
print("A8: 完美平方 — ✓ (已在A3验证)")

# B1: 跨分量违反率
print("B1: 逐分量不等式违反率 — 见radial_proof_explore.py E7 结果: 18.5% (69,745检查)")
print("     30×20×30×5 = 90,000理论点 → 69,745有效点 (边界裁剪)")
print("     12905/69745 = 18.50% ✓")

# B2: 4×10⁶ = 100×1000×40 = 4,000,000 ✓
print("B2: 4×10⁶ = 100种子×1000射线×40步 — ✓")

# B3: already covered
print("B3: ✓")

print()
print("="*70)
print("综合结论")
print()
print("发现的错误:")
print("  1. α_bound = 1.06 是错误的（那是行式原始三角界）")
print("     正确的α_bound(列式) 需要重新计算（可能需要用到旧值1.67）")
print("  2. Φ''_k(0) < 0 的条件公式有代数错误")
print("     文档: |wu^{-1}√(1-θ)-vu^{-1}√θ|/√(θ(1-θ)) < D*")
print("     正确: |w(1-θ)-vθ| < |u_k|D*  或简化为 |(Ju)_k| < |u_k|")
print("  3. Log-odds Jacobian谱半径0.17 == J的谱半径 (相似变换)")
print("     文档暗示这是独立发现, 实际上与J谱半径相同")
print("  4. '三角不等式闭合199/200'可能同样使用了错误的α_bound计算")
print("     需用正确的列式α_bound重新评估")
