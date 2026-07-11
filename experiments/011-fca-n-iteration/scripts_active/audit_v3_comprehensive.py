"""
全面审计 v3: 6.17A₃ 迭代界 + 6.17A₂ 行和界闭合 + 6.17B + 6.17D
=================================================================
审计清单:
  A1: m^(0) ≤ M* ≤ u^(0) (归纳基)
  A2: m^(t+1) ≤ M* ≤ u^(t+1) 从 m^(t) ≤ M* ≤ u^(t) (归纳步)  
  A3: m^(t) monotone INCREASING (单调性基+归纳)
  A4: u^(t) monotone DECREASING
  A5: m^(t) ≤ u^(t) 始终
  A6: 迭代界收紧后的行和界 (T=0,1,2)
  A7: 用D_low替代D*的行和界保守性
  A8: 6.17B α界与6.17A₂ RD/CD的一致性
  A9: 6.17D φ''框架无退化
  A10: Gershgorin链完整性
"""
import numpy as np

def n_operator(M, a, b, eps, W, V):
    num = a + W @ M; den = num + b + V @ M + eps
    return num / den

def compute_fp(a,b,eps,W,V):
    M = np.full(5, 0.5)
    for _ in range(20000):
        Mn = n_operator(M, a, b, eps, W, V)
        if np.max(np.abs(Mn - M)) < 1e-15: return Mn
        M = Mn
    return M

def gen_FCA(seed):
    rs = np.random.RandomState(seed % (2**31))
    a = rs.uniform(0.01, 0.5, 5); b = rs.uniform(0.01, 0.5, 5)
    e = rs.uniform(0.001, 0.1, 5)
    W = rs.uniform(0.01, 0.3, (5, 5)); V = rs.uniform(0.01, 0.3, (5, 5))
    np.fill_diagonal(W, 0.0); np.fill_diagonal(V, 0.0)
    t = a.sum()+b.sum()+W.sum()+V.sum()
    W *= 5./t; V *= 5./t
    return a, b, e, W, V

def gen_extended(seed):
    rs = np.random.RandomState(seed % (2**31))
    a = rs.uniform(0.005, 0.8, 5); b = rs.uniform(0.005, 0.8, 5)
    e = rs.uniform(0.0005, 0.2, 5)
    W = rs.uniform(0.005, 0.5, (5, 5)); V = rs.uniform(0.005, 0.5, (5, 5))
    np.fill_diagonal(W, 0.0); np.fill_diagonal(V, 0.0)
    return a, b, e, W, V

def g_k_val(k, x, W, V):
    return sum(abs(W[k,j] - (W[k,j]+V[k,j])*x) for j in range(5) if j != k)

print("=" * 70)
print("A1-A5: 迭代界数学有效性 (200 FCA + 500 扩展)")
print("=" * 70)

for label, gen_fn, n_seeds in [("FCA", gen_FCA, 200), ("扩展", gen_extended, 500)]:
    violations = {
        'A1_lower': 0, 'A1_upper': 0,
        'A2_lower_t1': 0, 'A2_upper_t1': 0,
        'A2_lower_t2': 0, 'A2_upper_t2': 0,
        'A3_mono_m': 0, 'A4_mono_u': 0,
        'A5_mlt_u': 0,
    }
    total = n_seeds * 5
    
    for s in range(n_seeds):
        a,b,e,W,V = gen_fn(s)
        Mstar = compute_fp(a,b,e,W,V)
        D_max = a+b+e+np.sum(W+V, axis=1)
        sum_w = np.sum(W, axis=1)
        
        m = a / D_max
        u = (a + sum_w) / (a + sum_w + b + e)
        
        # A1
        for k in range(5):
            if m[k] > Mstar[k] * (1 + 1e-12):
                violations['A1_lower'] += 1
            if u[k] < Mstar[k] * (1 - 1e-12):
                violations['A1_upper'] += 1
        
        # t=0→1
        m_prev, u_prev = m.copy(), u.copy()
        m = (a + W @ m_prev) / (a + b + e + (W+V) @ u_prev)
        u = (a + W @ u_prev) / (a + b + e + (W+V) @ m_prev)
        m = np.maximum(m, 0); u = np.minimum(u, 1)
        
        for k in range(5):
            # A2 t=1
            if m[k] > Mstar[k] * (1 + 1e-12):
                violations['A2_lower_t1'] += 1
            if u[k] < Mstar[k] * (1 - 1e-12):
                violations['A2_upper_t1'] += 1
            # A3/A4 base
            if m[k] < m_prev[k] * (1 - 1e-12):
                violations['A3_mono_m'] += 1
            if u[k] > u_prev[k] * (1 + 1e-12):
                violations['A4_mono_u'] += 1
            # A5
            if m[k] > u[k] * (1 + 1e-12):
                violations['A5_mlt_u'] += 1
        
        # t=1→2
        m_prev, u_prev = m.copy(), u.copy()
        m = (a + W @ m_prev) / (a + b + e + (W+V) @ u_prev)
        u = (a + W @ u_prev) / (a + b + e + (W+V) @ m_prev)
        m = np.maximum(m, 0); u = np.minimum(u, 1)
        
        for k in range(5):
            if m[k] > Mstar[k] * (1 + 1e-12):
                violations['A2_lower_t2'] += 1
            if u[k] < Mstar[k] * (1 - 1e-12):
                violations['A2_upper_t2'] += 1
            if m[k] < m_prev[k] * (1 - 1e-12):
                violations['A3_mono_m'] += 1
            if u[k] > u_prev[k] * (1 + 1e-12):
                violations['A4_mono_u'] += 1
    
    total_v = sum(violations.values())
    print(f"\n  {label} ({n_seeds}:")
    print(f"    A1 m^(0)≤M*≤u^(0):    {'✓' if violations['A1_lower']+violations['A1_upper']==0 else '✗'}")
    print(f"    A2 m^(1)≤M*≤u^(1):    {'✓' if violations['A2_lower_t1']+violations['A2_upper_t1']==0 else '✗'}")
    print(f"    A2 m^(2)≤M*≤u^(2):    {'✓' if violations['A2_lower_t2']+violations['A2_upper_t2']==0 else '✗'}")
    print(f"    A3 m↑ 单调:           {'✓' if violations['A3_mono_m']==0 else '✗ %d' % violations['A3_mono_m']}")
    print(f"    A4 u↓ 单调:           {'✓' if violations['A4_mono_u']==0 else '✗ %d' % violations['A4_mono_u']}")
    print(f"    A5 m≤u:               {'✓' if violations['A5_mlt_u']==0 else '✗'}")

# ============================================================
print(f"\n{'='*70}")
print("A3-DETAIL: m^(1) ≥ m^(0) 数值分析")
print("=" * 70)

min_ratio = np.inf
max_ratio = -np.inf
for s in range(200):
    a,b,e,W,V = gen_FCA(s)
    D_max = a+b+e+np.sum(W+V, axis=1)
    sum_w = np.sum(W, axis=1)
    m0 = a / D_max
    u0 = (a + sum_w) / (a + sum_w + b + e)
    
    m1 = (a + W @ m0) / (a + b + e + (W+V) @ u0)
    ratio = m1 / m0
    min_ratio = min(min_ratio, np.min(ratio))
    max_ratio = max(max_ratio, np.max(ratio))

print(f"  m^(1)/m^(0): min={min_ratio:.4f}  max={max_ratio:.4f}  "
      f"{'✓ m↑' if min_ratio >= 1 - 1e-12 else '✗ m可能下降'}")

# Also check why m^(1) ≥ m^(0) - which term dominates?
print(f"\n  最劣种子详细分析:")
worst_s = -1
worst_r = np.inf
for s in range(200):
    a,b,e,W,V = gen_FCA(s)
    D_max = a+b+e+np.sum(W+V, axis=1)
    sum_w = np.sum(W, axis=1)
    m0 = a / D_max
    u0 = (a + sum_w) / (a + sum_w + b + e)
    m1 = (a + W @ m0) / (a + b + e + (W+V) @ u0)
    r = np.min(m1/m0)
    if r < worst_r:
        worst_r = r
        worst_s = s

if worst_s >= 0:
    a,b,e,W,V = gen_FCA(worst_s)
    D_max = a+b+e+np.sum(W+V, axis=1)
    sum_w = np.sum(W, axis=1)
    m0 = a / D_max
    u0 = (a + sum_w) / (a + sum_w + b + e)
    m1 = (a + W @ m0) / (a + b + e + (W+V) @ u0)
    
    for k in range(5):
        print(f"    k={k}: m0={m0[k]:.6f} m1={m1[k]:.6f} ratio={m1[k]/m0[k]:.6f} "
              f"D_max={D_max[k]:.4f}  num0={a[k]:.4f} num1={a[k]+(W@m0)[k]:.4f}")

# ============================================================
print(f"\n{'='*70}")
print("A6-A7: 行和界覆盖率 (T=0,1,2, vs D* vs D_low)")
print("=" * 70)

for label, gen_fn, n_seeds in [("FCA", gen_FCA, 200), ("扩展", gen_extended, 500)]:
    cov_T0 = 0; cov_T1 = 0; cov_T2 = 0; cov_T2_dlow = 0
    max_r_T2 = 0; max_r_T2_s = -1
    
    for s in range(n_seeds):
        a,b,e,W,V = gen_fn(s)
        Mstar = compute_fp(a,b,e,W,V)
        Dstar = a+b+e+(W+V)@Mstar
        D_max = a+b+e+np.sum(W+V, axis=1)
        sum_w = np.sum(W, axis=1)
        
        D_low = a+b+e+(W+V)@(a/D_max)
        
        m = a / D_max
        u = (a + sum_w) / (a + sum_w + b + e)
        
        def check_cov(m_k, u_k, D_k):
            for k in range(5):
                g_m = g_k_val(k, m_k[k], W, V)
                g_u = g_k_val(k, u_k[k], W, V)
                if max(g_m, g_u) >= D_k[k] * (1 - 1e-10):
                    return False
            return True
        
        if check_cov(m, u, Dstar): cov_T0 += 1
        
        m1 = (a + W @ m) / (a + b + e + (W+V) @ u)
        u1 = (a + W @ u) / (a + b + e + (W+V) @ m)
        m1, u1 = np.maximum(m1,0), np.minimum(u1,1)
        if check_cov(m1, u1, Dstar): cov_T1 += 1
        
        m2 = (a + W @ m1) / (a + b + e + (W+V) @ u1)
        u2 = (a + W @ u1) / (a + b + e + (W+V) @ m1)
        m2, u2 = np.maximum(m2,0), np.minimum(u2,1)
        if check_cov(m2, u2, Dstar): cov_T2 += 1
        if check_cov(m2, u2, D_low): cov_T2_dlow += 1
        
        # Track max ratio
        for k in range(5):
            g_m = g_k_val(k, m2[k], W, V)
            g_u = g_k_val(k, u2[k], W, V)
            ratio = max(g_m, g_u) / Dstar[k]
            if ratio > max_r_T2:
                max_r_T2 = ratio
                max_r_T2_s = s
    
    print(f"  {label}: T=0→{cov_T0}/{n_seeds} T=1→{cov_T1}/{n_seeds} T=2→{cov_T2}/{n_seeds}"
          f" (max ratio={max_r_T2:.4f}@{max_r_T2_s})")
    print(f"          T=2+D_low→{cov_T2_dlow}/{n_seeds}")

# ============================================================
print(f"\n{'='*70}")
print("A8: 6.17B α界 + 6.17A₂ RD/CD一致性")
print("=" * 70)

for s in [0, 11, 21, 67, 149]:
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    Dstar = a+b+e+(W+V)@Mstar
    D_max = a+b+e+np.sum(W+V,axis=1)
    D_low_val = a+b+e+(W+V)@(a/D_max)
    
    J = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k != j:
                J[k,j] = (W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k]
    
    alpha = max([sum(abs(J[:,j]) * Dstar / D_low_val) for j in range(5)])
    
    A = np.eye(5) - J
    r = np.sum(np.abs(A - np.diag(np.diag(A))), axis=1)
    c = np.sum(np.abs(A - np.diag(np.diag(A))), axis=0)
    rd_ok = all(r < 1 - 1e-12)
    cd_ok = all(c < 1 - 1e-12)
    
    print(f"  seed {s}: α={alpha:.4f} (6.17B {'✓' if alpha<1 else '✗'})  "
          f"RD={rd_ok} CD={cd_ok} (6.17A₂)  Gershgorin r={(r+c)/2}")

# ============================================================
print(f"\n{'='*70}")
print("A9: 6.17D φ''框架无退化检查")
print("=" * 70)

for s in [0, 11, 21, 67, 149]:
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    theta = Mstar
    Dstar = a+b+e+(W+V)@Mstar
    
    # Verify perfect square decomposition at r=0
    J = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k != j:
                J[k,j] = (W[k,j]*(1-theta[k]) - V[k,j]*theta[k]) / Dstar[k]
    
    # Verify phi''(0) < 0 for 1000 random directions
    violations = 0
    for _ in range(1000):
        u = np.random.randn(5); u = u / np.linalg.norm(u)
        w_vec = W @ u; v_vec = V @ u
        
        ps = sum(((1-theta)*w_vec - theta*v_vec)**2 / (theta*(1-theta)*Dstar**2))
        kl = sum(u**2 / (theta*(1-theta)))
        phi_dd_0 = ps - kl
        
        if phi_dd_0 > 0:
            violations += 1
    
    # Verify eta''/psi'' < 1 at r>0
    worst_ratio = 0
    for _ in range(500):
        u = np.random.randn(5); u = u / np.linalg.norm(u)
        w_vec = W @ u; v_vec = V @ u
        
        for r in np.linspace(0.01, 2.0, 20):
            M = Mstar + r * u
            if np.any(M < 1e-6) or np.any(M > 1-1e-6): continue
            M = np.clip(M, 1e-6, 1-1e-6)
            A = a + W @ M; B = b + V @ M + e; D = A + B
            
            eta = sum(-(w_vec+v_vec)**2/D**2 + theta*w_vec**2/A**2 + (1-theta)*v_vec**2/B**2)
            psi = sum(u**2*(theta/M**2 + (1-theta)/(1-M)**2))
            
            if psi > 0:
                worst_ratio = max(worst_ratio, eta/psi)
    
    print(f"  seed {s}: φ''(0)>0违规={violations}/1000  "
          f"η''/ψ''比值max={worst_ratio:.4f}  "
          f"{'✓' if violations==0 and worst_ratio<1 else '✗'}")

# ============================================================
print(f"\n{'='*70}")
print("A10: Gershgorin链 + sym(I-J) 正定性 全200种子")
print("=" * 70)

for label, gen_fn, n_seeds in [("FCA", gen_FCA, 200), ("扩展", gen_extended, 500)]:
    ger_ok = 0
    sym_ok = 0
    min_eig_min = np.inf
    
    for s in range(n_seeds):
        a,b,e,W,V = gen_fn(s)
        Mstar = compute_fp(a,b,e,W,V)
        Dstar = a+b+e+(W+V)@Mstar
        
        J = np.zeros((5,5))
        for k in range(5):
            for j in range(5):
                if k != j:
                    J[k,j] = (W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k]
        
        A = np.eye(5) - J
        Sym = (A + A.T) / 2
        eig = np.linalg.eigvalsh(Sym)
        min_eig_min = min(min_eig_min, np.min(eig))
        
        r = np.sum(np.abs(A - np.diag(np.diag(A))), axis=1)
        c = np.sum(np.abs(A - np.diag(np.diag(A))), axis=0)
        g_circles = (r + c) / 2
        
        if all(g_circles < 1 - 1e-12): ger_ok += 1
        if all(eig > 1e-12): sym_ok += 1
    
    print(f"  {label}: Gershgorin链通过={ger_ok}/{n_seeds}  "
          f"sym(I-J)≻0={sym_ok}/{n_seeds}  "
          f"λ_min(min)={min_eig_min:.4f}")

print(f"\n{'='*70}")
print("审计结论")
print("=" * 70)
