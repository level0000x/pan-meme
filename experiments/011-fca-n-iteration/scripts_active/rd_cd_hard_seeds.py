"""
诊断剩余硬种子的参数模式
==============================
分析 行和界 + 迭代M*界 仍不闭合的种子，找统一特征。
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

def gen_extended(seed):
    rs = np.random.RandomState(seed + 10000)
    a = rs.uniform(0.005, 0.5, 5)
    b = rs.uniform(0.005, 0.5, 5)
    e = rs.uniform(0.0005, 0.5, 5)
    W = rs.uniform(0.005, 0.3, (5, 5))
    V = rs.uniform(0.005, 0.3, (5, 5))
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
        if np.max(np.abs(m_low_new - m_low)) < 1e-10:
            break
        m_low = m_low_new
        m_high = m_high_new
    return m_low, m_high, D_low

def rowsum_rd_bound_iter(a, b, e, W, V, max_iter=20):
    m_low, m_high, D_low = iterative_Mstar_bounds(a, b, e, W, V, max_iter)
    rd_bound = np.zeros(5)
    for k in range(5):
        # Tightened numerator using M* interval
        num_low = sum([abs(W[k,j]*(1-m_low[k]) - V[k,j]*m_low[k]) for j in range(5) if j != k])
        num_high = sum([abs(W[k,j]*(1-m_high[k]) - V[k,j]*m_high[k]) for j in range(5) if j != k])
        num_bound = max(num_low, num_high)
        rd_bound[k] = num_bound / D_low[k]
    return m_low, m_high, D_low, max(rd_bound), np.argmax(rd_bound)

print("=" * 70)
print("硬种子分析: 行和界 + 迭代M* 仍不闭合的种子")
print("=" * 70)

# Collect hard seeds for both domains
hard_fca = []
for seed in range(200):
    a, b, e, W, V = gen_FCA(seed)
    m_low, m_high, D_low, bound_val, worst_k = rowsum_rd_bound_iter(a, b, e, W, V)
    if bound_val >= 1:
        Mstar = compute_fp(a, b, e, W, V)
        Dstar = (a + W @ Mstar) + (b + V @ Mstar) + e
        # Actual RD
        J_act = np.zeros((5,5))
        for k in range(5):
            for j in range(5):
                if k != j:
                    J_act[k,j] = abs((W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k])
        rd_actual = max(J_act.sum(axis=1))
        
        # Simple row-sum bound
        rd_rowsum_Dv2 = np.zeros(5)
        D_low_v2 = a + b + e + (W+V) @ (a / (a + b + e + np.sum(W+V, axis=1)))
        for k in range(5):
            rd_rowsum_Dv2[k] = max(W[k].sum(), V[k].sum()) / D_low_v2[k]
        
        hard_fca.append((bound_val, worst_k, seed, rd_actual, max(rd_rowsum_Dv2), 
                         Dstar[worst_k], D_low[worst_k],
                         a[worst_k], b[worst_k], e[worst_k],
                         sum(W[worst_k]), sum(V[worst_k]),
                         m_low[worst_k], m_high[worst_k], Mstar[worst_k]))

hard_fca.sort(key=lambda x: -x[0])

print(f"\n{'='*70}")
print(f"FCA 硬种子 (bound ≥ 1, 共 {len(hard_fca)} 个)")
print(f"{'='*70}")

print(f"\n{'seed':>5} {'bound':>7} {'actual':>7} {'rowsum':>7} {'k':>3} "
      f"{'D*/Dlow':>8} {'a_k':>7} {'b_k':>7} {'eps_k':>7} {'S_w':>7} {'S_v':>7} "
      f"{'m_low':>7} {'m_high':>7} {'M*':>7}")

for bound_val, worst_k, seed, rd_act, rd_rs, Dstar, Dlow, ak, bk, ek, Sw, Sv, ml, mh, Mk in hard_fca[:25]:
    print(f"{seed:5d} {bound_val:7.3f} {rd_act:7.3f} {rd_rs:7.3f} {worst_k:3d} "
          f"{Dstar/Dlow:8.2f} {ak:7.3f} {bk:7.3f} {ek:7.3f} {Sw:7.3f} {Sv:7.3f} "
          f"{ml:7.3f} {mh:7.3f} {Mk:7.3f}")

# Pattern analysis
print(f"\n{'='*70}")
print("模式分析")
print(f"{'='*70}")

# Key metrics
print(f"\n关键统计 (worst rows of hard seeds):")
ratios_Sw_to_Dlow = [h[9] / h[6] for h in hard_fca]  # Σw / Dlow
ratios_Sv_to_Dlow = [h[10] / h[6] for h in hard_fca]  # Σv / Dlow
ratios_coupling_to_self = [(h[9]+h[10]) / (h[7]+h[8]+h[9]) for h in hard_fca]  # (Σw+Σv)/(a+b+ε)
Dstar_Dlow_ratios = [h[5] / h[6] for h in hard_fca]
m_widthes = [h[12] - h[11] for h in hard_fca]

print(f"  max(Σw,Σv) / D_low:  min={min(ratios_Sw_to_Dlow):.3f}  max={max(ratios_Sw_to_Dlow):.3f}  mean={np.mean(ratios_Sw_to_Dlow):.3f}")
print(f"  (Σw+Σv) / (a+b+ε):  min={min(ratios_coupling_to_self):.3f}  max={max(ratios_coupling_to_self):.3f}  mean={np.mean(ratios_coupling_to_self):.3f}")
print(f"  D* / D_low:         min={min(Dstar_Dlow_ratios):.3f}  max={max(Dstar_Dlow_ratios):.3f}  mean={np.mean(Dstar_Dlow_ratios):.3f}")
print(f"  M* width (m_high-m_low): min={min(m_widthes):.3f}  max={max(m_widthes):.3f}  mean={np.mean(m_widthes):.3f}")

# Compare with soft seeds
print(f"\n{'seed':>5} {'sw/Dlow':>8} {'(sw+sv)/(a+b+e)':>17} {'D*/Dlow':>8} {'M*width':>8} {'M*-m_low':>10} {'m_high-M*':>10}")
for bound_val, worst_k, seed, rd_act, rd_rs, Dstar, Dlow, ak, bk, ek, Sw, Sv, ml, mh, Mk in hard_fca[:25]:
    print(f"{seed:5d} {Sw/Dlow:8.3f} {(Sw+Sv)/(ak+bk+ek):17.3f} {Dstar/Dlow:8.2f} {mh-ml:8.3f} {Mk-ml:10.3f} {mh-Mk:10.3f}")

# Check: for seed 11, why is it so hard?
seed11_params = gen_FCA(11)
a11, b11, e11, W11, V11 = seed11_params
M11 = compute_fp(*seed11_params)
D11 = (a11 + W11 @ M11) + (b11 + V11 @ M11) + e11

print(f"\n{'='*70}")
print("Seed 11 详细分析")
print(f"{'='*70}")
print(f"\nM* = {M11}")
print(f"D* = {D11}")
print(f"D_min = {a11 + b11 + e11}")
print(f"D_low_v2 = {a11 + b11 + e11 + (W11+V11) @ (a11 / (a11 + b11 + e11 + np.sum(W11+V11, axis=1)))}")
print(f"\nW sum per row: {W11.sum(axis=1)}")
print(f"V sum per row: {V11.sum(axis=1)}")
print(f"max(W,V) sum per row: {np.maximum(W11, V11).sum(axis=1)}")

# Row 1 (B) is the worst
k = 1
print(f"\n行 {k} (最劣):")
print(f"  Σw_k = {W11[k].sum():.4f}, Σv_k = {V11[k].sum():.4f}")
print(f"  a_k = {a11[k]:.4f}, b_k = {b11[k]:.4f}, ε_k = {e11[k]:.4f}")
print(f"  D_min = {a11[k]+b11[k]+e11[k]:.4f}")
print(f"  D_low_v2 = {(a11 + b11 + e11 + (W11+V11) @ (a11 / (a11 + b11 + e11 + np.sum(W11+V11, axis=1))))[k]:.4f}")
print(f"  D* = {D11[k]:.4f}")
print(f"  max(Σw, Σv) = {max(W11[k].sum(), V11[k].sum()):.4f}")

# Numerator analysis
for x in np.linspace(0, 1, 21):
    num = sum([abs(W11[k,j]*(1-x) - V11[k,j]*x) for j in range(5) if j != k])
    if x == 0 or x == 1 or abs(x - M11[k]) < 0.01:
        print(f"  M_k = {x:.2f}: numerator = {num:.4f}")

# Can we get a better D* bound using the FP structure?
print(f"\n  Actual M*_k slope numerator/(a+b+e):")
print(f"    M*_k = {M11[k]:.4f} = A*_k / D*_k = {(a11[k] + sum(W11[k]*M11)):.4f} / {D11[k]:.4f}")
print(f"    D* lower bound from M*_k formula using M*_j lower bounds:")
M_low = np.clip(a11 / (a11 + b11 + e11 + np.sum(W11+V11, axis=1)), 0, 1)
D_low_v2 = a11 + b11 + e11 + (W11+V11) @ M_low
print(f"    M_low    = {M_low}")
print(f"    D_low_v2 = {D_low_v2}")
print(f"    A_low    = {a11 + W11 @ M_low}")
print(f"    M_low_v2 = A_low / (A_low + b11 + e11 + V11 @ M_low) = {A_low / (A_low + b11 + e11 + V11 @ M_low)}")
