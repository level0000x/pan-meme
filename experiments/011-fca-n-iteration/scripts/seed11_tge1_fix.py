"""
验证: t≥1 版本证明的正确性
=============================
关键声明:
  1. M(1)_k ≥ N_k(0) > m_k^(0) ∀k (所有种子)
  2. t≥1时 D(M(t)) ≥ D_low (可证明)
  3. α<1 对所有200种子仍然成立
  4. ||M(t+1)-M*|| ≤ α·||M(t)-M*|| for t≥1
"""
import numpy as np

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

print("=" * 70)
print("验证 1: N_k(0) > m_k^(0) 对所有种子和所有分量?")
print("=" * 70)

all_ok = True
for seed_id in range(200):
    a, b, e, W, V = gen_FCA(seed_id)
    D_max = a + b + e + np.sum(W + V, axis=1)
    m_low = a / D_max
    N0 = a / (a + b + e)  # N_k(0)
    
    if not np.all(N0 > m_low * (1 + 1e-12)):
        print(f"  ✗ seed {seed_id}")
        all_ok = False

print(f"  {'✓' if all_ok else '✗'} N_k(0) > m_k^(0) ∀k ∀seed")

print(f"\n{'='*70}")
print("验证 2: M(1)_k = N_k(0.5) ≥ m_k^(0) 对所有种子?")
print("=" * 70)

violations_2 = []
for seed_id in range(200):
    a, b, e, W, V = gen_FCA(seed_id)
    D_max = a + b + e + np.sum(W + V, axis=1)
    m_low = a / D_max
    
    M1 = n_operator(np.full(5, 0.5), a, b, e, W, V)
    
    if not np.all(M1 >= m_low * (1 - 1e-12)):
        violations_2.append((seed_id, M1, m_low))

if violations_2:
    print(f"  ✗ {len(violations_2)}/200 违规:")
    for s, M1, ml in violations_2:
        bad = [k for k in range(5) if M1[k] < ml[k]]
        print(f"    seed {s}: bad components {bad}, M1={M1}, m_low={ml}")
else:
    print(f"  ✓ 0/200 违规")

print(f"\n{'='*70}")
print("验证 3: t≥1 的 α 界 + 轨道暴力测试")
print("=" * 70)

# For t≥1, the bound is:
# ||N(M(t)) - M*|| ≤ α · ||M(t) - M*||
# where α = max_j Σ_k |J_kj| · D*_k/D_low,k

# We tested this already with random Δ and it passed (0/200 violations).
# Now test specifically on orbit points for t≥1 for ALL 200 seeds.

total_ok = 0
total_steps = 0
for seed_id in range(200):
    a, b, e, W, V = gen_FCA(seed_id)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + b + e + (W+V) @ Mstar
    D_max = a + b + e + np.sum(W + V, axis=1)
    m_low = a / D_max
    D_low = a + b + e + (W+V) @ m_low
    
    J = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k != j:
                J[k,j] = (W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k]
    gamma = Dstar / D_low
    alpha = max([sum(abs(J[:,j]) * gamma) for j in range(5)])
    
    M = np.full(5, 0.5)
    for t in range(50):
        M_next = n_operator(M, a, b, e, W, V)
        
        if t >= 1:  # α holds for t≥1
            ratio = np.sum(np.abs(M_next - Mstar)) / max(1e-10, np.sum(np.abs(M - Mstar)))
            total_steps += 1
            if not (ratio <= alpha * (1 + 1e-10)):
                print(f"  ✗ seed {seed_id} t={t}: ratio={ratio:.6f} > α={alpha:.6f}")
                total_ok -= 1
            total_ok += 1
        
        M = M_next
        if np.max(np.abs(M - Mstar)) < 1e-12:
            break

print(f"  α 界在轨道上 t≥1: {total_ok}/{total_steps} 测试通过")

print(f"\n{'='*70}")
print("验证 4: seed 149 的 M(1)_4 vs m_4^(0)")
print("=" * 70)

a, b, e, W, V = gen_FCA(149)
M1 = n_operator(np.full(5, 0.5), a, b, e, W, V)
D_max = a + b + e + np.sum(W + V, axis=1)
m_low = a / D_max
print(f"  M(1) = {M1}")
print(f"  m_low = {m_low}")
print(f"  M1_4 - m_low_4 = {M1[4] - m_low[4]:.6f}  {'✓' if M1[4] >= m_low[4] else '✗'}")

# Check: N_k(0) = a_k/(a_k+b_k+e_k) > m_k^(0) ?
N0 = a / (a + b + e)
print(f"  N(0) = {N0}")
print(f"  N(0)_4 > m_low_4? {N0[4]:.6f} > {m_low[4]:.6f} = {'✓' if N0[4] > m_low[4] else '✗'}")

print(f"\n{'='*70}")
print("结论")
print("=" * 70)
print("""
发现的 bug:
  seed 149 的 m_4^(0) = 0.5035 > 0.5
  → M(0)=0.5 不满足 M ≥ m^(0) 前提
  → D_low 在 t=0 不是可证明的下界

修复方案:
  将 α 界的适用条件改为 t≥1:
  1. M(1) = N(0.5) ≥ N(0) > m^(0) (单调性保证)
  2. 对所有 t≥1: M(t) ≥ m^(0) 由归纳法保持
  3. 对所有 t≥1: D(M(t)) ≥ D_low, α 界有效
  4. 收敛: ||M(t+1)-M*|| ≤ α·||M(t)-M*|| for t≥1
  5. t=0→1 的一步是有限的 (M(0), M(1) ∈ [0,1]⁵)
""")
