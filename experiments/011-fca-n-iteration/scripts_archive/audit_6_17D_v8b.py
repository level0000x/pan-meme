"""
第八轮审计 - 精简版 (减少计算量)
=================================
重点:
  [A] Cross/D 公式——δ+τ vs δ-τ (已完成)
  [B] Pinsker紧性 (已完成)
  [C] MVT验证 (已完成)
  [D] c_KL + 安全半径 (精简)
  [E] ‖M_H‖₂ / Gershgorin (精简)
  [H] 6.17C安全半径 (精简)
  [I] 数值盲测 (精简)
  [K] 新增：Bregman恒等式正负号独立验证
  [L] 新增：简化ΔV公式代数验证
"""
import numpy as np
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

def D_KL_b(p, q):
    eps = 1e-15
    pp = np.clip(p, eps, 1 - eps)
    qq = np.clip(q, eps, 1 - eps)
    return pp * np.log(pp / qq) + (1 - pp) * np.log((1 - pp) / (1 - qq))

def D_KL_vec(M1, M2):
    return np.sum([D_KL_b(M1[k], M2[k]) for k in range(5)])

# 预计算固定点缓存
print("预计算 200 组 FCA 参数...")
seeds_data = []
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
    Msym = 0.5 * (MH + MH.T)
    seeds_data.append({
        'a': a, 'b': b, 'e': e, 'W': W, 'V': V,
        'Mstar': Mstar, 'Dstar': Dstar, 'J': J,
        'H': H, 'MH': MH, 'Msym': Msym,
        'norm_MH': np.linalg.norm(MH, 2),
        'lambda_max_Msym': np.max(np.linalg.eigvalsh(Msym)),
        'lambda_min_IminusMsym': np.min(np.linalg.eigvalsh(np.eye(5) - Msym)),
        'r_max': max(np.sum(np.abs(MH), axis=1)),
        'c_max': max(np.sum(np.abs(MH), axis=0)),
        'lambda_min_HminusJTHJ': np.linalg.eigvalsh(H - J.T @ H @ J)[0],
        'lambda_min_sym_IminusJ': np.linalg.eigvalsh(0.5 * (np.eye(5) - J + (np.eye(5) - J).T))[0],
    })

print("完成！\n")

# ============================================================
# [D] c_KL 精简计算 (50种子 × 30方向 × 20半径)
# ============================================================
print("=" * 70)
print("[D] c_KL 精简计算 (50种子×30方向×20半径)")
c_kl_estimates = []
for s_idx in range(0, 200, 4):  # 50 seeds
    sd = seeds_data[s_idx]
    Mstar, J, H = sd['Mstar'], sd['J'], sd['H']
    a, b, e, W, V = sd['a'], sd['b'], sd['e'], sd['W'], sd['V']
    
    max_c = 0.0
    for _ in range(30):
        direction = np.random.randn(5)
        direction /= np.linalg.norm(direction)
        for r in np.logspace(-2.5, 0.5, 20):
            v = r * direction
            M = np.clip(Mstar + v, 1e-15, 1 - 1e-15)
            N = n_operator(M, a, b, e, W, V)
            M_s = np.clip(M, 1e-12, 1 - 1e-12)
            N_s = np.clip(N, 1e-12, 1 - 1e-12)
            f_v = D_KL_vec(Mstar, N_s) - D_KL_vec(Mstar, M_s)
            quad = 0.5 * v @ (J.T @ H @ J - H) @ v
            R3 = f_v - quad
            if r > 1e-3 and abs(R3) > 1e-15:
                c_candidate = abs(R3) / (r ** 3)
                if c_candidate > max_c:
                    max_c = c_candidate
    c_kl_estimates.append(max_c)

print(f"  c_KL: [{min(c_kl_estimates):.2f}, {max(c_kl_estimates):.2f}] median={np.median(c_kl_estimates):.2f}")
print(f"  文档: [7.8, 27.0]")
# Compute λ_min(H - J^T H J)
hmins = [sd['lambda_min_HminusJTHJ'] for sd in seeds_data]
print(f"  λ_min(H-J^THJ): [{min(hmins):.2f}, {max(hmins):.2f}]")
print(f"  文档: [3.8, 4.2]")
print(f"  安全半径: λ_min/c_KL ≈ [{min(hmins)/max(c_kl_estimates):.2f}, {max(hmins)/min(c_kl_estimates):.2f}]")
print()

# ============================================================
# [E] ‖M_H‖₂ 精简输出 (已预计算)
# ============================================================
print("=" * 70)
print("[E] ‖M_H‖₂ / Gershgorin / λ_max (200种子)")
norms = [sd['norm_MH'] for sd in seeds_data]
gersh = [np.sqrt(sd['r_max'] * sd['c_max']) for sd in seeds_data]
symax = [sd['lambda_max_Msym'] for sd in seeds_data]
symin = [sd['lambda_min_IminusMsym'] for sd in seeds_data]

print(f"  ‖M_H‖₂:         [{min(norms):.4f}, {max(norms):.4f}] mean={np.mean(norms):.4f}")
print(f"  文档:            [0.090, 0.252] mean=0.159")
print(f"  √(r·c):          [{min(gersh):.4f}, {max(gersh):.4f}] mean={np.mean(gersh):.4f}")
print(f"  文档:            [0.129, 0.341] mean=0.226")
print(f"  λ_max(M_sym):    [{min(symax):.4f}, {max(symax):.4f}]")
print(f"  文档:            < 0.197")
print(f"  λ_min(I-M_sym):  [{min(symin):.4f}, {max(symin):.4f}]")
print(f"  文档:            ≥ 0.803")
print(f"  全部<1: {'✓' if max(norms)<1.0 else '✗'}")
compression = [g/n for g, n in zip(gersh, norms)]
print(f"  压缩比:          [{min(compression):.2f}, {max(compression):.2f}]")
print(f"  文档:            1.18-1.68×")
print()

# ============================================================
# [H] 6.17C 安全半径精简 (25种子×20方向×15半径)
# ============================================================
print("=" * 70)
print("[H] 6.17C c_max 精简计算 (25种子×20方向×15半径)")
c_max_6_17C = []
for s_idx in range(0, 200, 8):  # 25 seeds
    sd = seeds_data[s_idx]
    Mstar, J = sd['Mstar'], sd['J']
    a, b, e, W, V = sd['a'], sd['b'], sd['e'], sd['W'], sd['V']
    IminusJ = np.eye(5) - J
    
    max_c = 0.0
    for _ in range(20):
        direction = np.random.randn(5)
        direction /= np.linalg.norm(direction)
        for r in np.logspace(-2.5, 0.3, 15):
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

lmin_IJ = [sd['lambda_min_sym_IminusJ'] for sd in seeds_data]
print(f"  λ_min(sym(I-J)): [{min(lmin_IJ):.4f}, {max(lmin_IJ):.4f}]")
print(f"  文档: [0.807, 0.970]")
print(f"  实测 c_max(6.17C): [{min(c_max_6_17C):.2f}, {max(c_max_6_17C):.2f}] median={np.median(c_max_6_17C):.2f}")
print(f"  文档声称: ~0.05-0.15")
print(f"  安全半径 ≈ [{min(lmin_IJ)/max(c_max_6_17C):.2f}, {max(lmin_IJ)/min(c_max_6_17C):.2f}]")
print()

# ============================================================
# [I] 数值盲测 (简化: 50种子×50点)
# ============================================================
print("=" * 70)
print("[I] 数值盲测 (50种子×50点)")
dVs = []; ratios_KL = []; lie_vals = []
for s_idx in range(0, 200, 4):
    sd = seeds_data[s_idx]
    Mstar = sd['Mstar']
    a, b, e, W, V = sd['a'], sd['b'], sd['e'], sd['W'], sd['V']
    for _ in range(50):
        M = np.random.uniform(0.05, 0.95, 5)
        N = n_operator(M, a, b, e, W, V)
        M_s = np.clip(M, 1e-12, 1-1e-12)
        N_s = np.clip(N, 1e-12, 1-1e-12)
        dV = D_KL_vec(Mstar, N_s) - D_KL_vec(Mstar, M_s)
        dVs.append(dV)
        V_M = D_KL_vec(Mstar, M_s)
        V_N = D_KL_vec(Mstar, N_s)
        if V_M > 1e-10:
            ratios_KL.append(V_N / V_M)
        grad = (M - Mstar) / (M * (1 - M))
        lie = grad @ (N - M)
        lie_vals.append(lie)

print(f"  ΔV: mean={np.mean(dVs):.2f}, min={np.min(dVs):.4f}, max={np.max(dVs):.6f}")
print(f"  文档: mean≈-1.91, min≈-9.1, max≈-0.001 ~ -0.01")
print(f"  V(N)/V(M): median={np.median(ratios_KL):.4f}")
print(f"  文档: 中位数约0.004")
print(f"  Lie导数: max={np.max(lie_vals):.6f}")
print(f"  文档: 零违规")
print(f"  零违规(ΔV): {'✓' if max(dVs)<0 else '✗'}")
print(f"  零违规(Lie): {'✓' if max(lie_vals)<0 else '✗'}")
print()

# ============================================================
# [K] Bregman恒等式正负号独立验证 ★新增★
# ============================================================
print("=" * 70)
print("[K] Bregman恒等式正负号独立验证")
print("""
  Bregman恒等式 (以二值熵为生成元):
    D(a||c) = D(a||b) + D(b||c) + (a-b)(logit b - logit c)
  
  代入 a=M*, b=N, c=M:
    D(M*||M) = D(M*||N) + D(N||M) + (M*-N)(logit N - logit M)
  
  → V_KL(M) = V_KL(N) + D(N||M) + (M*-N)(logit N - logit M)
  → V_KL(N) - V_KL(M) = -D(N||M) - (M*-N)(logit N - logit M)
  → ΔV = -D(N||M) + (M*-N)(logit M - logit N)
  
  文档公式:
    ΔV = -D_KL(N||M) + Σ(M*-N)(logit M - logit N)  ✓
  
  Case A: N在M与M*之间 → (M*-N)(logit M - logit N) < 0
  → ΔV ≤ -D_KL(N||M) < 0  ✓
  
  Case B: N穿越M* → 交叉项 > 0, 但有上界
  
  数值验证: 恒等式在数值上是否精确成立
""")

max_bregman_err = 0.0
max_dV_simple_err = 0.0
for s_idx in range(0, 200):
    sd = seeds_data[s_idx]
    Mstar = sd['Mstar']
    a, b, e, W, V = sd['a'], sd['b'], sd['e'], sd['W'], sd['V']
    for _ in range(10):
        M = np.random.uniform(0.05, 0.95, 5)
        N = n_operator(M, a, b, e, W, V)
        M_s = np.clip(M, 1e-12, 1-1e-12)
        N_s = np.clip(N, 1e-12, 1-1e-12)
        
        # 直接 ΔV
        dV_direct = D_KL_vec(Mstar, N_s) - D_KL_vec(Mstar, M_s)
        
        # Bregman 分解
        D_NM = D_KL_vec(N_s, M_s)
        cross_sum = 0.0
        for k in range(5):
            cross_sum += (Mstar[k] - N_s[k]) * (np.log(M_s[k]/(1-M_s[k])) - np.log(N_s[k]/(1-N_s[k])))
        dV_bregman = -D_NM + cross_sum
        
        # 简化 ΔV_k 公式
        dV_simple = 0.0
        for k in range(5):
            dV_simple += Mstar[k] * np.log(M_s[k]/N_s[k]) + (1-Mstar[k]) * np.log((1-M_s[k])/(1-N_s[k]))
        
        err1 = abs(dV_direct - dV_bregman)
        err2 = abs(dV_direct - dV_simple)
        if err1 > max_bregman_err:
            max_bregman_err = err1
        if err2 > max_dV_simple_err:
            max_dV_simple_err = err2

print(f"  Bregman分解误差: max={max_bregman_err:.2e}")
print(f"  简化ΔV公式误差: max={max_dV_simple_err:.2e}")
print(f"  两者皆精确: {'✓' if max_bregman_err<1e-13 and max_dV_simple_err<1e-13 else '✗'}")
print()

# ============================================================
# [L] 简化ΔV公式代数验证 ★新增★
# ============================================================
print("=" * 70)
print("[L] 简化ΔV公式的极限行为验证")
print("""
  ΔV_k = M*_k ln(M_k/N_k) + (1-M*_k)ln((1-M_k)/(1-N_k))
  
  测试极限情况: M*_k→0, M*_k→1, M_k→N_k, M_k→0, M_k→1 等
""")

test_cases = [
    (0.5, 0.3, 0.35, "一般Case A"),
    (0.5, 0.3, 0.7, "超调Case B"),
    (0.1, 0.2, 0.15, "小M* Case A"),
    (0.9, 0.8, 0.85, "大M* Case A"),
    (0.5, 0.3, 0.3, "M=N (应ΔV=0)"),
    (0.5, 0.3, 0.300001, "M≈N"),
    (0.01, 0.5, 0.3, "极小M*"),
    (0.99, 0.5, 0.7, "极大M*"),
]

for Mstar_k, M_k, N_k, label in test_cases:
    dV_direct = D_KL_b(Mstar_k, N_k) - D_KL_b(Mstar_k, M_k)
    dV_simple = Mstar_k * np.log(np.clip(M_k,1e-15,None) / np.clip(N_k,1e-15,None)) + \
                (1-Mstar_k) * np.log(np.clip(1-M_k,1e-15,None) / np.clip(1-N_k,1e-15,None))
    err = abs(dV_direct - dV_simple)
    print(f"  {label}: M*={Mstar_k}, M={M_k}, N={N_k}: "
          f"ΔV_direct={dV_direct:.6f}, ΔV_simple={dV_simple:.6f}, err={err:.2e} {'✓' if err<1e-14 else '✗'}")

print()

# ============================================================
# 总结
# ============================================================
print("=" * 70)
print("第八轮审计总结")
print("=" * 70)

issues = []
issues.append(f"[D] c_KL: [{min(c_kl_estimates):.1f}, {max(c_kl_estimates):.1f}] "
              f"{'✓' if min(c_kl_estimates)>5 and max(c_kl_estimates)<35 else '⚠️'}")
issues.append(f"[E] ‖M_H‖₂: [{min(norms):.4f}, {max(norms):.4f}] "
              f"{'✓' if abs(min(norms)-0.090)<0.01 and abs(max(norms)-0.252)<0.02 else '⚠️'}")
issues.append(f"[H] 6.17C c_max: [{min(c_max_6_17C):.2f}, {max(c_max_6_17C):.2f}] median={np.median(c_max_6_17C):.2f}")
issues.append(f"[I] ΔV 零违规: {'✓' if max(dVs)<0 else '✗'}")
issues.append(f"[K] Bregman恒等式精确: {'✓' if max_bregman_err<1e-13 else '✗'}")
issues.append(f"[L] 简化ΔV公式精确: {'✓' if max_dV_simple_err<1e-13 else '✗'}")

for issue in issues:
    print(f"  {issue}")

print(f"\n★ 核心发现 ★")
print(f"  [A] Cross/D公式分母: 文档使用 |δ+τ|，正确应为 |δ-τ|")
print(f"      → 不影响经验max=2.17 (经验值独立计算)")
print(f"      → 但公式推导本身有代数错误，需修正")
if np.median(c_max_6_17C) > 0.3:
    print(f"  [H] 6.17C c_max 实测 {np.median(c_max_6_17C):.1f} >> 文档声称 0.05-0.15")
    print(f"      → 6.17C安全半径被高估")
print()
