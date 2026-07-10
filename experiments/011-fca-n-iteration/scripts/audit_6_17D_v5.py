"""
第五轮审计 - 文档内部一致性检查
===============================
检查清单:
  [A] 定理索引中 ■ 项的实际计数 (声称34 vs 实际?)
  [B] "l₁ 比值 0.09" 的来源 — 遍历6.17B确认是否有出处
  [C] "KL 比值 0.008" 的实证重验
  [D] "Gershgorin 压缩 1.35-1.68×" 数值验证
  [E] 超调分量统计 (1246+502 vs 60684 是否自洽)
  [F] Lie导数 max = −0.032 的实证重验
  [G] 5分量超调 1.5-7% 概率验证
  [H] M_ℋ 构造中 J_kk=0 的直接检查
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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

def D_KL(p, q):
    s = 0.0
    for k in range(len(p)):
        pk, qk = float(p[k]), float(q[k])
        if pk > 1e-300 and qk > 1e-300:
            s += pk * np.log(pk / qk)
        if pk < 1.0 - 1e-300 and qk < 1.0 - 1e-300:
            s += (1.0 - pk) * np.log((1.0 - pk) / (1.0 - qk))
    return s

# ============================================================
# [B] & [C] KL 比值 vs l₁ 比值 — 独立重算
# ============================================================
print("=" * 70)
print("[B] & [C] KL 比值 (0.008) 和 l₁ 比值 (0.09)")
print("=" * 70)

kl_ratios = []  # |ΔV| / V(M)
l1_ratios = []  # ||N-M*||₁ / ||M-M*||₁

for seed in range(200):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    rs = np.random.RandomState(seed * 98765 + 1)
    
    for _ in range(20):
        v = (rs.rand(5) - 0.5) * 1.0
        M = np.clip(Mstar + v, 0.001, 0.999)
        N = n_operator(M, a, b, e, W, V)
        if np.max(np.abs(N - M)) < 1e-12:
            continue
        
        dV = D_KL(Mstar, N) - D_KL(Mstar, M)
        V_M = D_KL(Mstar, M)
        if V_M > 1e-12:
            kl_ratios.append(abs(dV) / V_M)
        
        l1_new = np.sum(np.abs(N - Mstar))
        l1_old = np.sum(np.abs(M - Mstar))
        if l1_old > 1e-12:
            l1_ratios.append(l1_new / l1_old)

kl_ratios = np.array(kl_ratios)
l1_ratios = np.array(l1_ratios)

print(f"  KL   |ΔV|/V(M):     mean={kl_ratios.mean():.4f}, median={np.median(kl_ratios):.4f}")
print(f"  l₁   ||N-M*||/||M-M*||: mean={l1_ratios.mean():.4f}, median={np.median(l1_ratios):.4f}")
print(f"  文档声称:                KL=0.008, l₁=0.09")
print(f"  KL 实际中位数:           {np.median(kl_ratios):.4f}")
print(f"  l₁ 实际中位数:           {np.median(l1_ratios):.4f}")
print()

# 解释: 0.008 可能是 KL 比值的均值/中位数，0.09 类似
# 但 6.17B 说的是最大收缩比 ≤ 0.54，平均/中位数约 0.09-0.18
# 如果 0.09 是均值，0.008 也是均值，则 0.008 < 0.09 确实表示 KL 收缩更强

kl_median = np.median(kl_ratios)
l1_median = np.median(l1_ratios)
print(f"  KL 收缩倍数 vs l₁: {l1_median/kl_median:.1f}×")
if l1_median > kl_median:
    print(f"  ✓ KL 确实比 l₁ 收缩更强")
else:
    print(f"  ✗ 方向反了!")
print()

# ============================================================
# [D] Gershgorin 压缩比
# ============================================================
print("=" * 70)
print("[D] Gershgorin 压缩比 (声称 1.35-1.68×)")
print("=" * 70)

compressions = []
for seed in range(200):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + W @ Mstar + b + V @ Mstar + e
    h = 1.0 / (Mstar * (1.0 - Mstar))
    Hsqrt = np.diag(np.sqrt(h))
    Hinvsqrt = np.diag(1.0 / np.sqrt(h))
    
    J = np.zeros((5, 5))
    for k in range(5):
        for j in range(5):
            J[k, j] = (W[k, j]*(1-Mstar[k]) - Mstar[k]*V[k, j]) / Dstar[k]
    
    Mh = Hsqrt @ J @ Hinvsqrt
    actual = np.linalg.norm(Mh, 2)
    
    rM = max(sum(abs(Mh[k, j]) for j in range(5) if j != k) for k in range(5))
    cM = max(sum(abs(Mh[i, j]) for i in range(5) if i != j) for j in range(5))
    ger = np.sqrt(rM * cM)
    
    compressions.append(ger / actual)

comp = np.array(compressions)
print(f"  Gershgorin/Actual ratio: [{comp.min():.4f}, {comp.max():.4f}], mean={comp.mean():.4f}")
print(f"  文档声称: [1.35, 1.68]")
print(f"  匹配: {'YES ✓' if abs(comp.min()-1.35)<0.05 and abs(comp.max()-1.68)<0.05 else '偏差 ✗'}")
print()

# ============================================================
# [E] 超调分量统计
# ============================================================
print("=" * 70)
print("[E] 超调分量统计 (文档: 1246+502 vs 60684)")
print("=" * 70)

total_components = 0
overshoot_low = 0    # M*_k < 0.25
overshoot_mid = 0    # M*_k ∈ [0.25, 0.75]
overshoot_high = 0   # M*_k > 0.75
non_overshoot = 0
cross_gt_D_low = 0
cross_gt_D_mid = 0
cross_gt_D_high = 0
total_overshoot_low = 0
total_overshoot_mid = 0
total_overshoot_high = 0

for seed in range(100):
    a, b, e, W, V = gen_FCA(seed * 3)
    Mstar = compute_fp(a, b, e, W, V)
    rs = np.random.RandomState(seed * 55555 + 99)
    
    for _ in range(500):
        M = np.clip(Mstar + (rs.rand(5)-0.5)*1.0, 0.001, 0.999)
        N = n_operator(M, a, b, e, W, V)
        if np.max(np.abs(N-M)) < 1e-12:
            continue
        
        for k in range(5):
            total_components += 1
            if (M[k]-Mstar[k]) * (N[k]-Mstar[k]) < 0:  # overshoot
                if Mstar[k] < 0.25:
                    overshoot_low += 1
                    total_overshoot_low += 1
                elif Mstar[k] > 0.75:
                    overshoot_high += 1
                    total_overshoot_high += 1
                else:
                    overshoot_mid += 1
                    total_overshoot_mid += 1
                
                cross = (Mstar[k]-N[k]) * (np.log(M[k]/(1-M[k])) - np.log(N[k]/(1-N[k])))
                dkl = D_KL(np.array([N[k]]), np.array([M[k]]))
                if dkl > 1e-15 and cross > dkl:
                    if Mstar[k] < 0.25:
                        cross_gt_D_low += 1
                    elif Mstar[k] > 0.75:
                        cross_gt_D_high += 1
                    else:
                        cross_gt_D_mid += 1
            else:
                non_overshoot += 1

print(f"  总分量数: {total_components}")
print(f"  非超调分量: {non_overshoot}")
print(f"  超调分量:   {overshoot_low + overshoot_mid + overshoot_high}")
print(f"    M*<0.25: {overshoot_low}, cross>D: {cross_gt_D_low}/{overshoot_low} = {100*cross_gt_D_low/max(1,overshoot_low):.2f}%")
print(f"    M*∈[0.25,0.75]: {overshoot_mid}, cross>D: {cross_gt_D_mid}/{overshoot_mid} = {100*cross_gt_D_mid/max(1,overshoot_mid):.2f}%")
print(f"    M*>0.75: {overshoot_high}, cross>D: {cross_gt_D_high}/{overshoot_high} = {100*cross_gt_D_high/max(1,overshoot_high):.2f}%")
print(f"  文档: 1246+502 vs 60684 (超调分量, M*<0.25+M*>0.75 vs total)")
print(f"  文档: 5.70%/3.98% (cross>D in M*<0.25 / M*>0.75)")
print()

# ============================================================
# [F] Lie 导数 max = −0.032
# ============================================================
print("=" * 70)
print("[F] Lie 导数 max 实证重验")
print("=" * 70)

all_lie = []
for seed in range(100):
    a, b, e, W, V = gen_FCA(seed * 11)
    Mstar = compute_fp(a, b, e, W, V)
    rs = np.random.RandomState(seed * 77777 + 1)
    
    for _ in range(500):
        v = (rs.rand(5) - 0.5) * 1.0
        M = np.clip(Mstar + v, 0.001, 0.999)
        N = n_operator(M, a, b, e, W, V)
        if np.max(np.abs(N-M)) < 1e-12:
            continue
        grad = (M - Mstar) / (M * (1.0 - M))
        lie = np.sum(grad * (N - M))
        all_lie.append(lie)

all_lie = np.array(all_lie)
print(f"  50,000 测试点")
print(f"  Lie max:  {all_lie.max():.6f}")
print(f"  Lie mean: {all_lie.mean():.6f}")
print(f"  文档声称: max = -0.032 (v4审计中修正为 约-0.02)")
print()

# ============================================================
# [G] 5分量同时超调概率
# ============================================================
print("=" * 70)
print("[G] 5分量同时超调概率 (文档: 1.5-7%)")
print("=" * 70)

simul_5 = 0
total_pts = 0
for seed in range(50):
    a, b, e, W, V = gen_FCA(seed * 17)
    Mstar = compute_fp(a, b, e, W, V)
    rs = np.random.RandomState(seed * 99999 + 7)
    
    for _ in range(400):
        M = np.clip(Mstar + (rs.rand(5)-0.5)*1.0, 0.001, 0.999)
        N = n_operator(M, a, b, e, W, V)
        if np.max(np.abs(N-M)) < 1e-12:
            continue
        total_pts += 1
        
        n_simul = sum(1 for k in range(5) if (M[k]-Mstar[k])*(N[k]-Mstar[k]) < 0)
        if n_simul == 5:
            simul_5 += 1

print(f"  总点数: {total_pts}")
print(f"  5分量同时超调: {simul_5}/{total_pts} = {100*simul_5/total_pts:.2f}%")
print(f"  文档声称: 1.5-7%")
print()

# ============================================================
# [H] J_kk = 0 在 M* 处验证
# ============================================================
print("=" * 70)
print("[H] J_kk(M*) = 0 直接验证 (零对角引理)")
print("=" * 70)

max_diag = 0.0
num_nonzero = 0
for seed in range(200):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = a + W @ Mstar + b + V @ Mstar + e
    
    for k in range(5):
        J_kk = (W[k, k]*(1-Mstar[k]) - Mstar[k]*V[k, k]) / Dstar[k]
        if abs(J_kk) > 1e-15:
            num_nonzero += 1
        if abs(J_kk) > abs(max_diag):
            max_diag = J_kk

print(f"  max |J_kk|: {abs(max_diag):.2e}")
print(f"  nonzero (|J_kk| > 1e-15): {num_nonzero}/1000")
print(f"  零对角成立: {'YES ✓' if abs(max_diag) < 1e-14 else 'NO ✗'}")

# 原因: 在 gen_FCA 中 np.fill_diagonal(W, 0) 和 np.fill_diagonal(V, 0)
# 所以 W_kk = V_kk = 0 → J_kk 解析为零
# 但 N 算子的通用定义中 w_kk, v_kk 不一定为 0
# 这取决于 FCA 的参数化约定
print()
print("  注: J_kk = 0 源于 W_kk = V_kk = 0 (FCA 参数化约定)")
print("  N_k = A_k/D_k 中 A_k 和 B_k 都不含 M_k 项")
print("  因此 ∂N_k/∂M_k = 0 是解析恒等式")
print()

# ============================================================
# 审计总结
# ============================================================
print("=" * 70)
print("第五轮审计 — 文档内部一致性总结")
print("=" * 70)
print(f"""
[B/C] KL 比值 & l₁ 比值:
  KL  |ΔV|/V 中位数:     {kl_median:.4f}
  l₁  ratio 中位数:      {l1_median:.4f}
  文档声称 KL=0.008, l₁=0.09
  实证 KL 中位数:         {np.median(kl_ratios):.4f}
  实证 l₁ 中位数:         {np.median(l1_ratios):.4f}

[D] Gershgorin 压缩: [{comp.min():.3f}, {comp.max():.3f}]
  文档: [1.35, 1.68]

[E] 超调统计: 文档声称 1246+502 超调分量 (M*<0.25+M*>0.75)
  本次实测 (50种子×400点): 超调总数见上方

[F] Lie 导数: max={all_lie.max():.6f}
  文档: max=-0.032 (旧) / "约-0.02" (修正后)

[G] 5分量同时超调: {100*simul_5/total_pts:.2f}%
  文档: 1.5-7%

[H] J_kk=0: ✓ (FCA参数化保证)
""")
