"""
RD/CD 行和界 —— 新的纯良基充分条件
====================================
关键洞察: RD 按行计算时, 分母 D*_k 对所有 j 相同, 可提因子:
  Σ_j |J_kj| = (1/D*_k) · Σ_j |w_kj(1-M*_k) - v_kj M*_k|

凸函数和的最大值在端点:
  sup_{M*_k∈[0,1]} Σ_j |w_kj - (w_kj+v_kj)M*_k| = max(Σ_j w_kj, Σ_j v_kj)

比原来的元素级界 Σ_j max(w,v) 紧约 2×。
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

print("=" * 70)
print("RD 行和界 vs. 元素级界 对比")
print("=" * 70)

# D* lower bounds
def D_low_vN(a, b, e, W, V, N=1):
    """N-th iteration lower bound on D*_k"""
    D_min = a + b + e
    D_max = a + b + e + np.sum(W + V, axis=1)
    M_low = np.clip(a / D_max, 0, 1)
    D_low = a + b + e + (W + V) @ M_low
    for _ in range(N - 1):
        M_low_next = np.clip((a + W @ M_low) / (a + b + e + (W + V) @ M_low), 0, 1)
        M_low = M_low_next
        D_low = a + b + e + (W + V) @ M_low
    return D_low

for gen_func, domain_name, n_seeds in [(gen_FCA, "FCA", 200), (gen_extended, "扩展域", 500)]:
    print(f"\n{'='*50}")
    print(f"  {domain_name} ({n_seeds} 种子)")
    print(f"{'='*50}")
    
    # Strategy OLD: 元素级 bound with D_min
    s_old_Dmin_rd = 0
    s_old_Dmin_cd = 0
    
    # Strategy OLD: 元素级 bound with D_low_v2
    s_old_Dv2_rd = 0
    s_old_Dv2_cd = 0
    
    # Strategy NEW: 行和 bound with D_low_v1
    s_new_Dv1_rd = 0
    s_new_Dv1_cd = 0
    
    # Strategy NEW: 行和 bound with D_low_v2
    s_new_Dv2_rd = 0
    s_new_Dv2_cd = 0
    
    # Strategy actual (reference)
    actual_rd_count = 0
    actual_cd_count = 0
    
    worst_seeds = []
    
    for seed in range(n_seeds):
        a, b, e, W, V = gen_func(seed)
        Mstar = compute_fp(a, b, e, W, V)
        Dstar = (a + W @ Mstar) + (b + V @ Mstar) + e
        
        D_min = a + b + e
        D_low_v2 = D_low_vN(a, b, e, W, V, N=2)
        D_low_v1 = D_low_vN(a, b, e, W, V, N=1)
        maxWV = np.maximum(W, V)
        
        # Actual RD/CD
        J_actual = np.zeros((5, 5))
        for k in range(5):
            for j in range(5):
                if k != j:
                    J_actual[k, j] = abs((W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k])
        actual_rd = max(J_actual.sum(axis=1))
        actual_cd = max(J_actual.sum(axis=0))
        if actual_rd < 1: actual_rd_count += 1
        if actual_cd < 1: actual_cd_count += 1
        
        # --- OLD: element-wise, D_min ---
        old_rd_Dmin = np.zeros(5)
        old_cd_Dmin = np.zeros(5)
        for k in range(5):
            old_rd_Dmin[k] = sum([maxWV[k,j] / D_min[k] for j in range(5) if j != k])
        for j in range(5):
            old_cd_Dmin[j] = sum([maxWV[k,j] / D_min[k] for k in range(5) if k != j])
        if max(old_rd_Dmin) < 1: s_old_Dmin_rd += 1
        if max(old_cd_Dmin) < 1: s_old_Dmin_cd += 1
        
        # --- OLD: element-wise, D_low_v2 ---
        old_rd_Dv2 = np.zeros(5)
        old_cd_Dv2 = np.zeros(5)
        for k in range(5):
            old_rd_Dv2[k] = sum([maxWV[k,j] / D_low_v2[k] for j in range(5) if j != k])
        for j in range(5):
            old_cd_Dv2[j] = sum([maxWV[k,j] / D_low_v2[k] for k in range(5) if k != j])
        if max(old_rd_Dv2) < 1: s_old_Dv2_rd += 1
        if max(old_cd_Dv2) < 1: s_old_Dv2_cd += 1
        
        # --- NEW: row-sum, D_low_v1 ---
        new_rd_Dv1 = np.zeros(5)
        for k in range(5):
            row_w_sum = W[k].sum()  # includes diag (0), same as Σ_{j≠k} w_kj
            row_v_sum = V[k].sum()
            num_bound = max(row_w_sum, row_v_sum)
            new_rd_Dv1[k] = num_bound / D_low_v1[k]
        if max(new_rd_Dv1) < 1: s_new_Dv1_rd += 1
        
        # --- NEW: row-sum, D_low_v2 ---
        new_rd_Dv2 = np.zeros(5)
        for k in range(5):
            row_w_sum = W[k].sum()
            row_v_sum = V[k].sum()
            num_bound = max(row_w_sum, row_v_sum)
            new_rd_Dv2[k] = num_bound / D_low_v2[k]
        if max(new_rd_Dv2) < 1: s_new_Dv2_rd += 1
        
        # Track worst seeds for the new bound
        worst_seeds.append((seed, max(new_rd_Dv2), max(old_rd_Dv2), actual_rd))
    
    print(f"  {'策略':<30} {'RD':<15}")
    print(f"  {'-'*45}")
    print(f"  {'OLD 元素级 D_min':<30} {s_old_Dmin_rd}/{n_seeds} ({100*s_old_Dmin_rd/n_seeds:.0f}%)")
    print(f"  {'OLD 元素级 D_low_v2':<30} {s_old_Dv2_rd}/{n_seeds} ({100*s_old_Dv2_rd/n_seeds:.0f}%)")
    print(f"  {'NEW 行和 D_low_v1':<30} {s_new_Dv1_rd}/{n_seeds} ({100*s_new_Dv1_rd/n_seeds:.0f}%)")
    print(f"  {'NEW 行和 D_low_v2':<30} {s_new_Dv2_rd}/{n_seeds} ({100*s_new_Dv2_rd/n_seeds:.0f}%)")
    print(f"  {'实际 (reference)':<30} {actual_rd_count}/{n_seeds} ({100*actual_rd_count/n_seeds:.0f}%)")
    
    # Show worst seeds for new method
    worst_seeds.sort(key=lambda x: -x[1])
    print(f"\n  NEW行和Dv2 最劣种子:")
    for s, nv, ov, av in worst_seeds[:5]:
        print(f"    seed {s:3d}: new={nv:.3f}  old_elem={ov:.3f}  actual={av:.3f}  gap={(nv-av):.3f}")


# ============================================================
# 理论分析：行和界的数学性质
# ============================================================
print(f"\n{'='*70}")
print("行和界 - 数学推导")
print(f"{'='*70}")
print()
print("对 RD (行对角占优) of I-J:")
print("  r_k = Σ_{j≠k} |J_kj|")
print("      = Σ_{j≠k} |w_kj(1-M*_k) - v_kj M*_k| / D*_k")
print("      = (1/D*_k) · Σ_{j≠k} |w_kj - (w_kj+v_kj)M*_k|")
print()
print("函数 f(M) = |w - (w+v)M| 在 [0,1] 上是凸的,")
print("  f(0) = w,  f(w/(w+v)) = 0,  f(1) = v")
print("  sup f = max(w, v)")
print()
print("但和 Σ_j f_j 的最大值不是 Σ_j max(w_j, v_j) !")
print("  sup_{M∈[0,1]} Σ_j |w_j - (w_j+v_j)M| = max(Σ_j w_j, Σ_j v_j)")
print("  原因: 凸函数的和在端点取最大值")
print()
print("因此:")
print("  r_k ≤ max(Σ_{j≠k} w_kj, Σ_{j≠k} v_kj) / D_low_k")
print()
print("对比元素级界:")
print("  r_k ≤ Σ_{j≠k} max(w_kj, v_kj) / D_low_k")
print()
print("改进因子 ≈ max(Σw, Σv) / Σ max(w,v)")
print("  当 w ≈ v 时: max(Σw, Σv) ≈ Σw,  Σ max(w,v) ≈ 2Σw → 改进约 2×")

# ============================================================
# CD 的困难
# ============================================================
print(f"\n{'='*70}")
print("CD 的额外困难")
print(f"{'='*70}")
print()
print("对 CD:")
print("  c_j = Σ_{k≠j} |J_kj| = Σ_{k≠j} |w_kj(1-M*_k) - v_kj M*_k| / D*_k")
print()
print("分母 D*_k 依赖 k, 不能像 RD 一样提因子.")
print("但实证中 CD 裕度远超 RD, 不是瓶颈.")
print()

# Check CD bottleneck
print("检查: CD 是否是瓶颈?")
print()
for gen_func, domain_name, n_seeds in [(gen_FCA, "FCA", 200)]:
    rd_vals = []
    cd_vals = []
    for seed in range(n_seeds):
        a, b, e, W, V = gen_func(seed)
        Mstar = compute_fp(a, b, e, W, V)
        Dstar = (a + W @ Mstar) + (b + V @ Mstar) + e
        J_actual = np.zeros((5, 5))
        for k in range(5):
            for j in range(5):
                if k != j:
                    J_actual[k, j] = abs((W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / Dstar[k])
        rd_vals.append(max(J_actual.sum(axis=1)))
        cd_vals.append(max(J_actual.sum(axis=0)))
    print(f"  {domain_name}: max RD = {max(rd_vals):.3f},  max CD = {max(cd_vals):.3f},  RD/CD = {max(rd_vals)/max(cd_vals) if max(cd_vals) > 0 else 0:.2f}")
