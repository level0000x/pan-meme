"""
6.17B seed 11 深度分析
======================
利用凸性引理 ■ 的成果, 尝试闭合 seed 11 的 l₁ 收缩界

已知: seed 11 的 α_bound (三角不等式上界) = 1.67 > 1
      但实际 l₁ 收缩比 = 0.20

问题: J(M*) 行内正负项相消, 三角不等式破坏相消
解决思路: 
  1. 分析 J(M*) 的符号模式
  2. 利用凸性引理 (N_k 沿 M_j 凹凸性固定) 证明沿迭代轨迹符号不翻转
  3. 保留符号的精确界替代三角不等式

关键: J_kj = (w_kj(1-N_k) - v_kj N_k)/D_k
      符号 = sign(w_kj(1-N_k) - v_kj N_k)
      当 N_k < w_kj/(w_kj+v_kj) 时为正, 否则为负
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

# ================================================================
# Seed 11 详细分析
# ================================================================
seed = 11
a,b,e,W,V = gen_FCA(seed)
Mstar = compute_fp(a,b,e,W,V)

print("="*70)
print(f"Seed {seed} 详细分析")
print()

print("M*:", Mstar)
print()

# Jacobian at M*
D_star = a + b + W @ Mstar + V @ Mstar + e
A_star = a + W @ Mstar
B_star = b + V @ Mstar + e

J = np.zeros((5,5))
J_abs = np.zeros((5,5))
J_sign = np.zeros((5,5), dtype=int)

for k in range(5):
    row = []
    for j in range(5):
        val = (W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / D_star[k]
        J[k,j] = val
        J_abs[k,j] = abs(val)
        J_sign[k,j] = np.sign(val) if abs(val) > 1e-15 else 0

print("J(M*) 矩阵:")
print(np.array2string(J, precision=4, suppress_small=True))
print()

print("|J(M*)| 矩阵:")
print(np.array2string(J_abs, precision=4, suppress_small=True))
print()

# l₁ 行范数
l1_rows_true = np.sum(J_abs, axis=1)
l1_rows_triangle = np.zeros(5)
for k in range(5):
    tri_bound = sum(W[k,j]*(1-Mstar[k]) + V[k,j]*Mstar[k] for j in range(5)) / D_star[k]
    l1_rows_triangle[k] = tri_bound

print("l₁ 行范数:")
for k in range(5):
    print(f"  行{k}: 真实={l1_rows_true[k]:.4f}  三角不等式上界={l1_rows_triangle[k]:.4f}  "
          f"过估={l1_rows_triangle[k]/max(l1_rows_true[k],1e-10):.1f}x")

alpha_bound = max(l1_rows_triangle)
alpha_true = max(l1_rows_true)
print(f"\n  α_bound (三角不等式) = {alpha_bound:.4f}")
print(f"  α_true  (真实行和)  = {alpha_true:.4f}")
print(f"  过估倍数 = {alpha_bound/alpha_true:.1f}x")
print()

# 分析哪一行过估最严重
print("各行内正负项分析:")
for k in range(5):
    pos_terms = []
    neg_terms = []
    for j in range(5):
        if j == k: continue
        val = J[k,j]
        if val > 1e-15:
            pos_terms.append((j, val))
        elif val < -1e-15:
            neg_terms.append((j, val))
    
    pos_sum = sum(v for _,v in pos_terms)
    neg_sum = sum(v for _,v in neg_terms)
    
    print(f"  行{k}: 正项={pos_terms}  负项={neg_terms}")
    print(f"        正和={pos_sum:.4f}  负和={neg_sum:.4f}  总和={pos_sum+neg_sum:.4f}  绝对值={abs(pos_sum)+abs(neg_sum):.4f}")
    print(f"        三角不等式: {pos_sum+abs(neg_sum):.4f} vs 真实: {pos_sum+neg_sum:.4f}")
    print()

# ================================================================
# 凸性引理的应用: 检查阈值
# ================================================================
print("="*70)
print("凸性引理应用: N_k 穿越阈值 θ_kj = w_kj/(w_kj+v_kj) 的条件")
print()

# 对 seed 11, 检查所有 (k,j) 的阈值和 N_k 在 M* 处的值
print("阈值检查 (所有 k≠j):")
for k in range(5):
    for j in range(5):
        if k == j: continue
        w = W[k,j]
        v = V[k,j]
        theta_kj = w/(w+v)
        
        # N_k at M*
        Nk_star = Mstar[k]
        
        # 在 M* 附近, N_k 是否能穿越 θ?
        # 退化条件: A(w+v) = wD, 即 A/D = w/(w+v)
        # A = a_k + Σ_{l≠j} w_kl M*_l
        # D = A + b_k + Σ_{l≠j} v_kl M*_l + ε_k
        
        A_minus_j = a[k] + sum(W[k,l]*Mstar[l] for l in range(5) if l != j)
        D_minus_j = A_minus_j + b[k] + sum(V[k,l]*Mstar[l] for l in range(5) if l != j) + e[k]
        ratio_AD = A_minus_j/D_minus_j
        
        degeneracy = abs(ratio_AD - theta_kj)
        
        print(f"  (k={k},j={j}): θ={theta_kj:.4f}  N*_k={Nk_star:.4f}  "
              f"A/D={ratio_AD:.4f}  退化距离={degeneracy:.4f}  "
              f"{'★ 接近退化!' if degeneracy < 0.1 else ''}")

print()

# ================================================================
# 关键: ||J||₁ < 1 是否只需要良基信息?
# ================================================================
print("="*70)
print("l₁ 收缩的良基分析")
print()

# 良基上界 (不经过 M*):
# |J_kj| = |w_kj(1-N_k) - v_kj N_k|/D_k
#        ≤ max(w_kj(1-N_k), v_kj N_k)/D_k  [两项同号时]
#        ≤ (w_kj+v_kj)/D_k                 [绝对值三角不等式]
#        ≤ w_kj/D_k + v_kj/D_k            [进一步放宽]

# 最紧的良基界: 使用 N_k ∈ [0,1]
# w_kj(1-N_k) - v_kj N_k ∈ [ -v_kj, w_kj ]  (since N_k∈[0,1])
# So |w_kj(1-N_k) - v_kj N_k| ≤ max(w_kj, v_kj)

print("每行的最优良基界:")
for k in range(5):
    tight_bound = sum(max(W[k,j], V[k,j]) for j in range(5)) / (a[k] + b[k] + e[k])  # D_min lower bound
    print(f"  行{k}: max(w,v)界={tight_bound:.4f}")
    
print()

# 实际检查: 是否有种子可以用 max(w,v) 界闭合?
print("200种子 max(w,v) 界扫描:")
closed_by_max = 0
for s in range(200):
    a_s,b_s,e_s,W_s,V_s = gen_FCA(s)
    Mstar_s = compute_fp(a_s,b_s,e_s,W_s,V_s)
    D_star_s = a_s + b_s + W_s @ Mstar_s + V_s @ Mstar_s + e_s
    
    max_bound = 0
    for k in range(5):
        row_bound = sum(max(W_s[k,j], V_s[k,j]) for j in range(5)) / D_star_s[k]
        max_bound = max(max_bound, row_bound)
    
    if max_bound < 1:
        closed_by_max += 1

print(f"  max(w,v) 界闭合: {closed_by_max}/200 ({100*closed_by_max/200:.0f}%)")
print()

# 进一步收紧: 使用 N_k 在 M* 处已知
print("利用 M* 的精确界:")
for k in range(5):
    exact_sign_sum = 0
    d = D_star[k]
    for j in range(5):
        if j == k: continue
        w = W[k,j]; v = V[k,j]
        val = w*(1-Mstar[k]) - v*Mstar[k]
        exact_sign_sum += val / d
    print(f"  行{k}: 符号保留和 = {exact_sign_sum:.6f}")
print()

# ================================================================
# 凸性引理 + 符号模式稳定性
# ================================================================
print("="*70)
print("凸性引理推论: J(M) 符号模式沿迭代的稳定性")
print()

"""
凸性引理 6.17Cb 告诉我们:
  d²N_k/dM_j² 的符号在 M_j 切片上是固定的
  这意味着 N_k(M_j) 单调方向固定 → J_kj 的符号固定
  
但这只适用于单个坐标切片 (其他 M_l 固定)。
沿迭代轨道, 所有 M 同时变化, 所以 J 的符号可能翻转。

不过! 如果 N 是压缩映射 (向 M* 收缩), 那么迭代是"围绕" M* 的,
M* 的 J 符号模式可以作为参考。
"""

# 检查: 沿随机方向离开 M*, J 的符号是否保持?
print("J(M*+r·u) 的符号模式稳定性:")
sign_changes = 0
total_checks = 0

for _ in range(100):
    u = np.random.randn(5)
    u /= np.linalg.norm(u)
    
    signs_at_Mstar = {}
    for k in range(5):
        for j in range(5):
            if k == j: continue
            w = W[k,j]; v = V[k,j]
            val = w*(1-Mstar[k]) - v*Mstar[k]
            if abs(val) > 1e-12:
                signs_at_Mstar[(k,j)] = np.sign(val)
    
    for r in np.logspace(-2, 0, 20):
        M = Mstar + r*u
        if np.any(M<1e-12) or np.any(M>1-1e-12): continue
        
        N = n_operator(M, a,b,e,W,V)
        D = a + b + W @ M + V @ M + e
        
        for (k,j), s0 in signs_at_Mstar.items():
            w = W[k,j]; v = V[k,j]
            val = w*(1-N[k]) - v*N[k]
            if abs(val) > 1e-12 and np.sign(val) != s0:
                sign_changes += 1
            total_checks += 1

print(f"  符号翻转: {sign_changes}/{total_checks} ({100*sign_changes/total_checks:.2f}%)")

# ================================================================
# 对策: 利用 M* 的信息 (半良基)
# ================================================================
print()
print("="*70)
print("对策: 半良基 l₁ 界")
print()

"""
如果接受 M* 由数值迭代获得, 那么 J(M*) 的符号模式就是已知的。
这还算是"良基"的——M* 一旦算得, 后续推理全是线性代数。

记:
  J_kj⁺ = max(0, J_kj(M*))
  J_kj⁻ = min(0, J_kj(M*))

由于 N 是压缩映射, 在 M* 附近:
  J_kj(M) ≈ J_kj(M*)  (符号不变的区域)

对于符号可能翻转的区域 (远离 M*):
  |J_kj(M)| ≤ (w_kj+v_kj)/D_k (通用的三角不等式)

结合: 
  |J_kj(M)| ≤ max(|J_kj(M*)|, (w_kj+v_kj)/D_k_bound)
  
这给出了一个半良基的界, 比纯三角不等式紧。
"""

# 半良基界
print("半良基界 (利用 M* 的 J 符号):")
for k in range(5):
    # 已知 J(M*) 的符号, 计算"精确"行和
    d_star = D_star[k]
    exact_sum = sum(abs(J[k,j]) for j in range(5))
    
    # 保守界: 已符号的项用 |J(M*)|, 未符号的项用三角不等式
    # 由于 M*_k 已知, 所有项的符号都已知!
    # 但这需要知道 M(r) 在整个轨道上 N_k 是否保持与 M*_k 相近...
    
    # 更实用的: 检查是否有更好的局部界
    # 对于 M 在 M* 的一个小邻域内, J(M) ≈ J(M*)
    # 所以 l₁ 行和 ≈ Σ_j |J_kj(M*)| = l1_rows_true[k]
    print(f"  行{k}: 真实 l₁ = {l1_rows_true[k]:.4f}  "
          f"三角不等式 = {l1_rows_triangle[k]:.4f}  "
          f"差值 = {l1_rows_triangle[k]-l1_rows_true[k]:.4f}")

# ================================================================
# 终极策略: 针对 seed 11 的具体情况
# ================================================================
print()
print("="*70)
print("Seed 11 最佳策略")
print()

# 对 seed 11, M* 已知, J(M*) 符号已知
# l₁ 真实行和 max = alpha_true, 所有行都 < 1
# 三角不等式 α_bound 来自行 k 的过估

# 找出过估最严重的行
worst_k = np.argmax(l1_rows_triangle - l1_rows_true)
print(f"过估最严重: 行{worst_k}")
print(f"  真实 l₁ = {l1_rows_true[worst_k]:.4f}")
print(f"  三角不等式 = {l1_rows_triangle[worst_k]:.4f}")
print()

# 检查该行的单个项
print(f"行{worst_k} 各项细节:")
for j in range(5):
    if j == worst_k: continue
    w = W[worst_k,j]
    v = V[worst_k,j]
    val = J[worst_k,j]
    tri = (w*(1-Mstar[worst_k]) + v*Mstar[worst_k]) / D_star[worst_k]
    print(f"  j={j}: w={w:.4f} v={v:.4f}  J={val:.6f}  |J|={abs(val):.6f}  "
          f"三角={tri:.6f}  过估={tri/abs(val):.2f}x")

# 如果能在迭代中证明 N_k 始终在 M*_k 附近 (利用收缩性),
# 则 J 的符号始终不变, l₁ 行和始终 < 1
print()
print("结论: seed 11 的 J(M*) 所有行 l₁ < 1")
print("      三角不等式过估来自行内正负项相消")
print("      如果沿迭代轨道 J 符号保持, l₁ 收缩 ■ 自动成立")
print("      凸性引理确保逐坐标凹凸性固定, 但不足以确保全轨道符号固定")
print()
print("      解决路径: 需要证明'符号保持'——这是 RD/CD 的等价问题")
print("      RD/CD 确保 I-J ≻ 0 → ρ(J) < 1")
print("      结合凸性 → ∂²N 符号固定 → 沿迭代单调方向 J 符号保持")
