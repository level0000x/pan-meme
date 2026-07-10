"""
凹凸性引理 - 解析证明 ∂²N_k/∂M_j² 不穿越零轴
================================================
核心代数发现:
  N_k(M_j) = (A + wM_j) / (D + (w+v)M_j)  [fixing other components]
  dN/dM_j = 0 ⇔ N_k = w/(w+v) ≡ θ
  Substituting: A(w+v) = wD  (independent of M_j!)
  
  Therefore: N_k NEVER equals θ unless A(w+v) = wD exactly.
  Since dN/dM_j > 0 when N_k < θ and < 0 when N_k > θ,
  and N_k is continuous in M_j, N_k stays on one side of θ.
  
  ⇒ sign of d²N_k/dM_j² = -sign(w(1-N_k)-vN_k) stays fixed.
  ⇒ CONVEXITY SIGN NEVER FLIPS along any M_j slice. ■
"""
import numpy as np

def n_operator(M, a, b, eps, W, V):
    num = a + W @ M
    den = num + b + V @ M + eps
    return num / den

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

# ============================================================
# Part 1: Analytic proof structure
# ============================================================
print("=" * 70)
print("引理 6.17Cb（凹凸性符号固定）■")
print("""
对于固定 M_l (l≠j)，考虑 N_k 作为 M_j 的单变量函数
  N_k(M_j) = (A + w_kj·M_j) / (D + (w_kj+v_kj)·M_j)
其中
  A ≡ a_k + Σ_{l≠j} w_kl M_l    (常数 > 0)
  D ≡ A + b_k + Σ_{l≠j} v_kl M_l + ε_k  (常数 > A)

一阶导:
  dN_k/dM_j = [w_kj(1-N_k) - v_kj·N_k] / D_k

二阶导:
  d²N_k/dM_j² = -2(w_kj+v_kj)·[w_kj(1-N_k)-v_kj·N_k] / D_k²

sign(d²N_k/dM_j²) = -sign(w_kj(1-N_k) - v_kj·N_k)
                   = -sign(θ - N_k)    where θ ≡ w_kj/(w_kj+v_kj)

dN_k/dM_j = 0  ⇔  N_k = θ
代入 N_k 的定义式:
  (A + wM_j)/(D + (w+v)M_j) = w/(w+v)
  ⇒ (A + wM_j)(w+v) = w(D + (w+v)M_j)
  ⇒ A(w+v) + w(w+v)M_j = wD + w(w+v)M_j
  ⇒ A(w+v) = wD                      ← M_j 消掉了!

由于 w>0, v>0, A≥0, D>A, 等式化为: A/D = w/(w+v)
即: A_k(-j) / D_k(-j) = w_kj/(w_kj+v_kj)

这是与 M_j 无关的退化条件——在一般参数下不成立。
因此 N_k(M_j) 在 M_j∈[0,1] 上永不等于 θ。

又因为 N_k 连续且 dN_k/dM_j ∝ (θ-N_k):
  - 若某点 N_k < θ, 则 dN_k/dM_j > 0 (N_k随M_j增)
  - 若某点 N_k > θ, 则 dN_k/dM_j < 0 (N_k随M_j减)

单调方向始终"指向"θ但永远到不了θ → N_k 保持与 θ 的相对大小关系不变。

⇒ sign(d²N_k/dM_j²) 在整条 M_j 切片上固定。∎
""")

# ============================================================
# Part 2: Numerical verification
# ============================================================
print("=" * 70)
print("数值验证")

total_tests = 0
sign_flips = 0
degenerate_cases = 0

for seed in range(200):
    a, b, e, W, V = gen_FCA(seed)
    
    for k in range(5):
        for j in range(5):
            if j == k: continue
            
            w = W[k,j]; v = V[k,j]
            theta = w / (w + v)
            
            for _ in range(50):
                M_other = np.random.uniform(0.02, 0.98, 5)
                
                # Degenerate check
                A = a[k] + sum(W[k,l] * M_other[l] for l in range(5) if l != j)
                D = A + b[k] + sum(V[k,l] * M_other[l] for l in range(5) if l != j) + e[k]
                
                if abs(A*(w+v) - w*D) < 1e-12:
                    degenerate_cases += 1
                
                signs = []
                for mj in np.linspace(0.01, 0.99, 501):
                    M = M_other.copy(); M[j] = mj
                    Nk = n_operator(M, a, b, e, W, V)[k]
                    F = w*(1-Nk) - v*Nk
                    signs.append(-np.sign(F))
                
                total_tests += 1
                unique = np.unique([s for s in signs if s != 0])
                if len(unique) > 1:
                    sign_flips += 1

print(f"  总测试: {total_tests:,}")
print(f"  符号翻转: {sign_flips}")
print(f"  退化条件命中: {degenerate_cases}")
print(f"  翻转率: {sign_flips}/{total_tests} = {100*sign_flips/total_tests:.4f}%")
print()

if sign_flips == 0:
    print("  ✓ 严格通过 — 凹凸性符号固定引理成立")
else:
    print("  ✗ 存在反例 — 需要分析退化条件")
    if degenerate_cases > 0:
        print(f"  可能有 {degenerate_cases} 个退化情况")
print()

# ============================================================
# Part 3: Degenerate case analysis
# ============================================================
print("=" * 70)
print("退化条件分析")
print()
print("退化条件: A(w+v) = wD")
print("即: A_k(-j) / D_k(-j) = w_kj/(w_kj+v_kj)")
print()
print("A_k(-j): a_k + sum of w_lk contributions (排除 j)")
print("D_k(-j): A_k(-j) + b_k + sum of v_lk contributions (排除 j) + ε_k")
print()
print("退化情形意味着: 当 M_j=0 时, N_k 恰好等于 θ")
print("但这不会导致符号翻转, 因为 N_k=θ 在 M_j=0 端点处")
print("而 dN_k/dM_j = 0 在该点, N_k 会 stay at θ 或 move away")
print()
print("FCA 参数随机生成几乎不可能命中此退化条件")
print()

# ============================================================
# Part 4: Implication for 6.17C
# ============================================================
print("=" * 70)
print("对 6.17C 的意义")
print()
print("已证: d²N_k/dM_j² 的符号在 M_j∈[0,1] 切片上固定 (对任意固定的 M_l, l≠j)")
print()
print("推论: N_k 沿 M_j 轴向的凹凸性固定 — 极值出现在端点 M_j=0 或 M_j=1")
print("       结合 6.17C 的 32 顶点全覆盖检验 (全部 min>0):")
print("       对于每个正单纯形切片, N 的方向单调性在保证顶点成立时, 即保证全域成立")
print()
print("这意味着 6.17C 的 ◆ (凹凸性引理) 变为 ■ !")
print()
print("6.17C 状态更新: 逐实例局部 ■ + 凹凸性 ■ + 顶点 ◆ → 可升级为 ■/◆ → ■")
