"""
深入验证：N 本身在加权 l∞ 范数下是全局收缩映射。
目标：从数值验证走向解析证明。

关键发现（来自 prove_new_routes.py Route 5）：
  N 在某个加权 l∞ 范数下 α = 0.69-0.93，无需 N³。

本脚本任务：
  1. 严密验证：网格扫描 [0,1]⁵ 所有角点和内部点
  2. 证明 l1 列和范数 < 1（如果是，则 N 是 l1 收缩）
  3. 探索 J = diag(α₁)w - diag(α₂)v 的分解结构
  4. 寻找普适性权重公式
"""

import numpy as np
from dataclasses import dataclass
from itertools import product
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Params:
    a: np.ndarray
    b: np.ndarray
    w: np.ndarray
    v: np.ndarray
    eps: np.ndarray

def N_op(M, p):
    A = p.a + p.w @ M
    B = p.b + p.v @ M
    return A / (A + B + p.eps)

def N_fixed_point(p, n_iter=5000, tol=1e-14):
    M = np.full(5, 0.5)
    for _ in range(n_iter):
        M_new = N_op(M, p)
        if np.max(np.abs(M_new - M)) < tol:
            return M_new
        M = M_new
    return M

def J_N(M, p):
    A = p.a + p.w @ M
    B = p.b + p.v @ M
    D = A + B + p.eps
    J = np.zeros((5, 5))
    for k in range(5):
        for j in range(5):
            J[k, j] = (p.w[k, j] * (B[k] + p.eps[k]) - A[k] * p.v[k, j]) / (D[k] ** 2)
    return J

# ============================================================
# 1. 严密网格验证：32^5 = 33M 点太多，
#    改为 3^5 = 243 角点 + 4^5 = 1024 等距内点 + 5000 随机点
# ============================================================
def rigorous_contraction_test(p, n_random=10000):
    """
    在 [0,1]⁵ 的密集采样点上验证 N 的收缩性。
    测试 l∞, l1, l2 范数下的收缩。
    """
    Mstar = N_fixed_point(p)
    rng = np.random.RandomState(42)
    
    # 角点 3^5
    corners = np.array(list(product([0.0, 0.5, 1.0], repeat=5)))
    
    # 等距内点 4^5
    grid = np.array(list(product([0.1, 0.4, 0.6, 0.9], repeat=5)))
    
    # 随机点
    random_pts = rng.uniform(0, 1, (n_random, 5))
    
    all_pts = np.vstack([corners, grid, random_pts])
    
    linf_ratios = []
    l1_ratios = []
    l2_ratios = []
    
    for M in all_pts:
        d = M - Mstar
        linf_before = np.max(np.abs(d))
        l1_before = np.sum(np.abs(d))
        l2_before = np.sqrt(np.sum(d * d))
        
        if l2_before < 1e-14:
            continue
        
        NM = N_op(M, p)
        d_after = NM - Mstar
        
        linf_after = np.max(np.abs(d_after))
        l1_after = np.sum(np.abs(d_after))
        l2_after = np.sqrt(np.sum(d_after * d_after))
        
        linf_ratios.append(linf_after / linf_before)
        l1_ratios.append(l1_after / l1_before)
        l2_ratios.append(l2_after / l2_before)
    
    return {
        'linf_max': max(linf_ratios),
        'linf_mean': np.mean(linf_ratios),
        'l1_max': max(l1_ratios),
        'l1_mean': np.mean(l1_ratios),
        'l2_max': max(l2_ratios),
        'l2_mean': np.mean(l2_ratios),
    }

# ============================================================
# 2. l1 列和范数（矩阵范数）检验
# ============================================================
def test_l1_matrix_norm(p, n_samples=5000):
    """
    检验 ||J(M)||_1 = max_j Σ_k |J_{kj}(M)| 是否 < 1。
    如果对所有 M 成立，则 N 是 l1 收缩映射。
    """
    rng = np.random.RandomState(42)
    Ms = rng.uniform(0, 1, (n_samples, 5))
    
    # Also check corners
    corners = np.array(list(product([0.0, 0.5, 1.0], repeat=5)))
    Ms = np.vstack([Ms, corners])
    
    max_norm = 0.0
    max_norm_M = None
    col_norms_history = []
    
    for M in Ms:
        J = J_N(M, p)
        col_sums = np.sum(np.abs(J), axis=0)
        norm = np.max(col_sums)
        col_norms_history.append(col_sums.copy())
        if norm > max_norm:
            max_norm = norm
            max_norm_M = M.copy()
    
    col_norms_history = np.array(col_norms_history)
    
    return {
        'max_l1_matrix_norm': max_norm,
        'max_at_M': max_norm_M,
        'max_per_column': np.max(col_norms_history, axis=0),
        'mean_per_column': np.mean(col_norms_history, axis=0),
    }

# ============================================================
# 3. 分解结构：J = diag(α₁)w - diag(α₂)v  
# ============================================================
def analyze_decomposition(p, n_samples=5000):
    """
    J = diag(α₁)w - diag(α₂)v
    where α₁_k = (B_k+ε_k)/(A_k+B_k+ε_k)²
          α₂_k = A_k/(A_k+B_k+ε_k)²
    
    Note: α₁_k + α₂_k = 1/(A_k+B_k+ε_k) = 1/D_k
    
    Key: α₁_k/α₂_k = (B_k+ε_k)/A_k = 1/r_k where r_k = A_k/(B_k+ε_k)
    
    For the weighted l∞ norm with weights ω:
    ||J||_{ω,∞} = max_i (1/ω_i) Σ_j |α₁_i w_{ij} - α₂_i v_{ij}| ω_j
    
    The sign of entry (i,j) is sign(α₁_i w_{ij} - α₂_i v_{ij}) 
    = sign(w_{ij} - (α₂_i/α₁_i) v_{ij})
    = sign(w_{ij} - r_i v_{ij})
    
    where r_i = A_i/(B_i+ε_i).
    """
    rng = np.random.RandomState(42)
    Ms = rng.uniform(0, 1, (n_samples, 5))
    
    alpha1_ranges = np.zeros((5, 2))
    alpha1_ranges[:, 0] = float('inf')
    alpha1_ranges[:, 1] = 0.0
    
    alpha2_ranges = np.zeros((5, 2))
    alpha2_ranges[:, 0] = float('inf')
    alpha2_ranges[:, 1] = 0.0
    
    r_ranges = np.zeros((5, 2))
    r_ranges[:, 0] = float('inf')
    r_ranges[:, 1] = 0.0
    
    for M in Ms:
        A = p.a + p.w @ M
        B = p.b + p.v @ M
        D = A + B + p.eps
        
        a1 = (B + p.eps) / (D ** 2)
        a2 = A / (D ** 2)
        r = A / (B + p.eps)
        
        for k in range(5):
            alpha1_ranges[k, 0] = min(alpha1_ranges[k, 0], a1[k])
            alpha1_ranges[k, 1] = max(alpha1_ranges[k, 1], a1[k])
            alpha2_ranges[k, 0] = min(alpha2_ranges[k, 0], a2[k])
            alpha2_ranges[k, 1] = max(alpha2_ranges[k, 1], a2[k])
            r_ranges[k, 0] = min(r_ranges[k, 0], r[k])
            r_ranges[k, 1] = max(r_ranges[k, 1], r[k])
    
    # Analytic bounds
    analytic_alpha1_min = np.zeros(5)
    analytic_alpha1_max = np.zeros(5)
    analytic_alpha2_min = np.zeros(5)
    analytic_alpha2_max = np.zeros(5)
    analytic_r_min = np.zeros(5)
    analytic_r_max = np.zeros(5)
    
    for k in range(5):
        A_min = p.a[k]
        A_max = p.a[k] + p.w[k].sum()
        B_min = p.b[k]
        B_max = p.b[k] + p.v[k].sum()
        
        D_min = A_min + B_min + p.eps[k]
        D_max = A_max + B_max + p.eps[k]
        
        # α₁ = (B+ε)/D²: max at large B, small D
        # α₁_max ≈ B_max+ε / D_min²
        analytic_alpha1_max[k] = (B_max + p.eps[k]) / (D_min ** 2)
        analytic_alpha1_min[k] = (B_min + p.eps[k]) / (D_max ** 2)
        
        # α₂ = A/D²: max at large A, small D
        analytic_alpha2_max[k] = A_max / (D_min ** 2)
        analytic_alpha2_min[k] = A_min / (D_max ** 2)
        
        # r = A/(B+ε): max at A_max, B_min
        analytic_r_max[k] = A_max / (B_min + p.eps[k])
        analytic_r_min[k] = A_min / (B_max + p.eps[k])
    
    return {
        'alpha1_numeric': alpha1_ranges,
        'alpha2_numeric': alpha2_ranges,
        'r_numeric': r_ranges,
        'alpha1_analytic': np.column_stack([analytic_alpha1_min, analytic_alpha1_max]),
        'alpha2_analytic': np.column_stack([analytic_alpha2_min, analytic_alpha2_max]),
        'r_analytic': np.column_stack([analytic_r_min, analytic_r_max]),
    }

# ============================================================
# 4. 寻找普适权重公式
# ============================================================
def find_universal_weight_formula(p):
    """
    尝试从参数结构推导一个解析权重公式。
    
    观察：如果取 ω_i = D_i(M*) = A_i* + B_i* + ε_i,
    是否在加权范数下 N 是收缩的？
    """
    Mstar = N_fixed_point(p)
    A_star = p.a + p.w @ Mstar
    B_star = p.b + p.v @ Mstar
    D_star = A_star + B_star + p.eps
    
    # Test weight = D_star (natural scale)
    rng = np.random.RandomState(42)
    Ms = rng.uniform(0, 1, (5000, 5))
    
    max_ratio_D = 0.0
    max_ratio_uniform = 0.0
    
    for M in Ms:
        J = J_N(M, p)
        
        # Weighted l∞ with w=D*
        weighted_row_sums = np.array([
            sum(abs(J[i, j]) * D_star[j] for j in range(5)) / D_star[i]
            for i in range(5)
        ])
        ratio_D = np.max(weighted_row_sums)
        max_ratio_D = max(max_ratio_D, ratio_D)
        
        # Standard l∞ (w=1)
        std_row_sums = np.sum(np.abs(J), axis=1)
        ratio_uniform = np.max(std_row_sums)
        max_ratio_uniform = max(max_ratio_uniform, ratio_uniform)
    
    # Also test: weight = D(M) itself (depends on M, not a valid norm...)
    # Test: w_i = D_min_i (worst-case D)
    D_min = p.eps.copy()
    D_max = p.a + p.w.sum(axis=1) + p.b + p.v.sum(axis=1) + p.eps
    
    print(f"\n  Weight candidates:")
    print(f"  D* = {D_star}")
    print(f"  D_min = {D_min}")
    print(f"  D_max = {D_max}")
    label_d = "||J||_(D*,inf)"
    label_1 = "||J||_(1,inf)"
    print(f"  {label_d} (worst sampled): {max_ratio_D:.4f}")
    print(f"  {label_1} (standard linf): {max_ratio_uniform:.4f}")
    
    # Search for best weight among simple formulas
    candidates = {
        'D_star': D_star,
        'D_min': D_min,
        'D_max': D_max,
        'sqrt(D_star)': np.sqrt(D_star),
        'ones': np.ones(5),
        'eps': p.eps,
        'A_star': A_star,
        'B_star+eps': B_star + p.eps,
    }
    
    for name, w in candidates.items():
        max_r = 0.0
        for M in Ms[:1000]:
            J = J_N(M, p)
            rs = np.array([
                sum(abs(J[i, j]) * w[j] for j in range(5)) / w[i]
                for i in range(5)
            ])
            max_r = max(max_r, np.max(rs))
        is_ok = "✓" if max_r < 1.0 else "✗"
        label_j = f"||J||_w_inf"
        print(f"  ω = {name:15s}: {label_j} = {max_r:.4f} {is_ok}")
    
    return max_ratio_D, max_ratio_uniform

# ============================================================
# 5. 最关键的检验：能否证明 l1 矩阵范数 < 1？
# ============================================================
def prove_l1_contraction(p):
    """
    Try to prove ||J(M)||_1 < 1 for all M ∈ [0,1]⁵.
    
    ||J||_1 = max_j C_j where C_j = Σ_k |J_{kj}|
    
    J_{kj} = (w_{kj}(B_k+ε_k) - A_k v_{kj}) / D_k²
    
    For each column j, we need:
    Σ_k |w_{kj}(B_k+ε_k) - A_k v_{kj}| / D_k² < 1
    
    Upper bound:
    Σ_k (w_{kj}(B_k+ε_k) + A_k v_{kj}) / D_k² < 1
    
    But D_k = A_k + B_k + ε_k, so:
    (B_k+ε_k)/D_k² ≤ 1/D_k and A_k/D_k² ≤ 1/D_k
    
    Therefore:
    Σ_k (w_{kj}(B_k+ε_k) + A_k v_{kj}) / D_k² ≤ Σ_k (w_{kj} + v_{kj}) / D_k
    
    D_k ≥ ε_k, so D_k is bounded below. But this bound might be loose.
    
    Better: use the actual structure.
    
    For each k: (B_k+ε_k)/D_k² + A_k/D_k² = 1/D_k
    
    So C_j ≤ Σ_k max(w_{kj}, v_{kj}) / D_k ≤ (1/min D_k) Σ_k max(w_{kj}, v_{kj})
    
    This bound is too loose when ε is small.
    
    Let's try a different bound:
    |w_{kj}(B_k+ε_k) - A_k v_{kj}| / D_k²
    = |w_{kj}·(B_k+ε_k)/D_k² - v_{kj}·A_k/D_k²|
    ≤ max(w_{kj}·(B_k+ε_k)/D_k², v_{kj}·A_k/D_k²)
    
    Because (B_k+ε_k)/D_k² = (B_k+ε_k)/D_k · 1/D_k ≤ 1/D_k
    and A_k/D_k² = A_k/D_k · 1/D_k ≤ 1/D_k
    
    So: |J_{kj}| ≤ max(w_{kj}, v_{kj}) / D_k
    
    C_j ≤ Σ_k max(w_{kj}, v_{kj}) / D_k
    """
    D_min = p.eps.copy()
    A_max = p.a + p.w.sum(axis=1)
    B_max = p.b + p.v.sum(axis=1)
    D_max = A_max + B_max + p.eps
    
    # Loose upper bound
    col_bound_loose = np.array([
        sum(max(p.w[k, j], p.v[k, j]) / D_min[k] for k in range(5))
        for j in range(5)
    ])
    
    # Tighter bound using max(w·(B+ε)/D², v·A/D²)
    # For each (k,j), the contribution is at most max of two terms:
    # T1_{kj} = w_{kj} * max_M(B_k+ε)/D_k²
    # T2_{kj} = v_{kj} * max_M A_k/D_k²
    
    # max(B+ε)/D²: occurs at max B and min D → B_max+ε / D_min²
    # max A/D²: occurs at max A and min D → A_max / D_min²
    col_bound_tight = np.zeros(5)
    for j in range(5):
        col_sum = 0.0
        for k in range(5):
            t1 = p.w[k, j] * (B_max[k] + p.eps[k]) / (D_min[k] ** 2)
            t2 = p.v[k, j] * A_max[k] / (D_min[k] ** 2)
            col_sum += max(t1, t2)
        col_bound_tight[j] = col_sum
    
    # Even tighter: don't use max of the two terms, use the fact that
    # at any given M, one of the two differences determines the sign
    # |J_{kj}| = J_{kj} if J_{kj} ≥ 0, else -J_{kj}
    # = w_{kj}(B_k+ε_k)/D_k² - v_{kj}A_k/D_k² (if ≥ 0)
    # or v_{kj}A_k/D_k² - w_{kj}(B_k+ε_k)/D_k² (if < 0)
    # = (1/D_k²) max(w_{kj}(B_k+ε_k), v_{kj}A_k) - ...
    # Actually no, |a-b| = max(a,b) - min(a,b) = max(a-b, b-a)
    
    # Actually the simplest bound:
    # |J_{kj}| ≤ max(w_{kj}, v_{kj}) · max(B_k+ε_k, A_k) / D_k²
    # and max(B_k+ε_k, A_k) ≤ D_k, so:
    # |J_{kj}| ≤ max(w_{kj}, v_{kj}) / D_k
    # This is the same as the loose bound. It's loose because D_k can be small.
    
    print(f"\n  Analytic bounds for ||J||_1:")
    print(f"  D_min = {D_min}")
    print(f"  D_max = {D_max}")
    print(f"  Loose bound (per column): {np.max(col_bound_loose):.4f}")
    print(f"  Tight bound (per column): {np.max(col_bound_tight):.4f}")
    
    return col_bound_tight

# ============================================================
# 6. 解析构造权重向量的尝试
# ============================================================
def construct_weight_analytically(p):
    """
    Given J = diag(α₁)w - diag(α₂)v, 
    try to analytically construct ω such that ||J||_{ω,∞} < 1.
    
    Idea: Choose ω proportional to the "influence" of each component.
    
    ω_i = D_i* = A_i* + B_i* + ε_i (natural scale at fixed point)
    
    Then for row i:
    Σ_j |J_ij| ω_j = Σ_j |α₁_i w_{ij} - α₂_i v_{ij}| ω_j
    = (1/D_i²) Σ_j |w_{ij}(B_i+ε_i) - A_i v_{ij}| ω_j
    
    For contraction, we need this < ω_i = D_i*.
    
    Equivalently: Σ_j |w_{ij}(B_i+ε_i) - A_i v_{ij}| ω_j < D_i² D_i*
    
    At M = M*: w_{ij}(B_i*+ε_i) - A_i* v_{ij} has the sign of 
    w_{ij}/v_{ij} - A_i*/(B_i*+ε_i) = w_{ij}/v_{ij} - r_i*
    
    We know |J(M*)| has spectral radius < 1. So ||J(M*)||_{ω,∞} < 1 for 
    ω being the dominant eigenvector of |J(M*)|.
    
    But this only works at M*, not for all M.
    """
    Mstar = N_fixed_point(p)
    J_star = J_N(Mstar, p)
    abs_J_star = np.abs(J_star)
    
    # Dominant eigenvector of |J(M*)| gives optimal weight at M*
    evals, evecs = np.linalg.eig(abs_J_star.T)
    dominant_idx = np.argmax(np.abs(evals))
    opt_w = np.abs(evecs[:, dominant_idx])
    opt_w = opt_w / opt_w.sum()
    
    rho_star = max(abs(evals))
    
    print(f"\n  At M*:")
    print(f"  ρ(|J(M*)|) = {rho_star:.6f}")
    print(f"  Optimal weight at M*: {opt_w}")
    
    # Test if this weight works globally
    rng = np.random.RandomState(42)
    Ms = rng.uniform(0, 1, (5000, 5))
    
    global_max = 0.0
    for M in Ms:
        J = J_N(M, p)
        rs = np.array([
            sum(abs(J[i, j]) * opt_w[j] for j in range(5)) / opt_w[i]
            for i in range(5)
        ])
        global_max = max(global_max, np.max(rs))
    
    label_j = "||J||_(w_opt,inf)"
    print(f"  {label_j} (global worst): {global_max:.4f} {'✓' if global_max < 1 else '✗'}")
    
    return rho_star, global_max, opt_w

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.RandomState(42)
    
    def sample_FCA_params(rng):
        a = rng.uniform(0.01, 0.5, 5)
        b = rng.uniform(0.01, 0.5, 5)
        eps = rng.uniform(0.001, 0.1, 5)
        w = rng.uniform(0.01, 0.3, (5, 5))
        v = rng.uniform(0.01, 0.3, (5, 5))
        for i in range(5):
            w[i, i] = 0.0
            v[i, i] = 0.0
        tot = a.sum() + b.sum() + w.sum() + v.sum()
        w = w / tot * 5.0
        v = v / tot * 5.0
        return Params(a=a.copy(), b=b.copy(), w=w.copy(), v=v.copy(), eps=eps.copy())
    
    def sample_BR_params(B_up, rho_up, rng):
        a = rng.uniform(0.01, 0.5, 5)
        eps = rng.uniform(0.001, 0.1, 5)
        w = rng.uniform(0.01, 0.3, (5, 5))
        v = rng.uniform(0.01, 0.3, (5, 5))
        for i in range(5):
            w[i, i] = 0.0
            v[i, i] = 0.0
        A_up = rng.uniform(0.5, 2.0)
        b_base = rng.uniform(0.01, 0.3, 5)
        b = b_base * (B_up / A_up)
        w_sum = w.sum()
        v_sum = v.sum()
        if v_sum > 0:
            scale = w_sum * rho_up / v_sum
            v = v * scale
        return Params(a=a.copy(), b=b.copy(), w=w.copy(), v=v.copy(), eps=eps.copy())
    
    test_params = [
        ("FCA", sample_FCA_params(rng)),
        ("BR(0.5,0.5)", sample_BR_params(0.5, 0.5, rng)),
        ("BR(1.0,0.3)", sample_BR_params(1.0, 0.3, rng)),
        ("BR(0.2,0.8)", sample_BR_params(0.2, 0.8, rng)),
    ]
    
    all_pass = True
    
    for name, p in test_params:
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")
        
        # Test 1: Rigorous contraction
        print("\n--- Test 1: Rigorous norm contraction ---")
        result = rigorous_contraction_test(p, n_random=5000)
        linf_ok = "✓" if result['linf_max'] < 1 else "✗"
        l1_ok = "✓" if result['l1_max'] < 1 else "✗"
        l2_ok = "✓" if result['l2_max'] < 1 else "✗"
        print(f"  linf max ratio: {result['linf_max']:.4f} ({linf_ok})")
        print(f"  l1  max ratio: {result['l1_max']:.4f} ({l1_ok})")
        print(f"  l2  max ratio: {result['l2_max']:.4f} ({l2_ok})")
        
        if result['linf_max'] >= 1 and result['l1_max'] >= 1:
            all_pass = False
        
        # Test 2: l1 matrix norm
        print("\n--- Test 2: l1 matrix norm (||J||_1) ---")
        l1_result = test_l1_matrix_norm(p)
        print(f"  max ||J||_1 = {l1_result['max_l1_matrix_norm']:.4f} {'✓' if l1_result['max_l1_matrix_norm'] < 1 else '✗'}")
        print(f"  Per-column max: {l1_result['max_per_column']}")
        
        # Test 3: Decomposition
        print("\n--- Test 3: J = diag(α₁)w - diag(α₂)v decomposition ---")
        decomp = analyze_decomposition(p)
        print(f"  α₁ ranges:")
        for k in range(5):
            print(f"    k={k}: numeric=[{decomp['alpha1_numeric'][k,0]:.6f}, {decomp['alpha1_numeric'][k,1]:.6f}], "
                  f"analytic=[{decomp['alpha1_analytic'][k,0]:.6f}, {decomp['alpha1_analytic'][k,1]:.6f}]")
        print(f"  r = A/(B+ε) ranges:")
        for k in range(5):
            print(f"    k={k}: numeric=[{decomp['r_numeric'][k,0]:.4f}, {decomp['r_numeric'][k,1]:.4f}], "
                  f"analytic=[{decomp['r_analytic'][k,0]:.4f}, {decomp['r_analytic'][k,1]:.4f}]")
        
        # Test 4: Universal weight
        print("\n--- Test 4: Weight formula search ---")
        find_universal_weight_formula(p)
        
        # Test 5: Analytic l1 bound
        print("\n--- Test 5: Analytic l1 bound attempt ---")
        prove_l1_contraction(p)
        
        # Test 6: Construct weight at M*
        print("\n--- Test 6: Weight from M* eigenvector ---")
        construct_weight_analytically(p)
    
    print(f"\n{'='*70}")
    print(f"OVERALL: All tests pass = {all_pass}")
    print(f"{'='*70}")
