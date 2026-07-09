"""
全新路线探索：定理6.18的解析证明。
前4条路线失败原因：
  Route 1: 区间算术依赖性问题 → N²解析界太松(3x)
  Route 2: Lipschitz常数~10 → 分支定界膨胀
  Route 3: 平均Jacobian范数~5.5 → MVT也超1
  Route 4: 局部盆半径0.2-0.35 < N²像半径0.32-0.52

新路线：
  Route 5: 加权范数收缩 (diagonal scaling)
  Route 6: Thompson度量 / Hilbert射影度量
  Route 7: 二次Lyapunov函数 (SOS思路)
  Route 8: 单调有界收敛 (cooperative system check)
  Route 9: N的符号结构分析 (sign-stable Jacobian)
  Route 10: 极限环排除 (Bendixson-Dulac类判断)
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

# ============================================================
# Parameter generation
# ============================================================
def sample_FCA_params(rng=None):
    """Generate params satisfying FCA lattice constraints."""
    if rng is None:
        rng = np.random.RandomState()
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

def sample_BR_grid_params(B_up, rho_up, rng=None):
    """From the (B_up, rho_up) parameterization used in 441K tests."""
    if rng is None:
        rng = np.random.RandomState()
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

# ============================================================
# N operator and Jacobian
# ============================================================
def N_op(M, p):
    A = p.a + p.w @ M
    B = p.b + p.v @ M
    return A / (A + B + p.eps)

def N_fixed_point(p, n_iter=2000, tol=1e-12):
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
# Route 5: Weighted l∞-norm contraction
# ============================================================
def find_weighted_norm_contraction(p, Mstar, n_samples=5000):
    """
    Search for diagonal weight vector w > 0 such that
    ||N(M) - M*||_{w,∞} ≤ α ||M - M*||_{w,∞} for all M, α < 1.
    
    Weighted l∞ norm: ||x||_{w,∞} = max_i |x_i|/w_i
    
    For contraction in this norm, we need:
    Σ_j |J_ij(M)| * w_j ≤ α w_i  for all i, all M
    
    We discretize M-space and search for w.
    """
    rng = np.random.RandomState(42)
    
    # Sample points in [0,1]^5
    Ms = rng.uniform(0, 1, (n_samples, 5))
    Ms = np.vstack([Ms, Mstar.reshape(1, -1)])
    
    # Compute Jacobians at sample points
    all_J = np.array([J_N(M, p) for M in Ms])
    
    # For each candidate weight vector, check if contraction holds
    # Search for weights via random sampling
    best_alpha = float('inf')
    best_w = None
    
    for _ in range(10000):
        w = rng.uniform(0.1, 2.0, 5)
        max_ratio = 0.0
        for J in all_J:
            for i in range(5):
                row_sum = sum(abs(J[i, j]) * w[j] for j in range(5))
                ratio = row_sum / w[i]
                max_ratio = max(max_ratio, ratio)
        if max_ratio < best_alpha:
            best_alpha = max_ratio
            best_w = w.copy()
            if best_alpha < 1.0:
                break
    
    return best_alpha, best_w

# ============================================================
# Route 6: Thompson metric / Hilbert projective metric
# ============================================================
def thompson_contraction_check(p, Mstar, n_samples=2000):
    """
    Thompson metric on positive orthant:
    d_T(x, y) = log max(max_i x_i/y_i, max_i y_i/x_i)
    
    N maps to [0,1]^5 but actual range is strictly interior.
    Check if d_T(N(x), N(y)) ≤ α d_T(x, y) for α < 1.
    """
    rng = np.random.RandomState(42)
    Ms = rng.uniform(0.001, 0.999, (n_samples, 5))
    
    max_ratio = 0.0
    for i in range(min(n_samples, 500)):
        x = Ms[i]
        Nx = N_op(x, p)
        for j in range(i+1, min(n_samples, 501)):
            y = Ms[j]
            Ny = N_op(y, p)
            
            # Thompson distance before
            ratios_before = np.maximum(x/y, y/x)
            d_before = np.log(np.max(ratios_before))
            
            # Thompson distance after
            ratios_after = np.maximum(Nx/Ny, Ny/Nx)
            d_after = np.log(np.max(ratios_after))
            
            if d_before > 1e-10:
                ratio = d_after / d_before
                max_ratio = max(max_ratio, ratio)
    
    return max_ratio

# ============================================================
# Route 7: Quadratic Lyapunov function search
# ============================================================
def search_quadratic_lyapunov(p, Mstar, n_samples=2000):
    """
    Search for symmetric positive definite Q such that
    V(M) = (M-M*)^T Q (M-M*) satisfies
    V(N(M)) ≤ β V(M) for some β < 1.
    
    Use diagonal Q for simplicity (separable Lyapunov).
    """
    rng = np.random.RandomState(42)
    Ms = rng.uniform(0, 1, (n_samples, 5))
    
    best_beta = float('inf')
    best_q = None
    
    for _ in range(5000):
        q = rng.uniform(0.5, 2.0, 5)
        max_ratio = 0.0
        for M in Ms[:500]:
            d = M - Mstar
            V_before = np.sum(q * d * d)
            if V_before < 1e-15:
                continue
            N_M = N_op(M, p)
            d_after = N_M - Mstar
            V_after = np.sum(q * d_after * d_after)
            ratio = V_after / V_before
            max_ratio = max(max_ratio, ratio)
        if max_ratio < best_beta:
            best_beta = max_ratio
            best_q = q.copy()
            if best_beta < 1.0:
                break
    
    return best_beta, best_q

# ============================================================
# Route 8: Monotonicity (cooperative system) check
# ============================================================
def check_monotonicity(p, n_samples=2000):
    """
    Check if N is order-preserving:
    M ≤ M' (componentwise) ⇒ N(M) ≤ N(M')
    
    A sufficient condition is ∂N_k/∂M_j ≥ 0 for all k, j.
    We check Jacobian sign pattern across the state space.
    """
    rng = np.random.RandomState(42)
    Ms = rng.uniform(0, 1, (n_samples, 5))
    
    J_signs = np.zeros((5, 5, 3))  # count [negative, zero, positive]
    
    for M in Ms:
        J = J_N(M, p)
        for k in range(5):
            for j in range(5):
                if J[k, j] < -1e-12:
                    J_signs[k, j, 0] += 1
                elif J[k, j] > 1e-12:
                    J_signs[k, j, 2] += 1
                else:
                    J_signs[k, j, 1] += 1
    
    mixed_signs = []
    all_nonneg = True
    for k in range(5):
        for j in range(5):
            if J_signs[k, j, 0] > 0 and J_signs[k, j, 2] > 0:
                mixed_signs.append((k, j, J_signs[k, j, 0], J_signs[k, j, 2]))
                all_nonneg = False
    
    return all_nonneg, mixed_signs, J_signs

# ============================================================
# Route 9: Sign-stable Jacobian structure
# ============================================================
def analyze_jacobian_structure(p, Mstar):
    """
    Analyze whether J has special structure that enables convergence proofs.
    - Diagonal dominance?
    - M-matrix? (off-diagonal ≤ 0, inverse ≥ 0)
    - Metzler matrix? (off-diagonal ≥ 0)
    """
    J = J_N(Mstar, p)
    
    diagonal = np.diag(J)
    off_diag = J - np.diagflat(diagonal)
    
    # Row diagonal dominance
    row_dd = np.all(np.abs(diagonal) >= np.sum(np.abs(off_diag), axis=1))
    row_dd_ratio = np.max(np.sum(np.abs(off_diag), axis=1) / (np.abs(diagonal) + 1e-12))
    
    # Column diagonal dominance
    col_dd = np.all(np.abs(diagonal) >= np.sum(np.abs(off_diag), axis=0))
    col_dd_ratio = np.max(np.sum(np.abs(off_diag), axis=0) / (np.abs(diagonal) + 1e-12))
    
    # Metzler check (off-diagonal ≥ 0)
    is_metzler = np.all(off_diag >= -1e-12)
    
    # M-matrix check (off-diagonal ≤ 0, and can be written as sI-B with B ≥ 0, s > ρ(B))
    is_m_matrix_candidate = np.all(off_diag <= 1e-12)
    
    # Sign pattern stability: does the sign of each J_{ij} stay constant?
    # (We check this via the parametric formula)
    sign_changes = []
    for k in range(5):
        for j in range(5):
            # ∂N_k/∂M_j = (w_{kj}(B_k+ε_k) - A_k v_{kj}) / D_k²
            # Sign = sign(w_{kj}(B_k+ε_k) - A_k v_{kj})
            # A_k = a_k + Σ w_{kℓ} M_ℓ, B_k = b_k + Σ v_{kℓ} M_ℓ
            if p.w[k, j] == 0 and p.v[k, j] == 0:
                continue
            # As M varies over [0,1]^5, A_k ∈ [a_k, a_k + Σw], B_k ∈ [b_k, b_k + Σv]
            A_min = p.a[k]
            A_max = p.a[k] + p.w[k].sum()
            B_min = p.b[k]
            B_max = p.b[k] + p.v[k].sum()
            
            term_min = p.w[k, j] * (B_min + p.eps[k]) - A_max * p.v[k, j]
            term_max = p.w[k, j] * (B_max + p.eps[k]) - A_min * p.v[k, j]
            
            if term_min * term_max < 0:
                # Sign CAN change
                sign_changes.append((k, j, term_min, term_max))
    
    return {
        'row_diag_dominant': row_dd,
        'row_dd_ratio': row_dd_ratio,
        'col_diag_dominant': col_dd,
        'col_dd_ratio': col_dd_ratio,
        'is_metzler': is_metzler,
        'is_m_matrix': is_m_matrix_candidate,
        'sign_changes': sign_changes,
        'n_sign_changes': len(sign_changes),
        'J_at_star': J,
        'spectral_radius': max(abs(np.linalg.eigvals(J)))
    }

# ============================================================
# Route 10: Bendixson-Dulac criterion for discrete systems
# ============================================================
def check_divergence_condition(p, n_samples=2000):
    """
    For discrete systems, if the Jacobian determinant is always < 1,
    and the system is orientation-preserving (det > 0), 
    certain limit cycle exclusion results apply.
    
    Check: |det(J)| < 1 everywhere?
    """
    rng = np.random.RandomState(42)
    Ms = rng.uniform(0, 1, (n_samples, 5))
    
    max_det = 0.0
    min_det = float('inf')
    dets_gt_1 = 0
    
    for M in Ms:
        J = J_N(M, p)
        det = abs(np.linalg.det(J))
        max_det = max(max_det, det)
        min_det = min(min_det, det)
        if det >= 1.0:
            dets_gt_1 += 1
    
    return max_det, min_det, dets_gt_1 / n_samples

# ============================================================
# Route 11: Norm inequality via convex envelope
# ============================================================
def check_convex_envelope_contraction(p, Mstar, n_samples=2000):
    """
    Check if ||N(M) - M*||_2 / ||M - M*||_2 is bounded by a convex function
    of (|M_k - M*_k|) that decreases.
    
    This tests whether N compresses in l2 sense even when l∞ fails.
    """
    rng = np.random.RandomState(42)
    Ms = rng.uniform(0, 1, (n_samples, 5))
    
    l2_ratios = []
    l1_ratios = []
    
    for M in Ms:
        d = M - Mstar
        l2_before = np.sqrt(np.sum(d * d))
        l1_before = np.sum(np.abs(d))
        if l2_before < 1e-12:
            continue
        N_M = N_op(M, p)
        d_after = N_M - Mstar
        l2_after = np.sqrt(np.sum(d_after * d_after))
        l1_after = np.sum(np.abs(d_after))
        l2_ratios.append(l2_after / l2_before)
        l1_ratios.append(l1_after / l1_before)
    
    return {
        'l2_max_ratio': max(l2_ratios),
        'l2_mean_ratio': np.mean(l2_ratios),
        'l1_max_ratio': max(l1_ratios),
        'l1_mean_ratio': np.mean(l1_ratios),
    }

# ============================================================
# Route 12: Proximal / gradient-descent interpretation
# ============================================================
def check_gradient_descent_interpretation(p, Mstar):
    """
    Can N be written as M - η ∇f(M) for some convex f?
    
    N_k = A_k/(A_k+B_k+ε_k) = 1/(1 + (B_k+ε_k)/A_k)
    
    Let s_k = (B_k+ε_k)/A_k. Then N_k = 1/(1+s_k).
    
    This is the sigmoid of -log(s_k) = log(A_k/(B_k+ε_k)).
    
    The map M → s_k is affine in M (linear fractional in a sense).
    
    If we can write N(M) - M as the gradient of something, then convergence
    to the fixed point follows from convex optimization theory.
    """
    J = J_N(Mstar, p)
    
    # Check if J is symmetric at M* (necessary for gradient field)
    is_symmetric = np.allclose(J, J.T, atol=1e-10)
    symmetry_error = np.max(np.abs(J - J.T))
    
    # Check if J - I is negative definite at M* (necessary for descent)
    evals = np.linalg.eigvals(J - np.eye(5))
    max_eval = np.max(np.real(evals))
    
    return {
        'J_symmetric': is_symmetric,
        'symmetry_error': symmetry_error,
        'max_eval_J_minus_I': max_eval,
        'J_minus_I_negative_def': max_eval < 0,
    }

# ============================================================
# Comprehensive test
# ============================================================
def comprehensive_test(p, label=""):
    print(f"\n{'='*70}")
    print(f"Testing: {label}")
    print(f"{'='*70}")
    
    Mstar = N_fixed_point(p)
    print(f"M* = {Mstar}")
    
    # Route 5
    print("\n--- Route 5: Weighted l∞ contraction ---")
    alpha_w, best_w = find_weighted_norm_contraction(p, Mstar)
    print(f"Best weighted l∞ contraction ratio: {alpha_w:.4f}")
    print(f"Best weights: {best_w}")
    print(f"Is contraction: {alpha_w < 1.0}")
    
    # Route 6
    print("\n--- Route 6: Thompson metric ---")
    thompson_ratio = thompson_contraction_check(p, Mstar)
    print(f"Thompson contraction ratio: {thompson_ratio:.4f}")
    print(f"Is contraction: {thompson_ratio < 1.0}")
    
    # Route 7
    print("\n--- Route 7: Quadratic Lyapunov ---")
    beta, best_q = search_quadratic_lyapunov(p, Mstar)
    print(f"Best Lyapunov decay ratio β: {beta:.4f}")
    print(f"Best Q diag: {best_q}")
    print(f"Is Lyapunov: {beta < 1.0}")
    
    # Route 8
    print("\n--- Route 8: Monotonicity ---")
    all_nonneg, mixed, J_signs = check_monotonicity(p)
    print(f"All Jacobian entries non-negative: {all_nonneg}")
    if mixed:
        print(f"Mixed sign entries ({len(mixed)}):")
        for k, j, n_neg, n_pos in mixed[:5]:
            print(f"  J[{k},{j}]: {n_neg} negative, {n_pos} positive samples")
    
    # Route 9
    print("\n--- Route 9: Jacobian structure ---")
    struct = analyze_jacobian_structure(p, Mstar)
    print(f"Row diag dominant: {struct['row_diag_dominant']} (ratio={struct['row_dd_ratio']:.4f})")
    print(f"Col diag dominant: {struct['col_diag_dominant']} (ratio={struct['col_dd_ratio']:.4f})")
    print(f"Metzler (off-diag ≥ 0): {struct['is_metzler']}")
    print(f"M-matrix candidate (off-diag ≤ 0): {struct['is_m_matrix']}")
    print(f"Sign changes possible: {struct['n_sign_changes']}")
    if struct['sign_changes']:
        for k, j, tmin, tmax in struct['sign_changes'][:3]:
            print(f"  J[{k},{j}] sign change: [{tmin:.4f}, {tmax:.4f}]")
    print(f"ρ(J(M*)): {struct['spectral_radius']:.6f}")
    
    # Route 10
    print("\n--- Route 10: Determinant condition ---")
    max_d, min_d, frac_gt1 = check_divergence_condition(p)
    print(f"|det(J)| range: [{min_d:.6f}, {max_d:.6f}]")
    print(f"Fraction with |det|≥1: {frac_gt1:.4f}")
    
    # Route 11
    print("\n--- Route 11: l2/l1 contraction ---")
    norms = check_convex_envelope_contraction(p, Mstar)
    print(f"l2 max ratio: {norms['l2_max_ratio']:.4f}")
    print(f"l2 mean ratio: {norms['l2_mean_ratio']:.4f}")
    print(f"l1 max ratio: {norms['l1_max_ratio']:.4f}")
    print(f"l1 mean ratio: {norms['l1_mean_ratio']:.4f}")
    
    # Route 12
    print("\n--- Route 12: Gradient descent interpretation ---")
    gd = check_gradient_descent_interpretation(p, Mstar)
    print(f"J symmetric at M*: {gd['J_symmetric']} (error={gd['symmetry_error']:.2e})")
    print(f"max Re(λ(J-I)): {gd['max_eval_J_minus_I']:.6f}")
    print(f"J-I negative definite: {gd['J_minus_I_negative_def']}")
    
    return {
        'weighted_contraction': alpha_w,
        'thompson': thompson_ratio,
        'lyapunov': beta,
        'monotone': all_nonneg,
        'jacobian_structure': struct,
    }

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    
    rng = np.random.RandomState(42)
    
    # Test on multiple parameter sets
    results = []
    
    # Test 1: FCA params
    p1 = sample_FCA_params(rng)
    r1 = comprehensive_test(p1, "FCA sampled params")
    results.append(('FCA', r1))
    
    # Test 2: B_up=0.5, rho_up=0.5
    p2 = sample_BR_grid_params(0.5, 0.5, rng)
    r2 = comprehensive_test(p2, "B_up=0.5, ρ_up=0.5")
    results.append(('BR(0.5,0.5)', r2))
    
    # Test 3: B_up=1.0, rho_up=0.3
    p3 = sample_BR_grid_params(1.0, 0.3, rng)
    r3 = comprehensive_test(p3, "B_up=1.0, ρ_up=0.3")
    results.append(('BR(1.0,0.3)', r3))
    
    # Test 4: B_up=0.2, rho_up=0.8
    p4 = sample_BR_grid_params(0.2, 0.8, rng)
    r4 = comprehensive_test(p4, "B_up=0.2, ρ_up=0.8")
    results.append(('BR(0.2,0.8)', r4))
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, r in results:
        print(f"\n{name}:")
        print(f"  Weighted l∞ contraction: {r['weighted_contraction']:.4f} {'✓' if r['weighted_contraction'] < 1 else '✗'}")
        print(f"  Thompson contraction:     {r['thompson']:.4f} {'✓' if r['thompson'] < 1 else '✗'}")
        print(f"  Lyapunov decay β:         {r['lyapunov']:.4f} {'✓' if r['lyapunov'] < 1 else '✗'}")
        print(f"  Monotone (cooperative):   {'✓' if r['monotone'] else '✗'}")
        struct = r['jacobian_structure']
        print(f"  ρ(J(M*)):                 {struct['spectral_radius']:.6f}")
        print(f"  Row DD ratio:             {struct['row_dd_ratio']:.4f}")
        print(f"  Sign changes in J:        {struct['n_sign_changes']}")
