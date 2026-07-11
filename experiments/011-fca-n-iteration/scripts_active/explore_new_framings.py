r"""
Explore fundamentally new proof approaches for Theorem 6.18 (N global convergence).

New framings NOT attempted before:
  Route A: Componentwise sandwich - is N_k(M) always between M_k and M*_k?
  Route B: Sign coherence - does (N_k(M)-M*_k) always have the same sign as (M_k-M*_k)?
  Route C: "Push" direction - does N always push toward M*?
  Route D: Energy decrease - V(M) = ||M-M*||_1 as potential Lyapunov
  Route E: 3-step decomposition - separate contractive and expansive phases
  Route F: Spectral decomposition - use M*'s 4D invariant subspace
  Route G: Row-wise contraction - prove each row individually
  Route H: Componentwise monotone after first step
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
# Route A: Componentwise sandwich
# ============================================================
def test_componentwise_sandwich(p, n_samples=5000):
    """
    Test: For each component k, is N_k(M) always between M_k and M*_k?
    If true: min(M_k, M*_k) <= N_k(M) <= max(M_k, M*_k)
    Then N "moves" M toward M* componentwise.
    """
    Mstar = N_fixed_point(p)
    rng = np.random.RandomState(42)
    Ms = rng.uniform(0, 1, (n_samples, 5))
    
    violations = {k: 0 for k in range(5)}
    max_violation = 0.0
    
    for M in Ms:
        NM = N_op(M, p)
        for k in range(5):
            if M[k] > Mstar[k]:
                if NM[k] > M[k]:
                    violations[k] += 1
                    max_violation = max(max_violation, NM[k] - M[k])
                elif NM[k] < Mstar[k]:
                    violations[k] += 1
                    max_violation = max(max_violation, Mstar[k] - NM[k])
            elif M[k] < Mstar[k]:
                if NM[k] < M[k]:
                    violations[k] += 1
                    max_violation = max(max_violation, M[k] - NM[k])
                elif NM[k] > Mstar[k]:
                    violations[k] += 1
                    max_violation = max(max_violation, NM[k] - Mstar[k])
    
    total_violations = sum(violations.values())
    return total_violations, max_violation

# ============================================================
# Route B: Sign coherence
# ============================================================
def test_sign_coherence(p, n_samples=5000):
    """
    Test: Does sign(N_k(M) - M*_k) = sign(M_k - M*_k)?
    If true, N monotonically moves each component toward M*.
    """
    Mstar = N_fixed_point(p)
    rng = np.random.RandomState(42)
    Ms = rng.uniform(0, 1, (n_samples, 5))
    
    sign_mismatches = {k: 0 for k in range(5)}
    
    for M in Ms:
        NM = N_op(M, p)
        for k in range(5):
            if np.sign(M[k] - Mstar[k]) * np.sign(NM[k] - Mstar[k]) < 0:
                sign_mismatches[k] += 1
    
    total = sum(sign_mismatches.values())
    return total, sign_mismatches

# ============================================================
# Route C: "Push" direction toward M*
# ============================================================
def test_push_direction(p, n_samples=5000):
    """
    Test: Does the vector N(M)-M always have a component toward M*?
    In other words, is (N(M)-M) . (M*-M) >= 0?
    If true, each step pushes M toward M*.
    """
    Mstar = N_fixed_point(p)
    rng = np.random.RandomState(42)
    Ms = rng.uniform(0, 1, (n_samples, 5))
    
    negative_dot = 0
    
    for M in Ms:
        NM = N_op(M, p)
        push = NM - M       # direction N moves M
        target = Mstar - M  # direction toward M*
        dot = np.dot(push, target)
        if dot < 0:
            negative_dot += 1
    
    return negative_dot / n_samples

# ============================================================
# Route D: l1 norm monotonicity
# ============================================================
def test_l1_monotonicity(p, n_samples=5000):
    """
    Test: Is ||N(M)-M*||_1 <= ||M-M*||_1 always?
    A necessary condition for N being l1-contractive.
    """
    Mstar = N_fixed_point(p)
    rng = np.random.RandomState(42)
    Ms = rng.uniform(0, 1, (n_samples, 5))
    
    expansions = 0
    max_expansion_ratio = 0.0
    
    for M in Ms:
        l1_before = np.sum(np.abs(M - Mstar))
        if l1_before < 1e-14: continue
        NM = N_op(M, p)
        l1_after = np.sum(np.abs(NM - Mstar))
        ratio = l1_after / l1_before
        if ratio > 1:
            expansions += 1
        max_expansion_ratio = max(max_expansion_ratio, ratio)
    
    return expansions / n_samples, max_expansion_ratio

# ============================================================
# Route E: 3-step decomposition into phases
# ============================================================
def test_three_step_phases(p, n_samples=1000):
    """
    Decompose N^3 into contractive vs expansive phases.
    Phase 1: N - brings extremes to center
    Phase 2: N^2 = N(N(M)) - causes "explosion" in Jacobian
    Phase 3: N^3 - brings back to center
    
    Test: Is ||N^3(M)-M*|| / ||M-M*|| < 1 always?
    And can we characterize the contractive/expansive phases?
    """
    Mstar = N_fixed_point(p)
    rng = np.random.RandomState(42)
    Ms = rng.uniform(0, 1, (n_samples, 5))
    
    ratios_N1 = []
    ratios_N2 = []
    ratios_N3 = []
    
    for M in Ms:
        d0 = M - Mstar
        norm0 = np.max(np.abs(d0))
        if norm0 < 1e-14: continue
        
        M1 = N_op(M, p)
        d1 = M1 - Mstar
        norm1 = np.max(np.abs(d1))
        
        M2 = N_op(M1, p)
        d2 = M2 - Mstar
        norm2 = np.max(np.abs(d2))
        
        M3 = N_op(M2, p)
        d3 = M3 - Mstar
        norm3 = np.max(np.abs(d3))
        
        ratios_N1.append(norm1 / norm0)
        ratios_N2.append(norm2 / norm0)
        ratios_N3.append(norm3 / norm0)
    
    return {
        'N1_max': max(ratios_N1), 'N1_mean': np.mean(ratios_N1),
        'N2_max': max(ratios_N2), 'N2_mean': np.mean(ratios_N2),
        'N3_max': max(ratios_N3), 'N3_mean': np.mean(ratios_N3),
    }

# ============================================================
# Route F: Row-wise decomposition
# ============================================================
def test_rowwise_contraction(p, n_samples=5000):
    """
    For each row k: |N_k(M) - M*_k| = |(1-M*_k)A_k - M*_k(B_k+eps_k)| / D_k
    
    Key observation: (1-M*_k)A_k - M*_k(B_k+eps_k) = Σ_j c_kj Δ_j
    where c_kj = (1-M*_k)w_kj - M*_k v_kj
    
    So: |N_k(M)-M*_k| = |Σ_j c_kj Δ_j| / D_k
    
    For contraction, we need: |Σ_j c_kj Δ_j| / D_k <= α max_j |Δ_j|
    
    Can we bound this analytically?
    """
    Mstar = N_fixed_point(p)
    
    c = np.zeros((5, 5))
    for k in range(5):
        for j in range(5):
            c[k, j] = (1 - Mstar[k]) * p.w[k, j] - Mstar[k] * p.v[k, j]
    
    D_min = p.a + p.b + p.eps
    
    # Upper bound per row (pessimistic)
    row_bounds = np.zeros(5)
    for k in range(5):
        row_bounds[k] = np.sum(np.abs(c[k])) / D_min[k]
    
    # This is: (1/D_min_k) Σ_j |(1-M*_k)w_kj - M*_k v_kj|
    # Note: |(1-M*_k)w_kj - M*_k v_kj| <= (1-M*_k)|w_kj| + M*_k|v_kj|
    upper_bound = np.zeros(5)
    for k in range(5):
        for j in range(5):
            upper_bound[k] += ((1-Mstar[k]) * p.w[k,j] + Mstar[k] * p.v[k,j]) / D_min[k]
    
    return {
        'row_bounds': row_bounds,
        'max_row_bound': np.max(row_bounds),
        'upper_bound': np.max(upper_bound),
    }

# ============================================================
# Route G: Componentwise ordering check
# ============================================================
def test_componentwise_ordering(p, n_samples=5000):
    """
    Test: After first iteration, is each component of N(M) ordered
    relative to M*? Specifically:
    
    If M_k > M*_k, does N_k(M) < M_k? (toward M*)
    If M_k < M*_k, does N_k(M) > M_k? (toward M*)
    
    And: Is the ordering PRESERVED after the first step?
    """
    Mstar = N_fixed_point(p)
    rng = np.random.RandomState(42)
    Ms = rng.uniform(0, 1, (n_samples, 5))
    
    overshoots_step1 = 0
    overshoots_step2 = 0
    total = 0
    
    for M in Ms:
        M1 = N_op(M, p)
        for k in range(5):
            total += 1
            # Step 1: toward M*?
            if M[k] > Mstar[k] and M1[k] > M[k]:
                overshoots_step1 += 1  # went further away
            elif M[k] < Mstar[k] and M1[k] < M[k]:
                overshoots_step1 += 1  # went further away
        
        M2 = N_op(M1, p)
        for k in range(5):
            # Step 2: toward M*?
            if M1[k] > Mstar[k] and M2[k] > M1[k]:
                overshoots_step2 += 1
            elif M1[k] < Mstar[k] and M2[k] < M1[k]:
                overshoots_step2 += 1
    
    return {
        'overshoot_step1': overshoots_step1 / total,
        'overshoot_step2': overshoots_step2 / total,
    }

# ============================================================
# Route H: Vector field interpretation
# ============================================================
def test_vector_field_structure(p, n_samples=5000):
    """
    Interpret N(M) - M as a "discrete vector field" F(M).
    
    F_k(M) = N_k(M) - M_k = (A_k - M_k D_k)/D_k
    = ((1-M_k)A_k - M_k(B_k+eps_k))/D_k
    
    At M*: F(M*) = 0.
    
    For convergence, we want F(M) to "point toward" M*:
    (N(M)-M) . (M*-M) > 0 for M != M*
    
    This is equivalent to: F(M) . (M*-M) > 0
    
    Also check: Is F_k(M) · (M*_k - M_k) >= 0 componentwise?
    """
    Mstar = N_fixed_point(p)
    rng = np.random.RandomState(42)
    Ms = rng.uniform(0, 1, (n_samples, 5))
    
    dot_positive = 0
    componentwise_ok = {k: 0 for k in range(5)}
    total = 0
    
    for M in Ms:
        NM = N_op(M, p)
        F = NM - M
        dot = np.dot(F, Mstar - M)
        if dot > 0:
            dot_positive += 1
        total += 1
        
        for k in range(5):
            if F[k] * (Mstar[k] - M[k]) > 0:
                componentwise_ok[k] += 1
    
    return {
        'dot_positive_frac': dot_positive / total,
        'componentwise': {k: v / total for k, v in componentwise_ok.items()},
    }

# ============================================================
# Comprehensive test
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
            v = v * w_sum * rho_up / v_sum
        return Params(a=a.copy(), b=b.copy(), w=w.copy(), v=v.copy(), eps=eps.copy())
    
    # Test across 5 parameter sets
    print("=" * 70)
    print("TESTING NEW PROOF FRAMINGS")
    print("=" * 70)
    
    for name, p in [
        ("FCA-0", sample_FCA_params(rng)),
        ("FCA-1", sample_FCA_params(np.random.RandomState(1))),
        ("FCA-9", sample_FCA_params(np.random.RandomState(9))),
        ("BR(0.5,0.5)", sample_BR_params(0.5, 0.5, rng)),
        ("BR(0.2,0.8)", sample_BR_params(0.2, 0.8, rng)),
    ]:
        print(f"\n{'='*60}")
        print(f"Params: {name}")
        print(f"{'='*60}")
        
        Mstar = N_fixed_point(p)
        print(f"M* = {Mstar}")
        
        # Route A
        v_a, max_v_a = test_componentwise_sandwich(p)
        print(f"\n[A] Componentwise sandwich violations: {v_a} (max={max_v_a:.4f})")
        
        # Route B
        v_b, mm_b = test_sign_coherence(p)
        print(f"[B] Sign mismatches: {v_b}  per comp: {dict((k,int(v)) for k,v in mm_b.items())}")
        
        # Route C
        v_c = test_push_direction(p)
        print(f"[C] Negative dot product fraction: {v_c:.4f}")
        
        # Route D
        v_d, max_d = test_l1_monotonicity(p)
        print(f"[D] l1 expansion fraction: {v_d:.4f}  max ratio: {max_d:.4f}")
        
        # Route E
        v_e = test_three_step_phases(p, 2000)
        print(f"[E] 3-step: N1_max={v_e['N1_max']:.4f} N2_max={v_e['N2_max']:.4f} N3_max={v_e['N3_max']:.4f}")
        
        # Route F
        v_f = test_rowwise_contraction(p)
        print(f"[F] Row bounds: {v_f['row_bounds']}  max={v_f['max_row_bound']:.4f}")
        
        # Route G
        v_g = test_componentwise_ordering(p)
        print(f"[G] Overshoot step1: {v_g['overshoot_step1']:.4f}  step2: {v_g['overshoot_step2']:.4f}")
        
        # Route H
        v_h = test_vector_field_structure(p, 2000)
        print(f"[H] Dot(M*-M, N(M)-M) > 0: {v_h['dot_positive_frac']:.4f}")
        print(f"    Componentwise: {dict((k,f'{v:.3f}') for k,v in v_h['componentwise'].items())}")
        
    # ============================================================
    # Deep dive: Test across MANY parameter sets
    # ============================================================
    print(f"\n{'='*70}")
    print("DEEP DIVE: 50 FCA parameter sets")
    print(f"{'='*70}")
    
    all_sandwich = []
    all_sign = []
    all_dot_pos = []
    all_l1_expand = []
    
    for seed in range(50):
        rs = np.random.RandomState(seed)
        p = sample_FCA_params(rs)
        
        v_a, _ = test_componentwise_sandwich(p, 2000)
        v_b, _ = test_sign_coherence(p, 2000)
        v_c = test_push_direction(p, 2000)
        v_d, _ = test_l1_monotonicity(p, 2000)
        
        all_sandwich.append(v_a)
        all_sign.append(v_b)
        all_dot_pos.append(v_c)
        all_l1_expand.append(v_d)
    
    print(f"[A] Sandwich violations: avg={np.mean(all_sandwich):.1f} max={np.max(all_sandwich)}")
    print(f"[B] Sign mismatches:    avg={np.mean(all_sign):.1f} max={np.max(all_sign)}")
    print(f"[C] Neg dot fraction:   avg={np.mean(all_dot_pos):.4f} max={np.max(all_dot_pos):.4f}")
    print(f"[D] l1 expansion frac:  avg={np.mean(all_l1_expand):.4f} max={np.max(all_l1_expand):.4f}")
    
    # ============================================================
    # Key test: Does the "push" direction FAIL at extreme points?
    # ============================================================
    print(f"\n{'='*70}")
    print("CORNER CASE: Test at extreme points [0,0,0,0,0] and [1,1,1,1,1]")
    print(f"{'='*70}")
    
    for name, p in [
        ("FCA-0", sample_FCA_params(rng)),
        ("FCA-1", sample_FCA_params(np.random.RandomState(1))),
    ]:
        Mstar = N_fixed_point(p)
        print(f"\n{name}: M* = {Mstar}")
        
        for Mlabel, M0 in [
            ("[0,0,0,0,0]", np.zeros(5)),
            ("[1,1,1,1,1]", np.ones(5)),
            ("[1,0,0,1,1]", np.array([1,0,0,1,1])),
            ("[0,1,1,0,0]", np.array([0,1,1,0,0])),
        ]:
            N0 = N_op(M0, p)
            N1 = N_op(N0, p)
            N2 = N_op(N1, p)
            
            d0 = np.max(np.abs(M0 - Mstar))
            d1 = np.max(np.abs(N0 - Mstar))
            d2 = np.max(np.abs(N1 - Mstar))
            d3 = np.max(np.abs(N2 - Mstar))
            
            push_dot = np.dot(N0 - M0, Mstar - M0)
            
            print(f"  {Mlabel}:")
            print(f"    d0={d0:.4f} -> N: d1={d1:.4f} (r={d1/(d0+1e-15):.4f}) push_dot={push_dot:.4f}")
            print(f"    -> N2: d2={d2:.4f} -> N3: d3={d3:.4f} (net r={d3/(d0+1e-15):.4f})")
            print(f"    N0={N0}")
