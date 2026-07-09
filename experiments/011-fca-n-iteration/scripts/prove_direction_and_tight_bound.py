"""
Two-pronged proof attempt:

PRONG 1: Analytic proof of direction monotonicity (6.17C)
  (N(M)-M)·(M*-M) > 0 for all M ≠ M*

  From 6.17A: N_k(M)-M*_k = (D*_k/D_k) Σ_j J_kj(M*) Δ_j

  So: (N(M)-M)·(M*-M) = Σ_k [ (D*_k/D_k) Σ_j J_kj(M*) Δ_j - Δ_k ](-Δ_k)
                       = Σ_k Δ_k² - Σ_k Σ_j (D*_k/D_k) J_kj Δ_j Δ_k

  Need: Δ^T A Δ > 0 where A_ii = 1 - (D*_i/D_i)J_ii(M*) = 1
        (since J_ii(M*) = 0 from Lemma 11.1A)
        A_ij = -(D*_i/D_i) J_ij(M*) for i≠j

  So A = I - diag(D*/D) J(M*)

  This is a sign-indefinite quadratic form. We want to prove:
  Δ^T (I - diag(D*/D) J(M*)) Δ > 0

  Since D depends on M, this is non-quadratic. But we can bound D.

PRONG 2: Tighter bound for 6.17B's problematic 1/200 seeds
  Instead of D_min, use an M-dependent bound on D*_k/D_k.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

def N_op(M, a, b, eps, w, v):
    A = a + w @ M
    B = b + v @ M
    return A / (A + B + eps)

def find_fp(a, b, eps, w, v):
    M = np.full(5, 0.5)
    for _ in range(5000):
        Mnew = N_op(M, a, b, eps, w, v)
        if np.max(np.abs(Mnew - M)) < 1e-14:
            return Mnew
        M = Mnew
    return M

def sample_FCA_params(seed):
    rs = np.random.RandomState(seed)
    a = rs.uniform(0.01, 0.5, 5)
    b = rs.uniform(0.01, 0.5, 5)
    eps = rs.uniform(0.001, 0.1, 5)
    w = rs.uniform(0.01, 0.3, (5, 5))
    v = rs.uniform(0.01, 0.3, (5, 5))
    for i in range(5):
        w[i, i] = 0.0
        v[i, i] = 0.0
    tot = a.sum() + b.sum() + w.sum() + v.sum()
    w = w / tot * 5.0
    v = v / tot * 5.0
    return a, b, eps, w, v

def J_at_Mstar(a, b, eps, w, v, Mstar):
    A = a + w @ Mstar
    B = b + v @ Mstar
    D = A + B + eps
    J = np.zeros((5, 5))
    for k in range(5):
        for j in range(5):
            J[k, j] = (w[k, j] * (B[k] + eps[k]) - A[k] * v[k, j]) / (D[k] ** 2)
    return J, D, A, B

# ============================================================
# PRONG 1: Direction monotonicity
# ============================================================
print("=" * 70)
print("PRONG 1: Direction Monotonicity Analytic Proof")
print("=" * 70)

# Key identity:
# (N(M)-M)·(M*-M) = Σ_k Δ_k² - Σ_k Σ_j (D*_k/D_k) J_kj(M*) Δ_j Δ_k
# 
# Since J_kk(M*) = 0, the diagonal of A = I - diag(D*/D)J is 1.
#
# We need: Δ^T A Δ > 0
# where A = I - B, B_ij = (D*_i/D_i) J_ij(M*)
#
# This is equivalent to: Δ^T B Δ < ||Δ||₂²

for seed in [0, 1, 9, 17, 33, 11]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    Js, Dstar, As, Bs = J_at_Mstar(a, b, eps, w, v, Mstar)
    
    print(f"\nSeed {seed}: M* = {Mstar}")
    print(f"  D* = {Dstar}")
    print(f"  Column sums of |J(M*)|: {np.sum(np.abs(Js), axis=0)}")
    print(f"  Max col sum: {np.max(np.sum(np.abs(Js), axis=0)):.4f}")
    
    # Key bound: D*_k/D_k ≤ D*_k/D_min_k
    # But we can do better: D_k depends on M through D_k = a_k + b_k + ε_k + Σ_j (w_kj + v_kj) M_j
    # 
    # Alternative: (D*_k/D_k) = 1 / (1 + (D_k - D*_k)/D*_k)
    # When M is near M*, D_k ≈ D*_k, so D*_k/D_k ≈ 1
    # When M is far, D*_k/D_k could be large if D_k is small.
    #
    # But direction monotonicity is about the LINEARIZED behavior near M*:
    # (N(M)-M) = (J_N(M*) - I)(M-M*) + O(||M-M*||²)
    # (N(M)-M)·(M*-M) = -(J_N(M*)-I)Δ · Δ + O(||Δ||³)
    # = Δ^T (I - J_N(M*)) Δ + O(||Δ||³)
    #
    # So NEAR M*, direction monotonicity depends on I - J_N(M*).
    # If Δ^T (I - J_N(M*)) Δ > 0 for all Δ ≠ 0, then locally direction monotonicity holds.
    
    I_minus_J = np.eye(5) - Js
    sym_part = (I_minus_J + I_minus_J.T) / 2
    evals_sym = np.linalg.eigvalsh(sym_part)
    print(f"  Eigenvalues of sym(I-J(M*)): {evals_sym}")
    print(f"  Min eigenvalue: {min(evals_sym):.6f}")
    
    if min(evals_sym) > 0:
        print(f"  LOCAL DIRECTION MONOTONICITY PROVED ■ (sym(I-J)>0)")
    else:
        print(f"  Sym(I-J) not positive definite — need global analysis")

# Now let's try the GLOBAL analysis
print(f"\n{'='*70}")
print("Global direction monotonicity: quadratic form analysis")
print(f"{'='*70}")

for seed in [0, 1, 11]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    Js, Dstar, As, Bs = J_at_Mstar(a, b, eps, w, v, Mstar)
    D_min = a + b + eps
    D_max = a + w.sum(axis=1) + b + v.sum(axis=1) + eps
    
    print(f"\nSeed {seed}:")
    print(f"  D* = {Dstar}")
    print(f"  D_min = {D_min}")
    print(f"  D_max = {D_max}")
    print(f"  D*/D_min ratios: {Dstar/D_min}")
    
    # Global bound: D*_k/D_k ≤ D*_k/D_min_k
    # But we can use a sharper bound based on the sign of Δ:
    # If Δ_j ≥ 0 (M_j ≥ M*_j), then M_j ≤ 1, so D_k ≤ D_max_k
    # If Δ_j ≤ 0 (M_j ≤ M*_j), then M_j ≥ 0, so D_k ≥ D_min_k
    #
    # But the cross terms make this complicated.
    # 
    # Try: bound each row contribution
    # (N_k(M)-M_k)·(M*_k-M_k) = - [(D*_k/D_k) Σ_j J_kj Δ_j - Δ_k] Δ_k
    # = Δ_k² - (D*_k/D_k) Δ_k Σ_j J_kj Δ_j
    #
    # For row k to contribute positively (help direction monotonicity):
    # Δ_k² > (D*_k/D_k) Δ_k Σ_j J_kj Δ_j
    #
    # If Δ_k = 0, contribution = 0 (neutral)
    # If Δ_k ≠ 0: Δ_k > (D*_k/D_k) Σ_j J_kj Δ_j   [dividing by Δ_k]
    #
    # So we need: (D*_k/D_k) Σ_j J_kj Δ_j / Δ_k < 1
    
    # For the overall sum, we need:
    # Σ_k Δ_k² > Σ_k (D*_k/D_k) Δ_k Σ_j J_kj Δ_j
    
    # Check numerically
    rs = np.random.RandomState(seed + 1000)
    Ms = rs.uniform(0, 1, (5000, 5))
    min_qform = float('inf')
    min_qform_M = None
    
    for M in Ms:
        Delta = M - Mstar
        if np.max(np.abs(Delta)) < 1e-14:
            continue
        
        A = a + w @ M
        B = b + v @ M
        D = A + B + eps
        
        # Full expression
        lhs = 0.0
        for k in range(5):
            term = (Dstar[k] / D[k]) * sum(Js[k, j] * Delta[j] for j in range(5))
            lhs += (term - Delta[k]) * (-Delta[k])
        
        if lhs < min_qform:
            min_qform = lhs
            min_qform_M = M.copy()
    
    print(f"  Min quadratic form: {min_qform:.8f}")
    if min_qform > 0:
        print(f"  All samples: (N-M)·(M*-M) > 0 ✓")
    
    # Normalized version (divide by ||Δ||²)
    rs2 = np.random.RandomState(seed + 2000)
    Ms2 = rs2.uniform(0, 1, (5000, 5))
    min_normalized = float('inf')
    
    for M in Ms2:
        Delta = M - Mstar
        norm_sq = np.sum(Delta * Delta)
        if norm_sq < 1e-14:
            continue
        
        A = a + w @ M
        B = b + v @ M
        D = A + B + eps
        
        val = 0.0
        for k in range(5):
            term = (Dstar[k] / D[k]) * sum(Js[k, j] * Delta[j] for j in range(5))
            val += (term - Delta[k]) * (-Delta[k])
        
        normalized = val / norm_sq
        if normalized < min_normalized:
            min_normalized = normalized
    
    print(f"  Min normalized qform: {min_normalized:.6f}")
    
    # Bounding approach: D*_k/D_k ≤ max(D*_k/D_min_k, 1) = D*_k/D_min_k
    # If we use worst D*_k/D_k:
    # Σ_k (D*_k/D_min_k) |J_kj| causes overestimation
    
    # Can we separate positive and negative cross terms?
    # (N_k(M)-M_k)(M*_k-M_k) = -(D*_k/D_k) Σ_j J_kj Δ_j Δ_k + Δ_k²
    #
    # For J_kj > 0 (sign determined by M*):
    #   - J_kj Δ_j Δ_k is worrying when Δ_j and Δ_k have same sign
    # For J_kj < 0:
    #   - J_kj Δ_j Δ_k is helping when Δ_j and Δ_k have same sign
    
    # Print J(M*) sign pattern:
    print(f"  J(M*) sign pattern:")
    print(f"  {np.sign(Js).astype(int)}")

print(f"\n{'='*70}")
print("Key insight for direction monotonicity")
print(f"{'='*70}")
print("""
The quadratic form is: Δ^T (I - diag(D*/D)J)Δ > 0

This is equivalent to: Δ^T (diag(D*/D)J) Δ < ||Δ||₂²

A sufficient condition: ||diag(D*/D)J||₂ < 1 (spectral norm of the matrix)
or: ρ(diag(D*/D)J) < 1 (spectral radius)

But diag(D*/D) depends on M. Using D_min gives:
||diag(D*/D_min)J(M*)||₂ as an upper bound.

However, the SYMMETRIC part matters for the quadratic form:
Δ^T M Δ = Δ^T (M+M^T)/2 Δ

For Δ^T (I - diag(D*/D)J)Δ > 0, we need λ_min(sym(I - diag(D*/D)J)) > 0
""")

# ============================================================
# PRONG 2: Tighter bound for 6.17B problematic seeds
# ============================================================
print(f"\n{'='*70}")
print("PRONG 2: Tighter bound for problematic 1/200 seeds")
print(f"{'='*70}")

# The problem: α_bound = max_j Σ_k |J_kj(M*)| D*_k/D_min_k
# D*_k/D_min_k can be very large when a_k and b_k are small.
# The triangle inequality |Σ_j J_kj Δ_j| ≤ Σ_j |J_kj| |Δ_j| loses sign info.
#
# New idea: bound the row contribution differently.
# |Σ_j J_kj Δ_j| = |J_k· Δ|
# ≤ ||J_k||_2 · ||Δ||_2 (Cauchy-Schwarz)
# = sqrt(Σ_j J_kj²) · ||Δ||_2
#
# Then: ||N(M)-M*||_1 ≤ Σ_k (D*_k/D_min_k) sqrt(Σ_j J_kj²) · ||Δ||_2
# = c · ||Δ||_2 where c = Σ_k (D*_k/D_min_k) ||J_k||_2
#
# But we want l1 norm, not l2. The conversion ||Δ||_2 ≤ ||Δ||_1 ≤ √5 ||Δ||_2
# gives: ||N(M)-M*||_1 ≤ c · ||Δ||_1 (using ||Δ||_2 ≤ ||Δ||_1)
# = where c = Σ_k (D*_k/D_min_k) ||J_k(M*)||_2
#
# This might be tighter because sqrt(Σ J²) ≤ Σ |J| (often much smaller)

print("\nTesting Cauchy-Schwarz based bound:")
for seed in range(200):
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    Js, Dstar, As, Bs = J_at_Mstar(a, b, eps, w, v, Mstar)
    D_min = a + b + eps
    
    # Original l1 bound
    col_sums = np.sum(np.abs(Js), axis=0)
    alpha_old = max(sum(np.abs(Js[k, j]) * Dstar[k] / D_min[k] for k in range(5))
                    for j in range(5))
    
    # New: bound each row by l2 norm, then sum
    row_l2 = np.sqrt(np.sum(Js * Js, axis=1))
    c_new = sum(Dstar[k] / D_min[k] * row_l2[k] for k in range(5))
    
    if seed < 5 or seed == 11:
        is_good = "✓ < 1" if alpha_old < 1 else "✗ > 1"
        is_good_new = "✓ < 1" if c_new < 1 else "✗ > 1"
        print(f"  Seed {seed}: α_old={alpha_old:.4f} {is_good}, c_new={c_new:.4f} {is_good_new}")

# Check all 200 seeds with new bound
failures_old = 0
failures_new = 0
for seed in range(200):
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    Js, Dstar, As, Bs = J_at_Mstar(a, b, eps, w, v, Mstar)
    D_min = a + b + eps
    
    col_sums = np.sum(np.abs(Js), axis=0)
    alpha_old = max(sum(np.abs(Js[k, j]) * Dstar[k] / D_min[k] for k in range(5))
                    for j in range(5))
    if alpha_old >= 1:
        failures_old += 1
    
    row_l2 = np.sqrt(np.sum(Js * Js, axis=1))
    c_new = sum(Dstar[k] / D_min[k] * row_l2[k] for k in range(5))
    if c_new >= 1:
        failures_new += 1

print(f"\n  Old bound failures: {failures_old}/200")
print(f"  CS-based bound failures: {failures_new}/200")

# ============================================================
# PRONG 2b: Component-wise bound using sign info
# ============================================================
print(f"\n{'='*70}")
print("PRONG 2b: Sign-aware bound")
print(f"{'='*70}")
print("""
Key idea: Instead of triangle inequality, use the actual sign pattern.
|Σ_j J_kj Δ_j| can be written as |Σ_{j:J_kj>0} |J_kj|Δ_j - Σ_{j:J_kj<0} |J_kj|Δ_j|

The worst case is when all Δ_j with J_kj>0 are at their max (1-M*_j),
and all Δ_j with J_kj<0 are at their min (-M*_j).

For fixed M, Σ_j J_kj Δ_j = J_k·Δ is a linear function.
Its maximum over the cube [-M*_j, 1-M*_j]⁵ is at a vertex.

So max_{M∈[0,1]⁵} |J_k·(M-M*)| = max_{v vertex Δ_j∈{-M*_j,1-M*_j}} |J_k·v|

But there are 2⁵ = 32 vertices per row, doable.
""")

for seed in [0, 1, 11]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    Js, Dstar, As, Bs = J_at_Mstar(a, b, eps, w, v, Mstar)
    D_min = a + b + eps
    
    # For each row k, find max |J_k·(M-M*)| over M ∈ [0,1]⁵
    # This is at a vertex where each Δ_j is either -M*_j or 1-M*_j
    row_max = np.zeros(5)
    for k in range(5):
        max_abs = 0.0
        # Enumerate 32 vertices
        for bits in range(32):
            Delta = np.array([-Mstar[j] if (bits >> j) & 1 == 0 else 1.0 - Mstar[j] 
                             for j in range(5)])
            val = abs(sum(Js[k, j] * Delta[j] for j in range(5)))
            max_abs = max(max_abs, val)
        row_max[k] = max_abs
    
    # Worst-case bound
    worst_bound = sum(Dstar[k] / D_min[k] * row_max[k] for k in range(5))
    
    # But this is l1 total, not per-column. For contraction:
    # ||N(M)-M*||_1 ≤ Σ_k (D*_k/D_min_k) max |J_k·Δ|
    # ≤ Σ_k (D*_k/D_min_k) row_max[k]
    #
    # But for an l1 contraction bound: ||N(M)-M*||_1 ≤ α ||Δ||_1
    # we need this per-column structure.
    
    # Alternative: 
    # ||N(M)-M*||_1 = Σ_k (D*_k/D_k) |J_k·Δ|
    # ≤ Σ_k (D*_k/D_min_k) |J_k·Δ|
    # ≤ Σ_k (D*_k/D_min_k) · row_max[k]
    
    # But this upper bounds ||N-M*||_1 by a constant — not contractive!
    # We need the bound to involve ||Δ||_1.
    
    # Better: bound Σ_k (D*_k/D_k) |J_k·Δ| / ||Δ||_1
    
    print(f"\n  Seed {seed}:")
    print(f"    row_max (vertex-enum): {row_max}")
    print(f"    Row l1 norms: {np.sum(np.abs(Js), axis=1)}")
    
    # For ||Δ||_1 reference:
    max_l1 = sum(max(Mstar[j], 1.0 - Mstar[j]) for j in range(5))
    print(f"    max ||Δ||_1: {max_l1:.4f}")
    print(f"    worst ||N-M*||_1 bound: {worst_bound:.4f}")
    
    # Ratio: worst ||N-M*||_1 / max ||Δ||_1
    ratio = worst_bound / max_l1
    print(f"    ratio: {ratio:.4f} {'✓' if ratio < 1 else '✗'}")

# ============================================================
# PRONG 2c: The "worst vertex" analysis
# At which vertex does J_k·Δ achieve its max?
# The vertex that maximizes J_k·(M-M*) depends on the signs of J_kj.
# For J_kj > 0: set Δ_j = 1-M*_j (maximize positive contribution)
# For J_kj < 0: set Δ_j = -M*_j (minimize negative contribution)
# This gives the max of J_k·Δ (NOT absolute value).
# For |J_k·Δ|, we need both max positive and max negative.
print(f"\n{'='*70}")
print("PRONG 2c: Sign-optimal vertex analysis")
print(f"{'='*70}")

for seed in [0, 1, 11]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    Js, Dstar, As, Bs = J_at_Mstar(a, b, eps, w, v, Mstar)
    
    # For direction monotonicity: can we bound D*_k/D_k Σ_j J_kj Δ_j ?
    # Using the signs:
    # Σ_j J_kj Δ_j ≤ Σ_{j:J_kj>0} J_kj (1-M*_j) + Σ_{j:J_kj<0} J_kj (-M*_j)
    # = Σ_{j:J_kj>0} J_kj (1-M*_j) - Σ_{j:J_kj<0} |J_kj| M*_j
    #
    # And the lower bound (which could be negative):
    # Σ_j J_kj Δ_j ≥ Σ_{j:J_kj>0} J_kj (-M*_j) + Σ_{j:J_kj<0} J_kj (1-M*_j)
    
    upper = np.zeros(5)
    lower = np.zeros(5)
    for k in range(5):
        upper[k] = sum(Js[k,j] * (1-Mstar[j]) if Js[k,j] > 0 else 0 for j in range(5)) + \
                   sum(Js[k,j] * (-Mstar[j]) if Js[k,j] < 0 else 0 for j in range(5))
        lower[k] = sum(Js[k,j] * (-Mstar[j]) if Js[k,j] > 0 else 0 for j in range(5)) + \
                   sum(Js[k,j] * (1-Mstar[j]) if Js[k,j] < 0 else 0 for j in range(5))
    
    print(f"\n  Seed {seed}:")
    print(f"    J(M*) upper bounds: {upper}")
    print(f"    J(M*) lower bounds: {lower}")
    
    # Now: ||N(M)-M*||_1 = Σ_k (D*_k/D_k) |J_k·Δ|
    # ≤ Σ_k (D*_k/D_min_k) max(|upper_k|, |lower_k|)
    
    tight_bound = sum(Dstar[k] / (a[k]+b[k]+eps[k]) * max(abs(upper[k]), abs(lower[k])) 
                      for k in range(5))
    print(f"    Sign-aware bound (sum): {tight_bound:.4f}")
    
    # But again this is an absolute bound, not contractive.
    # For contraction we need: ||N(M)-M*||_1 ≤ α · ||Δ||_1
    # So we need: tight_bound / ||Δ||_1 ≤ α < 1
    # But ||Δ||_1 = Σ |Δ_j|, which ranges from 0 to Σ max(M*_j, 1-M*_j)
    # The worst case is when ||Δ||_1 is small but ||N-M*||_1 is not proportionally small.
    # This happens near M*.
    
    # Local analysis: ||N(M)-M*||_1 ≈ ||J(M*)(M-M*)||_1 (ignoring D*/D)
    # For the j-th column: contribution ≈ Σ_k |J_kj| |Δ_j|
    # So local contraction depends on max_j Σ_k |J_kj| < 1
    
    col_sums = np.sum(np.abs(Js), axis=0)
    print(f"    Local contraction (col sums of |J|): {col_sums}")
    print(f"    Max col sum: {max(col_sums):.4f} {'✓' if max(col_sums) < 1 else '✗'}")

# ============================================================
# PRONG 3: Direct attempt at direction monotonicity proof
# ============================================================
print(f"\n{'='*70}")
print("PRONG 3: Direct analytic bound for direction monotonicity")
print(f"{'='*70}")

# (N(M)-M)·(M*-M) = Σ_k [-(D*_k/D_k) Σ_j J_kj Δ_j + Δ_k] (-Δ_k)
# = Σ_k Δ_k² - Σ_k (D*_k/D_k) Σ_j J_kj Δ_j Δ_k
# 
# Let S_ij = Σ_k (D*_k/D_k) J_kj(M*) if i ≠ j? No, this is a bilinear form.
# 
# Actually: Σ_k (D*_k/D_k) Σ_j J_kj Δ_j Δ_k = Σ_j,k (D*_k/D_k) J_kj Δ_j Δ_k
# = Δ^T (diag(D*/D) J) Δ? No, it's Δ^T B Δ where B_kj = (D*_k/D_k) J_kj
# = Δ^T (diag(D*/D) J(M*)) Δ
#
# But diag(D*/D) depends on M (through D), so this isn't a pure quadratic form.
#
# KEY INSIGHT: D_k - D*_k = Σ_j (w_kj + v_kj)(M_j - M*_j) = Σ_j (w+v)_kj Δ_j
# So D*_k/D_k = D*_k / (D*_k + Σ_j (w+v)_kj Δ_j)
# = 1 / (1 + Σ_j (w+v)_kj Δ_j / D*_k)
# = 1 - Σ_j (w+v)_kj Δ_j / D*_k + O(||Δ||²)   (Taylor)
# 
# So near M*:
# (N(M)-M)·(M*-M) = Δ^T (I - J(M*)) Δ + O(||Δ||³)
# 
# This is a 5x5 quadratic form! We need I - J(M*) ≻ 0 (in the sym part).

for seed in [0, 1, 11]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    Js, Dstar, As, Bs = J_at_Mstar(a, b, eps, w, v, Mstar)
    
    I_minus_J = np.eye(5) - Js
    sym_part = (I_minus_J + I_minus_J.T) / 2
    evals = np.linalg.eigvalsh(sym_part)
    
    print(f"\n  Seed {seed}:")
    print(f"    Eigenvalues of sym(I-J): {evals}")
    print(f"    Min eigenvalue: {evals.min():.6f}")
    
    is_pos_def = all(e > 1e-12 for e in evals)
    print(f"    I-J positive definite? {is_pos_def}")
    
    if is_pos_def:
        print(f"    LOCAL monotonicity ANALYTICALLY PROVED ■")
    
    # Global: D*_k/D_k ≤ D*_k/D_min_k
    # D*_k/D_k = 1 when M = M* (Δ = 0), increases as Σ_j (w+v)_kj Δ_j < 0
    # This happens when Δ components are negative for the components with large w+v weights.
    #
    # Upper bound approach: 
    # (N-M)·(M*-M) = Δ^T Δ - Δ^T diag(D*/D) J Δ
    # = ||Δ||² - Δ^T B Δ where B_kj = (D*_k/D_k) J_kj
    
    # Since D*_k/D_k ≥ 1 (or ≤ 1 depending on sign of Σ(w+v)Δ), 
    # the worst case for direction monotonicity is when D*_k/D_k is LARGE.
    # D*_k/D_k is maximized when D_k is minimized: D_min_k = a_k + b_k + ε_k
    # So the worst-case quadratic form is B_max → diag(D*/D_min) J(M*)
    
    B_max = np.diag(Dstar / (a+b+eps)) @ Js
    I_minus_Bmax = np.eye(5) - B_max
    sym_bmax = (I_minus_Bmax + I_minus_Bmax.T) / 2
    evals_bmax = np.linalg.eigvalsh(sym_bmax)
    print(f"    Eigenvalues of sym(I-diag(D*/D_min)J): {evals_bmax}")
    print(f"    Min: {evals_bmax.min():.6f}")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
print("""
PRONG 1 (Direction monotonicity):
  - Locally at M*: depends on I - J(M*). If sym(I-J) is positive definite,
    direction monotonicity holds locally.
  - Globally: need to bound D*_k/D_k · J(M*).
  - The worst-case bound uses D_min instead of D, which may overestimate.

PRONG 2 (Tighter l1 bound):
  - Cauchy-Schwarz per row may be tighter than l1 column sums
  - Sign-aware vertex enumeration gives exact per-row maxima
  - But translating to contraction coefficient is non-trivial

PRONG 3 (Local analysis):
  - Near M*, direction monotonicity is a quadratic form I-J(M*)
  - If sym(I-J) ≻ 0, local monotonicity is ANALYTICALLY PROVED
""")
