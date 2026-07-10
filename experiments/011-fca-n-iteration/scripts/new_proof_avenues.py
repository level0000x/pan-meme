"""
New proof avenues for the remaining ◆ gaps.
All three gaps converge on: analytically handling J(M*) sign cancellation.

APPROACH 1: KL divergence as global Lyapunov function
  V(M) = Σ_k [M_k ln(M_k/M*_k) + (1-M_k)ln((1-M_k)/(1-M*_k))]
  If V(N(M)) < V(M) for all M ≠ M*, global convergence follows directly.

APPROACH 2: Logit coordinate transformation
  y_k = log(M_k/(1-M_k)) = logit(M_k)
  The dynamics in y-space may linearize or take a simpler form.

APPROACH 3: Dual system from fixed-point equation
  M*_k D*_k = a_k + (wM*)_k
  (1-M*_k) D*_k = b_k + e_k + (vM*)_k
  These are LINEAR equations in (D*, M*). Can we bound J_kj sums?

APPROACH 4: Componentwise monotonicity + cooperative dynamics
  ∂N_k/∂M_j = J_kj(M) * (D*_k/D_k pattern?) - actually the Jacobian
  depends on M, but at M* it's J(M*). The sign pattern of J
  determines monotonicity direction.

APPROACH 5: Convexity of N_k and vertex principle
  We found d²N_k/dM_j² doesn't change sign. If N_k is componentwise
  convex/concave, then for any M, N_k(M) is bounded by convex
  combinations of vertex values. This + vertex l1 contraction
  might prove global l1 contraction.
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
    for _ in range(10000):
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

# ============================================================
# APPROACH 1: KL divergence as Lyapunov function
# ============================================================
print("=" * 70)
print("APPROACH 1: KL divergence as global Lyapunov function")
print("=" * 70)
print("""
V(M) = Σ_k [M_k ln(M_k/M*_k) + (1-M_k) ln((1-M_k)/(1-M*_k))]

This is the binary KL divergence, naturally convex for M_k ∈ (0,1).
M* is the unique global minimum (V ≥ 0, V = 0 iff M = M*).

If ΔV = V(N(M)) - V(M) < 0 for all M ≠ M*, then N is globally
convergent by Lyapunov's direct method.

Why this might work: N_k = A_k/D_k is a ratio of affine functions.
Taking logit: logit(N_k) = ln(A_k) - ln(B_k + eps_k)
The denominator structure might create cancellations in the KL form.
""")

def kl_div(M, Mstar):
    """Binary KL divergence."""
    val = 0.0
    for k in range(5):
        if M[k] > 1e-15 and M[k] < 1 - 1e-15:
            val += M[k] * np.log(M[k] / Mstar[k])
            val += (1 - M[k]) * np.log((1 - M[k]) / (1 - Mstar[k]))
    return val

for seed in [0, 1, 11, 42, 99]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    print(f"\n  Seed {seed}: M* = {Mstar}")
    
    # Test KL decrease
    rng = np.random.RandomState(seed + 5555)
    kl_ratios = []
    kl_diffs = []
    
    for _ in range(5000):
        M = rng.uniform(0.001, 0.999, 5)
        A = a + w @ M
        B = b + v @ M
        D = A + B + eps
        NM = A / D
        
        kl_before = kl_div(M, Mstar)
        kl_after = kl_div(NM, Mstar)
        
        if kl_before > 1e-14:
            kl_ratios.append(kl_after / kl_before)
        kl_diffs.append(kl_after - kl_before)
    
    kl_ratios = np.array(kl_ratios)
    kl_diffs = np.array(kl_diffs)
    
    print(f"    max KL ratio: {kl_ratios.max():.4f}")
    print(f"    mean KL ratio: {kl_ratios.mean():.4f}")
    print(f"    All KL decrease? {(kl_diffs < 0).all()}")
    if not (kl_diffs < 0).all():
        print(f"    KL INCREASE found! max diff = {kl_diffs.max():.10f}")

# Check vertex behavior of KL
print(f"\n  Vertex KL check:")
for seed in [0, 1, 11]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    min_kl_ratio = float('inf')
    max_kl_ratio = 0.0
    for bits in range(32):
        M = np.array([0.001 if (bits >> j) & 1 == 0 else 0.999 for j in range(5)])
        A = a + w @ M
        B = b + v @ M
        D = A + B + eps
        NM = A / D
        
        kl_before = kl_div(M, Mstar)
        kl_after = kl_div(NM, Mstar)
        if kl_before > 1e-14:
            ratio = kl_after / kl_before
            min_kl_ratio = min(min_kl_ratio, ratio)
            max_kl_ratio = max(max_kl_ratio, ratio)
    
    print(f"    Seed {seed}: KL ratio range [{min_kl_ratio:.4f}, {max_kl_ratio:.4f}]")

# ============================================================
# APPROACH 2: Logit coordinate transformation
# ============================================================
print(f"\n{'='*70}")
print("APPROACH 2: Logit coordinate transformation")
print(f"{'='*70}")
print("""
y_k = logit(M_k) = ln(M_k/(1-M_k))

N_k = A_k/D_k → logit(N_k) = ln A_k - ln(B_k + eps_k)

A_k = a_k + Σ_j w_kj M_j = a_k + Σ_j w_kj sigmoid(y_j)
B_k = b_k + Σ_j v_kj M_j = b_k + Σ_j v_kj sigmoid(y_j)

This is NOT linear in y-space.

But: near any M, the Jacobian in y-coordinates may be simpler.
dy'_k/dy_j = (∂logit(N_k)/∂M_j) * (∂M_j/∂y_j)
           = [J_kj(M) / (N_k(1-N_k))] * [M_j(1-M_j)]
           
At M = M*: N_k = M*_k, so:
dy'_k/dy_j|_{M*} = J_kj(M*) * M*_j(1-M*_j) / (M*_k(1-M*_k))
                = J_kj(M*) * (M*_j(1-M*_j)) / (M*_k(1-M*_k))

This is a SIMILARITY transformation of J(M*)!
Let S = diag(√(M*_k(1-M*_k)))
Then J_y = S^{-1} J(M*) S is a similarity transform → SAME eigenvalues!

So in logit space, the LINEARIZATION has the SAME spectral radius!
This means the local contraction rate is preserved under logit transform.
""")

# Verify spectral radius preservation
for seed in [0, 1, 11]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    D_s = a + w @ Mstar + b + v @ Mstar + eps
    
    Js = np.zeros((5, 5))
    for k in range(5):
        for j in range(5):
            Js[k, j] = (w[k, j] * (1 - Mstar[k]) - Mstar[k] * v[k, j]) / D_s[k]
    
    S = np.diag(np.sqrt(Mstar * (1 - Mstar)))
    S_inv = np.diag(1.0 / np.sqrt(Mstar * (1 - Mstar)))
    J_y = S_inv @ Js @ S
    
    evals_J = np.linalg.eigvals(Js)
    evals_Jy = np.linalg.eigvals(J_y)
    
    print(f"\n  Seed {seed}:")
    print(f"    ρ(J) = {max(abs(evals_J)):.4f}, ρ(J_y) = {max(abs(evals_Jy)):.4f}")
    print(f"    Match? {np.allclose(sorted(evals_J), sorted(evals_Jy))}")
    print(f"    Is J_y symmetric? {np.allclose(J_y, J_y.T, atol=1e-12)}")
    
    # Since J_y is similar to J, it preserves the same spectral radius.
    # But J_y might have special structure!
    # Check if J_y has any nice property
    print(f"    J_y row sums: {J_y.sum(axis=1)}")

# ============================================================
# APPROACH 3: Dual system / fixed-point constraints
# ============================================================
print(f"\n{'='*70}")
print("APPROACH 3: Fixed-point equation constraints on J")
print(f"{'='*70}")
print("""
Fixed-point equations:
  (1) M*_k D*_k = a_k + (wM*)_k  →  a_k = M*_k D*_k - (wM*)_k
  (2) (1-M*_k)D*_k = b_k + e_k + (vM*)_k  →  b_k = (1-M*_k)D*_k - e_k - (vM*)_k

The key: D*_k is constrained by these. M*_k ∈ (0,1).

Row DD for I-J: Σ_{j≠k} |J_kj| < D*_k
Let's express J_kj * D*_k = w_kj(1-M*_k) - M*_k v_kj

So row DD condition becomes:
  Σ_{j≠k} |w_kj(1-M*_k) - M*_k v_kj| < (a_k + b_k + e_k + (w+v)_k M*_k)

Split by sign of w_kj(1-M*_k) - M*_k v_kj:
Let P_k = {j ≠ k : w_kj(1-M*_k) > M*_k v_kj}  (positive terms)
Let N_k = {j ≠ k : w_kj(1-M*_k) < M*_k v_kj}  (negative terms)

Then:
  Σ_{j∈P_k} [w_kj(1-M*_k) - M*_k v_kj] + Σ_{j∈N_k} [M*_k v_kj - w_kj(1-M*_k)]
  < D*_k

Rearranged:
  (1-M*_k) Σ_{j∈P_k} w_kj - M*_k Σ_{j∈P_k} v_kj + M*_k Σ_{j∈N_k} v_kj - (1-M*_k) Σ_{j∈N_k} w_kj
  < a_k + b_k + e_k + Σ_j w_kj M*_j + Σ_j v_kj M*_j

This is a bound INVOLVING M*. Can we use the FP equation to simplify?
Let's compute actual values.
""")

for seed in range(5):
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    D_s = a + w @ Mstar + b + v @ Mstar + eps
    
    for k in range(5):
        Jk_D = np.array([w[k,j]*(1-Mstar[k]) - Mstar[k]*v[k,j] for j in range(5)])
        P_sum = sum(max(jd, 0) for jd in Jk_D)
        N_sum = sum(abs(min(jd, 0)) for jd in Jk_D)
        total = P_sum + N_sum
        
        # Row DD condition: total < D*_k
        margin = D_s[k] - total
        
        # Upper bound via triangle inequality
        upper_P = (1-Mstar[k]) * sum(w[k,j] for j in range(5) if w[k,j]*(1-Mstar[k]) > Mstar[k]*v[k,j])
        upper_N = Mstar[k] * sum(v[k,j] for j in range(5) if w[k,j]*(1-Mstar[k]) < Mstar[k]*v[k,j])
        
        print(f"  S{seed} row{k}: exact={total:.4f} margin={margin:.4f} D*={D_s[k]:.4f} " +
              f"DD={'✓' if total < D_s[k] else '✗'}")

# ============================================================
# APPROACH 4: Cooperative/monotone system structure
# ============================================================
print(f"\n{'='*70}")
print("APPROACH 4: Cooperative/monotone dynamical systems approach")
print(f"{'='*70}")
print("""
N_k = A_k/D_k = (a_k + w_k·M) / (a_k + b_k + e_k + (w+v)_k·M)

∂N_k/∂M_j = [w_kj D_k - A_k (w_kj + v_kj)] / D_k²
          = [w_kj D_k - N_k D_k (w_kj + v_kj)] / D_k²
          = [w_kj - N_k (w_kj + v_kj)] / D_k
          = [w_kj(1-N_k) - N_k v_kj] / D_k

So at ANY M: ∂N_k/∂M_j = [w_kj(1-N_k) - N_k v_kj] / D_k

This has the SAME sign structure as J_kj(M*) but with N_k instead of M*_k!

Crucially: if w_kj(1-N_k) > N_k v_kj at all M, then ∂N_k/∂M_j > 0 everywhere.
But sign can change because N_k changes.

However: sign of ∂N_k/∂M_j = sign of w_kj(1-N_k) - N_k v_kj
= sign of w_kj - N_k(w_kj + v_kj)
= sign of w_kj/(w_kj+v_kj) - N_k

So ∂N_k/∂M_j > 0 iff N_k < w_kj/(w_kj+v_kj)
∂N_k/∂M_j < 0 iff N_k > w_kj/(w_kj+v_kj)

As N_k evolves from 0 to M*_k to 1, the sign of ∂N_k/∂M_j may flip
when N_k crosses the threshold w_kj/(w_kj+v_kj).

This is a THRESHOLD MONOTONICITY structure! Very rich dynamics.
""")

# Check: for each (k,j), does N_k cross the threshold during iteration?
for seed in [0, 11]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    print(f"\n  Seed {seed}: Threshold analysis")
    for k in range(5):
        for j in range(5):
            if k == j:
                continue
            threshold = w[k,j] / (w[k,j] + v[k,j]) if (w[k,j] + v[k,j]) > 1e-15 else 0.5
            above = Mstar[k] > threshold
            sign_at_fp = "positive" if w[k,j]*(1-Mstar[k]) > Mstar[k]*v[k,j] else "negative"
            print(f"    ({k},{j}): threshold={threshold:.3f}, M*_k={Mstar[k]:.3f}, " +
                  f"M*_k vs thr: {'above' if above else 'below'}, sign={sign_at_fp}")

# ============================================================
# APPROACH 5: Convexity + vertex principle for global l1 contraction
# ============================================================
print(f"\n{'='*70}")
print("APPROACH 5: Convexity-based vertex principle")
print(f"{'='*70}")
print("""
We found: d²N_k/dM_j² has fixed sign (never crosses zero for each (k,j) pair).

This means N_k(M) as a function of M_j (holding other components fixed)
is either purely convex or purely concave.

If N_k is convex in M_j: N_k(M) ≤ λN_k(0) + (1-λ)N_k(1) for M_j = λ·0 + (1-λ)·1
  where other components are fixed.

For a function that is componentwise convex/concave in EACH variable,
the maximum over a hypercube is achieved at a VERTEX.

Let's verify this property for the full l1 contraction ratio.
""")

# Verify: is ||N(M)-M*||_1 / ||M-M*||_1 maximized at a vertex?
for seed in [0, 1, 11, 42]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    # Vertex max
    vertex_max = 0.0
    for bits in range(32):
        M = np.array([0.0 if (bits >> j) & 1 == 0 else 1.0 for j in range(5)])
        Delta = M - Mstar
        l1_bef = np.sum(np.abs(Delta))
        if l1_bef < 1e-14:
            continue
        A = a + w @ M
        B = b + v @ M
        D = A + B + eps
        NM = A / D
        ratio = np.sum(np.abs(NM - Mstar)) / l1_bef
        vertex_max = max(vertex_max, ratio)
    
    # Random interior points
    rng = np.random.RandomState(seed + 8888)
    interior_max = 0.0
    worst_interior = None
    for _ in range(10000):
        M = rng.uniform(0, 1, 5)
        Delta = M - Mstar
        l1_bef = np.sum(np.abs(Delta))
        if l1_bef < 1e-14:
            continue
        A = a + w @ M
        B = b + v @ M
        D = A + B + eps
        NM = A / D
        ratio = np.sum(np.abs(NM - Mstar)) / l1_bef
        if ratio > interior_max:
            interior_max = ratio
            worst_interior = M.copy()
    
    print(f"  Seed {seed}: vertex max={vertex_max:.4f}, interior max={interior_max:.4f}, " +
          f"{'vertex≥interior ✓' if vertex_max >= interior_max - 1e-12 else 'INTERIOR > VERTEX ✗!'}")

# ============================================================
# APPROACH 6: Bregman divergence — general convex function Lyapunov
# ============================================================
print(f"\n{'='*70}")
print("APPROACH 6: General Bregman divergence Lyapunov")
print(f"{'='*70}")
print("""
For any convex function φ, the Bregman divergence:
D_φ(M||M*) = φ(M) - φ(M*) - ∇φ(M*)·(M-M*)

KL is a special case with φ(M) = Σ M_k ln M_k + (1-M_k)ln(1-M_k)

The squared Euclidean ||M-M*||² is another (φ = ||·||²/2).

What if we use φ = -Σ ln M_k (Burg entropy)?
Or φ = -Σ ln(1-M_k)?

The key: find φ such that D_φ(N(M)||M*) ≤ θ D_φ(M||M*) with θ < 1.
This is equivalent to N being a Bregman contraction.
""")

# Try squared Euclidean: V(M) = 0.5 * ||M-M*||_2²
# ΔV = 0.5[||N-M*||² - ||M-M*||²]
# = 0.5[(N-M*)·(N-M*) - (M-M*)·(M-M*)]
# = (N-M)·(M-M*) + 0.5||N-M||²
#
# First term: -(N-M)·(M*-M) = -direction_monotonicity < 0
# Second term: 0.5||N-M||² > 0
# Net effect depends on step size.

for seed in [0, 1, 11, 99]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    # Max squared Euclidean decrease ratio
    rng = np.random.RandomState(seed + 9999)
    max_ratio = 0.0
    min_ratio = float('inf')
    
    for _ in range(5000):
        M = rng.uniform(0, 1, 5)
        Delta = M - Mstar
        norm_sq = np.sum(Delta * Delta)
        if norm_sq < 1e-14:
            continue
        
        A = a + w @ M
        B = b + v @ M
        D = A + B + eps
        NM = A / D
        norm_sq_new = np.sum((NM - Mstar) ** 2)
        
        ratio = norm_sq_new / norm_sq
        max_ratio = max(max_ratio, ratio)
        min_ratio = min(min_ratio, ratio)
    
    print(f"  Seed {seed}: ||N-M*||² ratio ∈ [{min_ratio:.4f}, {max_ratio:.4f}]")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("SUMMARY OF APPROACHES")
print(f"{'='*70}")
