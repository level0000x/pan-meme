"""
KL Lyapunov Proof - Version 2 (Corrected)
==========================================
Key correction: V(M) = D_KL(M* || M) (reverse KL), NOT D_KL(M || M*).

The simplified per-component formula IS correct:
    ΔV_k = M*_k ln(M_k/N_k) + (1-M*_k) ln((1-M_k)/(1-N_k))

This is algebraically exact from D_KL(M* || N) - D_KL(M* || M).

Strategy 1: Local KL Lyapunov via H-weighted contraction
Strategy 2: Bregman decomposition analysis
Strategy 3: CE/MM method connection
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

def D_KL_ber(p, q):
    """D_KL(p || q) for Bernoulli"""
    result = 0.0
    for k in range(len(p)):
        if p[k] > 0 and q[k] > 0:
            result += p[k] * np.log(p[k] / q[k])
        if p[k] < 1 and q[k] < 1:
            result += (1 - p[k]) * np.log((1 - p[k]) / (1 - q[k]))
    return result

def logit(x):
    return np.log(x / (1 - x))

print("=" * 70)
print("TEST 0: Verify simplified ΔV_k formula is EXACT")
print("=" * 70)

rs = np.random.RandomState(99)
for i in range(100):
    Mstar_k = rs.uniform(0.01, 0.99)
    M_k = rs.uniform(0.01, 0.99)
    N_k = rs.uniform(0.01, 0.99)
    
    dV_full = D_KL_ber([Mstar_k], [N_k]) - D_KL_ber([Mstar_k], [M_k])
    dV_simple = Mstar_k * np.log(M_k/N_k) + (1-Mstar_k) * np.log((1-M_k)/(1-N_k))
    
    err = abs(dV_full - dV_simple)
    if err > 1e-14:
        print(f"FAIL: err={err}")
        raise SystemExit(1)

print("All 100 tests: ΔV_k = M*_k ln(M_k/N_k) + (1-M*_k) ln((1-M_k)/(1-N_k)) ✓")
print("This formula is algebraically EXACT for V(M) = D_KL(M* || M)\n")

print("=" * 70)
print("STRATEGY 1: Local KL contraction condition")
print("V(M) ≈ (1/2)(M-M*)^T H (M-M*)  where H = diag(1/(M*_k(1-M*_k)))")
print("Near M*: N ≈ M* + J(M*)(M-M*)")
print("V(N)-V(M) ≈ (1/2)(M-M*)^T (J^T H J - H) (M-M*)")
print("Condition: J^T H J ≺ H  ⇔  ||H^{1/2} J H^{-1/2}|| < 1")
print("=" * 70)

max_norm = 0.0
min_norm = float('inf')
n_seeds = 200

for seed in range(n_seeds):
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    H = np.diag(1.0 / (Mstar * (1 - Mstar)))
    H_sqrt = np.diag(np.sqrt(1.0 / (Mstar * (1 - Mstar))))
    H_inv_sqrt = np.diag(np.sqrt(Mstar * (1 - Mstar)))
    
    J = np.zeros((5, 5))
    Dstar = a + w @ Mstar + b + v @ Mstar + eps
    for k in range(5):
        for j in range(5):
            J[k, j] = (w[k, j] * (1 - Mstar[k]) - Mstar[k] * v[k, j]) / Dstar[k]
    
    M = H_sqrt @ J @ H_inv_sqrt
    norm = np.linalg.norm(M, 2)
    
    max_norm = max(max_norm, norm)
    min_norm = min(min_norm, norm)

print(f"||H^{1/2} J H^{-1/2}||_2 over {n_seeds} FCA seeds:")
print(f"  min = {min_norm:.6f}")
print(f"  max = {max_norm:.6f}")
print(f"  condition held: {max_norm < 1.0}")
print()

print("=" * 70)
print("STRATEGY 2: Bregman decomposition analysis")
print("ΔV = -D_KL(N||M) + (M*-N)(logit M - logit N)")
print("")
print("Per-component analysis:")
print("  No overshoot (N_k between M_k and M*_k):")
print("    -D(N_k||M_k) ≤ 0, cross_k = (M*_k-N_k)(logit M_k-logit N_k) < 0")
print("    → ΔV_k ≤ -D(N_k||M_k) < 0  ✓")
print("  Overshoot (N_k crosses M*_k):")
print("    cross_k > 0, competes with -D(N_k||M_k)")
print("=" * 70)

n_points_per_seed = 500
total_positive = 0
total_tests = 0
cross_positive = 0
overshoot_count = 0
total_dV = []

for seed in range(100):
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    rs = np.random.RandomState(seed * 7 + 13)
    for _ in range(n_points_per_seed):
        M = np.clip(Mstar + rs.uniform(-0.5, 0.5, 5), 0.001, 0.999)
        N = N_op(M, a, b, eps, w, v)
        
        if np.max(np.abs(N - M)) < 1e-12:
            continue
        
        dV_total = D_KL_ber(Mstar, N) - D_KL_ber(Mstar, M)
        total_dV.append(dV_total)
        total_tests += 1
        
        if dV_total >= 0:
            total_positive += 1
        
        for k in range(5):
            if np.abs(M[k] - Mstar[k]) < 1e-10:
                continue
            if (M[k] - Mstar[k]) * (N[k] - Mstar[k]) < -1e-10:
                overshoot_count += 1
                cross = (Mstar[k] - N[k]) * (logit(M[k]) - logit(N[k]))
                if cross > 0:
                    cross_positive += 1

dVs = np.array(total_dV)
print(f"Global KL test ({total_tests} points across 100 seeds):")
print(f"  ΔV ≥ 0 violations: {total_positive} / {total_tests} ({100*total_positive/total_tests:.4f}%)")
print(f"  ΔV statistics: mean={dVs.mean():.6f}, max={dVs.max():.6f}, min={dVs.min():.6f}")
print(f"  Overshoot component count: {overshoot_count} ({100*overshoot_count/(total_tests*5):.2f}%)")
print(f"  Overshoot with positive cross: {cross_positive} ({100*cross_positive/max(1,overshoot_count):.2f}%)")
print()

print("=" * 70)
print("STRATEGY 3: Componentwise analysis - when can cross > D_KL?")
print("=" * 70)

max_cross_minus_D = -float('inf')
cross_gt_D_count = 0
total_components = 0

for seed in range(50):
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    rs = np.random.RandomState(seed * 17 + 31)
    for _ in range(200):
        M = np.clip(Mstar + rs.uniform(-0.5, 0.5, 5), 0.001, 0.999)
        N = N_op(M, a, b, eps, w, v)
        
        if np.max(np.abs(N - M)) < 1e-12:
            continue
        
        for k in range(5):
            total_components += 1
            dN = D_KL_ber([N[k]], [M[k]])
            cross_k = (Mstar[k] - N[k]) * (logit(M[k]) - logit(N[k]))
            delta = cross_k - dN
            if delta > max_cross_minus_D:
                max_cross_minus_D = delta
            if delta > 0:
                cross_gt_D_count += 1

print(f"Components tested: {total_components}")
print(f"cross_k > D_KL(N_k||M_k) occurrences: {cross_gt_D_count} ({100*cross_gt_D_count/total_components:.4f}%)")
print(f"max(cross_k - D_KL): {max_cross_minus_D:.6f}")
print()

print("=" * 70)
print("STRATEGY 4: Bound the Bregman decomposition analytically")
print("")
print("Goal: Show (M*-N)(logit M - logit N) ≤ D_KL(N||M) + D_KL(M*||M) - D_KL(M*||N)")
print("     → equivalent to ΔV ≤ 0")
print("")
print("Alternative: Show D_KL(M*||N) ≤ D_KL(M*||M)")
print("     → via KL contraction of N operator")
print("=" * 70)

print("""
Theoretical structure:
-----------------------
Let V(M) = D_KL(M* || M). This is a convex function with unique minimum at M = M*.

For any convex function V, the gradient step M → M - α∇V(M) decreases V.
The natural gradient step using Fisher metric is M → M - (M-M*) = M*, direct convergence.

The N operator is NOT a gradient step, but a CE-type update:
  N_k = A_k/D_k = (a_k + Σ_j w_kj M_j) / (a_k + Σ_j w_kj M_j + ...)

CE method interpretation:
  In the CE method for Bernoulli, given empirical elite distribution g*:
  M_new = argmin_M D_KL(g* || f(·; M))
  
  For Bernoulli: ∂/∂M_k D_KL(g* || f_M) = 0 ⇒ M_k = E_g*[X_k] = g*_k
  
  So the CE update is simply M_new = g* (that's too trivial for the general case).
  
  But CE with smoothed/regularized updates can be written as a convex combination.
  
N operator structure:
  N_k = (α_k·R + ε_1) / (denominator) — similar to an M-estimate with implicit g*.

  The key insight: at the fixed point M*, N(M*) = M* means:
    M*_k = A*_k / D*_k
  or equivalently: M*_k is the "optimal parameter" for the distribution 
  implicit in A*/D*.

For general M ≠ M*:
  The CE update for the target distribution encoded in (a, b, w, v, ε) IS N(M).
  And the CE method guarantees D_KL(· || f_{N(M)}) ≤ D_KL(· || f_M) 
  for the TRUE target distribution g* encoded by the N parameters.

  Since M* is the fixed point, g* = f(·; M*) (the target distribution equals 
  the fixed point distribution), so:
  
  D_KL(f(·; M*) || f(·; N)) ≤ D_KL(f(·; M*) || f(·; M))
  
  For product-of-Bernoulli: D_KL(f(·; p) || f(·; q)) = Σ D_KL(Ber(p_k) || Ber(q_k))
  
  Therefore: Σ D_KL(M*_k || N_k) ≤ Σ D_KL(M*_k || M_k)
  
  Wait — that's D_KL(M* || N) ≤ D_KL(M* || M) which is exactly V(N) ≤ V(M) !!

BUT: the CE method's monotonicity guarantee requires that M_new is obtained by 
EXACTLY minimizing D_KL(g* || f(·; M)). The N operator IS the minimizer for 
the problem where the target g* depends on the current M...

Hmm, this needs more careful analysis.

Actually, the standard CE method proof:
  At iteration t, we have M^(t).
  The update M^(t+1) minimizes D_KL(g*_{t} || f_M) where g*_{t} is the 
  empirical elite distribution at iteration t.
  
  The guarantee is:
    D_KL(g*_{t} || f_{M^(t+1)}) ≤ D_KL(g*_{t} || f_{M^(t)})
  
  But the target g*_{t} CHANGES with t! So this doesn't give monotonic
  decrease of D_KL(g* || f_{M^(t)}) for a FIXED g*.

CONCLUSION: CE method's monotonicity does NOT directly transfer to our case.
The N operator decreases different targets at different M values.

However, at the fixed point M*, the target "stabilizes" and the N operator
locally behaves like a contraction towards M*.

Let me now check the "reverse" formulation: is there a convex function
that the N operator naturally minimizes (like a Bregman proximal point)?

Bregman proximal point: M^(t+1) = argmin_M {φ(M) + D_H(M || M^(t))}
  for some convex φ and Bregman divergence D_H.

If N(M) = argmin_X {f(X) + D_KL(X || M)} for some f, then by the proximal
monotonicity property: D_KL(M* || N(M)) ≤ D_KL(M* || M).

Check: is N(M) a KL-proximal point?
  N(M) = argmin_X {F(X) + D_KL(X || M)}

First-order condition: ∇F(N) + ∇_X D_KL(N || M) = 0

∂/∂X_k D_KL(X || M) = ln(X_k/M_k) - ln((1-X_k)/(1-M_k)) = logit X_k - logit M_k

So: ∇F(N) + logit N - logit M = 0
  ⇒ logit N_k - logit M_k = -∂F/∂N_k

If F(X) = 0 at X = M* (the fixed point), then logit M*_k - logit M_k is the 
natural gradient direction.

The N operator: N_k = A_k/D_k. Can logit N be written as logit M - ∇F(M)?
Not obvious...

Let me abandon the proximal point approach and try something else.

NEW APPROACH: Information Geometry / Natural Gradient
======================================================
The Fisher information matrix for product-of-Bernoulli is diagonal:
  I(M)_kk = 1/(M_k(1-M_k))

The Riemannian distance (in Fisher metric) along a geodesic is NOT the KL
divergence symmetry, but they are related:
  D_KL(M* || M) ≈ (1/2) ||M - M*||²_I near M*

Natural gradient descent: N(M) = M - η · I(M)^{-1} · ∇V(M)
  = M - η · diag(M(1-M)) · (M-M*)/(M(1-M))
  = M - η · (M - M*)
  = (1-η)M + ηM*

This converges in 1 step with η=1, and gives monotonic decrease for η<1.

Our N operator is NOT of this form, but near M*:
  N(M) ≈ M* + J(M*)(M - M*)

If ρ(J(M*)) < 1, local convergence is guaranteed.

For KL monotonicity near M*, we need:
  (N-M*)^T H (N-M*) < (M-M*)^T H (M-M*)

where H = I(M*) = diag(1/(M*_k(1-M*_k))).

Substituting N ≈ M* + J(M*-M):
  (M-M*)^T J^T H J (M-M*) < (M-M*)^T H (M-M*)

i.e., J^T H J ≺ H, equivalent to ||H^{1/2} J H^{-1/2}|| < 1.

THIS is the local condition to verify numerically!
""")

print("=" * 70)
print("SUMMARY / NEXT STEPS")
print("=" * 70)
print("""
1. Local KL contraction: J^T H J ≺ H (condition equivalent to ||H^{1/2} J H^{-1/2}|| < 1)
   - If this holds for ALL FCA seeds, then KL descent is locally proven ■
   - For global: need to handle the convexity of V(M) + some structural property

2. Global via Bregman: ΔV = -D(N||M) + (M*-N)(logit M - logit N)
   - For non-overshoot: each term ≤ -D(N_k||M_k) < 0 → negative
   - For overshoot: need to bound cross_k relative to D_KL
   - Empirically: even with 5-component simultaneous crossing, total ΔV < 0

3. The simplified formula approach:
   ΔV_k = M*_k ln(M_k/N_k) + (1-M*_k) ln((1-M_k)/(1-N_k))
   - This IS correct (verified above)
   - ΔV_k < 0 iff N_k has higher Bernoulli log-likelihood under M*_k than M_k
   - This is the "best" interpretation: N moves M toward higher likelihood under M*

4. Key remaining challenge for global ■:
   - Prove that total log-likelihood gain from non-overshooters dominates
     possible loss from overshooters
   - Or: prove overshoot never happens (false empirically)
   - Or: use a different decomposition that avoids per-component competition
""")
