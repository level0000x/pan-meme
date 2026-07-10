"""
Deep verification of KL divergence as Lyapunov function.

Key formula from Approach 1:
V(N(M)) = Σ_k [N_k ln(N_k/M*_k) + (1-N_k) ln((1-N_k)/(1-M*_k))]

Using FP equation: M*_k = A*_k / D*_k, so:
  N_k/M*_k = (A_k/A*_k)(D*_k/D_k)
  (1-N_k)/(1-M*_k) = ((B_k+eps_k)/(B*_k+eps_k))(D*_k/D_k)

Thus:
  N_k ln(N_k/M*_k) + (1-N_k) ln((1-N_k)/(1-M*_k))
  = N_k ln(A_k/A*_k) + (1-N_k) ln((B_k+eps_k)/(B*_k+eps_k)) + ln(D*_k/D_k)

KL difference:
  ΔV_k = [N_k ln(A_k/A*_k) + (1-N_k) ln((B_k+eps_k)/(B*_k+eps_k)) + ln(D*_k/D_k)]
       - [M_k ln(M_k/M*_k) + (1-M_k) ln((1-M_k)/(1-M*_k))]

This is EXACT (no approximation). The question is whether it's always negative.
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

def kl_div(M, Mstar):
    val = 0.0
    for k in range(5):
        if M[k] <= 0 or M[k] >= 1:
            continue
        val += M[k] * np.log(M[k] / Mstar[k])
        val += (1 - M[k]) * np.log((1 - M[k]) / (1 - Mstar[k]))
    return val

# ============================================================
# TEST 1: 200 seeds, random + vertex, KL always decreases?
# ============================================================
print("=" * 70)
print("TEST 1: Universal KL decrease across 200 seeds")
print("=" * 70)

kl_violations = 0
kl_ratios_all = []
kl_diffs_all = []

for seed in range(200):
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    # Vertex test
    for bits in range(32):
        M = np.array([0.001 if (bits >> j) & 1 == 0 else 0.999 for j in range(5)])
        A = a + w @ M
        B = b + v @ M
        D = A + B + eps
        NM = A / D
        
        kl_b = kl_div(M, Mstar)
        kl_a = kl_div(NM, Mstar)
        diff = kl_a - kl_b
        if diff > 1e-12:
            kl_violations += 1
            if kl_violations <= 3:
                print(f"  VIOLATION! Seed {seed}, vertex {bits}: diff={diff:.10f}")
        kl_diffs_all.append(diff)
        if kl_b > 1e-14:
            kl_ratios_all.append(kl_a / kl_b)
    
    # Random interior test
    rng = np.random.RandomState(seed + 7777)
    for _ in range(500):
        M = rng.uniform(0.001, 0.999, 5)
        A = a + w @ M
        B = b + v @ M
        D = A + B + eps
        NM = A / D
        
        kl_b = kl_div(M, Mstar)
        kl_a = kl_div(NM, Mstar)
        diff = kl_a - kl_b
        if diff > 1e-12:
            kl_violations += 1
        kl_diffs_all.append(diff)
        if kl_b > 1e-14:
            kl_ratios_all.append(kl_a / kl_b)

kl_ratios_all = np.array(kl_ratios_all)
kl_diffs_all = np.array(kl_diffs_all)

print(f"\n  Total KL violations: {kl_violations}")
print(f"  Total samples: {len(kl_diffs_all)}")
print(f"  KL ratio max: {kl_ratios_all.max():.4f}")
print(f"  KL ratio mean: {kl_ratios_all.mean():.4f}")
print(f"  KL diff max: {kl_diffs_all.max():.12f}")
print(f"  KL diff min: {kl_diffs_all.min():.6f}")
print(f"  KL diff mean: {kl_diffs_all.mean():.6f}")

# ============================================================
# TEST 2: KL contraction is much stronger than l1
# ============================================================
print(f"\n{'='*70}")
print("TEST 2: KL vs l1 contraction comparison")
print(f"{'='*70}")

for seed in [0, 1, 11, 42, 99]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    # Track both KL and l1 ratios
    kl_ratios = []
    l1_ratios = []
    
    rng = np.random.RandomState(seed + 9999)
    for _ in range(2000):
        M = rng.uniform(0.001, 0.999, 5)
        Delta = M - Mstar
        l1_b = np.sum(np.abs(Delta))
        kl_b = kl_div(M, Mstar)
        if l1_b < 1e-14 or kl_b < 1e-14:
            continue
        
        A = a + w @ M
        B = b + v @ M
        D = A + B + eps
        NM = A / D
        
        l1_a = np.sum(np.abs(NM - Mstar))
        kl_a = kl_div(NM, Mstar)
        
        kl_ratios.append(kl_a / kl_b)
        l1_ratios.append(l1_a / l1_b)
    
    kl_ratios = np.array(kl_ratios)
    l1_ratios = np.array(l1_ratios)
    
    print(f"  Seed {seed}:")
    print(f"    KL:  max={kl_ratios.max():.4f}, mean={kl_ratios.mean():.4f}")
    print(f"    l1:  max={l1_ratios.max():.4f}, mean={l1_ratios.mean():.4f}")
    print(f"    KL/l1 ratio: {(kl_ratios.mean()/l1_ratios.mean()):.4f}")

# ============================================================
# TEST 3: Analytic structure of KL difference
# ============================================================
print(f"\n{'='*70}")
print("TEST 3: Analytic decomposition of KL difference")
print(f"{'='*70}")
print("""
Using the FP equation M*_k D*_k = A*_k:
  N_k/M*_k = (A_k/A*_k)(D*_k/D_k)
  (1-N_k)/(1-M*_k) = ((B_k+eps_k)/(B*_k+eps_k))(D*_k/D_k)

Per-component KL difference:
  ΔV_k = N_k ln(A_k/A*_k) + (1-N_k) ln((B_k+eps_k)/(B*_k+eps_k))
       + ln(D*_k/D_k) - M_k ln(M_k/M*_k) - (1-M_k) ln((1-M_k)/(1-M*_k))

Now M_k/M*_k = (1/M*_k) M_k ... this doesn't simplify as nicely.

Alternative: use the full expression directly.
N_k = A_k/D_k, M_k is variable.

KL_before = M_k ln M_k + (1-M_k) ln(1-M_k) - M_k ln M*_k - (1-M_k) ln(1-M*_k)
KL_after  = N_k ln N_k + (1-N_k) ln(1-N_k) - N_k ln M*_k - (1-N_k) ln(1-M*_k)

Difference:
  = [N_k ln N_k + (1-N_k) ln(1-N_k) - M_k ln M_k - (1-M_k) ln(1-M_k)]
  - [(N_k - M_k) ln M*_k + ((1-N_k) - (1-M_k)) ln(1-M*_k)]
  = [H(N_k) - H(M_k)] - (N_k-M_k)[ln M*_k - ln(1-M*_k)]
  = [H(N_k) - H(M_k)] - (N_k-M_k)logit(M*_k)

where H(p) = p ln p + (1-p) ln(1-p) is the binary entropy function (negated).

So ΔV = Σ_k { [H(N_k) - H(M_k)] - (N_k-M_k) · logit(M*_k) }

This is a per-component decomposition! No cross terms!
""")

# Verify this decomposition numerically
for seed in [0, 11]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    M = np.array([0.3, 0.7, 0.2, 0.8, 0.5])
    A = a + w @ M
    B = b + v @ M
    D = A + B + eps
    NM = A / D
    
    kl_b = kl_div(M, Mstar)
    kl_a = kl_div(NM, Mstar)
    delta_kl = kl_a - kl_b
    
    # Per-component formula
    delta_comp = 0.0
    for k in range(5):
        H_Nk = NM[k] * np.log(NM[k]) + (1-NM[k]) * np.log(1-NM[k])
        H_Mk = M[k] * np.log(M[k]) + (1-M[k]) * np.log(1-M[k])
        logit = np.log(Mstar[k] / (1-Mstar[k]))
        delta_comp += H_Nk - H_Mk - (NM[k] - M[k]) * logit
    
    print(f"  Seed {seed}: ΔV = {delta_kl:.10f}, per-comp formula = {delta_comp:.10f}, match={np.isclose(delta_kl, delta_comp)}")

# ============================================================
# TEST 4: Try to analytically bound KL contraction
# ============================================================
print(f"\n{'='*70}")
print("TEST 4: Can we analytically bound the per-component KL difference?")
print(f"{'='*70}")
print("""
Per component:
  ΔV_k = H(N_k) - H(M_k) - (N_k-M_k) logit(M*_k)

where H(p) = p ln p + (1-p) ln(1-p).

The second derivative: H''(p) = 1/p + 1/(1-p) = 1/(p(1-p))

By Taylor's theorem:
  H(N_k) = H(M_k) + H'(M_k)(N_k-M_k) + (1/2) H''(ξ_k) (N_k-M_k)²

where ξ_k is between M_k and N_k.

Since H'(p) = ln(p/(1-p)) = logit(p):
  H(N_k) - H(M_k) = logit(M_k)(N_k-M_k) + (1/2) (1/(ξ_k(1-ξ_k))) (N_k-M_k)²

Therefore:
  ΔV_k = [logit(M_k) - logit(M*_k)](N_k-M_k) + (1/2)(1/(ξ_k(1-ξ_k)))(N_k-M_k)²

The first term is: (logit(M_k) - logit(M*_k))(N_k - M_k)

KEY CONNECTION: logit(M_k) - logit(M*_k) and N_k - M_k have OPPOSITE signs! 
Because N_k sits between M_k and M*_k (from direction monotonicity). 

Wait - does N_k sit between M_k and M*_k componentwise? Direction monotonicity
gives (N-M)·(M*-M) > 0, not componentwise ordering.

But check numerically: is N_k between M_k and M*_k for each k?
""")

for seed in [0, 1, 11]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    violations = 0
    total = 0
    rng = np.random.RandomState(seed + 12345)
    
    for _ in range(5000):
        M = rng.uniform(0.001, 0.999, 5)
        A = a + w @ M
        B = b + v @ M
        D = A + B + eps
        NM = A / D
        
        for k in range(5):
            total += 1
            if M[k] > Mstar[k] and NM[k] > M[k]:
                violations += 1
            elif M[k] < Mstar[k] and NM[k] < M[k]:
                violations += 1
    
    print(f"  Seed {seed}: N_k outside [M_k, M*_k] in {violations}/{total} cases ({100*violations/total:.2f}%)")

# ============================================================
# TEST 5: THE KEY INSIGHT — componentwise monotonicity of N_k
# ============================================================
print(f"\n{'='*70}")
print("TEST 5: Componentwise analysis of N_k")
print(f"{'='*70}")
print("""
N_k = A_k/D_k = (a_k + w_k·M) / (a_k + b_k + e_k + (w+v)_k·M)

For fixed k, as a function of M_j:
  ∂N_k/∂M_j = [w_kj(1-N_k) - N_k v_kj] / D_k

At M = M*: ∂N_k/∂M_j|_{M*} = J_kj(M*) with the familiar sign structure.

But at general M, the sign of ∂N_k/∂M_j depends on whether N_k is above 
or below the threshold w_kj/(w_kj+v_kj).

The key: if we can prove N_k always moves TOWARD M*_k (componentwise),
then logit(M_k)-logit(M*_k) and N_k-M_k always have opposite signs,
giving ΔV_k < 0 strictly (via the convexity of H and the second term).

This is a PER-COMPONENT monotonicity property, stronger than the vector
direction monotonicity we've been studying.
""")

# Check: which components violate componentwise monotonicity?
for seed in [0, 1, 11]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    component_violations = {k: 0 for k in range(5)}
    total_per_comp = {k: 0 for k in range(5)}
    
    rng = np.random.RandomState(seed + 55555)
    for _ in range(5000):
        M = rng.uniform(0.001, 0.999, 5)
        A = a + w @ M
        B = b + v @ M
        D = A + B + eps
        NM = A / D
        
        for k in range(5):
            total_per_comp[k] += 1
            if M[k] < Mstar[k] and NM[k] < M[k]:
                component_violations[k] += 1
            elif M[k] > Mstar[k] and NM[k] > M[k]:
                component_violations[k] += 1
    
    print(f"\n  Seed {seed}:")
    for k in range(5):
        pct = 100 * component_violations[k] / total_per_comp[k] if total_per_comp[k] > 0 else 0
        print(f"    Component {k}: overshoots in {component_violations[k]}/{total_per_comp[k]} ({pct:.1f}%)")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("DISCUSSION OF NEW AVENUES")
print(f"{'='*70}")
