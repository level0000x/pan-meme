"""
The correct simplified formula for per-component KL difference:

ΔV_k = M*_k ln(M_k/N_k) + (1-M*_k) ln((1-M_k)/(1-N_k))

This is algebraically EXACT, derived from D_KL(M* || N) - D_KL(M* || M).

No cross terms, no approximations. Per-component factorization.

ΔV_k < 0 ⇔ (M_k/N_k)^{M*_k} · ((1-M_k)/(1-N_k))^{1-M*_k} < 1
         ⇔ M_k^{M*_k} (1-M_k)^{1-M*_k} < N_k^{M*_k} (1-N_k)^{1-M*_k}

So N_k has higher Bernoulli likelihood under parameter M*_k than M_k.
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
# TEST 1: Verify the simplified formula
# ============================================================
print("=" * 70)
print("TEST 1: Verify simplified per-component formula")
print("=" * 70)
print("ΔV_k = M*_k·ln(M_k/N_k) + (1-M*_k)·ln((1-M_k)/(1-N_k))")

for seed in [0, 1, 11, 42, 99]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    rng = np.random.RandomState(seed + 123456)
    max_err = 0.0
    
    for _ in range(5000):
        M = rng.uniform(0.001, 0.999, 5)
        A = a + w @ M
        B = b + v @ M
        D = A + B + eps
        NM = A / D
        
        for k in range(5):
            # Full formula
            dV_full = NM[k] * np.log(NM[k]/Mstar[k]) + (1-NM[k]) * np.log((1-NM[k])/(1-Mstar[k])) \
                    - M[k] * np.log(M[k]/Mstar[k]) - (1-M[k]) * np.log((1-M[k])/(1-Mstar[k]))
            
            # Simplified
            dV_simple = Mstar[k] * np.log(M[k]/NM[k]) + (1-Mstar[k]) * np.log((1-M[k])/(1-NM[k]))
            
            err = abs(dV_full - dV_simple)
            max_err = max(max_err, err)
    
    print(f"  Seed {seed}: max error = {max_err:.2e} ✓" if max_err < 1e-12 else f"  FAIL: {max_err:.2e}")

# ============================================================
# TEST 2: Check per-component negativity
# ============================================================
print(f"\n{'='*70}")
print("TEST 2: Is each component always negative?")
print(f"{'='*70}")

for seed in range(200):
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    per_comp_positive = 0
    total = 0
    
    rng = np.random.RandomState(seed + 99999)
    for _ in range(1000):
        M = rng.uniform(0.001, 0.999, 5)
        A = a + w @ M
        B = b + v @ M
        D = A + B + eps
        NM = A / D
        
        for k in range(5):
            total += 1
            dV_k = Mstar[k] * np.log(M[k]/NM[k]) + (1-Mstar[k]) * np.log((1-M[k])/(1-NM[k]))
            if dV_k > 1e-14:
                per_comp_positive += 1
    
    if seed < 5 or seed == 11:
        print(f"  Seed {seed}: per-comp positive in {per_comp_positive}/{total} ({100*per_comp_positive/total:.1f}%)")

# Full scan
total_positive = 0
total_all = 0
for seed in range(200):
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    rng = np.random.RandomState(seed + 55555)
    for _ in range(500):
        M = rng.uniform(0.001, 0.999, 5)
        A = a + w @ M
        B = b + v @ M
        D = A + B + eps
        NM = A / D
        for k in range(5):
            total_all += 1
            dV_k = Mstar[k] * np.log(M[k]/NM[k]) + (1-Mstar[k]) * np.log((1-M[k])/(1-NM[k]))
            if dV_k > 1e-14:
                total_positive += 1

print(f"\n  TOTAL: per-comp positive in {total_positive}/{total_all} ({100*total_positive/total_all:.1f}%)")

# ============================================================
# TEST 3: Total ΔV always negative?
# ============================================================
print(f"\n{'='*70}")
print("TEST 3: Total ΔV always negative?")
print(f"{'='*70}")

max_total_dV = -float('inf')
for seed in range(200):
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    rng = np.random.RandomState(seed + 77777)
    for _ in range(500):
        M = rng.uniform(0.001, 0.999, 5)
        A = a + w @ M
        B = b + v @ M
        D = A + B + eps
        NM = A / D
        dV = sum(Mstar[k] * np.log(M[k]/NM[k]) + (1-Mstar[k]) * np.log((1-M[k])/(1-NM[k])) for k in range(5))
        max_total_dV = max(max_total_dV, dV)

print(f"  Max total ΔV across 200 seeds × 500 points: {max_total_dV:.10f}")
print(f"  Always negative? {max_total_dV < 0}")

# ============================================================
# TEST 4: Analyze positivity mechanism - why does overshoot not break KL?
# ============================================================
print(f"\n{'='*70}")
print("TEST 4: Overshoot component analysis")
print(f"{'='*70}")
print("""
Per-component formula: ΔV_k = M* ln(M/N) + (1-M*) ln((1-M)/(1-N))

When N overshoots past M* (from M):
  There are 2 subcases:
  
  A) N < M < M*: both M and N are below M*
     M*/N > M*/M, N > 0 → first term M* ln(M/N) < 0 (small or moderate)
     (1-M)/(1-N) < 1 → second term (1-M*) ln((1-M)/(1-N)) < 0
     Both terms negative → ΔV_k < 0 ✓
  
  B) M < M* < N: M below, N above M*
     M*/N < 1 → M* ln(M/N) = M* ln(M) - M* ln(N)
         = M* ln(M) - M* ln(N) 
         Since M < M* < N, ln(M/N) < 0, so first term < 0.
     (1-M)/(1-N) > 1 → (1-M*) ln((1-M)/(1-N)) > 0
     Competition!
     
     But (1-M) is close to 1 and (1-N) < 1-M*, so (1-M)/(1-N) > 1.
     The M* ln(M/N) term must dominate.
     
  C) M > M* > N: symmetric to B.
  
  D) M > N > M*: all above M*, symmetric to A.
  
So only cases B and C have competition. In those cases, N crosses M*.
If |M-M*| is large (M far from M*), then |M-N| is also large (since N crossed M*),
making M* ln(M/N) very negative and (1-M*) ln((1-M)/(1-N)) positive but bounded.
""")

# Find worst cases of subcase B
for seed in [0, 1, 11]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    worst_B_dV = float('inf')
    worst_B_case = None
    
    rng = np.random.RandomState(seed + 88888)
    for _ in range(50000):
        M = rng.uniform(0.001, 0.999, 5)
        A = a + w @ M
        B = b + v @ M
        D = A + B + eps
        NM = A / D
        
        for k in range(5):
            if M[k] < Mstar[k] < NM[k]:  # case B
                dV_k = Mstar[k] * np.log(M[k]/NM[k]) + (1-Mstar[k]) * np.log((1-M[k])/(1-NM[k]))
                if dV_k > 1e-14 and dV_k > worst_B_dV:
                    worst_B_dV = dV_k
                    worst_B_case = (M[k], NM[k], Mstar[k], k)
            elif M[k] > Mstar[k] > NM[k]:  # case C
                dV_k = Mstar[k] * np.log(M[k]/NM[k]) + (1-Mstar[k]) * np.log((1-M[k])/(1-NM[k]))
                if dV_k > 1e-14 and dV_k > worst_B_dV:
                    worst_B_dV = dV_k
                    worst_B_case = (M[k], NM[k], Mstar[k], k)
    
    if worst_B_case:
        Mk, Nk, msk, k = worst_B_case
        term1 = msk * np.log(Mk/Nk)
        term2 = (1-msk) * np.log((1-Mk)/(1-Nk))
        crossing_type = "B (M<M*<N)" if Mk < msk < Nk else "C (M>M*>N)"
        print(f"\n  Seed {seed}: worst crossing case ({crossing_type})")
        print(f"    k={k}, M_k={Mk:.4f}, N_k={Nk:.4f}, M*_k={msk:.4f}")
        print(f"    Term1 (M*log(M/N)) = {term1:.6f}")
        print(f"    Term2 ((1-M*)log((1-M)/(1-N))) = {term2:.6f}")
        print(f"    ΔV_k = {term1+term2:.10f}")

# ============================================================
# TEST 5: The "crossing cancellation" conjecture
# ============================================================
print(f"\n{'='*70}")
print("TEST 5: The crossing cancellation conjecture")
print(f"{'='*70}")
print("""
When component k crosses M* (case B or C), its ΔV_k can be positive.
But at the SAME M, other components might NOT be crossing, so they
contribute negative ΔV.

Question: do the crossing components correlate with each other?
If all 5 components cross simultaneously, total ΔV could be positive.
""")

for seed in [0, 1, 11]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    # Count simultaneous crossings
    rng = np.random.RandomState(seed + 54321)
    cross_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    total_samples = 0
    max_total_dV_when_crossing = -float('inf')
    
    for _ in range(20000):
        M = rng.uniform(0.001, 0.999, 5)
        A = a + w @ M
        B = b + v @ M
        D = A + B + eps
        NM = A / D
        
        n_cross = 0
        for k in range(5):
            if (M[k] < Mstar[k] < NM[k]) or (M[k] > Mstar[k] > NM[k]):
                n_cross += 1
        
        total_samples += 1
        if n_cross > 0:
            cross_counts[n_cross] = cross_counts.get(n_cross, 0) + 1
            dV_total = sum(Mstar[k] * np.log(M[k]/NM[k]) + (1-Mstar[k]) * np.log((1-M[k])/(1-NM[k])) for k in range(5))
            max_total_dV_when_crossing = max(max_total_dV_when_crossing, dV_total)
    
    print(f"\n  Seed {seed}:")
    total_cross = sum(cross_counts.values())
    for n in range(1, 6):
        print(f"    {n} simultaneous crossings: {cross_counts[n]}/{total_cross} ({100*cross_counts[n]/total_cross:.1f}%)")
    print(f"    Max total ΔV when crossing: {max_total_dV_when_crossing:.10f}")
