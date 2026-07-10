"""
Continue proving — explore 4 new approaches to close remaining ◆ gaps.

APPROACH 1: Exact expression for direction monotonicity
  (N-M)·(M*-M) = ||Δ||² - Σ_k (JΔ)_k Δ_k / (1+r_k)
  with r_k = Σ_j (w+v)_kj Δ_j / D*_k
  Check if the cubic bound analysis (with quartic remainder) actually
  proves global positivity for all seeds.

APPROACH 2: Thompson / Hilbert projective metric contraction
  d_T(x,y) = log max_{i,j} (x_i/y_i)(y_j/x_j)
  Show N is a contraction in this metric → global convergence.

APPROACH 3: Componentwise monotonicity + comparison principle
  Use the monotone structure of the rational map to construct sub/supersolutions.

APPROACH 4: Fixed-point equation constraint on J(M*)
  M*_k = A*_k/D*_k imposes algebraic relations on a,b,w,v,ε.
  Can these relations force sym(I-J) ≻ 0 universally?
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
# APPROACH 1: Exact direction monotonicity with quartic remainder
# ============================================================
print("=" * 70)
print("APPROACH 1: Exact expression + quartic remainder bound")
print("=" * 70)

# Exact: (N-M)·(M*-M) = ||Δ||² - Σ_k (JΔ)_k Δ_k / (1+r_k)
# After linear expansion of 1/(1+r): = ||Δ||² - Σ_k (JΔ)_k Δ_k · (1-r_k+r_k²/(1+r_k))
# = Δ^T(I-J)Δ + Σ_k r_k(JΔ)_kΔ_k - Σ_k r_k²(JΔ)_kΔ_k/(1+r_k)
#
# The quartic remainder R_4 = Σ_k r_k²(JΔ)_kΔ_k/(1+r_k)
#
# Note: (1+r_k) = D_k/D*_k, so 1/(1+r_k) = D*_k/D_k
# Thus the quartic remainder = Σ_k r_k² (JΔ)_k Δ_k · (D*_k/D_k)

# For the cubic analysis we bounded:
# (N-M)·(M*-M) ≥ λ_min·||Δ||² - c_max·||Δ||³
# where λ_min = min eig of sym(I-J), c_max = max |Σ_k r_k(Jd)_k d_k| over unit vectors
#
# Adding quartic: ≥ λ_min·||Δ||² - c_max·||Δ||³ - q_max·||Δ||⁴
# where q_max bounds |Σ_k r_k² (Jd)_k d_k · (D*_k/D_k)| over unit vectors
#
# Since D*_k/D_k ≤ D*_k/D_min_k (worst case), we can bound:
# q_max ≤ max_d Σ_k (r_k(d))² · |(Jd)_k| · |d_k| · (D*_k/D_min_k)

for seed in [0, 1, 11, 42, 99]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    A_s = a + w @ Mstar
    B_s = b + v @ Mstar
    D_s = A_s + B_s + eps
    
    Js = np.zeros((5, 5))
    for k in range(5):
        for j in range(5):
            Js[k, j] = (w[k, j] * (B_s[k] + eps[k]) - A_s[k] * v[k, j]) / (D_s[k] ** 2)
    
    sym_IJ = np.eye(5) - (Js + Js.T) / 2
    lam_min = np.linalg.eigvalsh(sym_IJ).min()
    
    D_min = a + b + eps
    wv = w + v
    
    # Compute c_max and q_max bounds via random sampling
    c_max_est = 0.0
    q_max_est = 0.0
    
    for _ in range(10000):
        d = np.random.randn(5)
        d = d / np.linalg.norm(d)
        
        Jd = Js @ d
        r = d @ wv.T / D_s  # r_k = Σ_j wv_kj d_j/D*_k
        
        cubic = 0.0
        quartic = 0.0
        for k in range(5):
            cubic += r[k] * Jd[k] * d[k]
            quartic += r[k]**2 * abs(Jd[k]) * abs(d[k]) * (D_s[k] / D_min[k])
        
        c_max_est = max(c_max_est, abs(cubic))
        q_max_est = max(q_max_est, quartic)
    
    # Solve for when λ_min·t² - c_max·t³ - q_max·t⁴ = 0
    # = t² (λ_min - c_max·t - q_max·t²)
    # Need: λ_min - c_max·t - q_max·t² > 0
    # Quadratic: q_max·t² + c_max·t - λ_min < 0
    # t = (-c_max + sqrt(c_max² + 4·q_max·λ_min)) / (2·q_max)
    
    disc = c_max_est**2 + 4 * q_max_est * lam_min
    if q_max_est > 1e-15:
        safe_t = (-c_max_est + np.sqrt(disc)) / (2 * q_max_est)
    else:
        safe_t = lam_min / max(c_max_est, 1e-15)
    
    print(f"\n  Seed {seed}: λ_min={lam_min:.4f}, c_max={c_max_est:.4f}, q_max={q_max_est:.4f}")
    print(f"    Safe radius (cubic only): {lam_min/max(c_max_est,1e-15):.2f}")
    print(f"    Safe radius (cubic+quartic): {safe_t:.2f}")
    print(f"    Cube diameter √5 = 2.24: {'COVERS ✓' if safe_t > 2.24 else 'SHORT ✗'}")

# ============================================================
# APPROACH 2: Thompson metric contraction
# ============================================================
print(f"\n{'='*70}")
print("APPROACH 2: Thompson metric contraction")
print(f"{'='*70}")
print("""
Thompson (part metric): d_T(x,y) = log max_{i,j} (x_i/y_j) / (x_j/y_i) = log(max_i x_i/y_i) + log(max_i y_i/x_i)

For N_k = A_k/(A_k+B_k+ε_k), can we bound d_T(N(x), N(y)) ≤ θ d_T(x,y) with θ < 1?

The key property: N maps the positive cone to itself and is homogeneous of degree 0
in (A,B) jointly. This suggests Thompson contraction under certain conditions.

d_T(N(x),N(y)) = log max_{i,j} [A_i(x)/D_i(x) / (A_j(y)/D_j(y))] [A_j(y)/D_j(y) / (A_i(x)/D_i(x))]
= log max_{i,j} [A_i(x)D_j(y) / A_j(y)D_i(x)] [A_j(y)D_i(x) / A_i(x)D_j(y)]
= (max-log) + (max-log) of the same ratios
""")

# Compute Thompson distance for N
for seed in [0, 1, 11, 42, 99]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    # Test many pairs (M, Mstar) and compute Thompson contraction ratio
    max_ratio = 0.0
    rng = np.random.RandomState(seed + 7777)
    
    for _ in range(2000):
        M = rng.uniform(0.001, 1.0, 5)  # avoid 0 for log
        M = np.clip(M, 0.001, 1.0)
        
        A1 = a + w @ M
        B1 = b + v @ M
        D1 = A1 + B1 + eps
        NM = A1 / D1
        
        A2 = a + w @ Mstar
        B2 = b + v @ Mstar
        D2 = A2 + B2 + eps
        NMstar = A2 / D2
        
        # Thompson distance
        # d_T(x,y) = log(max_i max(x_i/y_i, y_i/x_i))
        # Actually: d_T(x,y) = log(max_i x_i/y_i · max_i y_i/x_i)
        ratio1 = max(NM[i] / NMstar[i] for i in range(5))
        ratio2 = max(NMstar[i] / NM[i] for i in range(5))
        d_after = np.log(ratio1 * ratio2)
        
        ratio_bef1 = max(M[i] / Mstar[i] for i in range(5))
        ratio_bef2 = max(Mstar[i] / M[i] for i in range(5))
        d_before = np.log(ratio_bef1 * ratio_bef2)
        
        if d_before > 1e-14:
            ratio = d_after / d_before
            max_ratio = max(max_ratio, ratio)
    
    print(f"  Seed {seed}: max Thompson ratio = {max_ratio:.4f} {'✓' if max_ratio < 1 else '✗'}")

# ============================================================
# APPROACH 3: l1 contraction with M-dependent D bound
# ============================================================
print(f"\n{'='*70}")
print("APPROACH 3: l1 contraction with M-dependent D bound")
print(f"{'='*70}")
print("""
Key idea: D*_k/D_k depends on M. Instead of using the worst case D*_k/D_min_k,
we can bound it using the fact that the worst contraction ratio
occurs at some specific M where D is small AND |JΔ| is large.

Let's search for the actual worst-case l1 contraction ratio analytically.
""")

for seed in [0, 1, 11, 42, 99]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    A_s = a + w @ Mstar
    B_s = b + v @ Mstar
    D_s = A_s + B_s + eps
    
    Js = np.zeros((5, 5))
    for k in range(5):
        for j in range(5):
            Js[k, j] = (w[k, j] * (B_s[k] + eps[k]) - A_s[k] * v[k, j]) / (D_s[k] ** 2)
    
    # Exhaustive vertex search for worst l1 ratio
    max_ratio = 0.0
    worst_M = None
    worst_Delta = None
    
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
        l1_aft = np.sum(np.abs(NM - Mstar))
        ratio = l1_aft / l1_bef
        
        if ratio > max_ratio:
            max_ratio = ratio
            worst_M = M.copy()
            worst_Delta = Delta.copy()
    
    print(f"\n  Seed {seed}: worst vertex l1 ratio = {max_ratio:.4f}")
    print(f"    Worst M = {worst_M}")
    print(f"    Worst Δ = {worst_Delta}")
    
    # What are D and J at this vertex?
    A_w = a + w @ worst_M
    B_w = b + v @ worst_M
    D_w = A_w + B_w + eps
    print(f"    D = {D_w}")
    print(f"    D*/D = {D_s / D_w}")
    print(f"    |Δ|₁ = {np.sum(np.abs(worst_Delta)):.4f}")
    
    # Is this at a simplex vertex or interior?
    print(f"    M components: {worst_M}")

# Check: do all 32 vertices produce l1 ratio < 1?
print(f"\n{'='*70}")
print("Check: l1 contraction ratio at ALL 32 vertices for ALL 200 seeds")
print(f"{'='*70}")

all_vertex_ratios = []
for seed in range(200):
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    max_vertex_ratio = 0.0
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
        max_vertex_ratio = max(max_vertex_ratio, ratio)
    
    all_vertex_ratios.append(max_vertex_ratio)

all_vertex_ratios = np.array(all_vertex_ratios)
print(f"  Max vertex l1 ratio: {all_vertex_ratios.max():.4f}")
print(f"  Mean: {all_vertex_ratios.mean():.4f}")
print(f"  All < 1? {all(all_vertex_ratios < 1)}")

# ============================================================
# APPROACH 4: Global l1 contraction via linear interpolation
# ============================================================
print(f"\n{'='*70}")
print("APPROACH 4: l1 contraction via decomposition into linear pieces")
print(f"{'='*70}")
print("""
If we can show N is l1-contractive at ALL vertices AND
the l1 contraction ratio is convex (or at least bounded by vertex values),
then N is globally l1-contractive.

The problem: N is nonlinear, so convexity of the ratio doesn't directly follow.

But: ||N(M)-M*||_1 = Σ_k |N_k(M)-M*_k|
N_k(M) is a COMPONENTWISE convex or concave function of M? Let's check.
""")

for seed in [0, 1, 11]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    # Check componentwise behavior of N_k along each axis
    for k in range(5):
        for j in range(5):
            # Vary M_j, fix others at M*
            Ms = []
            vals = []
            for t in np.linspace(0, 1, 20):
                M = Mstar.copy()
                M[j] = t
                A = a + w @ M
                B = b + v @ M
                D = A + B + eps
                vals.append(A[k] / D[k])
                Ms.append(t)
            
            # Check convexity
            diffs = np.diff(np.diff(vals))
            is_convex = all(d >= -1e-10 for d in diffs) or all(d <= 1e-10 for d in diffs)
            if seed == 0:
                print(f"  N_{k}(M) varying M_{j}: {'convex/concave ✓' if is_convex else 'NOT convex/concave ✗'}")

# ============================================================
# APPROACH 5: ANALYTIC proof of sym(I-J) ≻ 0 from FP equation
# ============================================================
print(f"\n{'='*70}")
print("APPROACH 5: Analytic proof sym(I-J) ≻ 0 from fixed-point equation")
print(f"{'='*70}")
print("""
Fixed-point equation: M*_k = A*_k/D*_k
where A*_k = a_k + (wM*)_k, D*_k = A*_k + B*_k + ε_k
B*_k = b_k + (vM*)_k

J_kj(M*) = [w_kj(B*_k+ε_k) - A*_k v_kj] / D*_k²
        = w_kj(1-M*_k)/D*_k - M*_k v_kj/D*_k

The second equality: B*_k+ε_k = D*_k - A*_k = D*_k - D*_k·M*_k = D*_k(1-M*_k)
And A*_k = D*_k·M*_k

So J_kj = [w_kj · D*_k(1-M*_k) - D*_k·M*_k · v_kj] / D*_k²
        = [w_kj(1-M*_k) - M*_k v_kj] / D*_k

This is much simpler! J_kj separates into a weight term and a decay term.

(I-J)_kk = 1 (since w_kk = v_kk = 0)
(I-J)_kj = -J_kj = [M*_k v_kj - w_kj(1-M*_k)] / D*_k for k ≠ j

For sym(I-J) to be positive definite, we need:
x^T sym(I-J) x > 0 for all x ≠ 0

sym(I-J)_kj = - (J_kj + J_jk)/2 for k ≠ j
            = - [M*_k v_kj - w_kj(1-M*_k)]/(2D*_k) - [M*_j v_jk - w_jk(1-M*_j)]/(2D*_j)

This seems messy. Let me try a different approach.

Instead of Gershgorin, let's use the Sylvester criterion or check
if I-J is a diagonally dominant M-matrix.
""")

# Check diagonal dominance
print("Checking row diagonal dominance of I-J:")
row_dd_fails = 0
col_dd_fails = 0

for seed in range(200):
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    D_s = a + w @ Mstar + b + v @ Mstar + eps
    Js = np.zeros((5, 5))
    for k in range(5):
        for j in range(5):
            Js[k, j] = (w[k, j] * (1 - Mstar[k]) - Mstar[k] * v[k, j]) / D_s[k]
    
    IJ = np.eye(5) - Js
    
    # Row diagonal dominance
    row_ok = True
    for k in range(5):
        off_sum = sum(abs(IJ[k, j]) for j in range(5) if j != k)
        if off_sum >= abs(IJ[k, k]):
            row_ok = False
    
    # Column diagonal dominance
    col_ok = True
    for j in range(5):
        off_sum = sum(abs(IJ[k, j]) for k in range(5) if k != j)
        if off_sum >= abs(IJ[j, j]):
            col_ok = False
    
    if not row_ok:
        row_dd_fails += 1
    if not col_ok:
        col_dd_fails += 1
    
    if seed < 5:
        print(f"  Seed {seed}: row DD={row_ok}, col DD={col_ok}")

print(f"\n  Row diagonal dominance failures: {row_dd_fails}/200")
print(f"  Col diagonal dominance failures: {col_dd_fails}/200")

# ============================================================
# APPROACH 6: LOG-COORDINATE CONTRACTION
# ============================================================
print(f"\n{'='*70}")
print("APPROACH 6: Log-coordinate contraction (y = log M)")
print(f"{'='*70}")
print("""
N_k = A_k/D_k = a_k + w_kj M_j / (a_k + w_kj M_j + b_k + v_kj M_j + ε_k)
In log coordinates y_k = log M_k, the dynamics might be simpler.

But M ∈ [0,1], so log M ∈ (-∞, 0]. The fixed point M* ∈ (0,1), so log M* is finite.
""")

for seed in [0, 1, 11]:
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    
    # Check: is ln(1-N_k(M)) a simple function?
    # N_k = A_k/D_k
    # 1-N_k = (B_k+ε_k)/D_k
    # ln(1-N_k) = ln(B_k+ε_k) - ln(D_k)
    # ln N_k = ln A_k - ln D_k
    # 
    # So ln(N_k/(1-N_k)) = ln A_k - ln(B_k+ε_k) = logit(N_k)
    # This is the log-odds ratio!
    
    M = np.array([0.2, 0.4, 0.6, 0.3, 0.1])
    A = a + w @ M
    B = b + v @ M
    D = A + B + eps
    NM = A / D
    
    logit_NM = np.log(NM / (1 - NM))
    logit_M = np.log(M / (1 - M))
    
    print(f"  Seed {seed}: logit shift = {logit_NM - logit_M}")

# ============================================================
# APPROACH 7: The key simplification
# ============================================================
print(f"\n{'='*70}")
print("APPROACH 7: The key simplification J_kj = [w(1-M*) - M*v]/D*")
print(f"{'='*70}")
print("""
J_kj(M*) = [w_kj(1-M*_k) - M*_k v_kj] / D*_k

This means:
- J_kj > 0 when w_kj/v_kj > M*_k/(1-M*_k), i.e., weight ratio exceeds odds ratio
- J_kj < 0 when v_kj dominates
- J_kj = 0 when w_kj(1-M*_k) = M*_k v_kj

And: (I-J)_kk = 1
     (I-J)_kj = [M*_k v_kj - w_kj(1-M*_k)] / D*_k

For sym(I-J)_kj = [(I-J)_kj + (I-J)_jk] / 2:

(I-J)_kj = [M*_k v_kj - w_kj(1-M*_k)] / D*_k
(I-J)_jk = [M*_j v_jk - w_jk(1-M*_j)] / D*_j

This has a specific structure. Let's check if there's a sign pattern.

Actually, let's compute the OFF-DIAGONAL SYMMETRIC PAIRS directly!
""")

# Check: is there a sign relationship between (I-J)_kj and (I-J)_jk?
for seed in range(5):
    a, b, eps, w, v = sample_FCA_params(seed)
    Mstar = find_fp(a, b, eps, w, v)
    D_s = a + w @ Mstar + b + v @ Mstar + eps
    
    Js = np.zeros((5, 5))
    for k in range(5):
        for j in range(5):
            Js[k, j] = (w[k, j] * (1 - Mstar[k]) - Mstar[k] * v[k, j]) / D_s[k]
    
    IJ = np.eye(5) - Js
    sym_IJ = (IJ + IJ.T) / 2
    
    print(f"\n  Seed {seed}: IJ signs:")
    print(f"  {np.array2string(IJ, precision=3, suppress_small=True)}")
    print(f"  sym(IJ) off-diagonal pairs with opposite signs: ", end="")
    opposite = 0
    same = 0
    for k in range(5):
        for j in range(k+1, 5):
            if IJ[k,j] * IJ[j,k] < 0:
                opposite += 1
            elif abs(IJ[k,j]) > 1e-10 and abs(IJ[j,k]) > 1e-10:
                same += 1
    print(f"{opposite} opposite, {same} same sign")

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")
