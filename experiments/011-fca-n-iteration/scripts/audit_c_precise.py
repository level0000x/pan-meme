"""
精确 c_KL / c_max 估计 - 仅用小半径，避免裁剪
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def n_operator(M, a, b, eps, W, V):
    num = a + W @ M
    den = num + b + V @ M + eps
    return num / den

def compute_fp(a, b, eps, W, V):
    M = np.full(5, 0.5)
    for _ in range(20000):
        M_new = n_operator(M, a, b, eps, W, V)
        if np.max(np.abs(M_new - M)) < 1e-15:
            return M_new
        M = M_new
    return M

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

def D_KL_b(p, q):
    eps = 1e-15
    pp = np.clip(p, eps, 1 - eps)
    qq = np.clip(q, eps, 1 - eps)
    return pp * np.log(pp / qq) + (1 - pp) * np.log((1 - pp) / (1 - qq))

def D_KL_vec(M1, M2):
    return np.sum([D_KL_b(M1[k], M2[k]) for k in range(5)])

# 预计算固定点
print("预计算 100 组 FCA 参数...")
seeds_data = []
for seed in range(100):
    a, b, e, W, V = gen_FCA(seed)
    Mstar = compute_fp(a, b, e, W, V)
    Dstar = (a + W @ Mstar) + (b + V @ Mstar) + e
    J = np.zeros((5, 5))
    for k in range(5):
        for j in range(5):
            J[k, j] = (W[k, j] * (1.0 - Mstar[k]) - Mstar[k] * V[k, j]) / Dstar[k]
    H = np.diag([1.0 / (Mstar[k] * (1.0 - Mstar[k])) for k in range(5)])
    seeds_data.append({
        'a': a, 'b': b, 'e': e, 'W': W, 'V': V,
        'Mstar': Mstar, 'Dstar': Dstar, 'J': J, 'H': H,
    })

# 半径范围: 只用小半径，确保 M*+v ∈ [0.01, 0.99]
def safe_radius(Mstar, direction):
    """Return max r such that Mstar + r*direction stays within [0.01, 0.99]"""
    r_max = 10.0
    for k in range(5):
        if direction[k] > 0:
            r_max = min(r_max, (0.99 - Mstar[k]) / direction[k])
        elif direction[k] < 0:
            r_max = min(r_max, (Mstar[k] - 0.01) / (-direction[k]))
    return r_max

print("=" * 70)
print("[D'] c_KL 精确估计 - 仅用小半径，无裁剪")
print("  direction × 50 per seed, radii: logspace(-3, min(log10(safe_r), -1), 15)")

c_kl_all = []
for sd in seeds_data:
    Mstar, J, H = sd['Mstar'], sd['J'], sd['H']
    a, b, e, W, V = sd['a'], sd['b'], sd['e'], sd['W'], sd['V']
    
    max_c = 0.0
    n_ok = 0
    for _ in range(50):
        direction = np.random.randn(5)
        direction /= np.linalg.norm(direction)
        safe_r = safe_radius(Mstar, direction)
        if safe_r < 0.01:
            continue
        
        r_max_test = min(safe_r, 0.1)  # max 0.1 radius
        for r in np.logspace(-3, np.log10(r_max_test), 15):
            v = r * direction
            M = Mstar + v  # no clipping needed
            N = n_operator(M, a, b, e, W, V)
            
            f_v = D_KL_vec(Mstar, N) - D_KL_vec(Mstar, M)
            quad = 0.5 * v @ (J.T @ H @ J - H) @ v
            R3 = f_v - quad
            
            n_ok += 1
            if abs(R3) > 1e-15:
                c_candidate = abs(R3) / (r ** 3)
                if c_candidate > max_c:
                    max_c = c_candidate
    
    c_kl_all.append(max_c)

print(f"  c_KL (r ≤ 0.1, no clip): [{min(c_kl_all):.2f}, {max(c_kl_all):.2f}] "
      f"median={np.median(c_kl_all):.2f}")
print(f"  文档: [7.8, 27.0]")

# 也计算 λ_min
hmins = []
for sd in seeds_data:
    J, H = sd['J'], sd['H']
    hmins.append(np.linalg.eigvalsh(H - J.T @ H @ J)[0])

print(f"  λ_min(H-J^THJ): [{min(hmins):.2f}, {max(hmins):.2f}]")
print(f"  文档: [3.8, 4.2]")
print(f"  安全半径: [{min(hmins)/max(c_kl_all):.4f}, {max(hmins)/min(c_kl_all):.4f}]")
print()

print("=" * 70)
print("[H'] 6.17C c_max 精确估计 - 仅用小半径，无裁剪")

c_max_all = []
for sd in seeds_data:
    Mstar, J = sd['Mstar'], sd['J']
    a, b, e, W, V = sd['a'], sd['b'], sd['e'], sd['W'], sd['V']
    IminusJ = np.eye(5) - J
    
    max_c = 0.0
    for _ in range(50):
        direction = np.random.randn(5)
        direction /= np.linalg.norm(direction)
        safe_r = safe_radius(Mstar, direction)
        if safe_r < 0.01:
            continue
        
        r_max_test = min(safe_r, 0.1)
        for r in np.logspace(-3, np.log10(r_max_test), 15):
            v = r * direction
            M = Mstar + v
            N = n_operator(M, a, b, e, W, V)
            
            actual = (N - M) @ (Mstar - M)
            approx = v @ IminusJ @ v
            R3 = actual - approx
            
            if abs(R3) > 1e-15:
                c = abs(R3) / (r**3)
                if c > max_c:
                    max_c = c
    
    c_max_all.append(max_c)

lmin_IJ = []
for sd in seeds_data:
    J = sd['J']
    sym = 0.5 * (np.eye(5) - J + (np.eye(5) - J).T)
    lmin_IJ.append(np.linalg.eigvalsh(sym)[0])

print(f"  λ_min(sym(I-J)): [{min(lmin_IJ):.4f}, {max(lmin_IJ):.4f}]")
print(f"  文档: [0.807, 0.970]")
print(f"  c_max (r ≤ 0.1, no clip): [{min(c_max_all):.2f}, {max(c_max_all):.2f}] median={np.median(c_max_all):.2f}")
print(f"  文档声称: ~0.05-0.15")
print(f"  安全半径: [{min(lmin_IJ)/max(c_max_all):.4f}, {max(lmin_IJ)/min(c_max_all):.4f}]")

# 分析不同半径下的c_max
print()
print("  按半径分层的c_max分析 (取最后一个seed做示意):")
for r_test in [0.001, 0.003, 0.01, 0.03, 0.1]:
    vals = []
    for sd in seeds_data:
        Mstar, J = sd['Mstar'], sd['J']
        a, b, e, W, V = sd['a'], sd['b'], sd['e'], sd['W'], sd['V']
        IminusJ = np.eye(5) - J
        for _ in range(10):
            direction = np.random.randn(5)
            direction /= np.linalg.norm(direction)
            if safe_radius(Mstar, direction) < 0.01:
                continue
            v = r_test * direction
            M = Mstar + v
            N = n_operator(M, a, b, e, W, V)
            actual = (N - M) @ (Mstar - M)
            approx = v @ IminusJ @ v
            R3 = actual - approx
            if abs(R3) > 1e-15:
                vals.append(abs(R3) / (r_test ** 3))
    if vals:
        print(f"    r={r_test:.3f}: c_max estimates: [{min(vals):.2f}, {max(vals):.2f}] median={np.median(vals):.2f}")

print()
print("=" * 70)
print("总结")
print(f"  c_KL 精确估计: [{min(c_kl_all):.1f}, {max(c_kl_all):.1f}] median={np.median(c_kl_all):.1f}")
print(f"  6.17C c_max 精确估计: [{min(c_max_all):.2f}, {max(c_max_all):.2f}] median={np.median(c_max_all):.2f}")
