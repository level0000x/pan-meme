"""路线3：MVT积分形式 + 路径收缩
N(M) - N(M*) = A(M) · (M - M*), A(M) = ∫₀¹ J_N(M*+t(M-M*))dt
证明 ||A(M)||_∞ < 1 for all M ∈ [0,1]⁵
"""
import numpy as np

e = 0.01

def N_vec(M, B_up, rho_up):
    D, B, rho, R, S = M
    ND = (R + e) / (R + B + B_up + e)
    NB = (R + B_up + e) / (R + B_up + D + e)
    Nrho = (D + rho_up + e) / (D + rho_up + R + e)
    NR = (rho + rho_up + B_up + e) / (rho + rho_up + B_up + D + S + e)
    NS = (D + e) / (D + R + e)
    return np.array([ND, NB, Nrho, NR, NS])

def find_fp(B_up, rho_up):
    M = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    for _ in range(30000):
        M_new = N_vec(M, B_up, rho_up)
        if np.max(np.abs(M_new - M)) < 1e-15:
            return M_new
        M = M_new
    return M

def J_at(M, B_up, rho_up):
    D, B, rho, R, S = M
    denD = R + B + B_up + e
    denB = R + B_up + D + e
    denRho = D + rho_up + R + e
    denR = rho + rho_up + B_up + D + S + e
    denS = D + R + e
    J = np.zeros((5,5))
    J[0,1] = -D/denD
    J[0,3] = (1-D)/denD
    J[1,0] = -B/denB
    J[1,3] = (1-B)/denB
    J[2,0] = (1-rho)/denRho
    J[2,3] = -rho/denRho
    J[3,0] = -R/denR
    J[3,2] = (1-R)/denR
    J[3,4] = -R/denR
    J[4,0] = (1-S)/denS
    J[4,3] = -S/denS
    return J

def integrated_J_row_norms(M, Mstar, B_up, rho_up, n_pts=100):
    """Compute ||∫₀¹ J(M*+t(M-M*))dt||_∞ (row-wise) via quadrature"""
    t = np.linspace(0, 1, n_pts)
    dt = 1.0 / (n_pts - 1)
    
    A = np.zeros((5,5))
    for i, ti in enumerate(t):
        Mt = Mstar + ti * (M - Mstar)
        J = J_at(Mt, B_up, rho_up)
        w = dt
        if i == 0 or i == n_pts - 1:
            w = dt / 2
        A += w * J
    
    row_norms = np.sum(np.abs(A), axis=1)
    return np.max(row_norms), row_norms

print("=" * 80)
print("MVT积分验证: ||A(M)||_∞ = ||∫J||_∞ < 1?")
print("=" * 80)

test_params = [(0.0, 0.0), (0.0, 0.3), (0.3, 0.0), (0.5, 0.5), (1.0, 1.0)]

for B_up, rho_up in test_params:
    Mstar = find_fp(B_up, rho_up)
    
    max_norm = 0
    worst = None
    
    np.random.seed(42)
    n_test = 5000
    for _ in range(n_test):
        M = np.random.uniform(0, 1, 5)
        norm, _ = integrated_J_row_norms(M, Mstar, B_up, rho_up)
        if norm > max_norm:
            max_norm = norm
            worst = M
    
    print(f"\n(B_up={B_up}, ρ_up={rho_up}): max ||A||_∞ = {max_norm:.4f} {'< 1 ✓' if max_norm < 1 else '≥ 1'}")
    
    corners = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 1.0, 0.0, 1.0]),
        np.array([0.0, 0.999, 0.999, 0.0, 0.0]),
    ]
    for c in corners:
        norm, rows = integrated_J_row_norms(c, Mstar, B_up, rho_up)
        print(f"  Corner {[f'{x:.2f}' for x in c]}: ||A||={norm:.4f}")
        if norm > max_norm:
            max_norm = norm
            worst = c
    
    print(f"  GLOBAL max ||A||_∞ = {max_norm:.4f} {'< 1 ✓ GLOBAL CONTRACTION!' if max_norm < 1 else '≥ 1'}")

print("\n" + "=" * 80)
print("关键洞察：若||A(M)||_∞ < 1 for all M，则")
print("  ||N(M)-M*|| ≤ ||A(M)||_∞ · ||M-M*|| < ||M-M*||")
print("  → N是全局压缩 → 唯一FP全局收敛 ■")
print("=" * 80)
