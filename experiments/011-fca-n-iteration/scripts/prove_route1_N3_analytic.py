"""
路线1：解析N²像集的严格区间上界，证明该区间上||J(N)||<1
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

def J_norm_on_box(D_lo, D_hi, B_lo, B_hi, rho_lo, rho_hi, R_lo, R_hi, S_lo, S_hi, B_up, rho_up):
    """Analytic worst-case ||J(N)||_inf over the box"""
    max_norm = 0
    
    for _, (lo, hi) in enumerate([
        (D_lo, D_hi), (B_lo, B_hi), (rho_lo, rho_hi), (R_lo, R_hi), (S_lo, S_hi)
    ]):
        pass
    
    # D row: ||J_D||_1 = (|J_DB| + |J_DR|) = (D + |1-D|)/Δ_D = (D + 1 - D)/Δ_D = 1/Δ_D
    # since D ∈ [0,1], |1-D| = 1-D
    # Δ_D = R + B + B_up + ε ≥ R_lo + B_lo + B_up + e
    denD_min = R_lo + B_lo + B_up + e
    assert denD_min > 0
    J_D_norm_max = 1.0 / denD_min
    max_norm = max(max_norm, J_D_norm_max)
    
    # B row: ||J_B||_1 = 1 / Δ_B, Δ_B = R + B_up + D + e ≥ R_lo + B_up + D_lo + e
    denB_min = R_lo + B_up + D_lo + e
    J_B_norm_max = 1.0 / denB_min
    max_norm = max(max_norm, J_B_norm_max)
    
    # ρ row: ||J_ρ||_1 = 1 / Δ_ρ, Δ_ρ = D + ρ_up + R + e ≥ D_lo + ρ_up + R_lo + e
    denRho_min = D_lo + rho_up + R_lo + e
    J_rho_norm_max = 1.0 / denRho_min
    max_norm = max(max_norm, J_rho_norm_max)
    
    # R row: ||J_R||_1 = (R + |1-R| + R)/Δ_R = (1+R)/Δ_R ≤ (1+R_hi)/(ρ_lo+ρ_up+B_up+D_lo+S_lo+e)
    denR_min = rho_lo + rho_up + B_up + D_lo + S_lo + e
    J_R_norm_max = (1.0 + R_hi) / denR_min
    max_norm = max(max_norm, J_R_norm_max)
    
    # S row: ||J_S||_1 = 1 / Δ_S, Δ_S = D + R + e ≥ D_lo + R_lo + e
    denS_min = D_lo + R_lo + e
    J_S_norm_max = 1.0 / denS_min
    max_norm = max(max_norm, J_S_norm_max)
    
    return max_norm

def N1_lower_upper_bounds(B_up, rho_up):
    """Analytic bounds for N(M) when M ∈ [0,1]^5"""
    e_local = e
    bounds = {}
    
    # N_D = (R+e)/(R+B+B_up+e)
    # min: R=0, B=1 → e/(1+B_up+e) = e/(1+B_up+e)
    # max: R=1, B=0 → (1+e)/(1+B_up+e)
    bounds['D'] = (e_local/(1+B_up+e_local), (1+e_local)/(1+B_up+e_local))
    
    # N_B = (R+B_up+e)/(R+B_up+D+e)
    # min: R=0, B_up small, D=1 → (B_up+e)/(B_up+1+e)
    # max: R=1, B_up large, D=0 → (1+B_up+e)/(1+B_up+e) = 1...no
    # Actually: when D=0, R=1: (1+B_up+e)/(1+B_up+e) = 1
    # min: when R=0, D=1: (B_up+e)/(B_up+1+e)
    bounds['B'] = ((B_up+e_local)/(B_up+1+e_local), 1.0)
    
    # N_ρ = (D+ρ_up+e)/(D+ρ_up+R+e)
    bounds['rho'] = ((rho_up+e_local)/(rho_up+1+e_local), (1+rho_up+e_local)/(rho_up+e_local))
    # Upper bound can exceed 1! Adjust:
    upper_rho = (1+rho_up+e_local)/(rho_up+e_local)
    bounds['rho'] = (bounds['rho'][0], min(1.0, upper_rho))
    
    # N_R = (ρ+ρ_up+B_up+e)/(ρ+ρ_up+B_up+D+S+e)
    bounds['R'] = ((rho_up+B_up+e_local)/(rho_up+B_up+2+e_local), 
                   min(1.0, (1+rho_up+B_up+e_local)/(rho_up+B_up+e_local)))
    
    # N_S = (D+e)/(D+R+e)
    bounds['S'] = (e_local/(1+e_local), (1+e_local)/(e_local))
    bounds['S'] = (bounds['S'][0], min(1.0, bounds['S'][1]))
    
    return bounds

def N2_lower_upper_bounds_analytic(B_up, rho_up):
    """Analytic bounds for N^2([0,1]^5)"""
    b1 = N1_lower_upper_bounds(B_up, rho_up)
    
    # N^2 means we apply N to points that satisfy the N^1 bounds
    # For the D component of N^2:
    #   N^2_D = N_D(N(M)) = (R'+e)/(R'+B'+B_up+e)
    # where R' and B' are components of N(M), bounded as:
    #   R' ∈ [b1['R'][0], b1['R'][1]]
    #   B' ∈ [b1['B'][0], b1['B'][1]]
    
    # N^2_D min: R'→min, B'→max
    N2_D_lo = (b1['R'][0] + e) / (b1['R'][1] + b1['B'][1] + B_up + e)
    N2_D_hi = (b1['R'][1] + e) / (b1['R'][0] + b1['B'][0] + B_up + e)
    
    # N^2_B: (R'+B_up+e)/(R'+B_up+D'+e), D' ∈ b1['D']
    N2_B_lo = (b1['R'][0] + B_up + e) / (b1['R'][1] + B_up + b1['D'][1] + e)
    N2_B_hi = (b1['R'][1] + B_up + e) / (b1['R'][0] + B_up + b1['D'][0] + e)
    
    # N^2_ρ: (D'+ρ_up+e)/(D'+ρ_up+R'+e)
    N2_rho_lo = (b1['D'][0] + rho_up + e) / (b1['D'][1] + rho_up + b1['R'][1] + e)
    N2_rho_hi = (b1['D'][1] + rho_up + e) / (b1['D'][0] + rho_up + b1['R'][0] + e)
    
    # N^2_R: (ρ'+ρ_up+B_up+e)/(ρ'+ρ_up+B_up+D'+S'+e)
    den_lo = b1['rho'][1] + b1['D'][1] + b1['S'][1] + rho_up + B_up + e
    den_hi = b1['rho'][0] + b1['D'][0] + b1['S'][0] + rho_up + B_up + e
    num_lo = b1['rho'][0] + rho_up + B_up + e
    num_hi = b1['rho'][1] + rho_up + B_up + e
    N2_R_lo = num_lo / den_hi
    N2_R_hi = min(1.0, num_hi / den_lo)
    
    # N^2_S: (D'+e)/(D'+R'+e)
    N2_S_lo = (b1['D'][0] + e) / (b1['D'][1] + b1['R'][1] + e)
    N2_S_hi = (b1['D'][1] + e) / (b1['D'][0] + b1['R'][0] + e)
    
    return {
        'D': (N2_D_lo, min(1.0, N2_D_hi)),
        'B': (N2_B_lo, min(1.0, N2_B_hi)),
        'rho': (N2_rho_lo, min(1.0, N2_rho_hi)),
        'R': (N2_R_lo, N2_R_hi),
        'S': (N2_S_lo, min(1.0, N2_S_hi)),
    }

print("=" * 80)
print("解析N^1上界 + N^2上界 → 计算||J(N)||在N^2像集上的最坏上界")
print("=" * 80)

for B_up in [0.0, 0.3, 0.5, 0.7, 1.0]:
    for rho_up in [0.0, 0.3, 0.5, 0.7, 1.0]:
        b2 = N2_lower_upper_bounds_analytic(B_up, rho_up)
        J_norm = J_norm_on_box(
            b2['D'][0], b2['D'][1],
            b2['B'][0], b2['B'][1],
            b2['rho'][0], b2['rho'][1],
            b2['R'][0], b2['R'][1],
            b2['S'][0], b2['S'][1],
            B_up, rho_up
        )
        status = "✓ <1" if J_norm < 1 else "✗ ≥1"
        print(f"  (B_up={B_up:.1f}, ρ_up={rho_up:.1f}): N² box=[{b2['D'][0]:.3f},{b2['D'][1]:.3f}] × [{b2['B'][0]:.3f},{b2['B'][1]:.3f}], max||J||={J_norm:.3f} {status}")

print("\n" + "=" * 80)
print("检验解析N²下界是否够紧（对比数值实际可达最小分量）")
print("=" * 80)

def actual_minmax_after_NK(K, B_up, rho_up, n=20000):
    mins, maxs = np.full(5, 1.0), np.full(5, 0.0)
    np.random.seed(42)
    for _ in range(n):
        M = np.random.uniform(0, 1, 5)
        for _ in range(K):
            M = N_vec(M, B_up, rho_up)
        mins = np.minimum(mins, M)
        maxs = np.maximum(maxs, M)
    return mins, maxs

for B_up, rho_up in [(0.0, 0.0), (0.5, 0.5)]:
    b2 = N2_lower_upper_bounds_analytic(B_up, rho_up)
    mins, maxs = actual_minmax_after_NK(2, B_up, rho_up)
    print(f"\n(B_up={B_up}, ρ_up={rho_up}):")
    for i, (name, (lo_a, hi_a)) in enumerate(zip(
        ['D','B','ρ','R','S'],
        [b2['D'], b2['B'], b2['rho'], b2['R'], b2['S']]
    )):
        print(f"  {name}: 解析 [{lo_a:.4f}, {hi_a:.4f}] 数值 [{mins[i]:.4f}, {maxs[i]:.4f}] 紧度={lo_a/max(1e-10,mins[i]):.1f}x")
