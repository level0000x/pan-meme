import numpy as np

eps = 1e-6
param = {
    'alpha1': 1.0, 'beta1': 1.0,
    'gamma1': 1.0, 'delta1': 1.0,
    'zeta1': 1.0, 'eta1': 1.0,
    'theta1': 1.0, 'kappa1': 1.0, 'kappa2': 1.0,
    'lambda1': 1.0, 'mu1': 1.0,
    'eps1': 0.01, 'eps2': 0.01, 'eps3': 0.01, 'eps4': 0.01, 'eps5': 0.01,
}

def N(M, B_up, rho_up, p):
    D, B, rho, R, S = M
    ND = (p['alpha1']*R + p['eps1']) / (p['alpha1']*R + p['beta1']*(B+B_up) + p['eps1'])
    NB = (p['gamma1']*(R+B_up) + p['eps2']) / (p['gamma1']*(R+B_up) + p['delta1']*D + p['eps2'])
    Nrho = (p['zeta1']*(D+rho_up) + p['eps3']) / (p['zeta1']*(D+rho_up) + p['eta1']*R + p['eps3'])
    NR = (p['theta1']*(rho+rho_up+B_up) + p['eps4']) / (p['theta1']*(rho+rho_up+B_up) + p['kappa1']*D + p['kappa2']*S + p['eps4'])
    NS = (p['lambda1']*D + p['eps5']) / (p['lambda1']*D + p['mu1']*R + p['eps5'])
    return np.array([ND, NB, Nrho, NR, NS])

def J_num(M, B_up, rho_up, p, h=1e-8):
    J = np.zeros((5, 5))
    f0 = N(M, B_up, rho_up, p)
    for i in range(5):
        Mh = M.copy()
        Mh[i] += h
        J[:, i] = (N(Mh, B_up, rho_up, p) - f0) / h
    return J

def J_analytic(M, B_up, rho_up, p):
    D, B, rho, R, S = M
    denD = p['alpha1']*R + p['beta1']*(B+B_up) + p['eps1']
    denB = p['gamma1']*(R+B_up) + p['delta1']*D + p['eps2']
    denRho = p['zeta1']*(D+rho_up) + p['eta1']*R + p['eps3']
    denR = p['theta1']*(rho+rho_up+B_up) + p['kappa1']*D + p['kappa2']*S + p['eps4']
    denS = p['lambda1']*D + p['mu1']*R + p['eps5']

    numD = p['alpha1']*R + p['eps1']
    numB = p['gamma1']*(R+B_up) + p['eps2']
    numRho = p['zeta1']*(D+rho_up) + p['eps3']
    numR = p['theta1']*(rho+rho_up+B_up) + p['eps4']
    numS = p['lambda1']*D + p['eps5']

    J = np.zeros((5, 5))

    J[0, 1] = -p['beta1'] * numD / denD**2
    J[0, 3] = p['alpha1'] * (denD - numD) / denD**2

    J[1, 0] = -p['delta1'] * numB / denB**2
    J[1, 3] = p['gamma1'] * (denB - numB) / denB**2

    J[2, 0] = p['zeta1'] * (denRho - numRho) / denRho**2
    J[2, 3] = -p['eta1'] * numRho / denRho**2

    J[3, 0] = -p['kappa1'] * numR / denR**2
    J[3, 1] = 0.0
    J[3, 2] = p['theta1'] * (denR - numR) / denR**2
    J[3, 4] = -p['kappa2'] * numR / denR**2

    J[4, 0] = p['lambda1'] * (denS - numS) / denS**2
    J[4, 3] = -p['mu1'] * numS / denS**2

    return J

def J_analytic_OLD(M, B_up, rho_up, p):
    D, B, rho, R, S = M
    denD = p['alpha1']*R + p['beta1']*(B+B_up) + p['eps1']
    denB = p['gamma1']*(R+B_up) + p['delta1']*D + p['eps2']
    denRho = p['zeta1']*(D+rho_up) + p['eta1']*R + p['eps3']
    denR = p['theta1']*(rho+rho_up+B_up) + p['kappa1']*D + p['kappa2']*S + p['eps4']
    denS = p['lambda1']*D + p['mu1']*R + p['eps5']

    J = np.zeros((5, 5))
    J[0, 1] = -p['beta1'] * D / denD
    J[0, 3] = p['alpha1'] * (1-D) / denD
    J[1, 0] = -p['delta1'] * B / denB
    J[1, 3] = p['gamma1'] * (1-B) / denB
    J[2, 0] = p['zeta1'] * (1-rho) / denRho
    J[2, 3] = -p['eta1'] * rho / denRho
    J[3, 0] = -p['kappa1'] * R / denR
    J[3, 1] = p['theta1'] * (1-R) / denR
    J[3, 2] = p['theta1'] * (1-R) / denR
    J[3, 4] = -p['kappa2'] * R / denR
    J[4, 0] = p['lambda1'] * (1-S) / denS
    J[4, 3] = -p['mu1'] * S / denS
    return J

def find_fp(B_up, rho_up, p, n_iters=10000):
    M = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    for _ in range(n_iters):
        M_new = N(M, B_up, rho_up, p)
        if np.max(np.abs(M_new - M)) < 1e-14:
            return M_new
        M = M_new
    return M

print("=" * 80)
print("J_RB = 0 验证")
print("=" * 80)

test_cases = [
    (0.0, 0.0, "叶"),
    (0.3, 0.3, "中间"),
    (0.8, 0.8, "大B_up大ρ_up"),
    (0.0, 0.5, "仅ρ_up"),
    (0.5, 0.0, "仅B_up"),
]

for B_up, rho_up, label in test_cases:
    M_star = find_fp(B_up, rho_up, param)
    J_num_mat = J_num(M_star, B_up, rho_up, param)
    J_ana = J_analytic(M_star, B_up, rho_up, param)

    jrb_num = J_num_mat[3, 1]
    jrb_ana = J_ana[3, 1]

    status = "OK" if abs(jrb_num) < 1e-6 else "VIOLATION"
    print(f"\n[{label}] B_up={B_up}, rho_up={rho_up}")
    print(f"  FP: D={M_star[0]:.6f}, B={M_star[1]:.6f}, rho={M_star[2]:.6f}, R={M_star[3]:.6f}, S={M_star[4]:.6f}")
    print(f"  J_RB (numerical) = {jrb_num:.12e}")
    print(f"  J_RB (analytic)  = {jrb_ana:.12e}  → {status}")

    max_diff = np.max(np.abs(J_num_mat - J_ana))
    print(f"  Max |J_num - J_ana| = {max_diff:.2e}")

    col_sum = np.sum(np.abs(J_ana), axis=0)
    row_sum = np.sum(np.abs(J_ana), axis=1)
    for k, name in enumerate(['D','B','rho','R','S']):
        print(f"  行{name} radius = {row_sum[k]:.6f}")

print("\n" + "=" * 80)
print("对比：旧版J（J_RB=θ₁(1-R)/Δ_R）vs 新版J（J_RB=0）")
print("=" * 80)

B_up, rho_up = 0.3, 0.3
M_star = find_fp(B_up, rho_up, param)
J_old = J_analytic_OLD(M_star, B_up, rho_up, param)
J_new = J_analytic(M_star, B_up, rho_up, param)
J_num_mat = J_num(M_star, B_up, rho_up, param)

print("\n旧版 J_R 行 (J_RB=θ₁(1-R)/Δ_R):")
print(f"  [J_RD={J_old[3,0]:.6f}, J_RB={J_old[3,1]:.6f}, J_Rρ={J_old[3,2]:.6f}, J_RR={J_old[3,3]:.6f}, J_RS={J_old[3,4]:.6f}]")
print(f"  Row radius = {np.sum(np.abs(J_old[3,:])):.6f}")

print("\n新版 J_R 行 (J_RB=0):")
print(f"  [J_RD={J_new[3,0]:.6f}, J_RB={J_new[3,1]:.6f}, J_Rρ={J_new[3,2]:.6f}, J_RR={J_new[3,3]:.6f}, J_RS={J_new[3,4]:.6f}]")
print(f"  Row radius = {np.sum(np.abs(J_new[3,:])):.6f}")

print("\n数值 J_R 行 (有限差分):")
print(f"  [J_RD={J_num_mat[3,0]:.6f}, J_RB={J_num_mat[3,1]:.6f}, J_Rρ={J_num_mat[3,2]:.6f}, J_RR={J_num_mat[3,3]:.6f}, J_RS={J_num_mat[3,4]:.6f}]")
print(f"  Row radius = {np.sum(np.abs(J_num_mat[3,:])):.6f}")

print("\n结论：")
print(f"  J_RB(数值) = {J_num_mat[3,1]:.2e}")
print(f"  J_RB(旧版) = {J_old[3,1]:.6f}")
print(f"  J_RB(新版) = {J_new[3,1]:.2e}")
print(f"  新版J_RB=0与数值一致: {'YES ■' if abs(J_num_mat[3,1]) < 1e-6 else 'NO ✗'}" )
print(f"  旧版J_RB≠0与数值不一致: {'YES ✗' if abs(J_old[3,1] - J_num_mat[3,1]) > 1e-4 else 'OK'}")
