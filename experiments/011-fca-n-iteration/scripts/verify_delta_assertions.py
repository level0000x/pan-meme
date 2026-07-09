"""验证 Δ_R ≥ Δ_B 断言"""
import numpy as np

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

def find_fp(B_up, rho_up, p, n_iters=20000):
    M = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    for _ in range(n_iters):
        M_new = N(M, B_up, rho_up, p)
        if np.max(np.abs(M_new - M)) < 1e-14:
            return M_new
        M = M_new
    return M

print("=" * 80)
print("验证 Δ_R ≥ Δ_B 断言")
print("=" * 80)

grid = np.linspace(0, 1, 21)
violations = []
diffs = []

for B_up in grid:
    for rho_up in grid:
        M = find_fp(B_up, rho_up, param)
        D, B, rho, R, S = M
        
        Delta_R = rho + rho_up + B_up + D + S + param['eps4']
        Delta_B = R + B_up + D + param['eps2']
        diff = Delta_R - Delta_B
        
        diffs.append(diff)
        if diff < -1e-10:
            violations.append((B_up, rho_up, diff, M))

print(f"21×21=441 扫参: Δ_R-Δ_B ∈ [{min(diffs):.4f}, {max(diffs):.4f}]")
print(f"违反 Δ_R ≥ Δ_B: {len(violations)} 组")

if violations:
    print("\n违反详情:")
    for B_up, rho_up, diff, M in violations[:10]:
        print(f"  (B_up={B_up:.2f}, ρ_up={rho_up:.2f}): Δ_R-Δ_B={diff:.6f}")
        print(f"    FP: ρ*={M[2]:.4f}, ρ_up={rho_up:.2f}, S*={M[4]:.4f}, R*={M[3]:.4f}")
else:
    print("✓ Δ_R ≥ Δ_B 在全部 441 网格点成立")

# Also check the D*B* > (1-D*)R* inequality
print("\n" + "=" * 80)
print("验证 D*B* > (1-D*)R* 不等式")
print("=" * 80)

violations_db = []
diffs_db = []
for B_up in grid:
    for rho_up in grid:
        M = find_fp(B_up, rho_up, param)
        D, B, rho, R, S = M
        lhs = D * B
        rhs = (1 - D) * R
        diff_db = lhs - rhs
        diffs_db.append(diff_db)
        if diff_db <= 0:
            violations_db.append((B_up, rho_up, diff_db, M))

print(f"21×21=441 扫参: D*B* - (1-D*)R* ∈ [{min(diffs_db):.4f}, {max(diffs_db):.4f}]")
print(f"违反 D*B* > (1-D*)R*: {len(violations_db)} 组")

if violations_db:
    for B_up, rho_up, diff, M in violations_db[:5]:
        print(f"  (B_up={B_up:.2f}, ρ_up={rho_up:.2f}): D*B*-(1-D*)R*={diff:.6f}")
        print(f"    FP: D*={M[0]:.4f}, B*={M[1]:.4f}, R*={M[3]:.4f}")
else:
    print("✓ D*B* > (1-D*)R* 在全部 441 网格点成立")
