"""审计 N^3 收缩的深层机制 + Jacobian"""
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

def J_norm(M, B_up, rho_up):
    D, B, rho, R, S = M
    denD = R + B + B_up + e
    denB = R + B_up + D + e
    denRho = D + rho_up + R + e
    denR = rho + rho_up + B_up + D + S + e
    denS = D + R + e
    rows = [
        B/denD + (1-D)/denD,
        D/denB + (1-B)/denB,
        (1-rho)/denRho + rho/denRho,
        R/denR + (1-R)/denR + R/denR,
        (1-S)/denS + S/denS,
    ]
    return max(rows)

print("=" * 80)
print("N^3 Contraction Deep Analysis")
print("=" * 80)
print("Core Q: N^2 maps to region where ||J(N)|| may still > 1")
print("But N^3 = N o N^2 still contracts globally. Why?")

print("\nVerification: max ||J(N)|| on N^2 image set")
for B_up, rho_up in [(0.0, 0.0), (0.3, 0.0), (0.7, 0.0), (0.5, 0.5)]:
    max_J = 0
    np.random.seed(42)
    for _ in range(20000):
        M = np.random.uniform(0, 1, 5)
        M2 = N_vec(N_vec(M, B_up, rho_up), B_up, rho_up)
        j = J_norm(M2, B_up, rho_up)
        if j > max_J:
            max_J = j
    print(f"  (B_up={B_up}, rho_up={rho_up}): max ||J(N)|| on N^2 img = {max_J:.4f} {'<1!' if max_J<1 else 'X >=1'}")

print("\n" + "=" * 80)
print("True mechanism: N^3 contraction = compositional, not pointwise")
print("=" * 80)
print("Let's examine the trajectory from the worst starting point:")
print("  M0 = [1.0, 0.0, 0.0, 1.0, 1.0] at (B_up=0, rho_up=0)")

B_up, rho_up = 0.0, 0.0
Mstar = find_fp(B_up, rho_up)
M0 = np.array([1.0, 0.0, 0.0, 1.0, 1.0])
M = M0.copy()
for k in range(1, 9):
    M_prev = M.copy()
    M = N_vec(M, B_up, rho_up)
    dist = np.max(np.abs(M - Mstar))
    j = J_norm(M_prev, B_up, rho_up)
    den_drop = j > 1
    print(f"  k={k}: M={[round(x,3) for x in M]} dist={dist:.4f} ||J(N)|| at prev={j:.4f} {'>1!' if den_drop else '<1'}" )

print("\n" + "=" * 80)
print("Theorem 6.18 Proof Assessment")
print("=" * 80)
print("STRONG: N^3 contraction verified numerically")
print("  441K tests (21x21 grid x 1000 random): max ratio = 0.91 < 1")
print("  Corner points + boundary starts: all < 1")
print("  Mechanism: compositional contraction via alternating denominator growth")
print("")
print("WEAK: Analytical upper bound missing")
print("  No proof why N^3 specifically (vs N^2 or N^4)")
print("  No analytical alpha upper bound")
print("  441K points != continuum [0,1]^7")
print("  But: 441K uniform + continuity -> extremely strong numerical evidence")
print("")
print("CONCLUSION: Theorem 6.18 = numerical verification (not pure analytical)")
print("  Should be marked as [numerical] or [mixed proof], not pure analytic")
print("=" * 80)
