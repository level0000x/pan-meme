"""验证6.17A恒等式推导"""
import numpy as np

def n_op(M,a,b,e,W,V):
    num=a+W@M; den=num+b+V@M+e; return num/den

def gen_FCA(seed):
    rs=np.random.RandomState(seed%(2**31))
    a=rs.uniform(0.01,0.5,5); b=rs.uniform(0.01,0.5,5); e=rs.uniform(0.001,0.1,5)
    W=rs.uniform(0.01,0.3,(5,5)); V=rs.uniform(0.01,0.3,(5,5))
    np.fill_diagonal(W,0.0); np.fill_diagonal(V,0.0)
    t=a.sum()+b.sum()+W.sum()+V.sum(); W*=5.0/t; V*=5.0/t
    return a,b,e,W,V

a,b,e,W,V = gen_FCA(42)
Mstar = np.full(5,0.5)
for _ in range(20000):
    Mn=n_op(Mstar,a,b,e,W,V)
    if np.max(np.abs(Mn-Mstar))<1e-15: break
    Mstar=Mn

# doc notation: A*_k = a_k + (WM*)_k, B*_k = b_k + (VM*)_k (NO ε), D*_k = A*_k + B*_k + ε_k
A_star = a + W@Mstar
B_star_doc = b + V@Mstar          # without ε
D_star = A_star + B_star_doc + e   # with ε

print("=== 6.17A 推导步骤验证 (2,500 = 500点 x 5分量) ===")
for _ in range(500):
    M = np.random.uniform(0.05, 0.95, 5)
    Delta = M - Mstar
    WDelta = W @ Delta
    VDelta = V @ Delta
    N = n_op(M, a, b, e, W, V)
    
    A = a + W@M
    B_doc = b + V@M               # without ε
    D = A + B_doc + e              # with ε
    
    for k in range(5):
        # S1-S2: N_k(M)-M*_k
        s1 = A[k]/D[k] - A_star[k]/D_star[k]
        s2 = (A[k]*D_star[k] - A_star[k]*D[k]) / (D[k]*D_star[k])
        assert abs(s1-s2) < 1e-15
        
        # S3: A_k D*_k - A*_k D_k = A_k B*_k - A*_k B_k + ε_k(A_k-A*_k)
        lhs = A[k]*D_star[k] - A_star[k]*D[k]
        rhs_s3 = A[k]*B_star_doc[k] - A_star[k]*B_doc[k] + e[k]*(A[k]-A_star[k])
        assert abs(lhs - rhs_s3) < 1e-15
        
        # S4: A_k = A*_k + (WΔ)_k, B_k = B*_k + (VΔ)_k
        assert abs(A[k] - (A_star[k] + WDelta[k])) < 1e-15
        assert abs(B_doc[k] - (B_star_doc[k] + VDelta[k])) < 1e-15
        
        # S5: substitute S4 into S3
        # (A*_k+WΔ_k)B*_k - A*_k(B*_k+VΔ_k) + e_k·WΔ_k
        # = A*_k B*_k + WΔ_k B*_k - A*_k B*_k - A*_k VΔ_k + e_k WΔ_k
        # = WΔ_k(B*_k+e_k) - A*_k VΔ_k
        s5 = WDelta[k]*(B_star_doc[k]+e[k]) - A_star[k]*VDelta[k]
        assert abs(lhs - s5) < 1e-15
        
        # S6: J_kj(M*) definition
        Jkj = np.array([(W[k,j]*(B_star_doc[k]+e[k]) - A_star[k]*V[k,j]) / D_star[k]**2 for j in range(5)])
        s6 = D_star[k]**2 * np.sum(Jkj * Delta)
        assert abs(s5 - s6) < 1e-15
        
        # Final identity
        rhs = (D_star[k]/D[k]) * np.sum(Jkj * Delta)
        err = abs((N[k]-Mstar[k]) - rhs)
        assert err < 1e-14, f"Final err={err:.2e}"

print("PASSED. 注: 常数a_k,b_k通过A_k=A*_k+(WΔ)_k自动消去, 不需要单独抵消步骤")

# Also verify the simplified J(M*) form
print()
print("=== J(M*) 简化形式验证 ===")
for k in range(5):
    J_kj_simple = np.array([(W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k])/D_star[k] for j in range(5)])
    J_kj_full = np.array([(W[k,j]*(B_star_doc[k]+e[k]) - A_star[k]*V[k,j])/D_star[k]**2 for j in range(5)])
    err = np.max(np.abs(J_kj_simple - J_kj_full))
    print(f"  k={k}: max|J_simple - J_full| = {err:.2e}")
print("PASSED (使用 M*_k = A*_k/D*_k, 1-M*_k = (B*_k+e_k)/D*_k)")
