"""验证引理11.1A的特征多项式形式"""
import numpy as np

# Test 1: random 5x5 zero-diag matrix - does CP = λ(λ⁴+a₂λ²+a₀)?
np.random.seed(123)
for t in range(5):
    J = np.random.randn(5,5)
    np.fill_diagonal(J, 0)
    coeffs = np.poly(J)
    # coeffs = [1, -tr, c3, -c2, c1, -det] for 5x5
    print(f"Random5x5 #{t}: coeff={[f'{c:+.4f}' for c in coeffs]}")
    # Check if odd-power terms vanish (coeffs indices: λ⁵ + c₄λ⁴ + c₃λ³ + c₂λ² + c₁λ + c₀)
    # For CP = λ(λ⁴+a₂λ²+a₀): c₄=0, c₃=a₂, c₂=0, c₁=a₀, c₀=0
    # But also NO λ⁴ term means tr(J)=0 (true for zero-diag)
    # And NO λ² term... not generally true!
    print(f"  |c₄(λ⁴)|={abs(coeffs[1]):.2e} |c₂(λ²)|={abs(coeffs[3]):.2e} |c₀(const)|={abs(coeffs[5]):.2e}")
print()

# Test 2: the specific J_N from 6.3.2 (chain C), generic 5x5
# with the sparsity pattern of J_N
# Cols: D,B,ρ,R,S (index 0,1,2,3,4)
# From the matrix at doc lines 612-618
print("=== Chain C J_N 特征多项式验证 ===")
for seed in range(10):
    rs = np.random.RandomState(seed)
    # Simulate the special J_N structure
    J = np.zeros((5,5))
    J[0,1] = -rs.uniform(0.1, 2.0)   # D row: j_DB negative
    J[0,3] = +rs.uniform(0.1, 2.0)   # D row: j_DR positive
    J[1,0] = -rs.uniform(0.1, 2.0)   # B row: j_BD negative
    J[1,3] = +rs.uniform(0.1, 2.0)   # B row: j_BR positive
    J[2,0] = +rs.uniform(0.1, 2.0)   # ρ row: j_ρD positive
    J[2,3] = -rs.uniform(0.1, 2.0)   # ρ row: j_ρR negative
    J[3,0] = -rs.uniform(0.1, 2.0)   # R row: j_RD negative
    J[3,2] = +rs.uniform(0.1, 2.0)   # R row: j_Rρ positive
    J[3,4] = -rs.uniform(0.1, 2.0)   # R row: j_RS negative
    J[4,0] = +rs.uniform(0.1, 2.0)   # S row: j_SD positive
    J[4,3] = -rs.uniform(0.1, 2.0)   # S row: j_SR negative
    
    coeffs = np.poly(J)
    lambda_vals = np.linalg.eigvals(J)
    print(f"Seed{seed}: λ={[f'{x.real:.4f}' for x in sorted(lambda_vals, key=abs)]}")
    print(f"  CP coeffs: λ⁵ {coeffs[1]:+.4f}λ⁴ {coeffs[2]:+.4f}λ³ {coeffs[3]:+.4f}λ² {coeffs[4]:+.4f}λ {coeffs[5]:+.4f}")
    print(f"  |c₄|={abs(coeffs[1]):.2e} |c₂|={abs(coeffs[3]):.2e} |c₀|={abs(coeffs[5]):.2e}")

print()
print("结论: J_N的特征多项式形式需要额外论证——不能仅从'零对角'推出")
print("c₄=0来自tr(J)=0(零对角保证) ✓")
print("c₀=0来自det(J)=0(结构奇异性) ✓")
print("c₂=0需要sign pattern的额外论证 ← 目前的声称有缺陷")

# Test 3: verify numerically for the actual FCA-generated J_N
print()
print("=== 用gen_FCA验证(随机参数) ===")
def gen_FCA(seed):
    rs=np.random.RandomState(seed%(2**31))
    a=rs.uniform(0.01,0.5,5); b=rs.uniform(0.01,0.5,5); e=rs.uniform(0.001,0.1,5)
    W=rs.uniform(0.01,0.3,(5,5)); V=rs.uniform(0.01,0.3,(5,5))
    np.fill_diagonal(W,0.0); np.fill_diagonal(V,0.0)
    t=a.sum()+b.sum()+W.sum()+V.sum(); W*=5.0/t; V*=5.0/t
    return a,b,e,W,V

for s in range(5):
    a,b,e,W,V = gen_FCA(s)
    Mstar = np.full(5,0.5)
    for _ in range(20000):
        Mn = (a+W@Mstar) / (a+W@Mstar + b+V@Mstar + e)
        if np.max(np.abs(Mn-Mstar))<1e-15: break
        Mstar=Mn
    D_star = a + b + W@Mstar + V@Mstar + e
    J = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            J[k,j] = (W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / D_star[k]
    
    coeffs = np.poly(J)
    lambda_vals = np.linalg.eigvals(J)
    print(f"Seed{s}: |c₄|={abs(coeffs[1]):.2e} |c₂|={abs(coeffs[3]):.2e} |c₀|={abs(coeffs[5]):.2e}")
    print(f"  λ={[f'{x.real:.6f}' for x in sorted(lambda_vals, key=abs)]}")
