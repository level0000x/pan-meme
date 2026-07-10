"""
颠覆性压力测试 v4: 6.17全链
============================
S1: CD（列对占优）的解析证明完整性
    - CD 不能用行和界的"同分母提因子"技巧
    - 列和 = Σ_{k≠j} |J_kj|, 每项分母 D*_k 不同
    - 验证 CD 是否数值上 200/200 通过
S2: D_low ≤ D* 严格性验证
S3: m^(t) ≤ M* ≤ u^(t) 归纳步的数学严密性 (逐项验证)
S4: 6.17C 的 O(‖Δ‖³) 项符号分析
S5: 极端参数域的边界行为
S6: 凸函数 g_k 在区间端点的 δ-分析
S7: α 界中 D_low 替代 D_min 的保守性因子
"""
import numpy as np

def n_operator(M,a,b,eps,W,V):
    num=a+W@M; return num/(num+b+V@M+eps)
def compute_fp(a,b,eps,W,V):
    M=np.full(5,.5)
    for _ in range(20000):
        Mn=n_operator(M,a,b,eps,W,V)
        if np.max(np.abs(Mn-M))<1e-15: return Mn
        M=Mn
    return M
def gen_FCA(seed):
    rs=np.random.RandomState(seed%(2**31))
    a=rs.uniform(.01,.5,5);b=rs.uniform(.01,.5,5);e=rs.uniform(.001,.1,5)
    W=rs.uniform(.01,.3,(5,5));V=rs.uniform(.01,.3,(5,5))
    np.fill_diagonal(W,0);np.fill_diagonal(V,0)
    t=a.sum()+b.sum()+W.sum()+V.sum();W*=5./t;V*=5./t
    return a,b,e,W,V
def gen_extreme(seed):
    """极端参数：极大a或极小b+ε"""
    rs=np.random.RandomState(seed%(2**31))
    a=rs.uniform(.001,2.0,5);b=rs.uniform(.001,2.0,5);e=rs.uniform(1e-5,.5,5)
    W=rs.uniform(.001,2.0,(5,5));V=rs.uniform(.001,2.0,(5,5))
    np.fill_diagonal(W,0);np.fill_diagonal(V,0)
    return a,b,e,W,V

# ============================================================
print("="*70)
print("S1: CD列对占优的解析证明完整性")
print("="*70)
print("""
问题: 行和界证明 RD 可通过提取 1/D*_k 的凸函数端点技巧。
      但 CD: c_j = Σ_{k≠j} |J_kj| 中每项的 D*_k 不同，无法提因子。
      文档声称 CD 也是"解析证明"——此声称是否正确？
""")

for label,gen_fn,n_seeds in [("FCA",gen_FCA,200),("扩展",gen_extreme,200)]:
    cd_fail=0; rd_fail=0; no_ger=0
    max_cd=0; max_rd=0; max_ger=0
    
    for s in range(n_seeds):
        a,b,e,W,V = gen_fn(s)
        Mstar=compute_fp(a,b,e,W,V); Dstar=a+b+e+(W+V)@Mstar
        D_max=a+b+e+np.sum(W+V,axis=1); sum_w=np.sum(W,axis=1)
        
        m=a/D_max; u=(a+sum_w)/(a+sum_w+b+e)
        for _ in range(2):
            m_n=(a+W@m)/(a+b+e+(W+V)@u); u_n=(a+W@u)/(a+b+e+(W+V)@m)
            m=np.maximum(m_n,0); u=np.minimum(u_n,1)
        
        J=np.zeros((5,5))
        for k in range(5):
            for j in range(5):
                if k!=j:
                    J[k,j]=(W[k,j]*(1-Mstar[k])-V[k,j]*Mstar[k])/Dstar[k]
        
        # RD using tightened bounds
        for k in range(5):
            rd_k=sum(abs(W[k,j]-(W[k,j]+V[k,j])*Mstar[k]) for j in range(5) if j!=k)/Dstar[k]
            max_rd=max(max_rd,rd_k)
            if rd_k>=1-1e-10: rd_fail+=1
        
        # CD: NO tightening possible (different denominators)
        for j in range(5):
            cd_j=sum(abs(J[k,j]) for k in range(5) if k!=j)
            max_cd=max(max_cd,cd_j)
            if cd_j>=1-1e-10: cd_fail+=1
        
        A=np.eye(5)-J; Sym=(A+A.T)/2
        for k in range(5):
            rk=sum(abs(A[k,j]) for j in range(5) if j!=k)
            ck=sum(abs(A[i,k]) for i in range(5) if i!=k)
            ger=(rk+ck)/2
            max_ger=max(max_ger,ger)
            if ger>=1-1e-10: no_ger+=1
    
    print(f"  {label} ({n_seeds}种子):")
    print(f"    RD违规={rd_fail}/{5*n_seeds}  max RD={max_rd:.4f}")
    print(f"    CD违规={cd_fail}/{5*n_seeds}  max CD={max_cd:.4f}")
    print(f"    Gershgorin违规={no_ger}/{5*n_seeds}  max ger={max_ger:.4f}")
    print(f"    CD可被Gershgorin容忍？  max RD={max_rd:.4f} max CD={max_cd:.4f} avg={(max_rd+max_cd)/2:.4f}")

# ============================================================
print(f"\n{'='*70}")
print("S2: D_low ≤ D* 严格性验证")
print("="*70)

for label,gen_fn,n_seeds in [("FCA",gen_FCA,200),("极端",gen_extreme,200)]:
    violations=0; max_Dstar_Dlow=0
    for s in range(n_seeds):
        a,b,e,W,V=gen_fn(s)
        Mstar=compute_fp(a,b,e,W,V); Dstar=a+b+e+(W+V)@Mstar
        D_max=a+b+e+np.sum(W+V,axis=1); m0=a/D_max
        D_low_val=a+b+e+(W+V)@m0
        
        if np.any(D_low_val>Dstar*(1+1e-12)):
            violations+=1
        max_Dstar_Dlow=max(max_Dstar_Dlow,np.max(Dstar/D_low_val))
    print(f"  {label}: D_low≤D*违规={violations}/{n_seeds}  最大D*/D_low={max_Dstar_Dlow:.2f}")

# ============================================================
print(f"\n{'='*70}")
print("S3: 6.17A₃ 归纳步 — 逐项逐种子验证")
print("="*70)
print("验证: m^(t+1) ≤ M* 的充分条件")
print("  num: a+W·m^(t) ≤ a+W·M* ✓ (m^(t)≤M*, w≥0)")
print("  den: a+b+ε+(W+V)·u^(t) ≥ a+b+ε+(W+V)·M* ✓ (u^(t)≥M*, w+v≥0)")
print("  ⇒ 分数 ≤ M* ✓ (x≤X, y≥Y ⇒ x/y≤X/Y)")

for label,gen_fn,n_seeds in [("FCA",gen_FCA,200),("极端",gen_extreme,200)]:
    num_viol=0; den_viol=0; frac_viol=0
    
    for s in range(n_seeds):
        a,b,e,W,V=gen_fn(s)
        Mstar=compute_fp(a,b,e,W,V)
        D_max=a+b+e+np.sum(W+V,axis=1); sum_w=np.sum(W,axis=1)
        m=a/D_max; u=(a+sum_w)/(a+sum_w+b+e)
        
        for t in range(5):
            for k in range(5):
                # Numerator check
                if a[k]+(W@m)[k] > a[k]+(W@Mstar)[k]*(1+1e-12):
                    num_viol+=1
                # Denominator check
                if a[k]+b[k]+e[k]+((W+V)@u)[k] < (a[k]+b[k]+e[k]+((W+V)@Mstar)[k])*(1-1e-12):
                    den_viol+=1
            
            m_n=(a+W@m)/(a+b+e+(W+V)@u); u_n=(a+W@u)/(a+b+e+(W+V)@m)
            m=np.maximum(m_n,0); u=np.minimum(u_n,1)
    
    print(f"  {label}: num违={num_viol} den违={den_viol} "
          f"{'✓' if num_viol==0 and den_viol==0 else '✗'}")

# ============================================================
print(f"\n{'='*70}")
print("S4: 凸函数 g_k 端点论证的完备性")
print("="*70)
print("g_k(x)=Σ_j|w-(w+v)x| 在区间[m_k,u_k]上凸 → max在端点")
print("M*_k∈[m_k,u_k] ⇒ g_k(M*_k)≤max(g_k(m_k),g_k(u_k))")

# Verify with counterexample search
for label,gen_fn,n_seeds in [("FCA",gen_FCA,200)]:
    conv_viol=0; endpt_viol=0
    
    for s in range(n_seeds):
        a,b,e,W,V=gen_fn(s)
        D_max=a+b+e+np.sum(W+V,axis=1); sum_w=np.sum(W,axis=1)
        m=a/D_max; u=(a+sum_w)/(a+sum_w+b+e)
        for _ in range(2):
            m_n=(a+W@m)/(a+b+e+(W+V)@u); u_n=(a+W@u)/(a+b+e+(W+V)@m)
            m=np.maximum(m_n,0); u=np.minimum(u_n,1)
        
        Mstar=compute_fp(a,b,e,W,V)
        
        for k in range(5):
            def gk(x):
                return sum(abs(W[k,j]-(W[k,j]+V[k,j])*x) for j in range(5) if j!=k)
            
            g_m=gk(m[k]); g_u=gk(u[k]); g_ms=gk(Mstar[k])
            
            if g_ms>max(g_m,g_u)*(1+1e-12):
                endpt_viol+=1
            
            # Check convexity: midpoint ≤ avg(endpoints)
            mid=(m[k]+u[k])/2; g_mid=gk(mid)
            if g_mid>(g_m+g_u)/2*(1+1e-12):
                conv_viol+=1
    
    print(f"  {label}: 端点最大违规={endpt_viol}/1000  凸性违规={conv_viol}/1000 "
          f"{'✓' if endpt_viol==0 else '✗凸函数端点论证有漏洞!'}")

# ============================================================
print(f"\n{'='*70}")
print("S5: 6.17C — O(‖Δ‖³) 余项的符号分析")
print("="*70)
print("(N-M)·(M*-M) = Δ^T(I-J)Δ + O(‖Δ‖³)")
print("验证: O(‖Δ‖³)项是否可能在大Δ时翻转符号?")

for s in [0,11,21,67,149]:
    a,b,e,W,V=gen_FCA(s)
    Mstar=compute_fp(a,b,e,W,V); Dstar=a+b+e+(W+V)@Mstar
    J=np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k!=j: J[k,j]=(W[k,j]*(1-Mstar[k])-V[k,j]*Mstar[k])/Dstar[k]
    
    A_I=np.eye(5)-J
    min_eig=np.min(np.linalg.eigvalsh((A_I+A_I.T)/2))
    
    # Search for sign violations at large Δ
    violations=0; total=0
    for _ in range(5000):
        u=np.random.randn(5); u/=np.linalg.norm(u)
        for r in np.linspace(.001,2.0,20):
            M=Mstar+r*u
            if np.any(M<1e-6) or np.any(M>1-1e-6): continue
            M=np.clip(M,1e-6,1-1e-6)
            N_val=n_operator(M,a,b,e,W,V)
            dot=(N_val-M)@(Mstar-M)
            total+=1
            if dot<=0: violations+=1
    
    print(f"  seed {s}: λ_min(sym(I-J))={min_eig:.4f}  "
          f"方向单调违规={violations}/{total}={100*violations/max(1,total):.1f}%")

# ============================================================
print(f"\n{'='*70}")
print("S6: 补充—J 符号模式与 g_k 凸性")
print("="*70)

# J_kj = (w_kj(1-M*_k)-v_kj M*_k)/D*_k
# sign(J_kj) = sign(w_kj(1-M*_k)-v_kj M*_k)
# = sign(w_kj - (w_kj+v_kj)M*_k)
#
# Each term in g_k(x) = |w - (w+v)x| 
# At x=M*_k: |w - (w+v)M*_k| = D*_k · |J_kj|
#
# The zero crossing is at x = w/(w+v) ∈ (1/2, 1) when w>v, ∈ (0,1/2) when w<v
# So the convexity argument is:
# g_k(x) = Σ|w-(w+v)x| is convex because each term is convex (absolute value of linear function)

for s in [0,11,149]:
    a,b,e,W,V=gen_FCA(s)
    Mstar=compute_fp(a,b,e,W,V)
    
    n_positive=0; n_negative=0; n_zero_cross=0
    for k in range(5):
        for j in range(5):
            if k==j: continue
            w=W[k,j]; v=V[k,j]; xstar=Mstar[k]
            x0=w/(w+v) if w+v>0 else 0
            n_positive+=(1 if w-(w+v)*xstar>0 else 0)
            n_negative+=(1 if w-(w+v)*xstar<0 else 0)
            if xstar<x0: n_zero_cross+=1
    
    print(f"  seed {s}: J正号={n_positive} J负号={n_negative} M*_k<w/(w+v)={n_zero_cross}/20")

# ============================================================
print(f"\n{'='*70}")
print("S7: α界中 D_low/D_min 替代比")
print("="*70)

for s in [0,11,67,149]:
    a,b,e,W,V=gen_FCA(s)
    Mstar=compute_fp(a,b,e,W,V); Dstar=a+b+e+(W+V)@Mstar
    D_max=a+b+e+np.sum(W+V,axis=1); m0=a/D_max
    D_min=a+b+e; D_low_val=a+b+e+(W+V)@m0
    
    J=np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k!=j: J[k,j]=(W[k,j]*(1-Mstar[k])-V[k,j]*Mstar[k])/Dstar[k]
    
    gamma_min=Dstar/D_min; gamma_low=Dstar/D_low_val
    alpha_min=max([sum(abs(J[:,j])*gamma_min) for j in range(5)])
    alpha_low=max([sum(abs(J[:,j])*gamma_low) for j in range(5)])
    
    print(f"  seed {s}: α(D_min)={alpha_min:.4f}  "
          f"α(D_low)={alpha_low:.4f}  "
          f"收紧比={alpha_low/alpha_min:.4f}  "
          f"D*/D_low={np.max(gamma_low):.2f}")

# ============================================================
print(f"\n{'='*70}")
print("S8: 6.17D η'' 上界的解析论证")
print("="*70)

# η''(r) = Σ_k [θ_k w_k²/A_k² + (1-θ_k)v_k²/B_k² - (w_k+v_k)²/D_k²]
#         ≤ Σ_k [θ_k w_k²/A_k² + (1-θ_k)v_k²/B_k²]
#         ≤ Σ_k [θ_k w_k²/a_k² + (1-θ_k)v_k²/(b_k+ε_k)²]
#
# Since A_k = a_k + (W·M)_k ≥ a_k, B_k = b_k+ε_k + (V·M)_k ≥ b_k+ε_k

for s in [0,11,67,149]:
    a,b,e,W,V=gen_FCA(s)
    Mstar=compute_fp(a,b,e,W,V); theta=Mstar
    
    # Theoretical upper bound on η''
    thet_ub=0
    for k in range(5):
        thet_ub+=theta[k]*np.sum(W[k,:]**2)/a[k]**2
        thet_ub+=(1-theta[k])*np.sum(V[k,:]**2)/(b[k]+e[k])**2
    
    # Actually test: how far is this upper bound from the actual max η''?
    max_eta=0
    for _ in range(2000):
        u=np.random.randn(5); u/=np.linalg.norm(u)
        w_vec=W@u; v_vec=V@u
        for r in np.linspace(.01,2.0,15):
            M=Mstar+r*u
            if np.any(M<1e-6) or np.any(M>1-1e-6): continue
            M=np.clip(M,1e-6,1-1e-6)
            A=a+W@M; B=b+V@M+e; D=A+B
            eta=sum(-(w_vec+v_vec)**2/D**2+theta*w_vec**2/A**2+(1-theta)*v_vec**2/B**2)
            max_eta=max(max_eta,eta)
    
    print(f"  seed {s}: η''理论UB={thet_ub:.4f}  实测max={max_eta:.4f}  "
          f"比率={max_eta/thet_ub:.4f}  ψ''≥4→φ''≤max_eta-4={max_eta-4:.4f}")

# ============================================================
print(f"\n{'='*70}")
print("审计结论")
print("="*70)
