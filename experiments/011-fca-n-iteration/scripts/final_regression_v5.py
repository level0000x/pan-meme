"""
最终回归 v5: 文档修正后的完整验证
"""
import numpy as np

def compute_fp(a,b,eps,W,V):
    M=np.full(5,.5)
    for _ in range(20000):
        Mn=(a+W@M)/(a+W@M+b+V@M+eps)
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

def g_k_val(k,x,W,V):
    return sum(abs(W[k,j]-(W[k,j]+V[k,j])*x) for j in range(5) if j!=k)

import time
t0=time.time()
print("="*70)
print("最终回归验证 (200 FCA 种子)")
print("="*70)

results={}

# R1: 6.17A₃ 有效性
viol=0
for s in range(200):
    a,b,e,W,V=gen_FCA(s); Mstar=compute_fp(a,b,e,W,V)
    D_max=a+b+e+np.sum(W+V,axis=1); sum_w=np.sum(W,axis=1)
    m=a/D_max; u=(a+sum_w)/(a+sum_w+b+e)
    for t in range(5):
        for k in range(5):
            if m[k]>Mstar[k]*(1+1e-12) or u[k]<Mstar[k]*(1-1e-12): viol+=1
        m_n=(a+W@m)/(a+b+e+(W+V)@u); u_n=(a+W@u)/(a+b+e+(W+V)@m)
        m=np.maximum(m_n,0); u=np.minimum(u_n,1)
results['R1_6.17A3'] = (viol==0, f"{viol}/5000")

# R2: T=2行和界
rd_fails=0
for s in range(200):
    a,b,e,W,V=gen_FCA(s); Mstar=compute_fp(a,b,e,W,V); Dstar=a+b+e+(W+V)@Mstar
    D_max=a+b+e+np.sum(W+V,axis=1); sum_w=np.sum(W,axis=1)
    m=a/D_max; u=(a+sum_w)/(a+sum_w+b+e)
    for _ in range(2):
        mn=(a+W@m)/(a+b+e+(W+V)@u); un=(a+W@u)/(a+b+e+(W+V)@m)
        m=np.maximum(mn,0); u=np.minimum(un,1)
    for k in range(5):
        B=max(g_k_val(k,m[k],W,V),g_k_val(k,u[k],W,V))
        if B>=Dstar[k]*(1-1e-10): rd_fails+=1
results['R2_RD_rowsum'] = (rd_fails==0, f"{rd_fails}/1000")

# R3: 6.17B α < 1
alphas=[]
for s in range(200):
    a,b,e,W,V=gen_FCA(s); Mstar=compute_fp(a,b,e,W,V); Dstar=a+b+e+(W+V)@Mstar
    D_max=a+b+e+np.sum(W+V,axis=1); D_low=a+b+e+(W+V)@(a/D_max)
    J=np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k!=j: J[k,j]=(W[k,j]*(1-Mstar[k])-V[k,j]*Mstar[k])/Dstar[k]
    alpha=max([sum(abs(J[:,j])*Dstar/D_low) for j in range(5)])
    alphas.append(alpha)
results['R3_alpha'] = (max(alphas)<1, f"max={max(alphas):.4f}")

# R4: sym(I-J)≻0
posdef=0
for s in range(200):
    a,b,e,W,V=gen_FCA(s); Mstar=compute_fp(a,b,e,W,V); Dstar=a+b+e+(W+V)@Mstar
    J=np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k!=j: J[k,j]=(W[k,j]*(1-Mstar[k])-V[k,j]*Mstar[k])/Dstar[k]
    A=np.eye(5)-J
    if np.min(np.linalg.eigvalsh((A+A.T)/2))>1e-12: posdef+=1
results['R4_symposdef'] = (posdef==200, f"{posdef}/200")

# R5: Gershgorin链
ger_ok=0
for s in range(200):
    a,b,e,W,V=gen_FCA(s); Mstar=compute_fp(a,b,e,W,V); Dstar=a+b+e+(W+V)@Mstar
    J=np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            if k!=j: J[k,j]=(W[k,j]*(1-Mstar[k])-V[k,j]*Mstar[k])/Dstar[k]
    A=np.eye(5)-J
    r=np.sum(np.abs(A-np.diag(np.diag(A))),axis=1)
    c=np.sum(np.abs(A-np.diag(np.diag(A))),axis=0)
    if all((r+c)/2<1-1e-12): ger_ok+=1
results['R5_Gershgorin'] = (ger_ok==200, f"{ger_ok}/200")

# R6: φ''(0)<0
v6=0
for s in range(200):
    a,b,e,W,V=gen_FCA(s); Mstar=compute_fp(a,b,e,W,V)
    Dstar=a+b+e+(W+V)@Mstar; theta=Mstar
    for _ in range(50):
        u=np.random.randn(5); u/=np.linalg.norm(u)
        w_vec=W@u; v_vec=V@u
        ps=sum(((1-theta)*w_vec-theta*v_vec)**2/(theta*(1-theta)*Dstar**2))
        kl=sum(u**2/(theta*(1-theta)))
        if ps-kl>0: v6+=1
results['R6_phi0'] = (v6==0, f"{v6}/10000")

# R7: CD 数值
cd_max=0
for s in range(200):
    a,b,e,W,V=gen_FCA(s); Mstar=compute_fp(a,b,e,W,V); Dstar=a+b+e+(W+V)@Mstar
    for j in range(5):
        cd=sum(abs((W[k,j]*(1-Mstar[k])-V[k,j]*Mstar[k])/Dstar[k]) for k in range(5) if k!=j)
        cd_max=max(cd_max,cd)
results['R7_CD'] = (cd_max<1, f"max={cd_max:.4f}")

# R8: D_low≤D*
v8=0
for s in range(200):
    a,b,e,W,V=gen_FCA(s); Mstar=compute_fp(a,b,e,W,V); Dstar=a+b+e+(W+V)@Mstar
    D_low_val=a+b+e+(W+V)@(a/(a+b+e+np.sum(W+V,axis=1)))
    if np.any(D_low_val>Dstar*(1+1e-12)): v8+=1
results['R8_Dlow_le_Dstar'] = (v8==0, f"{v8}/200")

# R9: 凸函数端点
v9=0
for s in range(200):
    a,b,e,W,V=gen_FCA(s); Mstar=compute_fp(a,b,e,W,V)
    D_max=a+b+e+np.sum(W+V,axis=1); sum_w=np.sum(W,axis=1)
    m=a/D_max; u=(a+sum_w)/(a+sum_w+b+e)
    for _ in range(2):
        mn=(a+W@m)/(a+b+e+(W+V)@u); un=(a+W@u)/(a+b+e+(W+V)@m)
        m=np.maximum(mn,0); u=np.minimum(un,1)
    for k in range(5):
        g_ms=sum(abs(W[k,j]-(W[k,j]+V[k,j])*Mstar[k]) for j in range(5) if j!=k)
        g_m=sum(abs(W[k,j]-(W[k,j]+V[k,j])*m[k]) for j in range(5) if j!=k)
        g_u=sum(abs(W[k,j]-(W[k,j]+V[k,j])*u[k]) for j in range(5) if j!=k)
        if g_ms>max(g_m,g_u)*(1+1e-12): v9+=1
results['R9_convex'] = (v9==0, f"{v9}/1000")

# R10: 6.17C 方向单调性
v10=0; total=0
for s in range(200):
    a,b,e,W,V=gen_FCA(s); Mstar=compute_fp(a,b,e,W,V)
    for _ in range(20):
        u=np.random.randn(5); u/=np.linalg.norm(u)
        for r in np.linspace(.001,2.0,10):
            M=Mstar+r*u
            if np.any(M<1e-6) or np.any(M>1-1e-6): continue
            M=np.clip(M,1e-6,1-1e-6)
            N_val=(a+W@M)/(a+W@M+b+V@M+e)
            total+=1
            if (N_val-M)@(Mstar-M)<=0: v10+=1
results['R10_dir_mono'] = (v10==0, f"{v10}/{total}")

print(f"  {'检查项':<20} {'结果':<15} {'状态':}")
print(f"  {'-'*45}")
for k,(ok,detail) in results.items():
    print(f"  {k:<20} {detail:<15} {'✓' if ok else '✗'}")

print(f"\n  耗时: {time.time()-t0:.1f}s")
print(f"  全部通过: {'✓' if all(ok for ok,_ in results.values()) else '✗'}")
