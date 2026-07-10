"""
CD 解析界：迭代收紧框架
======================
c_j = Σ_{k≠j} |w_kj(1-M*_k) - v_kj M*_k| / D*_k

对每个固定k: 
  |w(1-x) - vx| = |w - (w+v)x|  在上凸, 在 x∈[m_k,u_k] 上取端点最大值
  D*_k ≥ D_low,k (已知下界)

因此 c_j ≤ Σ_{k≠j} max(|w-(w+v)m_k|, |w-(w+v)u_k|) / D_low,k

其中 m_k, u_k 是 T 轮迭代界
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

def gen_extreme(seed):
    rs=np.random.RandomState(seed%(2**31))
    a=rs.uniform(.001,2.0,5);b=rs.uniform(.001,2.0,5);e=rs.uniform(1e-5,.5,5)
    W=rs.uniform(.001,2.0,(5,5));V=rs.uniform(.001,2.0,(5,5))
    np.fill_diagonal(W,0);np.fill_diagonal(V,0)
    return a,b,e,W,V

print("="*70)
print("CD迭代界：T=0,1,2 轮的解析界覆盖率")
print("="*70)

for label,gen_fn,n_seeds in [("FCA",gen_FCA,200),("极端",gen_extreme,200)]:
    for T in [0,1,2]:
        violations=0; max_gap=0
        for s in range(n_seeds):
            a,b,e,W,V=gen_fn(s)
            Mstar=compute_fp(a,b,e,W,V); Dstar=a+b+e+(W+V)@Mstar
            D_max=a+b+e+np.sum(W+V,axis=1); sum_w=np.sum(W,axis=1)
            
            m=a/D_max; u=(a+sum_w)/(a+sum_w+b+e)
            for _ in range(T):
                m_n=(a+W@m)/(a+b+e+(W+V)@u); u_n=(a+W@u)/(a+b+e+(W+V)@m)
                m=np.maximum(m_n,0); u=np.minimum(u_n,1)
            
            D_low_val=a+b+e+(W+V)@m
            
            for j in range(5):
                cd_bound=0; cd_true=0
                for k in range(5):
                    if k==j: continue
                    w=W[k,j]; v=V[k,j]
                    g_m=abs(w-(w+v)*m[k]); g_u=abs(w-(w+v)*u[k])
                    cd_bound+=max(g_m,g_u)/D_low_val[k]
                    cd_true+=abs(w*(1-Mstar[k])-v*Mstar[k])/Dstar[k]
                
                max_gap=max(max_gap,cd_bound-cd_true)
                if cd_bound>=1-1e-10: violations+=1
        
        print(f"  {label} T={T}: 解析界违规={violations}/{5*n_seeds}  "
              f"max overestimate={max_gap:.4f}")

# ============================================================
print(f"\n{'='*70}")
print("CD解析界最劣种子诊断")
print("="*70)

worst_seed=-1; worst_val=0; worst_j=-1
for s in range(200):
    a,b,e,W,V=gen_FCA(s)
    Mstar=compute_fp(a,b,e,W,V); Dstar=a+b+e+(W+V)@Mstar
    D_max=a+b+e+np.sum(W+V,axis=1); sum_w=np.sum(W,axis=1)
    m=a/D_max; u=(a+sum_w)/(a+sum_w+b+e)
    for _ in range(2):
        mn=(a+W@m)/(a+b+e+(W+V)@u); un=(a+W@u)/(a+b+e+(W+V)@m)
        m=np.maximum(mn,0); u=np.minimum(un,1)
    D_low_val=a+b+e+(W+V)@m
    
    for j in range(5):
        cb=0
        for k in range(5):
            if k==j: continue
            w=W[k,j]; v=V[k,j]
            g_m=abs(w-(w+v)*m[k]); g_u=abs(w-(w+v)*u[k])
            cb+=max(g_m,g_u)/D_low_val[k]
        if cb>worst_val: worst_val=cb; worst_seed=s; worst_j=j

print(f"  最劣: seed {worst_seed} j={worst_j} bound={worst_val:.4f}")

a,b,e,W,V=gen_FCA(worst_seed)
Mstar=compute_fp(a,b,e,W,V); Dstar=a+b+e+(W+V)@Mstar
D_max=a+b+e+np.sum(W+V,axis=1); sum_w=np.sum(W,axis=1)
m=a/D_max; u=(a+sum_w)/(a+sum_w+b+e)
for _ in range(2):
    mn=(a+W@m)/(a+b+e+(W+V)@u); un=(a+W@u)/(a+b+e+(W+V)@m)
    m=np.maximum(mn,0); u=np.minimum(un,1)
D_low_val=a+b+e+(W+V)@m

print(f"  M*={[f'{x:.4f}' for x in Mstar]}")
print(f"  m_(2)={[f'{x:.4f}' for x in m]}")
print(f"  u_(2)={[f'{x:.4f}' for x in u]}")
print(f"  D_low={[f'{x:.4f}' for x in D_low_val]} vs D*={[f'{x:.4f}' for x in Dstar]}")
for k in range(5):
    if k==worst_j: continue
    w=W[k,worst_j]; v=V[k,worst_j]
    g_m=abs(w-(w+v)*m[k]); g_u=abs(w-(w+v)*u[k])
    print(f"    k={k}: w={w:.4f} v={v:.4f}  "
          f"g(m)={g_m:.4f} g(u)={g_u:.4f}  "
          f"g/D_low={max(g_m,g_u)/D_low_val[k]:.4f}  "
          f"|J|={abs(w*(1-Mstar[k])-v*Mstar[k])/Dstar[k]:.4f}")
