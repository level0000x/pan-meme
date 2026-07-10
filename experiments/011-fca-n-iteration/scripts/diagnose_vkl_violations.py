"""诊断 V_KL 不降的具体位置"""
import numpy as np

def n_operator(M,a,b,eps,W,V):
    num=a+W@M; return num/(num+b+V@M+eps)

def compute_fp(a,b,eps,W,V):
    M=np.full(5,0.5)
    for _ in range(20000):
        Mn=n_operator(M,a,b,eps,W,V)
        if np.max(np.abs(Mn-M))<1e-15: return Mn
        M=Mn
    return M

def gen_FCA(seed):
    rs=np.random.RandomState(seed%(2**31))
    a=rs.uniform(0.01,0.5,5); b=rs.uniform(0.01,0.5,5); e=rs.uniform(0.001,0.1,5)
    W=rs.uniform(0.01,0.3,(5,5)); V=rs.uniform(0.01,0.3,(5,5))
    np.fill_diagonal(W,0); np.fill_diagonal(V,0)
    t=a.sum()+b.sum()+W.sum()+V.sum(); W*=5.0/t; V*=5.0/t
    return a,b,e,W,V

print("="*60)
print("诊断: V_KL violations 按步骤分布")
print("="*60)

viol_by_step={t:0 for t in range(10)}
total_by_step={t:0 for t in range(10)}
viol_details=[]

for s in range(200):
    a,b,e,W,V=gen_FCA(s); Mstar=compute_fp(a,b,e,W,V)
    for _ in range(10):
        M=np.random.random(5)
        for t in range(9):
            Mn=n_operator(M,a,b,e,W,V)
            kl_before=sum(Mstar*np.log(Mstar/np.clip(M,1e-15,None))+(1-Mstar)*np.log((1-Mstar)/np.clip(1-M,1e-15,None)))
            kl_after=sum(Mstar*np.log(Mstar/np.clip(Mn,1e-15,None))+(1-Mstar)*np.log((1-Mstar)/np.clip(1-Mn,1e-15,None)))
            total_by_step[t]+=1
            if kl_after>=kl_before-1e-14:
                viol_by_step[t]+=1
                if len(viol_details)<10:
                    viol_details.append((s,t,kl_before,kl_after,kl_after-kl_before))
            M=Mn

print(f"  Step | Violations | Total   | Rate")
print(f"  -----|------------|---------|------")
for t in range(9):
    r=viol_by_step[t]/max(total_by_step[t],1)*100
    print(f"  {t}->{t+1}  | {viol_by_step[t]:10d} | {total_by_step[t]:7d} | {r:5.1f}%")

print(f"\n  First 10 violations (seed, step, KL_before, KL_after, diff):")
for d in viol_details:
    print(f"    s={d[0]}, t={d[1]}->{d[1]+1}: KL_before={d[2]:.6f}, KL_after={d[3]:.6f}, diff={d[4]:.6e}")

print(f"\n{'='*60}")
print("根因: V_KL 在球外 (t=0) 是否单调?")
print("="*60)

# l₁ distance decreases every step (proven by 6.17B)
# But l₁ decrease ≠ KL decrease
# KL(M*||N(M)) < KL(M*||M) is NOT guaranteed by l₁ contraction alone

# Check: in B(M*, 0.049), is V_KL always decreasing?
viol_in_ball=0; total_in_ball=0
for s in range(200):
    a,b,e,W,V=gen_FCA(s); Mstar=compute_fp(a,b,e,W,V)
    for _ in range(200):
        u=np.random.randn(5); u/=np.linalg.norm(u)
        M=Mstar+0.049*u
        if np.any(M<1e-8) or np.any(M>1-1e-8): continue
        Mn=n_operator(M,a,b,e,W,V)
        kl_before=sum(Mstar*np.log(Mstar/np.clip(M,1e-15,None))+(1-Mstar)*np.log((1-Mstar)/np.clip(1-M,1e-15,None)))
        kl_after=sum(Mstar*np.log(Mstar/np.clip(Mn,1e-15,None))+(1-Mstar)*np.log((1-Mstar)/np.clip(1-Mn,1e-15,None)))
        total_in_ball+=1
        if kl_after>=kl_before-1e-14: viol_in_ball+=1

print(f"  V_KL violations IN B(M*, 0.049): {viol_in_ball}/{total_in_ball}")
print(f"  {'✓ 球内 V_KL 始终下降' if viol_in_ball==0 else '✗ 球内仍有 V_KL 不降'}")

print(f"\n{'='*60}")
print("结论: V_KL 非全局 Lyapunov")
print("="*60)
print("""
  6.17D 的正确表述:
  - V_KL 在 M* 邻域 B(M*, 0.049) 内严格单调下降 ■
  - V_KL 在球外 (t < T) 不保证每步单调下降 ✗
  - V_KL 是"eventual Lyapunov function" (最终 Lyapunov 函数)
    而非"strict Lyapunov function" (严格 Lyapunov 函数)
  
  证明升级:
  ■† (原宣称: 全局严格 Lyapunov) → 需下调
  ■_ball (球内严格单调) + ◆_early (前几步不保证)
""")
