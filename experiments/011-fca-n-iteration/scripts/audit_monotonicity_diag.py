"""
诊断: 6.17A₃ 单调性违规细节
============================
Q1: u^(1) ≤ u^(0) 的违规在哪里？幅度多大？
Q2: m^(2) ≥ m^(1) 的违规在哪里？
Q3: T=2 区间长度 vs T=0 区间长度？
Q4: 为什么 A2 有效性无违规但 A3/A4 单调性有违规？
"""
import numpy as np

def compute_fp(a,b,eps,W,V):
    M = np.full(5, 0.5)
    for _ in range(20000):
        Mn = (a+W@M)/(a+W@M+b+V@M+eps)
        if np.max(np.abs(Mn-M))<1e-15: return Mn
        M = Mn
    return M

def gen_FCA(seed):
    rs = np.random.RandomState(seed%(2**31))
    a=rs.uniform(.01,.5,5);b=rs.uniform(.01,.5,5);e=rs.uniform(.001,.1,5)
    W=rs.uniform(.01,.3,(5,5));V=rs.uniform(.01,.3,(5,5))
    np.fill_diagonal(W,0);np.fill_diagonal(V,0)
    t=a.sum()+b.sum()+W.sum()+V.sum();W*=5./t;V*=5./t
    return a,b,e,W,V

def gen_extended(seed):
    rs=np.random.RandomState(seed%(2**31))
    a=rs.uniform(.005,.8,5);b=rs.uniform(.005,.8,5);e=rs.uniform(.0005,.2,5)
    W=rs.uniform(.005,.5,(5,5));V=rs.uniform(.005,.5,(5,5))
    np.fill_diagonal(W,0);np.fill_diagonal(V,0)
    return a,b,e,W,V

print("="*70)
print("Q1: u^(1) ≤ u^(0) 违规分析 (FCA)")
print("="*70)

u1_violations = []
for s in range(200):
    a,b,e,W,V = gen_FCA(s)
    D_max = a+b+e+np.sum(W+V,axis=1)
    sum_w = np.sum(W,axis=1)
    m0 = a/D_max
    u0 = (a+sum_w)/(a+sum_w+b+e)
    m1 = (a+W@m0)/(a+b+e+(W+V)@u0)
    u1 = (a+W@u0)/(a+b+e+(W+V)@m0)
    m1=np.maximum(m1,0); u1=np.minimum(u1,1)
    
    for k in range(5):
        if u1[k] > u0[k]*(1+1e-12):
            u1_violations.append((s,k,u0[k],u1[k],u1[k]-u0[k]))

print(f"  u^(1)>u^(0) 违规数: {len(u1_violations)}/1000")
if u1_violations:
    print(f"\n  {'seed':>5} {'k':>3} {'u^(0)':>10} {'u^(1)':>10} {'delta':>10} {'delta/u^(0)':>12}")
    print(f"  {'-'*55}")
    for s,k,u0v,u1v,d in sorted(u1_violations, key=lambda x:-x[4])[:15]:
        print(f"  {s:>5} {k:>3} {u0v:>10.6f} {u1v:>10.6f} {d:>10.6f} {d/u0v:>12.4%}")

print(f"\n{'='*70}")
print("Q2: m^(2) ≥ m^(1) 违规分析 (FCA)")
print("="*70)

m2_violations = []
for s in range(200):
    a,b,e,W,V = gen_FCA(s)
    D_max = a+b+e+np.sum(W+V,axis=1)
    sum_w = np.sum(W,axis=1)
    m0 = a/D_max; u0 = (a+sum_w)/(a+sum_w+b+e)
    m1 = (a+W@m0)/(a+b+e+(W+V)@u0); u1 = (a+W@u0)/(a+b+e+(W+V)@m0)
    m1=np.maximum(m1,0); u1=np.minimum(u1,1)
    
    m2 = (a+W@m1)/(a+b+e+(W+V)@u1); u2 = (a+W@u1)/(a+b+e+(W+V)@m1)
    m2=np.maximum(m2,0); u2=np.minimum(u2,1)
    
    for k in range(5):
        if m2[k] < m1[k]*(1-1e-12):
            m2_violations.append((s,k,m1[k],m2[k],m2[k]-m1[k]))

print(f"  m^(2)<m^(1) 违规数: {len(m2_violations)}/1000")
if m2_violations:
    print(f"\n  {'seed':>5} {'k':>3} {'m^(1)':>10} {'m^(2)':>10} {'delta':>12}")
    print(f"  {'-'*50}")
    for s,k,m1v,m2v,d in sorted(m2_violations, key=lambda x:x[4])[:15]:
        print(f"  {s:>5} {k:>3} {m1v:>10.6f} {m2v:>10.6f} {d:>12.6e}")

print(f"\n{'='*70}")
print("Q3: 区间长度收缩分析")
print("="*70)

print(f"\n  {'seed':>5} {'|u^(0)-m^(0)|_∞':>18} {'|u^(2)-m^(2)|_∞':>18} {'收缩':>10}")
print(f"  {'-'*60}")
for s in [0, 11, 21, 67, 149]:
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    D_max = a+b+e+np.sum(W+V,axis=1); sum_w=np.sum(W,axis=1)
    m=a/D_max; u=(a+sum_w)/(a+sum_w+b+e)
    L0 = np.max(np.abs(u-m))
    
    for _ in range(2):
        m_next=(a+W@m)/(a+b+e+(W+V)@u); u_next=(a+W@u)/(a+b+e+(W+V)@m)
        m=np.maximum(m_next,0); u=np.minimum(u_next,1)
    
    L2 = np.max(np.abs(u-m))
    print(f"  {s:>5} {L0:>18.6f} {L2:>18.6f} {L2/L0:>10.4f}")

# 全统计
ratios = []
for s in range(200):
    a,b,e,W,V = gen_FCA(s)
    D_max=a+b+e+np.sum(W+V,axis=1); sum_w=np.sum(W,axis=1)
    m=a/D_max; u=(a+sum_w)/(a+sum_w+b+e)
    L0=np.max(np.abs(u-m))
    for _ in range(2):
        m_next=(a+W@m)/(a+b+e+(W+V)@u); u_next=(a+W@u)/(a+b+e+(W+V)@m)
        m=np.maximum(m_next,0); u=np.minimum(u_next,1)
    L2=np.max(np.abs(u-m))
    ratios.append(L2/L0)

ratios=np.array(ratios)
print(f"\n  全200种子: L2/L0 min={np.min(ratios):.4f} max={np.max(ratios):.4f} "
      f"mean={np.mean(ratios):.4f} median={np.median(ratios):.4f}")
print(f"  L2/L0 < 1: {np.sum(ratios<1)}/200 ({100*np.sum(ratios<1)/200:.1f}%)  "
      f"< 0.5: {np.sum(ratios<.5)}/200")

print(f"\n{'='*70}")
print("Q4: 为什么有效性无违规但单调性有违规?")
print("="*70)
print("""
答案: 单调性违规不影响有效性的原因——

m^(t+1) ≤ M* ≤ u^(t+1) 只需要:
  m^(t) ≤ M* ∧ u^(t) ≥ M* (归纳假设)
此条件对 u^(1) > u^(0) 的情况仍然成立，因为 u^(1) 虽然可能大于 u^(0)，
但仍然 ≥ M* (归纳步已证明)。

所以单调性声明 "(ii) m^(t)≤m^(t+1)≤u^(t+1)≤u^(t)" 并不总是成立。
但有效性 "(i) m^(t)≤M*≤u^(t)" 严格成立。

结论: 需要修正引理 6.17A₃，删除单调性声明，仅保留有效性声明。
收敛性 (iii) 可通过 m^(t) ≤ M* ≤ u^(t) + 区间收缩来证明（无需单调性）。
""")
