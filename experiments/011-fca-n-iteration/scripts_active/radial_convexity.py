"""
径向单调性 - 检查 φ''(r) 的符号
================================
φ(r) = V_KL(N(M*+ru)) - V_KL(M*+ru)

命题: 若 φ''(r) < 0 ∀r>0, 则 φ'(r) 单调递减
      φ'(0)=0 ⇒ φ'(r)<0 ∀r>0 ⇒ 径向单调性 ■

验证: φ''(r) 是否恒负?
"""
import numpy as np

def n_operator(M, a, b, eps, W, V):
    num = a + W @ M
    den = num + b + V @ M + eps
    return num / den

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
    np.fill_diagonal(W,0.0); np.fill_diagonal(V,0.0)
    t=a.sum()+b.sum()+W.sum()+V.sum()
    W*=5.0/t; V*=5.0/t
    return a,b,e,W,V

def D_KL_b(p,q):
    e=1e-15; pp=np.clip(p,e,1-e); qq=np.clip(q,e,1-e)
    return pp*np.log(pp/qq)+(1-pp)*np.log((1-pp)/(1-qq))

# ============================================================
# Test: φ''(r) sign
# ============================================================
print("="*70)
print("φ''(r) 符号检查")
print("""
如果 φ''(r) < 0 for all r > 0 on all rays:
  φ'(r) = φ'(0) + ∫₀^r φ''(t)dt  < 0 (since φ'(0)=0)
  φ(r)  = φ(0)  + ∫₀^r φ'(t)dt  < 0 (since φ(0)=0)
  
  → 径向单调性 ■ (纯积分)

验证: 100种子 × 50射线 × 30采样点 = 150K φ'' 检查
""")

violations = []
total_points = 0

for s in range(100):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    
    for _ in range(50):
        u = np.random.randn(5)
        u /= np.linalg.norm(u)
        
        prev_phi_prime = None
        prev_r = None
        
        for r in np.logspace(-2, np.log10(2.0), 30):
            M = Mstar + r*u
            if np.any(M < 1e-12) or np.any(M > 1-1e-12):
                continue
            
            N = n_operator(M, a,b,e,W,V)
            Mc = np.clip(M, 1e-12, 1-1e-12)
            Nc = np.clip(N, 1e-12, 1-1e-12)
            
            phi = sum(D_KL_b(Mstar[i], Nc[i]) - D_KL_b(Mstar[i], Mc[i]) for i in range(5))
            
            # φ'(r) via finite difference
            h = r * 1e-4
            M_p = Mstar + (r+h)*u
            M_m = Mstar + (r-h)*u
            if np.any(M_p<1e-12) or np.any(M_p>1-1e-12) or np.any(M_m<1e-12) or np.any(M_m>1-1e-12):
                continue
            N_p = n_operator(M_p, a,b,e,W,V)
            N_m = n_operator(M_m, a,b,e,W,V)
            M_pc = np.clip(M_p, 1e-12, 1-1e-12); N_pc = np.clip(N_p, 1e-12, 1-1e-12)
            M_mc = np.clip(M_m, 1e-12, 1-1e-12); N_mc = np.clip(N_m, 1e-12, 1-1e-12)
            
            phi_p = sum(D_KL_b(Mstar[i], N_pc[i]) - D_KL_b(Mstar[i], M_pc[i]) for i in range(5))
            phi_m = sum(D_KL_b(Mstar[i], N_mc[i]) - D_KL_b(Mstar[i], M_mc[i]) for i in range(5))
            
            phi_prime = (phi_p - phi_m) / (2*h)
            total_points += 1
            
            if prev_phi_prime is not None and prev_r is not None:
                phi_double = (phi_prime - prev_phi_prime) / (r - prev_r)
                if phi_double > 0:
                    violations.append((s, r, phi_double))
            
            prev_phi_prime = phi_prime
            prev_r = r

print(f"总φ''检查点: {total_points}")
print(f"φ''>0: {len(violations)}/{total_points} ({100*len(violations)/total_points:.4f}%)")

if violations:
    print(f"\n前5个违规:")
    for s,r,val in violations[:5]:
        print(f"  种子{s} r={r:.4f} φ''={val:.6f}")
else:
    print("\n✓ φ''(r) ≤ 0 全域成立! → φ'(r)单调递减")

print()

# ============================================================
# 如果φ''不恒负, 检查φ'本身
# ============================================================
print("="*70)
print("φ'(r) 直接检查")
print()

phi_prime_pos = 0
for s in range(100):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    
    for _ in range(50):
        u = np.random.randn(5)
        u /= np.linalg.norm(u)
        
        for r in np.logspace(-2, np.log10(2.0), 30):
            M = Mstar + r*u
            if np.any(M<1e-12) or np.any(M>1-1e-12): continue
            
            h = r*1e-4
            M_p = Mstar + (r+h)*u
            M_m = Mstar + (r-h)*u
            if np.any(M_p<1e-12) or np.any(M_p>1-1e-12): continue
            if np.any(M_m<1e-12) or np.any(M_m>1-1e-12): continue
            
            N_p=n_operator(M_p,a,b,e,W,V); N_m=n_operator(M_m,a,b,e,W,V)
            M_pc=np.clip(M_p,1e-12,1-1e-12); M_mc=np.clip(M_m,1e-12,1-1e-12)
            N_pc=np.clip(N_p,1e-12,1-1e-12); N_mc=np.clip(N_m,1e-12,1-1e-12)
            phi_p=sum(D_KL_b(Mstar[i],N_pc[i])-D_KL_b(Mstar[i],M_pc[i]) for i in range(5))
            phi_m=sum(D_KL_b(Mstar[i],N_mc[i])-D_KL_b(Mstar[i],M_mc[i]) for i in range(5))
            phi_prime = (phi_p-phi_m)/(2*h)
            
            if phi_prime > 1e-14:
                phi_prime_pos += 1

total = 100*50*30  # approximate
print(f"φ'(r)>0: {phi_prime_pos} (检测到正导数的次数)")
if phi_prime_pos == 0:
    print("✓ φ'(r)≤0 全域成立 — 径向单调性 100% 数值验证")
else:
    print(f"存在{phi_prime_pos}次φ'>0")
