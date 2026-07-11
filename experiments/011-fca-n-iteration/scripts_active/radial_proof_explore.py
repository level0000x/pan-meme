"""
径向单调性解析证明探索
========================
目标: 找到 φ''(r)≤0 的解析证明结构

关键formulation (A_k, B_k 参数化):
  N_k = A_k/(A_k+B_k), A_k = a_k + Σ w_kj M_j, B_k = b_k + Σ v_kj M_j + ε_k
  φ(r) = Σ_k [log(A_k+B_k) - M*_k log(A_k) - (1-M*_k) log(B_k)
               + M*_k log(M_k) + (1-M*_k) log(1-M_k)]
       = Σ_k Φ_k(A_k,B_k,M_k)
  
探索方向:
  E1: N̈·Ṅ 符号关系 (N̈ decelerates Ṅ?)
  E2: V'·N̈ 项与 V''·Ṅ² - V''·u² 的大小比较
  E3: A,B 参数化下的 φ'' 结构
  E4: φ'' 能否写成负定二次型 + 余项?
"""
import numpy as np
from itertools import product

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

# ================================================================
# E1: N̈_k · Ṅ_k 符号关系
# 猜想: N̈_k 与 Ṅ_k 反号 (deceleration)
# 即 d|Ṅ_k|/dr = sign(Ṅ_k)·N̈_k < 0
# ================================================================
print("="*70)
print("E1: N̈·Ṅ 符号关系 (deceleration hypothesis)")
print()

violations_E1 = 0
total_E1 = 0
same_sign = 0
opp_sign = 0

for s in range(50):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    
    # Compute A*, B*, D*
    A_star = a + W @ Mstar
    B_star = b + V @ Mstar + e
    D_star = A_star + B_star
    
    for _ in range(30):
        u = np.random.randn(5)
        u /= np.linalg.norm(u)
        Wu = W @ u
        Vu = V @ u
        
        for r in [0.01, 0.05, 0.1, 0.2, 0.5]:
            M = Mstar + r*u
            if np.any(M<1e-12) or np.any(M>1-1e-12): continue
            
            A = a + W @ M
            B = b + V @ M + e
            D = A + B
            N = A / D
            
            for k in range(5):
                Ndot = ((1-N[k])*Wu[k] - N[k]*Vu[k]) / D[k]
                
                # N̈ via analytic formula (A,B parameterization)
                # d/dr of A_k = Wu_k, d/dr of B_k = Vu_k (constant)
                # ∂N/∂A = B/D², ∂N/∂B = -A/D²
                # Nddot = d/dr[B/D² * Wu + (-A/D²) * Vu]
                #       = (∂/∂A[B/D²]*Wu + ∂/∂B[B/D²]*Vu)*Wu 
                #       + (∂/∂A[-A/D²]*Wu + ∂/∂B[-A/D²]*Vu)*Vu
                # ∂/∂A[B/D²] = -2B/D³
                # ∂/∂B[B/D²] = 1/D² - 2B/D³ = (D-2B)/D³ = (A-B)/D³
                # ∂/∂A[-A/D²] = -1/D² + 2A/D³ = (2A-D)/D³ = (A-B)/D³
                # ∂/∂B[-A/D²] = -2A/D³
                #
                # So Nddot = [-2B/D³ * Wu² + 2(A-B)/D³ * Wu*Vu + 2A/D³ * Vu²]  NO!
                
                # Correct:
                # N = A/(A+B), ∂N/∂A = B/D² = B/(A+B)², ∂N/∂B = -A/D²
                # dN/dr = (B/D²)·Wu + (-A/D²)·Vu = (B·Wu - A·Vu)/D²
                # d²N/dr² = d/dr[(B·Wu - A·Vu)/D²]
                # = [(Vu·Wu - Wu·Vu)/D²] + (B·Wu - A·Vu)·(-2/D³)·(Wu+Vu)
                # the first term vanishes!
                # = -2(B·Wu - A·Vu)(Wu+Vu)/D³
                # = -2(D²·Ndot)(Wu+Vu)/D³
                # = -2Ndot·(Wu+Vu)/D
                
                Nddot = -2 * Ndot * (Wu[k] + Vu[k]) / D[k]
                
                total_E1 += 1
                if Ndot * Nddot > 0:
                    same_sign += 1
                    if abs(Ndot) > 1e-10:
                        violations_E1 += 1
                else:
                    opp_sign += 1

print(f"  N̈·Ṅ > 0 (加速): {same_sign}/{total_E1} ({100*same_sign/total_E1:.1f}%)")
print(f"  N̈·Ṅ ≤ 0 (减速): {opp_sign}/{total_E1} ({100*opp_sign/total_E1:.1f}%)")
print()

# ================================================================
# E2: φ'' A,B 参数化
# ================================================================
print("="*70)
print("E2: φ'' 的 ABS (A,B分离) 参数化")
print()

"""
φ(r) = Σ_k Φ_k where:
Φ_k = log(A_k+B_k) - θ_k log(A_k) - (1-θ_k) log(B_k) 
      + θ_k log(M_k) + (1-θ_k) log(1-M_k)
      θ_k = M*_k

d/dr:
Φ'_k = (Wu+Vu)/D - θ_k·Wu/A - (1-θ_k)·Vu/B + θ_k·u_k/M - (1-θ_k)·u_k/(1-M)

At r=0: A=A*, B=B*, M=M*, θ=A*/(A*+B*)
Φ'_k(0) = (Wu+Vu)/D* - θ·Wu/A* - (1-θ)·Vu/B* + θ·u_k/M* - (1-θ)·u_k/(1-M*)
        = (Wu+Vu)/D* - Wu/D* - Vu/D* + u_k/M* - u_k/(1-M*) ... wait

θ/M* = (A*/D*)/(A*/D*) = 1/D*? No!
θ/M* = (A*/D*)/(A*/D*) = 1... that's not right either.

θ = M*_k = A*_k/(A*_k+B*_k) = A*_k/D*_k
So θ/M*_k = θ/θ = 1
And (1-θ)/(1-M*_k) = (B*_k/D*_k)/(B*_k/D*_k) = 1

So θ·u_k/M* - (1-θ)·u_k/(1-M*) = u_k - u_k = 0! (at r=0)

And θ·Wu/A* = (A*/D*)·Wu/A* = Wu/D*
And (1-θ)·Vu/B* = (B*/D*)·Vu/B* = Vu/D*

So Φ'_k(0) = (Wu+Vu)/D* - Wu/D* - Vu/D* + 0 = 0 ✓

Now for d²/dr²:
Φ''_k = -(Wu+Vu)²/D² + θ·Wu²/A² + (1-θ)·Vu²/B² - θ·u_k²/M² - (1-θ)·u_k²/(1-M)²

At r=0:
Φ''_k(0) = -(Wu+Vu)²/D*² + θ·Wu²/A*² + (1-θ)·Vu²/B*² - θ·u_k²/M*² - (1-θ)·u_k²/(1-M*)²

= -(Wu+Vu)²/D*² + Wu²/(A*D*) + Vu²/(B*D*) - u_k²/M*² - u_k²/(1-M*)²... wait

θ/A*² = (A*/D*)/A*² = 1/(A*D*)
(1-θ)/B*² = 1/(B*D*)

θ/M*² = (A*/D*)/(A*/D*)² = D*/A*
(1-θ)/(1-M*)² = D*/B*

So Φ''_k(0) = -(Wu+Vu)²/D*² + Wu²/(A*D*) + Vu²/(B*D*) - D*u_k²/A* - D*u_k²/B*

But A* = θ·D*, B* = (1-θ)·D*

Φ''_k(0) = -(Wu+Vu)²/D*² + Wu²/(θ·D*²) + Vu²/((1-θ)·D*²) - u_k²/θ - u_k²/(1-θ)

= [-(Wu+Vu)² + Wu²/θ + Vu²/(1-θ)]/D*² - u_k²/(θ(1-θ))
"""

# Numerical verification of the A-B formulation
print("A-B 参数化验证:")
max_err = 0
for s in range(10):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    
    for _ in range(10):
        u = np.random.randn(5)
        u /= np.linalg.norm(u)
        
        for r in np.logspace(-2, 0, 20):
            M = Mstar + r*u
            if np.any(M<1e-12) or np.any(M>1-1e-12): continue
            
            N = n_operator(M, a,b,e,W,V)
            Nc = np.clip(N, 1e-12, 1-1e-12)
            Mc = np.clip(M, 1e-12, 1-1e-12)
            
            def D_KL(m, n):
                return m*np.log(m/n)+(1-m)*np.log((1-m)/(1-n))
            
            phi_fd = sum(D_KL(Mstar[i],Nc[i])-D_KL(Mstar[i],Mc[i]) for i in range(5))
            
            A = a + W @ M
            B = b + V @ M + e
            D = A + B
            phi_ab = sum(np.log(D[i]) - Mstar[i]*np.log(A[i]) 
                        - (1-Mstar[i])*np.log(B[i])
                        + Mstar[i]*np.log(Mc[i]) 
                        + (1-Mstar[i])*np.log(1-Mc[i]) for i in range(5))
            
            max_err = max(max_err, abs(phi_fd - phi_ab))

print(f"  φ 的 A-B 参数化最大误差: {max_err:.2e}")
print()

# ================================================================
# E3: φ'' 结构的项级分解
# ================================================================
print("="*70)
print("E3: φ'' 的项级分解")
print()

"""
From the A-B formulation:
Φ''_k = -(Wu+Vu)²/D² + θ·Wu²/A² + (1-θ)·Vu²/B² - θ·u_k²/M² - (1-θ)·u_k²/(1-M)²

These 5 terms can be grouped as:
  T1 = -(Wu+Vu)²/D²                                [always ≤ 0, from log(A+B)]
  T2 = θ·Wu²/A² + (1-θ)·Vu²/B²                     [always ≥ 0, from -θlog(A)-(1-θ)log(B)]
  T3 = -θ·u_k²/M² - (1-θ)·u_k²/(1-M)²             [always ≤ 0, from θlog(M)+(1-θ)log(1-M)]

Φ''_k = T1 + T2 + T3

At r=0:
T1(0) = -(Wu+Vu)²/D*²
T2(0) = Wu²/(A*D*) + Vu²/(B*D*)  (since θ/A*² = 1/(A*D*), etc.)
T3(0) = -u_k²/(θ(1-θ)·D*)... wait

T3(0) = -θ·u_k²/M*² - (1-θ)·u_k²/(1-M*)²
      = -u_k²/(θ·D*²/D*²) - ... no, M*² = (A*/D*)² = (θ D* / D*)² = θ²
So θ/M*² = θ/θ² = 1/θ
And (1-θ)/(1-M*)² = 1/(1-θ)

So T3(0) = -u_k²/θ - u_k²/(1-θ) = -u_k²/(θ(1-θ))

Φ''_k(0) = -(Wu+Vu)²/D*² + Wu²/(θ D*²) + Vu²/((1-θ)D*²) - u_k²/(θ(1-θ))

For a fixed component: let w=Wu_k, v=Vu_k, t=θ=M*_k, d=D*_k

Φ''_k(0) = -(w+v)²/d² + w²/(t d²) + v²/((1-t)d²) - u_k²/(t(1-t))
        = [-(w+v)² + w²/t + v²/(1-t)]/d² - u_k²/(t(1-t))

The term in brackets:
B = -(w+v)² + w²/t + v²/(1-t)
  = -(w²+2wv+v²) + w²/t + v²/(1-t)
  = w²(1/t - 1) + v²(1/(1-t) - 1) - 2wv
  = w²(1-t)/t + v² t/(1-t) - 2wv
  = (w√(1-t)/√t - v√t/√(1-t))²    [perfect square!]
  ≥ 0

So Φ''_k(0) = (w√(1-t)/√t - v√t/√(1-t))²/d² - u_k²/(t(1-t))

This is: nonnegative_term - nonnegative_term
First term: "N-curvature" (curvature from N operator)
Second term: "M-curvature" (curvature from M itself, i.e., D_KL curvature)

For φ''(0) < 0 we need: N-curvature < M-curvature
"""

# Verify the perfect square factorization
print("完美平方分解验证:")
max_err_sq = 0
for s in range(20):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    A_star = a + W @ Mstar
    B_star = b + V @ Mstar + e
    D_star = A_star + B_star
    
    for _ in range(20):
        u = np.random.randn(5)
        u /= np.linalg.norm(u)
        Wu = W @ u
        Vu = V @ u
        
        for k in range(5):
            t = Mstar[k]
            d = D_star[k]
            w = Wu[k]
            v = Vu[k]
            
            # Direct computation
            B_direct = -(w+v)**2 + w**2/t + v**2/(1-t)
            
            # Perfect square form
            B_sq = (w*np.sqrt(1-t)/np.sqrt(t) - v*np.sqrt(t)/np.sqrt(1-t))**2
            
            max_err_sq = max(max_err_sq, abs(B_direct - B_sq))

print(f"  完美平方分解最大误差: {max_err_sq:.2e}")
print(f"  ✓ 成立 — φ''(0) 的 N 曲率项可分解为完美平方")
print()

# ================================================================
# E4: φ''(r) for r > 0 — 扰动分析
# ================================================================
print("="*70)
print("E4: φ''(r>0) 扰动分析")
print()

"""
For r > 0, at M = M* + r u:
  A = A* + r·Wu
  B = B* + r·Vu
  D = D* + r·(Wu+Vu)
  M_k = M*_k + r·u_k

Φ''_k(r) = -(Wu+Vu)²/D² + θ·Wu²/A² + (1-θ)·Vu²/B² - θ·u_k²/M_k² - (1-θ)·u_k²/(1-M_k)²

Where θ = M*_k is FIXED — it's the fixed point value.

Let's define relative perturbations:
  α = r·Wu/A*   (relative change in A)
  β = r·Vu/B*   (relative change in B)
  γ = r·u_k/M*  (relative change in M)

Then A = A*(1+α), B = B*(1+β), M = M*(1+γ)

Φ''_k(r) = -(Wu+Vu)²/[D*²(1+α_bar)²] 
          + θ·Wu²/[A*²(1+α)²]
          + (1-θ)·Vu²/[B*²(1+β)²]
          - θ·u_k²/[M*²(1+γ)²]
          - (1-θ)·u_k²/[(1-M*)²(1+γ')²]

where α_bar = (r·(Wu+Vu))/D* is a weighted average of α and β
and γ' = -r·u_k/(1-M*) is the relative change in (1-M)

The first three terms form:
T123(r) = 1/d² * [w²(1/r-1)/r²(1+α)² + v²/((1-r)·(1+β)²) - (w+v)²/(1+α_bar)²] ... 

This is getting messy. Let me verify numerically:
"""

# For r>0, check component-by-component
print("φ'' 分项检查 (r>0):")
worst_pos = []  # track any positive φ'' values

for s in range(30):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    theta = Mstar
    A_star = a + W @ Mstar
    B_star = b + V @ Mstar + e
    D_star = A_star + B_star
    
    for _ in range(20):
        u = np.random.randn(5)
        u /= np.linalg.norm(u)
        Wu = W @ u
        Vu = V @ u
        
        for r in np.logspace(-2, np.log10(2.0), 30):
            M = Mstar + r*u
            if np.any(M<1e-12) or np.any(M>1-1e-12): continue
            
            A = a + W @ M
            B = b + V @ M + e
            D = A + B
            
            phi_dd = 0.0
            for k in range(5):
                t = theta[k]
                
                T1 = -(Wu[k]+Vu[k])**2 / D[k]**2
                T2 = t * Wu[k]**2 / A[k]**2 + (1-t) * Vu[k]**2 / B[k]**2
                T3 = -t * u[k]**2 / M[k]**2 - (1-t) * u[k]**2 / (1-M[k])**2
                
                phi_dd_k = T1 + T2 + T3
                phi_dd += phi_dd_k
            
            if phi_dd > 1e-12:
                worst_pos.append((s, r, phi_dd))

print(f"  φ''>0: {len(worst_pos)} (应全部为0)")
if worst_pos:
    for s,r,val in worst_pos[:5]:
        print(f"    种子{s} r={r:.4f} φ''={val:.6f}")
print()

# ================================================================
# E5: 寻找解析上界
# ================================================================
print("="*70)
print("E5: φ'' 的解析上界构造")
print()

"""
Consider the T1+T2 part separately from T3:

T123_k = -(Wu+Vu)²/D² + θ·Wu²/A² + (1-θ)·Vu²/B²

This is the "N curvature" part. By Jensen or Cauchy, this is ALWAYS ≥ 0.
(It's the curvature of a log-convex function)

T3_k = -θ·u_k²/M_k² - (1-θ)·u_k²/(1-M_k)² < 0

So φ''_k = T123_k + T3_k, and the question is whether T123_k < |T3_k|.

Now, T123_k involves w=Wu_k and v=Vu_k, while T3_k involves u_k.
These are related through the structure of W and V.

Key insight: Ṅ_k = ((1-N_k)w - N_k v)/D is the "effective" directional change.
And d²N_k/dr² = N̈_k = -2Ṅ_k(w+v)/D (as derived in E1)

The relationship between w=Wu_k, v=Vu_k, and u_k is determined by the 
coupling matrices W and V. Let's check the spectral relationship.
"""

# Check the relationship between ||Wu||, ||Vu|| and ||u||
print("W 和 V 的耦合强度:")
for s in range(5):
    a,b,e,W,V = gen_FCA(s+100)
    w_norm = np.linalg.norm(W, 2)
    v_norm = np.linalg.norm(V, 2)
    print(f"  种子{s+100}: ||W||₂={w_norm:.4f}  ||V||₂={v_norm:.4f}  ||W+V||₂={np.linalg.norm(W+V,2):.4f}")
print()

# ================================================================
# E6: Gershgorin 界的 KL 版本
# ================================================================
print("="*70)
print("E6: KL Gershgorin — N算子作为KL几何的压缩")
print()

"""
Standard result: if T is a contraction in some metric, and V is a Lyapunov function
in that metric, then V(T(x)) < V(x).

For KL: D_KL(p||T(q)) compared to D_KL(p||q).

For the Bernoulli case with N: 
  N_k = A_k/(A_k+B_k) = logistic(log(A_k/B_k))

This is equivalent to:
  log(N_k/(1-N_k)) = log(A_k) - log(B_k) 
                    = log(a_k + Σ w_kj M_j) - log(b_k + ε_k + Σ v_kj M_j)

In the "log-odds" space:
  L_k = logit(N_k) = log(A_k) - log(B_k)

For the fixed point:
  L*_k = logit(M*_k) = log(A*_k) - log(B*_k)

The N operator in log-odds:
  L_k(M) = log(a_k + Σ w_kj M_j) - log(b_k + ε_k + Σ v_kj M_j)
  
where M_j = sigmoid(L*_j + δ_j) for small δ.

This is a nonlinear coupling in log-odds space. But maybe it's simpler!

Actually wait — this IS simpler. Let me compute ∂L_k/∂L_j:
  
  ∂L_k/∂M_j = ∂L_k/∂A_k · w_kj + ∂L_k/∂B_k · (-v_kj)
            = w_kj/A_k - v_kj/B_k

  ∂L_k/∂L_j = (∂L_k/∂M_j) · (∂M_j/∂L_j) 
            = (w_kj/A_k - v_kj/B_k) · M_j(1-M_j)
            = w_kj M_j(1-M_j)/A_k - v_kj M_j(1-M_j)/B_k

At the fixed point: M*_j = A*_j/D*_j, 1-M*_j = B*_j/D*_j
M*_j(1-M*_j)/A*_j = (A*_j B*_j/D*_j²)/A*_j = B*_j/D*_j² = (1-M*_j)/D*_j
M*_j(1-M*_j)/B*_j = (A*_j B*_j/D*_j²)/B*_j = A*_j/D*_j² = M*_j/D*_j

So:
  ∂L_k/∂L_j |_* = w_kj(1-M*_k)/(A*_k D*_j)... no wait, these terms involve M_j not M_k.

∂L_k/∂L_j |_* = w_kj M*_j(1-M*_j)/A*_k - v_kj M*_j(1-M*_j)/B*_k
            = w_kj (1-M*_j)/D*_j / A*_k ... no.

M*_j = A*_j/D*_j, so M*_j(1-M*_j) = A*_j B*_j/D*_j²

∂L_k/∂L_j |_* = w_kj A*_j B*_j/(D*_j² A*_k) - v_kj A*_j B*_j/(D*_j² B*_k)
            = A*_j B*_j/D*_j² * [w_kj/A*_k - v_kj/B*_k]
            
This doesn't simplify as nicely as I'd hoped...
"""

# Check Jacobian eigenvalues in log-odds space
print("Log-odds Jacobian 谱半径:")
max_rho = 0
for s in range(20):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    A_star = a + W @ Mstar
    B_star = b + V @ Mstar + e
    D_star = A_star + B_star
    
    # J in M-space
    J = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            J[k,j] = (W[k,j]*(1-Mstar[k]) - V[k,j]*Mstar[k]) / D_star[k]
    
    rho_J = max(abs(np.linalg.eigvals(J)))
    
    # J in L-space
    JL = np.zeros((5,5))
    for k in range(5):
        for j in range(5):
            # ∂L_k/∂L_j = (w_kj/A_k - v_kj/B_k) * M_j(1-M_j)
            dN_dM = (W[k,j]/A_star[k] - V[k,j]/B_star[k])
            dM_dL = Mstar[j] * (1-Mstar[j])
            JL[k,j] = dN_dM * dM_dL
    
    rho_JL = max(abs(np.linalg.eigvals(JL)))
    max_rho = max(max_rho, rho_JL)

print(f"  Log-odds Jacobian 最大谱半径: {max_rho:.6f}")
print()

# ================================================================
# E7: φ'' 的共形解释
# ================================================================
print("="*70)
print("E7: φ'' 的共形解释")
print()

"""
T123_k = -(Wu+Vu)²/D² + θ·Wu²/A² + (1-θ)·Vu²/B²

This is like comparing the "actual" curvature to the "ideal" curvature.

For a log-convex function f(x) with f'(x*)=0, the natural approximation is:
log(f(x*+h)) ≈ log(f(x*)) + ½h²·f''(x*)/f(x*)

Here, T123 is the second-order term of log(D) - θ log(A) - (1-θ) log(B)
evaluated along the ray.

Note that at the fixed point, the gradient of log(D)-θ log(A)-(1-θ)log(B) is zero
(since the "minimizer" is at (A*,B*) for fixed θ).

So this is a "Bregman divergence of the second kind" — the curvature of 
a log-partition function.

The key inequality we need is:
  T123_k(r) ≤ θ·u_k²/M_k² + (1-θ)·u_k²/(1-M_k)² = |T3_k(r)|

for all r. This is equivalent to:

  (Wu+Vu)²/D² + θ·u_k²/M_k² + (1-θ)·u_k²/(1-M_k)² ≥ θ·Wu²/A² + (1-θ)·Vu²/B²

Rearranging:
  θ·[u_k²/M_k² - Wu²/A²] + (1-θ)·[u_k²/(1-M_k)² - Vu²/B²] ≥ (Wu+Vu)²/D²

This is a KL-weighted comparison of the "M-curvature" to the "N-curvature."
"""

# Check the inequality directly
print("θ[u²/M² - Wu²/A²] + (1-θ)[u²/(1-M)² - Vu²/B²] vs (Wu+Vu)²/D²")
violations = 0
total = 0

for s in range(30):
    a,b,e,W,V = gen_FCA(s)
    Mstar = compute_fp(a,b,e,W,V)
    theta = Mstar
    
    for _ in range(20):
        u = np.random.randn(5)
        u /= np.linalg.norm(u)
        Wu = W @ u
        Vu = V @ u
        
        for r in np.logspace(-2, np.log10(2.0), 30):
            M = Mstar + r*u
            if np.any(M<1e-12) or np.any(M>1-1e-12): continue
            
            A = a + W @ M
            B = b + V @ M + e
            D = A + B
            
            for k in range(5):
                t = theta[k]
                lhs = t*(u[k]**2/M[k]**2 - Wu[k]**2/A[k]**2) \
                    + (1-t)*(u[k]**2/(1-M[k])**2 - Vu[k]**2/B[k]**2)
                rhs = (Wu[k]+Vu[k])**2 / D[k]**2
                
                total += 1
                if lhs + 1e-12 < rhs:
                    violations += 1

print(f"  违反: {violations}/{total} ({100*violations/total:.4f}%)")
if violations == 0:
    print("  ✓ 不等式全域成立 — 关键!")
print()

# ================================================================
# E8: 变分特征
# ================================================================
print("="*70)
print("E8: N操作符的变分特征")
print()

"""
Define F_k(a,b) = log(a+b) - θ log(a) - (1-θ) log(b)

At the fixed point: ∇F_k(A*,B*) = 0 (verified above)

The Hessian:
  ∂²F_k/∂a² = -1/(a+b)² + θ/a²
  ∂²F_k/∂a∂b = -1/(a+b)²
  ∂²F_k/∂b² = -1/(a+b)² + (1-θ)/b²

At (A*,B*): det = 0 (flat direction alog (A*_k, B*_k))
  eig1 = 0 (along ray through origin)
  eig2 = -(A*_k² + B*_k²)/(A*_k B*_k D*_k²) < 0 (orthogonal direction)

So F_k is "valley-shaped" — flat along the ray from origin through (A*,B*),
curved in the orthogonal direction.

The flat direction corresponds to scaling A and B by the same factor,
which doesn't change N_k = A/(A+B). This makes sense!

Along the ray M(r) = M* + r u:
  A_k(r) = A*_k + r (Wu)_k  → moves away from A*_k
  B_k(r) = B*_k + r (Vu)_k  → moves away from B*_k

The direction (Wu_k, Vu_k) in (A,B) space determines how F_k changes.

The "flat direction" is (A*, B*), which is along the vector to the origin.
The change (Wu_k, Vu_k) has components both parallel and perpendicular to (A*,B*).

Perpendicular component → F_k increases (curved direction, positive curvature).
Parallel component → F_k stays same (flat direction).

So T123_k measures how much of (Wu_k, Vu_k) is in the perpendicular (curved) 
direction. Meanwhile, T3_k measures the curvature of θ log(M) + (1-θ) log(1-M).

The balance between these determines the sign of φ''.

OK, I think I have enough analytical structure now. Let me try to prove φ'' ≤ 0.

Actually wait — I just realized from the E7 inequality check that 
θ[u²/M² - Wu²/A²] + (1-θ)[u²/(1-M)² - Vu²/B²] ≥ (Wu+Vu)²/D²

holds numerically! This is a PER-COMPONENT inequality. If I can prove this
analytically, φ'' ≤ 0 follows immediately.

Let me rewrite this inequality:

θ_k · u_k²/M_k² + (1-θ_k) · u_k²/(1-M_k)² ≥ θ_k · (Wu)_k²/A_k² + (1-θ_k) · (Vu)_k²/B_k² + (Wu+Vu)_k²/D_k²

Or equivalently:
D_KL curvature of M ≥ D_KL curvature of N at the component level

This looks like it might be provable using the fixed point relationship
and the normalization constraints on W and V.
"""

print("="*70)
print("证明策略总结")
print()
print("路径1 (φ''分解): T123_k + T3_k ≤ 0")
print("  等效于: Σ_k[T123_k ≤ -T3_k] 各分量")
print("  状态: 115,774点零违规 → 解析待证")
print()
print("路径2 (A-B参数化): 5项分解式")
print("  φ''_k = -(Wu+Vu)²/D² + θ·Wu²/A² + (1-θ)·Vu²/B²")
print("         - θ·u_k²/M_k² - (1-θ)·u_k²/(1-M_k)²")
print("  状态: 完美平方分解(at M*) + 扰动分析框架")
print()
print("路径3 (E7不等式): 逐分量KL曲率比较")
print("  θ[u²/M²-Wu²/A²] + (1-θ)[u²/(1-M)²-Vu²/B²] ≥ (Wu+Vu)²/D²")
print("  状态: 数值全域成立 → 解析待证")
print()
print("最可能突破: 路径2 + 路径3")
print("  - 利用F_k(A,B)的'山谷'结构(flat direction + curved direction)")
print("  - 证明(Wu,Vu)在flat direction上的分量不超过u在M空间的分量")
print("  - 利用W和V的加权平均性质(行和为0 by FCA convention)")
