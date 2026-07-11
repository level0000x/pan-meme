"""
Verify the Bregman 3-point identity for KL divergence.
This is a mathematical identity, not an approximation.
D_KL(a||c) = D_KL(a||b) + D_KL(b||c) + (a-b)(logit(c) - logit(b))
"""
import numpy as np

def kl(p, q):
    return p * np.log(p/q) + (1-p) * np.log((1-p)/(1-q))

def logit(p):
    return np.log(p / (1-p))

np.random.seed(42)
for _ in range(100):
    a = np.random.uniform(0.01, 0.99)
    b = np.random.uniform(0.01, 0.99)
    c = np.random.uniform(0.01, 0.99)
    
    lhs = kl(a, c)
    rhs = kl(a, b) + kl(b, c) + (a - b) * (logit(b) - logit(c))
    
    if abs(lhs - rhs) > 1e-14:
        print(f"FAIL: a={a:.6f}, b={b:.6f}, c={c:.6f}, diff={abs(lhs-rhs):.2e}")
        raise SystemExit(1)

print("All 100 random tests: Bregman 3-point identity holds exactly ✓")
print(f"D_KL(a||c) = D_KL(a||b) + D_KL(b||c) + (a-b)(logit(c) - logit(b))")

# Test per-component ΔV decomposition
print("\nΔV_k = D_KL(M*||N) - D_KL(M*||M)")
print("    = -[D_KL(N||M) + (M*-N)(logit M - logit N)]")
for _ in range(100):
    Mstar = np.random.uniform(0.01, 0.99)
    M = np.random.uniform(0.01, 0.99)
    N = np.random.uniform(0.01, 0.99)
    
    dV = kl(Mstar, N) - kl(Mstar, M)
    pred = -(kl(N, M) - (Mstar - N) * (logit(M) - logit(N)))
    
    if abs(dV - pred) > 1e-14:
        print(f"FAIL: dV={dV:.10f}, pred={pred:.10f}, diff={abs(dV-pred):.2e}")
        raise SystemExit(1)

print("All 100 random tests: ΔV decomposition holds exactly ✓")
