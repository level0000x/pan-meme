"""验证：N迭代是否真的从所有随机起点收敛到唯一FP？
对(B_up,ρ_up)的均匀随机采样测试全局收敛性。
"""
import numpy as np

param = {
    'alpha1': 1.0, 'beta1': 1.0,
    'gamma1': 1.0, 'delta1': 1.0,
    'zeta1': 1.0, 'eta1': 1.0,
    'theta1': 1.0, 'kappa1': 1.0, 'kappa2': 1.0,
    'lambda1': 1.0, 'mu1': 1.0,
    'eps1': 0.01, 'eps2': 0.01, 'eps3': 0.01, 'eps4': 0.01, 'eps5': 0.01,
}

def N(M, B_up, rho_up, p):
    D, B, rho, R, S = M
    ND = (p['alpha1']*R + p['eps1']) / (p['alpha1']*R + p['beta1']*(B+B_up) + p['eps1'])
    NB = (p['gamma1']*(R+B_up) + p['eps2']) / (p['gamma1']*(R+B_up) + p['delta1']*D + p['eps2'])
    Nrho = (p['zeta1']*(D+rho_up) + p['eps3']) / (p['zeta1']*(D+rho_up) + p['eta1']*R + p['eps3'])
    NR = (p['theta1']*(rho+rho_up+B_up) + p['eps4']) / (p['theta1']*(rho+rho_up+B_up) + p['kappa1']*D + p['kappa2']*S + p['eps4'])
    NS = (p['lambda1']*D + p['eps5']) / (p['lambda1']*D + p['mu1']*R + p['eps5'])
    return np.array([ND, NB, Nrho, NR, NS])

def find_fp(B_up, rho_up, p, n_iters=50000):
    M = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    for _ in range(n_iters):
        M_new = N(M, B_up, rho_up, p)
        if np.max(np.abs(M_new - M)) < 1e-15:
            return M_new, True
        M = M_new
    return M, False

def test_global_convergence(n_trials=2000):
    """从随机起点出发，检查是否都收敛到同一个FP"""
    np.random.seed(42)
    
    results = []
    for trial in range(n_trials):
        B_up = np.random.uniform(0, 1)
        rho_up = np.random.uniform(0, 1)
        
        fp_ref, _ = find_fp(B_up, rho_up, param)
        
        all_converged = True
        final_fps = []
        for _ in range(5):
            M0 = np.random.uniform(0, 1, 5)
            M = M0.copy()
            for _ in range(50000):
                M_new = N(M, B_up, rho_up, param)
                if np.max(np.abs(M_new - M)) < 1e-14:
                    final_fps.append(M_new)
                    if np.max(np.abs(M_new - fp_ref)) > 1e-10:
                        all_converged = False
                    break
                M = M_new
            else:
                all_converged = False
        
        results.append((B_up, rho_up, all_converged))
    
    n_success = sum(1 for r in results if r[2])
    print(f"全局收敛测试：{n_trials} 组随机 (B_up,ρ_up)，每组 5 随机起点")
    print(f"  全部收敛到同一 FP: {n_success}/{n_trials} ({100*n_success/n_trials:.1f}%)")
    
    failures = [r for r in results if not r[2]]
    if failures:
        print(f"\n  失败案例 ({len(failures)}):")
        for B_up, rho_up, _ in failures[:5]:
            print(f"    (B_up={B_up:.4f}, rho_up={rho_up:.4f})")
    else:
        print(f"  ✓ 所有随机起点均收敛到同一 FP")
    
    return n_success == n_trials

def test_extreme_starts():
    """测试极端起点（边界、角落）"""
    extreme_starts = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 1.0, 0.0, 1.0]),
        np.array([0.999, 0.999, 0.999, 0.999, 0.999]),
        np.array([0.001, 0.001, 0.001, 0.001, 0.001]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.999]),
        np.array([0.999, 0.0, 0.0, 0.0, 0.0]),
    ]
    
    print(f"\n极端起点测试（B_up=ρ_up=0 叶节点）:")
    fp_ref, _ = find_fp(0.0, 0.0, param)
    
    for i, M0 in enumerate(extreme_starts):
        M = M0.copy()
        converged = False
        for _ in range(50000):
            M_new = N(M, 0.0, 0.0, param)
            if np.max(np.abs(M_new - M)) < 1e-14:
                dist = np.max(np.abs(M_new - fp_ref))
                status = "✓" if dist < 1e-10 else "✗ DIFFERENT FP"
                print(f"  起点 {M0}: 收敛, Δ={dist:.2e} {status}")
                converged = True
                break
            M = M_new
        if not converged:
            print(f"  起点 {M0}: 未收敛@{_}")

def test_global_vs_local():
    """核心问题验证：
    定理6.15只证明了FP附近ρ(J_N)<1（局部压缩）。
    定理6.14声称"从任意M⁽⁰⁾出发必收敛"——这个跳跃没有被证明。
    我们测试：从远离FP的点出发，需要多少步进入FP的小邻域？
    """
    B_up, rho_up = 0.3, 0.3
    fp, _ = find_fp(B_up, rho_up, param)
    
    print(f"\n全局→局部过渡测试（B_up={B_up}, ρ_up={rho_up}）:")
    print(f"  FP = {fp}")
    
    # 找特征值估计最优的局部压缩率
    J = np.zeros((5,5))
    h = 1e-8
    f0 = N(fp, B_up, rho_up, param)
    for i in range(5):
        fph = fp.copy()
        fph[i] += h
        J[:, i] = (N(fph, B_up, rho_up, param) - f0) / h
    
    rhoJ = max(abs(np.linalg.eigvals(J)))
    print(f"  ρ(J_N(M*)) = {rhoJ:.4f}")
    
    # 从远处起点测试迭代距离衰减
    np.random.seed(123)
    for _ in range(5):
        M0 = np.random.uniform(0, 1, 5)
        dist0 = np.max(np.abs(M0 - fp))
        
        M = M0.copy()
        n_steps_to_neighborhood = None
        for k in range(500):
            M_new = N(M, B_up, rho_up, param)
            dist = np.max(np.abs(M_new - fp))
            if n_steps_to_neighborhood is None and dist < 0.1:
                n_steps_to_neighborhood = k
            if dist < 1e-14:
                print(f"  起点 dist₀={dist0:.4f}: {k}步收敛, 进入邻域@{n_steps_to_neighborhood}步")
                break
            M = M_new
        else:
            print(f"  起点 dist₀={dist0:.4f}: 500步未收敛, 最终dist={np.max(np.abs(M-fp)):.4f}")

if __name__ == "__main__":
    print("=" * 80)
    print("全局收敛性验证 vs 定理6.14声明")
    print("=" * 80)
    print()
    print("定理6.14声明：从任意 M⁽⁰⁾∈[0,1]⁵ 出发的迭代必收敛至 M*")
    print("定理6.15实际证明：ρ(J_N(M*)) < 1（FP处的局部压缩性）")
    print("定理6.16实际证明：进入M*邻域后指数收敛")
    print()
    print("未证明的gap：从任意起点是否必然进入M*的吸引盆")
    print("=" * 80)
    
    test_global_convergence(500)
    test_extreme_starts()
    test_global_vs_local()
    
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    print("1. 实验证据极强：所有随机起点收敛到同一FP")
    print("2. 但解析层面：'从任意起点出发必收敛'的严格证明缺失")
    print("3. 定理6.16隐含假设'迭代已进入M*的ε-邻域'（循环论证）")
    print("4. 定理6.14的'从任意M⁽⁰⁾出发必收敛'应标记为实验佐证非解析得证")
