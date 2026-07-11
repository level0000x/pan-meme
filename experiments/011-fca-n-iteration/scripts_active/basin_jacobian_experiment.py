"""
E3 + E4: M* 吸引子盆地边界 + 雅可比谱分类
===========================================
E3: 对每个概念，从多个随机初始 M₀ 运行 N-迭代，
    记录收敛到的 M*，分析盆地结构。

E4: 计算主 M* 处的 5×5 雅可比矩阵，
    按断裂类型（A-E + Normal）分类。
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_EPS = 1e-12
_MAX_ITER = 1000


def n_operator(M, b_up, rho_up, params):
    D, B, rho, R, S = M
    eps = params["eps"]
    a1, b1, g1, d1 = params["a1"], params["b1"], params["g1"], params["d1"]
    z1, e1, t1 = params["z1"], params["e1"], params["t1"]
    k1, k2, l1, m1 = params["k1"], params["k2"], params["l1"], params["m1"]

    den_d = a1 * R + b1 * (B + b_up) + eps
    den_b = g1 * (R + b_up) + d1 * D + eps
    den_rho = z1 * (D + rho_up) + e1 * R + eps
    den_r = t1 * (rho + rho_up + b_up) + k1 * D + k2 * S + eps
    den_s = l1 * D + m1 * R + eps

    return np.array([
        (a1 * R + eps) / den_d,
        (g1 * (R + b_up) + eps) / den_b,
        (z1 * (D + rho_up) + eps) / den_rho,
        (t1 * (rho + rho_up + b_up) + eps) / den_r,
        (l1 * D + eps) / den_s,
    ])


def run_to_fixed_point(m0, b_up, rho_up, params):
    M = m0.copy()
    for _ in range(_MAX_ITER):
        M_new = n_operator(M, b_up, rho_up, params)
        if np.max(np.abs(M_new - M)) < _EPS:
            return M_new
        M = M_new
    return M


def compute_jacobian(m_star, b_up, rho_up, params):
    D, B, rho, R, S = m_star
    eps = params["eps"]
    a1, b1, g1, d1 = params["a1"], params["b1"], params["g1"], params["d1"]
    z1, e1, t1 = params["z1"], params["e1"], params["t1"]
    k1, k2, l1, m1 = params["k1"], params["k2"], params["l1"], params["m1"]

    den_d = a1 * R + b1 * (B + b_up) + eps
    den_b = g1 * (R + b_up) + d1 * D + eps
    den_rho = z1 * (D + rho_up) + e1 * R + eps
    den_r = t1 * (rho + rho_up + b_up) + k1 * D + k2 * S + eps
    den_s = l1 * D + m1 * R + eps

    J = np.zeros((5, 5))

    J[0, 1] = -b1 * D / den_d
    J[0, 3] = +a1 * (1.0 - D) / den_d

    J[1, 0] = -d1 * B / den_b
    J[1, 3] = +g1 * (1.0 - B) / den_b

    J[2, 0] = +z1 * (1.0 - rho) / den_rho
    J[2, 3] = -e1 * rho / den_rho

    J[3, 0] = -k1 * R / den_r
    J[3, 1] = +t1 * (1.0 - R) / den_r
    J[3, 2] = +t1 * (1.0 - R) / den_r
    J[3, 4] = -k2 * R / den_r

    J[4, 0] = +l1 * (1.0 - S) / den_s
    J[4, 3] = -m1 * S / den_s

    return J


def classify_jacobian(J):
    """按断裂类型分类 5x5 雅可比矩阵。"""
    eigenvals = np.linalg.eigvals(J)
    rho = max(abs(ev) for ev in eigenvals)
    real_ev = [ev.real for ev in eigenvals if abs(ev.imag) < 1e-10]
    complex_pairs = sum(1 for ev in eigenvals if abs(ev.imag) > 1e-10)

    nz = np.sum(np.abs(J) > 1e-10)

    if rho >= 1.0:
        if nz <= 3:
            return "D-global (unstable, sparse)"
        elif complex_pairs >= 2:
            return "C-cycle (unstable, oscillatory)"
        else:
            return "D-global (unstable)"
    elif rho > 0.99:
        return "B-marginal (near-critical)"
    elif complex_pairs >= 2:
        if rho < 0.5:
            return "Normal (damped oscillatory)"
        else:
            return "Normal (oscillatory convergence)"
    elif nz <= 3:
        return "A-sparse (low coupling)"
    elif len(real_ev) == 5 and all(ev > 0 for ev in real_ev):
        return "Normal (monotonic)"
    elif rho < 0.3:
        return "A-weak (rapid convergence)"
    else:
        return "Normal (mixed)"


def analyze_basin(m_star_samples, tol=1e-6):
    """分析多个 M* 样本的盆地结构。"""
    if len(m_star_samples) <= 1:
        return {"n_unique": 1, "is_degenerate": False, "max_pairwise_diff": 0.0}

    ref = m_star_samples[0]
    diffs = [np.max(np.abs(m - ref)) for m in m_star_samples[1:]]
    max_diff = max(diffs)

    n_unique = 1
    if max_diff > tol:
        clusters = []
        for m in m_star_samples:
            found = False
            for c in clusters:
                if np.max(np.abs(m - c["center"])) < tol:
                    c["count"] += 1
                    found = True
                    break
            if not found:
                clusters.append({"center": m.copy(), "count": 1})
        n_unique = len(clusters)

    return {
        "n_unique": n_unique,
        "is_degenerate": n_unique > 1,
        "max_pairwise_diff": float(max_diff),
    }


def run_concept_basin_experiment(lattice, params, n_samples=20):
    """对单个概念运行盆地 + 雅可比实验。"""
    edges = [(p, c) for p, c in lattice["edges"]]
    concept_sizes = lattice["concept_sizes"]
    d_values = lattice["d_values"]
    n_concepts = len(concept_sizes)

    from collections import defaultdict

    hasse_parents = [set() for _ in range(n_concepts)]
    for p, c in edges:
        hasse_parents[c].add(p)

    heights = np.zeros(n_concepts, dtype=int)
    changed = True
    while changed:
        changed = False
        for i in range(n_concepts):
            if hasse_parents[i]:
                new_h = max(heights[p] for p in hasse_parents[i]) + 1
                if new_h != heights[i]:
                    heights[i] = new_h
                    changed = True
    topo_order = sorted(range(n_concepts), key=lambda i: -heights[i])

    total_extent = sum(cs["|A|"] for cs in concept_sizes)
    total_intent = sum(cs["|B|"] for cs in concept_sizes)
    valid_d = [d for d in d_values if d != float("inf") and d < 1e6]
    max_d = max(valid_d) if valid_d else 1.0

    results = []
    rng = np.random.RandomState(42)

    for ci in topo_order:
        cs = concept_sizes[ci]
        raw_d = d_values[ci]
        d_init = (raw_d / max_d) if (raw_d != float("inf") and raw_d < 1e6 and max_d > 0) else 0.8
        d_init = min(d_init, 1.0)
        b_init = max(0.0, min(1.0, 1.0 - cs["|B|"] / max(total_extent, 1)))
        rho_init = max(0.0, min(1.0, cs["|A|"] / max(total_intent, 1)))

        m0_base = np.array([d_init, b_init, rho_init, 0.5, 0.5])

        parents = hasse_parents[ci]
        b_up = 0.0
        rho_up = 0.0
        cnt = 0
        for p in parents:
            if p < len(results) and results[p] is not None:
                m_p = results[p]["m_star"]
                b_up += m_p[1]
                rho_up += m_p[2]
                cnt += 1
        if cnt > 0:
            b_up /= cnt
            rho_up /= cnt

        m_stars = []
        for s in range(n_samples):
            if s == 0:
                m0 = m0_base
            else:
                noise = rng.uniform(-0.15, 0.15, 5)
                m0 = np.clip(m0_base + noise, 0.01, 0.99)

            m_star = run_to_fixed_point(m0, b_up, rho_up, params)
            m_stars.append(m_star)

        basin_info = analyze_basin(m_stars)

        primary_m_star = m_stars[0]
        J = compute_jacobian(primary_m_star, b_up, rho_up, params)
        eigenvals = np.linalg.eigvals(J)
        rho_J = max(abs(ev) for ev in eigenvals)

        jac_class = classify_jacobian(J)
        tau_inv = 1.0 / (np.sum(primary_m_star) + params["eps"])

        results.append({
            "concept_idx": ci,
            "height": int(heights[ci]),
            "m_star": primary_m_star.tolist(),
            "tau_inv": float(tau_inv),
            "rho_J": float(rho_J),
            "eigenvalues": [{"real": float(ev.real), "imag": float(ev.imag)} for ev in eigenvals],
            "jacobian_class": jac_class,
            "n_basins": basin_info["n_unique"],
            "is_degenerate": basin_info["is_degenerate"],
            "max_basin_diff": basin_info["max_pairwise_diff"],
            "jacobian": J.tolist(),
        })

    return results


def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    lattice_files = sorted(results_dir.glob("*_lattice.json"))

    params = {
        "a1": 1.0, "b1": 1.0, "g1": 1.0, "d1": 1.0,
        "z1": 1.0, "e1": 1.0, "t1": 1.0,
        "k1": 1.0, "k2": 1.0, "l1": 1.0, "m1": 1.0,
        "eps": 0.01,
    }

    print("E3+E4: 吸引子盆地 + 雅可比谱分类")
    print("=" * 90)
    N_SAMPLES = 20

    all_concept_results = []
    basin_summary = {"global": 0, "single": 0, "multi": 0}
    jacobian_summary = {}

    for lf in lattice_files:
        try:
            with open(lf, "r", encoding="utf-8") as f:
                lattice = json.load(f)
            if "edges" not in lattice or lattice["n_concepts"] < 2:
                continue
        except (KeyError, json.JSONDecodeError):
            continue

        concept_name = lattice["concept_name"]
        per_concept = run_concept_basin_experiment(lattice, params, n_samples=N_SAMPLES)

        n_degenerate = sum(1 for r in per_concept if r["is_degenerate"])
        classes = [r["jacobian_class"] for r in per_concept]

        if n_degenerate == 0:
            basin_summary["single"] += 1
        elif n_degenerate == len(per_concept):
            basin_summary["global"] += 1
        else:
            basin_summary["multi"] += 1

        for c in classes:
            jacobian_summary[c] = jacobian_summary.get(c, 0) + 1

        all_concept_results.append({
            "concept": concept_name,
            "n_concepts": lattice["n_concepts"],
            "per_concept": per_concept,
        })

        print(f"{concept_name:<30} 概念={lattice['n_concepts']}  盆地唯一={n_degenerate==0}  "
              f"雅可比=[{', '.join(c[:12] for c in classes)}]")

    print()
    print("-" * 90)
    print(f"总计: {len(all_concept_results)} 概念")
    print(f"\nE3 盆地结构:")
    print(f"  全概念单一盆地: {basin_summary['single']}/{len(all_concept_results)} = "
          f"{100.0*basin_summary['single']/max(len(all_concept_results),1):.0f}%")
    print(f"  存在多重盆地:   {basin_summary['multi']}/{len(all_concept_results)}")

    if basin_summary["single"] == len(all_concept_results):
        print("  VERDICT: 全局单一吸引子 — M* 是全局稳定的")

    print(f"\nE4 雅可比谱分类:")
    for cls, count in sorted(jacobian_summary.items(), key=lambda x: -x[1]):
        print(f"  {cls:<40} {count:>4} ({100.0*count/max(sum(jacobian_summary.values()),1):.1f}%)")

    normal_count = sum(v for k, v in jacobian_summary.items() if k.startswith("Normal"))
    fracture_count = sum(v for k, v in jacobian_summary.items() if not k.startswith("Normal"))
    print(f"\n  正常: {normal_count}  断裂: {fracture_count}")
    if fracture_count == 0:
        print("  VERDICT: 全部雅可比谱分类为 Normal — 无断裂特征")

    summary_path = results_dir / "e3_e4_basin_jacobian.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "e3_basin": basin_summary,
            "e4_jacobian": jacobian_summary,
            "details": all_concept_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {summary_path}")


if __name__ == "__main__":
    main()
