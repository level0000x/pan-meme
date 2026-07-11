"""
B1-B3: 断裂边界实验
==================
主动制造反例，观察 5 种断裂类型（A-E）是否按理论预测出现。

B1 噪声注入 → A 型随机断裂：
  向 M 向量添加高斯噪声 N(0, σ²)，逐渐增大 σ，观察 τ⁻¹ 单调性何时失效。
  预测：断裂出现在 σ > σ_crit，且断边随机分布、无空间模式。

B2 偏序破坏 → B 型层级断裂：
  随机打乱某一层级中所有概念的父子关系，观察该层级的 τ⁻¹ 是否系统性失效。
  预测：断边集中在该层级的 Hasse 边上。

B3 循环引入 → C 型方向断裂：
  在某条 Hasse 边上强制添加反向边（形成 2-循环），观察双向信息流。
  预测：τ⁻¹ 在循环两侧都单调 → 信息流循环 → 可逆性失效。
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np

_EPS = 1e-12
_MAX_ITER = 1000


def n_operator(M, b_up, rho_up, params, noise_std=0.0):
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

    D_new = (a1 * R + eps) / den_d if den_d > 0 else 0.0
    B_new = (g1 * (R + b_up) + eps) / den_b if den_b > 0 else 0.0
    rho_new = (z1 * (D + rho_up) + eps) / den_rho if den_rho > 0 else 0.0
    R_new = (t1 * (rho + rho_up + b_up) + eps) / den_r if den_r > 0 else 0.0
    S_new = (l1 * D + eps) / den_s if den_s > 0 else 0.0

    M_new = np.array([D_new, B_new, rho_new, R_new, S_new])
    if noise_std > 0:
        M_new += np.random.randn(5) * noise_std
        M_new = np.clip(M_new, 0.01, 0.99)
    return M_new


def run_to_fixed_point(m0, b_up, rho_up, params, noise_std=0.0):
    M = m0.copy()
    for _ in range(_MAX_ITER):
        M_new = n_operator(M, b_up, rho_up, params, noise_std)
        if np.max(np.abs(M_new - M)) < _EPS:
            return M_new
        M = M_new
    return M


def rho_up_0(rho_up_0):
    return rho_up_0


def run_lattice_baseline(lattice, params):
    """对格运行干净的基线实验（无噪声，无破坏）。"""
    edges = [(p, c) for p, c in lattice["edges"]]
    concept_sizes = lattice["concept_sizes"]
    d_values = lattice["d_values"]
    n_concepts = len(concept_sizes)

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

    results = [None] * n_concepts
    for ci in topo_order:
        cs = concept_sizes[ci]
        raw_d = d_values[ci]
        d_init = (raw_d / max_d) if (raw_d != float("inf") and raw_d < 1e6 and max_d > 0) else 0.8
        d_init = min(d_init, 1.0)
        b_init = max(0.0, min(1.0, 1.0 - cs["|B|"] / max(total_extent, 1)))
        rho_init = max(0.0, min(1.0, cs["|A|"] / max(total_intent, 1)))
        m0 = np.array([d_init, b_init, rho_init, 0.5, 0.5])

        b_up = 0.0
        rho_up = 0.0
        cnt = 0
        for p in hasse_parents[ci]:
            if results[p] is not None:
                m_star_p = results[p][0]
                b_up += m_star_p[1]
                rho_up += m_star_p[2]
                cnt += 1
        if cnt > 0:
            b_up /= cnt
            rho_up /= cnt

        m_star = run_to_fixed_point(m0, b_up, rho_up, params)
        results[ci] = (m_star, b_up, rho_up)

    return results, edges, hasse_parents, heights, topo_order


def check_theorem_11_1(results, edges):
    """验证 Theorem 11.1: τ⁻¹ 沿 Hasse 边单调递减。"""
    violations = []
    for p, c in edges:
        m_p = results[p][0]
        m_c = results[c][0]
        tau_inv_p = 1.0 / (np.sum(m_p) + 0.01)
        tau_inv_c = 1.0 / (np.sum(m_c) + 0.01)
        if tau_inv_p < tau_inv_c:
            violations.append((p, c, tau_inv_p, tau_inv_c))
    return violations


def make_topological_labels(n_concepts, edges):
    """为每个概念分配拓扑层级（0=根, N-1=最深的叶）。"""
    parents = [set() for _ in range(n_concepts)]
    for p, c in edges:
        parents[c].add(p)
    heights = np.zeros(n_concepts, dtype=int)
    changed = True
    while changed:
        changed = False
        for i in range(n_concepts):
            if parents[i]:
                new_h = max(heights[p] for p in parents[i]) + 1
                if new_h != heights[i]:
                    heights[i] = new_h
                    changed = True
    return heights


def experiment_b1_noise(lattice, params, noise_levels):
    """B1: 噪声注入实验"""
    print("  B1: 噪声注入 — 搜索 A 型断裂阈值")
    results = {}
    for sigma in noise_levels:
        np.random.seed(42)
        params_noisy = dict(params)
        base, edges, parents, heights, topo = run_lattice_baseline(lattice, params)

        total_extent = sum(cs["|A|"] for cs in lattice["concept_sizes"])
        total_intent = sum(cs["|B|"] for cs in lattice["concept_sizes"])
        d_values = lattice["d_values"]
        concept_sizes = lattice["concept_sizes"]
        n_concepts = len(concept_sizes)
        valid_d = [d for d in d_values if d != float("inf") and d < 1e6]
        max_d = max(valid_d) if valid_d else 1.0

        noisy_results = [None] * n_concepts
        for ci in topo:
            cs = concept_sizes[ci]
            raw_d = d_values[ci]
            d_init = (raw_d / max_d) if (raw_d != float("inf") and raw_d < 1e6 and max_d > 0) else 0.8
            d_init = min(d_init, 1.0)
            b_init = max(0.0, min(1.0, 1.0 - cs["|B|"] / max(total_extent, 1)))
            rho_init = max(0.0, min(1.0, cs["|A|"] / max(total_intent, 1)))
            m0 = np.array([d_init, b_init, rho_init, 0.5, 0.5])

            b_up = 0.0
            rho_up = 0.0
            cnt = 0
            for p in parents[ci]:
                if noisy_results[p] is not None:
                    m_star_p = noisy_results[p][0]
                    b_up += m_star_p[1]
                    rho_up += m_star_p[2]
                    cnt += 1
            if cnt > 0:
                b_up /= cnt
                rho_up /= cnt

            m_star = run_to_fixed_point(m0, b_up, rho_up, params, noise_std=sigma)
            noisy_results[ci] = (m_star, b_up, rho_up)

        violations = check_theorem_11_1(noisy_results, edges)
        labels = make_topological_labels(n_concepts, edges)
        label_set = set(labels[v[0]] for v in violations)

        results[sigma] = {
            "n_violations": len(violations),
            "n_edges": len(edges),
            "violation_rate": len(violations) / len(edges) if edges else 0,
            "affected_levels": sorted(label_set),
            "random_pattern": (len(label_set) > 1 if len(violations) > 1 else True),
        }
        print(f"    σ={sigma:.3f}: {len(violations)}/{len(edges)} 断裂 "
              f"({100.0*len(violations)/max(len(edges),1):.1f}%) "
              f"层级={label_set}")

    return results


def experiment_b2_order_disruption(lattice, params, target_level, n_disrupt=2):
    """B2: 偏序破坏 — 目标层级上随机重组父子关系"""
    print(f"  B2: 偏序破坏 — 层级 {target_level}")
    edges = [(p, c) for p, c in lattice["edges"]]
    n_concepts = len(lattice["concept_sizes"])
    labels = make_topological_labels(n_concepts, edges)
    concept_sizes = lattice["concept_sizes"]
    d_values = lattice["d_values"]

    target_concepts = [i for i in range(n_concepts) if labels[i] == target_level]
    other_concepts = [i for i in range(n_concepts) if labels[i] != target_level]
    target_set = set(target_concepts)

    disrupted_edges = []
    orig_edges = set(edges)
    new_edges = set(edges)
    rng = np.random.RandomState(123)

    for ci in target_concepts:
        old_parents = [p for p, c in edges if c == ci]
        for p in old_parents:
            new_edges.discard((p, ci))
        possible_new = [oc for oc in other_concepts if labels[oc] < labels[ci]]
        if possible_new:
            new_parents = list(rng.choice(possible_new, size=min(n_disrupt, len(possible_new)), replace=False))
            for np_p in new_parents:
                if (np_p, ci) not in orig_edges:
                    disrupted_edges.append((np_p, ci))
                new_edges.add((np_p, ci))

    disrupted_edges_list = list(new_edges)

    total_extent = sum(cs["|A|"] for cs in concept_sizes)
    total_intent = sum(cs["|B|"] for cs in concept_sizes)
    valid_d = [d for d in d_values if d != float("inf") and d < 1e6]
    max_d = max(valid_d) if valid_d else 1.0

    parents_disrupted = [set() for _ in range(n_concepts)]
    for p, c in disrupted_edges_list:
        parents_disrupted[c].add(p)
    heights_d = np.zeros(n_concepts, dtype=int)
    changed = True
    while changed:
        changed = False
        for i in range(n_concepts):
            if parents_disrupted[i]:
                new_h = max(heights_d[p] for p in parents_disrupted[i]) + 1
                if new_h != heights_d[i]:
                    heights_d[i] = new_h
                    changed = True
    topo_d = sorted(range(n_concepts), key=lambda i: -heights_d[i])

    results = [None] * n_concepts
    for ci in topo_d:
        cs = concept_sizes[ci]
        raw_d = d_values[ci]
        d_init = (raw_d / max_d) if (raw_d != float("inf") and raw_d < 1e6 and max_d > 0) else 0.8
        d_init = min(d_init, 1.0)
        b_init = max(0.0, min(1.0, 1.0 - cs["|B|"] / max(total_extent, 1)))
        rho_init = max(0.0, min(1.0, cs["|A|"] / max(total_intent, 1)))
        m0 = np.array([d_init, b_init, rho_init, 0.5, 0.5])

        b_up = 0.0
        rho_up = 0.0
        cnt = 0
        for p in parents_disrupted[ci]:
            if results[p] is not None:
                m_star_p = results[p][0]
                b_up += m_star_p[1]
                rho_up += m_star_p[2]
                cnt += 1
        if cnt > 0:
            b_up /= cnt
            rho_up /= cnt

        m_star = run_to_fixed_point(m0, b_up, rho_up, params)
        results[ci] = (m_star, b_up, rho_up)

    violations = check_theorem_11_1(results, disrupted_edges_list)
    target_edges_violated = [v for v in violations
                              if v[0] in target_set or v[1] in target_set]

    layer_concentration = len(target_edges_violated) / len(violations) if violations else 0.0
    print(f"    总断裂: {len(violations)}/{len(disrupted_edges_list)} "
          f"目标层断裂: {len(target_edges_violated)} "
          f"集中度: {layer_concentration:.2f} "
          f"({'B 型 ✓' if layer_concentration > 0.5 else '非 B 型'})")

    return {
        "n_violations": len(violations),
        "n_edges": len(disrupted_edges_list),
        "target_level": target_level,
        "target_edges_violated": len(target_edges_violated),
        "layer_concentration": layer_concentration,
        "b_type_confirmed": layer_concentration > 0.5,
    }


def experiment_b3_cycle_injection(lattice, params):
    """B3: 循环引入 — 在一条 Hasse 边上添加反向边"""
    print("  B3: 循环引入 — 搜索 C 型断裂")
    edges = [(p, c) for p, c in lattice["edges"]]
    n_concepts = len(lattice["concept_sizes"])

    if not edges:
        print("    无 Hasse 边，跳过")
        return []

    middle_idx = len(edges) // 2
    p_test, c_test = edges[middle_idx]
    print(f"    测试边: {p_test} → {c_test}")

    parents_orig = [set() for _ in range(n_concepts)]
    for p, c in edges:
        parents_orig[c].add(p)

    concept_sizes = lattice["concept_sizes"]
    d_values = lattice["d_values"]
    total_extent = sum(cs["|A|"] for cs in concept_sizes)
    total_intent = sum(cs["|B|"] for cs in concept_sizes)
    valid_d = [d for d in d_values if d != float("inf") and d < 1e6]
    max_d = max(valid_d) if valid_d else 1.0

    heights = np.zeros(n_concepts, dtype=int)
    changed = True
    while changed:
        changed = False
        for i in range(n_concepts):
            if parents_orig[i]:
                new_h = max(heights[p] for p in parents_orig[i]) + 1
                if new_h != heights[i]:
                    heights[i] = new_h
                    changed = True

    results_no_cycle = []
    for bw in [False, True]:
        is_cycle = bw
        parents_use = [set(s) for s in parents_orig]
        if is_cycle:
            parents_use[p_test].add(c_test)

        heights_use = np.zeros(n_concepts, dtype=int)
        changed = True
        while changed:
            changed = False
            for i in range(n_concepts):
                if parents_use[i]:
                    new_h = max(heights_use[p] for p in parents_use[i]) + 1
                    if new_h != heights_use[i]:
                        heights_use[i] = new_h
                        changed = True

        topo = sorted(range(n_concepts), key=lambda i: -heights_use[i])
        results = [None] * n_concepts

        for ci in topo:
            cs = concept_sizes[ci]
            raw_d = d_values[ci]
            d_init = (raw_d / max_d) if (raw_d != float("inf") and raw_d < 1e6 and max_d > 0) else 0.8
            d_init = min(d_init, 1.0)
            b_init = max(0.0, min(1.0, 1.0 - cs["|B|"] / max(total_extent, 1)))
            rho_init = max(0.0, min(1.0, cs["|A|"] / max(total_intent, 1)))
            m0 = np.array([d_init, b_init, rho_init, 0.5, 0.5])

            b_up = 0.0
            rho_up = 0.0
            cnt = 0
            for p in parents_use[ci]:
                if results[p] is not None:
                    m_star_p = results[p][0]
                    b_up += m_star_p[1]
                    rho_up += m_star_p[2]
                    cnt += 1
            if cnt > 0:
                b_up /= cnt
                rho_up /= cnt

            m_star = run_to_fixed_point(m0, b_up, rho_up, params)
            results[ci] = (m_star, b_up, rho_up)

        results_no_cycle.append(results)

    m_p_nocycle = results_no_cycle[0][p_test][0]
    m_c_nocycle = results_no_cycle[0][c_test][0]
    m_p_cycle = results_no_cycle[1][p_test][0]
    m_c_cycle = results_no_cycle[1][c_test][0]

    tau_inv_p_no = 1.0 / (np.sum(m_p_nocycle) + 0.01)
    tau_inv_c_no = 1.0 / (np.sum(m_c_nocycle) + 0.01)
    tau_inv_p_cy = 1.0 / (np.sum(m_p_cycle) + 0.01)
    tau_inv_c_cy = 1.0 / (np.sum(m_c_cycle) + 0.01)

    delta_direction_p = tau_inv_p_cy - tau_inv_c_cy
    c_type = (tau_inv_p_no >= tau_inv_c_no
              and tau_inv_p_cy > tau_inv_c_cy
              and abs(delta_direction_p) > 1e-3)

    diff_p = np.max(np.abs(m_p_cycle - m_p_nocycle))
    diff_c = np.max(np.abs(m_c_cycle - m_c_nocycle))

    print(f"    τ⁻¹(p) no-cycle={tau_inv_p_no:.4f} cycle={tau_inv_p_cy:.4f} Δ={tau_inv_p_cy-tau_inv_p_no:.4f}")
    print(f"    τ⁻¹(c) no-cycle={tau_inv_c_no:.4f} cycle={tau_inv_c_cy:.4f} Δ={tau_inv_c_cy-tau_inv_c_no:.4f}")
    print(f"    M* 偏移 p: {diff_p:.2e}  c: {diff_c:.2e}")
    print(f"    C 型断裂: {'检测到 ✓' if c_type else '未检测到'}")

    return [{
        "edge": (int(p_test), int(c_test)),
        "tau_inv_no_cycle": {"p": float(tau_inv_p_no), "c": float(tau_inv_c_no)},
        "tau_inv_cycle": {"p": float(tau_inv_p_cy), "c": float(tau_inv_c_cy)},
        "m_star_shift": {"p": float(diff_p), "c": float(diff_c)},
        "c_type_fracture": bool(c_type),
    }]


def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    lattice_files = list(sorted(results_dir.glob("*_lattice.json")))

    params = {
        "a1": 1.0, "b1": 1.0, "g1": 1.0, "d1": 1.0,
        "z1": 1.0, "e1": 1.0, "t1": 1.0,
        "k1": 1.0, "k2": 1.0, "l1": 1.0, "m1": 1.0,
        "eps": 0.01,
    }

    test_concepts = ["DNA", "Bitcoin", "Democracy", "Christmas"]
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]

    print("=" * 70)
    print("断裂边界实验: B1（噪声）+ B2（偏序破坏）+ B3（循环引入）")
    print("=" * 70)

    all_b1 = {}
    all_b2 = {}
    all_b3 = {}

    for name in test_concepts:
        lf = results_dir / f"{name}_lattice.json"
        if not lf.exists():
            print(f"\n跳过 {name}: 文件不存在")
            continue

        with open(lf, "r", encoding="utf-8") as f:
            lattice = json.load(f)
        if "edges" not in lattice:
            print(f"\n跳过 {name}: 无 edges 字段")
            continue

        print(f"\n{'='*50}")
        print(f"概念: {name}  ({lattice['n_concepts']} 概念, {len(lattice['edges'])} 边)")
        print(f"{'='*50}")

        b1_result = experiment_b1_noise(lattice, params, noise_levels)
        all_b1[name] = b1_result

        labels = make_topological_labels(lattice["n_concepts"],
                                         [(p, c) for p, c in lattice["edges"]])
        max_level = max(labels) if len(labels) > 0 else 0
        if max_level >= 1:
            b2_result = experiment_b2_order_disruption(lattice, params, max_level)
            all_b2[name] = b2_result
        else:
            print(f"  B2: 跳过（层级数不足）")

        b3_result = experiment_b3_cycle_injection(lattice, params)
        all_b3[name] = b3_result

    print("\n" + "=" * 70)
    print("B1 汇总：噪声注入断裂阈值")
    print("-" * 70)
    for name, b1d in all_b1.items():
        thresholds = []
        for sigma in noise_levels:
            if sigma in b1d and b1d[sigma]["violation_rate"] > 0:
                thresholds.append(sigma)
        thresh_str = f"σ_crit ≈ {min(thresholds):.3f}" if thresholds else "σ_crit > 0.30"
        print(f"  {name:<20} {thresh_str}")

    print(f"\nB2 汇总：偏序破坏")
    print("-" * 70)
    for name, b2d in all_b2.items():
        print(f"  {name:<20} {'B 型确认 ✓' if b2d.get('b_type_confirmed') else '未确认'} "
              f"(集中度={b2d.get('layer_concentration', 0):.2f})")

    print(f"\nB3 汇总：循环引入")
    print("-" * 70)
    for name, b3r in all_b3.items():
        for r in b3r:
            print(f"  {name:<20} 边({r['edge'][0]}→{r['edge'][1]}) "
                  f"{'C 型确认 ✓' if r['c_type_fracture'] else 'C 型未触发'} "
                  f"Δτ⁻¹={r['tau_inv_cycle']['p']-r['tau_inv_no_cycle']['p']:.4f}")

    summary = {
        "B1_noise": {k: {str(s): v for s, v in d.items()} for k, d in all_b1.items()},
        "B2_order": all_b2,
        "B3_cycle": all_b3,
    }
    summary_path = results_dir / "boundary_fracture_experiments.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {summary_path}")


if __name__ == "__main__":
    main()
