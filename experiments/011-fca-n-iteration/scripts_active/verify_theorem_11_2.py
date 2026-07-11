"""
Theorem 11.2 实验验证: E_N = E_H
================================
验证 N-迭代耦合网络的边集恰好等于 FCA 概念格的 Hasse 边集。

策略:
  对每个概念格，分别以两种耦合模式运行 N-迭代:
  (A) Hasse-only: 每个节点只接收其 Hasse 父节点的 B↑/ρ↑
  (B) All-ancestors: 每个节点接收其所有祖先节点的 B↑/ρ↑

  Thm 11.2 预测: Mode A 和 Mode B 产生相同的不动点 M*，
  因为非 Hasse 边是冗余的——它们的耦合信息已通过 Hasse 路径传递。

  即: ||M*_A - M*_B|| ≈ 0，对所有概念。
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def n_operator(
    M: np.ndarray,
    b_up: float,
    rho_up: float,
    params: Dict[str, float],
) -> np.ndarray:
    D, B, rho, R, S = M
    eps = params.get("eps", 0.01)
    a1 = params["alpha1"]
    b1 = params["beta1"]
    g1 = params["gamma1"]
    d1 = params["delta1"]
    z1 = params["zeta1"]
    e1 = params["eta1"]
    t1 = params["theta1"]
    k1 = params["kappa1"]
    k2 = params["kappa2"]
    l1 = params["lambda1"]
    m1 = params["mu1"]

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

    return np.array([D_new, B_new, rho_new, R_new, S_new])


def run_iteration(
    m0: np.ndarray,
    b_up: float,
    rho_up: float,
    params: Dict[str, float],
    max_iter: int = 500,
    tol: float = 1e-12,
):
    M = m0.copy()
    traj = [M.copy()]
    for _ in range(max_iter):
        M_new = n_operator(M, b_up, rho_up, params)
        traj.append(M_new.copy())
        if np.max(np.abs(M_new - M)) < tol:
            return M_new, np.array(traj)
        M = M_new
    return M, np.array(traj)


def build_ancestor_map(n_concepts: int, hasse_edges: List[Tuple[int, int]]):
    """构建 ancestors[c] = 节点 c 的所有祖先（含间接）。"""
    ancestors = [set() for _ in range(n_concepts)]
    children = defaultdict(set)
    for p, c in hasse_edges:
        children[p].add(c)

    def dfs(node, root_ancestors):
        ancestors[node] = root_ancestors | {node}
        for child in children[node]:
            dfs(child, ancestors[node])

    roots = set(range(n_concepts)) - {c for _, c in hasse_edges}
    for r in roots:
        dfs(r, set())

    for i in range(n_concepts):
        ancestors[i].discard(i)

    return ancestors


def run_lattice_experiment(lattice_path: Path, params: Dict[str, float]):
    """对单个格运行双模式 N-迭代实验。"""
    with open(lattice_path, "r", encoding="utf-8") as f:
        lattice = json.load(f)

    concept_name = lattice["concept_name"]
    n_concepts = lattice["n_concepts"]
    edges = [(p, c) for p, c in lattice["edges"]]
    concept_sizes = lattice["concept_sizes"]
    d_values = lattice["d_values"]

    ancestors = build_ancestor_map(n_concepts, edges)

    hasse_parents = [set() for _ in range(n_concepts)]
    for p, c in edges:
        hasse_parents[c].add(p)

    total_extent = sum(cs["|A|"] for cs in concept_sizes)
    total_intent = sum(cs["|B|"] for cs in concept_sizes)
    valid_d = [d for d in d_values if d != float("inf") and d < 1e6]
    max_d = max(valid_d) if valid_d else 1.0

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

    def run_mode(use_ancestors: bool):
        results = [None] * n_concepts
        for ci in topo_order:
            cs = concept_sizes[ci]
            raw_d = d_values[ci]
            d_init = ((raw_d / max_d) if (raw_d != float("inf") and raw_d < 1e6 and max_d > 0) else 0.8)
            d_init = min(d_init, 1.0)
            b_init = max(0.0, min(1.0, 1.0 - cs["|B|"] / max(total_extent, 1)))
            rho_init = max(0.0, min(1.0, cs["|A|"] / max(total_intent, 1)))
            m0 = np.array([d_init, b_init, rho_init, 0.5, 0.5])

            if use_ancestors:
                feeders = ancestors[ci]
            else:
                feeders = hasse_parents[ci]

            b_up = 0.0
            rho_up = 0.0
            cnt = 0
            for f_idx in feeders:
                if results[f_idx] is not None:
                    m_star_f, _ = results[f_idx]
                    b_up += m_star_f[1]
                    rho_up += m_star_f[2]
                    cnt += 1
            if cnt > 0:
                b_up /= cnt
                rho_up /= cnt

            m_star, traj = run_iteration(m0, b_up, rho_up, params)
            results[ci] = (m_star, traj)
        return results

    results_hasse = run_mode(use_ancestors=False)
    results_ancestors = run_mode(use_ancestors=True)

    diffs = []
    for ci in range(n_concepts):
        m_h, _ = results_hasse[ci]
        m_a, _ = results_ancestors[ci]
        diff = np.max(np.abs(m_h - m_a))
        diffs.append(diff)

    n_hasse = len(edges)
    n_non_hasse = sum(len(ancestors[i]) - len(hasse_parents[i]) for i in range(n_concepts))
    hasse_only = (n_non_hasse == 0)

    return {
        "concept": concept_name,
        "n_concepts": n_concepts,
        "n_hasse_edges": n_hasse,
        "n_non_hasse_ancestors": n_non_hasse,
        "hasse_only": hasse_only,
        "max_diff": float(max(diffs)),
        "mean_diff": float(np.mean(diffs)),
        "diffs": [float(d) for d in diffs],
        "pass": all(d < 1e-6 for d in diffs) or hasse_only,
    }


def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    params = {
        "alpha1": 1.0, "beta1": 1.0, "gamma1": 1.0, "delta1": 1.0,
        "zeta1": 1.0, "eta1": 1.0, "theta1": 1.0, "kappa1": 1.0,
        "kappa2": 1.0, "lambda1": 1.0, "mu1": 1.0, "eps": 0.01,
    }

    lattice_files = sorted(results_dir.glob("*_lattice.json"))
    if not lattice_files:
        print("No lattice JSON files found.")
        sys.exit(1)

    print(f"Found {len(lattice_files)} lattice files.\n")
    print("Theorem 11.2: E_N = E_H — N-迭代耦合边 = Hasse 边")
    print("=" * 70)
    print(f"{'概念':<30} {'概念数':>6} {'Hasse边':>8} {'非Hasse':>8} {'max|ΔM*|':>12} {'通过':>6}")
    print("-" * 70)

    all_results = []
    passes = 0
    total = 0

    skipped = []
    for lf in lattice_files:
        try:
            result = run_lattice_experiment(lf, params)
        except (KeyError, json.JSONDecodeError) as e:
            name = lf.stem.replace("_lattice", "")
            skipped.append((name, str(e)))
            continue

        all_results.append(result)
        total += 1
        if result["pass"]:
            passes += 1

        status = "PASS" if result["pass"] else "FAIL"
        name = result["concept"][:28]
        tag = " (Hasse-only)" if result["hasse_only"] else ""
        print(f"{name:<30} {result['n_concepts']:>6} {result['n_hasse_edges']:>8} "
              f"{result['n_non_hasse_ancestors']:>8} {result['max_diff']:>12.2e} {status:>6}{tag}")

    if skipped:
        print(f"\n跳过 {len(skipped)} 个格式不兼容的文件:")
        for name, err in skipped:
            print(f"  {name}: {err}")

    print("-" * 70)
    rate = 100.0 * passes / total if total > 0 else 0.0
    print(f"\n总计: {passes}/{total} = {rate:.1f}% 通过")
    print()

    all_max_diffs = [r["max_diff"] for r in all_results if not r["hasse_only"]]
    if all_max_diffs:
        print(f"非平凡格（有非Hasse祖先关系）的 M* 差异统计:")
        print(f"  max   = {max(all_max_diffs):.2e}")
        print(f"  mean  = {np.mean(all_max_diffs):.2e}")
        print(f"  median = {np.median(all_max_diffs):.2e}")
        print()
        if np.mean(all_max_diffs) < 1e-6:
            print("VERDICT: Theorem 11.2 CONFIRMED — E_N = E_H")
            print("  非 Hasse 祖先关系不产生可检测的 M* 差异。")
        elif np.mean(all_max_diffs) < 1e-3:
            print("VERDICT: Theorem 11.2 LARGELY CONFIRMED")
            print(f"  M* 差异很小（均值 {np.mean(all_max_diffs):.2e}），")
            print("  非 Hasse 边对不动点的影响可忽略。")
        else:
            print("VERDICT: Theorem 11.2 needs further investigation")
            print(f"  M* 差异显著（均值 {np.mean(all_max_diffs):.2e}）。")

    summary_path = results_dir / "e1_theorem_11_2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "theorem": "11.2",
            "claim": "E_N = E_H",
            "total_concepts_tested": total,
            "passes": passes,
            "pass_rate": rate,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {summary_path}")


if __name__ == "__main__":
    main()
