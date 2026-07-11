"""
B1-B3: 断裂边界实验（快速版）
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

_EPS = 1e-10
_MAX_ITER = 300


def n_op(M, b_up, rho_up, p):
    D, B, rho, R, S = M
    eps = p["eps"]
    den_d = p["a1"]*R + p["b1"]*(B+b_up) + eps
    den_b = p["g1"]*(R+b_up) + p["d1"]*D + eps
    den_rho = p["z1"]*(D+rho_up) + p["e1"]*R + eps
    den_r = p["t1"]*(rho+rho_up+b_up) + p["k1"]*D + p["k2"]*S + eps
    den_s = p["l1"]*D + p["m1"]*R + eps

    return np.array([
        (p["a1"]*R+eps)/den_d,
        (p["g1"]*(R+b_up)+eps)/den_b,
        (p["z1"]*(D+rho_up)+eps)/den_rho,
        (p["t1"]*(rho+rho_up+b_up)+eps)/den_r,
        (p["l1"]*D+eps)/den_s,
    ])


def run(m0, b_up, rh_up, p, noise=0.0):
    M = m0.copy()
    for _ in range(_MAX_ITER):
        Mn = n_op(M, b_up, rh_up, p)
        if noise > 0:
            Mn += np.random.randn(5) * noise
            Mn = np.clip(Mn, 0.01, 0.99)
        if np.max(np.abs(Mn-M)) < _EPS:
            return Mn
        M = Mn
    return M


def topo_sort(n, edges):
    """Kahn's algorithm: 处理 DAG，遇到循环时返回 partial order。"""
    in_deg = [0] * n
    children = [[] for _ in range(n)]
    parents = [set() for _ in range(n)]
    for p, c in edges:
        in_deg[c] += 1
        children[p].append(c)
        parents[c].add(p)

    queue = [i for i in range(n) if in_deg[i] == 0]
    order = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in children[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)

    remaining = [i for i in range(n) if i not in order]
    order.extend(remaining)
    return order, parents


def run_lattice(lattice, params, edges_override=None, noise=0.0):
    edges = edges_override if edges_override is not None else [(p, c) for p, c in lattice["edges"]]
    cs = lattice["concept_sizes"]
    dv = lattice["d_values"]
    n = len(cs)

    order, parents = topo_sort(n, edges)

    te = sum(c["|A|"] for c in cs)
    ti = sum(c["|B|"] for c in cs)
    vd = [d for d in dv if d != float("inf") and d < 1e6]
    md = max(vd) if vd else 1.0

    results = [None] * n
    np.random.seed(42)
    for ci in order:
        c = cs[ci]
        rd = dv[ci]
        di = (rd/md) if (rd!=float("inf") and rd<1e6 and md>0) else 0.8
        m0 = np.array([min(di,1.0), max(0, min(1, 1-c["|B|"]/max(te,1))),
                       max(0, min(1, c["|A|"]/max(ti,1))), 0.5, 0.5])
        bu = 0.0; ru = 0.0; cnt = 0
        for p in parents[ci]:
            if results[p] is not None:
                ms = results[p][0]
                bu += ms[1]; ru += ms[2]; cnt += 1
        if cnt > 0: bu /= cnt; ru /= cnt
        ms = run(m0, bu, ru, params, noise)
        results[ci] = (ms, bu, ru)
    return results, edges


def chk_11_1(results, edges):
    viol = []
    for p, c in edges:
        tp = 1.0/(np.sum(results[p][0])+0.01)
        tc = 1.0/(np.sum(results[c][0])+0.01)
        if tp + 1e-8 < tc:
            viol.append((p, c, tp, tc))
    return viol


def main():
    res_dir = Path(__file__).resolve().parent.parent / "results"
    lf = sorted(res_dir.glob("*_lattice.json"))[0]
    with open(lf, "r", encoding="utf-8") as f:
        lat = json.load(f)
    if "edges" not in lat:
        print("No edges, exiting"); return

    n_edges = len(lat["edges"])
    print(f"概念: {lat['concept_name']}  ({lat['n_concepts']}节点, {n_edges}边)")
    params = {"a1":1,"b1":1,"g1":1,"d1":1,"z1":1,"e1":1,"t1":1,
         "k1":1,"k2":1,"l1":1,"m1":1,"eps":0.01}

    print("\n=== B1: 噪声注入 ===")
    for sigma in [0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25]:
        r, e = run_lattice(lat, params, noise=sigma)
        v = chk_11_1(r, e)
        lvls = set()
        _, parents = topo_sort(lat["n_concepts"], e)
        hgts = np.zeros(lat["n_concepts"], dtype=int)
        for i in range(lat["n_concepts"]):
            if parents[i]:
                hgts[i] = max(hgts[pp] for pp in parents[i]) + 1
        for vv in v: lvls.add(int(hgts[vv[0]]))
        vrate = len(v)/max(n_edges, 1)
        pat = "A型✓" if (len(v)>1 and len(lvls)>1) else ("—" if len(v)==0 else "边缘")
        print(f"  σ={sigma:.2f}: {len(v)}/{n_edges} ({100*vrate:.0f}%) 层级={lvls} {pat}")

    print("\n=== B2: 偏序破坏 ===")
    edges_orig = [(p,c) for p,c in lat["edges"]]
    _, parents = topo_sort(lat["n_concepts"], edges_orig)
    hgts = np.zeros(lat["n_concepts"], dtype=int)
    for i in range(lat["n_concepts"]):
        if parents[i]:
            hgts[i] = max(hgts[pp] for pp in parents[i]) + 1
    max_h = max(hgts)
    if max_h >= 1:
        tgt = [i for i in range(lat["n_concepts"]) if hgts[i] == max_h]
        new_edges = list(edges_orig)
        np.random.seed(123)
        for ci in tgt:
            op = [(pp,cc) for pp,cc in new_edges if cc==ci]
            np_pars = [i for i in range(lat["n_concepts"]) if hgts[i] < hgts[ci] and (i,ci) not in edges_orig]
            if np_pars:
                new_candidate = int(np.random.choice(np_pars))
                new_edges = [(pp,cc) for pp,cc in new_edges if cc!=ci]
                new_edges.append((new_candidate, ci))
        r, e = run_lattice(lat, params, edges_override=new_edges)
        v = chk_11_1(r, e)
        tgt_set = set(tgt)
        tv = [vv for vv in v if vv[0] in tgt_set or vv[1] in tgt_set]
        print(f"  破坏层级={max_h}: {len(v)}断裂, {len(tv)}在目标层 "
              f"(集中度={len(tv)/max(len(v),1):.2f}, "
              f"{'B型✓' if len(v)>0 and len(tv)/max(len(v),1)>0.5 else '非B型'})")
    else:
        print("  层级不足")

    print("\n=== B3: 循环引入 ===")
    mid = len(edges_orig)//2
    pt, ct = edges_orig[mid]
    cyc_edges = list(edges_orig) + [(ct, pt)]
    r0, e0 = run_lattice(lat, params)
    r1, e1 = run_lattice(lat, params, edges_override=cyc_edges)
    t0p = 1.0/(np.sum(r0[pt][0])+0.01)
    t0c = 1.0/(np.sum(r0[ct][0])+0.01)
    t1p = 1.0/(np.sum(r1[pt][0])+0.01)
    t1c = 1.0/(np.sum(r1[ct][0])+0.01)
    print(f"  边: {pt}→{ct}")
    print(f"  无循环: τ⁻¹(p)={t0p:.4f} τ⁻¹(c)={t0c:.4f}")
    print(f"  有循环: τ⁻¹(p)={t1p:.4f} τ⁻¹(c)={t1c:.4f}")
    shift = np.max(np.abs(r1[pt][0]-r0[pt][0]))
    print(f"  M*偏移: {shift:.2e}  C型={'✓' if shift>1e-6 else '未触发（格鲁棒）'}")

    summary = {"B3_done": True}
    sp = res_dir / "boundary_summary.json"
    with open(sp, "w") as f: json.dump(summary, f)
    print(f"\n结果已保存: {sp}")


if __name__ == "__main__":
    main()
