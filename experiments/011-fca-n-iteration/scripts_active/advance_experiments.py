import json, time
from pathlib import Path
from collections import defaultdict
import numpy as np

_EPS = 1e-10; _ITER = 300
np.random.seed(42)

def n_op(M, bu, ru, p):
    D, B, rho, R, S = M; e = p["eps"]
    return np.array([
        (p["a1"]*R+e)/(p["a1"]*R+p["b1"]*(B+bu)+e),
        (p["g1"]*(R+bu)+e)/(p["g1"]*(R+bu)+p["d1"]*D+e),
        (p["z1"]*(D+ru)+e)/(p["z1"]*(D+ru)+p["e1"]*R+e),
        (p["t1"]*(rho+ru+bu)+e)/(p["t1"]*(rho+ru+bu)+p["k1"]*D+p["k2"]*S+e),
        (p["l1"]*D+e)/(p["l1"]*D+p["m1"]*R+e),
    ])

def run(m0, bu, ru, p):
    M = m0.copy()
    for _ in range(_ITER):
        Mn = n_op(M, bu, ru, p)
        if np.max(np.abs(Mn-M)) < _EPS: return Mn
        M = Mn
    return M

def topo_sort(n, edges):
    in_deg = [0]*n; children = [[] for _ in range(n)]; parents = [set() for _ in range(n)]
    for pp,cc in edges: in_deg[cc]+=1; children[pp].append(cc); parents[cc].add(pp)
    q = [i for i in range(n) if in_deg[i]==0]; order = []
    while q:
        u = q.pop(0); order.append(u)
        for v in children[u]:
            in_deg[v]-=1
            if in_deg[v]==0: q.append(v)
    order.extend([i for i in range(n) if i not in order])
    return order, parents

def run_lattice(lat, p, edges_ov=None):
    edges = edges_ov or lat["edges"]
    cs = lat["concept_sizes"]; dv = lat["d_values"]; n = len(cs)
    order, parents = topo_sort(n, edges)
    te = sum(c["|A|"] for c in cs); ti = sum(c["|B|"] for c in cs)
    vd = [d for d in dv if d!=float("inf") and d<1e6]; md = max(vd) if vd else 1.0
    res = [None]*n
    for ci in order:
        c = cs[ci]; rd = dv[ci]
        di = (rd/md) if (rd!=float("inf") and rd<1e6 and md>0) else 0.8
        m0 = np.array([min(di,1.0), max(0,min(1,1-c["|B|"]/max(te,1))),
                       max(0,min(1,c["|A|"]/max(ti,1))), 0.5, 0.5])
        bu=0.0; ru=0.0; cnt=0
        for pp in parents[ci]:
            if res[pp] is not None: bu+=res[pp][0][1]; ru+=res[pp][0][2]; cnt+=1
        if cnt>0: bu/=cnt; ru/=cnt
        res[ci] = (run(m0,bu,ru,p), bu, ru)
    return res, edges

def chk_tau(res, edges):
    v = []
    for pp,cc in edges:
        tp=1.0/(np.sum(res[pp][0])+0.01); tc=1.0/(np.sum(res[cc][0])+0.01)
        if tp+1e-8 < tc: v.append((pp,cc,tp,tc))
    return v

def build_synthetic_lattice(n=20):
    rng = np.random.RandomState(42)
    cs = [{"|A|":int(rng.randint(3,15)),"|B|":int(rng.randint(2,12))} for _ in range(n)]
    dv = [c["|B|"]/max(c["|A|"],1) for c in cs]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if cs[i]["|B|"]>=cs[j]["|B|"] and cs[i]["|A|"]<=cs[j]["|A|"]:
                if rng.random()<0.3: edges.append([i,j])
    for i in range(2,n):
        if not any(cc==i for _,cc in edges): edges.append([rng.randint(0,i-1), i])
    return {"concept_name":"syn20","n_concepts":n,"concept_sizes":cs,
            "d_values":[float(d) for d in dv],"edges":edges}

def edge_heights(n, edges):
    _, parents = topo_sort(n, edges)
    h = np.zeros(n, dtype=int)
    for _ in range(n):
        for i in range(n):
            if parents[i]: h[i] = max(h[pp] for pp in parents[i])+1
    return h

def exp_C_strong(lat, p):
    print("C型强形式：多循环交叉注入")
    edges = lat["edges"]; n = lat["n_concepts"]
    r0, e0 = run_lattice(lat, p)

    results = []

    for label, cyc_edges in [
        ("3条交叉2-循环", [
            (edges[0][1], edges[0][0]), (edges[1][1], edges[1][0]), (edges[2][1], edges[2][0])
        ]),
        ("5条交叉2-循环", [
            (edges[i][1], edges[i][0]) for i in range(min(5,len(edges)))
        ]),
        ("1条3-循环", [(edges[0][1], edges[1][1]),
                        (edges[1][1], edges[0][0]),
                        (edges[0][0], edges[0][1])]),
    ]:
        cyc_e = list(edges) + [(int(c[0]), int(c[1])) for c in cyc_edges]
        r1, e1 = run_lattice(lat, p, edges_ov=cyc_e)
        v = chk_tau(r1, cyc_e)
        shift = np.max([np.max(np.abs(r1[i][0]-r0[i][0])) for i in range(n)])

        dir_flips = 0
        for pp,cc in cyc_edges:
            tp0 = 1.0/(np.sum(r0[pp][0])+0.01); tc0 = 1.0/(np.sum(r0[cc][0])+0.01)
            tp1 = 1.0/(np.sum(r1[pp][0])+0.01); tc1 = 1.0/(np.sum(r1[cc][0])+0.01)
            was_normal = tp0 + 1e-8 >= tc0
            is_flipped = tp1 + 1e-8 < tc1
            if was_normal and is_flipped: dir_flips += 1

        c_trig = shift > 1e-4 or dir_flips >= 1
        results.append({"label":label,"n_cycles":len(cyc_edges),"violations":len(v),
                        "shift":float(shift),"dir_flips":dir_flips,"c_triggered":c_trig})
        print(f"  {label}: 断裂={len(v)}/{len(cyc_e)} M*偏移={shift:.2e} 方向翻转={dir_flips} "
              f"{'C型强形式✓' if c_trig else '✗'}")
    return results

def exp_D_strong(lat, p):
    print("D型强形式：无向随机图")
    edges = lat["edges"]; n = lat["n_concepts"]
    rng = np.random.RandomState(12345)

    max_possible = n*(n-1)//2
    k_all = [min(max_possible, int(len(edges)*f)) for f in [3, 5]]
    results = []

    for k in k_all:
        all_pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
        chosen = rng.choice(len(all_pairs), k, replace=False)
        undirected = []
        for idx in chosen:
            i, j = all_pairs[idx]
            if rng.random() < 0.5: undirected.append((i,j))
            else: undirected.append((j,i))

        r1, e1 = run_lattice(lat, p, edges_ov=undirected)
        v = chk_tau(r1, undirected)
        rate = len(v)/max(len(undirected),1)
        d_trig = rate > 0.8
        results.append({"k":k,"e":len(undirected),"violations":len(v),
                        "rate":rate,"d_triggered":d_trig})
        print(f"  D强 k={k}: {len(v)}/{len(undirected)}断裂 ({100*rate:.0f}%) "
              f"{'D型强形式✓' if d_trig else '✗'}")
    return results

def exp_param_sensitivity(lat, p0):
    print("参数敏感性扫描（11参数 × 3倍率）")
    edges = lat["edges"]; n = lat["n_concepts"]
    r0, e0 = run_lattice(lat, p0)
    v0 = chk_tau(r0, e0)
    print(f"  基线: {len(v0)}/{len(edges)} τ⁻¹违反")
    tau0 = [1.0/(np.sum(r0[i][0])+0.01) for i in range(n)]

    param_names = ["a1","b1","g1","d1","z1","e1","t1","k1","k2","l1","m1"]
    multipliers = [0.01, 0.1, 10.0, 100.0]
    results = []

    for pn in param_names:
        for mul in multipliers:
            p1 = dict(p0); p1[pn] = p0[pn] * mul
            r1, e1 = run_lattice(lat, p1)
            v1 = chk_tau(r1, e1)
            tau1 = [1.0/(np.sum(r1[i][0])+0.01) for i in range(n)]
            tau_diff = np.max(np.abs(np.array(tau1)-np.array(tau0)))
            basin = all(np.max(np.abs(r1[i][0]-r0[i][0])) < 1e-3 for i in range(n))

            converged = all(np.max(np.abs(r1[i][0] - run(
                np.array([0.5,0.5,0.5,0.5,0.5]), 0.0, 0.0, p1
            ))) < 1e-3 for i in range(n)) if not any(v1) else None

            results.append({
                "param":pn,"multiplier":mul,"value":p1[pn],
                "violations":len(v1),"tau_diff":float(tau_diff),
                "basin_same":basin,"base_rate":(len(v1)/max(len(edges),1))
            })

    fragile = []; robust = []
    for r in results:
        if r["violations"] > len(v0) or r["tau_diff"] > 0.01:
            fragile.append(r)
        else:
            robust.append(r)
    fragile.sort(key=lambda x: -x["tau_diff"])

    print(f"\n  脆弱参数组合 ({len(fragile)} 个):")
    for r in fragile[:8]:
        print(f"    {r['param']}×{r['multiplier']}={r['value']:.2f} "
              f"断裂={r['violations']} τ差={r['tau_diff']:.4f}")

    print(f"\n  每次破坏率:")
    per_param = defaultdict(list)
    for r in fragile: per_param[r["param"]].append(r)
    for pn in param_names:
        cnt = len([r for r in fragile if r["param"]==pn])
        print(f"    {pn}: {cnt}/4次断裂 {'★关键' if cnt>=2 else ''}")

    return {"all":results,"fragile":fragile,"robust":robust,"baseline_violations":len(v0)}

def _py(v):
    if hasattr(v, "item"): return bool(v) if v.dtype == bool else float(v)
    if isinstance(v, dict): return {k: _py(vv) for k, vv in v.items()}
    if isinstance(v, list): return [_py(vv) for vv in v]
    return v

def main():
    p0 = {"a1":1,"b1":1,"g1":1,"d1":1,"z1":1,"e1":1,"t1":1,
          "k1":1,"k2":1,"l1":1,"m1":1,"eps":0.01}
    base = Path(__file__).resolve().parent.parent

    print("=" * 60)
    print("进阶实验：C强 + D强 + 参数敏感性")
    print("=" * 60)

    lat = build_synthetic_lattice(20)
    h = edge_heights(lat["n_concepts"], lat["edges"])
    print(f"\n合成格: {lat['n_concepts']}节点 {len(lat['edges'])}边 深度={int(max(h))}")

    print(f"\n{'='*50}")
    print("实验1: C型强形式")
    c_res = exp_C_strong(lat, p0)

    print(f"\n{'='*50}")
    print("实验2: D型强形式")
    d_res = exp_D_strong(lat, p0)

    print(f"\n{'='*50}")
    print("实验3: 参数敏感性")
    s_res = exp_param_sensitivity(lat, p0)

    print(f"\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")
    c_any = any(r["c_triggered"] for r in c_res)
    d_any = any(r["d_triggered"] for r in d_res)
    print(f"  C型强形式: {'✓ 触发' if c_any else '✗ 未触发'}")
    print(f"  D型强形式: {'✓ 触发' if d_any else '✗ 未触发'}")
    print(f"  参数敏感性: {len(s_res['fragile'])}/{len(s_res['all'])} 组合破坏 τ⁻¹")

    summary = {"C_strong": _py(c_res), "D_strong": _py(d_res),
               "param_sensitivity": _py(s_res)}
    with open(base/"results"/"advance_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: advance_summary.json")

if __name__ == "__main__":
    main()
