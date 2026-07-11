"""
规模化实验（快速版）：合成 20 节点格 + B2/B3/D + 模糊 FCA 15 文档
"""
import json, re, math, time
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

def exp_B2(lat, p):
    edges = lat["edges"]; n = lat["n_concepts"]
    _, parents = topo_sort(n, edges)
    h = np.zeros(n, dtype=int)
    for _ in range(n):
        for i in range(n):
            if parents[i]: h[i] = max(h[pp] for pp in parents[i])+1
    mh = int(max(h))
    results = []
    for lvl in range(mh, 0, -1):
        tgt = [i for i in range(n) if h[i]==lvl]
        if not tgt: continue
        new_e = list(edges)
        rng = np.random.RandomState(lvl*100+42)
        for ci in tgt:
            alt = [i for i in range(n) if h[i]<h[ci] and (i,ci) not in edges]
            if alt and any(pp!=ci for pp,_ in [(p,c) for p,c in new_e if c==ci]):
                new_c = int(rng.choice(alt))
                new_e = [(pp,cc) for pp,cc in new_e if cc!=ci]
                new_e.append((new_c, ci))
        r, e = run_lattice(lat, p, edges_ov=new_e)
        v = chk_tau(r, e)
        ts = set(tgt)
        tv = [vv for vv in v if vv[0] in ts or vv[1] in ts]
        conc = len(tv)/max(len(v),1) if v else 0.0
        results.append({"level":lvl,"n_target":len(tgt),"violations":len(v),
                        "target_violations":len(tv),"conc":conc,
                        "b_type":len(v)>0 and conc>0.5})
        print(f"  B2 层级={lvl}: {len(v)}/{len(e)}断裂 {len(tv)}在目标层 集中度={conc:.2f} {'B型✓' if results[-1]['b_type'] else ''}")
    return results

def exp_B3(lat, p, max_edges=10):
    edges = lat["edges"]
    r0, e0 = run_lattice(lat, p)
    results = []
    rng = np.random.RandomState(7)
    sample = edges if len(edges)<=max_edges else [edges[i] for i in rng.choice(len(edges), max_edges, replace=False)]
    for pt,ct in sample:
        cyc_e = list(edges)+[(ct,pt)]
        r1, e1 = run_lattice(lat, p, edges_ov=cyc_e)
        shift = np.max(np.abs(r1[pt][0]-r0[pt][0]))
        tp0=1.0/(np.sum(r0[pt][0])+0.01); tc0=1.0/(np.sum(r0[ct][0])+0.01)
        tp1=1.0/(np.sum(r1[pt][0])+0.01); tc1=1.0/(np.sum(r1[ct][0])+0.01)
        dir_br = (tp0>=tc0 and tp1>tc1) or (tp0<=tc0 and tp1<tc1)
        c_type = shift>1e-6 or dir_br
        results.append({"edge":(int(pt),int(ct)),"tau_no":[float(tp0),float(tc0)],
                        "tau_cyc":[float(tp1),float(tc1)],"shift":float(shift),"c_type":c_type})
        print(f"  B3 边{pt}→{ct}: τ⁻¹_no={[f'{x:.4f}' for x in [tp0,tc0]]} "
              f"τ⁻¹_cyc={[f'{x:.4f}' for x in [tp1,tc1]]} 偏移={shift:.2e} {'C型✓' if c_type else ''}")
    return results

def exp_D(lat, p):
    edges = lat["edges"]; n = lat["n_concepts"]
    rng = np.random.RandomState(999)
    random_e = []
    for i in range(1,n):
        poss = [(j,i) for j in range(i) if (j,i) not in edges]
        if poss and len(poss) > 0: random_e.append(poss[rng.randint(0,len(poss))])
        else: random_e.append((rng.randint(0,i-1), i))
    r1, e1 = run_lattice(lat, p, edges_ov=random_e)
    v = chk_tau(r1, random_e)
    rate = len(v)/max(len(random_e),1)
    d_t = rate>0.8
    print(f"  D: 原{len(edges)}边 → 随机{len(random_e)}边 断裂={len(v)}/{len(random_e)} ({100*rate:.0f}%) {'D型✓' if d_t else ''}")
    return {"n_orig":len(edges),"n_rand":len(random_e),"violations":len(v),"rate":rate,"d_type":d_t}

def crisp_fca_from_objects(I, nd, nv):
    t0 = time.time()
    concepts = []; seen = set()
    for g in range(nd):
        B = I[g] > 0
        B_up = B.astype(int)
        for m in np.where(B)[0]:
            objs = np.where(I[:,m] > 0)[0]
            for o in objs: B_up = B_up & (I[o] > 0).astype(int)
        key = tuple(B_up)
        if key not in seen:
            seen.add(key)
            A_down = np.array([all(I[o][B_up.astype(bool)]>0) or any(np.where(B_up)[0])==0 for o in range(nd)], dtype=float)
            concepts.append({"|A|":float(A_down.sum()), "|B|":float(B_up.sum())})
    concepts.sort(key=lambda c: c["|A|"], reverse=True)
    nc = len(concepts)
    edges = []
    for i in range(nc):
        for j in range(nc):
            if i == j: continue
            if concepts[j]["|A|"] >= concepts[i]["|A|"] + 0.01 and concepts[i]["|B|"] >= concepts[j]["|B|"] + 0.01:
                cov = True
                for k in range(nc):
                    if k == i or k == j: continue
                    if (concepts[i]["|B|"] >= concepts[k]["|B|"] >= concepts[j]["|B|"] and
                        concepts[j]["|A|"] >= concepts[k]["|A|"] >= concepts[i]["|A|"]):
                        cov = False; break
                if cov: edges.append((j, i))
    print(f"  概念={nc} Hasse边={len(edges)} ({time.time()-t0:.1f}s)")
    return concepts, edges

def fuzzy_scale(docs, vocab, p):
    nd = len(docs); nv = len(vocab)
    t0 = time.time()

    doc_words = [set(d.lower().split()) for d in docs]
    doc_counts = []
    for d in docs:
        wc = {}
        for w in d.lower().split(): wc[w] = wc.get(w, 0) + 1
        doc_counts.append(wc)
    vocab_df = {attr: sum(1 for dw in doc_words if attr in dw) for attr in vocab}
    I = np.zeros((nd, nv))
    for g in range(nd):
        wc = doc_counts[g]; mc = max(wc.values()) if wc else 1
        for m, attr in enumerate(vocab):
            if attr in wc:
                I[g, m] = min(1.0, (wc[attr] / mc) * math.log((nd + 1) / (vocab_df[attr] + 1)) / math.log(nd + 1))
    nz = 100 * (I > 0).sum() / I.size
    print(f"  模糊矩阵: {I.shape} 非零={nz:.1f}% ({time.time() - t0:.1f}s)")

    results = []
    for alpha in [0.0, 0.1, 0.2, 0.35, 0.5]:
        crisp = (I >= alpha).astype(int)
        concepts, edges = crisp_fca_from_objects(crisp, nd, nv)
        if len(edges) == 0:
            results.append({"alpha": alpha, "n": len(concepts), "e": 0, "pass": 0, "rate": 0.0})
            print(f"  α={alpha:.2f}: {len(concepts)}概念 0边 — 跳过")
            continue

        lat = {"concept_name": f"fuzzy_alpha_{alpha:.2f}", "n_concepts": len(concepts),
               "concept_sizes": concepts,
               "d_values": [float(c["|B|"] / max(c["|A|"], 0.001)) for c in concepts],
               "edges": edges}
        res_vals, e_chk = run_lattice(lat, p)
        v = chk_tau(res_vals, e_chk)
        results.append({"alpha": alpha, "n": len(concepts), "e": len(edges),
                        "pass": len(edges) - len(v), "rate": (len(edges) - len(v)) / max(len(edges), 1)})
        print(f"  α={alpha:.2f}: {results[-1]['n']}概念 {results[-1]['e']}边 "
              f"τ⁻¹={results[-1]['pass']}/{results[-1]['e']} ({100*results[-1]['rate']:.0f}%)")
    return results

def main():
    p = {"a1":1,"b1":1,"g1":1,"d1":1,"z1":1,"e1":1,"t1":1,
         "k1":1,"k2":1,"l1":1,"m1":1,"eps":0.01}
    base = Path(__file__).resolve().parent.parent

    print("="*60)
    print("规模化实验：合成 20 节点格 + B2/B3/D + 模糊FCA")
    print("="*60)

    lat = build_synthetic_lattice(20)
    print(f"\n合成格: {lat['n_concepts']}节点 {len(lat['edges'])}边")
    _, parents = topo_sort(20, lat["edges"])
    h = np.zeros(20, dtype=int)
    for _ in range(20):
        for i in range(20):
            if parents[i]: h[i] = max(h[pp] for pp in parents[i])+1
    print(f"  深度={max(h)} 层级分布={dict(zip(*np.unique(h, return_counts=True)))}")

    print(f"\n{'='*50}")
    print("B2: 偏序破坏（逐层）")
    b2r = exp_B2(lat, p)

    print(f"\n{'='*50}")
    print("B3: 循环引入")
    b3r = exp_B3(lat, p)

    print(f"\n{'='*50}")
    print("D: 全局随机化")
    dr = exp_D(lat, p)

    print(f"\n{'='*50}")
    print("Fuzzy FCA 规模化（15 文档）")
    ftdir = base.parent / "010-missing-archetypes" / "data" / "fulltext"
    docs = []
    for tf in sorted(ftdir.glob("*.txt"))[:15]:
        with open(tf,"r",encoding="utf-8") as f:
            txt = f.read()
            if len(txt)>200: docs.append(txt)
    print(f"  加载 {len(docs)} 文档")
    words_all = []
    for d in docs:
        for w in re.findall(r"[a-zA-Z]{4,}", d.lower()):
            if w not in {"the","and","for","are","was","were","that","this","with","from","have","been","their","which","they","not","but","has","had","its","can","all"}: words_all.append(w)
    wc_all = defaultdict(int)
    for w in words_all: wc_all[w] += 1
    vocab = sorted([w for w,c in wc_all.items() if c>=3])[:150]
    print(f"  词表={len(vocab)}")
    fr = fuzzy_scale(docs, vocab, p)
    best = max(fr, key=lambda x: x["rate"]) if fr else {"n":0,"e":0,"pass":0,"rate":0}
    print(f"  最佳: α={best.get('alpha','-')} {best['n']}概念 τ⁻¹={best['pass']}/{best['e']} ({100*best['rate']:.0f}%)")

    print(f"\n{'='*60}")
    print("规模化实验汇总")
    print(f"{'='*60}")
    n_b2 = sum(1 for r in b2r if r["b_type"])
    n_b3 = sum(1 for r in b3r if r["c_type"])
    print(f"  B2: {n_b2}/{len(b2r)}层级确认B型断裂（≥{lat['n_concepts']}节点格）")
    print(f"  B3: {n_b3}/{len(b3r)}边确认C型断裂")
    print(f"  D:  {'✓' if dr['d_type'] else '✗'} ({100*dr['rate']:.0f}%断裂)")
    print(f"  Fuzzy: {best['n']}概念 τ⁻¹={best['pass']}/{best['e']} ({100*best['rate']:.0f}%)")

    def _py(v):
        if hasattr(v, "item"): return bool(v)
        if isinstance(v, dict): return {k: _py(vv) for k, vv in v.items()}
        if isinstance(v, list): return [_py(vv) for vv in v]
        return v
    summary = {"B2": [_py(r) for r in b2r], "B3": [_py(r) for r in b3r],
               "D": _py(dr), "Fuzzy": [_py(r) for r in fr]}
    with open(base/"results"/"scale_all_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: scale_all_summary.json")

if __name__ == "__main__":
    main()
