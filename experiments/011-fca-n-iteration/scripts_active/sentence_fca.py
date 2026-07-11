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

STOPS = {"the","and","for","are","was","were","that","this","with","from",
         "have","been","their","which","they","not","but","has","had","its",
         "can","all","will","also","than","more","about","when","most","some",
         "into","only","other","between","after","over","many","such","each",
         "being","first","where","those","these","would","there","could","them",
         "what","than","then","just","like","well","very","because","how"}

def sentence_fca(docs, max_sentences=120, max_vocab=200):
    all_sents = []
    for d in docs:
        sents = re.split(r'[.!?;]\s+', d)
        for s in sents:
            words = re.findall(r'[a-zA-Z]{4,}', s.lower())
            words = [w for w in words if w not in STOPS]
            if len(words) >= 4: all_sents.append(words)

    if len(all_sents) > max_sentences:
        np.random.RandomState(42).shuffle(all_sents)
        all_sents = all_sents[:max_sentences]

    wc_all = defaultdict(int)
    for sw in all_sents:
        for w in sw: wc_all[w] += 1
    vocab = sorted([w for w,c in wc_all.items() if c >= 2])[:max_vocab]

    ns = len(all_sents); nv = len(vocab)
    print(f"  句子={ns} 词表={nv}")
    vocab_idx = {w: i for i,w in enumerate(vocab)}
    ctx = np.zeros((ns,nv), dtype=int)
    for g,sw in enumerate(all_sents):
        for w in sw:
            if w in vocab_idx: ctx[g, vocab_idx[w]] = 1

    idx_to_vocab = {i: w for i,w in enumerate(vocab)}

    concepts = []; seen = set()
    for g in range(ns):
        if not ctx[g].any(): continue
        B_up = ctx[g].copy()
        attrs = np.where(ctx[g])[0]
        for m in attrs:
            objs = np.where(ctx[:,m])[0]
            for o in objs: B_up = B_up & ctx[o]
        key = tuple(B_up)
        if key not in seen and B_up.sum() > 0:
            seen.add(key)
            A_down = np.array([all(ctx[o][B_up.astype(bool)] == 1) for o in range(ns)], dtype=float)
            concepts.append({
                "|A|": float(A_down.sum()),
                "|B|": float(B_up.sum()),
                "top_words": [idx_to_vocab[i] for i in np.where(B_up)[0][:5]]
            })
    for m in range(nv):
        if not ctx[:,m].any(): continue
        A_up = ctx[:,m].copy()
        objs = np.where(ctx[:,m])[0]
        for o in objs:
            attrs_o = np.where(ctx[o])[0]
            for a in attrs_o: A_up = A_up & ctx[:,a]
        key = tuple(A_up)
        if key not in seen and A_up.sum() > 0:
            seen.add(key)
            B_down = np.array([all(ctx[:,n][A_up.astype(bool)] == 1) for n in range(nv)], dtype=float)
            concepts.append({
                "|A|": float(A_up.sum()),
                "|B|": float(B_down.sum()),
                "top_words": [idx_to_vocab[i] for i in np.where(B_down)[0][:5]]
            })


    concepts.sort(key=lambda c: c["|A|"], reverse=True)
    nc = len(concepts)
    print(f"  初始概念={nc} |A|范围=[{concepts[-1]['|A|']:.0f},{concepts[0]['|A|']:.0f}] "
          f"|B|范围=[{min(c['|B|'] for c in concepts):.0f},{max(c['|B|'] for c in concepts):.0f}]")

    edges = []
    for i in range(nc):
        for j in range(nc):
            if i == j: continue
            a_bigger = concepts[j]["|A|"] > concepts[i]["|A|"] or (concepts[j]["|A|"] == concepts[i]["|A|"] and concepts[i]["|B|"] > concepts[j]["|B|"])
            b_bigger = concepts[i]["|B|"] > concepts[j]["|B|"] or (concepts[i]["|B|"] == concepts[j]["|B|"] and concepts[j]["|A|"] > concepts[i]["|A|"])
            if a_bigger and b_bigger:
                cov = True
                for k in range(nc):
                    if k == i or k == j: continue
                    if (concepts[i]["|B|"] >= concepts[k]["|B|"] >= concepts[j]["|B|"] and
                        concepts[j]["|A|"] >= concepts[k]["|A|"] >= concepts[i]["|A|"]):
                        cov = False; break
                if cov: edges.append((j, i))

    print(f"  Hasse边={len(edges)}")
    return concepts, edges

def main():
    p0 = {"a1":1,"b1":1,"g1":1,"d1":1,"z1":1,"e1":1,"t1":1,
          "k1":1,"k2":1,"l1":1,"m1":1,"eps":0.01}
    base = Path(__file__).resolve().parent.parent
    ftdir = base.parent / "010-missing-archetypes" / "data" / "fulltext"

    print("="*60)
    print("句子级 FCA：构建真实文本 ≥10 节点概念格")
    print("="*60)

    t0 = time.time()
    docs = []
    for tf in sorted(ftdir.glob("*.txt"))[:20]:
        with open(tf,"r",encoding="utf-8") as f:
            txt = f.read()
            if len(txt) > 400: docs.append(txt)
    print(f"加载 {len(docs)} 文档")

    for label, max_s in [("小规模", 80), ("中规模", 120), ("大规模", 180)]:
        print(f"\n{'='*50}")
        print(f"{label} 句子FCA (max {max_s} 句)")
        concepts, edges = sentence_fca(docs, max_sentences=max_s)

        if len(edges) == 0:
            print(f"  无Hasse边 — 格退化")
            continue

        lat = {"concept_name": f"sent_fca_{max_s}", "n_concepts": len(concepts),
               "concept_sizes": concepts,
               "d_values": [float(c["|B|"] / max(c["|A|"], 0.001)) for c in concepts],
               "edges": edges}
        res, e_chk = run_lattice(lat, p0)
        v = chk_tau(res, e_chk)
        depth = 0
        _, parents = topo_sort(len(concepts), edges)
        h = np.zeros(len(concepts), dtype=int)
        for _ in range(len(concepts)):
            for i in range(len(concepts)):
                if parents[i]: h[i] = max(h[pp] for pp in parents[i])+1
        depth = int(max(h))

        print(f"  N迭代完成: {len(v)}/{len(edges)} τ⁻¹违反 "
              f"({100*(len(edges)-len(v))/max(len(edges),1):.0f}%通过)")
        print(f"  格深度={depth} 概念={len(concepts)}")
        if concepts:
            print(f"  顶层: {concepts[0]['top_words'][:3]}")
            depth_bins = defaultdict(int)
            for i in range(len(concepts)):
                depth_bins[h[i]] += 1
            print(f"  层级分布: {dict(sorted(depth_bins.items()))}")

    print(f"\n总耗时: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
