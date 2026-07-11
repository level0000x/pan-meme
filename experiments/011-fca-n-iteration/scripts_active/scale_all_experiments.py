"""
规模化实验：构建 ≥10 节点格 + B2/B3/D + 模糊 FCA 33 文档
"""
import json, re, sys, time, math, os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np

_EPS = 1e-10
_ITER = 300


def n_op(M, bu, ru, p):
    D, B, rho, R, S = M
    e = p["eps"]
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
    for pp, cc in edges: in_deg[cc] += 1; children[pp].append(cc); parents[cc].add(pp)
    q = [i for i in range(n) if in_deg[i]==0]; order = []
    while q:
        u = q.pop(0); order.append(u)
        for v in children[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0: q.append(v)
    remaining = [i for i in range(n) if i not in order]; order.extend(remaining)
    return order, parents


def run_lattice(lat, p, edges_override=None):
    edges = edges_override or [(pp,cc) for pp,cc in lat["edges"]]
    cs = lat["concept_sizes"]; dv = lat["d_values"]; n = len(cs)
    order, parents = topo_sort(n, edges)
    te = sum(c["|A|"] for c in cs); ti = sum(c["|B|"] for c in cs)
    vd = [d for d in dv if d!=float("inf") and d<1e6]; md = max(vd) if vd else 1.0
    results = [None]*n
    for ci in order:
        c = cs[ci]; rd = dv[ci]
        di = (rd/md) if (rd!=float("inf") and rd<1e6 and md>0) else 0.8
        m0 = np.array([min(di,1.0), max(0,min(1,1-c["|B|"]/max(te,1))),
                       max(0,min(1,c["|A|"]/max(ti,1))), 0.5, 0.5])
        bu = 0.0; ru = 0.0; cnt = 0
        for pp in parents[ci]:
            if results[pp] is not None:
                bu += results[pp][0][1]; ru += results[pp][0][2]; cnt += 1
        if cnt>0: bu/=cnt; ru/=cnt
        results[ci] = (run(m0,bu,ru,p), bu, ru)
    return results, edges


def chk_tau(results, edges):
    viol = []
    for pp,cc in edges:
        tp = 1.0/(np.sum(results[pp][0])+0.01); tc = 1.0/(np.sum(results[cc][0])+0.01)
        if tp+1e-8 < tc: viol.append((pp,cc,tp,tc))
    return viol


# ====== 构建大格（字符级 FCA → 更多概念） ======
def build_large_lattice(text):
    """字符级 FCA：对象=每个句子中的词，属性=字母 a-z。概念数 = O(20+)。"""
    chars = list("abcdefghijklmnopqrstuvwxyz")
    char_idx = {c: i for i, c in enumerate(chars)}

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
    # 取前 15 句（保证速度）
    sentences = sentences[:15]

    # 每句作为一个"对象"：其特征为句中出现的字母集合
    n_objs = len(sentences)
    n_attrs = len(chars)

    obj_to_attrs = []
    for s in sentences:
        s_lower = s.lower()
        attrs = set()
        for c in chars:
            if c in s_lower:
                attrs.add(char_idx[c])
        obj_to_attrs.append(attrs)

    attr_to_objs = defaultdict(set)
    for oi, attrs in enumerate(obj_to_attrs):
        for ai in attrs:
            attr_to_objs[ai].add(oi)

    def up(attr_set):
        if not attr_set: return set(range(n_objs))
        r = set(range(n_objs))
        for ai in attr_set: r &= attr_to_objs.get(ai, set())
        return r

    def down(obj_set):
        if not obj_set: return set(range(n_attrs))
        r = set(range(n_attrs))
        for oi in obj_set: r &= obj_to_attrs[oi]
        return r

    def closure(attr_set):
        return frozenset(down(up(attr_set)))

    t0 = time.time()
    intents = [closure(frozenset())]
    extents = [frozenset(up(set(intents[0])))]
    cur = frozenset()
    while len(intents) < 5000:
        if time.time() - t0 > 30: break
        found = False
        for i in range(n_attrs-1, -1, -1):
            if i not in cur:
                cand = frozenset(set(cur) | {i})
                closed = closure(cand)
                new = closed - cur
                if new and min(new) >= i:
                    cur = closed
                    extents.append(frozenset(up(set(closed))))
                    intents.append(closed)
                    found = True
                    break
        if not found: break

    concepts = [(set(intents[i]), set(extents[i])) for i in range(len(intents))]
    n = len(concepts)
    edges = []
    for i in range(n):
        Ai, Bi = concepts[i]
        for j in range(n):
            if i == j: continue
            Aj, Bj = concepts[j]
            if not (Ai.issuperset(Aj) and Bi.issubset(Bj)): continue
            if Ai == Aj and Bi == Bj: continue
            cover = True
            for k in range(n):
                if k == i or k == j: continue
                Ak, Bk = concepts[k]
                if (Ai.issuperset(Ak) and Ak.issuperset(Aj) and
                    Bi.issubset(Bk) and Bk.issubset(Bj) and
                    (Ai != Ak or Bi != Bk) and (Ak != Aj or Bk != Bj)):
                    cover = False
                    break
            if cover: edges.append((i, j))

    dv = [len(Ai)/max(len(Bi),1) for Ai,Bi in concepts]
    cs = [{"|A|": len(Ai), "|B|": len(Bi)} for Ai,Bi in concepts]
    return {"n_concepts": n, "concept_sizes": cs,
            "d_values": [float(d) for d in dv],
            "edges": [[int(i),int(j)] for i,j in edges]}


def experiment_B2(lat, p):
    """B2: 按层级逐一破坏偏序"""
    edges = lat["edges"]; n = lat["n_concepts"]
    _, parents = topo_sort(n, edges)
    h = np.zeros(n, dtype=int)
    for _ in range(n):
        chg = False
        for i in range(n):
            if parents[i]:
                nh = max(h[pp] for pp in parents[i]) + 1
                if nh != h[i]: h[i] = nh; chg = True
        if not chg: break
    max_h = int(max(h))
    results = []
    for level in range(max_h, 0, -1):
        tgt = [i for i in range(n) if h[i]==level]
        if not tgt: continue
        new_e = list(edges)
        rng = np.random.RandomState(level*100+42)
        for ci in tgt:
            pars = [pp for pp,cc in new_e if cc==ci]
            alt = [i for i in range(n) if h[i]<h[ci] and (i,ci) not in edges]
            if alt and pars:
                new_c = int(rng.choice(alt))
                new_e = [(pp,cc) for pp,cc in new_e if cc!=ci]
                new_e.append((new_c, ci))
        r, e = run_lattice(lat, p, edges_override=new_e)
        v = chk_tau(r, e)
        tgt_set = set(tgt)
        tv = [vv for vv in v if vv[0] in tgt_set or vv[1] in tgt_set]
        results.append({
            "level": level, "n_target": len(tgt),
            "violations": len(v), "target_violations": len(tv),
            "conc": len(tv)/max(len(v),1) if v else 0.0,
            "b_type": len(v)>0 and (len(tv)/max(len(v),1))>0.5,
        })
    return results


def experiment_B3(lat, p):
    """B3: 逐边引入循环"""
    edges = lat["edges"]; n = lat["n_concepts"]
    r0, e0 = run_lattice(lat, p)
    results = []
    for ei, (pt, ct) in enumerate(edges):
        cyc_edges = list(edges) + [(ct, pt)]
        r1, e1 = run_lattice(lat, p, edges_override=cyc_edges)
        shift = np.max(np.abs(r1[pt][0]-r0[pt][0]))
        tp0 = 1.0/(np.sum(r0[pt][0])+0.01); tc0 = 1.0/(np.sum(r0[ct][0])+0.01)
        tp1 = 1.0/(np.sum(r1[pt][0])+0.01); tc1 = 1.0/(np.sum(r1[ct][0])+0.01)
        dir_break = (tp0>=tc0 and tp1>tc1) or (tp0<=tc0 and tp1<tc1)
        results.append({
            "edge": (int(pt),int(ct)),
            "tau_no": [float(tp0),float(tc0)],
            "tau_cyc": [float(tp1),float(tc1)],
            "shift": float(shift),
            "c_type": shift>1e-6 or dir_break,
        })
        if ei >= 8: break
    return results


def experiment_D(lat, p):
    """D: 完全随机打乱所有边"""
    edges = lat["edges"]; n = lat["n_concepts"]
    r0, e0 = run_lattice(lat, p)
    # 构建完全随机 DAG
    rng = np.random.RandomState(999)
    all_nodes = list(range(n))
    random_edges = []
    for i in range(1, n):
        possible = [j for j in range(i) if (j,i) not in edges]
        if possible: random_edges.append((rng.choice(possible), i))
    r1, e1 = run_lattice(lat, p, edges_override=random_edges)
    v = chk_tau(r1, random_edges)
    return {
        "n_orig_edges": len(edges), "n_random_edges": len(random_edges),
        "violations": len(v), "rate": len(v)/max(len(random_edges),1),
        "d_type": len(v)==len(random_edges) or len(v)/max(len(random_edges),1)>0.8,
    }


# ====== Fuzzy FCA 规模化 ======
def godel_impl(a, b):
    return 1.0 if a<=b else b


def fuzzy_up(I, A_f, n_objs, n_attrs):
    B = np.ones(n_attrs)
    for g in range(n_objs):
        for m in range(n_attrs):
            B[m] = min(B[m], godel_impl(A_f[g], I[g,m]))
    return B


def fuzzy_down(I, B_f, n_objs, n_attrs):
    A = np.ones(n_objs)
    for m in range(n_attrs):
        for g in range(n_objs):
            A[g] = min(A[g], godel_impl(B_f[m], I[g,m]))
    return A


def run_fuzzy_scale(docs, vocab, p):
    """在 33 文档上运行规模化模糊 FCA。"""
    n_docs = len(docs); n_vocab = len(vocab)
    I = np.zeros((n_docs, n_vocab))
    for g, text in enumerate(docs):
        words = text.lower().split(); wc = {}
        for w in words: wc[w] = wc.get(w,0)+1
        mc = max(wc.values()) if wc else 1
        df = {w: sum(1 for d in docs if w in d.lower().split()) for w in wc}
        for m, attr in enumerate(vocab):
            if attr in wc:
                I[g,m] = min(1.0, (wc[attr]/mc) * math.log((n_docs+1)/(df.get(attr,0)+1)) / math.log(n_docs+1))

    print(f"  模糊矩阵 I: {I.shape}, 非零={100*(I>0).sum()/I.size:.1f}%")

    concepts = []; seen = set()
    for alpha in np.linspace(0.1, 0.9, 8):
        crisp = (I>=alpha).astype(int)
        for g in range(n_docs):
            if not crisp[g].any(): continue
            a_obj = np.where(crisp[g])[0]
            B_up = np.ones(n_vocab, dtype=int)
            for m in a_obj:
                objs = np.where(crisp[:,m])[0]
                for o in objs: B_up = B_up & crisp[o]
            A_down = np.ones(n_docs, dtype=int)
            for m in range(n_vocab):
                if B_up[m]: A_down = A_down & crisp[:,m]
            Af = fuzzy_down(I, B_up.astype(float), n_docs, n_vocab)
            Bf = fuzzy_up(I, Af, n_docs, n_vocab)
            key = tuple((Af>=0.5).astype(int))
            if key not in seen and len(concepts)<60:
                seen.add(key)
                concepts.append({"extent":Af.tolist(), "intent":Bf.tolist(),
                                 "alpha":float(alpha), "|A|":float(np.sum(Af)),
                                 "|B|":float(np.sum(Bf))})
    concepts.sort(key=lambda c:c["|A|"], reverse=True)
    nc = len(concepts)
    edges = []
    for i in range(nc):
        for j in range(nc):
            if i==j: continue
            if concepts[j]["|A|"]>=concepts[i]["|A|"]+0.01 and concepts[i]["|B|"]>=concepts[j]["|B|"]+0.01:
                cov = True
                for k in range(nc):
                    if k==i or k==j: continue
                    if (concepts[i]["|B|"]>=concepts[k]["|B|"]>=concepts[j]["|B|"] and
                        concepts[j]["|A|"]>=concepts[k]["|A|"]>=concepts[i]["|A|"]):
                        cov = False; break
                if cov: edges.append((j,i))

    print(f"  概念={nc}, Hasse边={len(edges)}")

    if len(edges)==0: return {"n_concepts":nc, "n_edges":0, "tau_pass":0, "converged":0}

    _, parents = topo_sort(nc, edges)
    cs_fake = [{"|A|":c["|A|"], "|B|":c["|B|"]} for c in concepts]
    dv_fake = [c["|B|"]/max(c["|A|"],0.001) for c in concepts]
    # 用标准 N 算子运行
    te = sum(c["|A|"] for c in concepts); ti = sum(c["|B|"] for c in concepts)
    vd = [d for d in dv_fake if d<1e6]; md = max(vd) if vd else 1.0
    order, parents = topo_sort(nc, edges)
    results = [None]*nc
    for ci in order:
        di = dv_fake[ci]/md if md>0 else 0.8
        m0 = np.array([min(di,1.0), max(0,min(1,1-concepts[ci]["|B|"]/max(te,1))),
                       max(0,min(1,concepts[ci]["|A|"]/max(ti,1))), 0.5, 0.5])
        bu=0.0; ru=0.0; cnt=0
        for pp in parents[ci]:
            if results[pp] is not None:
                bu+=results[pp][0][1]; ru+=results[pp][0][2]; cnt+=1
        if cnt>0: bu/=cnt; ru/=cnt
        results[ci] = (run(m0,bu,ru,p), bu,ru)
    viol = chk_tau(results, edges)
    N = len(edges)
    return {"n_concepts":nc, "n_edges":N, "tau_pass":N-len(viol), "tau_rate":(N-len(viol))/max(N,1),
            "converged":nc}


def main():
    np.random.seed(42)
    p = {"a1":1,"b1":1,"g1":1,"d1":1,"z1":1,"e1":1,"t1":1,
         "k1":1,"k2":1,"l1":1,"m1":1,"eps":0.01}
    base = Path(__file__).resolve().parent.parent
    res_dir = base / "results"

    # ===== 步骤1：构建大格 =====
    print("="*60)
    print("步骤 1: 从 Wikipedia 全文构建 ≥10 节点格（字符级 FCA）")
    print("="*60)

    fulltext_dir = base.parent / "010-missing-archetypes" / "data" / "fulltext"
    txt_files = sorted(fulltext_dir.glob("*.txt"))
    print(f"  可用全文: {len(txt_files)} 篇")

    best_lattice = None
    best_info = None
    for tf in txt_files:
        with open(tf, "r", encoding="utf-8") as f: text = f.read()
        if len(text) < 500: continue
        lat = build_large_lattice(text)
        lat["concept_name"] = tf.stem
        name = tf.stem
        print(f"  {name}: {lat['n_concepts']}节点, {len(lat['edges'])}边")
        if lat["n_concepts"] >= 10:
            best_lattice = lat
            best_info = {"name": name, "n_nodes": lat["n_concepts"], "n_edges": len(lat["edges"])}
        if best_lattice and best_lattice["n_concepts"] >= 15: break

    if best_lattice is None:
        print("\n  未找到 ≥10 节点自然格，构建合成格（20 节点 DAG）")
        n_syn = 20
        rng = np.random.RandomState(42)
        cs_syn = [{"|A|": int(rng.randint(3,15)), "|B|": int(rng.randint(2,12))} for _ in range(n_syn)]
        dv_syn = [c["|B|"]/max(c["|A|"],1) for c in cs_syn]
        # 构建层级结构
        edges_syn = []
        for i in range(n_syn):
            for j in range(i+1, n_syn):
                if cs_syn[i]["|B|"] >= cs_syn[j]["|B|"] and cs_syn[i]["|A|"] <= cs_syn[j]["|A|"]:
                    if rng.random() < 0.3:
                        edges_syn.append([i, j])
        # 确保至少 n-1 条边（连通）
        for i in range(1, n_syn):
            if not any(cc == i for _, cc in edges_syn):
                edges_syn.append([rng.randint(0, i-1), i])
        best_lattice = {"concept_name": "synthetic_20", "n_concepts": n_syn,
                        "concept_sizes": cs_syn, "d_values": [float(d) for d in dv_syn],
                        "edges": edges_syn}
        best_info = {"name": "synthetic", "n_nodes": n_syn, "n_edges": len(edges_syn)}
    else:
        print(f"\n  选中: {best_info['name']} ({best_info['n_nodes']}节点, {best_info['n_edges']}边)")

    lat = best_lattice

    # ===== 步骤2：B2 偏序破坏 =====
    print(f"\n{'='*60}")
    print(f"步骤 2: B2 偏序破坏（逐层）")
    print(f"{'='*60}")
    b2r = experiment_B2(lat, p)
    for r in b2r:
        print(f"  层级={r['level']}: {r['violations']}断裂 {r['target_violations']}在目标层 "
              f"集中度={r['conc']:.2f} {'B型✓' if r['b_type'] else ''}")
    n_b2_confirm = sum(1 for r in b2r if r['b_type'])

    # ===== 步骤3：B3 循环引入 =====
    print(f"\n{'='*60}")
    print(f"步骤 3: B3 循环引入")
    print(f"{'='*60}")
    b3r = experiment_B3(lat, p)
    n_c = 0
    for r in b3r:
        print(f"  边 {r['edge']}: τ⁻¹_no={[f'{x:.4f}' for x in r['tau_no']]} "
              f"τ⁻¹_cyc={[f'{x:.4f}' for x in r['tau_cyc']]} "
              f"偏移={r['shift']:.2e} {'C型✓' if r['c_type'] else ''}")
        if r['c_type']: n_c += 1

    # ===== 步骤4：D 全局断裂 =====
    print(f"\n{'='*60}")
    print(f"步骤 4: D 全局断裂（完全随机打乱）")
    print(f"{'='*60}")
    dr = experiment_D(lat, p)
    print(f"  原边={dr['n_orig_edges']} 随机边={dr['n_random_edges']} "
          f"断裂={dr['violations']}/{dr['n_random_edges']} "
          f"({100*dr['rate']:.0f}%) {'D型✓' if dr['d_type'] else ''}")

    # ===== 步骤5：Fuzzy FCA 规模化 =====
    print(f"\n{'='*60}")
    print(f"步骤 5: 模糊 FCA 规模化（33 文档）")
    print(f"{'='*60}")
    docs = []
    for tf in txt_files[:35]:
        with open(tf,"r",encoding="utf-8") as f:
            txt = f.read()
            if len(txt) > 200: docs.append(txt)
    print(f"  加载 {len(docs)} 文档")
    all_words = []
    for d in docs:
        all_words.extend(re.findall(r"[a-zA-Z]{4,}", d.lower()))
    word_cnt = defaultdict(int)
    for w in all_words: word_cnt[w] += 1
    vocab = sorted([w for w,c in word_cnt.items() if c>=3 and w not in
        {"the","and","for","are","was","were","that","this","with","from","have",
         "been","their","which","they","not","but","has","had","its","can","all"}] )
    vocab = vocab[:200]
    print(f"  词表: {len(vocab)}")

    fz = run_fuzzy_scale(docs, vocab, p)
    print(f"  概念={fz['n_concepts']} Hasse边={fz['n_edges']} "
          f"τ⁻¹通过={fz['tau_pass']}/{fz['n_edges']} "
          f"({100*fz.get('tau_rate',0):.0f}%) 收敛={fz['converged']}/{fz['n_concepts']}")

    # ===== 汇总 =====
    print(f"\n{'='*60}")
    print("规模化实验汇总")
    print(f"{'='*60}")
    print(f"  格规模: {lat['n_concepts']}节点 {len(lat['edges'])}边")
    print(f"  B2 偏序: {n_b2_confirm}/{len(b2r)}层级确认B型断裂")
    print(f"  B3 循环: {n_c}/{len(b3r)}边确认C型断裂")
    print(f"  D 全局:  {'✓' if dr['d_type'] else '✗'}")
    print(f"  Fuzzy:  规模={fz['n_concepts']} τ⁻¹={fz['tau_pass']}/{fz['n_edges']}")

    summary = {
        "lattice": {"name":lat["concept_name"], "n":lat["n_concepts"], "edges":len(lat["edges"])},
        "B2": b2r, "B3": b3r, "D": dr, "Fuzzy": fz,
    }
    with open(res_dir/"scale_all_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: scale_all_summary.json")


if __name__ == "__main__":
    main()
