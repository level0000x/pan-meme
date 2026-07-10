"""
E-1 v4: 用 v5 N-迭代不动点 M* 检验社区可分性
==============================================
核心假说: 同语义域的概念 → 相似的不动点 M*
特征: (D*, rho, tau_inv) per concept
使用 v5_d_monotonic（概念特化参数）而非 uniform
"""
import json
import sys
import numpy as np
from collections import defaultdict, Counter

with open("experiments/011-fca-n-iteration/results/e1_v5_analytical_rho.json") as f:
    v5_data = json.load(f)

v5_detail = v5_data["v5_d_monotonic"]["e3"]["details"]
print(f"v5_d_monotonic: total={v5_data['v5_d_monotonic']['e3']['total']}, "
      f"pass_rate={v5_data['v5_d_monotonic']['e3']['pass_rate']:.3f}")

# Aggregate per-concept
concept_stats = defaultdict(lambda: {"D_star": [], "rho": [], "tau_inv": []})
for d in v5_detail:
    name = d["concept"].replace("_", " ")
    concept_stats[name]["D_star"].append(d["D_star_parent"])
    concept_stats[name]["D_star"].append(d["D_star_child"])
    concept_stats[name]["rho"].append(d["rho_parent"])
    concept_stats[name]["rho"].append(d["rho_child"])
    concept_stats[name]["tau_inv"].append(d["tau_inv_parent"])
    concept_stats[name]["tau_inv"].append(d["tau_inv_child"])

CATEGORIES = {
    "Academy Awards":"Entertainment","Adobe Flash":"Technology",
    "Algebra":"Science","Barbie (film)":"Entertainment",
    "Beethoven":"Culture","Bitcoin":"Technology",
    "Black Lives Matter":"Politics","BlackBerry":"Technology",
    "Brexit":"Politics","COVID-19":"Health",
    "Calculus":"Science","Carbon":"Science",
    "ChatGPT":"Technology","Christmas":"Culture",
    "Clubhouse (app)":"Technology","DNA":"Science",
    "Darwin":"Science","Democracy":"Politics",
    "Elon Musk":"Business","Evolution":"Science",
    "FIFA World Cup":"Sports","FarmVille":"Technology",
    "Flash Player":"Technology","GameStop short squeeze":"Business",
    "Google%2B":"Technology","Google+":"Technology","Grammy Awards":"Entertainment",
    "Gravity":"Science","Halloween":"Culture",
    "Internet Explorer":"Technology","LimeWire":"Technology",
    "MySpace":"Technology","NFT":"Technology",
    "Newton":"Science","Nokia":"Technology",
    "Olympic Games":"Sports","Oppenheimer (film)":"Entertainment",
    "Oxygen":"Science","Periodic table":"Science",
    "Philosophy":"Science","Photosynthesis":"Science",
    "Queen Elizabeth II":"Politics","Second Life":"Technology",
    "Shakespeare":"Culture","Super Bowl":"Sports",
    "Taylor Swift":"Entertainment","Tesla, Inc.":"Business",
    "Thanksgiving":"Culture","Trump":"Politics",
    "Ukraine":"Politics","Vine (service)":"Technology",
    "Windows XP":"Technology","Yahoo!":"Technology",
}

# Build feature matrix
names = []
y_str = []
feats = []
for name, stats in concept_stats.items():
    cat = CATEGORIES.get(name, "Other")
    D_arr = np.array(stats["D_star"])
    rho_arr = np.array(stats["rho"])
    tau_arr = np.array(stats["tau_inv"])
    
    # Features: mean, min, max, std of D*, rho, tau_inv
    f = np.array([
        D_arr.mean(), D_arr.min(), D_arr.max(), D_arr.std(),
        rho_arr.mean(), rho_arr.min(), rho_arr.max(), rho_arr.std(),
        tau_arr.mean(), tau_arr.min(), tau_arr.max(), tau_arr.std(),
    ])
    names.append(name)
    y_str.append(cat)
    feats.append(f)

X = np.array(feats)
y = np.array([{c: i for i, c in enumerate(sorted(set(y_str)))}[c] for c in y_str])
cats_list = sorted(set(y_str))

# Diagnostic: data concepts vs CATEGORIES
data_concepts = set(concept_stats.keys())
cat_concepts = set(CATEGORIES.keys())
matched = data_concepts & cat_concepts
unmatched_data = data_concepts - cat_concepts
print(f"Data concepts: {len(data_concepts)}, CATEGORIES: {len(cat_concepts)}, Matched: {len(matched)}")
if unmatched_data:
    print(f"UNMATCHED: {sorted(unmatched_data)}")
    for u in unmatched_data:
        cat = "Other"
        print(f"  Assigning {u} -> {cat}")
        CATEGORIES[u] = cat

print(f"\nConcepts: {len(names)}, Features: {X.shape[1]}, Categories: {len(cats_list)}")
print(f"Category distribution: {dict(Counter(y_str))}")

# Standardize
Xc = (X - X.mean(0)) / (X.std(0) + 1e-12)

# PCA
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
var_r = S[:2]**2 / np.sum(S**2)
X_pca = U[:,:2] * S[:2]
print(f"\nPCA: PC1={var_r[0]:.3f}, PC2={var_r[1]:.3f}")

# 5-fold nearest-centroid
rng = np.random.RandomState(42)
idx = rng.permutation(len(y)); N = len(y); K = 5
preds = np.zeros(N, int)
for f in range(K):
    s, e = f*N//K, min((f+1)*N//K, N)
    tst = np.zeros(N, bool); tst[s:e]=True; trn=~tst
    Xtr, Xte = Xc[trn], Xc[tst]
    cents = np.zeros((len(cats_list), Xtr.shape[1]))
    for ci in range(len(cats_list)):
        m = (y[trn]==ci)
        if m.sum()>0: cents[ci] = Xtr[m].mean(0)
        else: cents[ci] = np.inf
    for i, j in enumerate(np.where(tst)[0]):
        d2 = np.sum((Xte[i]-cents)**2, axis=1)
        preds[j] = np.argmin(np.nan_to_num(d2, nan=1e10))
acc = np.mean(preds==y)

# Baseline: random (expected)
import math
n_cl = len(cats_list)
chance = 1.0 / n_cl

print(f"Nearest-centroid 5-fold: {acc:.3f} (chance={chance:.3f})")
print(f"Gain over chance: {acc/chance:.1f}x")

# K-means
cents = np.array([Xc[y==ci].mean(0) for ci in range(len(cats_list))])
for _ in range(20):
    labs = np.argmin(np.sum((Xc[:,None,:]-cents[None,:,:])**2, axis=2), axis=1)
    for ci in range(len(cats_list)):
        m = labs==ci
        if m.sum(): cents[ci] = Xc[m].mean(0)

purities = []
for cl in range(len(cats_list)):
    m = labs==cl
    if m.sum()==0: continue
    cnt = Counter(y[m]); dcat, dc = cnt.most_common(1)[0]
    p = dc/m.sum()
    purities.append(p)
    print(f"  Cluster {cl}: {m.sum()} samples, {cats_list[dcat]} purity={p:.2f}")
avg_pur = np.mean(purities) if purities else 0
print(f"Avg purity: {avg_pur:.3f} (chance={chance:.3f}, ratio={avg_pur/chance:.1f}x)")

# Per-category accuracy
print(f"\nPer-category:")
for ci, cname in enumerate(cats_list):
    m = y==ci
    if m.sum()==0: continue
    ca = np.mean(preds[m]==ci)
    pm = preds==ci
    pr = np.mean(y[pm]==ci) if pm.sum()>0 else 0
    print(f"  {cname:15s} n={m.sum():2d} acc={ca:.2f} prec={pr:.2f}")

# Show concepts in PCA space
print(f"\nPCA coordinates:")
for i in range(min(30, len(names))):
    print(f"  {names[i]:<25s} {y_str[i]:<12s} PC1={X_pca[i,0]:>7.2f} PC2={X_pca[i,1]:>7.2f}")

# Feature importance
print(f"\nTop PC1 loadings:")
for idx in np.argsort(np.abs(Vt[0]))[::-1][:5]:
    feat_names = ["D*_mean","D*_min","D*_max","D*_std",
                  "rho_mean","rho_min","rho_max","rho_std",
                  "tau_mean","tau_min","tau_max","tau_std"]
    print(f"  {feat_names[idx]:12s} loading={Vt[0,idx]:.3f}")

print(f"\n{'='*50}")
print(f"VERDICT")
print(f"{'='*50}")
print(f"  N-iteration fixed point M* features:")
print(f"  Classification: {acc:.3f} (chance={chance:.3f}, {acc/chance:.1f}x)")
print(f"  Cluster purity: {avg_pur:.3f} (chance={chance:.3f}, {avg_pur/chance:.1f}x)")
if acc > chance * 2.5 and avg_pur > chance * 2:
    print(f"  PASS: M* carries significant community information")
else:
    print(f"  MIXED: M* has some community signal but not decisive")
    print(f"  (expected: v5 uses UNIFORM parameters → same M* for same structure)")
    print(f"  Next step: concept-specific parameters needed for M* differentiation")
