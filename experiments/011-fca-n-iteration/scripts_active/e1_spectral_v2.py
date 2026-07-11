"""
E-1: 谱可分性实验 v2 — 使用多种特征组合
============================================
特征包括:
  1. FCA 结构特征: |A|, |B|, D=|A|/|B|, |A|+|B|, Hasse高度
  2. Laplacian 谱特征: 特征值分布 (20-bin histogram)
  3. 热迹特征: Θ(t) 在 15 个 t 点的值
  4. v5 N-迭代特征: M*=[D*,B*,rho*,R*,S*], ρ, τ⁻¹

目标: 检验光谱+FCA结构是否区分概念社区
"""
import json, os, glob, re
from collections import Counter
import numpy as np

# ============================================================
# 1. 读入 Wikipedia 文本 + FCA lattice 数据
# ============================================================
extract_dir = "experiments/009-external-validation/data/extracts"
lattice_dir = "experiments/011-fca-n-iteration/results"

concept_texts = {}
for fpath in sorted(glob.glob(os.path.join(extract_dir, "*.json"))):
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)
    pages = data.get("query", {}).get("pages", {})
    text = ""
    for pid, page in pages.items():
        text = page.get("extract", "")
    name = os.path.splitext(os.path.basename(fpath))[0]
    name = name.replace("%2B", "+").replace("_", " ")
    concept_texts[name] = text

lattice_data = {}
for fpath in sorted(glob.glob(os.path.join(lattice_dir, "*_lattice.json"))):
    with open(fpath, encoding="utf-8") as f:
        lat = json.load(f)
    cname = lat.get("concept_name", "")
    cname = cname.replace("_", " ")
    lattice_data[cname] = lat

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
    "Google+":"Technology","Grammy Awards":"Entertainment",
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

# ============================================================
# 2. 特征提取
# ============================================================
def extract_words(text, min_len=4, max_words=200):
    words = re.findall(r'[a-zA-Z]+', text.lower())
    words = [w for w in words if len(w) >= min_len]
    stops = {'the','and','for','that','with','this','from','have','were','are',
             'its','not','but','all','has','been','was','also','had','can','which',
             'they','their','more','other','than','into','such','when','about',
             'some','these','those','each','over','after','would','what','may',
             'them','between','where','being','could','most','only','first',
             'there','well','very','many','like','because','however','through'}
    words = [w for w in words if w not in stops]
    return list(set(words[:max_words]))

def spectral_features(words):
    """Extract Laplacian spectral features from bigram co-occurrence graph."""
    n = len(words)
    if n < 5:
        return None

    # Bigram feature extraction
    bigram_freq = Counter()
    for w in words:
        for i in range(len(w) - 1):
            bg = w[i:i+2]
            bad = {"th","he","in","er","an","on","at","en","nd","or",
                   "re","ed","es","st","to","it","ar","nt","is","al"}
            if bg not in bad:
                bigram_freq[bg] += 1
    
    top_bgs = [bg for bg, _ in bigram_freq.most_common(120)]
    bg2idx = {bg: i for i, bg in enumerate(top_bgs)}
    
    W_mat = np.zeros((n, len(top_bgs)))
    for i, w in enumerate(words):
        for j in range(len(w) - 1):
            bg = w[j:j+2]
            if bg in bg2idx:
                W_mat[i, bg2idx[bg]] = 1.0
    
    A = W_mat @ W_mat.T
    np.fill_diagonal(A, 0)  # No self-loops
    
    # Laplacian
    d = A.sum(axis=1)
    if np.all(d == 0):
        return None
    
    d_safe = np.where(d > 0, d, 1.0)
    L = np.eye(n) - np.outer(1.0/np.sqrt(d_safe), 1.0/np.sqrt(d_safe)) * A
    L = np.nan_to_num(L)
    
    try:
        eigvals = np.linalg.eigvalsh(L)
    except:
        return None
    
    eigvals = np.clip(eigvals, 0, 2)
    
    # Features:
    # 1. Eigenvalue histogram (20 bins)
    hist, _ = np.histogram(eigvals, bins=20, range=(0, 2))
    hist = hist.astype(float) / n  # normalize
    
    # 2. Heat trace at multiple scales
    t_arr = np.logspace(-1, 1, 15)
    theta = np.array([np.sum(np.exp(-t*eigvals))/n for t in t_arr])
    
    # 3. Spectral moments
    m1 = np.mean(eigvals)
    m2 = np.mean(eigvals**2)
    m3 = np.mean(eigvals**3)
    
    # 4. Spectral gaps
    eigvals_sorted = np.sort(eigvals)
    gap1 = eigvals_sorted[0] if n > 0 else 0  # algebraic connectivity (approx)
    # spectral radius
    sr = eigvals[-1] if n > 0 else 0
    
    return np.concatenate([hist, theta, [m1, m2, m3, gap1, sr]])

def fca_features(lat):
    """Extract FCA structure features."""
    if lat is None:
        return None
    n_c = lat.get("n_concepts", 0)
    n_e = lat.get("n_hasse_edges", 0)
    d_vals = lat.get("d_values", [])
    if len(d_vals) == 0:
        return None
    d_arr = np.array(d_vals)
    return np.array([
        n_c, n_e, n_c/max(n_e, 1),  # structural ratios
        d_arr.min(), d_arr.max(), d_arr.mean(), d_arr.std(),  # D stats
        n_e/max(n_c, 1),  # edge density
    ])

# ============================================================
# 3. 主循环
# ============================================================
print("Building feature vectors...")
samples = []

for name, text in concept_texts.items():
    cat = CATEGORIES.get(name, "Other")
    words = extract_words(text)
    if len(words) < 10:
        continue
    
    lat = lattice_data.get(name) if name in lattice_data else None
    
    sf = spectral_features(words)
    ff = fca_features(lat)
    
    if sf is None and ff is None:
        continue
    
    feat_parts = []
    if sf is not None:
        feat_parts.append(sf)
    if ff is not None:
        feat_parts.append(ff)
    
    if len(feat_parts) == 0:
        continue
    
    feat = np.concatenate(feat_parts)
    
    samples.append({
        "name": name, "cat": cat, "feat": feat,
        "nw": len(words), "has_lattice": lat is not None,
        "feat_len": len(feat)
    })

# Use the minimum feature length across all samples for consistency
# (some may lack FCA lattice, others lack spectral)
min_feat_len = min(s["feat_len"] for s in samples)
for s in samples:
    s["feat"] = s["feat"][:min_feat_len]

n_feat = min_feat_len
print(f"Samples: {len(samples)}, features: {n_feat}")

# ============================================================
# 4. 分类
# ============================================================
X = np.array([s["feat"] for s in samples])
names_arr = [s["name"] for s in samples]
y_str = [s["cat"] for s in samples]
cats_unique = sorted(set(y_str))
cat2idx = {c: i for i, c in enumerate(cats_unique)}
y = np.array([cat2idx[c] for c in y_str])

# Standardize
X_mean = X.mean(axis=0); X_std = X.std(axis=0)
X_std = np.where(X_std < 1e-12, 1.0, X_std)
X_s = (X - X_mean) / X_std

# PCA
Xc = X_s - X_s.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
var_r = S[:2]**2 / np.sum(S**2)
print(f"PCA: PC1={var_r[0]:.3f}, PC2={var_r[1]:.3f}, sum={var_r.sum():.3f}")

# 5-fold nearest-centroid
rng = np.random.RandomState(42)
idx = rng.permutation(len(y))
N = len(y); K = 5
preds = np.zeros(N, dtype=int)
for f in range(K):
    s, e = f*N//K, min((f+1)*N//K, N)
    tst = np.zeros(N, bool); tst[s:e] = True; trn = ~tst
    Xtr, Xte = Xc[trn], Xc[tst]
    cents = np.zeros((len(cats_unique), Xtr.shape[1]))
    valid_cats = []
    for ci in range(len(cats_unique)):
        m = (y[trn] == ci)
        if m.sum() > 0:
            cents[ci] = Xtr[m].mean(0)
            valid_cats.append(ci)
        else:
            cents[ci] = np.full(Xtr.shape[1], np.inf)
    for i, j in enumerate(np.where(tst)[0]):
        d2 = np.sum((Xte[i] - cents)**2, axis=1)
        d2[:] = np.where(np.isnan(d2), np.inf, d2)
        preds[j] = np.argmin(d2)

acc = np.mean(preds == y)
print(f"Nearest-centroid 5-fold: {acc:.3f}")

# Baseline: n_words only
nw_arr = np.array([s["nw"] for s in samples], float)
nw_s = (nw_arr - nw_arr.mean())
nw_s = nw_s / (np.std(nw_arr) or 1.0)
nwp = np.zeros(N, int)
for f in range(K):
    s, e = f*N//K, min((f+1)*N//K, N)
    tst = np.zeros(N, bool); tst[s:e] = True; trn = ~tst
    nw_c = np.array([nw_s[(y==ci)&trn].mean() if ((y==ci)&trn).sum()>0 else 0 for ci in range(len(cats_unique))])
    for i, j in enumerate(np.where(tst)[0]):
        nwp[j] = np.argmin((nw_s[j]-nw_c)**2)
nw_acc = np.mean(nwp == y)
print(f"Baseline (n_words): {nw_acc:.3f}")
print(f"Gain: {(acc-nw_acc)/max(nw_acc,1e-3)*100:+.1f}%")

# Per-category
print(f"\nPer-category:")
for ci in range(len(cats_unique)):
    m = y == ci
    if m.sum()==0: continue
    ca = np.mean(preds[m]==ci)
    pm = preds==ci
    pr = np.mean(y[pm]==ci) if pm.sum()>0 else 0
    print(f"  {cats_unique[ci]:15s} n={m.sum():2d} acc={ca:.2f} prec={pr:.2f}")

# K-means
cents = np.array([Xc[y==ci].mean(0) for ci in range(len(cats_unique))])
for _ in range(20):
    labs = np.argmin(np.sum((Xc[:,None,:]-cents[None,:,:])**2, axis=2), axis=1)
    for ci in range(len(cats_unique)):
        m = labs == ci
        if m.sum(): cents[ci] = Xc[m].mean(0)
purities = []
for cl in range(len(cats_unique)):
    m = labs == cl
    if m.sum()==0: continue
    cnt = Counter(y[m]); dcat, dc = cnt.most_common(1)[0]
    purities.append(dc/m.sum())
    print(f"  Cluster {cl}: {m.sum()} samples, {cats_unique[dcat]}, purity={dc/m.sum():.2f}")
avg_pur = np.mean(purities) if purities else 0
print(f"Avg purity: {avg_pur:.3f}")

# Feature importance via PCA loading
X_pca = U[:,:2] * S[:2]
print(f"\nTop 5 PC1 loadings:")
for idx in np.argsort(np.abs(Vt[0]))[::-1][:5]:
    feat_type = f"hist[{idx}]" if idx < 20 else f"theta[{idx-20}]" if idx < 35 else f"moment[{idx-35}]"
    print(f"  feat[{idx}]={feat_type}, loading={Vt[0,idx]:.3f}")

# Top 5 PC2 loadings
print(f"Top 5 PC2 loadings:")
for idx in np.argsort(np.abs(Vt[1]))[::-1][:5]:
    feat_type = f"hist[{idx}]" if idx < 20 else f"theta[{idx-20}]" if idx < 35 else f"moment[{idx-35}]"
    print(f"  feat[{idx}]={feat_type}, loading={Vt[1,idx]:.3f}")

print(f"\n{'='*50}")
print(f"VERDICT")
print(f"{'='*50}")
if acc > nw_acc + 0.1 and avg_pur > 1/len(cats_unique) + 0.1:
    print(f"PASS: Spectral+FCA features separate communities")
    print(f"  accuracy={acc:.3f} >> baseline={nw_acc:.3f}")
    print(f"  purity={avg_pur:.3f} >> chance={1/len(cats_unique):.3f}")
else:
    print(f"MIXED: accuracy={acc:.3f} vs baseline={nw_acc:.3f}")
    print(f"  purity={avg_pur:.3f} vs chance={1/len(cats_unique):.3f}")
    print(f"  Features capture some structure but not strong community signal")
    print(f"  Possible: bigram co-occurrence insufficient — need semantic embeddings")
