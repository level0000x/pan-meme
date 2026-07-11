"""
E-1: 热迹谱 Θ(t) 社区可分性实验（纯 numpy 版）
"""
import json, os, glob, re
from collections import Counter
import numpy as np

extract_dir = "experiments/009-external-validation/data/extracts"
concept_data = {}
for fpath in sorted(glob.glob(os.path.join(extract_dir, "*.json"))):
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)
    pages = data.get("query", {}).get("pages", {})
    text = ""
    for pid, page in pages.items():
        text = page.get("extract", "")
    name = os.path.splitext(os.path.basename(fpath))[0]
    name = name.replace("%2B", "+").replace("_", " ")
    concept_data[name] = text

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

def extract_words(text, min_len=3, max_words=200):
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

def build_cooccurrence(words, top_bigrams=200):
    bigram_freq = Counter()
    for w in words:
        for i in range(len(w) - 1):
            bg = w[i:i+2]
            bad = {"th","he","in","er","an","on","at","en","nd","or",
                   "re","ed","es","st","to","it","ar","nt","is","al"}
            if bg not in bad:
                bigram_freq[bg] += 1
    top_bgs = [bg for bg, _ in bigram_freq.most_common(top_bigrams)]
    bg_to_idx = {bg: i for i, bg in enumerate(top_bgs)}
    n_w, n_b = len(words), len(top_bgs)
    W_mat = np.zeros((n_w, n_b))
    for i, w in enumerate(words):
        for j in range(len(w) - 1):
            bg = w[j:j+2]
            if bg in bg_to_idx:
                W_mat[i, bg_to_idx[bg]] = 1.0
    return W_mat @ W_mat.T

def heat_trace(A, t_points=30, t_max=10.0):
    n = A.shape[0]
    t_arr = np.logspace(-1, np.log10(t_max), t_points)
    if n <= 1:
        return np.ones(t_points)
    d = A.sum(axis=1)
    d_safe = np.where(d > 0, d, 1.0)
    L = np.eye(n) - np.outer(1.0/np.sqrt(d_safe), 1.0/np.sqrt(d_safe)) * A
    L = np.nan_to_num(L)
    try:
        eigvals = np.linalg.eigvalsh(L)
    except:
        return np.ones(t_points)
    eigvals = np.clip(eigvals, 0, None)
    theta = np.array([np.sum(np.exp(-t*eigvals)) for t in t_arr])
    return theta / n

print(f"Concepts total: {len(concept_data)}")
results = {}
for name, text in concept_data.items():
    cat = CATEGORIES.get(name, "Other")
    words = extract_words(text)
    if len(words) < 10:
        continue
    A = build_cooccurrence(words)
    theta = heat_trace(A)
    results[name] = {"theta": theta, "cat": cat, "nw": len(words)}

names = list(results.keys())
X_raw = np.array([results[n]["theta"] for n in names])
y_str = [results[n]["cat"] for n in names]
print(f"Valid concepts: {len(names)}")

# PCA via SVD
X_ctr = X_raw - X_raw.mean(axis=0)
U, S, Vt = np.linalg.svd(X_ctr, full_matrices=False)
var_ratio = S[:2]**2 / np.sum(S**2)
print(f"PCA: PC1={var_ratio[0]:.3f}, PC2={var_ratio[1]:.3f}")

cats_unique = sorted(set(y_str))
cat2idx = {c: i for i, c in enumerate(cats_unique)}
y = np.array([cat2idx[c] for c in y_str])

# 5-fold nearest-centroid
rng = np.random.RandomState(42)
idx = rng.permutation(len(y))
fold_size = len(y) // 5
preds = np.zeros(len(y), dtype=int)
for f in range(5):
    s, e = f*fold_size, min((f+1)*fold_size, len(y))
    tst = np.zeros(len(y), bool); tst[s:e] = True
    trn = ~tst
    X_tr, X_te = X_ctr[trn], X_ctr[tst]
    cents = np.array([X_tr[y[trn]==ci].mean(0) for ci in range(len(cats_unique))])
    for i, j in enumerate(np.where(tst)[0]):
        preds[j] = np.argmin(np.sum((X_te[i]-cents)**2, axis=1))
acc = np.mean(preds == y)
print(f"Nearest-centroid 5-fold: {acc:.3f}")

# per-category
for ci in range(len(cats_unique)):
    m = y == ci
    if m.sum() == 0: continue
    ca = np.mean(preds[m] == y[m])
    pm = preds == ci
    pr = np.mean(y[pm]==ci) if pm.sum() > 0 else 0
    print(f"  {cats_unique[ci]:15s} n={m.sum():2d} acc={ca:.3f} prec={pr:.3f}")

# baseline: n_words only
nw = np.array([results[n]["nw"] for n in names], float)
nw_c = nw - nw.mean()
nw_s = np.std(nw) or 1.0
nw_sc = nw_c / nw_s
nwp = np.zeros(len(y), int)
for f in range(5):
    s, e = f*fold_size, min((f+1)*fold_size, len(y))
    tst = np.zeros(len(y), bool); tst[s:e] = True
    trn = ~tst
    nw_cents = np.array([nw_sc[(y==ci)&trn].mean() for ci in range(len(cats_unique))])
    for i, j in enumerate(np.where(tst)[0]):
        nwp[j] = np.argmin((nw_sc[j]-nw_cents)**2)
nw_acc = np.mean(nwp == y)
print(f"Baseline (n_words): {nw_acc:.3f}")
print(f"Gain: {(acc-nw_acc)/max(nw_acc,1e-3)*100:+.1f}%")

# k-means purity
n_cl = len(cats_unique)
cents = np.array([X_ctr[y==ci].mean(0) for ci in range(len(cats_unique))])
for _ in range(20):
    labs = np.argmin(np.sum((X_ctr[:,None,:]-cents[None,:,:])**2, axis=2), axis=1)
    for ci in range(len(cats_unique)):
        m = labs == ci
        if m.sum(): cents[ci] = X_ctr[m].mean(0)
purities = []
for cl in range(n_cl):
    m = labs == cl
    if m.sum() == 0: continue
    cnt = Counter(y[m]); dcat, dc = cnt.most_common(1)[0]; p = dc/m.sum()
    purities.append(p)
    print(f"  Cluster {cl}: {m.sum()} samples, dominant={cats_unique[dcat]}, purity={p:.2f}")
avg_purity = np.mean(purities) if purities else 0
print(f"Avg purity: {avg_purity:.3f}")

# PC1 loadings
t_arr = np.logspace(-1, 1, 30)
for idx in np.argsort(np.abs(Vt[0]))[::-1][:5]:
    print(f"  t={t_arr[idx]:.3f} PC1_load={Vt[0,idx]:.3f}")

# print PCA coords
X_pca = U[:,:2] * S[:2]
print("\nPCA coordinates (first 20):")
for i in range(min(20, len(names))):
    print(f"  {names[i]:<25s} {y_str[i]:<12s} {X_pca[i,0]:>8.2f} {X_pca[i,1]:>8.2f}")

# save
out = {
    "n": len(names), "pca_var": [float(v) for v in var_ratio],
    "acc": float(acc), "nw_acc": float(nw_acc), "purity": float(avg_purity),
    "per_concept": {
        names[i]: {"cat": y_str[i], "nw": results[names[i]]["nw"],
                   "pc1": float(X_pca[i,0]), "pc2": float(X_pca[i,1])}
        for i in range(len(names))
    }
}
with open("experiments/011-fca-n-iteration/results/e1_heat_trace.json", "w") as f:
    json.dump(out, f, indent=2)

print(f"\n{'='*50}")
print(f"VERDICT")
print(f"{'='*50}")
if acc > nw_acc + 0.05:
    print(f"PASS: Θ(t) captures semantic structure beyond word count (acc={acc:.3f} vs baseline={nw_acc:.3f})")
else:
    print(f"WEAK: Θ(t) {acc:.3f} vs baseline {nw_acc:.3f} — need richer spectral features")
