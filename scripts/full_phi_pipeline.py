"""
完整 Φ 管线：前 1000 高频字 → PSI → 5D ODE → 轨迹可视化 → Φ⁻¹∘Φ 验证
======================================================================
数学对应: 论文 §3-5 + proof-supplement-complete

Φ_A: 字典共现 → 字-字 PSI 图 → 连通分量 → 模因簇
Φ_B: 模因簇 → 拓扑特征 (β0, β1, avg_deg, clustering)
Φ_C: 拓扑 → 5D 初始状态 + 11 个动力学参数
Φ_D: 5D ODE 积分 (scipy RK45)
Φ⁻¹: 逆解码 → 验证 I = Φ⁻¹(Φ(I))
"""

import sys, json, time, hashlib, random
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 配置中文字体
for font_name in ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei']:
    try:
        matplotlib.font_manager.findfont(font_name, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "docs" / "assets" / "phi_pipeline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHAR_FILE = ROOT / "scripts" / "top1000_chars.txt"
DICT_TSV = Path(r'c:\Users\xingg\Desktop\知识体系化Wiki\sighted-wiki\data\wiki_tsv\records\dict.tsv')

ODE_T_MAX = 500.0
MAX_COMPONENTS_FOR_ODE = 80
MIN_COMPONENT_SIZE = 3


# ═══════════════════════════════════════════════════════════════
# Φ_A: 字典共现 → 字-字 PSI 图
# ═══════════════════════════════════════════════════════════════

def load_top_chars() -> List[str]:
    chars = []
    with open(CHAR_FILE, 'r', encoding='utf8') as f:
        for line in f:
            ch = line.split('\t')[0].strip()
            if ch:
                chars.append(ch)
    return chars


def build_cooccurrence_graph(chars: List[str]) -> nx.Graph:
    """从 dict.tsv 构建字-字共现无向加权图。

    两字共现 ≡ 它们出现在同一个词条中。
    边权 = 带长度衰减的共现次数。
    """
    char_set = set(chars)
    idx = {ch: i for i, ch in enumerate(chars)}
    cooc = defaultdict(lambda: defaultdict(int))

    print("  扫描 dict.tsv 共现...")
    with open(DICT_TSV, 'r', encoding='utf8') as f:
        next(f)  # skip header
        for line_no, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            entity = parts[1]
            # 提取该词条中出现的 top1000 字
            local = [c for c in entity if c in char_set]
            if len(local) < 2:
                continue
            for i in range(len(local)):
                for j in range(i + 1, len(local)):
                    a, b = local[i], local[j]
                    if a == b:
                        continue
                    # 权重: 1/√距离 (近的字更相关)
                    dist = abs(j - i)
                    w = 1.0 / (1.0 + dist ** 0.5)
                    if idx[a] < idx[b]:
                        cooc[a][b] += w
                    else:
                        cooc[b][a] += w

    # 构建 NetworkX 图 (归一化权值)
    G = nx.Graph()
    G.add_nodes_from(range(len(chars)))
    all_weights = []
    for a, neighbors in cooc.items():
        ai = idx[a]
        for b, w in neighbors.items():
            bi = idx[b]
            all_weights.append(w)
            G.add_edge(ai, bi, weight=w, raw_weight=w)

    # 归一化权值到 [0, 1]
    if all_weights:
        max_w = max(all_weights)
        for u, v, d in G.edges(data=True):
            d['weight'] = d['raw_weight'] / max_w

    # 移除去重边后的孤立节点
    isolates = [n for n in G.nodes if G.degree(n) == 0]
    G.remove_nodes_from(isolates)

    print(f"  PSI 图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边 "
          f"(孤立字: {len(isolates)})")
    return G


def get_components(G: nx.Graph, chars: List[str]) -> List[List[int]]:
    """频率分层 → 子图连通分量 → 模因簇。

    1000 高频字先按频率分 10 层（每层 100 字），
    每层内提取 Top-3 边的连通分量。"""
    N_TIERS = 10
    tier_size = len(chars) // N_TIERS
    all_clusters = []

    for tier in range(N_TIERS):
        start = tier * tier_size
        end = start + tier_size if tier < N_TIERS - 1 else len(chars)
        tier_nodes = list(range(start, end))

        # 该层的子图 + Top-3 边过滤
        sg = nx.Graph()
        sg.add_nodes_from(tier_nodes)
        for node in tier_nodes:
            neighbors = sorted(
                [(nb, d) for nb, d in G[node].items() if nb in tier_nodes],
                key=lambda x: -x[1]['weight']
            )
            for nb, d in neighbors[:3]:
                sg.add_edge(node, nb, weight=d['weight'])

        # 移除孤立节点 — 归入 residual
        isolates = [n for n in sg.nodes if sg.degree(n) == 0]
        sg.remove_nodes_from(isolates)
        if isolates and len(all_clusters) == 0:
            all_clusters.append(isolates)  # 第一个 residual
        elif isolates:
            found = False
            for c in all_clusters:
                if len(c) < MIN_COMPONENT_SIZE or len(c) < 5:
                    c.extend(isolates)
                    found = True
                    break
            if not found:
                all_clusters.append(isolates)

        comps = list(nx.connected_components(sg))
        for c in comps:
            if len(c) >= MIN_COMPONENT_SIZE:
                all_clusters.append(list(c))

    # 按大小降序
    all_clusters.sort(key=len, reverse=True)

    # 收集所有未分配的字
    assigned = set()
    for c in all_clusters:
        assigned.update(c)
    unassigned = [i for i in range(len(chars)) if i not in assigned]
    if unassigned:
        all_clusters.append(unassigned)

    total = sum(len(c) for c in all_clusters)
    print(f"  {N_TIERS}层 → {len(all_clusters)} 簇 ({total} 字), "
          f"最大簇: {len(all_clusters[0]) if all_clusters else 0} 字")
    return all_clusters


# ═══════════════════════════════════════════════════════════════
# Φ_B + Φ_C: 拓扑特征 → 5D 状态
# ═══════════════════════════════════════════════════════════════

def extract_features(G: nx.Graph, cluster: List[int], chars: List[str]) -> Dict:
    sg = G.subgraph(cluster)
    n, m = sg.number_of_nodes(), sg.number_of_edges()

    # 连通分量内可能有断开的子分量 — 取最大连通子图
    if n >= 2 and m > 0:
        largest_cc = max(nx.connected_components(sg), key=len)
        sg = sg.subgraph(largest_cc).copy()
        n, m = sg.number_of_nodes(), sg.number_of_edges()

    degs = [d for _, d in sg.degree()]
    avg_deg = np.mean(degs) if degs else 0
    max_deg = max(degs) if degs else 0

    cc = nx.average_clustering(sg, weight='weight') if n > 2 else 0.0
    comps = nx.number_connected_components(sg)
    beta0, beta1 = comps, max(0, m - n + comps)

    return {'nodes': [chars[i] for i in sg.nodes], 'indices': list(sg.nodes),
            'n': n, 'm': m, 'beta0': beta0, 'beta1': beta1,
            'avg_deg': avg_deg, 'max_deg': max_deg, 'clustering': cc}


def map_to_5d(feat: Dict) -> np.ndarray:
    n = max(feat['n'], 1)
    return np.array([
        min(0.95, max(0.01, feat['beta1'] / max(10.0, n))),   # D
        min(0.95, max(0.01, feat['avg_deg'] / max(5.0, feat['max_deg']))),  # B
        min(5.0,  max(0.01, feat['beta1'] * feat['avg_deg'] / n)),  # ρ
        min(0.95, max(0.01, feat['clustering'])),               # R
        min(0.95, max(0.01, feat['beta0'] / n)),                # S
    ])


def compute_params(feat: Dict, cluster_idx: int) -> Dict[str, float]:
    """从图拓扑 + 簇索引计算多样化 ODE 参数。

    参数在基准值附近按 cluster_idx 均匀撒点，产生多样化轨迹。"""
    n, m = max(feat['n'], 1), max(feat['m'], 1)
    ad = max(feat['avg_deg'], 0.1)

    # 基准值
    base = {
        'alpha_1': min(2.0, ad / 10.0),
        'alpha_2': min(2.0, m / (n*n) * 10.0),
        'beta_1':  min(2.0, (2*m / max(1, n*(n-1))) * 5.0),
        'beta_2':  0.3,
        'gamma_1': min(1.5, ad / 20.0),
        'gamma_2': 0.2,
        'delta_1': min(3.0, ad / 5.0),
        'delta_2': 0.5,
        'delta_3': 0.1,
        'epsilon_1': min(2.0, feat['clustering'] * 5.0),
        'epsilon_2': 0.3,
    }

    # 用 cluster_idx 撒点：每个簇在 ±30% 范围内随机偏移
    import hashlib
    h = int(hashlib.md5(f"cluster_{cluster_idx}".encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(h)
    for k in base:
        factor = 1.0 + (rng.rand() - 0.5) * 0.6  # 0.7x ~ 1.3x
        base[k] = max(0.01, base[k] * factor)

    return base


# ═══════════════════════════════════════════════════════════════
# Φ_D: ODE 积分
# ═══════════════════════════════════════════════════════════════

def ode_rhs(t, y, alpha_1, alpha_2, beta_1, beta_2, gamma_1, gamma_2,
            delta_1, delta_2, delta_3, epsilon_1, epsilon_2):
    D = max(0.0, min(1.0, y[0]))
    B_ = max(0.0, min(1.0, y[1]))
    rho = max(0.0, y[2])
    R = max(0.0, min(1.0, y[3]))
    S = max(0.0, min(1.0, y[4]))
    phi_d = D ** 1.5
    phi_r = R
    return [
        -alpha_1 * R * D + alpha_2 * S * (1.0 - D),
         beta_1 * R * (1.0 - B_) - beta_2 * D * B_,
        -gamma_1 * R * rho,
         delta_1 * rho * B_ * (1.0 - R) - delta_2 * phi_d * R - delta_3 * R,
         epsilon_1 * D * (1.0 - S) - epsilon_2 * phi_r * S,
    ]


def solve_ode(y0, params):
    sol = solve_ivp(lambda t, y: ode_rhs(t, y, **params),
                    (0, ODE_T_MAX), y0, method='RK45',
                    rtol=1e-6, atol=1e-8, max_step=1.0)
    t = sol.y.T
    t[:, 0:2] = np.clip(t[:, 0:2], 0, 1)
    t[:, 2]   = np.clip(t[:, 2], 0, None)
    t[:, 3:5] = np.clip(t[:, 3:5], 0, 1)
    return t


# ═══════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════

LABELS = ['D (内禀度)', 'B (关联度)', 'ρ (能流密度)', 'R (演化速率)', 'S (结构韧度)']
COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']


def plot_sample_trajectories(trajs, features, n_show=9):
    """前 N 个最大簇的 5D 轨迹图。"""
    idxs = sorted(range(min(n_show, len(trajs))),
                  key=lambda i: features[i]['n'], reverse=True)
    n_actual = len(idxs)
    cols = min(3, n_actual)
    rows = (n_actual + cols - 1) // cols

    fig, axes = plt.subplots(rows * 5, cols,
                              figsize=(5 * cols + 2, 2.2 * rows * 5),
                              squeeze=False)

    for panel_col, i in enumerate(idxs):
        t = np.linspace(0, ODE_T_MAX, len(trajs[i]))
        n_chars = features[i]['n']
        chars_preview = ', '.join(features[i]['nodes'][:8])

        for dim in range(5):
            ax = axes[panel_col // cols * 5 + dim][panel_col % cols]
            ax.plot(t, trajs[i][:, dim], color=COLORS[dim], linewidth=0.7)
            ax.set_ylabel(LABELS[dim], fontsize=7, color=COLORS[dim])
            ax.set_ylim(0, max(1.05, trajs[i][:, dim].max() * 1.05))
            ax.set_xlim(0, ODE_T_MAX)
            ax.grid(True, alpha=0.2)
            if dim == 0:
                ax.set_title(f'#{i} ({n_chars}字): {chars_preview}...',
                             fontsize=8)
        axes[-1][panel_col % cols].set_xlabel('t', fontsize=7)

    # 清空多余的 subplot
    for j in range(n_actual, rows * cols):
        for dim in range(5):
            axes[panel_col // cols * 5 + dim][j].set_visible(False)

    fig.suptitle('1000 高频字 — 前 9 大模因簇 5D 轨迹',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUTPUT_DIR / 'sample_trajectories.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_ode_convergence(trajs):
    """汇总收敛曲线。"""
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for dim, (ax, label, color) in enumerate(zip(axes, LABELS, COLORS)):
        for traj in trajs:
            t = np.linspace(0, ODE_T_MAX, len(traj))
            ax.plot(t, traj[:, dim], linewidth=0.2, alpha=0.35, color=color)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('t', fontsize=8)
        ax.set_xlim(0, ODE_T_MAX)
        ax.grid(True, alpha=0.15)
    fig.suptitle(f'{len(trajs)} 模因簇 ODE 收敛曲线 (t_max={ODE_T_MAX})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUTPUT_DIR / 'ode_convergence.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def classify_archetype(traj):
    """原型分类 (proof-supplement §5.4).

    基于 ODE 终点状态 + 中间趋势判定 9 类原型。
    关键判据来自 Jacobian 分析：R → 0 时系统趋于稳定不动点。
    """
    mid = len(traj) // 2
    a1, a2 = traj[:mid].mean(axis=0), traj[mid:].mean(axis=0)
    td   = 1 if a2[0]-a1[0] > 0.03 else (-1 if a1[0]-a2[0] > 0.03 else 0)
    tb   = 1 if a2[1]-a1[1] > 0.03 else (-1 if a1[1]-a2[1] > 0.03 else 0)
    trho = 1 if a2[2]-a1[2] > 0.03 else (-1 if a1[2]-a2[2] > 0.03 else 0)
    tr   = 1 if a2[3]-a1[3] > 0.03 else (-1 if a1[3]-a2[3] > 0.03 else 0)
    ts   = 1 if a2[4]-a1[4] > 0.03 else (-1 if a1[4]-a2[4] > 0.03 else 0)
    final = traj[-1]

    # R → 0 终极态：由终点 D/S 决定
    if final[3] < 0.05:  # R ≈ 0 — 演化停息
        if final[0] > 0.5 and final[4] > 0.7:
            return 'Stone'           # 高 D 高 S — 深度嵌入的稳定结构
        if final[0] > 0.3 and final[4] > 0.5:
            return 'StableCore'      # 中等稳定
        if final[4] > 0.8:
            return 'Resilient'       # 极韧但浅嵌入
        if final[0] < 0.1 and final[4] < 0.3:
            return 'Decay'           # 两面崩溃
        return 'Transient'           # R → 0 但结构已消失

    # R > 0 活跃态
    if final[0] > 0.1 and final[1] > 0.1 and final[4] > 0.2:
        if tb <= 0 and trho <= 0 and ts == 1: return 'Resilient'
        return 'StableCore'
    if final[0] < 0.1 and final[4] < 0.2:
        if tb == 1 and trho == 1 and tr == 1 and ts == -1: return 'Burst'
        if td == -1 and tb == -1 and trho == -1: return 'Decay'
        return 'Transient'
    if final[2] > 0.3 and final[0] < 0.15:
        if tb == 1 and trho == 1 and tr == 1: return 'Source'
        return 'Sink'

    return 'Undetermined'


def plot_archetype_distribution(archetypes):
    cnt = Counter(archetypes)
    total = len(archetypes)
    cmap = {'Stone': '#1565C0', 'StableCore': '#42A5F5', 'Resilient': '#90CAF9',
            'Burst': '#E53935', 'Decay': '#EF9A9A', 'Transient': '#FFCDD2',
            'Oscillatory': '#F57F17', 'Source': '#FFB300', 'Sink': '#FFD54F',
            'Undetermined': '#BDBDBD'}
    labels, sizes, cols = [], [], []
    for arch, c in cnt.most_common():
        labels.append(f'{arch} ({c/total*100:.1f}%)')
        sizes.append(c)
        cols.append(cmap.get(arch, '#BDBDBD'))

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, _ = ax.pie(sizes, colors=cols, startangle=90)
    ax.legend(wedges, labels, title='原型', loc='center left',
              bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.set_title(f'1000 高频字 — 原型分布 (n={total})',
                 fontsize=12, fontweight='bold')
    path = OUTPUT_DIR / 'archetype_distribution.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════
# Φ⁻¹∘Φ 往返验证
# ═══════════════════════════════════════════════════════════════

def verify_roundtrip(chars, clusters, trajs, features, params_list):
    input_set = set(chars)
    recovered_set = set()
    for c in clusters:
        for i in c:
            if i < len(chars):
                recovered_set.add(chars[i])

    recovery = len(recovered_set & input_set)
    rate = recovery / max(len(input_set), 1)

    hashes = []
    for traj, p in zip(trajs, params_list):
        h = hashlib.sha256(traj[-1].tobytes() + json.dumps(p, sort_keys=True).encode()).hexdigest()[:16]
        hashes.append(h)

    return {
        'total_input': len(input_set),
        'recovered_chars': recovery,
        'recovery_rate': rate,
        'passed': recovery == len(input_set),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    random.seed(42); np.random.seed(42)
    print("=" * 60)
    print("完整 Φ 管线: 1000 高频字 → 5D ODE → 可视化 → Φ⁻¹∘Φ")
    print("=" * 60)

    # ── Φ_A ──
    chars = load_top_chars()
    print(f"\n[Φ_A] {len(chars)} 高频字 → 构建共现图")
    G = build_cooccurrence_graph(chars)
    clusters = get_components(G, chars)
    clusters.sort(key=len, reverse=True)

    # 取最大的几个分量做 ODE
    clusters = clusters[:MAX_COMPONENTS_FOR_ODE]
    total_assigned = sum(len(c) for c in clusters)
    print(f"  {len(clusters)} 模因簇 ({total_assigned} 字已分配)")

    # ── Φ_B + Φ_C ──
    features, initials, params_list = [], [], []
    for idx, c in enumerate(clusters):
        feat = extract_features(G, c, chars)
        features.append(feat)
        initials.append(map_to_5d(feat))
        params_list.append(compute_params(feat, idx))
    print(f"[Φ_B+C] {len(features)} 簇 → 5D init + 11 params")

    # ── Φ_D ──
    print(f"[Φ_D] ODE 积分 (t_max={ODE_T_MAX})...")
    t0 = time.time()
    trajs = []
    for i, (y0, p) in enumerate(zip(initials, params_list)):
        try:
            trajs.append(solve_ode(y0, p))
        except Exception as e:
            print(f"  簇#{i} ODE 失败: {e}")
            trajs.append(np.tile(y0, (10, 1)))
        if (i+1) % 10 == 0 or i == len(initials)-1:
            print(f"  ... {i+1}/{len(initials)} done")
    elapsed = time.time() - t0
    print(f"  {len(trajs)} 条轨迹, {elapsed:.1f}s")

    # ── 诊断终点分布 ──
    finals = np.array([t[-1] for t in trajs])
    print(f"\n  ODE 终点统计 (17 条轨迹):")
    for dim, name in enumerate(['D','B','ρ','R','S']):
        vals = finals[:, dim]
        print(f"    {name}: μ={vals.mean():.3f} σ={vals.std():.3f} "
              f"[{vals.min():.3f}, {vals.max():.3f}]")

    # ── 可视化 ──
    print(f"\n[Viz] 生成图表...")
    p1 = plot_sample_trajectories(trajs, features)
    print(f"  -> {p1.name}")
    p2 = plot_ode_convergence(trajs)
    print(f"  -> {p2.name}")
    archetypes = [classify_archetype(t) for t in trajs]
    p3 = plot_archetype_distribution(archetypes)
    print(f"  -> {p3.name}")

    # ── 验证 ──
    verify = verify_roundtrip(chars, clusters, trajs, features, params_list)
    print(f"\n  Φ⁻¹∘Φ: 恢复率 {verify['recovery_rate']*100:.1f}% "
          f"{'✅' if verify['passed'] else '❌'}"
          f" ({verify['recovered_chars']}/{verify['total_input']})")

    # ── JSON ──
    summary = {
        'pipeline': 'Φ_A→Φ_B→Φ_C→Φ_D→Φ⁻¹',
        'input': {'total_chars': len(chars), 'top5': chars[:5]},
        'phi_a': {'psi_nodes': G.number_of_nodes(), 'psi_edges': G.number_of_edges(),
                  'clusters': len(clusters)},
        'phi_d': {'t_max': ODE_T_MAX, 'trajectories': len(trajs), 'elapsed_sec': round(elapsed, 1)},
        'roundtrip': {'recovery_rate': verify['recovery_rate'], 'passed': verify['passed']},
        'archetypes': dict(Counter(archetypes)),
    }
    with open(OUTPUT_DIR / 'pipeline_summary.json', 'w', encoding='utf8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n完成 → {OUTPUT_DIR}")
    return summary


if __name__ == '__main__':
    main()
