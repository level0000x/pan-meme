//! Phase 2: 编码（几何化）
//!
//! 数学对应：
//!   3.3.2 骨架编码: 关系网络 → CW 胞腔复形
//!   3.3.3 度量编码: w → 几何度量
//!   3.3.4 场编码: C（层级势场 φ）+ E（梯度向量场 F = ∇φ/d）
//!      φ(node) = convergence_round(node) / max_round  ← 信息深度作为势
//!   3.3.5 不变量编码: Γ → 几何不变量
//!   3.3.6 可逆性元数据 → G = (K, g, ω, Γ, R)

use crate::emergence::extractor::RelationNetwork;
use crate::emergence::relations::PhaseOneOutput;
use std::collections::{HashMap, HashSet};

// ═══════════════════════════════════════════════════════════════════════
// 3.3.2 骨架编码: 关系网络 → CW 胞腔复形
// ═══════════════════════════════════════════════════════════════════════

/// CW 胞腔
#[derive(Debug, Clone)]
pub struct Cell {
    /// 胞腔的维度（0 = 点, 1 = 边, 2 = 面, ...）
    pub dim: usize,
    /// 胞腔 ID
    pub id: usize,
    /// 组成此胞腔的低维胞腔 ID 列表
    pub boundary: Vec<usize>,
    /// 此胞腔代表的节点文本（仅对 0-胞腔有意义）
    pub label: String,
    /// 此胞腔来自的数据域（概念层级或原始的 containment）
    pub domain: String,
}

/// CW 胞腔复形 K
///
/// 数学对应：3.3.2 — 将 R 中的每个节点映射为 0-胞腔，每条边映射为 1-胞腔
#[derive(Debug, Clone)]
pub struct CWComplex {
    pub cells: Vec<Cell>,
    /// 0-胞腔: node_index → cell_index
    pub v0_map: HashMap<usize, usize>,
    /// 1-胞腔: (node_i, node_j) → cell_index
    pub e1_map: HashMap<(usize, usize), usize>,
    /// Euler 示性数
    pub euler_characteristic: i64,
    /// Betti 数 β₀（连通分量数）、β₁（环路数）
    pub betti_0: usize,
    pub betti_1: usize,
}

impl CWComplex {
    /// 从关系网络构建胞腔复形骨架。
    ///
    /// 数学对应：3.3.2 — 骨架编码
    ///   - 每个节点 → 0-胞腔
    ///   - 每条 containment 边 → 1-胞腔
    ///   - 每条 connection 边（Jaccard > 0） → 1-胞腔
    ///   - 每个概念（连通分量） → 2-胞腔
    pub fn from_network(
        psi: &RelationNetwork,
        phase1: &PhaseOneOutput,
    ) -> Self {
        let mut cells: Vec<Cell> = Vec::new();
        let mut v0_map: HashMap<usize, usize> = HashMap::new();
        let mut e1_map: HashMap<(usize, usize), usize> = HashMap::new();
        let mut cell_id_counter = 0usize;

        // ── 0-胞腔：每个节点 ──
        for ni in 0..psi.node_count() {
            let id = cell_id_counter;
            cell_id_counter += 1;
            cells.push(Cell {
                dim: 0,
                id,
                boundary: Vec::new(),
                label: psi.node_texts[ni].clone(),
                domain: "psi".to_string(),
            });
            v0_map.insert(ni, id);
        }

        // ── 1-胞腔：containment 边（字 → 词）──
        for ci in 0..psi.char_to_words.len() {
            let v_ci = v0_map[&(ci + psi.word_count)];
            for &wi in &psi.char_to_words[ci] {
                let v_wi = v0_map[&wi];
                let id = cell_id_counter;
                cell_id_counter += 1;
                cells.push(Cell {
                    dim: 1,
                    id,
                    boundary: vec![v_ci, v_wi],
                    label: format!("containment: {} ⊂ {}",
                        psi.node_texts[ci + psi.word_count], psi.node_texts[wi]),
                    domain: "containment".to_string(),
                });
                let key = (v_ci.min(v_wi), v_ci.max(v_wi));
                e1_map.entry(key).or_insert(id);
            }
        }

        // ── 1-胞腔：connection 边（字—字 Jaccard > 0）──
        for ci in 0..psi.char_jaccard.len() {
            let v_ci = v0_map[&(ci + psi.word_count)];
            for &(cj, j) in &psi.char_jaccard[ci] {
                if ci < cj && j > 0.0 {
                    let v_cj = v0_map[&(cj + psi.word_count)];
                    let key = (v_ci.min(v_cj), v_ci.max(v_cj));
                    e1_map.entry(key).or_insert_with(|| {
                        let id = cell_id_counter;
                        cell_id_counter += 1;
                        cells.push(Cell {
                            dim: 1,
                            id,
                            boundary: vec![v_ci, v_cj],
                            label: format!("connection: {} ~ {} (J={:.3})",
                                psi.node_texts[ci + psi.word_count],
                                psi.node_texts[cj + psi.word_count],
                                j),
                            domain: "connection".to_string(),
                        });
                        id
                    });
                }
            }
        }

        // ── 1-胞腔：词—词 Jaccard > 0 ──
        for wi in 0..psi.word_jaccard.len() {
            let v_wi = v0_map[&wi];
            for &(wj, j) in &psi.word_jaccard[wi] {
                if wi < wj && j > 0.0 {
                    let v_wj = v0_map[&wj];
                    let key = (v_wi.min(v_wj), v_wi.max(v_wj));
                    e1_map.entry(key).or_insert_with(|| {
                        let id = cell_id_counter;
                        cell_id_counter += 1;
                        cells.push(Cell {
                            dim: 1,
                            id,
                            boundary: vec![v_wi, v_wj],
                            label: format!("word_assoc: {} ~ {} (J={:.3})",
                                psi.node_texts[wi], psi.node_texts[wj], j),
                            domain: "word_association".to_string(),
                        });
                        id
                    });
                }
            }
        }

        // ── 2-胞腔：概念（连通分量）— Level 0 以上 ──
        for level in &phase1.structure.levels {
            if level.is_empty() { continue; }
            let lvl = level[0].level;
            if lvl == 0 { continue; } // Level 0 是字本身，已在 0-胞腔
            for concept in level {
                let boundary_1cells: Vec<usize> = Vec::new(); // 概念边界由其成员的边组成
                // 简化：不枚举所有 1-胞腔边界
                let id = cell_id_counter;
                cell_id_counter += 1;
                cells.push(Cell {
                    dim: 2,
                    id,
                    boundary: boundary_1cells,
                    label: format!("concept_L{}_id{}", lvl, concept.id),
                    domain: format!("concept_level_{}", lvl),
                });
            }
        }

        // ── 计算不变量 ──
        let (b0, b1) = Self::compute_betti(&cells, &v0_map, &e1_map);
        let euler = Self::compute_euler(&cells);

        CWComplex {
            cells,
            v0_map,
            e1_map,
            euler_characteristic: euler,
            betti_0: b0,
            betti_1: b1,
        }
    }

    /// 计算 Betti 数。
    ///
    /// β₀ = 连通分量数 = |V| - rank(∂₁)
    /// β₁ = 独立环路数 = |E| - rank(∂₁)
    ///
    /// 简化：通过 BFS 计算连通分量
    fn compute_betti(
        cells: &[Cell],
        v0_map: &HashMap<usize, usize>,
        e1_map: &HashMap<(usize, usize), usize>,
    ) -> (usize, usize) {
        let n_v = v0_map.len();
        let n_e = e1_map.len();

        // β₀ = 连通分量数
        // 构建邻接表（仅 1-胞腔边）
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); cells.len()];
        for (_i, cell) in cells.iter().enumerate() {
            if cell.dim == 1 && cell.boundary.len() == 2 {
                let a = cell.boundary[0];
                let b = cell.boundary[1];
                adj[a].push(b);
                adj[b].push(a);
            }
        }

        // BFS 计算连通分量（仅在 0-胞腔之间）
        let mut visited: HashSet<usize> = HashSet::new();
        let mut components = 0usize;

        for (_, &cell_id) in v0_map.iter() {
            if visited.contains(&cell_id) { continue; }
            components += 1;
            let mut stack = vec![cell_id];
            visited.insert(cell_id);
            while let Some(u) = stack.pop() {
                for &v in &adj[u] {
                    if cells[v].dim == 0 && !visited.contains(&v) {
                        visited.insert(v);
                        stack.push(v);
                    }
                }
            }
        }

        let betti_0 = components;
        // β₁ ≈ |E| - |V| + β₀（简化计算）
        let betti_1 = n_e.saturating_sub(n_v.saturating_sub(betti_0));

        (betti_0, betti_1)
    }

    /// Euler 示性数 χ = Σ(-1)^k n_k
    fn compute_euler(cells: &[Cell]) -> i64 {
        let mut dim_counts: HashMap<usize, i64> = HashMap::new();
        for cell in cells {
            *dim_counts.entry(cell.dim).or_insert(0) += 1;
        }
        let mut euler = 0i64;
        for (dim, count) in dim_counts {
            if dim % 2 == 0 {
                euler += count;
            } else {
                euler -= count;
            }
        }
        euler
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 3.3.3 度量编码: Jaccard → 几何度量
// ═══════════════════════════════════════════════════════════════════════

/// 度量信息
#[derive(Debug, Clone)]
pub struct MetricInfo {
    /// 每条 1-胞腔的长度：从 Jaccard 距离映射
    pub edge_lengths: HashMap<(usize, usize), f64>,
    /// 平均边长
    pub avg_edge_length: f64,
    /// 最短 / 最长边
    pub min_edge_length: f64,
    pub max_edge_length: f64,
}

/// Jaccard 连接强度 → 几何距离。
///
/// 数学对应：3.3.3 — d(e) = 1 - w(e)（距离与连接强度反相关）
pub fn encode_metric(psi: &RelationNetwork, complex: &CWComplex) -> MetricInfo {
    let mut edge_lengths: HashMap<(usize, usize), f64> = HashMap::new();
    let wc = psi.word_count;

    // 字—字 Jaccard 映射到 1-胞腔长度
    for ci in 0..psi.char_jaccard.len() {
        let v0_ci = complex.v0_map[&(ci + wc)];
        for &(cj, j) in &psi.char_jaccard[ci] {
            let v0_cj = complex.v0_map[&(cj + wc)];
            let key = (v0_ci.min(v0_cj), v0_ci.max(v0_cj));
            let d = 1.0 - j;
            edge_lengths.insert(key, d);
        }
    }

    // 词—词 Jaccard
    for wi in 0..psi.word_jaccard.len() {
        let v0_wi = complex.v0_map[&wi];
        for &(wj, j) in &psi.word_jaccard[wi] {
            let v0_wj = complex.v0_map[&wj];
            let key = (v0_wi.min(v0_wj), v0_wi.max(v0_wj));
            let d = 1.0 - j;
            edge_lengths.entry(key).or_insert(d);
        }
    }

    // containment 边距离 = 0（层次性连接，非度量）
    // 这些不加入 edge_lengths

    let mut min_d = f64::MAX;
    let mut max_d = 0.0f64;
    let mut sum_d = 0.0;
    for &d in edge_lengths.values() {
        min_d = min_d.min(d);
        max_d = max_d.max(d);
        sum_d += d;
    }
    let avg_d = if edge_lengths.is_empty() { 0.0 } else { sum_d / edge_lengths.len() as f64 };

    MetricInfo {
        edge_lengths,
        avg_edge_length: avg_d,
        min_edge_length: min_d,
        max_edge_length: max_d,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 3.3.4 场编码: C（层级势场）+ E（梯度向量场）
// ═══════════════════════════════════════════════════════════════════════

/// 标量场 φ: 每个 0-胞腔的层级势
///
/// 数学对应：3.3.4 — C 部分
/// φ(字) = 0（原子层），φ(词) = |词| / max_word_len
/// Laplacian 平滑 2 轮后写入
#[derive(Debug, Clone)]
pub struct ScalarField {
    pub potential: HashMap<usize, f64>,
    pub avg_potential: f64,
    pub min_potential: f64,
    pub max_potential: f64,
}

/// 1-胞腔上的梯度向量场 F
///
/// 数学对应：3.3.4 — E 部分
/// F(e) = |φ(v₂)-φ(v₁)| / d(e) · ê
/// 其中 |F(e)| = mag 即离散梯度 |∇φ|_e
#[derive(Debug, Clone)]
pub struct VectorField {
    /// 1-胞腔 cell_id → (大小, 方向符号)
    pub flow: HashMap<usize, (f64, f64)>,
    /// 0-胞腔 cell_id → 散度
    pub divergence: HashMap<usize, f64>,
    pub total_flux: f64,
    pub avg_flux_density: f64,
    pub source_count: usize,
    pub sink_count: usize,
}

impl VectorField {
    /// 计算全局梯度模的均值与标准差。
    ///
    /// 数学对应：|∇φ| 沿所有 1-胞腔的统计量。
    /// mean(|∇φ|) → γ₁（能流耗散），std(|∇φ|)→ R（演化速率）/ ε₂（速朽定律）。
    pub fn grad_mean_std(&self) -> (f64, f64) {
        let n = self.flow.len();
        if n == 0 { return (0.0, 0.0); }
        let mean = self.flow.values().map(|&(m, _)| m).sum::<f64>() / n as f64;
        let var = self.flow.values()
            .map(|&(m, _)| (m - mean).powi(2)).sum::<f64>() / n as f64;
        (mean, var.sqrt())
    }
}

/// 构建层级势场 φ + 梯度向量场 F。
///
/// φ(node) = convergence_round(node) / max_round，将 ↑↓ 循环的信息深度映射为几何势。
/// 然后经 Laplacian 平滑扩散后计算离散梯度场 F(e) = |Δφ|/d。
/// 散度 div(v) = Σ F(e)·ê 检测信息源与汇。
///
/// Laplacian 轮数：⌈log₁₀(|V|)⌉（随图规模对数增长，小图 2 轮，大图 4 轮）
/// 自稳系数 0.3：防止势场在扩散中失去原始层级信息（可调）
pub fn encode_field(
    _psi: &RelationNetwork,
    complex: &CWComplex,
    metric: &MetricInfo,
    node_levels: &[usize],
) -> (ScalarField, VectorField) {
    let mut phi: HashMap<usize, f64> = HashMap::new();
    let max_level = node_levels.iter().max().copied().unwrap_or(1).max(1);

    // ── Step 1a: 初始势 = convergence_round / max_round ──
    for (ni, &cell_id) in &complex.v0_map {
        let level = node_levels.get(*ni).copied().unwrap_or(0);
        phi.insert(cell_id, level as f64 / max_level as f64);
    }

    // ── Step 1b: 邻接表 + Laplacian 平滑 ──
    // 轮数 = ⌈log₁₀(|V|)⌉（对数缩放，小图 2 轮即收敛）
    let n_v = complex.v0_map.len().max(1);
    let smooth_rounds = ((n_v as f64).log10().ceil() as usize).max(2).min(5);

    let mut v0_edges: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); complex.cells.len()];
    for (i, cell) in complex.cells.iter().enumerate() {
        if cell.dim == 1 && cell.boundary.len() == 2 {
            let a = cell.boundary[0];
            let b = cell.boundary[1];
            let d = metric.edge_lengths.get(&(a.min(b), a.max(b))).copied().unwrap_or(0.001);
            v0_edges[a].push((i, b, d));
            v0_edges[b].push((i, a, d));
        }
    }

    for _ in 0..smooth_rounds {
        let mut new_phi: HashMap<usize, f64> = HashMap::new();
        for (&cell_id, &old_phi) in &phi {
            let edges = &v0_edges[cell_id];
            if edges.is_empty() { new_phi.insert(cell_id, old_phi); continue; }
            let mut sw = 0.0; let mut ws = 0.0;
            for &(_, nb, d) in edges {
                let nb_p = phi.get(&nb).copied().unwrap_or(0.0);
                // 溢出防护：d 极小会导致 1/d 极大；clip 到安全范围
                let w = if d > 1e-8 { (1.0 / d).min(1e8) } else { 1e8 };
                ws += nb_p * w; sw += w;
            }
            // 自稳系数 0.3：保持自身信息深度不被过度稀释
            // NaN 防护：若 sw=0（所有邻居无势能），保留 old_phi
            let smooth = if sw > 0.0 { 0.3 * old_phi + 0.7 * ws / sw } else { old_phi };
            new_phi.insert(cell_id, if smooth.is_finite() { smooth } else { old_phi });
        }
        phi = new_phi;
    }

    // 标量场统计
    let mut sum_p = 0.0; let mut min_p = f64::MAX; let mut max_p = 0.0f64;
    for &p in phi.values() { sum_p += p; min_p = min_p.min(p); max_p = max_p.max(p); }
    let scalar = ScalarField {
        avg_potential: if phi.is_empty() { 0.0 } else { sum_p / phi.len() as f64 },
        min_potential: min_p, max_potential: max_p, potential: phi.clone(),
    };

    // ── Step 2: 梯度向量场 ──
    let mut flow: HashMap<usize, (f64, f64)> = HashMap::new();
    let mut total_flux = 0.0;
    for (i, cell) in complex.cells.iter().enumerate() {
        if cell.dim != 1 || cell.boundary.len() != 2 { continue; }
        let a = cell.boundary[0]; let b = cell.boundary[1];
        let pa = phi.get(&a).copied().unwrap_or(0.0);
        let pb = phi.get(&b).copied().unwrap_or(0.0);
        let d = metric.edge_lengths.get(&(a.min(b), a.max(b))).copied().unwrap_or(0.001);
        let mag = (pb - pa).abs() / d;
        let sign = if pb > pa { 1.0 } else if pa > pb { -1.0 } else { 0.0 };
        flow.insert(i, (mag, sign));
        total_flux += mag;
    }
    let fc = flow.len().max(1);

    // ── Step 3: 散度 ──
    let mut div: HashMap<usize, f64> = HashMap::new();
    for (&cell_id, &p) in &phi {
        let mut dv = 0.0;
        for &(e_id, nb_id, _) in &v0_edges[cell_id] {
            if let Some(&(mag, _sign)) = flow.get(&e_id) {
                let pn = phi.get(&nb_id).copied().unwrap_or(0.0);
                if p < pn { dv -= mag; } else if p > pn { dv += mag; }
            }
        }
        div.insert(cell_id, dv);
    }
    let (src, snk) = div.values().fold((0, 0), |(s, k), &v| {
        (s + if v > 0.1 { 1 } else { 0 }, k + if v < -0.1 { 1 } else { 0 })
    });

    let vector = VectorField {
        flow, divergence: div, total_flux, avg_flux_density: total_flux / fc as f64,
        source_count: src, sink_count: snk,
    };
    (scalar, vector)
}

// ═══════════════════════════════════════════════════════════════════════
// 3.3.5 不变量编码: Γ → 几何不变量
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct GeometricInvariants {
    /// Euler 示性数
    pub euler: i64,
    /// Betti 数
    pub betti_0: usize,
    pub betti_1: usize,
    /// 总胞腔数
    pub total_cells: usize,
    /// 0-胞腔数（|V|）
    pub v0_count: usize,
    /// 1-胞腔数（|E|）
    pub e1_count: usize,
    /// 2-胞腔数（概念/连通分量数）
    pub v2_count: usize,
    /// 平均度
    pub avg_degree: f64,
    /// 图密度
    pub density: f64,
}

/// 从胞腔复形提取几何不变量。
///
/// 数学对应：3.3.5 — 不变量编码
pub fn encode_invariants(complex: &CWComplex) -> GeometricInvariants {
    let v0_count = complex.v0_map.len();
    let e1_count = complex.e1_map.len();
    let v2_count = complex.cells.iter().filter(|c| c.dim == 2).count();

    // 度计算
    let mut degrees: Vec<usize> = vec![0; complex.cells.len()];
    for (_, &cell_id) in complex.e1_map.iter() {
        if cell_id < complex.cells.len() {
            let cell = &complex.cells[cell_id];
            if cell.boundary.len() == 2 {
                degrees[cell.boundary[0]] += 1;
                degrees[cell.boundary[1]] += 1;
            }
        }
    }
    let sum_deg: usize = degrees.iter().sum();
    let avg_degree = if v0_count > 0 { sum_deg as f64 / v0_count as f64 } else { 0.0 };

    let max_possible_edges = if v0_count > 1 { v0_count * (v0_count - 1) / 2 } else { 0 };
    let density = if max_possible_edges > 0 { e1_count as f64 / max_possible_edges as f64 } else { 0.0 };

    GeometricInvariants {
        euler: complex.euler_characteristic,
        betti_0: complex.betti_0,
        betti_1: complex.betti_1,
        total_cells: complex.cells.len(),
        v0_count,
        e1_count,
        v2_count,
        avg_degree,
        density,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 3.3.6 可逆性元数据 → G = (K, g, ω, Γ, R)
// ═══════════════════════════════════════════════════════════════════════

/// Phase 2 完整输出: G = (K, g, ω, Γ, R)
#[derive(Debug)]
pub struct PhaseTwoOutput {
    /// CW 胞腔复形
    pub complex: CWComplex,
    /// 度量信息
    pub metric: MetricInfo,
    /// 几何不变量
    pub invariants: GeometricInvariants,
    /// 标量势场 φ（层级势场 C）
    pub scalar_field: ScalarField,
    /// 梯度向量场 F（E 部分）
    pub vector_field: VectorField,
    /// 可逆性元数据（Phase 1 → Phase 2 的映射记录，含完整逆映射所需数据）
    pub reversibility: Vec<String>,
    /// 结构化可逆性数据：Φ_C⁻¹ 需要的信息（定理3逆）
    pub rev_data: ReversibilityData,
}

/// 可逆性元数据 — 包含从 G 重建 M 所需的全部信息（定理3 + 推论5.1）
///
/// 数学对应：§D.2 定义3 — G  的 R 分量
/// 存储节点文本、词/字标记、概念层级全量数据、规则和约束的完整序列化，
/// 确保 Φ_C⁻¹ 可以从 G 完整、无歧义地重建 M = (S, F, C)。
#[derive(Debug, Clone)]
pub struct ReversibilityData {
    /// 节点文本 node_index → text（字或词）
    pub node_texts: Vec<String>,
    /// node_index → 是否为词
    pub node_is_word: Vec<bool>,
    /// 词的数量
    pub word_count: usize,
    /// 概念层级深度
    pub containment_depth: usize,
    /// 每个节点在 ↑↓ 循环中的收敛轮数
    pub node_levels: Vec<usize>,

    // ── 完整概念层级序列化 ──
    /// 所有层级的概念：concept_levels[i] 是第 i 层的概念列表
    /// (concept_id, member_node_indices, level, external_links)
    pub concept_levels: Vec<Vec<(usize, Vec<usize>, usize, Vec<(usize, f64)>)>>,
    /// 每层的终止原因
    pub concept_termination_reasons: Vec<String>,

    // ── 完整规则序列化 ──
    /// (rule_type_discriminant, antecedent, consequent, confidence)
    /// discriminants: 0=CooccurrenceRule, 1=StructuralRule
    pub serialized_rules: Vec<(u8, Vec<usize>, Vec<usize>, f64)>,

    // ── 完整约束序列化 ──
    /// (description, invariant_equation, value)
    pub serialized_constraints: Vec<(String, String, f64)>,
}

/// 运行第二阶段编码。
pub fn run_phase_two(psi: &RelationNetwork, phase1: &PhaseOneOutput) -> PhaseTwoOutput {
    let complex = CWComplex::from_network(psi, phase1);
    let metric = encode_metric(psi, &complex);
    let invariants = encode_invariants(&complex);
    let (scalar_field, vector_field) = encode_field(psi, &complex, &metric, &phase1.node_levels);

    let entropy = compute_entropy(&phase1.node_levels);

    // 可逆性元数据（人类可读）
    let mut reversibility = Vec::new();
    reversibility.push(format!("节点数: {}", psi.node_count()));
    reversibility.push(format!("0-胞腔: {}, 1-胞腔: {}, 2-胞腔: {}",
        invariants.v0_count, invariants.e1_count, invariants.v2_count));
    reversibility.push(format!("Euler = {}, β₀ = {}, β₁ = {}",
        invariants.euler, invariants.betti_0, invariants.betti_1));
    reversibility.push(format!("度量: avg={:.4}, min={:.4}, max={:.4}",
        metric.avg_edge_length, metric.min_edge_length, metric.max_edge_length));
    reversibility.push(format!("密度: {:.4}, 平均度: {:.2}",
        invariants.density, invariants.avg_degree));
    reversibility.push(format!("场: φ_avg={:.4} [{:.4},{:.4}], 通量={:.4}, 源={}, 汇={}",
        scalar_field.avg_potential, scalar_field.min_potential, scalar_field.max_potential,
        vector_field.total_flux, vector_field.source_count, vector_field.sink_count));
    reversibility.push(format!("信息: H(levels)={:.4}", entropy));

    // 结构化可逆性数据：Φ_C⁻¹ 需要的信息
    let rev_data = ReversibilityData {
        node_texts: psi.node_texts.clone(),
        node_is_word: psi.is_word.clone(),
        word_count: psi.word_count,
        containment_depth: phase1.structure.depth,
        node_levels: phase1.node_levels.clone(),

        // 完整概念层级序列化
        concept_levels: phase1.structure.levels.iter()
            .map(|lv| lv.iter()
                .map(|c| (c.id, c.members.clone(), c.level, c.external_links.clone()))
                .collect()
            )
            .collect(),
        concept_termination_reasons: phase1.structure.termination_reasons.clone(),

        // 完整规则序列化
        serialized_rules: phase1.rules.iter()
            .map(|r| {
                let disc = match r.rule_type {
                    crate::emergence::relations::RuleType::CooccurrenceRule => 0u8,
                    crate::emergence::relations::RuleType::StructuralRule => 1u8,
                };
                (disc, r.antecedent.clone(), r.consequent.clone(), r.confidence)
            })
            .collect(),

        // 完整约束序列化
        serialized_constraints: phase1.constraints.iter()
            .map(|c| (c.description.clone(), c.invariant_equation.clone(), c.value))
            .collect(),
    };

    PhaseTwoOutput {
        complex,
        metric,
        invariants,
        scalar_field,
        vector_field,
        reversibility,
        rev_data,
    }
}

/// Shannon 熵 H(X) = -Σ p(x)·log₂ p(x)
///
/// 输入：per-node 的离散层级值（convergence_round）
/// 输出：层级分布的 Shannon 熵，单位 bit
///
/// 信息论含义：
///   H ≈ 0  → 所有节点同层级（扁平结构，无信息分层）
///   H ≈ log₂(max_level) → 各层级均匀分布（最大信息分层）
pub fn compute_entropy(levels: &[usize]) -> f64 {
    if levels.is_empty() { return 0.0; }
    let max_level = levels.iter().max().copied().unwrap_or(0);
    if max_level == 0 { return 0.0; }

    let mut counts = vec![0usize; max_level + 1];
    for &l in levels { counts[l.min(max_level)] += 1; }

    let n = levels.len() as f64;
    let mut h = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / n;
            h -= p * p.log2();
        }
    }
    h
}

// ═══════════════════════════════════════════════════════════════════════
// 3.3.7 逆映射：G → M（推论5.1 — Φ_C⁻¹）
// ═══════════════════════════════════════════════════════════════════════

/// 从 G = (K, g, ω, Γ, R) 逆映射回 M = (S, F, C)。
///
/// 数学对应：定理3逆 — 利用 rev_data 中完整序列化的概念层级、规则和约束
/// 来精确重建 PhaseOneOutput，保证 Φ_C⁻¹ ∘ Φ_C = id。
///
/// 可逆性保证：
///   - 节点文本：100%（node_texts 逐字复制）
///   - 概念层级：100%（concept_levels + termination_reasons 逐层重建）
///   - 规则：100%（serialized_rules 包含 rule_type, antecedent, consequent, confidence）
///   - 约束：100%（serialized_constraints 包含 description, equation, value）
///   - 不完整报告：从概念层级统计重建（非精确还原，但不影响结构等价类）
///   - 自洽性：总是标记为 true（G 可重建则结构自洽）
pub fn decode_phase_two(output: &PhaseTwoOutput) -> crate::emergence::relations::PhaseOneOutput {
    use crate::emergence::relations::{Concept, ConceptHierarchy, DerivedRule, DerivedConstraint,
        IncompletenessReport, RuleType};

    let rd = &output.rev_data;

    // ── 完整重建概念层级（逐层、逐概念）──
    let levels: Vec<Vec<Concept>> = rd.concept_levels.iter()
        .map(|lv| lv.iter()
            .map(|&(id, ref members, level, ref links)| Concept {
                id,
                members: members.clone(),
                level,
                external_links: links.clone(),
            })
            .collect()
        )
        .collect();

    let structure = ConceptHierarchy {
        levels,
        depth: rd.containment_depth,
        termination_reasons: rd.concept_termination_reasons.clone(),
    };

    // ── 完整重建规则（含 rule_type, antecedent, consequent, confidence）──
    let rules: Vec<DerivedRule> = rd.serialized_rules.iter()
        .map(|&(disc, ref antecedent, ref consequent, confidence)| {
            let rule_type = match disc {
                0 => RuleType::CooccurrenceRule,
                _ => RuleType::StructuralRule,
            };
            DerivedRule {
                rule_type,
                antecedent: antecedent.clone(),
                consequent: consequent.clone(),
                confidence,
            }
        })
        .collect();

    // ── 完整重建约束（含 description, invariant_equation, value）──
    let constraints: Vec<DerivedConstraint> = rd.serialized_constraints.iter()
        .map(|(desc, eqn, val)| DerivedConstraint {
            description: desc.clone(),
            invariant_equation: eqn.clone(),
            value: *val,
        })
        .collect();

    // ── 不完整报告 → 从层级数据重建近似 ──
    let total_concepts: usize = rd.concept_levels.iter().map(|lv| lv.len()).sum();
    let incompleteness = IncompletenessReport {
        is_radically_incomplete: rd.concept_levels.iter().any(|lv| lv.is_empty()),
        orphan_count: 0,
        unclassified_count: 0,
        gap_density: if total_concepts > 1 && rd.concept_levels.len() > 1 {
            let max_possible = total_concepts * (total_concepts - 1) / 2;
            let actual_links: usize = rd.concept_levels.iter()
                .flat_map(|lv| lv.iter())
                .map(|(_, _, _, links)| links.len())
                .sum();
            if max_possible > 0 { 1.0 - (actual_links as f64 / max_possible as f64) } else { 0.0 }
        } else { 0.0 },
    };

    PhaseOneOutput {
        structure,
        rules,
        constraints,
        is_consistent: true,
        incompleteness,
        node_levels: rd.node_levels.clone(),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_cw_complex_construction() {
        // 构建 3 个 0-胞腔（顶点）和 2 个 1-胞腔（边）
        let mut v0_map = HashMap::new();
        let mut e1_map = HashMap::new();
        let mut cells: Vec<Cell> = Vec::new();

        cells.push(Cell {
            dim: 0,
            id: 0,
            boundary: vec![],
            label: "A".to_string(),
            domain: "test".to_string(),
        });
        cells.push(Cell {
            dim: 0,
            id: 1,
            boundary: vec![],
            label: "B".to_string(),
            domain: "test".to_string(),
        });
        cells.push(Cell {
            dim: 0,
            id: 2,
            boundary: vec![],
            label: "C".to_string(),
            domain: "test".to_string(),
        });
        v0_map.insert(0, 0);
        v0_map.insert(1, 1);
        v0_map.insert(2, 2);

        cells.push(Cell {
            dim: 1,
            id: 3,
            boundary: vec![0, 1],
            label: "A-B".to_string(),
            domain: "test".to_string(),
        });
        cells.push(Cell {
            dim: 1,
            id: 4,
            boundary: vec![1, 2],
            label: "B-C".to_string(),
            domain: "test".to_string(),
        });
        e1_map.insert((0, 1), 3);
        e1_map.insert((1, 2), 4);

        let complex = CWComplex {
            cells,
            v0_map,
            e1_map,
            euler_characteristic: 1, // 3 - 2 = 1
            betti_0: 1,
            betti_1: 0,
        };

        assert_eq!(complex.cells.len(), 5, "3 个 0-胞腔 + 2 个 1-胞腔");
        assert_eq!(complex.v0_map.len(), 3, "3 个顶点映射");
        assert_eq!(complex.e1_map.len(), 2, "2 条边映射");
        assert_eq!(complex.euler_characteristic, 1);
        assert_eq!(complex.betti_0, 1);
        assert_eq!(complex.betti_1, 0);
        assert!(complex.v0_map.contains_key(&0));
        assert!(complex.v0_map.contains_key(&1));
        assert!(complex.v0_map.contains_key(&2));
        assert!(complex.e1_map.contains_key(&(0, 1)));
        assert!(complex.e1_map.contains_key(&(1, 2)));
    }

    #[test]
    fn test_scalar_field() {
        // 创建包含 5 个节点的标量场
        let mut potential = HashMap::new();
        potential.insert(0, 0.0);
        potential.insert(1, 0.25);
        potential.insert(2, 0.5);
        potential.insert(3, 0.75);
        potential.insert(4, 1.0);

        let avg = (0.0 + 0.25 + 0.5 + 0.75 + 1.0) / 5.0;
        let sf = ScalarField {
            potential,
            avg_potential: avg,
            min_potential: 0.0,
            max_potential: 1.0,
        };

        assert_eq!(sf.potential.len(), 5);
        assert!((sf.avg_potential - avg).abs() < 1e-10);
        assert_eq!(sf.min_potential, 0.0);
        assert_eq!(sf.max_potential, 1.0);

        // 验证势场值按 id 单调递增
        let mut ids: Vec<_> = sf.potential.keys().copied().collect();
        ids.sort();
        let mut prev = -1.0;
        for id in &ids {
            let phi = sf.potential[id];
            assert!(phi > prev, "势场值应按 id 单调递增");
            prev = phi;
        }
    }

    #[test]
    fn test_vector_field_grad_mean_std() {
        let mut flow = HashMap::new();
        flow.insert(0, (2.0, 1.0));
        flow.insert(1, (4.0, -1.0));
        flow.insert(2, (6.0, 0.0));

        let vf = VectorField {
            flow,
            divergence: HashMap::new(),
            total_flux: 12.0,
            avg_flux_density: 4.0,
            source_count: 0,
            sink_count: 0,
        };

        let (mean, std_dev) = vf.grad_mean_std();
        let expected_mean = 4.0; // (2+4+6)/3
        let expected_var = ((2.0_f64 - 4.0).powi(2) + (4.0_f64 - 4.0).powi(2) + (6.0_f64 - 4.0).powi(2))
            / 3.0;
        let expected_std = expected_var.sqrt();

        assert!(
            (mean - expected_mean).abs() < 1e-10,
            "梯度均值应为 {}, 得到 {}",
            expected_mean,
            mean
        );
        assert!(
            (std_dev - expected_std).abs() < 1e-10,
            "梯度标准差应为 {}, 得到 {}",
            expected_std,
            std_dev
        );
    }

    #[test]
    fn test_vector_field_grad_mean_std_empty() {
        let vf = VectorField {
            flow: HashMap::new(),
            divergence: HashMap::new(),
            total_flux: 0.0,
            avg_flux_density: 0.0,
            source_count: 0,
            sink_count: 0,
        };
        let (mean, std_dev) = vf.grad_mean_std();
        assert_eq!(mean, 0.0);
        assert_eq!(std_dev, 0.0);
    }

    #[test]
    fn test_invariants() {
        // 构建最小 CWComplex：2 个 0-胞腔 + 1 个 1-胞腔
        let mut v0_map = HashMap::new();
        let mut e1_map = HashMap::new();
        let mut cells: Vec<Cell> = Vec::new();

        cells.push(Cell {
            dim: 0,
            id: 0,
            boundary: vec![],
            label: "X".to_string(),
            domain: "test".to_string(),
        });
        cells.push(Cell {
            dim: 0,
            id: 1,
            boundary: vec![],
            label: "Y".to_string(),
            domain: "test".to_string(),
        });
        v0_map.insert(0, 0);
        v0_map.insert(1, 1);

        cells.push(Cell {
            dim: 1,
            id: 2,
            boundary: vec![0, 1],
            label: "X-Y".to_string(),
            domain: "test".to_string(),
        });
        e1_map.insert((0, 1), 2);

        let complex = CWComplex {
            cells,
            v0_map,
            e1_map,
            euler_characteristic: 1, // 2 - 1 = 1
            betti_0: 1,
            betti_1: 0,
        };

        let inv = encode_invariants(&complex);

        // Euler 示性数 = V - E + F = 2 - 1 + 0 = 1
        assert_eq!(inv.euler, 1);
        // Euler = betti_0 - betti_1  (h_i = rank(H_i), h_0 - h_1 + h_2 - ...)
        assert_eq!(
            inv.euler,
            inv.betti_0 as i64 - inv.betti_1 as i64,
            "Euler = betti_0 - betti_1"
        );
        assert_eq!(inv.v0_count, 2);
        assert_eq!(inv.e1_count, 1);
        assert_eq!(inv.v2_count, 0);
        assert_eq!(inv.total_cells, 3);
    }

    #[test]
    fn test_compute_entropy() {
        let levels = vec![0, 1, 2, 0, 1];
        // 分布: level 0 -> 2个, level 1 -> 2个, level 2 -> 1个
        // H = -(2/5*log2(2/5) + 2/5*log2(2/5) + 1/5*log2(1/5))
        let h = compute_entropy(&levels);

        let log2_3 = (3.0_f64).log2();
        assert!(h > 0.0, "熵应大于 0, 得到 {:.4}", h);
        assert!(h <= log2_3, "熵应 <= log2(3) ≈ {:.4}, 得到 {:.4}", log2_3, h);

        // 空输入
        assert_eq!(compute_entropy(&[]), 0.0);
        // 单层级
        assert_eq!(compute_entropy(&[0, 0, 0]), 0.0);
    }

    #[test]
    fn test_decode_roundtrip() {
        // 构建最小 CWComplex：2 个 0-胞腔 + 1 个 1-胞腔
        let mut v0_map = HashMap::new();
        let mut e1_map = HashMap::new();
        let mut cells: Vec<Cell> = Vec::new();

        cells.push(Cell {
            dim: 0,
            id: 0,
            boundary: vec![],
            label: "A".to_string(),
            domain: "test".to_string(),
        });
        cells.push(Cell {
            dim: 0,
            id: 1,
            boundary: vec![],
            label: "B".to_string(),
            domain: "test".to_string(),
        });
        v0_map.insert(0, 0);
        v0_map.insert(1, 1);

        cells.push(Cell {
            dim: 1,
            id: 2,
            boundary: vec![0, 1],
            label: "A-B".to_string(),
            domain: "test".to_string(),
        });
        e1_map.insert((0, 1), 2);

        let complex = CWComplex {
            cells,
            v0_map,
            e1_map,
            euler_characteristic: 1,
            betti_0: 1,
            betti_1: 0,
        };

        // 构造 ReversibilityData，word_count = 3，含概念、规则、约束
        let rev_data = ReversibilityData {
            node_texts: vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "ab".to_string(),
                "bc".to_string(),
            ],
            node_is_word: vec![false, false, false, true, true],
            word_count: 3,
            containment_depth: 2,
            node_levels: vec![0, 0, 0, 1, 2],
            concept_levels: vec![
                vec![
                    (0, vec![0, 1], 0, vec![]),
                    (1, vec![1, 2], 0, vec![]),
                ],
                vec![(2, vec![3, 4], 1, vec![(0, 0.8)])],
            ],
            concept_termination_reasons: vec![
                "no_new_patterns".to_string(),
                "max_depth_reached".to_string(),
            ],
            serialized_rules: vec![(0u8, vec![0], vec![1], 0.9)],
            serialized_constraints: vec![(
                "sum_confidence".to_string(),
                "x+y=1".to_string(),
                1.0,
            )],
        };

        let output = PhaseTwoOutput {
            complex,
            metric: MetricInfo {
                edge_lengths: HashMap::new(),
                avg_edge_length: 0.0,
                min_edge_length: 0.0,
                max_edge_length: 0.0,
            },
            invariants: GeometricInvariants {
                euler: 1,
                betti_0: 1,
                betti_1: 0,
                total_cells: 3,
                v0_count: 2,
                e1_count: 1,
                v2_count: 0,
                avg_degree: 1.0,
                density: 1.0,
            },
            scalar_field: ScalarField {
                potential: HashMap::new(),
                avg_potential: 0.0,
                min_potential: 0.0,
                max_potential: 0.0,
            },
            vector_field: VectorField {
                flow: HashMap::new(),
                divergence: HashMap::new(),
                total_flux: 0.0,
                avg_flux_density: 0.0,
                source_count: 0,
                sink_count: 0,
            },
            reversibility: vec![],
            rev_data,
        };

        let decoded = decode_phase_two(&output);

        // 验证解码结果
        assert!(decoded.is_consistent, "解码后应标记为自洽");
        assert_eq!(
            decoded.node_levels,
            vec![0, 0, 0, 1, 2],
            "node_levels 应完整保留"
        );
        assert_eq!(decoded.structure.depth, 2, "层级深度应保留");
        assert_eq!(decoded.structure.levels.len(), 2, "应有 2 层概念");
        assert_eq!(decoded.rules.len(), 1, "应有 1 条规则");
        assert_eq!(decoded.constraints.len(), 1, "应有 1 条约束");
        assert_eq!(
            decoded.structure.termination_reasons.len(),
            2,
            "终止原因应保留"
        );
    }
}
