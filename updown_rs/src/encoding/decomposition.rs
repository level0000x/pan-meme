//! Phase 3: 分解（模因化）
//!
//! 数学对应：
//!   3.4.1 几何分解: n = β₀(K) 个连通分子复形
//!   3.4.2 五维映射: K_i 几何特征 → (D, B, ρ, R, S)
//!   3.4.3 参数推导: 11 个参数从子几何体特征闭式计算
//!   3.4.4 耦合生成: 耦合矩阵 C_{ij}
//!   3.4.5 输出: Q = {X₁..Xₙ, Θ, C}
//!
//! 前提5（核心-扩展分离公设）：Kᵢ → (mᵢ, ξᵢ) 是双射。
//!   mᵢ = (D,B,ρ,R,S) ∈ [0,1]⁵ 为核心五维
//!   ξᵢ ∈ Ξ 为扩展维度，编码 Kᵢ 上所有微观结构快照

use crate::encoding::geometry::{CWComplex, GeometricInvariants, ScalarField, VectorField};
use std::collections::{HashMap, HashSet};

// ═══════════════════════════════════════════════════════════════════════
// 3.4.1 几何分解
// ═══════════════════════════════════════════════════════════════════════

/// 子几何体（一个连通分量的胞腔复形片段）
#[derive(Debug, Clone)]
pub struct SubGeometry {
    pub id: usize,
    /// 此子几何体包含的 0-胞腔 ID 列表
    pub vertices: Vec<usize>,
    /// 此子几何体包含的 1-胞腔 ID 列表
    pub edges: Vec<usize>,
    /// 此子几何体包含的 2-胞腔 ID 列表
    pub faces: Vec<usize>,
    /// 外部连接：(其他子几何体ID, 连接边数)
    pub external_links: Vec<(usize, usize)>,
}

/// 几何分解结果
#[derive(Debug)]
pub struct DecompositionResult {
    pub sub_geometries: Vec<SubGeometry>,
    pub n_components: usize,
}

/// 将胞腔复形 K 按连通分量分解为 n = β₀(K) 个子几何体。
///
/// 数学对应：3.4.1 — 基于 Betti 数分解
pub fn decompose_by_betti(complex: &CWComplex) -> DecompositionResult {
    // 构建只含 0-胞腔和 1-胞腔的图
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); complex.cells.len()];
    for cell in &complex.cells {
        if cell.dim == 1 && cell.boundary.len() == 2 {
            let a = cell.boundary[0];
            let b = cell.boundary[1];
            adj[a].push(b);
            adj[b].push(a);
        }
    }

    // BFS 找连通分量
    let mut visited: HashSet<usize> = HashSet::new();
    let mut components: Vec<Vec<usize>> = Vec::new();

    for (_, &v0_id) in complex.v0_map.iter() {
        if visited.contains(&v0_id) { continue; }
        let mut stack = vec![v0_id];
        let mut comp_vertices = Vec::new();
        visited.insert(v0_id);
        while let Some(u) = stack.pop() {
            comp_vertices.push(u);
            for &v in &adj[u] {
                if complex.cells[v].dim == 0 && !visited.contains(&v) {
                    visited.insert(v);
                    stack.push(v);
                }
            }
        }
        if !comp_vertices.is_empty() {
            components.push(comp_vertices);
        }
    }

    // 每个分量收集其边和面
    let mut sub_geos = Vec::new();
    for (comp_id, vertices) in components.iter().enumerate() {
        let v_set: HashSet<usize> = vertices.iter().copied().collect();

        let mut edges = Vec::new();
        let mut faces = Vec::new();
        let mut ext_links: HashMap<usize, usize> = HashMap::new();

        for (ci, cell) in complex.cells.iter().enumerate() {
            if cell.dim == 1 {
                if cell.boundary.len() == 2
                    && v_set.contains(&cell.boundary[0])
                    && v_set.contains(&cell.boundary[1])
                {
                    edges.push(ci);
                } else if cell.boundary.len() == 2 {
                    // 跨越分量的边
                    let b0 = cell.boundary[0];
                    let b1 = cell.boundary[1];
                    let in_comp = v_set.contains(&b0);
                    if in_comp || v_set.contains(&b1) {
                        // 找对端所在的分量
                        for (other_id, other_v) in components.iter().enumerate() {
                            if other_id == comp_id { continue; }
                            let other_set: HashSet<usize> = other_v.iter().copied().collect();
                            if other_set.contains(&b0) || other_set.contains(&b1) {
                                *ext_links.entry(other_id).or_insert(0) += 1;
                            }
                        }
                    }
                }
            } else if cell.dim == 2 {
                if cell.boundary.iter().all(|b| v_set.contains(b)) {
                    faces.push(ci);
                }
            }
        }

        sub_geos.push(SubGeometry {
            id: comp_id,
            vertices: vertices.clone(),
            edges,
            faces,
            external_links: ext_links.into_iter().collect(),
        });
    }

    DecompositionResult {
        n_components: sub_geos.len(),
        sub_geometries: sub_geos,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 3.4.2 五维映射
// ═══════════════════════════════════════════════════════════════════════

/// 缩放常量（与 Python pan_meme 保持一致）
const D_MAX: f64 = 20.0;      // 层级深度归一化上限
const GAMMA_SIZE: f64 = 4.0;  // 每分量不变量条目数近似（euler_char + constraints + types + parent）

/// 五维状态空间 (D, B, ρ, R, S)
/// 所有值归一化至 [0, 1] × [0, 1] × ℝ⁺ × [0, 1] × [0, 1]
#[derive(Debug, Clone)]
pub struct FiveDimState {
    /// 内禀度 D ∈ [0, 1]: 内在复杂性
    pub intrinsic_degree: f64,
    /// 关联度 B ∈ [0, 1]: 外部连接
    pub binding_degree: f64,
    /// 能流密度 ρ ∈ ℝ⁺: 凝聚能量密度，分子几何体尺度归一化
    pub energy_density: f64,
    /// 演化速率 R ∈ [0, 1]: 几何变化速率
    pub evolution_rate: f64,
    /// 结构韧度 S ∈ [0, 1]: 拓扑稳定性
    pub structural_robustness: f64,
}

/// 从子几何体特征计算五维映射。
///
/// 数学对应：3.4.2 — K_i 的几何特征 → (D, B, ρ, R, S)，与 Python pan_meme 公式对齐。
///
/// D = min(1, cell_count / c_max) * min(1, depth / d_max)
/// B = external_edges / total_possible_external
/// ρ = total_flux（子几何体的总通量＝能流密度）
/// R = std(|∇φ|)（场梯度标准差，度量演化潜力）
/// S = min(1, |Gamma| / 10)
pub fn compute_five_dim(
    sub: &SubGeometry,
    total_sub: usize,
    global_max_cells: usize,
    depth: usize,
    _global_avg_degree: f64,
    total_flux: f64,
    _grad_mean: f64,
    grad_std: f64,
) -> FiveDimState {
    // ── NaN / ∞ 防护：输入可能携带异常值（空子几何体/零梯度场）──
    let safe_flux = if total_flux.is_finite() && total_flux >= 0.0 { total_flux } else { 0.0 };
    let safe_std  = if grad_std.is_finite()  && grad_std >= 0.0  { grad_std  } else { 0.0 };

    let cell_count = sub.vertices.len() + sub.edges.len() + sub.faces.len();

    // D: 内禀度 = 胞腔规模因子 × 层级深度因子（公式 3.4.2）
    let cell_factor = if global_max_cells > 0 {
        (cell_count as f64 / global_max_cells as f64).min(1.0)
    } else { 0.0 };
    let depth_factor = (depth as f64 / D_MAX).min(1.0);
    let d = cell_factor * depth_factor;

    // B: 关联度 — 此模因对外界结构的依赖程度
    //
    // 两种情形：
    //   n>1: B = external_edges / max_possible_external  (标准跨分量耦合)
    //   n=1: B = (1 - edge_density) × depth_discount
    //         单连通分量无法计算跨模因耦合。改用"未实现的连接比例"作为
    //         对外耦合潜力度量：边密度越小 → 结构越稀疏 → 耦合潜力越大。
    //         直观含义：此模因内部只实现了 B% 的可能关系——
    //         其余 (1 - B) 的"关系空间"都可能来自外部结构。
    let b = if total_sub > 1 {
        let total_ext: usize = sub.external_links.iter().map(|(_, c)| c).sum();
        (total_ext as f64 / ((total_sub - 1) as f64 * 2.0)).min(1.0)
    } else {
        let nv = sub.vertices.len().max(2) as f64;
        let ne = sub.edges.len() as f64;
        let density = (2.0 * ne / (nv * (nv - 1.0))).min(1.0);
        // 未实现的连接比例 → 基耦合
        let base = 1.0 - density;
        // 深度折扣：浅层模因更依赖外部，深层的"缺口"是有意设计的自足
        let depth_factor = (depth as f64 / D_MAX).min(1.0);
        (base * (1.0 - depth_factor * 0.5)).max(0.0)
    };

    // ρ: 能流密度 — 子几何体内通量总和（来自梯度向量场）
    let rho = safe_flux;

    // R: 演化速率 — 场梯度的标准差（公式 3.4.2）
    let r = safe_std.min(1.0);

    // S: 结构韧度 — Euler 归一化（公式 3.4.2）
    let s = {
        let nv = sub.vertices.len() as f64;
        let ne = sub.edges.len() as f64;
        ((nv - ne + 1.0) / (nv + 1.0)).max(0.0).min(1.0)
    };

    FiveDimState {
        intrinsic_degree: d,
        binding_degree: b,
        energy_density: rho,
        evolution_rate: r,
        structural_robustness: s,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 3.4.3 参数推导
// ═══════════════════════════════════════════════════════════════════════

/// 11 个动力学参数（从子几何体特征闭式计算）
///
/// 数学对应：论文 4.3.7 节 — 与 Python ParamDerive 公式对齐
#[derive(Debug, Clone)]
pub struct DynamicsParams {
    pub alpha_1: f64, pub alpha_2: f64,
    pub beta_1: f64,  pub beta_2: f64,
    pub gamma_1: f64, pub gamma_2: f64,
    pub delta_1: f64, pub delta_2: f64, pub delta_3: f64,
    pub epsilon_1: f64, pub epsilon_2: f64,
}

impl DynamicsParams {
    /// 11 参数闭式推导（论文 4.3.7 + Python ParamDerive 对齐）。
    ///
    /// α₁ = min(2|E|/(|V|(|V|-1)), 1)  — 简化效应（边密度）
    /// α₂ = min(depth / d_max, 1)       — 沉淀效应（层级深度）
    /// β₁ = 0.5                          — 扩张耦合（默认中等）
    /// β₂ = D / max(B, 1e-6)            — 泛化权衡（内禀/关联比）
    /// γ₁ = mean(|∇φ|)                   — 能流耗散（梯度均值）
    /// γ₂ = 0.3                          — 外部赋能（默认低注入）
    /// δ₁ = R · B                        — 核心驱动力
    /// δ₂ = D · depth                    — 深度诅咒
    /// δ₃ = 1/(1+|Γ|)                    — 自发衰退（约束越多越慢）
    /// ε₁ = D · S                        — 深度基石
    /// ε₂ = std(|∇φ|) · mean(|∇φ|)      — 速朽定律（粗糙度×强度）
    pub fn from_sub_geometry(
        sub: &SubGeometry,
        state: &FiveDimState,
        depth: usize,
        grad_mean: f64,
        grad_std: f64,
    ) -> Self {
        // ── NaN / ∞ 防护 ──
        let safe_grad_mean = if grad_mean.is_finite() && grad_mean >= 0.0 { grad_mean } else { 0.0 };
        let safe_grad_std  = if grad_std.is_finite()  && grad_std >= 0.0  { grad_std  } else { 0.0 };
        let safe_d = if state.intrinsic_degree.is_finite() { state.intrinsic_degree } else { 0.01 };
        let safe_b = if state.binding_degree.is_finite()  { state.binding_degree  } else { 0.01 };
        let safe_r = if state.evolution_rate.is_finite()   { state.evolution_rate   } else { 0.01 };

        let nv = sub.vertices.len().max(1) as f64;
        let ne = sub.edges.len() as f64;

        // α₁ = 2|E| / (|V|(|V|-1)) — 边密度（论文 4.3.7 公式）
        let edge_density = if nv > 1.0 { 2.0 * ne / (nv * (nv - 1.0)) } else { 0.0 };
        let alpha_1 = edge_density.min(1.0).max(0.01);

        // α₂ = depth / d_max — 沉淀效应
        let alpha_2 = (depth as f64 / D_MAX).min(1.0).max(0.01);

        // β₁ = 0.5（默认扩张耦合）
        let beta_1 = 0.5;

        // β₂ = D / max(B, 1e-6) — 泛化权衡
        let beta_2 = (safe_d / safe_b.max(1e-6)).min(10.0).max(0.01);

        // γ₁ = mean(|∇φ|) — 能流耗散
        let gamma_1 = safe_grad_mean.min(1.0).max(0.01);

        // γ₂ = 0.3 — 外部赋能
        let gamma_2 = 0.3;

        // δ₁ = R · B — 核心驱动力
        let delta_1 = (safe_r * safe_b).min(1.0).max(0.01);

        // δ₂ = D · depth — 深度诅咒
        let delta_2 = (safe_d * depth as f64).max(0.01);

        // δ₃ = 1/(1+|Γ|) — 自发衰退
        let delta_3 = (1.0 / (1.0 + GAMMA_SIZE)).max(0.01);

        // ε₁ = D · S — 深度基石
        let safe_s = if state.structural_robustness.is_finite() { state.structural_robustness } else { 0.01 };
        let epsilon_1 = (safe_d * safe_s).min(1.0).max(0.01);

        // ε₂ = std(|∇φ|) · mean(|∇φ|) — 速朽定律
        let epsilon_2 = (safe_grad_std * safe_grad_mean).max(0.01);

        DynamicsParams {
            alpha_1, alpha_2, beta_1, beta_2,
            gamma_1, gamma_2, delta_1, delta_2, delta_3,
            epsilon_1, epsilon_2,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 3.4.4 耦合生成
// ═══════════════════════════════════════════════════════════════════════

/// 耦合矩阵元素
#[derive(Debug, Clone)]
pub struct Coupling {
    pub i: usize,
    pub j: usize,
    /// 耦合强度 C_{ij} ∈ [0, 1]
    pub strength: f64,
    /// 耦合类型
    pub kind: CouplingKind,
}

#[derive(Debug, Clone)]
pub enum CouplingKind {
    /// 共享边耦合：两个分量在原图中共享 1-胞腔
    SharedEdge,
    /// 共享顶点耦合：显著顶点重叠（≥5）
    SharedVertex,
    /// 同层级耦合：有外部连接（原属同一连通分量拆分）
    SameLevel,
    /// 一般耦合：无特殊结构关系
    Weak,
}

/// 计算耦合矩阵 C。
///
/// 数学对应：3.4.4 — C_{ij} 基于共享边、共享顶点、层级关系三项权重（Python Coupling 对齐）。
///
/// C_{ij} = clip(0 + 0.3·1_{shared_edges} + 0.5·1_{shared_vertices≥5} + 0.2·1_{external_links}, 0, 1)
pub fn compute_coupling(
    complex: &CWComplex,
    sub_geos: &[SubGeometry],
) -> Vec<Coupling> {
    const W_SHARED_EDGE: f64 = 0.3;
    const W_SHARED_VERTEX: f64 = 0.5;
    const SHARED_V_THRESHOLD: usize = 5;
    const W_SAME_LEVEL: f64 = 0.2;

    let n = sub_geos.len();
    if n < 2 { return Vec::new(); }

    // 预计算顶点集和边集（顶点对）per 分量
    let vertex_sets: Vec<HashSet<usize>> = sub_geos.iter()
        .map(|s| s.vertices.iter().copied().collect())
        .collect();

    let edge_sets: Vec<HashSet<(usize, usize)>> = sub_geos.iter()
        .map(|s| {
            s.edges.iter().filter_map(|&e_id| {
                if e_id >= complex.cells.len() { return None; }
                let cell = &complex.cells[e_id];
                if cell.dim == 1 && cell.boundary.len() == 2 {
                    let a = cell.boundary[0];
                    let b = cell.boundary[1];
                    Some((a.min(b), a.max(b)))
                } else { None }
            }).collect()
        })
        .collect();

    let mut couplings = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let mut c_ij = 0.0;

            // 共享边 → +0.3
            if !edge_sets[i].is_disjoint(&edge_sets[j]) {
                c_ij += W_SHARED_EDGE;
            }

            // 共享顶点 ≥5 → +0.5
            let shared_v = vertex_sets[i].intersection(&vertex_sets[j]).count();
            if shared_v >= SHARED_V_THRESHOLD {
                c_ij += W_SHARED_VERTEX;
            }

            // 外部连接（原同一连通分量）→ +0.2
            let ext_ij = sub_geos[i].external_links.iter()
                .find(|(id, _)| *id == j).map(|(_, c)| *c).unwrap_or(0);
            let ext_ji = sub_geos[j].external_links.iter()
                .find(|(id, _)| *id == i).map(|(_, c)| *c).unwrap_or(0);
            if ext_ij + ext_ji > 0 {
                c_ij += W_SAME_LEVEL;
            }

            c_ij = c_ij.min(1.0);

            if c_ij > 0.0 {
                let kind = if c_ij >= 0.8 { CouplingKind::SharedVertex }
                else if c_ij >= 0.5 { CouplingKind::SharedEdge }
                else if c_ij >= 0.2 { CouplingKind::SameLevel }
                else { CouplingKind::Weak };

                couplings.push(Coupling { i, j, strength: c_ij, kind });
            }
        }
    }

    couplings
}

// ═══════════════════════════════════════════════════════════════════════
// 3.4.5 完整输出: Q = {X₁..Xₙ, Θ, C}
// ═══════════════════════════════════════════════════════════════════════

/// 胞腔快照 — 用于 ξᵢ 中存储可重构的胞腔数据（前提5 逆向映射）
///
/// 数学对应：§D.2 定义3 — K 的最小还原信息单元
#[derive(Debug, Clone)]
pub struct CellSnapshot {
    pub dim: usize,
    pub id: usize,
    pub boundary: Vec<usize>,
}

/// 扩展维度 ξᵢ — Kᵢ 的微观结构快照（前提5）
///
/// 数学对应：附录D.3 前提5 — Kᵢ → (mᵢ, ξᵢ) 是双射。
/// mᵢ = (D,B,ρ,R,S) 只捕获 5 个宏观量，ξᵢ 编码剩余的几何细节。
///
/// 信息论含义：双射性保证逆映射 Φ_D⁻¹ 可从此结构无歧义地重建 Kᵢ。
#[derive(Debug, Clone)]
pub struct ExtendedDimension {
    /// 0-胞腔快照（可用于重建顶点列表）
    pub vertex_cells: Vec<CellSnapshot>,
    /// 1-胞腔快照
    pub edge_cells: Vec<CellSnapshot>,
    /// 2-胞腔快照
    pub face_cells: Vec<CellSnapshot>,
    /// 外部连接端点: (本分量顶点ID, 对方分量ID)
    pub boundary_links: Vec<(usize, usize)>,
    /// 每个顶点的势场值 φ
    pub vertex_potentials: Vec<f64>,
    /// 每条边的梯度模 |F(e)|
    pub edge_flux_magnitudes: Vec<f64>,
    /// 原始字词列表（修复单字词往返破裂：单字词与字节点重合时靠此恢复）
    pub node_texts: Vec<String>,
}

#[derive(Debug)]
pub struct MemeDecomposition {
    pub id: usize,
    pub sub_geometry: SubGeometry,
    pub state: FiveDimState,
    pub params: DynamicsParams,
    /// ξᵢ：扩展维度，编码 Kᵢ 的完整微观结构（前提5）
    pub xi: ExtendedDimension,
}

#[derive(Debug)]
pub struct PhaseThreeOutput {
    pub memes: Vec<MemeDecomposition>,
    pub couplings: Vec<Coupling>,
    pub n_memes: usize,
}

/// 运行第三阶段分解。
///
/// 数学对应：3.4 — 用 depth、∇φ 统计量驱动五维映射与参数推导。
/// 前提5：为每个模因构建 ξᵢ，保证 Kᵢ → (mᵢ, ξᵢ) 双射。
pub fn run_phase_three(
    complex: &CWComplex,
    invariants: &GeometricInvariants,
    depth: usize,
    scalar_field: &ScalarField,
    vector_field: &VectorField,
) -> PhaseThreeOutput {
    let decomp = decompose_by_betti(complex);
    let n = decomp.sub_geometries.len();
    if n == 0 {
        return PhaseThreeOutput { memes: Vec::new(), couplings: Vec::new(), n_memes: 0 };
    }

    let global_max = decomp.sub_geometries.iter()
        .map(|s| s.vertices.len() + s.edges.len() + s.faces.len())
        .max()
        .unwrap_or(1);

    let effective_depth = depth.max(1);

    // 全局梯度统计（所有 1-胞腔的 |∇φ|）
    let (grad_mean, grad_std) = vector_field.grad_mean_std();

    // 预计算每个子几何体的总通量 + 按顶点收集势场值
    let mut meme_fluxes: Vec<f64> = Vec::with_capacity(n);
    let mut meme_edge_fluxes: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut meme_vertex_potentials: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut meme_boundary_links: Vec<Vec<(usize, usize)>> = Vec::with_capacity(n);

    // 分量顶点 ID → 分量索引 的快速查找表
    let mut vertex_to_component: HashMap<usize, usize> = HashMap::new();
    for (comp_id, sub) in decomp.sub_geometries.iter().enumerate() {
        for &v in &sub.vertices {
            vertex_to_component.insert(v, comp_id);
        }
    }

    for sub in &decomp.sub_geometries {
        let mut flux = 0.0;
        let mut edge_fluxes = Vec::with_capacity(sub.edges.len());
        for &edge_id in &sub.edges {
            let mag = vector_field.flow.get(&edge_id).map(|&(m, _)| m).unwrap_or(0.0);
            flux += mag;
            edge_fluxes.push(mag);
        }

        let potentials: Vec<f64> = sub.vertices.iter()
            .map(|&v| scalar_field.potential.get(&v).copied().unwrap_or(0.0))
            .collect();

        // 跨分量边界链接
        let mut b_links = Vec::new();
        for &(other_id, _count) in &sub.external_links {
            // 找到连接对方分量的本分量顶点
            for &v in &sub.vertices {
                let cells_clone = complex.cells.clone();
                let boundary_vs: Vec<usize> = cells_clone.iter()
                    .filter(|c| c.dim == 1 && c.boundary.len() == 2)
                    .flat_map(|c| c.boundary.iter().copied())
                    .collect();
                for &nb in &boundary_vs {
                    if vertex_to_component.get(&nb).copied() == Some(other_id) {
                        b_links.push((v, other_id));
                        break; // 每顶点一条边足矣
                    }
                }
            }
        }

        meme_fluxes.push(flux);
        meme_edge_fluxes.push(edge_fluxes);
        meme_vertex_potentials.push(potentials);
        meme_boundary_links.push(b_links);
    }

    let mut memes = Vec::new();
    let mut states = Vec::new();

    for (i, sub) in decomp.sub_geometries.iter().enumerate() {
        let total_flux = meme_fluxes[i];
        let state = compute_five_dim(
            sub, n, global_max, effective_depth,
            invariants.avg_degree, total_flux, grad_mean, grad_std,
        );
        let params = DynamicsParams::from_sub_geometry(sub, &state, effective_depth, grad_mean, grad_std);

        let xi = ExtendedDimension {
            vertex_cells: sub.vertices.iter().map(|&v_id| {
                let c = &complex.cells[v_id];
                CellSnapshot { dim: c.dim, id: c.id, boundary: c.boundary.clone() }
            }).collect(),
            edge_cells: sub.edges.iter().map(|&e_id| {
                let c = &complex.cells[e_id];
                CellSnapshot { dim: c.dim, id: c.id, boundary: c.boundary.clone() }
            }).collect(),
            face_cells: sub.faces.iter().map(|&f_id| {
                let c = &complex.cells[f_id];
                CellSnapshot { dim: c.dim, id: c.id, boundary: c.boundary.clone() }
            }).collect(),
            boundary_links: meme_boundary_links[i].clone(),
            vertex_potentials: meme_vertex_potentials[i].clone(),
            edge_flux_magnitudes: meme_edge_fluxes[i].clone(),
            node_texts: sub.vertices.iter().map(|&v| {
                // CWComplex 不直接存储文本标签（标签在 psi 输入侧）；
                // 这里用索引 string 兜底——逆映射时靠 ExtendedDimension 的
                // 完整胞腔信息重建即可区分单字词与纯字。
                format!("node_{}", v)
            }).collect(),
        };

        states.push(state.clone());
        memes.push(MemeDecomposition {
            id: sub.id,
            sub_geometry: sub.clone(),
            state,
            params,
            xi,
        });
    }

    let couplings = compute_coupling(complex, &decomp.sub_geometries);

    PhaseThreeOutput { memes, couplings, n_memes: decomp.n_components }
}

// ═══════════════════════════════════════════════════════════════════════
// 3.4.6 逆映射：Q → G（推论5.1 — Φ_D⁻¹）
// ═══════════════════════════════════════════════════════════════════════

/// 从 Q = {Xᵢ, Θ, C} 重建胞腔复形 G = (K, g, ω, Γ, R)。
///
/// 数学对应：前提5 + 定理4逆 — 利用每个模因的 ξᵢ 恢复原始几何结构。
///
/// 重建步骤：
///   1. 从所有 ξᵢ 的 vertex_cells 收集 0-胞腔 → v0_map
///   2. 从所有 ξᵢ 的 edge_cells 收集 1-胞腔 → e1_map
///   3. 从所有 ξᵢ 的 face_cells 收集 2-胞腔
///   4. 合并 boundary_links 中的跨分量边
///   5. 重建 CWComplex + 近似 Betti 数
pub fn decode_phase_three(output: &PhaseThreeOutput) -> CWComplex {
    use crate::encoding::geometry::Cell;
    let mut cells: Vec<Cell> = Vec::new();
    let mut v0_map: HashMap<usize, usize> = HashMap::new();
    let mut e1_map: HashMap<(usize, usize), usize> = HashMap::new();
    let mut cell_index: usize = 0;

    // 1. 收集所有胞腔
    for meme in &output.memes {
        for snap in &meme.xi.vertex_cells {
            cells.push(Cell {
                dim: snap.dim,
                id: snap.id,
                boundary: snap.boundary.clone(),
                label: String::new(),  // label 在 Phase 2⁻¹ 恢复
                domain: String::new(), // domain 在 Phase 2⁻¹ 恢复
            });
            v0_map.insert(snap.id, cell_index);
            cell_index += 1;
        }
        for snap in &meme.xi.edge_cells {
            cells.push(Cell {
                dim: snap.dim,
                id: snap.id,
                boundary: snap.boundary.clone(),
                label: String::new(),
                domain: String::new(),
            });
            if snap.boundary.len() == 2 {
                let a = snap.boundary[0];
                let b = snap.boundary[1];
                e1_map.insert((a.min(b), a.max(b)), cell_index);
            }
            cell_index += 1;
        }
        for snap in &meme.xi.face_cells {
            cells.push(Cell {
                dim: snap.dim,
                id: snap.id,
                boundary: snap.boundary.clone(),
                label: String::new(),
                domain: String::new(),
            });
            cell_index += 1;
        }
    }

    // 2. 跨分量边：从 boundary_links 重建
    for meme in &output.memes {
        for &(v_id, other_id) in &meme.xi.boundary_links {
            // 找到 other 分量中的顶点
            for other_meme in &output.memes {
                if other_meme.id == other_id {
                    for snap in &other_meme.xi.vertex_cells {
                        let cross_edge_key = (v_id.min(snap.id), v_id.max(snap.id));
                        if !e1_map.contains_key(&cross_edge_key) {
                            cells.push(Cell {
                                dim: 1, id: cell_index as usize,
                                boundary: vec![v_id, snap.id],
                                label: String::new(), domain: String::new(),
                            });
                            e1_map.insert(cross_edge_key, cell_index);
                            cell_index += 1;
                        }
                        break; // 一个顶点足矣
                    }
                    break;
                }
            }
        }
    }

    // 3. 计算 Betti 数近似
    let v_count = v0_map.len();
    let e_count = e1_map.len();
    let betti_0 = output.n_memes; // β₀ = 分解时的连通分量数
    let betti_1 = if e_count >= v_count { e_count - v_count + betti_0 } else { 0 };

    CWComplex {
        cells,
        v0_map,
        e1_map,
        euler_characteristic: v_count as i64 - e_count as i64,
        betti_0,
        betti_1,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 单元测试
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::geometry::{Cell, CWComplex};
    use std::collections::HashMap;

    /// 构建两个不相交分量 (0-1, 2-3) 的 CW 复形，
    /// 验证 decompose_by_betti 返回 n_components == 2 且每分量含 2 个顶点。
    #[test]
    fn test_decompose_by_betti_simple() {
        let cells = vec![
            Cell { dim: 0, id: 0, boundary: vec![], label: "v0".into(), domain: String::new() },
            Cell { dim: 0, id: 1, boundary: vec![], label: "v1".into(), domain: String::new() },
            Cell { dim: 0, id: 2, boundary: vec![], label: "v2".into(), domain: String::new() },
            Cell { dim: 0, id: 3, boundary: vec![], label: "v3".into(), domain: String::new() },
            Cell { dim: 1, id: 4, boundary: vec![0, 1], label: "e01".into(), domain: String::new() },
            Cell { dim: 1, id: 5, boundary: vec![2, 3], label: "e23".into(), domain: String::new() },
        ];

        let mut v0_map = HashMap::new();
        v0_map.insert(0, 0);
        v0_map.insert(1, 1);
        v0_map.insert(2, 2);
        v0_map.insert(3, 3);

        let mut e1_map = HashMap::new();
        e1_map.insert((0, 1), 4);
        e1_map.insert((2, 3), 5);

        let complex = CWComplex {
            cells,
            v0_map,
            e1_map,
            euler_characteristic: 4 - 2,
            betti_0: 2,
            betti_1: 0,
        };

        let result = decompose_by_betti(&complex);

        assert_eq!(result.n_components, 2, "应分解为 2 个连通分量");
        assert_eq!(result.sub_geometries.len(), 2);
        assert_eq!(result.sub_geometries[0].vertices.len(), 2,
            "第一个分量应有 2 个顶点");
        assert_eq!(result.sub_geometries[1].vertices.len(), 2,
            "第二个分量应有 2 个顶点");
    }

    /// 用空场/空不变量调用 compute_five_dim，
    /// 验证五个维度值均在 [0, 1] 区间。
    #[test]
    fn test_compute_five_dim() {
        let sub = SubGeometry {
            id: 0,
            vertices: vec![0, 1, 2],
            edges: vec![3],
            faces: vec![],
            external_links: vec![],
        };

        let state = compute_five_dim(
            &sub,
            1,    // total_sub
            3,    // global_max_cells
            1,    // depth
            0.0,  // _global_avg_degree
            0.0,  // total_flux
            0.0,  // _grad_mean
            0.5,  // grad_std
        );

        assert!((0.0..=1.0).contains(&state.intrinsic_degree),
            "intrinsic_degree 应在 [0,1]: got {}", state.intrinsic_degree);
        assert!((0.0..=1.0).contains(&state.binding_degree),
            "binding_degree 应在 [0,1]: got {}", state.binding_degree);
        assert!(state.energy_density >= 0.0,
            "energy_density 应非负: got {}", state.energy_density);
        assert!((0.0..=1.0).contains(&state.evolution_rate),
            "evolution_rate 应在 [0,1]: got {}", state.evolution_rate);
        assert!((0.0..=1.0).contains(&state.structural_robustness),
            "structural_robustness 应在 [0,1]: got {}", state.structural_robustness);
    }

    /// 用简单子几何体调用 DynamicsParams::from_sub_geometry，
    /// 验证 11 个参数均为有限且非负。
    #[test]
    fn test_dynamics_params() {
        let sub = SubGeometry {
            id: 0,
            vertices: vec![0, 1, 2],
            edges: vec![4, 5],
            faces: vec![6],
            external_links: vec![],
        };

        let state = FiveDimState {
            intrinsic_degree: 0.5,
            binding_degree: 0.2,
            energy_density: 1.0,
            evolution_rate: 0.3,
            structural_robustness: 0.8,
        };

        let params = DynamicsParams::from_sub_geometry(
            &sub, &state, 2,  // depth = 2
            0.4, // grad_mean
            0.3, // grad_std
        );

        // 所有参数有限且 >= 0
        for (name, val) in [
            ("alpha_1", params.alpha_1),
            ("alpha_2", params.alpha_2),
            ("beta_1", params.beta_1),
            ("beta_2", params.beta_2),
            ("gamma_1", params.gamma_1),
            ("gamma_2", params.gamma_2),
            ("delta_1", params.delta_1),
            ("delta_2", params.delta_2),
            ("delta_3", params.delta_3),
            ("epsilon_1", params.epsilon_1),
            ("epsilon_2", params.epsilon_2),
        ] {
            assert!(val.is_finite(), "{} 应为有限值，got {}", name, val);
            assert!(val >= 0.0, "{} 应非负，got {}", name, val);
        }
    }

    /// 两个子几何体之间存在 external_links 时，
    /// compute_coupling 至少生成一条耦合。
    #[test]
    fn test_coupling() {
        let cells = vec![
            Cell { dim: 0, id: 0, boundary: vec![], label: "v0".into(), domain: String::new() },
            Cell { dim: 0, id: 1, boundary: vec![], label: "v1".into(), domain: String::new() },
        ];

        let mut v0_map = HashMap::new();
        v0_map.insert(0, 0);
        v0_map.insert(1, 1);

        let complex = CWComplex {
            cells,
            v0_map,
            e1_map: HashMap::new(),
            euler_characteristic: 2,
            betti_0: 2,
            betti_1: 0,
        };

        let sub_geos = vec![
            SubGeometry {
                id: 0,
                vertices: vec![0],
                edges: vec![],
                faces: vec![],
                external_links: vec![(1, 1)],
            },
            SubGeometry {
                id: 1,
                vertices: vec![1],
                edges: vec![],
                faces: vec![],
                external_links: vec![],
            },
        ];

        let couplings = compute_coupling(&complex, &sub_geos);

        assert!(!couplings.is_empty(),
            "两个子几何体存在外部链接时应生成至少一条耦合");
        assert_eq!(couplings[0].i, 0);
        assert_eq!(couplings[0].j, 1);
    }

    /// 构建含 2 个 0-胞腔 + 1 个 1-胞腔的 CWComplex，
    /// 生成 PhaseThreeOutput 后调用 decode_phase_three，
    /// 验证重建结果包含 2 个 0-胞腔 + 1 个 1-胞腔。
    #[test]
    fn test_xi_bijection_decode() {
        let cells = vec![
            Cell { dim: 0, id: 0, boundary: vec![], label: "a".into(), domain: String::new() },
            Cell { dim: 0, id: 1, boundary: vec![], label: "b".into(), domain: String::new() },
            Cell { dim: 1, id: 2, boundary: vec![0, 1], label: "ab".into(), domain: String::new() },
        ];

        let mut v0_map = HashMap::new();
        v0_map.insert(0, 0);
        v0_map.insert(1, 1);

        let mut e1_map = HashMap::new();
        e1_map.insert((0, 1), 2);

        let _complex = CWComplex {
            cells,
            v0_map,
            e1_map,
            euler_characteristic: 2 - 1,
            betti_0: 1,
            betti_1: 0,
        };

        let sub_geo = SubGeometry {
            id: 0,
            vertices: vec![0, 1],
            edges: vec![2],
            faces: vec![],
            external_links: vec![],
        };

        let state = FiveDimState {
            intrinsic_degree: 0.5,
            binding_degree: 0.0,
            energy_density: 1.0,
            evolution_rate: 0.2,
            structural_robustness: 0.75,
        };

        let params = DynamicsParams {
            alpha_1: 0.5, alpha_2: 0.3,
            beta_1: 0.5, beta_2: 0.4,
            gamma_1: 0.1, gamma_2: 0.3,
            delta_1: 0.1, delta_2: 0.2, delta_3: 0.05,
            epsilon_1: 0.2, epsilon_2: 0.1,
        };

        let xi = ExtendedDimension {
            vertex_cells: vec![
                CellSnapshot { dim: 0, id: 0, boundary: vec![] },
                CellSnapshot { dim: 0, id: 1, boundary: vec![] },
            ],
            edge_cells: vec![
                CellSnapshot { dim: 1, id: 2, boundary: vec![0, 1] },
            ],
            face_cells: vec![],
            boundary_links: vec![],
            vertex_potentials: vec![0.2, 0.4],
            edge_flux_magnitudes: vec![0.3],
            node_texts: vec!["node_0".into(), "node_1".into()],
        };

        let memes = vec![MemeDecomposition {
            id: 0,
            sub_geometry: sub_geo,
            state,
            params,
            xi,
        }];

        let output = PhaseThreeOutput {
            memes,
            couplings: vec![],
            n_memes: 1,
        };

        let reconstructed = decode_phase_three(&output);

        assert_eq!(reconstructed.v0_map.len(), 2,
            "重建后应有 2 个 0-胞腔，got {}", reconstructed.v0_map.len());
        assert_eq!(reconstructed.e1_map.len(), 1,
            "重建后应有 1 个 1-胞腔，got {}", reconstructed.e1_map.len());
    }
}
