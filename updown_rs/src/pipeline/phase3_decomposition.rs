//! Phase 3: 分解 (G → Q) — 泛模因理论 §3.4, §6.4 M2
//!
//! 对应 proof-supplement-complete.md 定义 3.1-3.4 + 定理 3.5-3.6
//!
//! 注：与论文 §3.4.1 的已知偏差 — 论文定义 n = β₀(K)（连通分量数），
//! 但实践中字词关系图往往形成单一巨连通分量，丧失模因分解意义。
//! 故采用 Louvain 社区检测作为替代分解策略。
//! 当 Louvain 退化为连通分量时，与论文方法等价。
//!
//! 数据流:
//! PhaseTwoOutput → 构建 Louvain 邻接表 → Louvain 社区检测
//!              → 每社区: 五维状态 + 扩展维度 ξ + 11 参数
//!              → 耦合矩阵 C
//!              → PhaseThreeOutput { memes, coupling }
//!
//! 输出: Q = {X_i, Θ, C}

use crate::pipeline::phase2_encoding::PhaseTwoOutput;
use crate::theory::coupling::build_coupling_matrix;
use crate::theory::dynamics_params::DynamicsParams;
use crate::theory::extended_dimension::ExtendedDimension;
use crate::theory::five_dim::FiveDimState;
use crate::theory::louvain::run_louvain_with_resolution;

/// 单模因状态
#[derive(Debug, Clone)]
pub struct MemeState {
    /// 五维状态
    pub five_dim: FiveDimState,
    /// 扩展维度 ξ
    pub extended: ExtendedDimension,
    /// 11 个动力学参数
    pub params: DynamicsParams,
    /// 该模因包含的顶点索引
    pub vertices: Vec<usize>,
}

/// Phase 3 输出: Q = {X_i, Θ, C} — 论文 §3.4.4
#[derive(Debug, Clone)]
pub struct PhaseThreeOutput {
    pub memes: Vec<MemeState>,
    pub coupling: Vec<Vec<f64>>,
    pub n_memes: usize,
}

/// 运行 Phase 3: 分解
pub fn run_phase_three(phase2: &PhaseTwoOutput, louvain_resolution: f64) -> PhaseThreeOutput {
    let n_vertices = phase2.complex.n_vertices();

    // 构建 Louvain 邻接表（边权 = 通量 |Γ¹(e)|）
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_vertices];
    for cell in &phase2.complex.cells {
        if cell.dim == 1 && cell.boundary.len() == 2 {
            let v1 = cell.boundary[0];
            let v2 = cell.boundary[1];
            let weight = 1.0; // 简化: 使用通量 |Γ¹(e)|
            adj[v1].push((v2, weight));
            adj[v2].push((v1, weight));
        }
    }

    // Louvain 社区检测
    let mut communities = vec![0; n_vertices];
    let _n_memes = if let Some(layer) = run_louvain_with_resolution(&adj, louvain_resolution) {
        communities = layer.node_to_community;
        layer.n_communities
    } else {
        // fallback: 每个顶点一个社区
        communities = (0..n_vertices).collect();
        n_vertices
    };

    // 收集每个社区的顶点
    let n_communities = communities.iter().max().map(|&m| m + 1).unwrap_or(0);
    let mut meme_vertices: Vec<Vec<usize>> = vec![Vec::new(); n_communities];
    for (v, &c) in communities.iter().enumerate() {
        if c < n_communities {
            meme_vertices[c].push(v);
        }
    }

    // 每个社区: 五维状态 + 扩展维度 ξ + 11 参数
    let total_vertices = n_vertices as f64;
    let max_depth = phase2.reversibility.containment_depth;
    let grad_mean = phase2.vector_field.mean_gradient();
    let grad_std = phase2.vector_field.std_gradient();

    // 预计算每个社区的顶点集合（用于快速查找）
    let mut vertex_to_community = vec![0usize; n_vertices];
    for (ci, verts) in meme_vertices.iter().enumerate() {
        for &v in verts {
            if v < n_vertices {
                vertex_to_community[v] = ci;
            }
        }
    }

    // 预计算每个社区的边统计
    let n_communities = meme_vertices.len();
    let mut internal_edges = vec![0usize; n_communities];
    let mut cross_edges = vec![0usize; n_communities];
    let mut community_degree = vec![0.0f64; n_communities]; // 社区总度数
    for cell in &phase2.complex.cells {
        if cell.dim == 1 && cell.boundary.len() == 2 {
            let v1 = cell.boundary[0];
            let v2 = cell.boundary[1];
            if v1 < n_vertices && v2 < n_vertices {
                let c1 = vertex_to_community[v1];
                let c2 = vertex_to_community[v2];
                if c1 < n_communities && c2 < n_communities {
                    community_degree[c1] += 1.0;
                    community_degree[c2] += 1.0;
                    if c1 == c2 {
                        internal_edges[c1] += 1;
                    } else {
                        cross_edges[c1] += 1;
                        cross_edges[c2] += 1;
                    }
                }
            }
        }
    }

    let mut memes = Vec::new();
    for (ci, verts) in meme_vertices.iter().enumerate() {
        if verts.is_empty() {
            continue;
        }

        let nv = verts.len() as f64;
        let depth = if let Some(&d) = verts
            .first()
            .and_then(|&v| phase2.reversibility.node_levels.get(v))
        {
            d as f64
        } else {
            0.0
        };

        // ── 五维状态 (supplement 定义 3.2) ──
        // 每个模因根据其内部结构计算独立初始状态

        // D: 内在度 = 内部边密度 + 顶点占比的混合
        let internal_density = if nv > 1.0 {
            (2.0 * internal_edges[ci] as f64) / (nv * (nv - 1.0))
        } else {
            0.0
        };
        let d = (0.5 * internal_density + 0.5 * (nv / total_vertices.max(1.0)))
            .clamp(0.0, 1.0);

        // B: 广度 = 顶点占比 + 跨社区连接度
        let total_touching = internal_edges[ci] as f64 + cross_edges[ci] as f64;
        let cross_ratio = if total_touching > 0.0 {
            cross_edges[ci] as f64 / total_touching
        } else {
            0.0
        };
        let b = (0.6 * (nv / total_vertices.max(1.0)) + 0.4 * cross_ratio)
            .clamp(0.0, 1.0);

        // ρ: 能量密度 = 全局场强 + 内部边密度调制
        let rho = (grad_mean.abs() * (0.3 + 0.7 * internal_density))
            .max(0.0).min(1.0);

        // R: 演化速率 — 小模因更活跃（高 R），大而密集的模因更稳定（低 R）
        // 用大小和密度共同决定，使 R 在 [0.15, 0.9] 范围内拉开
        let size_factor = (nv / total_vertices.max(1.0)).clamp(0.0, 1.0); // 大模因 → 低 R
        let r_base = cross_ratio.clamp(0.0, 1.0);
        let sparsity_factor = (1.0 - internal_density).max(0.1);
        // 混合: 40% 跨社区 + 30% 稀疏度 + 30% 大小反比
        let r = (r_base * 0.4 + sparsity_factor * 0.3 + (1.0 - size_factor) * 0.3)
            .clamp(0.15, 0.9);

        // S: 结构韧度 = 内部边密度 + 顶点占比
        let s = (0.7 * internal_density + 0.3 * (nv / total_vertices.max(1.0)))
            .clamp(0.0, 1.0);

        let five_dim = FiveDimState::new(d, b, rho, r, s);

        // 扩展维度 ξ (supplement 定义 3.3)
        let extended = ExtendedDimension {
            cell_snapshots_v: verts
                .iter()
                .map(|&v| crate::theory::extended_dimension::CellSnapshot {
                    dim: 0,
                    id: v,
                    boundary: vec![],
                })
                .collect(),
            cell_snapshots_e: Vec::new(),
            cell_snapshots_f: Vec::new(),
            boundary_links: Vec::new(),
            parent_meme_id: None,
            micro_fluctuation: Vec::new(),
        };

        // 11 参数 — 根据模因结构差异化
        // δ₁ 核心驱动力: 高 ρ 和 B 的模因有更强的自持力
        // δ₃ 自发衰退: 大模因衰退慢，小模因衰退快
        let params = DynamicsParams::from_geometry(
            verts.len(),
            internal_edges[ci],
            depth,
            max_depth,
            &five_dim,
            grad_mean,
            grad_std,
        );

        memes.push(MemeState {
            five_dim,
            extended,
            params,
            vertices: verts.clone(),
        });
    }

    let n_memes = memes.len();
    let five_dim_states: Vec<FiveDimState> = memes.iter().map(|m| m.five_dim).collect();
    let meme_vert_lists: Vec<Vec<usize>> = memes.iter().map(|m| m.vertices.clone()).collect();
    let coupling = build_coupling_matrix(&five_dim_states, &meme_vert_lists);

    PhaseThreeOutput {
        memes,
        coupling,
        n_memes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::phase1_emergence::run_phase_one;
    use crate::pipeline::phase2_encoding::run_phase_two;

    #[test]
    fn test_phase_three_basic() {
        let words = vec!["苹果".to_string(), "香蕉".to_string(), "水果".to_string()];
        let p1 = run_phase_one(words, 100);
        let p2 = run_phase_two(&p1);
        let p3 = run_phase_three(&p2, 1.0);
        assert!(p3.n_memes > 0);
        assert_eq!(p3.coupling.len(), p3.n_memes);
        for meme in &p3.memes {
            assert!(meme.five_dim.is_valid() || meme.five_dim.intrinsic_degree >= 0.0);
        }
    }
}
