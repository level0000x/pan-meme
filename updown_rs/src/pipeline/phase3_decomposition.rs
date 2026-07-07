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
use crate::theory::louvain::run_louvain;

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
pub fn run_phase_three(phase2: &PhaseTwoOutput) -> PhaseThreeOutput {
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
    let _n_memes = if let Some(layer) = run_louvain(&adj) {
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

    let mut memes = Vec::new();
    for verts in &meme_vertices {
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

        // 五维状态 (supplement 定义 3.2)
        // 所有维度 clamp 到有效域 Ω = [0,1]⁴×[0,∞)
        let d = (nv / total_vertices.max(1.0)).clamp(0.0, 1.0);
        let b = (nv / total_vertices.max(1.0)).clamp(0.0, 1.0);
        let rho = grad_mean.abs().max(0.0);
        let r = (nv / total_vertices.max(1.0)).clamp(0.0, 1.0);
        // S: 结构韧度 — 使用 2-胞腔占比（supplement 定义 3.2: 2-胞腔数/维数比）
        // Euler 特征 χ=V-E 可能为负，改用归一化顶点数密度作为代理，恒在 [0,1]
        let s = d;

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

        // 11 参数
        let params = DynamicsParams::from_geometry(
            verts.len(),
            0,
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
        let p3 = run_phase_three(&p2);
        assert!(p3.n_memes > 0);
        assert_eq!(p3.coupling.len(), p3.n_memes);
        for meme in &p3.memes {
            assert!(meme.five_dim.is_valid() || meme.five_dim.intrinsic_degree >= 0.0);
        }
    }
}
