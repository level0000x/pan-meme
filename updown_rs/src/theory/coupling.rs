//! 耦合矩阵 C — 泛模因理论 §3.4.4, §6.4.4
//!
//! 核心函数：
//! - compute_coupling: 计算两个模因间的耦合强度
//! - build_coupling_matrix: 构建完整耦合矩阵 C

use crate::theory::five_dim::FiveDimState;

/// 计算两个模因间的耦合强度 — 论文 §3.4.4
///
/// C_{ij} = 基于共享顶点数、总顶点数、状态相似度的耦合度量
pub fn compute_coupling(
    state_i: &FiveDimState,
    state_j: &FiveDimState,
    shared: usize,
    total_i: usize,
    total_j: usize,
) -> f64 {
    if total_i == 0 || total_j == 0 {
        return 0.0;
    }

    // Jaccard 相似度
    let union = total_i + total_j - shared;
    let jaccard = if union > 0 {
        shared as f64 / union as f64
    } else {
        0.0
    };

    // 状态相似度: 1 - 归一化欧氏距离
    let state_diff = ((state_i.intrinsic_degree - state_j.intrinsic_degree).powi(2)
        + (state_i.binding_degree - state_j.binding_degree).powi(2)
        + (state_i.energy_density - state_j.energy_density).powi(2)
        + (state_i.evolution_rate - state_j.evolution_rate).powi(2)
        + (state_i.structural_robustness - state_j.structural_robustness).powi(2))
    .sqrt();

    let state_similarity = 1.0 / (1.0 + state_diff);

    // 耦合强度 = Jaccard 相似度 × 状态相似度
    let coupling = jaccard * state_similarity;
    if coupling.is_nan() || coupling.is_infinite() {
        0.0
    } else {
        coupling
    }
}

/// 构建耦合矩阵 C — 论文 §6.4.4
///
/// 输入: 模因状态列表 + 每个模因的顶点列表
/// 输出: n × n 耦合矩阵 C
pub fn build_coupling_matrix(
    memes: &[FiveDimState],
    meme_vertices: &[Vec<usize>],
) -> Vec<Vec<f64>> {
    let n = memes.len();
    let mut c = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let shared = meme_vertices[i]
                .iter()
                .filter(|v| meme_vertices[j].contains(v))
                .count();
            let coupling = compute_coupling(
                &memes[i],
                &memes[j],
                shared,
                meme_vertices[i].len(),
                meme_vertices[j].len(),
            );
            c[i][j] = coupling;
            c[j][i] = coupling;
        }
    }

    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_shared() {
        let a = FiveDimState::zero();
        let b = FiveDimState::zero();
        let c = compute_coupling(&a, &b, 0, 10, 10);
        assert!((c).abs() < 1e-10);
    }

    #[test]
    fn test_perfect_shared() {
        let a = FiveDimState::new(0.5, 0.5, 1.0, 0.3, 0.8);
        let b = FiveDimState::new(0.5, 0.5, 1.0, 0.3, 0.8);
        let c = compute_coupling(&a, &b, 10, 10, 10);
        assert!(c > 0.0);
        assert!(c <= 1.0);
    }

    #[test]
    fn test_coupling_matrix() {
        let memes = vec![
            FiveDimState::new(0.5, 0.5, 1.0, 0.3, 0.8),
            FiveDimState::new(0.5, 0.5, 1.0, 0.3, 0.8),
        ];
        let vertices = vec![vec![0, 1, 2], vec![2, 3, 4]];
        let c = build_coupling_matrix(&memes, &vertices);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].len(), 2);
        assert_eq!(c[0][1], c[1][0]);
    }
}
