//! 跳变点处理 + Π 投影映射 — 泛模因理论 定理 6
//!
//! 核心构造：
//! - JumpHandler: 跳变处理器
//! - JumpType: 跳变类型（合并/分裂/消失）
//!
//! 定理 6 规则:
//! 1. 模因追踪: 新几何中存在与旧几何结构连续的子复形 → 状态承接
//! 2. 新模因初始化: n_i > n_{i-1} → 从父模因 ξ 解码初始化
//! 3. 模因消失: n_i < n_{i-1} → 消失模因状态被吸收进存续模因的 ξ

use crate::theory::extended_dimension::ExtendedDimension;
use crate::theory::five_dim::FiveDimState;

/// 跳变类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JumpType {
    /// 模因数量不变 — 无跳变
    NoJump,
    /// 模因数量增加 — 分裂
    Split,
    /// 模因数量减少 — 合并/消失
    Merge,
}

/// 跳变处理器 — 定理 6
#[derive(Debug, Clone)]
pub struct JumpHandler;

impl JumpHandler {
    /// 检测跳变类型
    pub fn detect_jump(betti_0_old: usize, betti_0_new: usize) -> JumpType {
        if betti_0_new > betti_0_old {
            JumpType::Split
        } else if betti_0_new < betti_0_old {
            JumpType::Merge
        } else {
            JumpType::NoJump
        }
    }

    /// 投影映射 Π: 从旧几何到新几何的可逆投影 — 定理 6
    ///
    /// 对应论文定理 6 的模因追踪规则:
    /// M_j(t_i^+) = Π(M_j(t_i^-), K_j(t_i^+))
    pub fn project_state(
        old_state: &FiveDimState,
        old_vertices: &[usize],
        new_vertices: &[usize],
    ) -> FiveDimState {
        let overlap = old_vertices
            .iter()
            .filter(|v| new_vertices.contains(v))
            .count();
        let retention = if old_vertices.is_empty() {
            0.0
        } else {
            overlap as f64 / old_vertices.len() as f64
        };

        FiveDimState {
            intrinsic_degree: old_state.intrinsic_degree * retention,
            binding_degree: old_state.binding_degree * retention,
            energy_density: old_state.energy_density * retention,
            evolution_rate: old_state.evolution_rate * retention,
            structural_robustness: old_state.structural_robustness * retention,
        }
    }

    /// 从父模因的扩展维度 ξ 解码初始化新模因 — 定理 6 规则 2
    ///
    /// 对应论文: n_i > n_{i-1} → 新增模因从父模因的 ξ 解码初始化
    pub fn init_new_meme(parent_xi: &ExtendedDimension) -> FiveDimState {
        if parent_xi.micro_fluctuation.len() == 5 {
            FiveDimState::new(
                parent_xi.micro_fluctuation[0],
                parent_xi.micro_fluctuation[1],
                parent_xi.micro_fluctuation[2],
                parent_xi.micro_fluctuation[3],
                parent_xi.micro_fluctuation[4],
            )
        } else {
            FiveDimState::zero()
        }
    }

    /// 吸收消失模因的状态到存续模因的 ξ — 定理 6 规则 3
    ///
    /// 对应论文: n_i < n_{i-1} → 消失模因状态被吸收进存续模因的 ξ 中
    pub fn absorb_meme(target_xi: &mut ExtendedDimension, vanished_state: &FiveDimState) {
        target_xi.absorb(&vanished_state.to_array());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_no_jump() {
        assert_eq!(JumpHandler::detect_jump(3, 3), JumpType::NoJump);
    }

    #[test]
    fn test_detect_split() {
        assert_eq!(JumpHandler::detect_jump(1, 3), JumpType::Split);
    }

    #[test]
    fn test_detect_merge() {
        assert_eq!(JumpHandler::detect_jump(3, 1), JumpType::Merge);
    }

    #[test]
    fn test_project_state_full_overlap() {
        let old = FiveDimState::new(0.5, 0.5, 1.0, 0.3, 0.8);
        let old_verts = vec![0, 1, 2];
        let new_verts = vec![0, 1, 2, 3];
        let projected = JumpHandler::project_state(&old, &old_verts, &new_verts);
        assert!((projected.intrinsic_degree - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_project_state_partial_overlap() {
        let old = FiveDimState::new(0.5, 0.5, 1.0, 0.3, 0.8);
        let old_verts = vec![0, 1, 2, 3];
        let new_verts = vec![0, 1];
        let projected = JumpHandler::project_state(&old, &old_verts, &new_verts);
        assert!((projected.intrinsic_degree - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_init_new_meme() {
        let mut xi = ExtendedDimension::with_parent(0);
        let state = [0.5, 0.3, 1.0, 0.2, 0.8];
        xi.absorb(&state);
        let new_meme = JumpHandler::init_new_meme(&xi);
        assert!((new_meme.intrinsic_degree - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_absorb_meme() {
        let mut xi = ExtendedDimension::new();
        let vanished = FiveDimState::new(0.1, 0.1, 0.5, 0.05, 0.2);
        JumpHandler::absorb_meme(&mut xi, &vanished);
        assert_eq!(xi.micro_fluctuation.len(), 5);
    }
}
