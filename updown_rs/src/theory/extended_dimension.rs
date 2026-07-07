//! 扩展维度 ξ — 泛模因理论 §6.4.3, 前提 5
//!
//! 核心构造：ExtendedDimension = (V_i, E_i, F_i, B_i)
//!
//! 对应 proof-supplement-complete.md 定义 3.3
//! "核心五维驱动动力学，扩展维度编码微观涨落保证双射性"

/// 胞腔快照 — 用于 ξ 中记录胞腔的完整信息
#[derive(Debug, Clone, PartialEq)]
pub struct CellSnapshot {
    /// 维数: 0=顶点, 1=边, 2=面
    pub dim: u8,
    /// 胞腔 id
    pub id: usize,
    /// 边界
    pub boundary: Vec<usize>,
}

/// 扩展维度 ξ — 论文 §6.4.3, 前提 5 + supplement 定义 3.3
///
/// ξ = (V_i, E_i, F_i, B_i) 四元组，编码微观涨落保证双射性
/// 当 n 个模因合并/分裂时（定理 6 跳变），ξ 承载子几何体的完整信息
#[derive(Debug, Clone)]
pub struct ExtendedDimension {
    /// 顶点快照 V_i — supplement 定义 3.3
    pub cell_snapshots_v: Vec<CellSnapshot>,
    /// 边快照 E_i — supplement 定义 3.3
    pub cell_snapshots_e: Vec<CellSnapshot>,
    /// 面快照 F_i — supplement 定义 3.3
    pub cell_snapshots_f: Vec<CellSnapshot>,
    /// 跨子几何体边界链接 B_i — supplement 定义 3.3
    /// (source_vertex_id, target_vertex_id, link_strength)
    pub boundary_links: Vec<(usize, usize, f64)>,
    /// 分裂来源（新模因从父模因的 ξ 解码初始化）— 定理 6
    pub parent_meme_id: Option<usize>,
    /// 微观涨落模式（5D 残差向量）— 前提 5
    pub micro_fluctuation: Vec<f64>,
}

impl ExtendedDimension {
    /// 创建空的扩展维度
    pub fn new() -> Self {
        ExtendedDimension {
            cell_snapshots_v: Vec::new(),
            cell_snapshots_e: Vec::new(),
            cell_snapshots_f: Vec::new(),
            boundary_links: Vec::new(),
            parent_meme_id: None,
            micro_fluctuation: Vec::new(),
        }
    }

    /// 带父模因信息的扩展维度（用于跳变后的新模因初始化）
    pub fn with_parent(parent_id: usize) -> Self {
        ExtendedDimension {
            cell_snapshots_v: Vec::new(),
            cell_snapshots_e: Vec::new(),
            cell_snapshots_f: Vec::new(),
            boundary_links: Vec::new(),
            parent_meme_id: Some(parent_id),
            micro_fluctuation: Vec::new(),
        }
    }

    /// 记录消失模因的状态到边界链接（用于模因吸收）
    pub fn absorb(&mut self, vanished_state: &[f64; 5]) {
        self.micro_fluctuation = vanished_state.to_vec();
    }
}

impl Default for ExtendedDimension {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_extended_dimension() {
        let xi = ExtendedDimension::new();
        assert!(xi.cell_snapshots_v.is_empty());
        assert!(xi.parent_meme_id.is_none());
    }

    #[test]
    fn test_with_parent() {
        let xi = ExtendedDimension::with_parent(42);
        assert_eq!(xi.parent_meme_id, Some(42));
    }

    #[test]
    fn test_absorb() {
        let mut xi = ExtendedDimension::new();
        let state = [0.5, 0.3, 1.0, 0.2, 0.8];
        xi.absorb(&state);
        assert_eq!(xi.micro_fluctuation, state.to_vec());
    }
}
