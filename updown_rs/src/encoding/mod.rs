//! Phase 2-3: 编码 — 模型到几何表示
//!
//! geometry:       CW 复形 → 离散梯度 → 向量场 → 不变量 → G = (K, g, Γ, R)
//! decomposition:  Betti 分解 → 五维映射 → 11 参数闭式解 → Q = ({Xᵢ}, Θ, C)

pub mod geometry;
pub mod decomposition;
