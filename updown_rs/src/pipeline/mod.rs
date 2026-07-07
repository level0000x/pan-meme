//! 五阶段管线层 — 泛模因理论 §6.2-§6.4
//!
//! 管线分为五个阶段，每阶段调用 theory 层的数学构造：
//! - Phase 1: 浮现 (I → M = (S, F, C))
//! - Phase 2: 编码 (M → G = (K, g, ω, Γ, R))
//! - Phase 3: 分解 (G → Q = {X_i, Θ, C})
//! - Phase 4: 固化 (Q → Credential)
//! - Phase 5: 演化 (Q → ODE 时间序列 → 原型分类)

pub mod phase1_emergence;
pub mod phase2_encoding;
pub mod phase3_decomposition;
pub mod phase4_binding;
pub mod phase5_evolution;
