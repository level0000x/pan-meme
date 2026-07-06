//! Phase 4-5: 固化与演化
//!
//! binding:    SHA-256 哈希绑定 → 凭证
//! merkle:     Merkle 树 / 证明 / 层验证 / 篡改定位
//! ode:        RKF45 求解器 → 5D ODE → 收敛分类 → 九类原型
//! optimizer:  H = T × F × Θ × N 全局优化

pub mod binding;
pub mod merkle;
pub mod ode;
pub mod optimizer;
