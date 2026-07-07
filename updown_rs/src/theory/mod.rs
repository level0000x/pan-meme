//! 核心数学构造层 — 泛模因理论 §3-§4, 附录 A-E
//!
//! 本模块封装了论文中所有核心数学构造，包括：
//! - 形式概念分析 (FCA) ↑↓ 伽罗瓦连接
//! - CW 胞腔复形 + 离散梯度场
//! - Louvain 社区检测
//! - 五维状态 + 扩展维度 ξ
//! - 五族函数 (Power/Exponential/Sigmoid/Logarithm/PiecewiseLinear)
//! - 11 参数动力学 + 耦合矩阵
//! - 5D ODE + RKF45 + 原型分类
//! - 跳变点处理 + Π 投影映射
//! - 假设 0 全局优化

pub mod coupling;
pub mod cw_complex;
pub mod dynamics_params;
pub mod extended_dimension;
pub mod fca;
pub mod five_dim;
pub mod function_families;
pub mod jump_handler;
pub mod louvain;
pub mod ode;
pub mod optimizer;
pub mod types;
pub mod vector_field;
