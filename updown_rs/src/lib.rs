//! ↑↓ 泛模因建模引擎 — 库入口
//!
//! 三层架构: theory/ (核心数学构造) → pipeline/ (五阶段管线) → io/ (输入输出)
//!
//! 数学对应: 泛模因理论 §3-§4, §6.2-§6.4

pub mod io;
pub mod pipeline;
pub mod theory;
