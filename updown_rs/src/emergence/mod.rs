//! Phase 0-1: 浮现 — 从原始输入到结构化模型
//!
//! extractor: 词列表 → 关系网络 Ψ = A(I)
//! cycle:     ↑↓ 伽罗瓦循环、涌现
//! relations: 关系自组织 → 五类推理 → 概念层级 → M = (S, F, C)
//! louvain:   Louvain 多层社区检测 → 多层级结构域 S

pub mod extractor;
pub mod cycle;
pub mod relations;
pub mod louvain;
