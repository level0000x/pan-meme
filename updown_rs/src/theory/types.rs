//! 语义新类型 — 防止单位混淆，利用类型系统
//!
//! 裸 `f64` 可以表示 Jaccard 阈值、ODE 步长、Shannon 熵等完全不同语义的值。
//! 编译器不会阻止你把熵值当成步长用。newtype 提供零成本编译期安全。

use std::fmt;

/// Jaccard 相似度阈值 T ∈ [0, 1]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct JaccardThreshold(pub f64);

impl JaccardThreshold {
    pub fn new(v: f64) -> Self {
        JaccardThreshold(v.clamp(0.0, 1.0))
    }
}

impl fmt::Display for JaccardThreshold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

/// Shannon 熵值（非负）
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ShannonEntropy(pub f64);

impl ShannonEntropy {
    pub fn new(v: f64) -> Self {
        ShannonEntropy(v.max(0.0))
    }
}

impl fmt::Display for ShannonEntropy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4} bits", self.0)
    }
}

/// 模块度 Q ∈ [-1, 1]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Modularity(pub f64);

impl Modularity {
    pub fn new(v: f64) -> Self {
        Modularity(v.clamp(-1.0, 1.0))
    }
}

impl fmt::Display for Modularity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Q={:.4}", self.0)
    }
}

/// 信息深度 d ≥ 0（概念层级深度）
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct InfoDepth(pub f64);

impl InfoDepth {
    pub fn new(v: f64) -> Self {
        InfoDepth(v.max(0.0))
    }
}

impl fmt::Display for InfoDepth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "d={:.1}", self.0)
    }
}

/// 恢复率 r ∈ [0, 1]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct RecoveryRate(pub f64);

impl RecoveryRate {
    pub fn new(v: f64) -> Self {
        RecoveryRate(v.clamp(0.0, 1.0))
    }
}

impl fmt::Display for RecoveryRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1}%", self.0 * 100.0)
    }
}
