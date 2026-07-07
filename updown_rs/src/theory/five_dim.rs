//! 五维状态 — 泛模因理论 §3.4.2, §4.1
//!
//! 核心构造：FiveDimState = (D, B, ρ, R, S)
//!
//! 对应 proof-supplement-complete.md 定义 3.2

/// 五维状态向量 M = (D, B, ρ, R, S) — 论文 §3.4.2
///
/// 每个维度对应论文附录 A.2 的定义：
/// - D: 内禀度 (Intrinsic Degree) — 结构自身的复杂性与自洽性程度
/// - B: 关联度 (Binding Degree) — 结构与其他系统或要素的连接广度
/// - ρ: 能流密度 (Energy Density) — 单位时间/空间内承载或转换的能量/信息流强度
/// - R: 演化速率 (Evolution Rate) — 结构状态变化或扩散的瞬时速度
/// - S: 结构韧度 (Structural Robustness) — 结构抵抗扰动、维持核心特征的能力
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiveDimState {
    /// 内禀度 D ∈ [0, 1]
    pub intrinsic_degree: f64,
    /// 关联度 B ∈ [0, 1]
    pub binding_degree: f64,
    /// 能流密度 ρ ∈ [0, ∞)
    pub energy_density: f64,
    /// 演化速率 R ∈ [0, 1]
    pub evolution_rate: f64,
    /// 结构韧度 S ∈ [0, 1]
    pub structural_robustness: f64,
}

impl FiveDimState {
    /// 创建五维状态，自动 NaN 防护
    pub fn new(d: f64, b: f64, rho: f64, r: f64, s: f64) -> Self {
        FiveDimState {
            intrinsic_degree: if d.is_nan() || d.is_infinite() {
                0.0
            } else {
                d
            },
            binding_degree: if b.is_nan() || b.is_infinite() {
                0.0
            } else {
                b
            },
            energy_density: if rho.is_nan() || rho.is_infinite() {
                0.0
            } else {
                rho
            },
            evolution_rate: if r.is_nan() || r.is_infinite() {
                0.0
            } else {
                r
            },
            structural_robustness: if s.is_nan() || s.is_infinite() {
                0.0
            } else {
                s
            },
        }
    }

    /// 零状态
    pub fn zero() -> Self {
        FiveDimState {
            intrinsic_degree: 0.0,
            binding_degree: 0.0,
            energy_density: 0.0,
            evolution_rate: 0.0,
            structural_robustness: 0.0,
        }
    }

    /// 验证状态在有效域 Ω = [0,1]⁴ × [0,∞) 内 — 定理 7
    pub fn is_valid(&self) -> bool {
        self.intrinsic_degree >= 0.0
            && self.intrinsic_degree <= 1.0
            && self.binding_degree >= 0.0
            && self.binding_degree <= 1.0
            && self.energy_density >= 0.0
            && self.evolution_rate >= 0.0
            && self.evolution_rate <= 1.0
            && self.structural_robustness >= 0.0
            && self.structural_robustness <= 1.0
    }

    /// 裁剪到 Ω = [0,1]⁴ × [0,∞) — 定理 7 不变性
    pub fn clamp_to_omega(&self) -> Self {
        FiveDimState {
            intrinsic_degree: self.intrinsic_degree.clamp(0.0, 1.0),
            binding_degree: self.binding_degree.clamp(0.0, 1.0),
            energy_density: self.energy_density.max(0.0),
            evolution_rate: self.evolution_rate.clamp(0.0, 1.0),
            structural_robustness: self.structural_robustness.clamp(0.0, 1.0),
        }
    }

    /// 状态间总变化量
    pub fn total_change(&self, other: &FiveDimState) -> f64 {
        (self.intrinsic_degree - other.intrinsic_degree).abs()
            + (self.binding_degree - other.binding_degree).abs()
            + (self.energy_density - other.energy_density).abs()
            + (self.evolution_rate - other.evolution_rate).abs()
            + (self.structural_robustness - other.structural_robustness).abs()
    }

    /// 转换为数组 [D, B, ρ, R, S]
    pub fn to_array(&self) -> [f64; 5] {
        [
            self.intrinsic_degree,
            self.binding_degree,
            self.energy_density,
            self.evolution_rate,
            self.structural_robustness,
        ]
    }

    /// 从数组创建
    pub fn from_array(arr: &[f64; 5]) -> Self {
        FiveDimState::new(arr[0], arr[1], arr[2], arr[3], arr[4])
    }
}

impl Default for FiveDimState {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_state() {
        let s = FiveDimState::new(0.5, 0.5, 1.0, 0.3, 0.8);
        assert!(s.is_valid());
    }

    #[test]
    fn test_invalid_state() {
        let s = FiveDimState::new(1.5, 0.5, 1.0, 0.3, 0.8);
        assert!(!s.is_valid());
    }

    #[test]
    fn test_nan_guard() {
        let s = FiveDimState::new(f64::NAN, 0.5, 1.0, 0.3, 0.8);
        assert_eq!(s.intrinsic_degree, 0.0);
    }

    #[test]
    fn test_clamp_to_omega() {
        let s = FiveDimState::new(1.5, -0.5, -1.0, 2.0, 100.0);
        let clamped = s.clamp_to_omega();
        assert_eq!(clamped.intrinsic_degree, 1.0);
        assert_eq!(clamped.binding_degree, 0.0);
        assert_eq!(clamped.energy_density, 0.0);
        assert_eq!(clamped.evolution_rate, 1.0);
        assert_eq!(clamped.structural_robustness, 1.0);
    }

    #[test]
    fn test_total_change() {
        let a = FiveDimState::zero();
        let b = FiveDimState::new(1.0, 1.0, 1.0, 1.0, 1.0);
        assert!((a.total_change(&b) - 5.0).abs() < 1e-10);
    }
}
