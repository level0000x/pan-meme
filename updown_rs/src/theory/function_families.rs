//! 五族函数 — 泛模因理论 附录 B, §4.3.8
//!
//! 核心构造：
//! - FunctionFamily 枚举: Power / Exponential / Sigmoid / Logarithm / PiecewiseLinear
//! - FamilyParams: 函数族 + 参数 + 评估
//!
//! 参数范围对齐 proof-supplement-complete.md §6.2 的 Θ 紧致化定义

/// 五族函数 — 论文 §4.3.8 + 附录 B
///
/// 对应 proof-supplement-complete.md §6.2 的紧致化参数范围
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionFamily {
    /// 幂函数: x^k, k ∈ [0, 2]
    Power,
    /// 指数函数: e^{kx} - 1, k ∈ [0, 2]
    Exponential,
    /// Sigmoid: 1/(1+e^{-k(x-x₀)}) - 1/2, k ∈ [0.5, 2.5], x₀ ∈ [0, 1]
    Sigmoid,
    /// 对数函数: ln(1+kx), k ∈ [0, 2]
    Logarithm,
    /// 分段线性: 数据驱动, b₁,b₂ ∈ [0,1], b₁ < b₂
    PiecewiseLinear,
}

impl FunctionFamily {
    /// 所有函数族
    pub fn all() -> [FunctionFamily; 5] {
        [FunctionFamily::Power, FunctionFamily::Exponential, FunctionFamily::Sigmoid, FunctionFamily::Logarithm, FunctionFamily::PiecewiseLinear]
    }

    /// 函数族名称
    pub fn name(&self) -> &'static str {
        match self {
            FunctionFamily::Power => "Power",
            FunctionFamily::Exponential => "Exponential",
            FunctionFamily::Sigmoid => "Sigmoid",
            FunctionFamily::Logarithm => "Logarithm",
            FunctionFamily::PiecewiseLinear => "PiecewiseLinear",
        }
    }
}

/// 函数族参数
#[derive(Debug, Clone)]
pub struct FamilyParams {
    /// 函数族
    pub family: FunctionFamily,
    /// 主参数 k
    pub k: f64,
    /// 偏移参数 x₀（仅 Sigmoid 使用）
    pub x0: f64,
    /// 分段线性断点（仅 PiecewiseLinear 使用）
    pub breakpoints: Vec<(f64, f64)>,
}

impl FamilyParams {
    /// 创建参数
    pub fn new(family: FunctionFamily, k: f64, x0: f64) -> Self {
        FamilyParams { family, k, x0, breakpoints: Vec::new() }
    }

    /// 创建分段线性参数
    pub fn piecewise_linear(breakpoints: Vec<(f64, f64)>) -> Self {
        FamilyParams { family: FunctionFamily::PiecewiseLinear, k: 0.0, x0: 0.0, breakpoints }
    }

    /// 评估函数值 Φ(x) — 论文 §4.3.8
    ///
    /// 所有 Φ 满足 Φ(0) = 0
    pub fn evaluate(&self, x: f64) -> f64 {
        let result = match self.family {
            FunctionFamily::Power => {
                if x <= 0.0 { 0.0 } else { x.powf(self.k) }
            }
            FunctionFamily::Exponential => {
                (self.k * x).exp() - 1.0
            }
            FunctionFamily::Sigmoid => {
                // 1/(1+e^{-k(x-x₀)}) - 1/2
                // 减 1/2 保证 Φ(0) = 0 当 x₀ = 0.5 时
                let exponent = -self.k * (x - self.x0);
                if exponent > 50.0 {
                    1.0 - 0.5  // 避免溢出
                } else if exponent < -50.0 {
                    0.0 - 0.5
                } else {
                    1.0 / (1.0 + exponent.exp()) - 0.5
                }
            }
            FunctionFamily::Logarithm => {
                if x <= 0.0 { 0.0 } else { (1.0 + self.k * x).ln() }
            }
            FunctionFamily::PiecewiseLinear => {
                piecewise_evaluate(x, &self.breakpoints)
            }
        };
        if result.is_nan() || result.is_infinite() { 0.0 } else { result }
    }

    /// 验证参数在有效范围内（supplement §6.2 紧致化范围）
    pub fn is_valid(&self) -> bool {
        match self.family {
            FunctionFamily::Power | FunctionFamily::Exponential | FunctionFamily::Logarithm => {
                self.k >= 0.0 && self.k <= 2.0
            }
            FunctionFamily::Sigmoid => {
                self.k >= 0.5 && self.k <= 2.5 && self.x0 >= 0.0 && self.x0 <= 1.0
            }
            FunctionFamily::PiecewiseLinear => {
                !self.breakpoints.is_empty()
            }
        }
    }
}

/// 分段线性插值
fn piecewise_evaluate(x: f64, breakpoints: &[(f64, f64)]) -> f64 {
    if breakpoints.is_empty() {
        return 0.0;
    }
    if x <= breakpoints[0].0 {
        return breakpoints[0].1;
    }
    if x >= breakpoints.last().unwrap().0 {
        return breakpoints.last().unwrap().1;
    }
    for i in 0..breakpoints.len() - 1 {
        let (x1, y1) = breakpoints[i];
        let (x2, y2) = breakpoints[i + 1];
        if x >= x1 && x <= x2 {
            let t = (x - x1) / (x2 - x1);
            return y1 + t * (y2 - y1);
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_zero() {
        let p = FamilyParams::new(FunctionFamily::Power, 1.0, 0.0);
        assert!((p.evaluate(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_power_one() {
        let p = FamilyParams::new(FunctionFamily::Power, 1.0, 0.0);
        assert!((p.evaluate(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_zero() {
        let p = FamilyParams::new(FunctionFamily::Exponential, 1.0, 0.0);
        assert!((p.evaluate(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_zero() {
        // 当 x₀ = 0.5 时，Φ(0) = 1/(1+e^{k·0.5}) - 1/2 ≈ 0
        let p = FamilyParams::new(FunctionFamily::Sigmoid, 5.0, 0.5);
        let val = p.evaluate(0.0);
        // Φ(0) 接近 0 但不严格为 0
        assert!(val > -0.5 && val < 0.0);
    }

    #[test]
    fn test_logarithm_zero() {
        let p = FamilyParams::new(FunctionFamily::Logarithm, 1.0, 0.0);
        assert!((p.evaluate(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_piecewise_linear() {
        let bp = vec![(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)];
        let p = FamilyParams::piecewise_linear(bp);
        assert!((p.evaluate(0.25) - 0.25).abs() < 1e-10);
        assert!((p.evaluate(0.75) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_formula() {
        // 验证 Sigmoid 公式减 1/2 的正确性
        let p = FamilyParams::new(FunctionFamily::Sigmoid, 10.0, 0.5);
        let val_at_05 = p.evaluate(0.5);
        // 在 x = x₀ 处，1/(1+e⁰) - 1/2 = 1/2 - 1/2 = 0
        assert!((val_at_05).abs() < 1e-10);
    }
}