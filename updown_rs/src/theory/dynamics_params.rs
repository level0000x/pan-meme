//! 11 参数动力学 — 泛模因理论 §3.4.3, §4.3.7, 附录 A.3
//!
//! 核心构造：DynamicsParams = {α₁, α₂, β₁, β₂, γ₁, γ₂, δ₁, δ₂, δ₃, ε₁, ε₂}
//!
//! 参数来源按论文 §3.4.3 的几何推导方向。
//! 部分参数的具体公式为简化近似，后续可迭代精确化为论文的几何推导。

use crate::theory::five_dim::FiveDimState;

/// 11 个动力学参数 — 论文 §4.3.7, 附录 A.3
///
/// 每个参数对应 ODE 方程中的一个系数
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicsParams {
    /// α₁: 简化效应 — R 对 D 的稀释速率
    pub alpha_1: f64,
    /// α₂: 沉淀效应 — S 对 D 的深化速率
    pub alpha_2: f64,
    /// β₁: 扩张耦合 — R 对 B 的扩张效率
    pub beta_1: f64,
    /// β₂: 泛化权衡 — D 对 B 的抑制强度
    pub beta_2: f64,
    /// γ₁: 能流耗散 — R 对 ρ 的消耗速率
    pub gamma_1: f64,
    /// γ₂: 外部赋能 — I_ext 对 ρ 的注入效率
    pub gamma_2: f64,
    /// δ₁: 核心驱动力 — ρ·B 对 R 的促进效率
    pub delta_1: f64,
    /// δ₂: 深度诅咒 — Φ_D(D) 对 R 的抑制强度
    pub delta_2: f64,
    /// δ₃: 自发衰退 — R 的自然衰减率
    pub delta_3: f64,
    /// ε₁: 深度基石 — D 对 S 的奠基效率
    pub epsilon_1: f64,
    /// ε₂: 速朽定律 — Φ_R(R) 对 S 的消耗强度
    pub epsilon_2: f64,
}

impl DynamicsParams {
    /// 从几何特征推导 11 参数 — 论文 §3.4.3
    ///
    /// 参数来源按论文的几何推导方向：
    /// - α₁: 边密度 2|E|/(|V|(|V|-1))
    /// - α₂: 层级深度 depth/d_max
    /// - β₁: 外部连接数（论文 §3.4.3）
    /// - β₂: 内部复杂度/外部连接数（论文 §3.4.3）
    /// - γ₁: 场散度 mean(|∇φ|)
    /// - γ₂: 边界流入通量（论文 §3.4.3）
    /// - δ₁: 曲率 × 连接密度
    /// - δ₂: 层级深度 × 内部复杂度
    /// - δ₃: 拓扑不变量数量倒数 1/(1+|Γ|)
    /// - ε₁: 层级深度 × 稳定性指标
    /// - ε₂: 曲率变化率 × 场散度
    pub fn from_geometry(
        n_vertices: usize,
        n_edges: usize,
        depth: f64,
        max_depth: f64,
        state: &FiveDimState,
        grad_mean: f64,
        grad_std: f64,
    ) -> Self {
        let nv = n_vertices as f64;
        let ne = n_edges as f64;

        // α₁: 边密度
        let alpha_1 = if nv > 1.0 {
            (2.0 * ne) / (nv * (nv - 1.0))
        } else {
            0.0
        };

        // α₂: 层级深度
        let alpha_2 = if max_depth > 0.0 { depth / max_depth } else { 0.0 };

        // β₁: 外部连接数 (简化近似)
        let beta_1 = if nv > 0.0 { ne / nv } else { 0.5 };

        // β₂: 内部复杂度/外部连接数
        let beta_2 = if nv > 0.0 && ne > 0.0 {
            state.intrinsic_degree / (ne / nv).max(0.001)
        } else {
            0.5
        };

        // γ₁: 场散度
        let gamma_1 = grad_mean.abs();

        // γ₂: 边界流入通量 (简化近似)
        let gamma_2 = 0.3;

        // δ₁: 曲率 × 连接密度
        let delta_1 = state.evolution_rate * state.binding_degree;

        // δ₂: 层级深度 × 内部复杂度
        let delta_2 = depth * state.intrinsic_degree;

        // δ₃: 拓扑不变量数量倒数
        let delta_3 = 1.0 / (1.0 + (ne - nv + 1.0).abs());

        // ε₁: 层级深度 × 稳定性指标
        let epsilon_1 = depth * state.structural_robustness;

        // ε₂: 曲率变化率 × 场散度
        let epsilon_2 = grad_std * grad_mean.abs();

        DynamicsParams {
            alpha_1: nan_guard(alpha_1),
            alpha_2: nan_guard(alpha_2),
            beta_1: nan_guard(beta_1),
            beta_2: nan_guard(beta_2),
            gamma_1: nan_guard(gamma_1),
            gamma_2: nan_guard(gamma_2),
            delta_1: nan_guard(delta_1),
            delta_2: nan_guard(delta_2),
            delta_3: nan_guard(delta_3),
            epsilon_1: nan_guard(epsilon_1),
            epsilon_2: nan_guard(epsilon_2),
        }
    }

    /// 默认参数（用于测试）
    pub fn default_params() -> Self {
        DynamicsParams {
            alpha_1: 0.1, alpha_2: 0.2,
            beta_1: 0.1, beta_2: 0.1,
            gamma_1: 0.1, gamma_2: 0.3,
            delta_1: 0.2, delta_2: 0.1, delta_3: 0.05,
            epsilon_1: 0.2, epsilon_2: 0.1,
        }
    }
}

/// NaN 防护
fn nan_guard(v: f64) -> f64 {
    if v.is_nan() || v.is_infinite() { 0.0 } else { v }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_geometry() {
        let state = FiveDimState::new(0.5, 0.5, 1.0, 0.3, 0.8);
        let params = DynamicsParams::from_geometry(10, 20, 3.0, 5.0, &state, 0.2, 0.05);
        assert!(params.alpha_1 >= 0.0);
        assert!(params.alpha_2 >= 0.0);
        assert!(params.gamma_1 >= 0.0);
    }

    #[test]
    fn test_default_params() {
        let p = DynamicsParams::default_params();
        assert!(p.alpha_1 > 0.0);
        assert!(p.epsilon_1 > 0.0);
    }
}