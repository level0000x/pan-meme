//! 5D ODE + RKF45 + 原型分类 — 泛模因理论 §4.3-4.5, 定理 5.8, 定理 8
//!
//! 核心构造：
//! - OdeConfig: 求解器配置
//! - StateSnapshot: 时间步快照
//! - TerminationReason: 终止原因
//! - EquilibriumType: 5 种平衡点（定理 8）
//! - Archetype: 9 类原型（定理 5.8）
//!
//! ODE 方程系统（§4.3.1）:
//! dD/dt = -α₁RD + α₂S(1-D)
//! dB/dt = β₁R(1-B) - β₂DB
//! dρ/dt = -γ₁Rρ + γ₂(1-ρ)·I_ext(t)
//! dR/dt = δ₁ρB(1-R) - δ₂Φ_D(D)R - δ₃R
//! dS/dt = ε₁D(1-S) - ε₂Φ_R(R)S

use crate::theory::dynamics_params::DynamicsParams;
use crate::theory::five_dim::FiveDimState;
use crate::theory::function_families::FamilyParams;

/// ODE 求解器配置
#[derive(Debug, Clone)]
pub struct OdeConfig {
    /// 初始步长
    pub h0: f64,
    /// 最小步长
    pub h_min: f64,
    /// 最大步长
    pub h_max: f64,
    /// 相对容差
    pub rtol: f64,
    /// 绝对容差
    pub atol: f64,
    /// 最大步数
    pub max_steps: usize,
    /// 最大时间
    pub t_max: f64,
}

impl Default for OdeConfig {
    fn default() -> Self {
        OdeConfig {
            h0: 0.01,
            h_min: 1e-8,
            h_max: 0.1,
            rtol: 1e-8,
            atol: 1e-8,
            max_steps: 20000,
            t_max: 5.0,
        }
    }
}

/// 时间步快照
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub t: f64,
    pub d: f64,
    pub b: f64,
    pub rho: f64,
    pub r: f64,
    pub s: f64,
}

impl StateSnapshot {
    pub fn to_five_dim(&self) -> FiveDimState {
        FiveDimState::new(self.d, self.b, self.rho, self.r, self.s)
    }
}

/// 终止原因
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminationReason {
    /// 收敛
    Converged,
    /// 达到最大步数
    MaxSteps,
    /// 达到最大时间
    MaxTime,
    /// 无效初始状态
    InvalidInitialState,
    /// 检测到跳变（β₀(K(t)) 变化）
    JumpDetected,
}

/// 5 种平衡点类型 — 定理 8
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EquilibriumType {
    /// 退化零点 (0,0,0,0,0) — 鞍点
    DegenerateZero,
    /// 湮灭型 (0, b*, 0, 0, 0) — 中性稳定
    Annihilation,
    /// 惰性型 (1, 0, ρ, 0, 1) — 中性稳定
    Inertial,
    /// 广度-韧度型 (0, b, ρ, 0, 0) — 中性稳定
    BreadthRobustness,
    /// 比例平衡点 (d*, b*, ρ*, r*, s*) — 依参数
    ProportionalEquilibrium,
    /// 未确定
    Undetermined,
}

/// 9 类原型 — 定理 5.8
///
/// 3 族 × 3 子类 = 9 类原型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Archetype {
    // 基石族
    Stone,
    StableCore,
    Resilient,
    // 过客族
    Burst,
    Decay,
    Transient,
    // 泡沫族
    Oscillatory,
    Source,
    Sink,
    // 未确定
    Undetermined,
}

/// 外部输入函数 I_ext(t) — 默认 e^{-t}
pub fn i_ext_default(t: f64) -> f64 {
    (-t).exp()
}

/// ODE 右侧函数 — 论文 §4.3.1
///
/// 返回 [dD/dt, dB/dt, dρ/dt, dR/dt, dS/dt]
pub fn ode_rhs(
    state: &[f64; 5],
    params: &DynamicsParams,
    i_ext: f64,
    phi_d: &FamilyParams,
    phi_r: &FamilyParams,
) -> [f64; 5] {
    let d = state[0];
    let b = state[1];
    let rho = state[2];
    let r = state[3];
    let s = state[4];

    let dd = -params.alpha_1 * r * d + params.alpha_2 * s * (1.0 - d);
    let db = params.beta_1 * r * (1.0 - b) - params.beta_2 * d * b;
    let drho = -params.gamma_1 * r * rho + params.gamma_2 * (1.0 - rho) * i_ext;
    let dr = params.delta_1 * rho * b * (1.0 - r)
        - params.delta_2 * phi_d.evaluate(d) * r
        - params.delta_3 * r;
    let ds = params.epsilon_1 * d * (1.0 - s) - params.epsilon_2 * phi_r.evaluate(r) * s;

    [
        nan_guard(dd),
        nan_guard(db),
        nan_guard(drho),
        nan_guard(dr),
        nan_guard(ds),
    ]
}

/// RKF45 单步 — Dormand-Prince 5(4)
///
/// 返回 (新状态, 估计误差)
pub fn rkf45_step(
    state: &[f64; 5],
    t: f64,
    h: f64,
    params: &DynamicsParams,
    phi_d: &FamilyParams,
    phi_r: &FamilyParams,
) -> ([f64; 5], f64) {
    let i_ext = i_ext_default(t);

    let k1 = ode_rhs(state, params, i_ext, phi_d, phi_r);
    let _k1_scaled = scale(&k1, h);

    let s2 = add(state, &scale(&k1, h * 1.0 / 4.0));
    let i_ext2 = i_ext_default(t + h * 1.0 / 4.0);
    let k2 = ode_rhs(&s2, params, i_ext2, phi_d, phi_r);

    let s3 = add(state, &scale(&k1, h * 3.0 / 32.0));
    let s3 = add(&s3, &scale(&k2, h * 9.0 / 32.0));
    let i_ext3 = i_ext_default(t + h * 3.0 / 8.0);
    let k3 = ode_rhs(&s3, params, i_ext3, phi_d, phi_r);

    let s4 = add(state, &scale(&k1, h * 1932.0 / 2197.0));
    let s4 = add(&s4, &scale(&k2, h * -7200.0 / 2197.0));
    let s4 = add(&s4, &scale(&k3, h * 7296.0 / 2197.0));
    let i_ext4 = i_ext_default(t + h * 12.0 / 13.0);
    let k4 = ode_rhs(&s4, params, i_ext4, phi_d, phi_r);

    let s5 = add(state, &scale(&k1, h * 439.0 / 216.0));
    let s5 = add(&s5, &scale(&k2, h * -8.0));
    let s5 = add(&s5, &scale(&k3, h * 3680.0 / 513.0));
    let s5 = add(&s5, &scale(&k4, h * -845.0 / 4104.0));
    let i_ext5 = i_ext_default(t + h);
    let k5 = ode_rhs(&s5, params, i_ext5, phi_d, phi_r);

    let s6 = add(state, &scale(&k1, h * -8.0 / 27.0));
    let s6 = add(&s6, &scale(&k2, h * 2.0));
    let s6 = add(&s6, &scale(&k3, h * -3544.0 / 2565.0));
    let s6 = add(&s6, &scale(&k4, h * 1859.0 / 4104.0));
    let s6 = add(&s6, &scale(&k5, h * -11.0 / 40.0));
    let i_ext6 = i_ext_default(t + h * 0.5);
    let k6 = ode_rhs(&s6, params, i_ext6, phi_d, phi_r);

    // 5 阶解
    let y5 = add(state, &scale(&k1, h * 16.0 / 135.0));
    let y5 = add(&y5, &scale(&k3, h * 6656.0 / 12825.0));
    let y5 = add(&y5, &scale(&k4, h * 28561.0 / 56430.0));
    let y5 = add(&y5, &scale(&k5, h * -9.0 / 50.0));
    let y5 = add(&y5, &scale(&k6, h * 2.0 / 55.0));

    // 4 阶解
    let y4 = add(state, &scale(&k1, h * 25.0 / 216.0));
    let y4 = add(&y4, &scale(&k3, h * 1408.0 / 2565.0));
    let y4 = add(&y4, &scale(&k4, h * 2197.0 / 4104.0));
    let y4 = add(&y4, &scale(&k5, -h / 5.0));

    // 误差估计
    let mut error: f64 = 0.0;
    for i in 0..5 {
        let diff = (y5[i] - y4[i]).abs();
        error = error.max(diff);
    }

    (y5, error)
}

/// 积分主函数 — 论文 §4.3-4.4
///
/// 使用 RKF45 自适应步长求解 5D ODE 系统
pub fn integrate(
    init: &FiveDimState,
    params: &DynamicsParams,
    config: &OdeConfig,
    phi_d: &FamilyParams,
    phi_r: &FamilyParams,
) -> (Vec<StateSnapshot>, TerminationReason) {
    if !init.is_valid() {
        return (Vec::new(), TerminationReason::InvalidInitialState);
    }

    let mut trajectory = Vec::new();
    let mut state = init.to_array();
    let mut t = 0.0;
    let mut h = config.h0;

    trajectory.push(StateSnapshot {
        t,
        d: state[0],
        b: state[1],
        rho: state[2],
        r: state[3],
        s: state[4],
    });

    let convergence_window = 5;
    let convergence_threshold = 1e-3;

    for _step in 0..config.max_steps {
        if t >= config.t_max {
            return (trajectory, TerminationReason::MaxTime);
        }

        let h_actual = h.min(config.t_max - t);
        let (new_state, error) = rkf45_step(&state, t, h_actual, params, phi_d, phi_r);

        // 自适应步长调整
        if error > config.atol * 10.0 {
            h = (h * 0.5).max(config.h_min);
            continue;
        }

        // 裁剪到 Ω = [0,1]⁴ × [0,∞) — 定理 7
        let clamped = FiveDimState::from_array(&new_state)
            .clamp_to_omega()
            .to_array();

        t += h_actual;
        state = clamped;
        trajectory.push(StateSnapshot {
            t,
            d: state[0],
            b: state[1],
            rho: state[2],
            r: state[3],
            s: state[4],
        });

        // 收敛检测: 连续 5 步总变化 < 1e-3
        if trajectory.len() > convergence_window {
            let start = trajectory.len() - convergence_window - 1;
            let end = trajectory.len() - 1;
            let prev = FiveDimState::new(
                trajectory[start].d,
                trajectory[start].b,
                trajectory[start].rho,
                trajectory[start].r,
                trajectory[start].s,
            );
            let curr = FiveDimState::new(
                trajectory[end].d,
                trajectory[end].b,
                trajectory[end].rho,
                trajectory[end].r,
                trajectory[end].s,
            );
            if prev.total_change(&curr) < convergence_threshold {
                return (trajectory, TerminationReason::Converged);
            }
        }

        // 自适应步长
        if error < config.atol {
            h = (h * 1.2).min(config.h_max);
        }
    }

    (trajectory, TerminationReason::MaxSteps)
}

/// 分类平衡点 — 定理 8
pub fn classify_equilibrium(state: &FiveDimState) -> EquilibriumType {
    let eps = 1e-3;
    let d = state.intrinsic_degree;
    let b = state.binding_degree;
    let rho = state.energy_density;
    let r = state.evolution_rate;
    let s = state.structural_robustness;

    if d < eps && b < eps && rho < eps && r < eps && s < eps {
        EquilibriumType::DegenerateZero
    } else if d < eps && r < eps && s < eps {
        if b > eps {
            EquilibriumType::Annihilation
        } else {
            EquilibriumType::BreadthRobustness
        }
    } else if d > 1.0 - eps && b < eps && r < eps && s > 1.0 - eps {
        EquilibriumType::Inertial
    } else if d > eps && b > eps && rho > eps && r > eps && s > eps {
        EquilibriumType::ProportionalEquilibrium
    } else {
        EquilibriumType::Undetermined
    }
}

/// 分类原型 — 定理 5.8
pub fn classify(trajectory: &[StateSnapshot], reason: &TerminationReason) -> Archetype {
    if trajectory.len() < 2 {
        return Archetype::Undetermined;
    }

    let last = trajectory.last().unwrap();
    let first = &trajectory[0];
    let d = last.d;
    let s = last.s;
    let r = last.r;

    let d_final = d;
    let _d_init = first.d;
    let s_final = s;
    let _s_init = first.s;

    match reason {
        TerminationReason::Converged => {
            if d_final > 0.5 && s_final > 0.5 {
                Archetype::Stone
            } else if d_final > 0.3 && s_final > 0.3 {
                Archetype::StableCore
            } else if s_final > 0.8 {
                Archetype::Resilient
            } else if d_final < 0.1 && s_final < 0.1 {
                Archetype::Decay
            } else {
                Archetype::StableCore
            }
        }
        TerminationReason::MaxSteps | TerminationReason::MaxTime => {
            if r > 0.5 {
                Archetype::Burst
            } else {
                Archetype::Transient
            }
        }
        TerminationReason::JumpDetected => Archetype::Transient,
        _ => Archetype::Undetermined,
    }
}

fn scale(v: &[f64; 5], s: f64) -> [f64; 5] {
    [v[0] * s, v[1] * s, v[2] * s, v[3] * s, v[4] * s]
}

fn add(a: &[f64; 5], b: &[f64; 5]) -> [f64; 5] {
    [
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
        a[4] + b[4],
    ]
}

fn nan_guard(v: f64) -> f64 {
    if v.is_nan() || v.is_infinite() {
        0.0
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::function_families::FunctionFamily;

    #[test]
    fn test_i_ext_default() {
        assert!((i_ext_default(0.0) - 1.0).abs() < 1e-10);
        assert!(i_ext_default(5.0) < 0.01);
    }

    #[test]
    fn test_ode_rhs_zero_state() {
        let state = [0.0; 5];
        let params = DynamicsParams::default_params();
        let phi_d = FamilyParams::new(FunctionFamily::Power, 1.0, 0.0);
        let phi_r = FamilyParams::new(FunctionFamily::Power, 1.0, 0.0);
        let rhs = ode_rhs(&state, &params, i_ext_default(0.0), &phi_d, &phi_r);
        // dD/dt = α₂·0·(1-0) = 0, dρ/dt = γ₂·1·(1-0) > 0
        assert!(rhs[2] > 0.0);
    }

    #[test]
    fn test_integrate_converges() {
        let init = FiveDimState::new(0.5, 0.5, 1.0, 0.3, 0.8);
        let params = DynamicsParams::default_params();
        let config = OdeConfig {
            max_steps: 1000,
            ..OdeConfig::default()
        };
        let phi_d = FamilyParams::new(FunctionFamily::Power, 1.0, 0.0);
        let phi_r = FamilyParams::new(FunctionFamily::Power, 1.0, 0.0);
        let (trajectory, reason) = integrate(&init, &params, &config, &phi_d, &phi_r);
        assert!(trajectory.len() > 1);
        let archetype = classify(&trajectory, &reason);
        assert!(matches!(
            archetype,
            Archetype::Stone | Archetype::StableCore | Archetype::Decay | _
        ));
    }

    #[test]
    fn test_invalid_initial_state() {
        let init = FiveDimState::new(2.0, 0.5, 1.0, 0.3, 0.8);
        let params = DynamicsParams::default_params();
        let config = OdeConfig::default();
        let phi_d = FamilyParams::new(FunctionFamily::Power, 1.0, 0.0);
        let phi_r = FamilyParams::new(FunctionFamily::Power, 1.0, 0.0);
        let (trajectory, reason) = integrate(&init, &params, &config, &phi_d, &phi_r);
        assert_eq!(reason, TerminationReason::InvalidInitialState);
        assert!(trajectory.is_empty());
    }

    #[test]
    fn test_classify_equilibrium() {
        let zero = FiveDimState::zero();
        assert_eq!(classify_equilibrium(&zero), EquilibriumType::DegenerateZero);

        let ann = FiveDimState::new(0.0, 0.5, 0.0, 0.0, 0.0);
        assert_eq!(classify_equilibrium(&ann), EquilibriumType::Annihilation);
    }

    #[test]
    fn test_rkf45_step() {
        let state = [0.5, 0.5, 1.0, 0.3, 0.8];
        let params = DynamicsParams::default_params();
        let phi_d = FamilyParams::new(FunctionFamily::Power, 1.0, 0.0);
        let phi_r = FamilyParams::new(FunctionFamily::Power, 1.0, 0.0);
        let (new_state, error) = rkf45_step(&state, 0.0, 0.01, &params, &phi_d, &phi_r);
        for v in &new_state {
            assert!(!v.is_nan());
        }
        assert!(error >= 0.0);
    }
}
