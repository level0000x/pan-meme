//! Phase 5: ODE 求解器
//!
//! 数学对应：§D.5 五维动力学系统 + 定理6+7+8+10
//!
//! 5D ODE 系统（每个模因独立）：
//!   dD/dt = α₁·Φ_D(D) - α₂·D·B²
//!   dB/dt = β₁·B·(1-B) - β₂·D·B
//!   dρ/dt = γ₁·D·B - γ₂·ρ
//!   dR/dt = δ₁·ρ·R·(1-R) - δ₂·D·R - δ₃·R
//!   dS/dt = ε₁·D·ρ - ε₂·S
//!
//! 求解器：RKF45 (Runge-Kutta-Fehlberg 4(5)) 自适应步长
//! 定理7：Ω = [0,1]⁵ 是不变集（逐维边界裁剪）

use crate::encoding::decomposition::{DynamicsParams, Coupling};

// ═══════════════════════════════════════════════════════════════════════
// ODE 配置
// ═══════════════════════════════════════════════════════════════════════

/// ODE 求解器配置
#[derive(Debug, Clone)]
pub struct OdeConfig {
    /// 初始步长
    pub h0: f64,
    /// 最小步长（低于此值视为收敛）
    pub h_min: f64,
    /// 最大步长
    pub h_max: f64,
    /// 相对误差容限
    pub rtol: f64,
    /// 绝对误差容限
    pub atol: f64,
    /// 最大积分步数
    pub max_steps: usize,
    /// 仿真时间上限
    pub t_max: f64,
    /// 函数族选择: 0=power, 1=exp, 2=sigmoid, 3=log, 4=piecewise
    pub function_family: usize,
    /// 函数族参数 (k for power/exp/log, (k, x0) for sigmoid)
    pub fn_params: Vec<f64>,
}

impl Default for OdeConfig {
    fn default() -> Self {
        Self {
            h0: 0.01,
            h_min: 1e-8,
            h_max: 0.5,
            rtol: 1e-6,
            atol: 1e-8,
            max_steps: 100_000,
            t_max: 100.0,
            function_family: 0,  // 默认幂函数
            fn_params: vec![1.5], // k=1.5 超线性
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 5 个函数族（§D.3 假设0 — F 空间）
// ═══════════════════════════════════════════════════════════════════════

/// 候选函数族 F = {幂, 指数, Sigmoid, 对数, 分段线性}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionFamily {
    Power = 0,
    Exp = 1,
    Sigmoid = 2,
    Log = 3,
    Piecewise = 4,
}

impl FunctionFamily {
    pub const ALL: [FunctionFamily; 5] = [
        FunctionFamily::Power,
        FunctionFamily::Exp,
        FunctionFamily::Sigmoid,
        FunctionFamily::Log,
        FunctionFamily::Piecewise,
    ];

    /// 参数个数
    pub fn n_params(self) -> usize {
        match self {
            FunctionFamily::Power | FunctionFamily::Exp | FunctionFamily::Log => 1,
            FunctionFamily::Sigmoid => 2,
            FunctionFamily::Piecewise => 3, // 2 断点 → 3 参数
        }
    }

    /// Lipschitz 常数估计（用于步长上界 h ≤ 1/max(δ₃, ε₂, L)）
    pub fn lipschitz_constant(self, params: &[f64]) -> f64 {
        let k = params.first().copied().unwrap_or(1.0);
        match self {
            FunctionFamily::Power => k.abs(),
            FunctionFamily::Exp => (k.abs() * (k.abs() * 1.0).exp()).min(20.0),
            FunctionFamily::Sigmoid => k.abs() / 4.0,
            FunctionFamily::Log => k.abs(),
            FunctionFamily::Piecewise => 1.0,
        }
    }

    /// 计算 f(x) with x ∈ [0,1]
    pub fn evaluate(self, x: f64, params: &[f64]) -> f64 {
        let k = params.first().copied().unwrap_or(1.0);
        let x_clamped = x.max(0.0).min(10.0);
        match self {
            FunctionFamily::Power => x_clamped.powf(k),
            FunctionFamily::Exp => (k * x_clamped).exp() - 1.0,
            FunctionFamily::Sigmoid => {
                let x0 = params.get(1).copied().unwrap_or(0.5);
                1.0 / (1.0 + (-k * (x_clamped - x0)).exp()) - if x0 < -50.0 { 0.0 } else { 0.5 }
            }
            FunctionFamily::Log => (1.0 + k * x_clamped).ln(),
            FunctionFamily::Piecewise => {
                let b1 = params.get(1).copied().unwrap_or(0.3);
                let b2 = params.get(2).copied().unwrap_or(0.7);
                if x < b1 { x / b1 * 0.3 }
                else if x < b2 { 0.3 + (x - b1) / (b2 - b1) * 0.6 }
                else { 0.9 + (x - b2) / (1.0 - b2) * 0.1 }
            }
        }
    }

    /// 导数 f'(x)
    pub fn derivative(self, x: f64, params: &[f64]) -> f64 {
        let k = params.first().copied().unwrap_or(1.0);
        let x_clamped = x.max(0.0).min(10.0);
        match self {
            FunctionFamily::Power => {
                if x_clamped < 1e-10 && k < 1.0 { 1e10 } else { k * x_clamped.powf(k - 1.0) }
            }
            FunctionFamily::Exp => k * (k * x_clamped).exp(),
            FunctionFamily::Sigmoid => {
                let x0 = params.get(1).copied().unwrap_or(0.5);
                let s = 1.0 / (1.0 + (-k * (x_clamped - x0)).exp());
                k * s * (1.0 - s)
            }
            FunctionFamily::Log => k / (1.0 + k * x_clamped),
            FunctionFamily::Piecewise => {
                let b1 = params.get(1).copied().unwrap_or(0.3);
                let b2 = params.get(2).copied().unwrap_or(0.7);
                if x < b1 { 0.3 / b1 }
                else if x < b2 { 0.6 / (b2 - b1) }
                else { 0.1 / (1.0 - b2) }
            }
        }
    }

    /// 从 usize 构造
    pub fn from_usize(n: usize) -> Self {
        match n {
            1 => FunctionFamily::Exp,
            2 => FunctionFamily::Sigmoid,
            3 => FunctionFamily::Log,
            4 => FunctionFamily::Piecewise,
            _ => FunctionFamily::Power,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 单步五维状态（轨迹快照）
// ═══════════════════════════════════════════════════════════════════════

/// 五维状态在时刻 t 的快照
#[derive(Debug, Clone)]
pub struct FiveDimSnapshot {
    pub t: f64,
    pub d: f64,
    pub b: f64,
    pub rho: f64,
    pub r: f64,
    pub s: f64,
}

// ═══════════════════════════════════════════════════════════════════════
// 单模因轨迹
// ═══════════════════════════════════════════════════════════════════════

/// 单个模因的 ODE 轨迹
#[derive(Debug, Clone)]
pub struct MemeTrajectory {
    pub meme_id: usize,
    pub trajectory: Vec<FiveDimSnapshot>,
    pub terminated_at: f64,
    pub termination_reason: TerminationReason,
    pub hopf_warning: bool,
}

/// 轨迹终止原因
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminationReason {
    /// 收敛于平衡点
    Converged,
    /// 达到最大步数
    MaxSteps,
    /// 达到最大时间
    MaxTime,
    /// 步长低于阈值（接近奇点）
    StepTooSmall,
    /// 发散（数值不稳定）
    Diverged,
}

// ═══════════════════════════════════════════════════════════════════════
// 收敛分类（定理8 + 定理10）
// ═══════════════════════════════════════════════════════════════════════

/// 平衡点分类（定理8）
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EquilibriumClass {
    /// 退化零点 (0,0,0,0,0) — 孤立稳定点
    DegenerateZero,
    /// 湮灭型 (0,b*,0,0,0) — 中性稳定流形
    Annihilation,
    /// 惰性型 (1,0,ρ,0,1) — 中性稳定流形
    Inert,
    /// 广度-韧度型 (0,b,ρ,0,0) — 中性稳定流形
    BreadthRobustness,
    /// 比例平衡点 (d*,b*,ρ*,r*,s*) — 稳定性依参数
    ProportionalEquilibrium,
    /// 无法归类
    Unclassified,
}

/// 三类原型（定理10 — 能量函数 V + LaSalle 不变原理）
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemeArchetype {
    /// 基石模因：V 快速下降到低值，所有维度稳定
    Stone,
    /// 过客模因：V 先上升后下降，短暂活跃后湮灭
    Transient,
    /// 泡沫模因：V 持续下降但 D→0，最终坍缩
    Bubble,
    /// 无法判断
    Undetermined,
}

// ═══════════════════════════════════════════════════════════════════════
// ODE 求解器输出
// ═══════════════════════════════════════════════════════════════════════

/// Phase 5（ODE 演化）完整输出
#[derive(Debug)]
pub struct OdeOutput {
    /// 每个模因的独立轨迹
    pub trajectories: Vec<MemeTrajectory>,
    /// 收敛分类结果
    pub archetypes: Vec<(usize, MemeArchetype, EquilibriumClass)>,
    /// 跳变点（拓扑变点 t 值）
    pub jump_points: Vec<f64>,
    /// 跳变时的模因数量变化
    pub jump_meme_counts: Vec<(f64, usize)>,
}

// ═══════════════════════════════════════════════════════════════════════
// RKF45 内核
// ═══════════════════════════════════════════════════════════════════════

/// RKF45 单步积分。
///
/// 返回 (y_new, error_estimate) 其中 error = ||y5 - y4||。
fn rkf45_step(
    y: &[f64; 5],
    t: f64,
    h: f64,
    rhs: impl Fn(f64, &[f64; 5]) -> [f64; 5],
) -> ([f64; 5], f64) {
    // Butcher tableau for RKF45
    let a2 = [1.0/4.0];
    let a3 = [3.0/32.0, 9.0/32.0];
    let a4 = [1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0];
    let a5 = [439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0];
    let a6 = [-8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0];

    let b4 = [25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0, 0.0];
    let b5 = [16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0];

    let c = [0.0, 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0];

    // k1
    let k1 = rhs(t, y);

    // k2
    let mut y2 = [0.0; 5];
    for i in 0..5 { y2[i] = y[i] + h * (a2[0] * k1[i]); }
    let k2 = rhs(t + c[1] * h, &y2);

    // k3
    let mut y3 = [0.0; 5];
    for i in 0..5 { y3[i] = y[i] + h * (a3[0] * k1[i] + a3[1] * k2[i]); }
    let k3 = rhs(t + c[2] * h, &y3);

    // k4
    let mut y4 = [0.0; 5];
    for i in 0..5 { y4[i] = y[i] + h * (a4[0] * k1[i] + a4[1] * k2[i] + a4[2] * k3[i]); }
    let k4 = rhs(t + c[3] * h, &y4);

    // k5
    let mut y5 = [0.0; 5];
    for i in 0..5 { y5[i] = y[i] + h * (a5[0] * k1[i] + a5[1] * k2[i] + a5[2] * k3[i] + a5[3] * k4[i]); }
    let k5 = rhs(t + c[4] * h, &y5);

    // k6
    let mut y6 = [0.0; 5];
    for i in 0..5 { y6[i] = y[i] + h * (a6[0] * k1[i] + a6[1] * k2[i] + a6[2] * k3[i] + a6[3] * k4[i] + a6[4] * k5[i]); }
    let k6 = rhs(t + c[5] * h, &y6);

    // 4th order
    let mut y4_out = [0.0; 5];
    for i in 0..5 {
        y4_out[i] = y[i] + h * (b4[0] * k1[i] + b4[1] * k2[i] + b4[2] * k3[i] + b4[3] * k4[i] + b4[4] * k5[i] + b4[5] * k6[i]);
    }

    // 5th order
    let mut y5_out = [0.0; 5];
    for i in 0..5 {
        y5_out[i] = y[i] + h * (b5[0] * k1[i] + b5[1] * k2[i] + b5[2] * k3[i] + b5[3] * k4[i] + b5[4] * k5[i] + b5[5] * k6[i]);
    }

    // 误差估计
    let mut err = 0.0;
    for i in 0..5 {
        let diff = (y5_out[i] - y4_out[i]).abs();
        err += diff * diff;
    }
    err = err.sqrt();

    (y5_out, err)
}

/// 自适应步长调整
fn adapt_step(h: f64, err: f64, rtol: f64, atol: f64, h_max: f64, h_min: f64) -> (f64, bool) {
    if err < 1e-15 { return (h.min(h_max * 1.5), true); }

    let safety = 0.9;
    let beta = 0.2; // PI controller

    let h_new = safety * h * (atol / err.max(atol)).powf(beta);
    let h_new = h_new.max(h_min).min(h_max);

    let accepted = err < rtol || err < atol;
    (h_new, accepted)
}

// ═══════════════════════════════════════════════════════════════════════
// 五维 RHS 函数
// ═══════════════════════════════════════════════════════════════════════

/// 5D ODE 右侧函数。
///
/// Ω 裁剪（定理7）：每步后裁剪到 [0,1]⁵ 保证不变性。
pub fn five_dim_rhs<'a>(params: &'a DynamicsParams, ff: FunctionFamily, fn_params: &'a [f64])
    -> impl Fn(f64, &[f64; 5]) -> [f64; 5] + 'a
{
    let a1 = params.alpha_1;
    let a2 = params.alpha_2;
    let b1 = params.beta_1;
    let b2 = params.beta_2;
    let g1 = params.gamma_1;
    let g2 = params.gamma_2;
    let d1 = params.delta_1;
    let d2 = params.delta_2;
    let d3 = params.delta_3;
    let e1 = params.epsilon_1;
    let e2 = params.epsilon_2;

    move |_t: f64, y: &[f64; 5]| -> [f64; 5] {
        let d = y[0].max(0.0).min(1.0);
        let b = y[1].max(0.0).min(1.0);
        let rho = y[2].max(0.0);  // ρ ∈ ℝ⁺ 无上界
        let r = y[3].max(0.0).min(1.0);
        let s = y[4].max(0.0).min(1.0);

        let phi_d = ff.evaluate(d, fn_params);

        [
            a1 * phi_d - a2 * d * b * b,        // dD/dt
            b1 * b * (1.0 - b) - b2 * d * b,    // dB/dt
            g1 * d * b - g2 * rho,               // dρ/dt
            d1 * rho * r * (1.0 - r) - d2 * d * r - d3 * r,  // dR/dt
            e1 * d * rho - e2 * s,               // dS/dt
        ]
    }
}

/// Ω 裁剪：保持 D,B,R,S ∈ [0,1], ρ ≥ 0
fn clip_to_omega(y: &mut [f64; 5]) {
    y[0] = y[0].max(0.0).min(1.0);
    y[1] = y[1].max(0.0).min(1.0);
    y[2] = y[2].max(0.0);
    y[3] = y[3].max(0.0).min(1.0);
    y[4] = y[4].max(0.0).min(1.0);
}

// ═══════════════════════════════════════════════════════════════════════
// 单个模因求解
// ═══════════════════════════════════════════════════════════════════════

/// 求解单个模因的 5D ODE。
///
/// 定理6（分段解存在唯一性）：
///   在 n(t) 不变的段内，RHS 局部 Lipschitz → 皮卡-林德勒夫保证唯一解。
pub fn solve_single_meme(
    meme_id: usize,
    initial: &FiveDimSnapshot,
    params: &DynamicsParams,
    config: &OdeConfig,
) -> MemeTrajectory {
    let ff = FunctionFamily::from_usize(config.function_family);
    let rhs = five_dim_rhs(params, ff, &config.fn_params);

    let mut y = [initial.d, initial.b, initial.rho, initial.r, initial.s];
    let mut t = initial.t;
    let mut h = config.h0;

    let mut trajectory = vec![initial.clone()];

    let mut hopf_warning = false;

    for _step_idx in 0..config.max_steps {
        if t >= config.t_max {
            return MemeTrajectory {
                meme_id, trajectory, terminated_at: t,
                termination_reason: TerminationReason::MaxTime, hopf_warning,
            };
        }

        h = h.min(config.t_max - t);

        let (y_new, err) = rkf45_step(&y, t, h, &rhs);
        let (h_new, accepted) = adapt_step(h, err, config.rtol, config.atol, config.h_max, config.h_min);

        if accepted {
            y = y_new;
            clip_to_omega(&mut y);
            t += h;

            let snap = FiveDimSnapshot {
                t, d: y[0], b: y[1], rho: y[2], r: y[3], s: y[4],
            };
            trajectory.push(snap);

            // 收敛检测：连续 10 步变化 < atol
            if trajectory.len() > 10 {
                let last = &trajectory[trajectory.len() - 1];
                let prev = &trajectory[trajectory.len() - 11];
                let change = (last.d - prev.d).abs() + (last.b - prev.b).abs()
                    + (last.rho - prev.rho).abs() + (last.r - prev.r).abs() + (last.s - prev.s).abs();
                if change < config.atol * 50.0 {
                    return MemeTrajectory {
                        meme_id, trajectory, terminated_at: t,
                        termination_reason: TerminationReason::Converged, hopf_warning,
                    };
                }
            }
        }

        h = h_new;
        if h < config.h_min {
            return MemeTrajectory {
                meme_id, trajectory, terminated_at: t,
                termination_reason: TerminationReason::StepTooSmall, hopf_warning,
            };
        }

        // Hopf 预警：S·R 乘积振荡（符号变换 ≥3 次）
        if trajectory.len() > 20 {
            let recent = &trajectory[trajectory.len() - 20..];
            let sr: Vec<f64> = recent.iter().map(|sn| sn.s * sn.r).collect();
            let mut sign_changes = 0;
            for i in 1..sr.len() {
                if (sr[i] - sr[i-1]) * (sr[i-1] - if i > 1 { sr[i-2] } else { sr[0] }) < 0.0 {
                    sign_changes += 1;
                }
            }
            if sign_changes >= 3 { hopf_warning = true; }
        }
    }

    MemeTrajectory {
        meme_id, trajectory, terminated_at: t,
        termination_reason: TerminationReason::MaxSteps, hopf_warning,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 所有模因求解 + 收敛分类
// ═══════════════════════════════════════════════════════════════════════

/// 对所有模因求解 ODE 并做收敛分类。
pub fn solve_all_memes(
    initial_states: &[(usize, FiveDimSnapshot)],
    params: &[DynamicsParams],
    _couplings: &[Coupling],
    config: &OdeConfig,
) -> OdeOutput {
    let mut trajectories = Vec::new();
    for &(meme_id, ref init) in initial_states {
        if meme_id < params.len() {
            trajectories.push(solve_single_meme(meme_id, init, &params[meme_id], config));
        }
    }

    let archetypes: Vec<_> = trajectories.iter().map(|traj| {
        let eq = classify_equilibrium(traj);
        let arch = classify_archetype(traj);
        (traj.meme_id, arch, eq)
    }).collect();

    OdeOutput {
        trajectories,
        archetypes,
        jump_points: Vec::new(),
        jump_meme_counts: Vec::new(),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 收敛分类（定理8 + 定理10）
// ═══════════════════════════════════════════════════════════════════════

/// 平衡点分类（定理8）。
fn classify_equilibrium(traj: &MemeTrajectory) -> EquilibriumClass {
    if traj.trajectory.len() < 2 { return EquilibriumClass::Unclassified; }
    let final_state = traj.trajectory.last().unwrap();
    let d = final_state.d;
    let b = final_state.b;
    let rho = final_state.rho;
    let r = final_state.r;
    let s = final_state.s;

    let eps = 0.01;
    if d < eps && b < eps && rho < eps && r < eps && s < eps {
        EquilibriumClass::DegenerateZero
    } else if d < eps && b > eps && rho < eps && r < eps && s < eps {
        EquilibriumClass::Annihilation
    } else if d > 1.0 - eps && b < eps && r > eps && r < eps && s > 1.0 - eps {
        EquilibriumClass::Inert
    } else if d < eps && b > eps && rho > eps && r < eps && s < eps {
        EquilibriumClass::BreadthRobustness
    } else if d > eps && b > eps && rho > eps && r > eps && s > eps {
        EquilibriumClass::ProportionalEquilibrium
    } else {
        EquilibriumClass::Unclassified
    }
}

/// 三类原型分类（定理10 — 能量函数判定）。
///
/// 能量函数 V(t) = D² + B² + ρ² + R² + S²
fn classify_archetype(traj: &MemeTrajectory) -> MemeArchetype {
    if traj.trajectory.len() < 10 { return MemeArchetype::Undetermined; }

    let vs: Vec<f64> = traj.trajectory.iter()
        .map(|s| s.d.powi(2) + s.b.powi(2) + s.rho.powi(2) + s.r.powi(2) + s.s.powi(2))
        .collect();

    let v_start = vs[0];
    let v_end = *vs.last().unwrap();
    let v_max = vs.iter().cloned().fold(0.0_f64, f64::max);

    let final_d = traj.trajectory.last().unwrap().d;
    let final_b = traj.trajectory.last().unwrap().b;

    // 基石：V 持续下降，最终 D,B > 0.1
    if v_end < 0.01 && final_d > 0.1 && final_b > 0.1 {
        return MemeArchetype::Stone;
    }

    // 过客：V 有峰值（先升后降），终值 < 初始
    if v_max > v_start * 1.5 && v_end < v_start * 0.3 {
        return MemeArchetype::Transient;
    }

    // 泡沫：V 下降但 D → 0
    if v_end < v_start * 0.1 && final_d < 0.05 {
        return MemeArchetype::Bubble;
    }

    MemeArchetype::Undetermined
}
