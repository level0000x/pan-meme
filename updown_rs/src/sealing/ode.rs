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
    /// Φ_D 函数族: 0=power, 1=exp, 2=sigmoid, 3=log, 4=piecewise (§4.3.5 R方程)
    pub function_family: usize,
    /// Φ_D 参数 (k for power/exp/log, (k, x0) for sigmoid)
    pub fn_params: Vec<f64>,
    /// Φ_R 函数族: 0=power, 1=exp, 2=sigmoid, 3=log, 4=piecewise (§4.3.6 S方程)
    pub function_family_r: usize,
    /// Φ_R 参数
    pub fn_params_r: Vec<f64>,
    /// 外部输入 I_ext(t) (§4.3.4 ρ 方程) — 常数或可扩展为函数
    pub i_ext: f64,
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
            t_max: 1000.0,
            function_family: 0,  // Φ_D: Power
            fn_params: vec![1.0],
            function_family_r: 0, // Φ_R: Power
            fn_params_r: vec![1.0],
            i_ext: 0.0,           // 默认零外部输入
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

/// 九类演化原型 — 基于五维轨迹的细粒度分类
///
/// 三类主原型（论文 §4.4）细化为九种子类型（README §4.4.1-§4.4.9）：
///   基石族 → Stone, StableCore, Resilient
///   过客族 → Burst, Transient, Decay
///   泡沫族 → Oscillatory, Source, Sink
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemeArchetype {
    /// 基石型：D↑ B↑ ρ↑ R→ S↑ — 深度稳固的核心结构
    Stone,
    /// 稳定核：D→ B→ ρ→ R→ S→ — 长期稳态平衡
    StableCore,
    /// 爆发型：D↓ B↑↑ ρ↑↑ R↑↑ S↓ — 快速扩张后收缩
    Burst,
    /// 衰减型：D↓ B↓ ρ↓ R→ S↓ — 渐近消亡的结构
    Decay,
    /// 振荡型：D↻ B↻ ρ↻ R↻ S↻ — 周期性波动
    Oscillatory,
    /// 瞬变型：D↑↓ B↑↓ ρ↑↓ R↑↓ S↑↓ — 短暂出现后消失
    Transient,
    /// 韧态： D→ B↓ ρ↓ R→ S↑↑ — 低活动高稳定
    Resilient,
    /// 源型： D→ B↑ ρ↑↑ R↑ S→ — 信息-结构净产出者
    Source,
    /// 汇型： D→ B↓ ρ↑ R↓ S→ — 信息-结构净吸收者
    Sink,
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
// 五维 RHS 函数 — 严格对齐论文 §4.3.5-§4.3.6
// ═══════════════════════════════════════════════════════════════════════
//
//  dD/dt = -α₁·R·D + α₂·S·(1-D)                     (§4.3.5 D)
//  dB/dt =  β₁·R·(1-B) - β₂·D·B                      (§4.3.5 B)
//  dρ/dt = -γ₁·R·ρ + γ₂·(1-ρ)·I_ext                   (§4.3.5 ρ)
//  dR/dt =  δ₁·ρ·B·(1-R) - δ₂·Φ_D(D)·R - δ₃·R       (§4.3.5 R)
//  dS/dt =  ε₁·D·(1-S) - ε₂·Φ_R(R)·S                 (§4.3.6 S)
//
//  Φ_D, Φ_R ∈ F 独立选择，I_ext ∈ ℝ，Ω 裁剪保证不变性（定理7）

/// 5D ODE 右侧函数。
///
/// Ω 裁剪（定理7）：每步后裁剪到 [0,1] × [0,1] × ℝ⁺ × [0,1] × [0,1] 保证不变性。
pub fn five_dim_rhs<'a>(
    params: &'a DynamicsParams,
    ff_d: FunctionFamily, fp_d: &'a [f64],
    ff_r: FunctionFamily, fp_r: &'a [f64],
    i_ext: f64,
) -> impl Fn(f64, &[f64; 5]) -> [f64; 5] + 'a
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
        let d   = y[0].max(0.0).min(1.0);
        let b   = y[1].max(0.0).min(1.0);
        let rho = y[2].max(0.0);
        let r   = y[3].max(0.0).min(1.0);
        let s   = y[4].max(0.0).min(1.0);

        let phi_d = ff_d.evaluate(d, fp_d);
        let phi_r = ff_r.evaluate(r, fp_r);

        [
            -a1 * r * d + a2 * s * (1.0 - d),                    // dD/dt
             b1 * r * (1.0 - b) - b2 * d * b,                    // dB/dt
            -g1 * r * rho + g2 * (1.0 - rho) * i_ext,            // dρ/dt
             d1 * rho * b * (1.0 - r) - d2 * phi_d * r - d3 * r, // dR/dt
             e1 * d * (1.0 - s) - e2 * phi_r * s,                // dS/dt
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
    let ff_d = FunctionFamily::from_usize(config.function_family);
    let ff_r = FunctionFamily::from_usize(config.function_family_r);
    let rhs = five_dim_rhs(params, ff_d, &config.fn_params, ff_r, &config.fn_params_r, config.i_ext);

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

/// 九类原型分类（定理10 — 三类主原型 + 细粒度子类型）
///
/// 第一步：按论文 §4.4 判定三类主原型（基石/过客/泡沫）
/// 第二步：在每个主原型内按五维趋势细分为子类型
///
/// 趋势判定：比较轨迹前半段 vs 后半段各维度的均值
fn classify_archetype(traj: &MemeTrajectory) -> MemeArchetype {
    let n = traj.trajectory.len();
    if n < 10 { return MemeArchetype::Undetermined; }

    let mid = n / 2;
    let first_half = &traj.trajectory[..mid];
    let second_half = &traj.trajectory[mid..];

    let avg = |half: &[FiveDimSnapshot]| -> [f64; 5] {
        let len = half.len() as f64;
        [
            half.iter().map(|s| s.d).sum::<f64>() / len,
            half.iter().map(|s| s.b).sum::<f64>() / len,
            half.iter().map(|s| s.rho).sum::<f64>() / len,
            half.iter().map(|s| s.r).sum::<f64>() / len,
            half.iter().map(|s| s.s).sum::<f64>() / len,
        ]
    };

    let a1 = avg(first_half);
    let a2 = avg(second_half);

    // 趋势方向：+1=↑, -1=↓, 0=→（阈值 0.03 防止微小波动误判）
    let trend = |v1: f64, v2: f64| -> i8 {
        if v2 - v1 > 0.03 { 1 } else if v1 - v2 > 0.03 { -1 } else { 0 }
    };

    let td = trend(a1[0], a2[0]);
    let tb = trend(a1[1], a2[1]);
    let trho = trend(a1[2], a2[2]);
    let tr = trend(a1[3], a2[3]);
    let ts = trend(a1[4], a2[4]);

    let final_state = traj.trajectory.last().unwrap();
    let final_d = final_state.d;
    let final_b = final_state.b;
    let final_rho = final_state.rho;
    let final_s = final_state.s;

    // 检测振荡：标准差 / 均值 > 0.2
    let osc = |vals: &[f64]| -> bool {
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        if mean < 0.01 { return false; }
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        (var.sqrt() / mean) > 0.2
    };

    let dims: Vec<Vec<f64>> = vec![
        traj.trajectory.iter().map(|s| s.d).collect(),
        traj.trajectory.iter().map(|s| s.b).collect(),
        traj.trajectory.iter().map(|s| s.rho).collect(),
        traj.trajectory.iter().map(|s| s.r).collect(),
        traj.trajectory.iter().map(|s| s.s).collect(),
    ];
    let osc_count = dims.iter().filter(|v| osc(v)).count();

    // ── 基石族：终态 D > 0.1, B > 0.1, S > 0.2 ──
    if final_d > 0.1 && final_b > 0.1 && final_s > 0.2 {
        if td >= 0 && tb >= 0 && ts >= 0 && tr == 0 && osc_count <= 1 {
            return MemeArchetype::StableCore;        // 全维度稳态
        }
        if td == 1 && ts == 1 && tr == 0 {
            return MemeArchetype::Stone;              // D,S 攀升 + R 平稳
        }
        if tb <= 0 && trho <= 0 && ts == 1 {
            return MemeArchetype::Resilient;           // 低关联低能流 + S 攀升
        }
        return MemeArchetype::Stone;                  // 基石族兜底
    }

    // ── 过客族：终态 D < 0.1 且 S < 0.2 ──
    if final_d < 0.1 && final_s < 0.2 {
        if tb == 1 && trho == 1 && tr == 1 && ts == -1 {
            return MemeArchetype::Burst;               // B,ρ,R 冲高 + S 崩溃
        }
        if td == -1 && tb == -1 && trho == -1 {
            return MemeArchetype::Decay;               // 全面衰减
        }
        if osc_count >= 3 {
            return MemeArchetype::Oscillatory;         // 多维度振荡
        }
        return MemeArchetype::Transient;               // 过客族兜底
    }

    // ── 泡沫族 / 中间态 ──
    if final_rho > 0.3 && final_d < 0.15 {
        if tb == 1 && trho == 1 && tr == 1 {
            return MemeArchetype::Source;              // B,ρ 净产出 + R 高
        }
        if tb == -1 && trho == 1 && tr == -1 {
            return MemeArchetype::Sink;                // 关联萎缩 + 能流累积 + 慢演化
        }
    }

    // 振荡检测（跨族）
    if osc_count >= 3 {
        return MemeArchetype::Oscillatory;
    }

    // 终态判定兜底
    if final_d < 0.05 && final_s < 0.1 {
        return MemeArchetype::Transient;
    }
    if final_d > 0.1 && final_s > 0.3 {
        return MemeArchetype::Stone;
    }

    MemeArchetype::Undetermined
}

// ═══════════════════════════════════════════════════════════════════════
// 测试
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::decomposition::DynamicsParams;

    /// 构造一个所有参数均为 0.1 的 DynamicsParams，用于快速测试。
    fn test_params() -> DynamicsParams {
        DynamicsParams {
            alpha_1: 0.1, alpha_2: 0.1,
            beta_1: 0.1,  beta_2: 0.1,
            gamma_1: 0.1, gamma_2: 0.1,
            delta_1: 0.1, delta_2: 0.1, delta_3: 0.1,
            epsilon_1: 0.1, epsilon_2: 0.1,
        }
    }

    /// 小步数、短时间的快速配置，避免测试耗时过长。
    fn fast_config() -> OdeConfig {
        OdeConfig {
            max_steps: 1000,
            t_max: 10.0,
            ..Default::default()
        }
    }

    fn half_initial() -> FiveDimSnapshot {
        FiveDimSnapshot { t: 0.0, d: 0.5, b: 0.5, rho: 0.5, r: 0.5, s: 0.5 }
    }

    // ── 1. FunctionFamily::evaluate ──────────────────────────────────

    #[test]
    fn test_function_family_evaluate() {
        // Power: f(x)=x^k, k=2.0, x=0.5 → 0.25
        let power_val = FunctionFamily::Power.evaluate(0.5, &[2.0]);
        assert!((power_val - 0.25).abs() < 1e-10,
            "Power(2.0) at x=0.5 expected 0.25, got {}", power_val);

        // Exp: f(x)=exp(k*x)-1, k=1.0, x=0.5 → e^0.5 - 1 ≈ 0.64872
        let exp_val = FunctionFamily::Exp.evaluate(0.5, &[1.0]);
        assert!((exp_val - 0.6487212707).abs() < 1e-6,
            "Exp(1.0) at x=0.5 expected ~0.6487, got {}", exp_val);

        // Sigmoid: centered at x0, f(x)=σ(k(x-x0))-0.5.
        // k=2.0, x0=0.5, x=0.5 → σ(0)-0.5 = 0.0
        let sig_val = FunctionFamily::Sigmoid.evaluate(0.5, &[2.0, 0.5]);
        assert!(sig_val.abs() < 1e-10,
            "Sigmoid(2.0,0.5) at x=0.5 expected 0.0, got {}", sig_val);

        // Log: f(x)=ln(1+k*x), k=1.0, x=0.5 → ln(1.5) ≈ 0.4055
        let log_val = FunctionFamily::Log.evaluate(0.5, &[1.0]);
        assert!((log_val - 0.405465108).abs() < 1e-4,
            "Log(1.0) at x=0.5 expected ~0.4055, got {}", log_val);

        // Piecewise: x=0.5 落在 [0.3, 0.7] 中段
        // → 0.3 + (0.5-0.3)/(0.7-0.3)*0.6 = 0.6
        let pw_val = FunctionFamily::Piecewise.evaluate(0.5, &[1.0, 0.3, 0.7]);
        assert!((pw_val - 0.6).abs() < 1e-10,
            "Piecewise at x=0.5 expected 0.6, got {}", pw_val);
    }

    #[test]
    fn test_function_family_all() {
        assert_eq!(FunctionFamily::ALL.len(), 5);
        // 确保 5 个枚举值互不相同
        let all = &FunctionFamily::ALL;
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j]);
            }
        }
    }

    // ── 2. OdeConfig::default ───────────────────────────────────────

    #[test]
    fn test_ode_config_default() {
        let cfg = OdeConfig::default();
        assert!((cfg.h0 - 0.01).abs() < 1e-10, "h0 should be 0.01");
        assert!((cfg.h_min - 1e-8).abs() < 1e-10, "h_min should be 1e-8");
        assert!((cfg.h_max - 0.5).abs() < 1e-10, "h_max should be 0.5");
        assert!((cfg.rtol - 1e-6).abs() < 1e-10, "rtol should be 1e-6");
        assert!((cfg.atol - 1e-8).abs() < 1e-10, "atol should be 1e-8");
        assert_eq!(cfg.max_steps, 100_000, "max_steps should be 100000");
        assert!((cfg.t_max - 1000.0).abs() < 1e-10, "t_max should be 1000.0");
        assert_eq!(cfg.function_family, 0, "default Φ_D family should be 0 (Power)");
        assert_eq!(cfg.fn_params, vec![1.0], "default Φ_D params should be [1.0]");
        assert_eq!(cfg.function_family_r, 0, "default Φ_R family should be 0 (Power)");
        assert_eq!(cfg.fn_params_r, vec![1.0], "default Φ_R params should be [1.0]");
        assert!((cfg.i_ext - 0.0).abs() < 1e-10, "default i_ext should be 0.0");
    }

    // ── 3. five_dim_rhs ─────────────────────────────────────────────

    #[test]
    fn test_five_dim_rhs() {
        let params = test_params();
        let ff_d = FunctionFamily::Power;
        let fp_d = vec![1.5];
        let ff_r = FunctionFamily::Exp;
        let fp_r = vec![1.0];
        let i_ext = 0.0;
        let rhs = five_dim_rhs(&params, ff_d, &fp_d, ff_r, &fp_r, i_ext);
        let state = [0.5; 5];
        let derivatives = rhs(0.0, &state);
        assert_eq!(derivatives.len(), 5);
        for (i, d) in derivatives.iter().enumerate() {
            assert!(d.is_finite(),
                "derivative[{}] should be finite, got {}", i, d);
        }
    }

    // ── 4. solve_single_meme ────────────────────────────────────────

    #[test]
    fn test_solve_single_meme() {
        let params = test_params();
        let config = OdeConfig {
            max_steps: 1000,
            t_max: 10.0,
            function_family: 1, // Exp
            fn_params: vec![1.0],
            ..Default::default()
        };
        let initial = half_initial();
        let traj = solve_single_meme(0, &initial, &params, &config);

        assert!(!traj.trajectory.is_empty(),
            "trajectory should contain at least the initial snapshot");

        // 终止原因应是 Converged / MaxSteps / MaxTime 之一
        assert!(
            traj.termination_reason == TerminationReason::Converged
                || traj.termination_reason == TerminationReason::MaxSteps
                || traj.termination_reason == TerminationReason::MaxTime,
            "unexpected termination: {:?}", traj.termination_reason
        );

        assert_eq!(traj.meme_id, 0);
        assert!(traj.terminated_at > 0.0, "terminated_at should be positive");
    }

    // ── 5. classify_equilibrium ────────────────────────────────────

    #[test]
    fn test_classify_equilibrium() {
        let params = test_params();
        let config = fast_config();
        let initial = half_initial();
        let traj = solve_single_meme(0, &initial, &params, &config);

        // 必须有足够快照用于分类
        assert!(traj.trajectory.len() >= 2,
            "need at least 2 snapshots for classification");

        let eq = super::classify_equilibrium(&traj);
        assert!(matches!(eq,
            EquilibriumClass::DegenerateZero
            | EquilibriumClass::Annihilation
            | EquilibriumClass::Inert
            | EquilibriumClass::BreadthRobustness
            | EquilibriumClass::ProportionalEquilibrium
            | EquilibriumClass::Unclassified
        ), "unexpected equilibrium class: {:?}", eq);
    }

    // ── 6. classify_archetype ───────────────────────────────────────

    #[test]
    fn test_classify_archetype() {
        let params = test_params();
        let config = fast_config();
        let initial = half_initial();
        let traj = solve_single_meme(0, &initial, &params, &config);

        let arch = super::classify_archetype(&traj);
        assert!(matches!(arch,
            MemeArchetype::Stone | MemeArchetype::StableCore
            | MemeArchetype::Burst | MemeArchetype::Decay
            | MemeArchetype::Oscillatory | MemeArchetype::Transient
            | MemeArchetype::Resilient | MemeArchetype::Source
            | MemeArchetype::Sink | MemeArchetype::Undetermined
        ), "unexpected archetype: {:?}", arch);
    }

    // ── 7. solve_all_memes ──────────────────────────────────────────

    #[test]
    fn test_solve_all_memes() {
        let params = vec![test_params(), test_params()];
        let couplings: Vec<crate::encoding::decomposition::Coupling> = Vec::new();
        let config = OdeConfig {
            max_steps: 500,
            t_max: 10.0,
            function_family: 0,
            fn_params: vec![1.5],
            ..Default::default()
        };

        let initials = vec![
            (0, half_initial()),
            (1, FiveDimSnapshot { t: 0.0, d: 0.3, b: 0.7, rho: 0.2, r: 0.4, s: 0.6 }),
        ];

        let output = solve_all_memes(&initials, &params, &couplings, &config);

        assert_eq!(output.trajectories.len(), 2,
            "should have 2 trajectories");
        assert_eq!(output.archetypes.len(), 2,
            "should have 2 archetype classifications");

        for (i, (_id, arch, eq)) in output.archetypes.iter().enumerate() {
            assert!(matches!(arch,
                MemeArchetype::Stone | MemeArchetype::StableCore
                | MemeArchetype::Burst | MemeArchetype::Decay
                | MemeArchetype::Oscillatory | MemeArchetype::Transient
                | MemeArchetype::Resilient | MemeArchetype::Source
                | MemeArchetype::Sink | MemeArchetype::Undetermined
            ), "meme {} unexpected archetype: {:?}", i, arch);
            assert!(matches!(eq,
                EquilibriumClass::DegenerateZero | EquilibriumClass::Annihilation
                | EquilibriumClass::Inert | EquilibriumClass::BreadthRobustness
                | EquilibriumClass::ProportionalEquilibrium | EquilibriumClass::Unclassified
            ), "meme {} unexpected equilibrium: {:?}", i, eq);
        }
    }

    // ── 8. h-不变性：Ω=[0,1]⁵ 不变集检验（定理7）────────────────

    #[test]
    fn test_h_invariance() {
        let params = test_params();
        // 多组不同的函数族和初始状态，确保不变性在各种条件下成立
        let families: Vec<(usize, Vec<f64>)> = vec![
            (0, vec![1.5]),          // Power  k=1.5
            (1, vec![1.0]),          // Exp    k=1.0
        ];
        let initials: Vec<FiveDimSnapshot> = vec![
            FiveDimSnapshot { t: 0.0, d: 0.5, b: 0.5, rho: 0.5, r: 0.5, s: 0.5 },
            FiveDimSnapshot { t: 0.0, d: 0.1, b: 0.9, rho: 0.2, r: 0.8, s: 0.3 },
        ];

        for (ff_idx, fnp) in &families {
            let config = OdeConfig {
                max_steps: 1000,
                t_max: 10.0,
                function_family: *ff_idx,
                fn_params: fnp.clone(),
                function_family_r: *ff_idx,
                fn_params_r: fnp.clone(),
                i_ext: 0.0,
                ..Default::default()
            };
            for init in &initials {
                let traj = solve_single_meme(99, init, &params, &config);
                for (step, snap) in traj.trajectory.iter().enumerate() {
                    assert!(snap.d >= 0.0 && snap.d <= 1.0,
                        "family={} d={} at step {} out of [0,1]", ff_idx, snap.d, step);
                    assert!(snap.b >= 0.0 && snap.b <= 1.0,
                        "family={} b={} at step {} out of [0,1]", ff_idx, snap.b, step);
                    assert!(snap.rho >= 0.0,
                        "family={} rho={} at step {} < 0", ff_idx, snap.rho, step);
                    assert!(snap.r >= 0.0 && snap.r <= 1.0,
                        "family={} r={} at step {} out of [0,1]", ff_idx, snap.r, step);
                    assert!(snap.s >= 0.0 && snap.s <= 1.0,
                        "family={} s={} at step {} out of [0,1]", ff_idx, snap.s, step);
                }
            }
        }
    }
}
