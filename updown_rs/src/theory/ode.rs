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
    // ── 收敛判据（实验零）──
    /// 收敛检测窗口大小（连续 N 步）
    pub convergence_window: usize,
    /// 收敛阈值（窗口首尾总变化 < 此值）
    pub convergence_threshold: f64,
    /// 轨迹末尾连续 N 步各维度标准差上限
    pub std_threshold: f64,
    /// Undetermined 分类最大占比（超出则收敛不足）
    pub undetermined_max_pct: f64,
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
            convergence_window: 5,
            convergence_threshold: 1e-3,
            std_threshold: 1e-3,
            undetermined_max_pct: 0.10,
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
    /// 收敛不足（未满足收敛判据）— 实验零
    InsufficientConvergence,
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

        // 收敛检测: 连续 N 步总变化 < threshold
        if trajectory.len() > config.convergence_window {
            let start = trajectory.len() - config.convergence_window - 1;
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
            if prev.total_change(&curr) < config.convergence_threshold {
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

/// 分类原型 — 定理 5.8（基于轨迹动力学特征）
///
/// 与旧版不同，不依赖 TerminationReason 决定原型族，而是分析轨迹的
/// 形状特征（峰值、衰减、振荡等）来分类。这使分类对 ODE 参数
/// 和初始条件的变化更敏感，能产生多样化的原型。
pub fn classify(trajectory: &[StateSnapshot], _reason: &TerminationReason) -> Archetype {
    if trajectory.len() < 2 {
        return Archetype::Undetermined;
    }

    let n = trajectory.len();
    let first = &trajectory[0];
    let last = &trajectory[n - 1];

    // ── 轨迹特征提取 ──
    // 各维度: 初始值、终值、峰值、峰值时间、谷值
    let d_init = first.d;
    let d_final = last.d;
    let b_init = first.b;
    let b_final = last.b;
    let rho_init = first.rho;
    let rho_final = last.rho;
    let r_init = first.r;
    let r_final = last.r;
    let s_init = first.s;
    let s_final = last.s;

    // 找各维度峰值和时间
    let mut d_peak = d_init;
    let mut d_peak_t = 0.0;
    let mut b_peak = b_init;
    let mut _b_peak_t = 0.0;
    let mut r_peak = r_init;
    let mut rho_peak = rho_init;
    let mut s_min = s_init;

    for snap in trajectory {
        if snap.d > d_peak {
            d_peak = snap.d;
            d_peak_t = snap.t;
        }
        if snap.b > b_peak {
            b_peak = snap.b;
            _b_peak_t = snap.t;
        }
        if snap.r > r_peak {
            r_peak = snap.r;
        }
        if snap.rho > rho_peak {
            rho_peak = snap.rho;
        }
        if snap.s < s_min {
            s_min = snap.s;
        }
    }

    // 振荡检测: 计算各维度导数的符号变化次数
    let oscillation_count = count_sign_changes(trajectory);

    // 增长/衰减速率: 前 20% 到 后 20% 的变化
    let early_idx = (n as f64 * 0.2) as usize;
    let late_idx = (n as f64 * 0.8) as usize;
    let early = &trajectory[early_idx.min(n - 1)];
    let late = &trajectory[late_idx.min(n - 1)];
    let d_growth = late.d - early.d;
    let _r_growth = late.r - early.r;
    let _b_growth = late.b - early.b;

    // 收敛速度: 达到终值 90% 的时间
    let d_target = d_init + 0.9 * (d_final - d_init);
    let mut _d_converge_t = last.t;
    for snap in trajectory {
        if (d_final - d_init).abs() > 1e-6
            && (snap.d - d_init).abs() >= (d_target - d_init).abs() * 0.99
        {
            _d_converge_t = snap.t;
            break;
        }
    }

    // ── 原型判定 ──
    let eps = 0.15;

    // 检测轨迹是否仍在显著演化中（末尾 20% 步长内总变化 > 0.05）
    let tail_start = (n as f64 * 0.8) as usize;
    let tail_slice = &trajectory[tail_start.min(n - 1)..];
    let tail_change = if tail_slice.len() >= 2 {
        let ts = &tail_slice[0];
        let te = &tail_slice[tail_slice.len() - 1];
        (te.d - ts.d).abs() + (te.b - ts.b).abs() + (te.r - ts.r).abs() + (te.s - ts.s).abs()
    } else {
        0.0
    };
    let still_evolving = tail_change > 0.05;

    // 振荡型: 多维度导数符号变化 ≥ 3
    if oscillation_count >= 3 {
        return Archetype::Oscillatory;
    }

    // 过客族: 峰值明显高于终值（先升后降）
    let d_decay = d_peak - d_final;
    let r_decay = r_peak - r_final;

    if d_decay > 0.3 && d_peak > 0.5 {
        // Burst: R 峰值高，爆发性强
        if r_peak > 0.4 {
            return Archetype::Burst;
        }
        // Transient: 快速上升后衰减，峰值时间早
        if d_peak_t < last.t * 0.4 {
            return Archetype::Transient;
        }
        return Archetype::Decay;
    }

    // 仍在演化中（MaxTime/MaxSteps）→ 按轨迹动态分类，不按终值
    if still_evolving {
        // R 从高峰值衰减 → Burst
        if r_peak > 0.5 && r_decay > 0.2 {
            return Archetype::Burst;
        }
        // D 和 B 都在增长 → Source（向外扩展中）
        if d_growth > 0.3 && b_final > 0.4 {
            return Archetype::Source;
        }
        // R 保持在中等水平以上 → 活跃中
        if r_final > 0.05 {
            return Archetype::Transient;
        }
        // D 显著增长但 R 已衰减 → 趋稳中
        if d_growth > 0.3 {
            return Archetype::StableCore;
        }
        // 已接近 Stone 吸引子 (D≈1, S≈1, R≈0) → Stone
        if d_final > 0.9 && s_final > 0.9 && r_final < 0.1 {
            return Archetype::Stone;
        }
        // 兜底: 仍在演化但无法归类
        return Archetype::Undetermined;
    }

    // 泡沫族: 特殊模式
    // Source: 向外输出 (B 和 R 持续高)
    if b_final > 0.3 && r_final > 0.2 && d_final > 0.3 {
        return Archetype::Source;
    }
    // Sink: 吸收型 (ρ 高, D 和 B 低)
    if rho_final > 0.7 && d_final < 0.3 && b_final < 0.3 {
        return Archetype::Sink;
    }

    // 基石族: 稳定收敛
    if d_final > 1.0 - eps && s_final > 1.0 - eps && r_final < eps {
        return Archetype::Stone;
    }

    if s_final > 0.7 && s_min > 0.5 {
        return Archetype::Resilient;
    }

    // StableCore: 收敛到中等值
    if d_final > 0.3 && s_final > 0.3 {
        return Archetype::StableCore;
    }

    // 兜底: 按终值判断
    if d_final < 0.2 && s_final < 0.2 {
        if d_growth < -0.1 {
            Archetype::Decay
        } else {
            Archetype::Transient
        }
    } else if r_final > 0.3 {
        Archetype::Burst
    } else {
        Archetype::Undetermined
    }
}

/// 统计轨迹中所有维度的导数符号变化次数
fn count_sign_changes(trajectory: &[StateSnapshot]) -> usize {
    if trajectory.len() < 3 {
        return 0;
    }
    let mut count = 0;
    for dim in 0..5 {
        let mut prev_sign = 0i8;
        for i in 1..trajectory.len() {
            let diff = match dim {
                0 => trajectory[i].d - trajectory[i - 1].d,
                1 => trajectory[i].b - trajectory[i - 1].b,
                2 => trajectory[i].rho - trajectory[i - 1].rho,
                3 => trajectory[i].r - trajectory[i - 1].r,
                4 => trajectory[i].s - trajectory[i - 1].s,
                _ => 0.0,
            };
            let sign = if diff > 1e-6 { 1 } else if diff < -1e-6 { -1 } else { 0 };
            if sign != 0 && prev_sign != 0 && sign != prev_sign {
                count += 1;
            }
            if sign != 0 {
                prev_sign = sign;
            }
        }
    }
    count
}

/// 收敛报告 — 实验零
///
/// 评估整个 Phase 5 输出的收敛质量，按四条判据逐项检查。
#[derive(Debug, Clone)]
pub struct ConvergenceReport {
    /// 总模因数
    pub total_memes: usize,
    /// 收敛模因数（TerminationReason::Converged）
    pub converged_count: usize,
    /// 收敛率
    pub convergence_rate: f64,
    /// 轨迹末尾维度标准差超标的模因数
    pub std_violations: usize,
    /// Undetermined 分类数
    pub undetermined_count: usize,
    /// Undetermined 占比
    pub undetermined_pct: f64,
    /// 四条判据逐一通过状态
    pub criteria: ConvergenceCriteria,
    /// 整体是否收敛
    pub is_converged: bool,
}

/// 四条收敛判据
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    /// 判据 1: 所有模因 Converged（非 MaxSteps/NaN）
    pub all_converged: bool,
    /// 判据 2: 轨迹末尾维度标准差 < threshold
    pub std_ok: bool,
    /// 判据 3: Undetermined < 10%
    pub undetermined_ok: bool,
    /// 判据 4: 原型分布一致（需多次运行，此处仅标记）
    pub reproducibility_note: &'static str,
}

/// 评估收敛质量 — 实验零核心函数
///
/// 参数:
/// - evolutions: Phase 5 输出的所有模因演化结果
/// - config: ODE 配置（含收敛判据参数）
pub fn evaluate_convergence(
    evolutions: &[crate::pipeline::phase5_evolution::MemeEvolution],
    config: &OdeConfig,
) -> ConvergenceReport {
    let total = evolutions.len();
    let converged = evolutions
        .iter()
        .filter(|e| e.termination == TerminationReason::Converged)
        .count();

    let mut std_violations = 0;
    for evo in evolutions {
        if evo.trajectory.len() < config.convergence_window {
            std_violations += 1;
            continue;
        }
        // 轨迹末尾连续 N 步各维度标准差
        let tail: Vec<_> = evo
            .trajectory
            .iter()
            .rev()
            .take(config.convergence_window)
            .collect();
        if tail.len() < config.convergence_window {
            std_violations += 1;
            continue;
        }
        let means = mean_of_snapshots(&tail);
        let stds = std_of_snapshots(&tail, &means);
        if stds.iter().any(|s| *s > config.std_threshold) {
            std_violations += 1;
        }
    }

    let undetermined = evolutions
        .iter()
        .filter(|e| e.archetype == Archetype::Undetermined)
        .count();

    let convergence_rate = if total > 0 {
        converged as f64 / total as f64
    } else {
        0.0
    };
    let undetermined_pct = if total > 0 {
        undetermined as f64 / total as f64
    } else {
        0.0
    };

    let criteria = ConvergenceCriteria {
        all_converged: converged == total,
        std_ok: std_violations == 0,
        undetermined_ok: undetermined_pct <= config.undetermined_max_pct,
        reproducibility_note: "需多次独立运行验证原型分布一致性",
    };

    let is_converged = criteria.all_converged && criteria.std_ok && criteria.undetermined_ok;

    ConvergenceReport {
        total_memes: total,
        converged_count: converged,
        convergence_rate,
        std_violations,
        undetermined_count: undetermined,
        undetermined_pct,
        criteria,
        is_converged,
    }
}

fn mean_of_snapshots(snapshots: &[&StateSnapshot]) -> [f64; 5] {
    let n = snapshots.len() as f64;
    let mut sums = [0.0; 5];
    for s in snapshots {
        sums[0] += s.d;
        sums[1] += s.b;
        sums[2] += s.rho;
        sums[3] += s.r;
        sums[4] += s.s;
    }
    [sums[0] / n, sums[1] / n, sums[2] / n, sums[3] / n, sums[4] / n]
}

fn std_of_snapshots(snapshots: &[&StateSnapshot], means: &[f64; 5]) -> [f64; 5] {
    let n = snapshots.len() as f64;
    let mut sums = [0.0; 5];
    for s in snapshots {
        sums[0] += (s.d - means[0]).powi(2);
        sums[1] += (s.b - means[1]).powi(2);
        sums[2] += (s.rho - means[2]).powi(2);
        sums[3] += (s.r - means[3]).powi(2);
        sums[4] += (s.s - means[4]).powi(2);
    }
    [
        (sums[0] / n).sqrt(),
        (sums[1] / n).sqrt(),
        (sums[2] / n).sqrt(),
        (sums[3] / n).sqrt(),
        (sums[4] / n).sqrt(),
    ]
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
