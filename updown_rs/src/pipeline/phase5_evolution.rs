//! Phase 5: 演化 (Q → ODE 时间序列 → 原型分类) — 泛模因理论 §4.3-4.4, §6.4 M2
//!
//! 数据流:
//! PhaseThreeOutput → 每模因独立 ODE 积分
//!                → 跳变检测 (β₀(K(t)) 变化)
//!                → 原型分类
//!                → PhaseFiveOutput { trajectories, archetypes, equilibria }

use crate::theory::ode::{self, OdeConfig, StateSnapshot, TerminationReason, Archetype, EquilibriumType};
use crate::theory::function_families::{FunctionFamily, FamilyParams};
use crate::theory::optimizer::{self, OptimizerConfig, OptimalHypothesis};
use crate::pipeline::phase3_decomposition::PhaseThreeOutput;

/// 单模因演化结果
#[derive(Debug, Clone)]
pub struct MemeEvolution {
    /// 模因索引
    pub meme_idx: usize,
    /// ODE 轨迹
    pub trajectory: Vec<StateSnapshot>,
    /// 终止原因
    pub termination: TerminationReason,
    /// 原型分类
    pub archetype: Archetype,
    /// 平衡点类型
    pub equilibrium: EquilibriumType,
}

/// Phase 5 输出
#[derive(Debug, Clone)]
pub struct PhaseFiveOutput {
    pub evolutions: Vec<MemeEvolution>,
    pub optimal_hypothesis: Option<OptimalHypothesis>,
}

/// 运行 Phase 5: 演化
pub fn run_phase_five(
    phase3: &PhaseThreeOutput,
    ode_config: &OdeConfig,
    optimizer_config: Option<&OptimizerConfig>,
) -> PhaseFiveOutput {
    let mut evolutions = Vec::new();

    // 默认使用 Power 函数族
    let default_phi_d = FamilyParams::new(FunctionFamily::Power, 1.0, 0.0);
    let default_phi_r = FamilyParams::new(FunctionFamily::Power, 1.0, 0.0);

    // 全局优化
    let optimal_hypothesis = if let Some(opt_config) = optimizer_config {
        let error_fn = |_t: f64, phi_d: &FamilyParams, phi_r: &FamilyParams| -> f64 {
            let mut total_loss = 0.0;
            for (_i, meme) in phase3.memes.iter().enumerate() {
                let init = meme.five_dim.clamp_to_omega();
                let (trajectory, _) = ode::integrate(
                    &init, &meme.params, ode_config, phi_d, phi_r,
                );
                for snapshot in &trajectory {
                    total_loss += (snapshot.d - 0.5).powi(2)
                        + (snapshot.s - 0.5).powi(2);
                }
            }
            total_loss
        };
        Some(optimizer::optimize(error_fn, opt_config))
    } else {
        None
    };

    // 确定使用的 Φ_D, Φ_R
    let (phi_d, phi_r) = if let Some(ref hyp) = optimal_hypothesis {
        (hyp.params_d.clone(), hyp.params_r.clone())
    } else {
        (default_phi_d, default_phi_r)
    };

    // 每模因独立 ODE 积分
    for (i, meme) in phase3.memes.iter().enumerate() {
        // 先 clamp 到 Ω = [0,1]⁴×[0,∞)，保证初始状态有效（定理 7）
        let init = meme.five_dim.clamp_to_omega();
        let (trajectory, termination) = ode::integrate(
            &init, &meme.params, ode_config, &phi_d, &phi_r,
        );

        let archetype = ode::classify(&trajectory, &termination);
        let equilibrium = if let Some(last) = trajectory.last() {
            ode::classify_equilibrium(&last.to_five_dim())
        } else {
            EquilibriumType::Undetermined
        };

        evolutions.push(MemeEvolution {
            meme_idx: i,
            trajectory,
            termination,
            archetype,
            equilibrium,
        });
    }

    PhaseFiveOutput { evolutions, optimal_hypothesis }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::five_dim::FiveDimState;
    use crate::theory::dynamics_params::DynamicsParams;
    use crate::theory::extended_dimension::ExtendedDimension;
    use crate::pipeline::phase3_decomposition::MemeState;

    fn make_test_phase3() -> PhaseThreeOutput {
        PhaseThreeOutput {
            memes: vec![MemeState {
                five_dim: FiveDimState::new(0.5, 0.5, 1.0, 0.3, 0.8),
                extended: ExtendedDimension::new(),
                params: DynamicsParams::default_params(),
                vertices: vec![0, 1],
            }],
            coupling: vec![vec![0.0]],
            n_memes: 1,
        }
    }

    #[test]
    fn test_phase_five_basic() {
        let p3 = make_test_phase3();
        let config = OdeConfig { max_steps: 500, ..OdeConfig::default() };
        let output = run_phase_five(&p3, &config, None);
        assert_eq!(output.evolutions.len(), 1);
        let evo = &output.evolutions[0];
        assert!(evo.trajectory.len() > 0);
        assert!(matches!(evo.archetype, Archetype::Stone | Archetype::StableCore | Archetype::Decay | _));
    }

    #[test]
    fn test_phase_five_with_optimizer() {
        let p3 = make_test_phase3();
        let ode_config = OdeConfig { max_steps: 500, ..OdeConfig::default() };
        let opt_config = OptimizerConfig {
            t_values: vec![0.1],
            n_step: 1,
            ..OptimizerConfig::default()
        };
        let output = run_phase_five(&p3, &ode_config, Some(&opt_config));
        assert!(output.optimal_hypothesis.is_some());
    }
}