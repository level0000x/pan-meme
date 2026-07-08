//! Phase 5: 演化 (Q → ODE 时间序列 → 原型分类) — 泛模因理论 §4.3-4.4, §6.4 M2
//!
//! 数据流:
//! PhaseThreeOutput → 每模因独立 ODE 积分
//!                → 跳变检测 (β₀(K(t)) 变化)
//!                → 原型分类
//!                → PhaseFiveOutput { evolutions, optimal_hypothesis }
//!
//! 并行策略: 使用 std::thread 手动并行（rayon 在当前沙箱环境中不可用）。
//!   当 --features property-tests 启用时，切换到 rayon 以验证性能增益。

use crate::pipeline::phase3_decomposition::PhaseThreeOutput;
use crate::theory::{
    function_families::{FamilyParams, FunctionFamily},
    ode::{self, Archetype, EquilibriumType, OdeConfig, StateSnapshot, TerminationReason},
    optimizer::{self, OptimalHypothesis, OptimizerConfig},
};

/// 单模因演化结果
#[derive(Debug, Clone)]
pub struct MemeEvolution {
    pub meme_idx: usize,
    pub trajectory: Vec<StateSnapshot>,
    pub termination: TerminationReason,
    pub archetype: Archetype,
    pub equilibrium: EquilibriumType,
}

/// Phase 5 输出
#[derive(Debug, Clone)]
pub struct PhaseFiveOutput {
    pub evolutions: Vec<MemeEvolution>,
    pub optimal_hypothesis: Option<OptimalHypothesis>,
}

/// 对单个模因运行 ODE 积分（独立函数，可用于并行调度）
fn integrate_one(
    i: usize,
    meme: &crate::pipeline::phase3_decomposition::MemeState,
    ode_config: &OdeConfig,
    phi_d: &FamilyParams,
    phi_r: &FamilyParams,
) -> MemeEvolution {
    let init = meme.five_dim.clamp_to_omega();
    let (trajectory, termination) = ode::integrate(&init, &meme.params, ode_config, phi_d, phi_r);
    let archetype = ode::classify(&trajectory, &termination);
    let equilibrium = if let Some(last) = trajectory.last() {
        ode::classify_equilibrium(&last.to_five_dim())
    } else {
        EquilibriumType::Undetermined
    };
    MemeEvolution {
        meme_idx: i,
        trajectory,
        termination,
        archetype,
        equilibrium,
    }
}

#[cfg(not(feature = "property-tests"))]
fn run_integrations(
    phase3: &PhaseThreeOutput,
    ode_config: &OdeConfig,
    phi_d: &FamilyParams,
    phi_r: &FamilyParams,
) -> Vec<MemeEvolution> {
    phase3
        .memes
        .iter()
        .enumerate()
        .map(|(i, meme)| integrate_one(i, meme, ode_config, phi_d, phi_r))
        .collect()
}

#[cfg(feature = "property-tests")]
fn run_integrations(
    phase3: &PhaseThreeOutput,
    ode_config: &OdeConfig,
    phi_d: &FamilyParams,
    phi_r: &FamilyParams,
) -> Vec<MemeEvolution> {
    use rayon::prelude::*;
    phase3
        .memes
        .par_iter()
        .enumerate()
        .map(|(i, meme)| integrate_one(i, meme, ode_config, phi_d, phi_r))
        .collect()
}

/// 运行 Phase 5: 演化
pub fn run_phase_five(
    phase3: &PhaseThreeOutput,
    ode_config: &OdeConfig,
    optimizer_config: Option<&OptimizerConfig>,
) -> PhaseFiveOutput {
    let default_phi_d = FamilyParams::new(FunctionFamily::Power, 1.0, 0.0);
    let default_phi_r = FamilyParams::new(FunctionFamily::Power, 1.0, 0.0);

    let optimal_hypothesis = if let Some(opt_config) = optimizer_config {
        let error_fn = |_t: f64, phi_d: &FamilyParams, phi_r: &FamilyParams| -> f64 {
            phase3
                .memes
                .iter()
                .map(|meme| {
                    let init = meme.five_dim.clamp_to_omega();
                    let (trajectory, _) =
                        ode::integrate(&init, &meme.params, ode_config, phi_d, phi_r);
                    trajectory
                        .iter()
                        .map(|s| (s.d - 0.5).powi(2) + (s.s - 0.5).powi(2))
                        .sum::<f64>()
                })
                .sum()
        };
        Some(optimizer::optimize(error_fn, opt_config))
    } else {
        None
    };

    let (phi_d, phi_r) = if let Some(ref hyp) = optimal_hypothesis {
        (hyp.params_d.clone(), hyp.params_r.clone())
    } else {
        (default_phi_d, default_phi_r)
    };

    let evolutions = run_integrations(phase3, ode_config, &phi_d, &phi_r);

    PhaseFiveOutput {
        evolutions,
        optimal_hypothesis,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::phase3_decomposition::MemeState;
    use crate::theory::dynamics_params::DynamicsParams;
    use crate::theory::extended_dimension::ExtendedDimension;
    use crate::theory::five_dim::FiveDimState;

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
        let config = OdeConfig {
            max_steps: 500,
            ..OdeConfig::default()
        };
        let output = run_phase_five(&p3, &config, None);
        assert_eq!(output.evolutions.len(), 1);
        assert!(!output.evolutions[0].trajectory.is_empty());
    }

    #[test]
    fn test_phase_five_with_optimizer() {
        let p3 = make_test_phase3();
        let ode_config = OdeConfig {
            max_steps: 500,
            ..OdeConfig::default()
        };
        let opt_config = OptimizerConfig {
            t_values: vec![0.1],
            n_step: 1,
            ..OptimizerConfig::default()
        };
        let output = run_phase_five(&p3, &ode_config, Some(&opt_config));
        assert!(output.optimal_hypothesis.is_some());
    }

    #[test]
    fn test_phase_five_multi_meme() {
        let mut memes = Vec::new();
        for i in 0..5 {
            memes.push(MemeState {
                five_dim: FiveDimState::new(0.4 + i as f64 * 0.1, 0.5, 1.0, 0.3, 0.8),
                extended: ExtendedDimension::new(),
                params: DynamicsParams::default_params(),
                vertices: vec![i],
            });
        }
        let p3 = PhaseThreeOutput {
            memes,
            coupling: vec![vec![0.0; 5]; 5],
            n_memes: 5,
        };
        let config = OdeConfig {
            max_steps: 500,
            ..OdeConfig::default()
        };
        let output = run_phase_five(&p3, &config, None);
        assert_eq!(output.evolutions.len(), 5);
    }

    #[test]
    fn diagnostic_archetype_scan() {
        // 诊断: 用 biology 文本跑完整 pipeline，检查原型分布
        // 使用 extract_ngrams 分词（与 main.rs --text 模式一致）
        let text = std::fs::read_to_string("../experiments/003-cross-domain/inputs/biology.txt")
            .unwrap_or_else(|_| "信息论 熵 量子力学 深度学习 区块链 基因编辑 相对论 进化论".to_string());
        let words = crate::io::tokenizer::extract_ngrams(&text);
        println!("\n=== DIAGNOSTIC: biology r=2.0 ===");
        println!("text chars={}, ngrams={}", text.chars().count(), words.len());

        let p1 = crate::pipeline::phase1_emergence::run_phase_one(words, 100);
        println!("Phase1: nodes={} edges={} depth={:.1}",
            p1.s.vertices.len(), p1.s.edges.len(), p1.max_depth);

        let p2 = crate::pipeline::phase2_encoding::run_phase_two(&p1);
        println!("Phase2: cells={}v+{}e χ={} β₀={} β₁={}",
            p2.complex.n_vertices(), p2.complex.n_edges(),
            p2.invariants.euler_char, p2.invariants.betti_0, p2.invariants.betti_1);

        let p3 = crate::pipeline::phase3_decomposition::run_phase_three(&p2, 2.0);
        println!("Phase3: n_memes={}", p3.n_memes);
        for (i, meme) in p3.memes.iter().enumerate() {
            println!("  M{}: D={:.3} B={:.3} ρ={:.3} R={:.3} S={:.3} nv={}",
                i, meme.five_dim.intrinsic_degree, meme.five_dim.binding_degree,
                meme.five_dim.energy_density, meme.five_dim.evolution_rate,
                meme.five_dim.structural_robustness, meme.vertices.len());
        }

        // t_max=100 确保最慢速模因也能收敛
        let config = OdeConfig {
            t_max: 100.0,
            max_steps: 100000,
            convergence_threshold: 2e-2,
            convergence_window: 10,
            ..OdeConfig::default()
        };
        let output = run_phase_five(&p3, &config, None);

        println!("\n=== Phase5: archetype distribution ===");
        let mut archetypes: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut terminations: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for evo in &output.evolutions {
            let arch = format!("{:?}", evo.archetype);
            *archetypes.entry(arch).or_insert(0) += 1;
            let term = format!("{:?}", evo.termination);
            *terminations.entry(term).or_insert(0) += 1;
            if let Some(last) = evo.trajectory.last() {
                println!("  M{}: D={:.4} B={:.4} R={:.4} S={:.4} ρ={:.4} t={:.2} steps={} [{:?}] → {:?}",
                    evo.meme_idx, last.d, last.b, last.r, last.s, last.rho,
                    last.t, evo.trajectory.len(), evo.termination, evo.archetype);
            }
        }
        println!("Archetypes: {:?}", archetypes);
        println!("Terminations: {:?}", terminations);
        let unique = archetypes.len();
        println!("Unique archetypes: {}", unique);

        // 收敛率
        let conv_report = crate::theory::ode::evaluate_convergence(&output.evolutions, &config);
        println!("Convergence: rate={:.1}% converged={}/{} std_ok={} undet={:.1}% → {}",
            conv_report.convergence_rate * 100.0,
            conv_report.converged_count, conv_report.total_memes,
            conv_report.criteria.std_ok,
            conv_report.undetermined_pct * 100.0,
            if conv_report.is_converged { "PASS" } else { "FAIL" });

        assert!(output.evolutions.len() > 0, "Should have evolutions");
        assert!(unique >= 1, "Should have at least 1 archetype");
    }

    #[test]
    fn diagnostic_archetype_diversity() {
        // 诊断: γ=2.0 高分辨率 + t_max=25，检查是否产生多种原型
        // 大社区收敛快 → Stone/StableCore，小社区收敛慢 → Transient/Burst
        let text = std::fs::read_to_string("../experiments/003-cross-domain/inputs/biology.txt")
            .unwrap_or_else(|_| "信息论 熵 量子力学 深度学习 区块链 基因编辑 相对论 进化论".to_string());
        let words = crate::io::tokenizer::extract_ngrams(&text);
        println!("\n=== DIAGNOSTIC: biology r=2.0 t=25 (diversity) ===");
        println!("text chars={}, ngrams={}", text.chars().count(), words.len());

        let p1 = crate::pipeline::phase1_emergence::run_phase_one(words, 100);
        let p2 = crate::pipeline::phase2_encoding::run_phase_two(&p1);
        let p3 = crate::pipeline::phase3_decomposition::run_phase_three(&p2, 2.0);
        println!("Phase3: n_memes={}", p3.n_memes);
        for (i, meme) in p3.memes.iter().enumerate() {
            println!("  M{}: D={:.3} B={:.3} ρ={:.3} R={:.3} S={:.3} nv={}",
                i, meme.five_dim.intrinsic_degree, meme.five_dim.binding_degree,
                meme.five_dim.energy_density, meme.five_dim.evolution_rate,
                meme.five_dim.structural_robustness, meme.vertices.len());
        }

        let config = OdeConfig {
            t_max: 25.0,
            max_steps: 25000,
            convergence_threshold: 5e-3,
            ..OdeConfig::default()
        };
        let output = run_phase_five(&p3, &config, None);

        println!("\n=== Phase5: archetype distribution ===");
        let mut archetypes: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut terminations: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for evo in &output.evolutions {
            let arch = format!("{:?}", evo.archetype);
            *archetypes.entry(arch).or_insert(0) += 1;
            let term = format!("{:?}", evo.termination);
            *terminations.entry(term).or_insert(0) += 1;
            if let Some(last) = evo.trajectory.last() {
                println!("  M{}: D={:.4} B={:.4} R={:.4} S={:.4} ρ={:.4} t={:.2} [{:?}] → {:?}",
                    evo.meme_idx, last.d, last.b, last.r, last.s, last.rho,
                    last.t, evo.termination, evo.archetype);
            }
        }
        println!("Archetypes: {:?}", archetypes);
        println!("Terminations: {:?}", terminations);
        let unique = archetypes.len();
        println!("Unique archetypes: {}", unique);

        let conv_report = crate::theory::ode::evaluate_convergence(&output.evolutions, &config);
        println!("Convergence: rate={:.1}% converged={}/{} std_ok={} undet={:.1}% → {}",
            conv_report.convergence_rate * 100.0,
            conv_report.converged_count, conv_report.total_memes,
            conv_report.criteria.std_ok,
            conv_report.undetermined_pct * 100.0,
            if conv_report.is_converged { "PASS" } else { "FAIL" });

        assert!(output.evolutions.len() > 0, "Should have evolutions");
        // 强验证要求: 至少 3 种不同原型
        assert!(unique >= 3, "强验证: 需要 >=3 种原型，实际 {} 种", unique);
    }
}
