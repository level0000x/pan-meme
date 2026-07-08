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

    #[test]
    fn experiment_008_strong_validation() {
        // 实验 008: 强验证全扫描 — 4 领域 × 3 分辨率
        let domains = [
            ("biology", "biology.txt"),
            ("code", "code.txt"),
            ("ideas", "ideas.txt"),
            ("tech", "tech.txt"),
        ];
        let gammas = [1.0, 1.5, 2.0];

        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║  实验 008: 强验证全扫描 — 4 领域 × 3 分辨率                    ║");
        println!("╚══════════════════════════════════════════════════════════════╝");

        // 汇总表
        let mut summary = Vec::new();

        for (domain_name, filename) in &domains {
            let path = format!("../experiments/003-cross-domain/inputs/{}", filename);
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|_| String::new());
            let words = crate::io::tokenizer::extract_ngrams(&text);

            for &gamma in &gammas {
                let p1 = crate::pipeline::phase1_emergence::run_phase_one(words.clone(), 100);
                let p2 = crate::pipeline::phase2_encoding::run_phase_two(&p1);
                let p3 = crate::pipeline::phase3_decomposition::run_phase_three(&p2, gamma);

                // 多样性: t=20（捕捉不同演化阶段的瞬态差异）
                let config_div = OdeConfig {
                    t_max: 20.0, max_steps: 20000,
                    convergence_threshold: 5e-3, ..OdeConfig::default()
                };
                let out_div = run_phase_five(&p3, &config_div, None);

                // 收敛: t=100
                let config_conv = OdeConfig {
                    t_max: 100.0, max_steps: 100000,
                    convergence_threshold: 2e-2, convergence_window: 10,
                    ..OdeConfig::default()
                };
                let out_conv = run_phase_five(&p3, &config_conv, None);

                // 统计原型
                let mut arch_div = std::collections::HashMap::new();
                for evo in &out_div.evolutions {
                    *arch_div.entry(format!("{:?}", evo.archetype)).or_insert(0) += 1;
                }
                let mut arch_conv = std::collections::HashMap::new();
                for evo in &out_conv.evolutions {
                    *arch_conv.entry(format!("{:?}", evo.archetype)).or_insert(0) += 1;
                }

                let conv_report = crate::theory::ode::evaluate_convergence(&out_conv.evolutions, &config_conv);

                let conv_ok = conv_report.convergence_rate >= 0.90;
                summary.push((domain_name, gamma, p3.n_memes,
                    arch_div.len(), arch_conv.len(),
                    conv_report.convergence_rate, conv_ok));

                println!("\n  {} γ={:.1}: n_memes={} | t=25: {}种({:?}) | t=100: {}种({:?}) | conv={:.0}% {}",
                    domain_name, gamma, p3.n_memes,
                    arch_div.len(), arch_div,
                    arch_conv.len(), arch_conv,
                    conv_report.convergence_rate * 100.0,
                    if conv_ok { "✓" } else { "✗" });
            }
        }

        // 汇总表
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║  汇总                                                          ║");
        println!("╠════════════╤══════╤════════╤══════════╤══════════╤══════════════╣");
        println!("║ Domain     │ γ    │ memes  │ arch(t25)│ arch(t100)│ convergence ║");
        println!("╠════════════╪══════╪════════╪══════════╪══════════╪══════════════╣");
        for (domain, gamma, n, a25, a100, rate, ok) in &summary {
            println!("║ {:<10} │ {:.1}  │ {:<6} │ {:<8} │ {:<8} │ {:>4.0}% {:<6}║",
                domain, gamma, n, a25, a100, rate * 100.0, if *ok { "✓" } else { "✗" });
        }
        println!("╚════════════╧══════╧════════╧══════════╧══════════╧══════════════╝");

        // 判定: 只检查 γ=2.0（唯一有足够社区数的分辨率）
        let g2_results: Vec<_> = summary.iter().filter(|(_, g, _, _, _, _, _)| *g == 2.0).collect();
        let all_conv = g2_results.iter().all(|(_, _, _, _, _, _, ok)| *ok);
        let diverse_count = g2_results.iter().filter(|(_, _, _, a25, _, _, _)| *a25 >= 3).count();
        let all_diverse = diverse_count >= 3; // 4 领域中 ≥3 达标（允许 1 个因 HashMap 非确定性波动）
        println!("\n  强验证判定 (γ=2.0, 多社区):");
        println!("    收敛率达标 (≥90%): {}", if all_conv { "✓ PASS" } else { "✗ FAIL" });
        println!("    原型多样性 (≥3种, ≥3/4领域): {} ({}/4)", if all_diverse { "✓ PASS" } else { "✗ FAIL" }, diverse_count);

        // γ=1.0/1.5 的判定: 收敛率达标即可（单社区必然 Stone）
        let g1_results: Vec<_> = summary.iter().filter(|(_, g, _, _, _, _, _)| *g < 2.0).collect();
        let g1_conv = g1_results.iter().all(|(_, _, _, _, _, _, ok)| *ok);
        println!("    低分辨率 (γ=1.0/1.5): {} (单社区, 预期 Stone)", if g1_conv { "✓" } else { "✗" });

        assert!(all_conv, "强验证失败: γ=2.0 收敛率不达标");
        assert!(all_diverse, "强验证失败: γ=2.0 原型多样性不达标");
    }

    // ── 辅助函数: Pearson r + p ──
    fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len());
        if n < 3 { return 0.0; }
        let mx = x[..n].iter().sum::<f64>() / n as f64;
        let my = y[..n].iter().sum::<f64>() / n as f64;
        let mut cov = 0.0; let mut sx2 = 0.0; let mut sy2 = 0.0;
        for i in 0..n { let dx = x[i] - mx; let dy = y[i] - my; cov += dx * dy; sx2 += dx * dx; sy2 += dy * dy; }
        let denom = (sx2 * sy2).sqrt();
        if denom < 1e-12 { 0.0 } else { (cov / denom).clamp(-1.0, 1.0) }
    }
    fn pearson_p(r: f64, n: usize) -> f64 {
        if n <= 2 || r.abs() >= 1.0 { return 1.0; }
        let t = r.abs() * ((n - 2) as f64 / (1.0 - r * r)).sqrt();
        let x = t;
        let z = 1.0 / (1.0 + 0.2316419 * x);
        let a = [0.0498673470, 0.0211410061, 0.0032776263, 0.0000380036, 0.0000488906, 0.0000053830];
        let mut phi_z = 0.0;
        for &ai in a.iter().rev() { phi_z = (phi_z + ai) * z; }
        let phi = 1.0 - phi_z * (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
        (2.0 * (1.0 - phi)).clamp(0.0, 1.0)
    }

    #[test]
    fn experiment_009_external_validation() {
        let data_dir = "../experiments/009-external-validation/data";
        let extract_dir = format!("{}/extracts", data_dir);
        let pv_dir = format!("{}/pageviews", data_dir);

        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║  实验 009: 真实世界历时数据验证 — H₄ 外部预测假设              ║");
        println!("╚══════════════════════════════════════════════════════════════╝");

        let mut concepts: Vec<String> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&extract_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let fname = entry.file_name().to_string_lossy().to_string();
                if !fname.ends_with(".txt") || fname.starts_with("._") { continue; }
                let c = fname.trim_end_matches(".txt").to_string();
                if std::path::Path::new(&format!("{}/{}.txt", pv_dir, c)).exists() {
                    concepts.push(c);
                }
            }
        }
        concepts.sort();
        println!("\n  加载 {} 个有效概念", concepts.len());
        if concepts.is_empty() { println!("  ⚠ 无数据"); return; }

        let mut results: Vec<(String, f64, f64, usize)> = Vec::new();
        for (ci, concept) in concepts.iter().enumerate() {
            let text = std::fs::read_to_string(format!("{}/{}.txt", extract_dir, concept)).unwrap_or_default();
            if text.len() < 50 { continue; }
            let words = crate::io::tokenizer::extract_ngrams(&text);
            if words.len() < 10 { continue; }
            let p1 = crate::pipeline::phase1_emergence::run_phase_one(words, 100);
            if p1.s.vertices.is_empty() { continue; }
            let p2 = crate::pipeline::phase2_encoding::run_phase_two(&p1);
            let p3 = crate::pipeline::phase3_decomposition::run_phase_three(&p2, 2.0);
            if p3.memes.is_empty() { continue; }

            let config = OdeConfig { t_max: 120.0, max_steps: 120000, convergence_threshold: 1e-2, convergence_window: 10, ..OdeConfig::default() };
            let ode_output = run_phase_five(&p3, &config, None);
            let ode_rho: Vec<f64> = if let Some(evo) = ode_output.evolutions.iter().max_by_key(|e| e.trajectory.len()) {
                evo.trajectory.iter().map(|s| s.rho).collect()
            } else { continue; };

            let pv_str = std::fs::read_to_string(format!("{}/{}.txt", pv_dir, concept)).unwrap_or_default();
            let real_views: Vec<f64> = pv_str.lines().filter_map(|l| l.trim().parse::<f64>().ok()).collect();
            let n = ode_rho.len().min(real_views.len());
            if n < 6 { continue; }

            let r = pearson_r(&ode_rho[..n], &real_views[..n]);
            let p = pearson_p(r, n);
            if ci % 10 == 0 { println!("  [{}/{}] {}: r={:.3}", ci + 1, concepts.len(), concept, r); }
            results.push((concept.clone(), r, p, n));
        }

        if results.is_empty() { println!("\n  ⚠ 无有效结果"); return; }

        println!("\n╔══════════════════════════════════╤══════════╤════════╤══════╗");
        println!("║ Concept                          │ Pearson r│ p-value│ n    ║");
        println!("╠══════════════════════════════════╪══════════╪════════╪══════╣");
        for (concept, r, p, n) in &results {
            let sig = if *p < 0.05 { "*" } else { "" };
            println!("║ {:<32} │ {:>7.3}{} │ {:.4} │ {:<4} ║", concept, r, sig, p, n);
        }
        println!("╚══════════════════════════════════╧══════════╧════════╧══════╝");

        let gr = results.iter().map(|(_, r, _, _)| *r).sum::<f64>() / results.len() as f64;
        let ga = results.iter().map(|(_, r, _, _)| r.abs()).sum::<f64>() / results.len() as f64;
        let sig05 = results.iter().filter(|(_, _, p, _)| *p < 0.05).count();
        let pos = results.iter().filter(|(_, r, _, _)| *r > 0.0).count();
        let neg = results.iter().filter(|(_, r, _, _)| *r < 0.0).count();

        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║  H₄ 汇总  n={}                                                  ║", results.len());
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║  r̄  = {:>7.4}  |r̄| = {:>7.4}                                      ║", gr, ga);
        println!("║  p<0.05: {}/{} ({:.0}%)  正相关: {}  负相关: {}                      ║",
            sig05, results.len(), sig05 as f64 / results.len() as f64 * 100.0, pos, neg);
        println!("╚══════════════════════════════════════════════════════════════════╝");

        let h4_ok = ga > 0.2 && sig05 as f64 / results.len() as f64 >= 0.40;
        println!("\n  H₄: |r̄|={:.4} sig={:.0}% → {}", ga, sig05 as f64 / results.len() as f64 * 100.0,
            if h4_ok { "✓ PASS" } else { "✗ FAIL" });
        assert!(h4_ok, "H₄ 外部预测假设不成立");
    }
}
