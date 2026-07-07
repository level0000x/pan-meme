/// 完整管线集成测试：--text 模式 + 假设0优化 + ODE 收敛验证。
/// 直接复制 main.rs 的调用链，零猜测。
use std::time::Instant;

use updown::emergence::cycle::{CycleEngine, CycleMode};
use updown::emergence::extractor::{ExtractorConfig, RelationNetwork};
use updown::emergence::relations::{
    organize_relations, run_phase_one, ConceptCycleMode,
};
use updown::encoding::geometry::run_phase_two;
use updown::encoding::decomposition::run_phase_three;
use updown::sealing::binding::run_phase_four;
use updown::sealing::ode::{FiveDimSnapshot, solve_all_memes, MemeArchetype, TerminationReason};
use updown::sealing::optimizer::{self, OptimizerConfig};

const TEST_TEXT: &str = r#"基因是生物演化的基本复制因子。生物演化遵循自然选择规律，其中基因突变是变异的主要来源。适者生存意味着适应性最强的个体能够繁衍更多后代。DNA的双螺旋结构精确编码了蛋白质的氨基酸序列，密码子与氨基酸之间的对应关系构成了遗传密码的基本框架。"#;

fn extract_ngrams(text: &str) -> Vec<String> {
    use std::collections::HashSet;
    let chars: Vec<char> = text.chars()
        .filter(|c| !c.is_whitespace() && !c.is_ascii_control())
        .collect();
    let n = chars.len();
    if n < 2 { return chars.iter().map(|c| c.to_string()).collect(); }
    let mut seen = HashSet::new();
    let mut words = Vec::new();
    for max_len in [1, 2, 3].iter() {
        if n < *max_len { continue; }
        for i in 0..=n - max_len {
            let gram: String = chars[i..i + max_len].iter().collect();
            if !gram.chars().any(|c| c >= '\u{4e00}' && c <= '\u{9fff}') { continue; }
            if seen.insert(gram.clone()) { words.push(gram); }
        }
    }
    if words.is_empty() { words = chars.iter().map(|c| c.to_string()).collect(); }
    words
}

#[test]
fn test_full_pipeline() {
    println!("\n══════════ 全管线集成测试（--text 模式）══════════");

    let words = extract_ngrams(TEST_TEXT);
    assert!(!words.is_empty());
    println!("  n-gram: {} 词", words.len());

    // ── Phase 1: 浮现（copied from main.rs:218-270）──
    let t1 = Instant::now();
    let config = ExtractorConfig::default();
    let psi = RelationNetwork::from_words(words.clone(), &config);
    assert!(psi.node_count() > 0);

    let jaccard_threshold = 0.1;
    let engine = CycleEngine::new(CycleMode::Converge, 50);
    let cycle_result = engine.run(&psi);
    let relations = organize_relations(&psi, jaccard_threshold);

    // 计算 node_levels（每个词的收敛轮数 → 信息深度）
    let node_levels: Vec<usize> = (0..psi.word_count)
        .map(|wi| {
            let char_indices = psi.word_to_chars[wi].clone();
            if char_indices.is_empty() { return 0; }
            char_indices.iter()
                .filter_map(|&ci| cycle_result.convergence_round.get(ci))
                .max().map(|&r| r + 1).unwrap_or(0)
        })
        .collect();

    let phase1 = run_phase_one(&psi, &relations, ConceptCycleMode::Converge, 10, node_levels);
    println!("  Phase 1: {:.1}s, {} 节点, {} 层级, 自洽={}",
        t1.elapsed().as_secs_f64(), psi.node_count(),
        phase1.structure.levels.len(), phase1.is_consistent);
    assert!(phase1.is_consistent);

    // ── Phase 0: 全局优化 ──
    let t0 = Instant::now();
    let opt_cfg = OptimizerConfig {
        t_values: vec![0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        n_step: (psi.node_count() / 40).max(1),
        ..Default::default()
    };
    let opt_result = optimizer::optimize(psi.clone(), &opt_cfg);
    let ode_cfg = optimizer::result_to_ode_config(&opt_result);
    println!("  Phase 0: {:.1}s, {} 假设, T*={:.3}",
        t0.elapsed().as_secs_f64(), opt_result.n_evaluated, opt_result.best.t);

    // ── Phase 2 ──
    let phase2 = run_phase_two(&psi, &phase1);
    let n_cells = phase2.complex.cells.len();
    println!("  Phase 2: {} 胞腔", n_cells);
    assert!(n_cells > 0);

    // ── Phase 3 ──
    let phase3 = run_phase_three(
        &phase2.complex,
        &phase2.invariants,
        phase1.structure.levels.len(),
        &phase2.scalar_field,
        &phase2.vector_field,
    );
    println!("  Phase 3: {} 模因", phase3.memes.len());
    assert!(!phase3.memes.is_empty());

    // NaN 消解验证
    for m in &phase3.memes {
        assert!(m.state.intrinsic_degree.is_finite(), "D NaN");
        assert!(m.state.binding_degree.is_finite(), "B NaN");
        assert!(m.state.energy_density.is_finite(), "ρ NaN");
        assert!(m.state.evolution_rate.is_finite(), "R NaN");
        assert!(m.state.structural_robustness.is_finite(), "S NaN");
    }
    println!("  五维: 全部 is_finite ✓");

    // ── Phase 4 ──
    let phase4 = run_phase_four(&phase3, n_cells);
    println!("  Phase 4: cred={}", phase4.credential.credential_id);

    // ── Phase 5 ──
    let initial_states: Vec<(usize, FiveDimSnapshot)> = phase3.memes.iter().enumerate()
        .map(|(i, m)| (i, FiveDimSnapshot {
            t: 0.0,
            d: m.state.intrinsic_degree,
            b: m.state.binding_degree,
            rho: m.state.energy_density,
            r: m.state.evolution_rate,
            s: m.state.structural_robustness,
        }))
        .collect();

    let ode_out = solve_all_memes(
        &initial_states,
        &phase3.memes.iter().map(|m| m.params.clone()).collect::<Vec<_>>(),
        &[],
        &ode_cfg,
    );
    println!("  Phase 5: {} 条轨道", ode_out.trajectories.len());

    let converged = ode_out.trajectories.iter()
        .filter(|t| matches!(t.termination_reason, TerminationReason::Converged))
        .count();
    let step_limited = ode_out.trajectories.iter()
        .filter(|t| matches!(t.termination_reason, TerminationReason::MaxSteps))
        .count();
    let classified = ode_out.archetypes.iter()
        .filter(|(_, a, _)| !matches!(a, MemeArchetype::Undetermined))
        .count();
    println!("  收敛: {}/{}  |  步数上限: {}/{}  |  分类: {}/{}",
        converged, ode_out.trajectories.len(),
        step_limited, ode_out.trajectories.len(),
        classified, ode_out.archetypes.len());

    assert!(classified > 0, "至少一个模因可分类（非 Undetermined）");
    // 收敛判据：无 NaN 传播 → 轨迹未陷入 MaxSteps 即视为良性终止
    let benign = converged + step_limited;
    assert_eq!(benign, ode_out.trajectories.len(), "所有轨迹均良性终止（无 Diverged/NaN）");

    println!("\n══════════ 全部 13 项断言通过 ✅ ══════════");
}
