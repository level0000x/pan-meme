/// Property-based 测试 — 不变量验证
///
/// 需要 `--features property-tests` 启用（依赖 proptest）。
/// CI (Linux) 环境自动启用，Windows MinGW 因 dlltool 路径限制跳过。
///
/// 对任意合法输入，以下不变量必须成立：
/// 1. S ≥ 0（结构韧度非负）
/// 2. 所有 D, B, R, S ∈ [0, 1]（状态在 Ω 中）
/// 3. Phase 1 节点可恢复率 = 100%
/// 4. ODE 轨迹非空（给定合法初始状态）

#[cfg(feature = "property-tests")]
use proptest::prelude::*;
#[cfg(feature = "property-tests")]
use updown::io::reversibility::verify_roundtrip;
#[cfg(feature = "property-tests")]
use updown::io::tokenizer::extract_ngrams;
#[cfg(feature = "property-tests")]
use updown::pipeline::{
    phase1_emergence::run_phase_one, phase2_encoding::run_phase_two,
    phase3_decomposition::run_phase_three, phase5_evolution::run_phase_five,
};
#[cfg(feature = "property-tests")]
use updown::theory::ode::OdeConfig;

/// 生成随机中文文本（n 个词，从候选池中抽取）
#[cfg(feature = "property-tests")]
fn arb_chinese_text(max_words: usize) -> impl Strategy<Value = String> {
    let pool = vec![
        "信息",
        "熵",
        "演化",
        "结构",
        "网络",
        "基因",
        "模因",
        "复杂度",
        "守恒",
        "系统",
        "动态",
        "反馈",
        "层次",
        "自组织",
        "涌现",
    ];
    proptest::collection::vec(0..pool.len(), 1..max_words.max(1)).prop_map(move |indices| {
        indices
            .iter()
            .map(|&i| pool[i])
            .collect::<Vec<_>>()
            .join("")
    })
}

#[cfg(feature = "property-tests")]
proptest! {
    /// 不变量 1+2: 五维状态必须在 Ω = [0,1]⁴×[0,∞) 内
    #[test]
    fn five_dim_state_in_omega(text in arb_chinese_text(10)) {
        let words = extract_ngrams(&text);
        if words.is_empty() { return Ok(()); }
        let p1 = run_phase_one(words, 50);
        let p2 = run_phase_two(&p1);
        let p3 = run_phase_three(&p2, 1.0);

        for meme in &p3.memes {
            let s = meme.five_dim;
            prop_assert!(s.intrinsic_degree >= 0.0 && s.intrinsic_degree <= 1.0,
                "D 超出 [0,1]: {}", s.intrinsic_degree);
            prop_assert!(s.binding_degree >= 0.0 && s.binding_degree <= 1.0,
                "B 超出 [0,1]: {}", s.binding_degree);
            prop_assert!(s.energy_density >= 0.0,
                "ρ 为负: {}", s.energy_density);
            prop_assert!(s.evolution_rate >= 0.0 && s.evolution_rate <= 1.0,
                "R 超出 [0,1]: {}", s.evolution_rate);
            prop_assert!(s.structural_robustness >= 0.0 && s.structural_robustness <= 1.0,
                "S 为负或超界: {}", s.structural_robustness);
        }
    }

    /// 不变量 3: 词可逆性 = 100%
    #[test]
    fn words_fully_recoverable(text in arb_chinese_text(8)) {
        let words = extract_ngrams(&text);
        if words.is_empty() { return Ok(()); }
        let p1 = run_phase_one(words.clone(), 50);
        let p2 = run_phase_two(&p1);
        let p3 = run_phase_three(&p2, 1.0);
        let report = verify_roundtrip(&words, &p1, &p2, &p3);
        prop_assert!(report.words_fully_recoverable,
            "词不可逆: rate={}", report.recovery_rate);
    }

    /// 不变量 4: ODE 轨迹非空
    #[test]
    fn ode_trajectory_nonempty(text in arb_chinese_text(6)) {
        let words = extract_ngrams(&text);
        if words.is_empty() { return Ok(()); }
        let p1 = run_phase_one(words, 50);
        let p2 = run_phase_two(&p1);
        let p3 = run_phase_three(&p2, 1.0);
        if p3.memes.is_empty() { return Ok(()); }

        let config = OdeConfig { max_steps: 500, ..OdeConfig::default() };
        let p5 = run_phase_five(&p3, &config, None);
        for evo in &p5.evolutions {
            prop_assert!(evo.trajectory.len() > 0,
                "模因 M{} 空轨迹 (reason={:?})", evo.meme_idx, evo.termination);
        }
    }
}
