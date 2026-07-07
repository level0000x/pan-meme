/// Golden file 测试 — 防止回归
///
/// 每次修改代码后运行，对比输出是否与 gold 一致。
/// 若有意改变行为，删除旧的 .gold 文件重新生成。
use std::fs;

use updown::io::reversibility::verify_roundtrip;
use updown::io::tokenizer::extract_ngrams;
use updown::pipeline::{
    phase1_emergence::run_phase_one, phase2_encoding::run_phase_two,
    phase3_decomposition::run_phase_three, phase5_evolution::run_phase_five,
};
use updown::theory::ode::OdeConfig;

const GOLDEN_INPUT: &str = r#"信息是物理世界的基本构成要素，而非仅仅是认识论意义上的抽象概念。熵增原理与信息守恒定律构成了宇宙演化的两条基本脉络。"#;

/// 采集当前管线输出，写入字符串
fn run_and_collect() -> String {
    let words = extract_ngrams(GOLDEN_INPUT);
    let p1 = run_phase_one(words.clone(), 100);
    let p2 = run_phase_two(&p1);
    let p3 = run_phase_three(&p2, 1.0);
    let report = verify_roundtrip(&words, &p1, &p2, &p3);

    let mut out = String::new();
    out.push_str(&format!(
        "words={} vertices={} depths={:.1}\n",
        words.len(),
        p1.s.vertices.len(),
        p1.max_depth
    ));
    out.push_str(&format!(
        "betti_0={} betti_1={}\n",
        p2.invariants.betti_0, p2.invariants.betti_1
    ));
    out.push_str(&format!("n_memes={}\n", p3.n_memes));
    for (i, meme) in p3.memes.iter().enumerate() {
        out.push_str(&format!(
            "meme_{}: D={:.3} B={:.3} S={:.3}\n",
            i,
            meme.five_dim.intrinsic_degree,
            meme.five_dim.binding_degree,
            meme.five_dim.structural_robustness
        ));
    }
    out.push_str(&format!(
        "recovery_rate={:.4} entropy_conserved={}\n",
        report.recovery_rate, report.entropy_conserved
    ));

    let ode_config = OdeConfig {
        max_steps: 500,
        ..OdeConfig::default()
    };
    let p5 = run_phase_five(&p3, &ode_config, None);
    out.push_str(&format!("n_evolutions={}\n", p5.evolutions.len()));
    for evo in &p5.evolutions {
        out.push_str(&format!(
            "evo_{}: steps={} archetype={:?}\n",
            evo.meme_idx,
            evo.trajectory.len(),
            evo.archetype
        ));
    }

    out
}

#[test]
fn test_golden() {
    let current = run_and_collect();
    let gold_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/pipeline.gold");

    // 若 gold 文件不存在，创建它（首次运行）
    if !std::path::Path::new(gold_path).exists() {
        let parent = std::path::Path::new(gold_path).parent().unwrap();
        fs::create_dir_all(parent).unwrap();
        fs::write(gold_path, &current).unwrap();
        println!("Golden file created at {}", gold_path);
        return;
    }

    let expected = fs::read_to_string(gold_path).unwrap();
    assert_eq!(
        current, expected,
        "Golden test failed! Output has changed.\n\
        If this is intentional, delete {} and re-run.",
        gold_path
    );
}
