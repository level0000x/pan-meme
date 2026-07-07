/// Pipeline benchmarks
///
/// Run: cargo bench
use std::time::Instant;
use updown::io::tokenizer::extract_ngrams;
use updown::pipeline::{
    phase1_emergence::run_phase_one, phase2_encoding::run_phase_two,
    phase3_decomposition::run_phase_three, phase5_evolution::run_phase_five,
};
use updown::theory::ode::OdeConfig;

const BENCH_TEXT: &str = r#"信息是物理世界的基本构成要素，而非仅仅是认识论意义上的抽象概念。熵增原理与信息守恒定律构成了宇宙演化的两条基本脉络。生命系统通过负熵流的持续输入来维持其有序结构，这一过程与热力学第二定律并不矛盾，而是对其在开放系统中的补充。"#;

fn bench_phase1() {
    let words = extract_ngrams(BENCH_TEXT);
    let start = Instant::now();
    let _ = run_phase_one(words, 100);
    println!("  Phase 1: {:?}", start.elapsed());
}

fn bench_phase1_to_3() {
    let words = extract_ngrams(BENCH_TEXT);
    let p1 = run_phase_one(words.clone(), 100);
    let p2 = run_phase_two(&p1);
    let start = Instant::now();
    let _ = run_phase_three(&p2);
    println!("  Phase 3: {:?}", start.elapsed());
}

fn bench_phase5() {
    let words = extract_ngrams(BENCH_TEXT);
    let p1 = run_phase_one(words, 100);
    let p2 = run_phase_two(&p1);
    let p3 = run_phase_three(&p2);
    let config = OdeConfig {
        max_steps: 2000,
        ..OdeConfig::default()
    };
    let start = Instant::now();
    let _ = run_phase_five(&p3, &config, None);
    println!("  Phase 5: {:?}", start.elapsed());
}

fn bench_full_pipeline() {
    let words = extract_ngrams(BENCH_TEXT);
    let start = Instant::now();
    let p1 = run_phase_one(words.clone(), 100);
    let p2 = run_phase_two(&p1);
    let p3 = run_phase_three(&p2);
    let config = OdeConfig {
        max_steps: 2000,
        ..OdeConfig::default()
    };
    let _ = run_phase_five(&p3, &config, None);
    println!("  Full pipeline: {:?}", start.elapsed());
}

fn main() {
    println!("=== Pipeline Benchmarks ===");
    println!(
        "Text: {} chars, n-gram extract then run",
        BENCH_TEXT.chars().count()
    );
    println!();
    bench_phase1();
    bench_phase1_to_3();
    bench_phase5();
    bench_full_pipeline();
}
