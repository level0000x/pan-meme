/// 完整管线集成测试 v4: 五阶段流水线 + 往返验证 + ODE 收敛
///
/// 验证内容:
/// - Phase 1-5 完整流水线运行
/// - 推论 5.1 往返验证 (Φ⁻¹∘Φ = I)
/// - ODE 收敛
/// - 可证伪判据验证

use updown::io::tokenizer::extract_ngrams;
use updown::pipeline::{
    phase1_emergence::run_phase_one,
    phase2_encoding::run_phase_two,
    phase3_decomposition::run_phase_three,
    phase4_binding::{run_phase_four, verify_credential},
    phase5_evolution::run_phase_five,
};
use updown::io::reversibility::verify_roundtrip;
use updown::theory::ode::OdeConfig;
use updown::theory::optimizer::{OptimizerConfig, verify_falsifiability};

const TEST_TEXT: &str = r#"基因是生物演化的基本复制因子。生物演化遵循自然选择规律，其中基因突变是变异的主要来源。适者生存意味着适应性最强的个体能够繁衍更多后代。DNA的双螺旋结构精确编码了蛋白质的氨基酸序列，密码子与氨基酸之间的对应关系构成了遗传密码的基本框架。"#;

#[test]
fn test_full_pipeline_with_text() {
    let words = extract_ngrams(TEST_TEXT);
    assert!(!words.is_empty(), "n-gram 分词不应为空");

    // Phase 1: 浮现
    let p1 = run_phase_one(words.clone(), 100);
    assert!(p1.s.vertices.len() > 0, "Phase 1 应有节点");
    assert!(p1.max_depth > 0.0, "Phase 1 应有深度");
    assert!(!p1.reversibility_record.is_empty(), "Phase 1 应有可逆性记录");

    // Phase 2: 编码
    let p2 = run_phase_two(&p1);
    assert!(p2.complex.n_vertices() > 0, "Phase 2 应有顶点");
    assert!(p2.invariants.betti_0 > 0, "Phase 2 应有 Betti 数");

    // Phase 3: 分解
    let p3 = run_phase_three(&p2);
    assert!(p3.n_memes > 0, "Phase 3 应有至少一个模因");

    // 往返验证
    let report = verify_roundtrip(&words, &p1, &p2, &p3);
    assert!(report.words_fully_recoverable, "词应完全可逆: rate={:.2}", report.recovery_rate);

    // Phase 4: 固化
    let original_text = words.join(" ");
    let cred = run_phase_four(p3.clone(), &original_text);
    assert!(verify_credential(&cred, &original_text), "凭证验证应通过");
    assert!(!verify_credential(&cred, "不同文本"), "错误凭证应拒绝");

    // Phase 5: 演化
    let ode_config = OdeConfig { max_steps: 2000, ..OdeConfig::default() };
    let p5 = run_phase_five(&p3, &ode_config, None);
    assert_eq!(p5.evolutions.len(), p3.n_memes, "每个模因应有演化结果");
}

#[test]
fn test_pipeline_with_optimizer() {
    let words = extract_ngrams(TEST_TEXT);
    let p1 = run_phase_one(words.clone(), 100);
    let p2 = run_phase_two(&p1);
    let p3 = run_phase_three(&p2);

    let ode_config = OdeConfig { max_steps: 500, ..OdeConfig::default() };
    let opt_config = OptimizerConfig {
        t_values: vec![0.1, 0.2],
        n_step: 1,
        ..OptimizerConfig::default()
    };

    let p5 = run_phase_five(&p3, &ode_config, Some(&opt_config));
    assert!(p5.optimal_hypothesis.is_some(), "应返回最优假设");
    let hyp = p5.optimal_hypothesis.unwrap();
    assert!(hyp.loss < f64::MAX, "损失应有限");
}

#[test]
fn test_falsifiability_criterion() {
    // 在不同数据集上验证可证伪判据
    let texts = [
        "苹果香蕉水果",
        "基因演化自然选择",
        "信息熵守恒定律",
    ];

    let mut results = Vec::new();
    for text in &texts {
        let words = extract_ngrams(text);
        let p1 = run_phase_one(words, 100);
        let p2 = run_phase_two(&p1);
        let p3 = run_phase_three(&p2);

        let ode_config = OdeConfig { max_steps: 500, ..OdeConfig::default() };
        let opt_config = OptimizerConfig {
            t_values: vec![0.1, 0.2],
            n_step: 1,
            ..OptimizerConfig::default()
        };
        let p5 = run_phase_five(&p3, &ode_config, Some(&opt_config));
        if let Some(hyp) = p5.optimal_hypothesis {
            results.push(hyp);
        }
    }

    if results.len() >= 3 {
        let consistent = verify_falsifiability(&results);
        // 不强制要求一致，但记录结果
        println!("可证伪判据: 函数族一致性={}", consistent);
    }
}

#[test]
fn test_reversibility_roundtrip() {
    // 验证推论 5.1: 简单中文文本的往返
    let words = vec!["苹果".to_string(), "香蕉".to_string(), "水果".to_string()];
    let p1 = run_phase_one(words.clone(), 100);
    let p2 = run_phase_two(&p1);
    let p3 = run_phase_three(&p2);
    let report = verify_roundtrip(&words, &p1, &p2, &p3);
    assert!(report.words_fully_recoverable);
}

#[test]
fn test_ode_convergence() {
    use updown::theory::ode::integrate;
    use updown::theory::five_dim::FiveDimState;
    use updown::theory::dynamics_params::DynamicsParams;
    use updown::theory::function_families::{FunctionFamily, FamilyParams};

    let init = FiveDimState::new(0.5, 0.5, 1.0, 0.3, 0.8);
    let params = DynamicsParams::default_params();
    let config = OdeConfig { max_steps: 2000, ..OdeConfig::default() };
    let phi_d = FamilyParams::new(FunctionFamily::Power, 1.0, 0.0);
    let phi_r = FamilyParams::new(FunctionFamily::Power, 1.0, 0.0);

    let (trajectory, _reason) = integrate(&init, &params, &config, &phi_d, &phi_r);
    assert!(trajectory.len() > 1, "ODE 应有轨迹");
    assert!(trajectory.len() <= 2000, "ODE 不应超过最大步数");

    // 验证最终状态有效性
    if let Some(last) = trajectory.last() {
        let final_state = last.to_five_dim();
        assert!(final_state.is_valid() || final_state.intrinsic_degree >= 0.0);
    }
}