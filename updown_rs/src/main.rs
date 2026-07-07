//! ↑↓ 泛模因建模引擎 — 主入口（完整五阶段）
//!
//! 数学对应: 泛模因理论 §3.2-§4.5, §6.2-§6.4
//!
//! 流程:
//!   Phase 1: 浮现 I → M = (S, F, C)
//!   Phase 2: 编码 M → G = (K, g, ω, Γ, R)
//!   Phase 3: 分解 G → Q = ({X_i}, Θ, C)
//!   Phase 4: 固化 Q → Credential
//!   Phase 5: 演化 Q → ODE 时间序列 → 原型分类
//!
//! 用法:
//!   updown input.txt                     # 完整五阶段流水线
//!   updown input.txt --text              # 自然语言文本模式
//!   updown input.txt --phase 1           # 仅 Phase 1
//!   updown input.txt -o output_dir       # 指定输出目录
//!   updown input.txt --auto-optimize     # 启用假设0全局优化

use updown::theory::{
    ode::OdeConfig,
    optimizer::OptimizerConfig,
};
use updown::pipeline::{
    phase1_emergence::run_phase_one,
    phase2_encoding::run_phase_two,
    phase3_decomposition::run_phase_three,
    phase4_binding::run_phase_four,
    phase5_evolution::run_phase_five,
};
use updown::io::{
    tokenizer::extract_ngrams,
    reversibility::verify_roundtrip,
};

use std::fs;
use std::io::{self, BufRead};
use std::path::Path;
use std::time::Instant;

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("用法: updown <输入文件> [选项]");
        eprintln!("选项:");
        eprintln!("  --phase <1|2|3|4|5>  只运行指定阶段 (默认: 全部)");
        eprintln!("  -o <目录>            输出目录 (默认: ./output)");
        eprintln!("  -r <轮数>             ↑↓ 最大循环轮数 (默认: 50)");
        eprintln!("  --text                自然语言文本模式（自动 n-gram 分词）");
        eprintln!("  --auto-optimize       启用假设0全局优化器");
        return Ok(());
    }

    let input_path = &args[1];
    let mut output_dir = "output".to_string();
    let mut max_rounds = 50usize;
    let mut phase_filter: Option<u8> = None;
    let mut text_mode = false;
    let mut auto_optimize = false;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--phase" => { i += 1; if i < args.len() { phase_filter = Some(args[i].parse().unwrap_or(0)); } }
            "-o" => { i += 1; if i < args.len() { output_dir = args[i].clone(); } }
            "-r" => { i += 1; if i < args.len() { max_rounds = args[i].parse().unwrap_or(50); } }
            "--text" => { text_mode = true; }
            "--auto-optimize" => { auto_optimize = true; }
            _ => {}
        }
        i += 1;
    }

    let source = Path::new(input_path).file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    println!("╔══════════════════════════════════════════════╗");
    println!("║  ↑↓ 泛模因建模引擎 — 五阶段流水线 v4        ║");
    println!("╚══════════════════════════════════════════════╝");
    println!("  输入: {}", input_path);
    println!("  输出: {}", output_dir);
    println!();

    let t_total = Instant::now();

    // ── 读取输入 ──
    let words: Vec<String> = if text_mode {
        let raw_text = fs::read_to_string(input_path)?;
        println!("  文本模式: {} 字符", raw_text.chars().count());
        let w = extract_ngrams(&raw_text);
        println!("  n-gram 词表: {} 词", w.len());
        w
    } else {
        let file = fs::File::open(input_path)?;
        let reader = io::BufReader::new(file);
        let mut w = Vec::new();
        for line in reader.lines() {
            let line = line?;
            let line = line.trim().to_string();
            if !line.is_empty() { w.push(line); }
        }
        println!("  词表模式: {} 词", w.len());
        w
    };

    fs::create_dir_all(&output_dir).ok();

    // ── Phase 1: 浮现 ──
    if phase_filter.map_or(true, |p| p <= 1) {
        println!();
        println!("  ┌─ Phase 1: 浮现 (I → M) ────────────────┐");
        let t1 = Instant::now();
        let phase1 = run_phase_one(words.clone(), max_rounds);
        println!("  │ 节点: {} 词 + {} 字 = {} 总节点",
            words.len(),
            phase1.s.vertices.len() - words.len(),
            phase1.s.vertices.len(),
        );
        println!("  │ 收敛轮数: {}, 最大深度: {:.1}", phase1.convergence_rounds, phase1.max_depth);
        println!("  │ 规则: {} 条, 约束: {} 项", phase1.f.len(), phase1.c.len());
        for detail in &phase1.reversibility_record {
            println!("  │ {}", detail);
        }
        println!("  └─ Phase 1 完成: {:.2}s ────────────────┘", t1.elapsed().as_secs_f64());

        if phase_filter == Some(1) {
            println!("\n总耗时: {:.2}s", t_total.elapsed().as_secs_f64());
            return Ok(());
        }

        // ── Phase 2: 编码 ──
        if phase_filter.map_or(true, |p| p <= 2) {
            println!();
            println!("  ┌─ Phase 2: 编码 (M → G) ────────────────┐");
            let t2 = Instant::now();
            let phase2 = run_phase_two(&phase1);
            println!("  │ 胞腔: {} 0-cells + {} 1-cells + {} 2-cells",
                phase2.complex.n_vertices(), phase2.complex.n_edges(),
                phase2.complex.cells.iter().filter(|c| c.dim == 2).count());
            println!("  │ Euler χ = {}, Betti: β₀={}, β₁={}",
                phase2.invariants.euler_char, phase2.invariants.betti_0, phase2.invariants.betti_1);
            println!("  │ 场: mean(|∇φ|)={:.4}, std={:.4}",
                phase2.vector_field.mean_gradient(), phase2.vector_field.std_gradient());
            println!("  └─ Phase 2 完成: {:.2}s ────────────────┘", t2.elapsed().as_secs_f64());

            if phase_filter == Some(2) {
                println!("\n总耗时: {:.2}s", t_total.elapsed().as_secs_f64());
                return Ok(());
            }

            // ── Phase 3: 分解 ──
            if phase_filter.map_or(true, |p| p <= 3) {
                println!();
                println!("  ┌─ Phase 3: 分解 (G → Q) ────────────────┐");
                let t3 = Instant::now();
                let phase3 = run_phase_three(&phase2);
                println!("  │ 模因数量: {} 个", phase3.n_memes);
                for (i, meme) in phase3.memes.iter().enumerate() {
                    println!("  │   M{}: D={:.3} B={:.3} ρ={:.3} R={:.3} S={:.3} | 顶点: {}",
                        i,
                        meme.five_dim.intrinsic_degree,
                        meme.five_dim.binding_degree,
                        meme.five_dim.energy_density,
                        meme.five_dim.evolution_rate,
                        meme.five_dim.structural_robustness,
                        meme.vertices.len(),
                    );
                }
                println!("  │ 耦合矩阵: {}×{}", phase3.coupling.len(),
                    if phase3.coupling.is_empty() { 0 } else { phase3.coupling[0].len() });
                println!("  └─ Phase 3 完成: {:.2}s ────────────────┘", t3.elapsed().as_secs_f64());

                // ── 往返验证 ──
                println!();
                println!("  ═══ 推论 5.1 验证: Φ⁻¹∘Φ = I ═══");
                let report = verify_roundtrip(&words, &phase1, &phase2, &phase3);
                for detail in &report.details {
                    println!("  {}", detail);
                }
                if report.words_fully_recoverable {
                    println!("  ▶ 推论 5.1 成立: 词完全可逆");
                }

                if phase_filter == Some(3) {
                    println!("\n总耗时: {:.2}s", t_total.elapsed().as_secs_f64());
                    return Ok(());
                }

                // ── Phase 4: 固化 ──
                if phase_filter.map_or(true, |p| p <= 4) {
                    println!();
                    println!("  ┌─ Phase 4: 固化 (Q → Credential) ───────┐");
                    let t4 = Instant::now();
                    let original_text = words.join(" ");
                    let phase4 = run_phase_four(phase3.clone(), &original_text);
                    println!("  │ 凭证: v{}", phase4.header.version);
                    println!("  │ 数据哈希: {}", phase4.data_hash);
                    println!("  │ 模因数: {}, 原始大小: {} 字节",
                        phase4.metadata.meme_count, phase4.metadata.original_size);
                    println!("  └─ Phase 4 完成: {:.2}s ────────────────┘", t4.elapsed().as_secs_f64());

                    if phase_filter == Some(4) {
                        println!("\n总耗时: {:.2}s", t_total.elapsed().as_secs_f64());
                        return Ok(());
                    }

                    // ── Phase 5: 演化 ──
                    if phase_filter.map_or(true, |p| p <= 5) {
                        println!();
                        println!("  ┌─ Phase 5: 演化 (Q → ODE → 原型) ──────┐");
                        let t5 = Instant::now();

                        let ode_config = OdeConfig {
                            max_steps: 20000,
                            ..OdeConfig::default()
                        };

                        let opt_config = if auto_optimize {
                            Some(OptimizerConfig::default())
                        } else {
                            None
                        };

                        let phase5 = run_phase_five(&phase3, &ode_config, opt_config.as_ref());

                        if let Some(ref hyp) = phase5.optimal_hypothesis {
                            println!("  │ 假设0: T*={:.3} Φ_D*={:?} Φ_R*={:?} L={:.6}",
                                hyp.threshold, hyp.family_d, hyp.family_r, hyp.loss);
                        }

                        for evo in &phase5.evolutions {
                            let n_pts = evo.trajectory.len();
                            if let Some(last) = evo.trajectory.last() {
                                println!("  │  M{}: t={:.2} 步={} D={:.4} B={:.4} ρ={:.4} R={:.4} S={:.4} [{}] → {:?}",
                                    evo.meme_idx, last.t, n_pts,
                                    last.d, last.b, last.rho, last.r, last.s,
                                    match evo.termination {
                                        _ => "OK",
                                    },
                                    evo.archetype,
                                );
                            }
                        }
                        println!("  └─ Phase 5 完成: {:.2}s ────────────────┘", t5.elapsed().as_secs_f64());

                        // 输出 ODE 轨迹 CSV
                        let ode_path = Path::new(&output_dir)
                            .join(format!("{}_ode_trajectory.csv", source));
                        let mut of = fs::File::create(&ode_path)?;
                        use std::io::Write;
                        writeln!(of, "meme_id,t,d,b,rho,r,s,archetype")?;
                        for evo in &phase5.evolutions {
                            for snap in &evo.trajectory {
                                writeln!(of, "{},{},{:.8},{:.8},{:.8},{:.8},{:.8},{:?}",
                                    evo.meme_idx, snap.t, snap.d, snap.b, snap.rho, snap.r, snap.s,
                                    evo.archetype)?;
                            }
                        }
                    }
                }
            }
        }
    }

    println!();
    println!("════════════════════════════════════════════════");
    println!("  总耗时: {:.2}s", t_total.elapsed().as_secs_f64());
    println!("════════════════════════════════════════════════");

    Ok(())
}