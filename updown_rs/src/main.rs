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
//!   updown input.txt --verbose           # 详细日志输出

use std::fs;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;

use updown::io::{reversibility::verify_roundtrip, tokenizer::extract_ngrams};
use updown::pipeline::{
    phase1_emergence::run_phase_one, phase2_encoding::run_phase_two,
    phase3_decomposition::run_phase_three, phase4_binding::run_phase_four,
    phase5_evolution::run_phase_five,
};
use updown::theory::{ode::OdeConfig, ode::evaluate_convergence, optimizer::OptimizerConfig};

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
        eprintln!("  --verbose             详细日志输出");
        return Ok(());
    }

    let input_path = &args[1];
    let mut output_dir = "output".to_string();
    let mut max_rounds = 50usize;
    let mut phase_filter: Option<u8> = None;
    let mut text_mode = false;
    let mut auto_optimize = false;
    let mut verbose = false;
    let mut t_max = 5.0;
    let mut max_steps = 20000usize;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--phase" => {
                i += 1;
                if i < args.len() {
                    phase_filter = Some(args[i].parse().unwrap_or(0));
                }
            }
            "-o" => {
                i += 1;
                if i < args.len() {
                    output_dir = args[i].clone();
                }
            }
            "-r" => {
                i += 1;
                if i < args.len() {
                    max_rounds = args[i].parse().unwrap_or(50);
                }
            }
            "--text" => text_mode = true,
            "--auto-optimize" => auto_optimize = true,
            "--verbose" => verbose = true,
            "--t-max" => {
                i += 1;
                if i < args.len() {
                    t_max = args[i].parse().unwrap_or(5.0);
                }
            }
            "--max-steps" => {
                i += 1;
                if i < args.len() {
                    max_steps = args[i].parse().unwrap_or(20000);
                }
            }
            _ => {}
        }
        i += 1;
    }

    let source = Path::new(input_path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    println!("╔══════════════════════════════════════════════╗");
    println!("║  ↑↓ 泛模因建模引擎 — 五阶段流水线 v4        ║");
    println!("╚══════════════════════════════════════════════╝");
    println!("  输入: {}  输出: {}", input_path, output_dir);
    println!();

    let t_total = Instant::now();

    // ── 读取输入 ──
    let words: Vec<String> = if text_mode {
        let raw_text = fs::read_to_string(input_path)?;
        if verbose {
            eprintln!("[text] chars={}", raw_text.chars().count());
        }
        let w = extract_ngrams(&raw_text);
        if verbose {
            eprintln!("[text] n_grams={}", w.len());
        }
        w
    } else {
        let file = fs::File::open(input_path)?;
        let reader = io::BufReader::new(file);
        let mut w = Vec::new();
        for line in reader.lines() {
            let line = line?;
            let line = line.trim().to_string();
            if !line.is_empty() {
                w.push(line);
            }
        }
        if verbose {
            eprintln!("[words] count={}", w.len());
        }
        w
    };

    fs::create_dir_all(&output_dir)?;

    // ── Phase 1: 浮现 ──
    if phase_filter.is_none_or(|p| p <= 1) {
        let t1 = Instant::now();
        let phase1 = run_phase_one(words.clone(), max_rounds);
        let dt = t1.elapsed();
        println!(
            "  [Phase 1] nodes={} rounds={} depth={:.1} rules={} constraints={} — {:.2}s",
            phase1.s.vertices.len(),
            phase1.convergence_rounds,
            phase1.max_depth,
            phase1.f.len(),
            phase1.c.len(),
            dt.as_secs_f64()
        );
        for detail in &phase1.reversibility_record {
            println!("    {}", detail);
        }

        if phase_filter == Some(1) {
            println!("\n总耗时: {:.2}s", t_total.elapsed().as_secs_f64());
            return Ok(());
        }

        // ── Phase 2: 编码 ──
        if phase_filter.is_none_or(|p| p <= 2) {
            let t2 = Instant::now();
            let phase2 = run_phase_two(&phase1);
            let dt = t2.elapsed();
            println!(
                "  [Phase 2] cells: {}v+{}e, χ={} β₀={} β₁={}, |∇φ|={:.4}±{:.4} — {:.2}s",
                phase2.complex.n_vertices(),
                phase2.complex.n_edges(),
                phase2.invariants.euler_char,
                phase2.invariants.betti_0,
                phase2.invariants.betti_1,
                phase2.vector_field.mean_gradient(),
                phase2.vector_field.std_gradient(),
                dt.as_secs_f64()
            );

            if phase_filter == Some(2) {
                println!("\n总耗时: {:.2}s", t_total.elapsed().as_secs_f64());
                return Ok(());
            }

            // ── Phase 3: 分解 ──
            if phase_filter.is_none_or(|p| p <= 3) {
                let t3 = Instant::now();
                let phase3 = run_phase_three(&phase2);
                let dt = t3.elapsed();
                println!(
                    "  [Phase 3] memes={} — {:.2}s",
                    phase3.n_memes,
                    dt.as_secs_f64()
                );
                for (i, meme) in phase3.memes.iter().enumerate() {
                    println!(
                        "    M{}: D={:.3} B={:.3} ρ={:.3} R={:.3} S={:.3} | vertices={}",
                        i,
                        meme.five_dim.intrinsic_degree,
                        meme.five_dim.binding_degree,
                        meme.five_dim.energy_density,
                        meme.five_dim.evolution_rate,
                        meme.five_dim.structural_robustness,
                        meme.vertices.len()
                    );
                }
                if verbose && !phase3.coupling.is_empty() {
                    eprintln!(
                        "  [Phase 3] coupling: {}×{}",
                        phase3.coupling.len(),
                        phase3.coupling[0].len()
                    );
                }

                // 往返验证
                let report = verify_roundtrip(&words, &phase1, &phase2, &phase3);
                for detail in &report.details {
                    println!("  [Verify] {}", detail);
                }

                if phase_filter == Some(3) {
                    println!("\n总耗时: {:.2}s", t_total.elapsed().as_secs_f64());
                    return Ok(());
                }

                // ── Phase 4: 固化 ──
                if phase_filter.is_none_or(|p| p <= 4) {
                    let t4 = Instant::now();
                    let original_text = words.join(" ");
                    let phase4 = run_phase_four(phase3.clone(), &original_text);
                    let dt = t4.elapsed();
                    println!(
                        "  [Phase 4] hash={} memes={} bytes={} — {:.2}s",
                        phase4.data_hash,
                        phase4.metadata.meme_count,
                        phase4.metadata.original_size,
                        dt.as_secs_f64()
                    );

                    if phase_filter == Some(4) {
                        println!("\n总耗时: {:.2}s", t_total.elapsed().as_secs_f64());
                        return Ok(());
                    }

                    // ── Phase 5: 演化 ──
                    if phase_filter.is_none_or(|p| p <= 5) {
                        let t5 = Instant::now();
                        let ode_config = OdeConfig {
                            t_max,
                            max_steps,
                            ..OdeConfig::default()
                        };
                        if verbose {
                            eprintln!(
                                "  [ODE] t_max={:.1} max_steps={} window={} thresh={:.4}",
                                ode_config.t_max,
                                ode_config.max_steps,
                                ode_config.convergence_window,
                                ode_config.convergence_threshold
                            );
                        }
                        let opt_config = if auto_optimize {
                            Some(OptimizerConfig::default())
                        } else {
                            None
                        };
                        let phase5 = run_phase_five(&phase3, &ode_config, opt_config.as_ref());
                        let dt = t5.elapsed();

                        if let Some(ref hyp) = phase5.optimal_hypothesis {
                            println!(
                                "  [Phase 5] 假设0: T*={:.3} ΦD*={:?} ΦR*={:?} L={:.6}",
                                hyp.threshold, hyp.family_d, hyp.family_r, hyp.loss
                            );
                        }

                        println!(
                            "  [Phase 5] evolutions={} — {:.2}s",
                            phase5.evolutions.len(),
                            dt.as_secs_f64()
                        );
                        for evo in &phase5.evolutions {
                            if let Some(last) = evo.trajectory.last() {
                                println!("    M{}: t={:.2} steps={} D={:.4} B={:.4} R={:.4} S={:.4} [{:?}] → {:?}",
                                    evo.meme_idx, last.t, evo.trajectory.len(),
                                    last.d, last.b, last.r, last.s,
                                    evo.termination, evo.archetype);
                            } else {
                                eprintln!(
                                    "    [WARN] M{}: 空轨迹 (reason={:?})",
                                    evo.meme_idx, evo.termination
                                );
                            }
                        }

                        // ── 收敛报告（实验零）──
                        let conv_report = evaluate_convergence(&phase5.evolutions, &ode_config);
                        println!(
                            "  [Convergence] rate={:.1}% converged={}/{} std_ok={} undetermined={:.1}% → {}",
                            conv_report.convergence_rate * 100.0,
                            conv_report.converged_count,
                            conv_report.total_memes,
                            conv_report.criteria.std_ok,
                            conv_report.undetermined_pct * 100.0,
                            if conv_report.is_converged {
                                "PASS"
                            } else {
                                "FAIL"
                            }
                        );
                        if !conv_report.is_converged {
                            eprintln!(
                                "  [Convergence] 判据: all_converged={} std_ok={} undet_ok={}",
                                conv_report.criteria.all_converged,
                                conv_report.criteria.std_ok,
                                conv_report.criteria.undetermined_ok
                            );
                        }

                        // 输出 ODE 轨迹 CSV
                        let ode_path =
                            Path::new(&output_dir).join(format!("{}_ode_trajectory.csv", source));
                        let mut of = fs::File::create(&ode_path)?;
                        writeln!(of, "meme_id,t,d,b,rho,r,s,archetype")?;
                        for evo in &phase5.evolutions {
                            for snap in &evo.trajectory {
                                writeln!(
                                    of,
                                    "{},{},{:.8},{:.8},{:.8},{:.8},{:.8},{:?}",
                                    evo.meme_idx,
                                    snap.t,
                                    snap.d,
                                    snap.b,
                                    snap.rho,
                                    snap.r,
                                    snap.s,
                                    evo.archetype
                                )?;
                            }
                        }
                        if verbose {
                            eprintln!("  [Phase 5] CSV → {}", ode_path.display());
                        }
                    }
                }
            }
        }
    }

    println!("\n════════════════════════════════════════════════");
    println!("  总耗时: {:.2}s", t_total.elapsed().as_secs_f64());
    println!("════════════════════════════════════════════════");
    Ok(())
}
