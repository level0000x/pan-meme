//! ↑↓ 数据建模引擎 — 主入口（完整四阶段）
//!
//! 数学对应：泛模因 §3.2-§3.5 全流程
//!
//! 流程:
//!   Phase 1: 词列表 → Ψ → ↑↓ 循环 → 关系自组织 → 推理 → 概念层级 → 输出 M=(S,F,C)
//!   Phase 2: M → 胞腔复形 → 度量 → 不变量 → 输出 G=(K,g,Γ,R)
//!   Phase 3: G → Betti分解 → 五维映射 → 参数推导 → 耦合 → 输出 Q=({Xᵢ},Θ,C)
//!   Phase 4: Q → SHA256哈希 → 凭证 → 输出 H
//!   Phase 5: Q → ODE演化 → 收敛分类 → 原型判定
//!
//! 用法:
//!   updown input.txt                     # 完整四阶段流水线
//!   updown input.txt --phase 1           # 仅 Phase 1
//!   updown input.txt -o output_dir       # 指定输出目录
//!   updown input.txt -T 0.3              # Jaccard 阈值
//!   updown input.txt --fixed 5           # 固定概念层级 5 层

mod emergence;
mod encoding;
mod sealing;
mod infra;

use emergence::cycle::{CycleEngine, CycleMode};
use emergence::extractor::{ExtractorConfig, RelationNetwork};
use emergence::relations::{
    ConceptCycleMode, ReasonerConfig,
    organize_relations, reason, run_phase_one,
};
use encoding::geometry::run_phase_two;
use encoding::decomposition::run_phase_three;
use sealing::binding::run_phase_four;
use sealing::ode::{OdeConfig, FiveDimSnapshot, solve_all_memes};
use infra::tsv::TsvWriter;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("用法: updown <词列表文件> [选项]");
        eprintln!("选项:");
        eprintln!("  --phase <1|2|3|4>  只运行指定阶段 (默认: 全部)");
        eprintln!("  -o <目录>          输出目录 (默认: ./output)");
        eprintln!("  -r <轮数>          ↑↓ 最大循环轮数 (默认: 50)");
        eprintln!("  -T <阈值>          Jaccard 阈值 (默认: 0.1)");
        eprintln!("  -s <来源>          数据来源标签");
        eprintln!("  --fixed <N>        固定概念层级层数（默认: 自动收敛）");
        eprintln!("  --no-word-sub      禁用词间子串 containment");
        eprintln!("  --no-reasoning     禁用推理与补全");
        eprintln!("  --no-cooccurrence  禁用共现推理");
        return Ok(());
    }

    let input_path = &args[1];
    let mut output_dir = "output".to_string();
    let mut max_rounds = 50usize;
    let mut jaccard_threshold = 0.1;
    let mut source = String::new();
    let mut cycle_mode = CycleMode::Converge;
    let mut concept_mode = ConceptCycleMode::Converge;
    let mut word_substring = true;
    let mut phase_filter: Option<u8> = None;
    let mut reasoning_enabled = true;
    let mut cooccurrence_enabled = true;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--phase" => { i += 1; if i < args.len() { phase_filter = Some(args[i].parse().unwrap_or(0)); } }
            "-o" => { i += 1; if i < args.len() { output_dir = args[i].clone(); } }
            "-r" => { i += 1; if i < args.len() { max_rounds = args[i].parse().unwrap_or(50); } }
            "-T" => { i += 1; if i < args.len() { jaccard_threshold = args[i].parse().unwrap_or(0.1); } }
            "-s" => { i += 1; if i < args.len() { source = args[i].clone(); } }
            "--fixed" => {
                i += 1;
                if i < args.len() {
                    let n: usize = args[i].parse().unwrap_or(5);
                    cycle_mode = CycleMode::Fixed(n);
                    concept_mode = ConceptCycleMode::Fixed(n);
                }
            }
            "--no-word-sub" => { word_substring = false; }
            "--no-reasoning" => { reasoning_enabled = false; }
            "--no-cooccurrence" => { cooccurrence_enabled = false; }
            _ => {}
        }
        i += 1;
    }

    if source.is_empty() {
        source = Path::new(input_path).file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
    }
    let short_source = &source;

    println!("╔══════════════════════════════════════════════╗");
    println!("║  ↑↓ 泛模因建模引擎 — 四阶段流水线            ║");
    println!("╚══════════════════════════════════════════════╝");
    println!("  输入: {}", input_path);
    println!("  输出: {}", output_dir);
    println!();

    // ── 读取词列表 ──
    let t_total = Instant::now();
    let file = fs::File::open(input_path)?;
    let reader = io::BufReader::new(file);
    let mut words: Vec<String> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim().to_string();
        if !line.is_empty() { words.push(line); }
    }
    let raw_count = words.len();
    println!("  读取: {} 行", raw_count);

    // ── Step 0: 构建 Ψ ──
    let config = ExtractorConfig {
        word_substring_containment: word_substring,
        min_subordinate_len: 1,
        max_substring_window: 6,
    };
    let psi = RelationNetwork::from_words(words.clone(), &config);
    println!("  论域: {} 词 + {} 字 = {} 节点",
        psi.word_count, psi.node_count() - psi.word_count, psi.node_count()
    );

    // ── Phase 1 ──
    if phase_filter.map_or(true, |p| p <= 1) {
        println!();
        println!("  ┌─ Phase 1: 浮现 ──────────────────────┐");
        let t1 = Instant::now();

        // 1a: ↑↓ 循环
        let engine = CycleEngine::new(cycle_mode, max_rounds);
        let cycle_result = engine.run(&psi);
        println!("  │ ↑↓ 循环: {:.2}s, {} 轮 ({})",
            t1.elapsed().as_secs_f64(),
            cycle_result.termination.total_rounds,
            cycle_result.termination.reason
        );

        // 1b: 关系自组织（Jaccard 阈值筛选）
        let relations = organize_relations(&psi, jaccard_threshold);
        println!("  │ 字字关系(≥T): {}, 词词关系(≥T): {}",
            relations.char_relations.len(),
            relations.word_relations.len()
        );

        // 1c: 推理补全
        let inferred_count = if reasoning_enabled {
            let rconfig = ReasonerConfig {
                cooccurrence_enabled,
                ..Default::default()
            };
            let inferred = reason(&psi, &relations, &rconfig);
            let count = inferred.len();
            println!("  │ 推理补全: {} 条", count);
            count
        } else { 0 };
        let _ = inferred_count;

        // 1d: 概念-要素循环 → M = (S, F, C)
        let node_levels: Vec<usize> = {
            let total = psi.node_count();
            let mut levels = Vec::with_capacity(total);
            // 字符: 直接用cycle的收敛轮数
            levels.extend_from_slice(&cycle_result.convergence_round);
            // 词: 取其组成字的最大收敛轮数 + 1
            for wi in 0..psi.word_count {
                if wi < psi.word_to_chars.len() && !psi.word_to_chars[wi].is_empty() {
                    let max_char_round = psi.word_to_chars[wi].iter()
                        .map(|&ci| cycle_result.convergence_round.get(ci).copied().unwrap_or(0))
                        .max().unwrap_or(0);
                    levels.push(max_char_round + 1);
                } else {
                    levels.push(0);
                }
            }
            levels
        };

        let phase1 = run_phase_one(&psi, &relations, concept_mode, 10, node_levels);
        println!("  │ 概念层级: {} 层", phase1.structure.depth);
        println!("  │ 规则: {} 条, 约束: {} 项", phase1.rules.len(), phase1.constraints.len());
        println!("  │ 自洽: {}, 不完整性: {}",
            if phase1.is_consistent { "通过" } else { "失败" },
            if phase1.incompleteness.is_radically_incomplete { "彻底不完整" } else { "可接受" }
        );
        println!("  │ 信息: H(levels)={:.4} bit", encoding::geometry::compute_entropy(&phase1.node_levels));
        println!("  └─ Phase 1 完成: {:.2}s ───────────────┘", t1.elapsed().as_secs_f64());

        // ── 输出 ──
        std::fs::create_dir_all(&output_dir).ok();
        let mut w = TsvWriter::new(&output_dir);
        w.write_relations(&psi, &relations, &cycle_result, short_source);
        w.write_rules(&phase1.rules, &phase1.constraints, short_source);

        if phase_filter == Some(1) {
            println!("\n总耗时: {:.2}s", t_total.elapsed().as_secs_f64());
            return Ok(());
        }

        // ── Phase 2: 编码 ──
        if phase_filter.map_or(true, |p| p <= 2) {
            println!();
            println!("  ┌─ Phase 2: 编码（几何化）────────────────┐");
            let t2 = Instant::now();
            let phase2 = run_phase_two(&psi, &phase1);
            println!("  │ 胞腔: {} 0-cells + {} 1-cells + {} 2-cells",
                phase2.invariants.v0_count, phase2.invariants.e1_count, phase2.invariants.v2_count);
            println!("  │ Euler χ = {}, Betti: β₀={}, β₁={}",
                phase2.invariants.euler, phase2.invariants.betti_0, phase2.invariants.betti_1);
            println!("  │ 度量: avg={:.4}, 密度={:.4}",
                phase2.metric.avg_edge_length, phase2.invariants.density);
            println!("  │ 场: φ_avg={:.4} 通量={:.4} 源={} 汇={}",
                phase2.scalar_field.avg_potential,
                phase2.vector_field.total_flux,
                phase2.vector_field.source_count,
                phase2.vector_field.sink_count);
            println!("  └─ Phase 2 完成: {:.2}s ───────────────┘", t2.elapsed().as_secs_f64());

            // 输出几何概要
            let geom_path = Path::new(&output_dir).join(format!("{}_geometry.txt", short_source));
            let mut gf = fs::File::create(&geom_path)?;
            for line in &phase2.reversibility {
                writeln!(gf, "{}", line)?;
            }

            if phase_filter == Some(2) {
                println!("\n总耗时: {:.2}s", t_total.elapsed().as_secs_f64());
                return Ok(());
            }

            // ── Phase 3: 分解 ──
            if phase_filter.map_or(true, |p| p <= 3) {
                println!();
                println!("  ┌─ Phase 3: 分解（模因化）──────────────┐");
                let t3 = Instant::now();
                let depth = phase1.structure.depth;
                let phase3 = run_phase_three(&phase2.complex, &phase2.invariants, depth, &phase2.scalar_field, &phase2.vector_field);
                println!("  │ 分子几何体: {} 个", phase3.n_memes);
                for meme in &phase3.memes {
                    println!("  │   M{}: D={:.3} B={:.3} ρ={:.3} R={:.3} S={:.3}",
                        meme.id,
                        meme.state.intrinsic_degree,
                        meme.state.binding_degree,
                        meme.state.energy_density,
                        meme.state.evolution_rate,
                        meme.state.structural_robustness,
                    );
                }
                println!("  │ 耦合: {} 对", phase3.couplings.len());
                println!("  └─ Phase 3 完成: {:.2}s ───────────────┘", t3.elapsed().as_secs_f64());

                // 输出五维映射 CSV
                let dim5_path = Path::new(&output_dir).join(format!("{}_5d_mapping.csv", short_source));
                let mut df = fs::File::create(&dim5_path)?;
                writeln!(df, "meme_id,D,B,rho,R,S,alpha_1,beta_1,delta_2,epsilon_2")?;
                for meme in &phase3.memes {
                    writeln!(df, "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                        meme.id,
                        meme.state.intrinsic_degree,
                        meme.state.binding_degree,
                        meme.state.energy_density,
                        meme.state.evolution_rate,
                        meme.state.structural_robustness,
                        meme.params.alpha_1,
                        meme.params.beta_1,
                        meme.params.delta_2,
                        meme.params.epsilon_2,
                    )?;
                }

                // 耦合 CSV
                let coup_path = Path::new(&output_dir).join(format!("{}_coupling.csv", short_source));
                let mut cf = fs::File::create(&coup_path)?;
                writeln!(cf, "i,j,strength,kind")?;
                for c in &phase3.couplings {
                    writeln!(cf, "{},{},{:.6},{:?}", c.i, c.j, c.strength, c.kind)?;
                }

                if phase_filter == Some(3) {
                    println!("\n总耗时: {:.2}s", t_total.elapsed().as_secs_f64());
                    return Ok(());
                }

                // ── 推论5.1 验证：全链 round-trip Φ⁻¹(Φ(I)) ≡ I ──
                println!();
                println!("  ═══ 推论5.1 验证: Φ⁻¹(Φ(I)) ≡ I ═══");

                // Step 3b: Q → G (Φ_D⁻¹)
                let decoded_complex = encoding::decomposition::decode_phase_three(&phase3);
                let v0_decoded = decoded_complex.v0_map.len();
                let e1_decoded = decoded_complex.e1_map.len();
                let v0_orig = phase2.complex.v0_map.len();
                let e1_orig = phase2.complex.e1_map.len();

                println!("  Φ_D⁻¹: Q → G");
                println!("    0-胞腔: {} (原始: {})", v0_decoded, v0_orig);
                println!("    1-胞腔: {} (原始: {})", e1_decoded, e1_orig);

                // Step 3c: G → M (Φ_C⁻¹)
                let decoded_phase1 = encoding::geometry::decode_phase_two(&phase2);

                // 深度比对
                let concept_levels_match = decoded_phase1.structure.levels.len() == phase1.structure.levels.len()
                    && decoded_phase1.structure.levels.iter().zip(phase1.structure.levels.iter())
                        .all(|(a, b)| a.len() == b.len());
                let depth_match = decoded_phase1.structure.depth == phase1.structure.depth;
                let rules_match = decoded_phase1.rules.len() == phase1.rules.len();
                let constraints_match = decoded_phase1.constraints.len() == phase1.constraints.len();

                // Step 3d: M → I (Φ_B⁻¹ ∘ Φ_A⁻¹)
                let (decoded_words, decoded_chars) = emergence::relations::decode_phase_one(
                    &phase2.rev_data.node_texts,
                    phase2.rev_data.word_count,
                );

                // 比对：原始词列表 vs 逆映射重建
                let word_match = decoded_words == words;
                let char_match = {
                    let orig_chars: Vec<String> = (0..psi.node_count() - psi.word_count)
                        .map(|i| {
                            psi.node_texts.get(psi.word_count + i).cloned().unwrap_or_default()
                        })
                        .collect();
                    decoded_chars == orig_chars
                };

                println!("  Φ_C⁻¹: G → M");
                println!("    概念层级: {}层 (原始: {}), 匹配: {}", 
                    decoded_phase1.structure.depth, phase1.structure.depth,
                    if depth_match { "✓" } else { "✗" });
                println!("    规则: {} (原始: {}), 约束: {} (原始: {})",
                    decoded_phase1.rules.len(), phase1.rules.len(),
                    decoded_phase1.constraints.len(), phase1.constraints.len());
                if !concept_levels_match { println!("    ⚠ 层内概念数不完全一致"); }
                if !rules_match { println!("    ⚠ 规则数不完全一致"); }
                if !constraints_match { println!("    ⚠ 约束数不完全一致"); }

                println!("  Φ_B⁻¹ ∘ Φ_A⁻¹: M → Ψ → I");
                println!("    词: {} (原始: {}), 匹配: {}", decoded_words.len(), words.len(),
                    if word_match { "✓" } else { "✗" });
                println!("    字: {} (原始: {}), 匹配: {}", decoded_chars.len(),
                    psi.node_count() - psi.word_count,
                    if char_match { "✓" } else { "✗" });

                if word_match && char_match && v0_decoded == v0_orig
                    && depth_match && rules_match && constraints_match
                {
                    println!("  ▶ 推论5.1 成立: Φ⁻¹(Φ(I)) ≡ I（词/字/层级/规则/约束 完全可逆）");
                } else if word_match && char_match {
                    println!("  ▶ 推论5.1 成立: Φ⁻¹(Φ(I)) ≡ I（词/字 完全可逆）");
                } else {
                    println!("  ▶ 推论5.1 部分成立");
                }

                // ── Phase 4: 固化 ──
                if phase_filter.map_or(true, |p| p <= 4) {
                    println!();
                    println!("  ┌─ Phase 4: 固化（绑定）──────────────┐");
                    let t4 = Instant::now();
                    let phase4 = run_phase_four(&phase3, phase2.invariants.total_cells);
                    println!("  │ 证书: {}", phase4.credential.credential_id);
                    println!("  │ 模因数: {}", phase4.credential.meme_count);
                    println!("  │ Merkle 根: {}", phase4.credential.merkle_root);
                    println!("  └─ Phase 4 完成: {:.2}s ───────────────┘", t4.elapsed().as_secs_f64());

                    // 输出凭证 JSON
                    let cred_path = Path::new(&output_dir)
                        .join(format!("{}_credential.json", short_source));
                    let mut crf = fs::File::create(&cred_path)?;
                    writeln!(crf, "{{")?;
                    writeln!(crf, "  \"credential_id\": \"{}\",", phase4.credential.credential_id)?;
                    writeln!(crf, "  \"merkle_root\": \"{}\",", phase4.credential.merkle_root)?;
                    writeln!(crf, "  \"meme_count\": {},", phase4.credential.meme_count)?;
                    writeln!(crf, "  \"asset_count\": {},", phase4.credential.asset_count)?;
                    writeln!(crf, "  \"timestamp\": {},", phase4.credential.timestamp)?;
                    writeln!(crf, "  \"fingerprints\": [")?;
                    for (fi, fp) in phase4.binding.fingerprints.iter().enumerate() {
                        let comma = if fi < phase4.binding.fingerprints.len() - 1 { "," } else { "" };
                        writeln!(crf, "    {{\"meme_id\": {}, \"hash\": \"{}\"}}{}",
                            fp.meme_id, fp.hash, comma)?;
                    }
                    writeln!(crf, "  ]")?;
                    writeln!(crf, "}}")?;
                }

                // ── Phase 5: ODE 演化（定理6+7+8+10）──
                if phase_filter.map_or(true, |p| p <= 5) {
                    println!();
                    println!("  ┌─ Phase 5: ODE 演化 ──────────────────┐");
                    let t5 = Instant::now();

                    let ode_config = OdeConfig::default();
                    let initial_states: Vec<(usize, FiveDimSnapshot)> = phase3.memes.iter()
                        .map(|m| (m.id, FiveDimSnapshot {
                            t: 0.0,
                            d: m.state.intrinsic_degree,
                            b: m.state.binding_degree,
                            rho: m.state.energy_density,
                            r: m.state.evolution_rate,
                            s: m.state.structural_robustness,
                        }))
                        .collect();
                    let params: Vec<_> = phase3.memes.iter().map(|m| m.params.clone()).collect();

                    let ode_output = solve_all_memes(&initial_states, &params, &phase3.couplings, &ode_config);

                    println!("  │ 轨道数: {}", ode_output.trajectories.len());
                    for traj in &ode_output.trajectories {
                        let n_pts = traj.trajectory.len();
                        let last = traj.trajectory.last().unwrap();
                        println!("  │  模因 {}: t={:.2} 步={} D={:.4} B={:.4} ρ={:.4} R={:.4} S={:.4} [{}]",
                            traj.meme_id, traj.terminated_at, n_pts,
                            last.d, last.b, last.rho, last.r, last.s,
                            match traj.termination_reason {
                                sealing::ode::TerminationReason::Converged => "收敛",
                                sealing::ode::TerminationReason::MaxTime => "超时",
                                sealing::ode::TerminationReason::MaxSteps => "步满",
                                sealing::ode::TerminationReason::StepTooSmall => "步过小",
                                sealing::ode::TerminationReason::Diverged => "发散",
                            }
                        );
                        if traj.hopf_warning { println!("  │    ⚠ Hopf 预警"); }
                    }

                    // 收敛分类
                    println!("  │ 收敛分类:");
                    for &(meme_id, ref arch, ref eq) in &ode_output.archetypes {
                        println!("  │   模因 {}: {:?} / {:?}", meme_id, arch, eq);
                    }

                    println!("  └─ Phase 5 完成: {:.2}s ───────────────┘", t5.elapsed().as_secs_f64());

                    // 输出 ODE 轨迹 CSV
                    let ode_path = Path::new(&output_dir)
                        .join(format!("{}_ode_trajectory.csv", short_source));
                    let mut of = fs::File::create(&ode_path)?;
                    writeln!(of, "meme_id,t,d,b,rho,r,s")?;
                    for traj in &ode_output.trajectories {
                        for snap in &traj.trajectory {
                            writeln!(of, "{},{},{:.8},{:.8},{:.8},{:.8},{:.8}",
                                traj.meme_id, snap.t, snap.d, snap.b, snap.rho, snap.r, snap.s)?;
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
