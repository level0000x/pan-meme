use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use n_iter_rs::five_dim;
use n_iter_rs::n_operator::{self, DynamicsParams};
use n_iter_rs::fca;
use n_iter_rs::io;

fn main() {
    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().to_path_buf();
    let extract_dir = base_dir
        .join("experiments")
        .join("009-external-validation")
        .join("data")
        .join("extracts");
    let output_dir = base_dir
        .join("experiments")
        .join("011-fca-n-iteration")
        .join("output");

    let _ = fs::create_dir_all(&output_dir);

    let entries = fs::read_dir(&extract_dir).expect("Failed to read extracts");
    let json_files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "json").unwrap_or(false))
        .collect();

    println!("Found {} Wikipedia extracts", json_files.len());
    let cat_map = io::build_category_map();

    let mut all_articles: Vec<(String, String, String)> = Vec::new();
    for path in &json_files {
        let concept = path.file_stem().unwrap().to_string_lossy().to_string();
        let (_title, text) = match io::read_wikipedia_extract(path) {
            Some(t) => t,
            None => continue,
        };
        let cat = cat_map.get(concept.as_str()).map(|s| s.as_str()).unwrap_or("Other").to_string();
        all_articles.push((concept, text, cat));
    }

    let mut by_cat: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for (name, text, cat) in &all_articles {
        by_cat.entry(cat.clone()).or_default().push((name.clone(), text.clone()));
    }

    println!("Articles by category:");
    for (cat, arts) in by_cat.iter() {
        println!("  {:>15}: {} articles", cat, arts.len());
    }

    let max_attrs: usize = 5000;
    let max_concepts: usize = 5000;
    let time_limit: f64 = 1800.0;

    {
        let mut merged = String::new();
        for (_, text, _) in &all_articles { merged.push_str(text); merged.push(' '); }

        let art_texts: Vec<String> = all_articles.iter().map(|(_, t, _)| t.clone()).collect();
        run_experiment(
            &format!("ALL {} articles (word-level article FCA)", all_articles.len()),
            &merged,
            &art_texts,
            max_attrs,
            max_concepts,
            time_limit,
            &output_dir,
        );

        let tsv_path = output_dir.join("unified_lattice.tsv");
        println!("  TSV output: {}", tsv_path.display());
    }

    for (cat, arts) in &by_cat {
        if arts.len() < 3 { continue; }
        let mut merged = String::new();
        for (_, text) in arts { merged.push_str(text); merged.push(' '); }

        let art_texts: Vec<String> = arts.iter().map(|(_, t)| t.clone()).collect();
        run_experiment(
            &format!("{} ({}) articles", cat, arts.len()),
            &merged,
            &art_texts,
            max_attrs,
            max_concepts,
            time_limit,
            &output_dir,
        );
    }
}

fn run_experiment(label: &str, _merged_text: &str, art_texts: &[String], max_attrs: usize, max_concepts: usize, time_limit: f64, _output_dir: &PathBuf) {
    let bg_chars: usize = art_texts.iter().map(|t| t.chars().count()).sum();

    println!("\n{}\n  {}\n  chars={} n_articles={} attrs={} concepts_limit={} time_limit={:.0}s\n{}",
        "=".repeat(64), label, bg_chars, art_texts.len(), max_attrs, max_concepts, time_limit, "=".repeat(64));

    let start = std::time::Instant::now();
    println!("  Building FCA lattice...");
    let min_arts = 2usize;
    let lattice = fca::build_article_lattice(art_texts, max_attrs, min_arts, max_concepts, time_limit);
    let n = lattice.concepts.len();
    let elapsed = start.elapsed().as_secs_f64();
    println!("  Lattice: {} formal concepts, {} Hasse edges  (built in {:.1}s)",
        n, lattice.edges.len(), elapsed);

    if n < 3 {
        println!("  Too few concepts ({}), skipping.", n);
        return;
    }

    let heights = fca::hasse_heights(n, &lattice.edges);
    let mut feeders: Vec<Vec<usize>> = vec![vec![]; n];
    for &(p, c) in &lattice.edges { feeders[p].push(c); }

    let params = DynamicsParams::uniform();
    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by_key(|&i| std::cmp::Reverse(heights[i]));
    let mut results: Vec<Option<n_operator::IterResult>> = vec![None; n];

    let total_extent: usize = lattice.concept_sizes.iter().map(|(_, b)| *b).sum();
    let total_intent: usize = lattice.concept_sizes.iter().map(|(a, _)| *a).sum();
    let valid_d: Vec<f64> = lattice.d_values.iter()
        .filter(|&&d| d.is_finite() && d < 1e6).copied().collect();
    let max_d_valid = valid_d.iter().cloned().fold(1.0_f64, f64::max);

    println!("  Running N-iteration on {} concepts...", n);
    let n_start = std::time::Instant::now();

    for &ci in &sorted {
        let csize = &lattice.concept_sizes[ci];
        let raw_d = lattice.d_values[ci];
        let d_init = if raw_d.is_finite() && raw_d < 1e6 {
            (raw_d / max_d_valid).min(1.0)
        } else {
            0.8
        };
        let b_init = (1.0 - csize.1 as f64 / total_extent.max(1) as f64).clamp(0.0, 1.0);
        let rho_init = (csize.0 as f64 / total_intent.max(1) as f64).clamp(0.0, 1.0);

        let (b_up, rho_up) = get_upstream(ci, &feeders, &results);
        let m0 = five_dim::make_state(d_init, b_init, rho_init, 0.5, 0.5);
        let result = n_operator::run_iteration(&m0, b_up, rho_up, &params, 500, 1e-12);
        results[ci] = Some(result);
    }

    let n_elapsed = n_start.elapsed().as_secs_f64();

    let d_vals: Vec<f64> = results.iter().filter_map(|r| r.as_ref()).map(|r| r.m_star[0]).collect();
    let all_tau: Vec<f64> = results.iter().filter_map(|r| r.as_ref()).map(|r| r.tau_inv).collect();
    let all_rho: Vec<f64> = results.iter().filter_map(|r| r.as_ref()).map(|r| r.rho_spectral).collect();

    let fca_d_mono = fca::verify_theorem_11_3(&lattice.concepts, &lattice.d_values, &lattice.edges);
    let d_star_mono = fca::verify_theorem_11_3(&lattice.concepts, &d_vals, &lattice.edges);
    let t_mono = fca::verify_theorem_11_1(&all_tau, &lattice.edges);

    let mut max_traj_dev = 0.0_f64;
    for (ci, opt_r) in results.iter().enumerate() {
        if let Some(ref r) = opt_r {
            let (b_up, rho_up) = get_upstream(ci, &feeders, &results);
            if let Some(_d) = r.verify_trajectory_conservation(&params, b_up, rho_up) {
                max_traj_dev = max_traj_dev.max(_d);
            }
        }
    }

    let d_std = std_dev(&d_vals);
    let t_std = std_dev(&all_tau);

    println!("  N-iteration done in {:.1}s", n_elapsed);
    println!();
    println!("  Concepts:             {}", n);
    println!("  Hasse edges:          {}", lattice.edges.len());
    println!("  Max lattice height:   {}", heights.iter().max().unwrap_or(&0));
    println!("  D*  range:            [{:.4}, {:.4}]  std={:.4}",
        d_vals.iter().cloned().fold(f64::INFINITY, f64::min),
        d_vals.iter().cloned().fold(0.0_f64, f64::max), d_std);
    println!("  τ⁻¹ range:            [{:.4}, {:.4}]  std={:.4}",
        all_tau.iter().cloned().fold(f64::INFINITY, f64::min),
        all_tau.iter().cloned().fold(0.0_f64, f64::max), t_std);
    println!("  ρ(J_N) range:         [{:.4}, {:.4}]",
        all_rho.iter().cloned().fold(f64::INFINITY, f64::min),
        all_rho.iter().cloned().fold(0.0_f64, f64::max));

    println!();
    println!("  Theorem 11.3  (FCA D₀=|A|/|B|)      {}/{} = {:.1}%",
        fca_d_mono.0, fca_d_mono.0 + fca_d_mono.1,
        pct(fca_d_mono.0, fca_d_mono.0 + fca_d_mono.1));
    println!("  Empirical      (N-iter D*)          {}/{} = {:.1}%",
        d_star_mono.0, d_star_mono.0 + d_star_mono.1,
        pct(d_star_mono.0, d_star_mono.0 + d_star_mono.1));
    println!("  Theorem 11.1  (τ⁻¹ monotonicity)    {}/{} = {:.1}%",
        t_mono.0, t_mono.0 + t_mono.1,
        pct(t_mono.0, t_mono.0 + t_mono.1));
    println!("  Theorem 6.17  (trajectory)           max dev = {:.2e}", max_traj_dev);

    let mut height_stats: HashMap<usize, Vec<f64>> = HashMap::new();
    for (ci, opt_r) in results.iter().enumerate() {
        if let Some(ref r) = opt_r {
            height_stats.entry(heights[ci]).or_default().push(r.tau_inv);
        }
    }
    let mut hs: Vec<usize> = height_stats.keys().copied().collect();
    hs.sort();
    println!("\n  τ⁻¹ by lattice height:");
    for h in &hs {
        let vals = &height_stats[h];
        if vals.is_empty() { continue; }
        let min_v = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_v = vals.iter().cloned().fold(0.0_f64, f64::max);
        let avg_v = vals.iter().sum::<f64>() / vals.len() as f64;
        println!("    h={}: τ⁻¹∈[{:.4},{:.4}] avg={:.4}  n={}", h, min_v, max_v, avg_v, vals.len());
    }

    println!("\n  M* (per concept)  D₀=|A|/|B| (FCA) vs D* (N-iter):");
    let mut by_h: Vec<(usize, Vec<usize>)> = Vec::new();
    for h in &hs { by_h.push((*h, vec![])); }
    for (ci, opt_r) in results.iter().enumerate() {
        if opt_r.is_some() {
            for (h, v) in by_h.iter_mut() {
                if heights[ci] == *h { v.push(ci); break; }
            }
        }
    }
    for (h, cis) in &by_h {
        for &ci in cis {
            if let Some(ref r) = results[ci] {
                let m = &r.m_star;
                let d0 = lattice.d_values[ci];
                let (b_up, _rho_up) = get_upstream(ci, &feeders, &results);
                let d0_str = if d0.is_finite() && d0 < 1e6 {
                    format!("{:.4}→{:.4}", d0 / max_d_valid.min(1.0).max(0.01), m[0])
                } else {
                    format!("∞→{:.4}", m[0])
                };
                println!("    C{:<3} h={} D={} B*={:.4} ρ*={:.4} R*={:.4} S*={:.4} B↑={:.4} τ⁻¹={:.4} ρ_J={:.4}",
                    ci, h, d0_str, m[1], m[2], m[3], m[4], b_up, r.tau_inv, r.rho_spectral);
            }
        }
    }

    println!("\n  N_D formula at M*: D* ≈ (R+ε)/(R+B+B_up+ε)  (self-sufficiency ratio)");
    for (h, cis) in &by_h {
        for &ci in cis {
            if let Some(ref r) = results[ci] {
                let m = &r.m_star;
                let (b_up, _rho_up) = get_upstream(ci, &feeders, &results);
                let ratio = (m[3] + 0.01) / (m[3] + m[1] + b_up + 0.01);
                println!("    C{:<3} h={} R*={:.4} B*+B↑={:.4} ratio={:.4} vs D*={:.4}",
                    ci, h, m[3], m[1] + b_up, ratio, m[0]);
            }
        }
    }

    t_mono_verdict(t_mono.0, t_mono.0 + t_mono.1);

    println!("  FCA D₀=|A|/|B|:  mean={:.4}  std={:.4}  range=({:.4}, {:.4})",
        mean(&lattice.d_values.iter().filter(|&&d| d.is_finite() && d < 1e6).copied().collect::<Vec<_>>()),
        std_dev(&lattice.d_values.iter().filter(|&&d| d.is_finite() && d < 1e6).copied().collect::<Vec<_>>()),
        lattice.d_values.iter().filter(|&&d| d.is_finite() && d < 1e6).fold(f64::INFINITY, |a, &b| a.min(b)),
        lattice.d_values.iter().filter(|&&d| d.is_finite() && d < 1e6).fold(0.0_f64, |a, &b| a.max(b)));
    println!("  N-iter D*:        mean={:.4}  std={:.4}  range=({:.4}, {:.4})",
        mean(&d_vals), d_std,
        d_vals.iter().cloned().fold(f64::INFINITY, f64::min),
        d_vals.iter().cloned().fold(0.0_f64, f64::max));
    println!("  B* values:        mean={:.4}  std={:.4}",
        mean(&results.iter().filter_map(|r| r.as_ref()).map(|r| r.m_star[1]).collect::<Vec<_>>()),
        std_dev(&results.iter().filter_map(|r| r.as_ref()).map(|r| r.m_star[1]).collect::<Vec<_>>()));

    let d0_to_dstar: Vec<f64> = results.iter().enumerate().filter_map(|(ci, r)| {
        let d0 = lattice.d_values[ci];
        if d0.is_finite() && d0 < 1e6 {
            r.as_ref().map(|rr| rr.m_star[0] / (d0 / max_d_valid).max(0.01))
        } else { None }
    }).collect();
    if !d0_to_dstar.is_empty() {
        println!("  D*/D₀ ratio:      mean={:.4}  std={:.4}  range=({:.4}, {:.4})",
            mean(&d0_to_dstar), std_dev(&d0_to_dstar),
            d0_to_dstar.iter().cloned().fold(f64::INFINITY, f64::min),
            d0_to_dstar.iter().cloned().fold(0.0_f64, f64::max));
    }
}

fn get_upstream(ci: usize, feeders: &[Vec<usize>], results: &[Option<n_operator::IterResult>]) -> (f64, f64) {
    let mut b_up = 0.0;
    let mut rho_up = 0.0;
    if !feeders[ci].is_empty() {
        let mut sums = (0.0, 0.0);
        let mut cnt = 0;
        for &f_idx in &feeders[ci] {
            if let Some(Some(ref r)) = results.get(f_idx) {
                sums.0 += r.m_star[1];
                sums.1 += r.m_star[2];
                cnt += 1;
            }
        }
        if cnt > 0 { b_up = sums.0 / cnt as f64; rho_up = sums.1 / cnt as f64; }
    }
    (b_up, rho_up)
}

fn mean(vals: &[f64]) -> f64 {
    if vals.is_empty() { return 0.0; }
    vals.iter().sum::<f64>() / vals.len() as f64
}

fn std_dev(vals: &[f64]) -> f64 {
    let n = vals.len() as f64;
    if n < 2.0 { return 0.0; }
    let avg = mean(vals);
    let var = vals.iter().map(|x| (x - avg).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

fn pct(passes: usize, total: usize) -> f64 {
    if total == 0 { 0.0 } else { 100.0 * passes as f64 / total as f64 }
}

fn t_mono_verdict(passes: usize, total: usize) {
    let p = pct(passes, total);
    println!();
    if p >= 99.0 {
        println!("  VERDICT: Theorem 11.1 CONFIRMED — uniform params, τ⁻¹≥99%");
    } else if p >= 90.0 {
        println!("  VERDICT: Theorem 11.1 LARGELY CONFIRMED — τ⁻¹≥90%");
    } else if p >= 70.0 {
        println!("  VERDICT: Partial — τ⁻¹={:.1}% (theory ~74% with per-concept params)", p);
    } else if p > 0.0 {
        println!("  VERDICT: Weak — τ⁻¹={:.1}% ({}/{})", p, passes, total);
    } else {
        println!("  VERDICT: No evidence for Theorem 11.1 — τ⁻¹={:.1}%", p);
    }
}
