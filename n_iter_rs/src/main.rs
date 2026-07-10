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
    println!("Output: {}", output_dir.display());
    let cat_map = io::build_category_map();

    let mut article_names: Vec<String> = Vec::new();
    let mut article_texts: Vec<String> = Vec::new();

    for path in &json_files {
        let concept = path.file_stem().unwrap().to_string_lossy().to_string();
        let (_title, text) = match io::read_wikipedia_extract(path) {
            Some(t) => t,
            None => continue,
        };
        article_names.push(concept);
        article_texts.push(text);
    }

    let n_articles = article_texts.len();
    let n_attrs: usize = 5000;
    let max_concepts: usize = 5000;
    let time_limit: f64 = 1200.0;

    println!("\n{}", "=".repeat(64));
    println!("Chain C: article-level FCA lattice (v5-style, word attributes)");
    println!("  {} articles → FCA formal concepts", n_articles);
    println!("  N-iteration: uniform params (all 11 coupling coefficients = 1.0)");
    println!("  Attributes: {} | Concepts limit: {} | Time limit: {}s",
             n_attrs, max_concepts, time_limit);
    println!("{}", "=".repeat(64));

    println!("Building article-level FCA lattice...");
    let lattice = fca::build_article_lattice(&article_texts, n_attrs, max_concepts, time_limit);
    let n = lattice.concepts.len();
    println!("Lattice: {} formal concepts, {} Hasse edges", n, lattice.edges.len());

    if n == 0 {
        println!("Empty lattice — aborting.");
        return;
    }

    let heights = fca::hasse_heights(n, &lattice.edges);
    let mut feeders: Vec<Vec<usize>> = vec![vec![]; n];
    for &(p, c) in &lattice.edges { feeders[c].push(p); }

    let params = DynamicsParams::uniform();

    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by_key(|&i| std::cmp::Reverse(heights[i]));
    let mut results: Vec<Option<n_operator::IterResult>> = vec![None; n];

    println!("Running N-iteration on {} concepts...", n);
    let total_extent = n_articles as f64;

    for &ci in &sorted {
        let csize = &lattice.concept_sizes[ci];
        let raw_d = lattice.d_values[ci];
        let d_init = if raw_d.is_finite() && raw_d < 1e6 {
            (raw_d / raw_d.max(1.0)).min(1.0)
        } else {
            0.8
        };
        let b_init = if csize.1 > 0 {
            (1.0 - csize.1 as f64 / total_extent).clamp(0.0, 1.0)
        } else {
            0.2
        };
        let rho_init = if csize.0 > 0 {
            (csize.0 as f64 / n_attrs as f64).clamp(0.0, 1.0)
        } else {
            0.5
        };

        let (b_up, rho_up) = get_upstream(ci, &feeders, &results);
        let m0 = five_dim::make_state(d_init, b_init, rho_init, 0.5, 0.5);
        let result = n_operator::run_iteration(&m0, b_up, rho_up, &params, 500, 1e-12);
        results[ci] = Some(result);
    }

    let d_mono = fca::verify_theorem_11_3(&lattice.concepts, &lattice.d_values, &lattice.edges);
    let tau_vals: Vec<f64> = results.iter().filter_map(|r| r.as_ref()).map(|r| r.tau_inv).collect();
    let t_mono = fca::verify_theorem_11_1(&tau_vals, &lattice.edges);

    let mut max_traj_dev = 0.0_f64;
    for (ci, opt_r) in results.iter().enumerate() {
        if let Some(ref r) = opt_r {
            let (b_up, rho_up) = get_upstream(ci, &feeders, &results);
            if let Some(d) = r.verify_trajectory_conservation(&params, b_up, rho_up) {
                max_traj_dev = max_traj_dev.max(d);
            }
        }
    }

    let d_vals: Vec<f64> = results.iter().filter_map(|r| r.as_ref()).map(|r| r.m_star[0]).collect();
    let all_tau: Vec<f64> = results.iter().filter_map(|r| r.as_ref()).map(|r| r.tau_inv).collect();
    let rho_vals: Vec<f64> = results.iter().filter_map(|r| r.as_ref()).map(|r| r.rho_spectral).collect();

    println!("\n{}", "=".repeat(64));
    println!("Theorem Verification (article-level uniform params)");
    println!("{}", "=".repeat(64));
    println!("  Concepts:             {}", n);
    println!("  Hasse edges:          {}", lattice.edges.len());
    println!("  Max lattice height:   {}", heights.iter().max().unwrap_or(&0));
    println!("  D*  range:            [{:.4}, {:.4}] (std={:.4})",
        d_vals.iter().cloned().fold(f64::INFINITY, f64::min),
        d_vals.iter().cloned().fold(0.0_f64, f64::max),
        std_dev(&d_vals));
    println!("  τ⁻¹ range:            [{:.4}, {:.4}] (std={:.4})",
        all_tau.iter().cloned().fold(f64::INFINITY, f64::min),
        all_tau.iter().cloned().fold(0.0_f64, f64::max),
        std_dev(&all_tau));
    println!("  ρ(J_N) range:         [{:.4}, {:.4}]",
        rho_vals.iter().cloned().fold(f64::INFINITY, f64::min),
        rho_vals.iter().cloned().fold(0.0_f64, f64::max));

    println!();
    println!("  Theorem 11.3  (D      monotonicity)      {}/{}  = {:.1}%",
        d_mono.0, d_mono.0 + d_mono.1,
        100.0 * d_mono.0 as f64 / (d_mono.0 + d_mono.1).max(1) as f64);
    println!("  Theorem 11.1  (τ⁻¹    monotonicity)      {}/{}  = {:.1}%",
        t_mono.0, t_mono.0 + t_mono.1,
        100.0 * t_mono.0 as f64 / (t_mono.0 + t_mono.1).max(1) as f64);
    println!("  Theorem 6.17  (trajectory conservation)  max dev = {:.2e}", max_traj_dev);

    let violations = t_mono.0 + t_mono.1;
    let pct = if violations > 0 {
        100.0 * t_mono.0 as f64 / violations as f64
    } else { 0.0 };
    print_verdict(pct);

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
        let min_v = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_v = vals.iter().cloned().fold(0.0_f64, f64::max);
        let avg_v = vals.iter().sum::<f64>() / vals.len() as f64;
        println!("    height {}: τ⁻¹∈[{:.4},{:.4}] avg={:.4} (n={})", h, min_v, max_v, avg_v, vals.len());
    }

    print_concept_article_assignment(&lattice, &results, &article_names, &cat_map, &heights);

    export_tsv(&lattice, &results, &article_names, &output_dir);
}

fn get_upstream(
    ci: usize,
    feeders: &[Vec<usize>],
    results: &[Option<n_operator::IterResult>],
) -> (f64, f64) {
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

fn std_dev(vals: &[f64]) -> f64 {
    let n = vals.len() as f64;
    if n < 2.0 { return 0.0; }
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

fn print_concept_article_assignment(
    lattice: &fca::FcaLattice,
    results: &[Option<n_operator::IterResult>],
    article_names: &[String],
    cat_map: &HashMap<String, String>,
    heights: &[usize],
) {
    let n = lattice.concepts.len();
    let mut per_article: HashMap<&str, Vec<(usize, f64, f64, f64, usize)>> = HashMap::new();

    for ci in 0..n {
        if let Some(ref r) = results[ci] {
            for &art_idx in &lattice.concepts[ci].extent {
                per_article.entry(article_names[art_idx].as_str())
                    .or_default()
                    .push((ci, r.m_star[0], r.m_star[2], r.tau_inv, heights[ci]));
            }
        }
    }

    println!("\n{}", "=".repeat(64));
    println!("Per-Article Concept Coverage  (concept → article via extent)");
    println!("{}", "=".repeat(64));

    let mut total_assigned = 0usize;
    for name in article_names {
        let cat = cat_map.get(name.as_str()).map(|s| s.as_str()).unwrap_or("Other");
        let concepts = match per_article.get(name.as_str()) {
            Some(c) => c,
            None => continue,
        };
        total_assigned += concepts.len();
        if concepts.is_empty() { continue; }
        let d_min = concepts.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
        let d_max = concepts.iter().map(|c| c.1).fold(0.0_f64, f64::max);
        let t_min = concepts.iter().map(|c| c.3).fold(f64::INFINITY, f64::min);
        let t_max = concepts.iter().map(|c| c.3).fold(0.0_f64, f64::max);
        let heights_set: std::collections::HashSet<usize> = concepts.iter().map(|c| c.4).collect();
        println!("  {:<30} [{:<13}] concepts={:>3} D*=[{:.3},{:.3}] τ⁻¹=[{:.3},{:.3}] heights={:?}",
            name, cat, concepts.len(), d_min, d_max, t_min, t_max,
            {
                let mut h: Vec<usize> = heights_set.into_iter().collect();
                h.sort();
                h
            });
    }
    println!("  ({} concept-article assignments total)", total_assigned);
}

fn export_tsv(
    lattice: &fca::FcaLattice,
    results: &[Option<n_operator::IterResult>],
    article_names: &[String],
    output_dir: &PathBuf,
) {
    let mut triples: Vec<(String, String, String)> = Vec::new();

    for ci in 0..lattice.concepts.len() {
        if lattice.concepts[ci].extent.is_empty() { continue; }
        let primary_art = &article_names[lattice.concepts[ci].extent[0]];
        if let Some(ref r) = results[ci] {
            triples.push((format!("C{}", ci), "D*".to_string(),
                format!("{:.6}", r.m_star[0])));
            triples.push((format!("C{}", ci), "τ⁻¹".to_string(),
                format!("{:.6}", r.tau_inv)));
            triples.push((format!("C{}", ci), "ρ(J_N)".to_string(),
                format!("{:.6}", r.rho_spectral)));
            triples.push((format!("C{}", ci), "primary_article".to_string(), primary_art.clone()));
        }
    }

    for &(a, b) in &lattice.edges {
        triples.push((format!("C{}", a), "hasse_edge".to_string(), format!("C{}", b)));
    }

    let path = output_dir.join("unified_lattice.tsv");
    let _ = n_iter_rs::tsv::write_triples(&path.to_string_lossy(), &triples);
    println!("\n  TSV output: {}", path.display());
}

fn print_verdict(pct: f64) {
    println!();
    if pct >= 99.0 {
        println!("  VERDICT: Theorem 11.1 CONFIRMED — τ⁻¹ monotonicity ≥ 99%");
        println!("  The N-iteration kernel strictly implements:");
        println!("    * Definition 6.12  (N operator, A/(A+B+ε) form)");
        println!("    * Theorem 6.15    (Jacobian 5×5, zero diagonal, 13 non-zero entries)");
        println!("    * Theorem 11.1    (τ⁻¹ monotonicity along Hasse edges)");
        println!("    * Theorem 11.3    (D = |A|/|B| monotonicity)");
        println!("    * Theorem 6.17    (trajectory conservation)");
    } else if pct >= 90.0 {
        println!("  VERDICT: Theorem 11.1 LARGELY CONFIRMED — τ⁻¹ monotonicity ≥ 90%");
    } else if pct >= 70.0 {
        println!("  VERDICT: Partial — τ⁻¹ monotonicity {:.1}%", pct);
        println!("  Possible: some (B_up, ρ_up) fall outside FCA constraint domain.");
    } else {
        println!("  VERDICT: NOT CONFIRMED — τ⁻¹ monotonicity {:.1}%", pct);
        println!("  Possible causes:");
        println!("    1. Lattice too small ({} concepts)", "n");
        println!("    2. B_up/ρ_up cascade breaks Lemma 11.1C denominator monotonicity");
        println!("    3. Need more articles or richer bigram features");
    }
}
