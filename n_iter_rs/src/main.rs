use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use n_iter_rs::n_operator::{self, DynamicsParams};
use n_iter_rs::fca;
use n_iter_rs::io;
use n_iter_rs::pipeline;

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

    run_param_scan(&all_articles, max_attrs, max_concepts, time_limit, &output_dir);
    run_lattice_richness_scan(&by_cat, max_concepts, time_limit, &output_dir);
    run_isolation_experiment(&all_articles, max_attrs, max_concepts, time_limit, &output_dir);
    run_delta1_scan(&all_articles, max_attrs, max_concepts, time_limit, &output_dir);
}

fn run_param_scan(all_articles: &[(String, String, String)], max_attrs: usize, max_concepts: usize, time_limit: f64, output_dir: &PathBuf) {
    let art_texts: Vec<String> = all_articles.iter().map(|(_, t, _)| t.clone()).collect();
    let n_articles = art_texts.len();

    println!();
    println!("{}", "=".repeat(64));
    println!("  PARAMETER SCAN: β₁ variation on ALL {} articles", n_articles);
    println!("{}", "=".repeat(64));

    let lattice = fca::build_article_lattice(&art_texts, max_attrs, 2, max_concepts, time_limit);
    let n = lattice.concepts.len();
    if n < 3 {
        println!("  Too few concepts ({}), cannot scan.", n);
        return;
    }
    println!("  Lattice: {} concepts, {} edges", n, lattice.edges.len());

    let tsv_path = output_dir.join("param_scan_beta1.tsv");
    let mut tsv_lines: Vec<String> = vec![];
    tsv_lines.push("beta1\ttau_inv_root\ttau_inv_leaf\trho_J_root\trho_J_leaf\tDstar_root\tDstar_leaf\tpass\tfail\trate".to_string());

    let base_params = DynamicsParams::uniform();
    let beta1_values: Vec<f64> = vec![
        0.01, 0.02, 0.05, 0.10, 0.20, 0.50,
        0.75, 1.00, 1.50, 2.00, 3.00, 5.00, 10.0,
    ];

    println!();
    println!("  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>6}  {:>3}",
        "beta1", "τ⁻¹_0", "τ⁻¹_n", "ρ0", "ρn", "D*_0", "D*_n", "pass", "%");

    for &beta1 in &beta1_values {
        let params = base_params.with_beta1(beta1);
        let (tau_inv, rho_j, dstar, heights, edges) = run_pipeline(&lattice, &art_texts, &params);

        let root_idx = heights.iter().position(|&h| h == 0).unwrap_or(0);
        let leaf_idx = heights.iter().enumerate()
            .max_by_key(|(_, &h)| h).map(|(i, _)| i).unwrap_or(n - 1);

        let tau_root = tau_inv[root_idx];
        let tau_leaf = tau_inv[leaf_idx];
        let rho_root = rho_j[root_idx];
        let rho_leaf = rho_j[leaf_idx];
        let d_root = dstar[root_idx];
        let d_leaf = dstar[leaf_idx];

        let _d_mono = fca::verify_theorem_11_3(&lattice.concepts, &dstar, &edges);
        let t_mono = fca::verify_theorem_11_1(&tau_inv, &edges);
        let total = t_mono.0 + t_mono.1;
        let rate = if total > 0 { 100.0 * t_mono.0 as f64 / total as f64 } else { 0.0 };

        println!("  {:>8.3}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>3}/{:>3}  {:>5.1}%",
            beta1, tau_root, tau_leaf, rho_root, rho_leaf, d_root, d_leaf,
            t_mono.0, total, rate);

        tsv_lines.push(format!("{:.4}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{:.4}",
            beta1, tau_root, tau_leaf, rho_root, rho_leaf, d_root, d_leaf,
            t_mono.0, t_mono.1, rate));
    }

    let tsv_content = tsv_lines.join("\n");
    if let Err(e) = fs::write(&tsv_path, &tsv_content) {
        println!("  WARN: could not write TSV: {}", e);
    } else {
        println!("\n  TSV: {}", tsv_path.display());
    }
}

fn run_lattice_richness_scan(by_cat: &HashMap<String, Vec<(String, String)>>, max_concepts: usize, time_limit: f64, _output_dir: &PathBuf) {
    println!();
    println!("{}", "=".repeat(64));
    println!("  FCA LATTICE RICHNESS SCAN (Technology, 14 articles)");
    println!("{}", "=".repeat(64));

    let tech_texts: Vec<&str> = by_cat.get("Technology")
        .map(|a| a.iter().map(|(_, t)| t.as_str()).collect())
        .unwrap_or_default();
    if tech_texts.len() < 3 {
        println!("  Not enough Technology articles");
        return;
    }
    let art_texts: Vec<String> = tech_texts.iter().map(|t| t.to_string()).collect();
    let merged: String = tech_texts.join(" ");

    #[derive(Clone)]
    struct Conf { label: &'static str, is_merged: bool, chars: bool, attrs: usize, min_a: usize }
    let configs = vec![
        Conf { label: "word-FCA  merged     attr=5000  min=2",  is_merged: true,  chars: false, attrs: 5000,  min_a: 2 },
        Conf { label: "word-FCA  article    attr=5000  min=2",  is_merged: false, chars: false, attrs: 5000,  min_a: 2 },
        Conf { label: "word-FCA  article    attr=5000  min=1",  is_merged: false, chars: false, attrs: 5000,  min_a: 1 },
        Conf { label: "word-FCA  article    attr=200   min=2",  is_merged: false, chars: false, attrs: 200,   min_a: 2 },
        Conf { label: "word-FCA  article    attr=200   min=1",  is_merged: false, chars: false, attrs: 200,   min_a: 1 },
        Conf { label: "word-FCA  article    attr=20000 min=2",  is_merged: false, chars: false, attrs: 20000, min_a: 2 },
        Conf { label: "word-FCA  article    attr=20000 min=1",  is_merged: false, chars: false, attrs: 20000, min_a: 1 },
        Conf { label: "char-FCA  article    attr=5000  min=2",  is_merged: false, chars: true,  attrs: 5000,  min_a: 2 },
        Conf { label: "char-FCA  article    attr=5000  min=1",  is_merged: false, chars: true,  attrs: 5000,  min_a: 1 },
        Conf { label: "char-FCA  article    attr=50000 min=2",  is_merged: false, chars: true,  attrs: 50000, min_a: 2 },
        Conf { label: "char-FCA  article    attr=50000 min=1",  is_merged: false, chars: true,  attrs: 50000, min_a: 1 },
        Conf { label: "word-FCA  merged     attr=1000  min=2",  is_merged: true,  chars: false, attrs: 1000,  min_a: 2 },
        Conf { label: "word-FCA  merged     attr=200   min=2",  is_merged: true,  chars: false, attrs: 200,   min_a: 2 },
    ];

    println!();
    println!("  {:>40}  {:>5}  {:>5}  {:>4}", "config", "conc", "edge", "hmax");

    let mut best_conf = None;
    let mut best_n = 0;

    for c in &configs {
        let lat = if c.chars {
            if c.is_merged { fca::build_lattice_chars(&merged, c.attrs, 3, max_concepts, time_limit) }
            else { fca::build_article_lattice_chars(&art_texts, c.attrs, c.min_a, max_concepts, time_limit) }
        } else {
            if c.is_merged { fca::build_lattice(&merged, c.attrs, max_concepts, time_limit) }
            else { fca::build_article_lattice(&art_texts, c.attrs, c.min_a, max_concepts, time_limit) }
        };
        let n = lat.concepts.len();
        let hm = fca::hasse_heights(n, &lat.edges).iter().cloned().max().unwrap_or(0);
        println!("  {:>40}  {:>5}  {:>5}  {:>4}", c.label, n, lat.edges.len(), hm);
        if n > best_n { best_n = n; best_conf = Some(c.clone()); }
    }

    if let Some(best) = best_conf {
        println!();
        println!("  BEST: {} → {} concepts", best.label, best_n);
        if best_n > 4 {
            let lat = if best.chars {
                if best.is_merged { fca::build_lattice_chars(&merged, best.attrs, 3, max_concepts, time_limit) }
                else { fca::build_article_lattice_chars(&art_texts, best.attrs, best.min_a, max_concepts, time_limit) }
            } else {
                if best.is_merged { fca::build_lattice(&merged, best.attrs, max_concepts, time_limit) }
                else { fca::build_article_lattice(&art_texts, best.attrs, best.min_a, max_concepts, time_limit) }
            };
            let params = DynamicsParams::uniform();
            let (tau_inv, rho_j, dstar, heights, edges) = run_pipeline(&lat, &art_texts, &params);
            let t_mono = fca::verify_theorem_11_1(&tau_inv, &edges);
            let d_mono = fca::verify_theorem_11_3(&lat.concepts, &dstar, &edges);
            let total = t_mono.0 + t_mono.1;
            println!("  N-iter on BEST ({} concepts, {} edges):", best_n, edges.len());
            println!("    Theorem 11.1 (τ⁻¹): {}/{} = {:.1}%",
                t_mono.0, total, if total > 0 { 100.0 * t_mono.0 as f64 / total as f64 } else { 0.0 });
            println!("    Theorem 11.3 (D*):  {}/{} = {:.1}%",
                d_mono.0, d_mono.0 + d_mono.1,
                if d_mono.0 + d_mono.1 > 0 { 100.0 * d_mono.0 as f64 / (d_mono.0 + d_mono.1) as f64 } else { 0.0 });
            println!("    τ⁻¹ range: [{:.4}, {:.4}]  ρ(J) range: [{:.4}, {:.4}]",
                tau_inv.iter().cloned().fold(f64::INFINITY, f64::min),
                tau_inv.iter().cloned().fold(0.0_f64, f64::max),
                rho_j.iter().cloned().fold(f64::INFINITY, f64::min),
                rho_j.iter().cloned().fold(0.0_f64, f64::max));
            println!("    By height:");
            let mut by_h: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
            for (ci, &h) in heights.iter().enumerate() { by_h.entry(h).or_default().push(ci); }
            for (h, cis) in &by_h {
                let taus: Vec<f64> = cis.iter().map(|&ci| tau_inv[ci]).collect();
                println!("      h={}: τ⁻¹∈[{:.4},{:.4}] ({} concepts)", h,
                    taus.iter().cloned().fold(f64::INFINITY, f64::min),
                    taus.iter().cloned().fold(0.0_f64, f64::max),
                    cis.len());
            }
        }
    }
}

fn run_isolation_experiment(all_articles: &[(String, String, String)], max_attrs: usize, max_concepts: usize, time_limit: f64, output_dir: &PathBuf) {
    let art_texts: Vec<String> = all_articles.iter().map(|(_, t, _)| t.clone()).collect();

    println!();
    println!("{}", "=".repeat(64));
    println!("  ISOLATION EXPERIMENT: b_up=0 vs b_up>0 on ALL 45 articles");
    println!("{}", "=".repeat(64));

    let lattice = fca::build_article_lattice(&art_texts, max_attrs, 2, max_concepts, time_limit);
    let n = lattice.concepts.len();
    if n < 3 { return; }
    println!("  Lattice: {} concepts, {} edges", n, lattice.edges.len());

    let stats = pipeline::compute_lattice_stats(&lattice);
    let base_params = DynamicsParams::uniform();

    let mut isolated_results: Vec<Option<n_operator::IterResult>> = vec![None; n];
    let mut partial_results: Vec<Option<n_operator::IterResult>> = vec![None; n];
    let mut full_results: Vec<Option<n_operator::IterResult>> = vec![None; n];

    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by_key(|&i| std::cmp::Reverse(stats.heights[i]));

    for &ci in &sorted {
        let m0 = pipeline::init_state(ci, &lattice, &stats);
        let (b_up, rho_up) = pipeline::get_upstream(ci, &stats.feeders, &full_results);
        let iso = n_operator::run_iteration(&m0, 0.0, 0.0, &base_params, 500, 1e-12);
        let part = n_operator::run_iteration(&m0, b_up, 0.0, &base_params, 500, 1e-12);
        let full = n_operator::run_iteration(&m0, b_up, rho_up, &base_params, 500, 1e-12);
        isolated_results[ci] = Some(iso);
        partial_results[ci] = Some(part);
        full_results[ci] = Some(full);
    }

    println!();
    println!("  {:>4} {:>3} {:>10} {:>10} {:>10}  {:>10} {:>10} {:>10}",
        "C#", "h", "τ⁻¹(iso)", "τ⁻¹(+B↑)", "τ⁻¹(full)", "ρ(iso)", "ρ(+B↑)", "ρ(full)");

    let mut by_h: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
    for (ci, &h) in stats.heights.iter().enumerate() { by_h.entry(h).or_default().push(ci); }

    for (h, cis) in &by_h {
        for &ci in cis {
            let ti = |r: &Option<n_operator::IterResult>| r.as_ref().map(|x| x.tau_inv).unwrap_or(0.0);
            let rj = |r: &Option<n_operator::IterResult>| r.as_ref().map(|x| x.rho_spectral).unwrap_or(0.0);
            println!("  {:>4} {:>3} {:>10.4} {:>10.4} {:>10.4}  {:>10.4} {:>10.4} {:>10.4}",
                ci, h,
                ti(&isolated_results[ci]), ti(&partial_results[ci]), ti(&full_results[ci]),
                rj(&isolated_results[ci]), rj(&partial_results[ci]), rj(&full_results[ci]));
        }
    }

    println!();
    println!("  Effect of B↑ injection (τ⁻¹ change):");
    for (h, cis) in &by_h {
        for &ci in cis {
            let ti = |r: &Option<n_operator::IterResult>| r.as_ref().map(|x| x.tau_inv).unwrap_or(0.0);
            let rj = |r: &Option<n_operator::IterResult>| r.as_ref().map(|x| x.rho_spectral).unwrap_or(0.0);
            let delta_tau = ti(&full_results[ci]) - ti(&isolated_results[ci]);
            let delta_rho = rj(&full_results[ci]) - rj(&isolated_results[ci]);
            let (b_up, _) = pipeline::get_upstream(ci, &stats.feeders, &full_results);
            println!("    C{:<3} h={} B↑={:.4}  Δτ⁻¹={:+.4}  Δρ={:+.4}", ci, h, b_up, delta_tau, delta_rho);
        }
    }

    let tsv_path = output_dir.join("isolation_experiment.tsv");
    let mut lines = vec!["ci\theight\tB_up\ttau_iso\ttau_partial\ttau_full\trho_iso\trho_partial\trho_full\tDstar_iso\tDstar_full".to_string()];
    for ci in 0..n {
        let (b_up, _) = pipeline::get_upstream(ci, &stats.feeders, &full_results);
        let ti = |r: &Option<n_operator::IterResult>| r.as_ref().map(|x| x.tau_inv).unwrap_or(0.0);
        let rj = |r: &Option<n_operator::IterResult>| r.as_ref().map(|x| x.rho_spectral).unwrap_or(0.0);
        let ds = |r: &Option<n_operator::IterResult>| r.as_ref().map(|x| x.m_star[0]).unwrap_or(0.0);
        lines.push(format!("{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            ci, stats.heights[ci], b_up,
            ti(&isolated_results[ci]), ti(&partial_results[ci]), ti(&full_results[ci]),
            rj(&isolated_results[ci]), rj(&partial_results[ci]), rj(&full_results[ci]),
            ds(&isolated_results[ci]), ds(&full_results[ci])));
    }
    let _ = fs::write(&tsv_path, lines.join("\n"));
    println!("\n  TSV: {}", tsv_path.display());
}

fn run_delta1_scan(all_articles: &[(String, String, String)], max_attrs: usize, max_concepts: usize, time_limit: f64, output_dir: &PathBuf) {
    let art_texts: Vec<String> = all_articles.iter().map(|(_, t, _)| t.clone()).collect();
    let n_articles = art_texts.len();

    println!();
    println!("{}", "=".repeat(64));
    println!("  δ₁ PARAMETER SCAN + JACOBIAN HEATMAP on ALL {} articles", n_articles);
    println!("{}", "=".repeat(64));

    let lattice = fca::build_article_lattice(&art_texts, max_attrs, 2, max_concepts, time_limit);
    let n = lattice.concepts.len();
    if n < 3 { return; }
    println!("  Lattice: {} concepts, {} edges", n, lattice.edges.len());

    let base_params = DynamicsParams::uniform();
    let delta1_values: Vec<f64> = vec![
        0.01, 0.05, 0.10, 0.20, 0.50,
        0.75, 1.00, 2.00, 5.00, 10.0, 20.0, 50.0, 100.0,
    ];

    println!();
    println!("  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>6}  {:>3}",
        "delta1", "τ⁻¹_0", "τ⁻¹_n", "ρ0", "ρn", "D*_0", "D*_n", "pass", "%");

    let mut all_results: Vec<(f64, Vec<f64>, Vec<f64>, Vec<f64>)> = Vec::new();

    for &delta1 in &delta1_values {
        let params = base_params.with_delta1(delta1);
        let (tau_inv, rho_j, dstar, heights, edges) = run_pipeline(&lattice, &art_texts, &params);

        let root_idx = heights.iter().position(|&h| h == 0).unwrap_or(0);
        let leaf_idx = heights.iter().enumerate()
            .max_by_key(|(_, &h)| h).map(|(i, _)| i).unwrap_or(n - 1);

        let tau_root = tau_inv[root_idx];
        let tau_leaf = tau_inv[leaf_idx];
        let rho_root = rho_j[root_idx];
        let rho_leaf = rho_j[leaf_idx];
        let d_root = dstar[root_idx];
        let d_leaf = dstar[leaf_idx];

        let t_mono = fca::verify_theorem_11_1(&tau_inv, &edges);
        let total = t_mono.0 + t_mono.1;
        let rate = if total > 0 { 100.0 * t_mono.0 as f64 / total as f64 } else { 0.0 };

        println!("  {:>8.3}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>3}/{:>3}  {:>5.1}%",
            delta1, tau_root, tau_leaf, rho_root, rho_leaf, d_root, d_leaf,
            t_mono.0, total, rate);

        all_results.push((delta1, tau_inv.clone(), rho_j.clone(), dstar.clone()));
    }

    let tsv_path = output_dir.join("param_scan_delta1.tsv");
    let mut lines = vec!["delta1\ttau_inv_root\ttau_inv_leaf\trho_J_root\trho_J_leaf\tDstar_root\tDstar_leaf\tpass\tfail\trate".to_string()];
    for (delta1, tau_inv, rho_j, dstar) in &all_results {
        let root_idx = 0;
        let leaf_idx = n - 1;
        let t_mono = fca::verify_theorem_11_1(tau_inv, &lattice.edges);
        let total = t_mono.0 + t_mono.1;
        let rate = if total > 0 { 100.0 * t_mono.0 as f64 / total as f64 } else { 0.0 };
        lines.push(format!("{:.4}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{:.4}",
            delta1, tau_inv[root_idx], tau_inv[leaf_idx],
            rho_j[root_idx], rho_j[leaf_idx],
            dstar[root_idx], dstar[leaf_idx],
            t_mono.0, t_mono.1, rate));
    }
    let _ = fs::write(&tsv_path, lines.join("\n"));
    println!("\n  TSV: {}", tsv_path.display());

    println!();
    println!("{}", "=".repeat(64));
    println!("  JACOBIAN HEATMAP at M* (δ₁=1.0, ALL 45 articles, 4-concept lattice)");
    println!("{}", "=".repeat(64));

    let stats = pipeline::compute_lattice_stats(&lattice);
    let results = pipeline::run_topological_iteration(&lattice, &stats, &DynamicsParams::uniform());

    let labels = ["D", "B", "ρ", "R", "S"];
    let mut by_h: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
    for (ci, &h) in stats.heights.iter().enumerate() { by_h.entry(h).or_default().push(ci); }

    for (h, cis) in &by_h {
        for &ci in cis {
            if let Some(ref r) = results[ci] {
                let (b_up, _) = pipeline::get_upstream(ci, &stats.feeders, &results);
                println!();
                println!("  C{} (h={}, B↑={:.4}, τ⁻¹={:.4}, ρ={:.4}):", ci, h, b_up, r.tau_inv, r.rho_spectral);
                println!("      {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}", "∂/∂", "D", "B", "ρ", "R", "S");
                for row in 0..5 {
                    let prefix = format!("      N_{}", labels[row]);
                    let mut vals = String::new();
                    for col in 0..5 {
                        let v = r.jacobian[row * 5 + col];
                        if v.abs() < 1e-8 {
                            vals.push_str(&format!("  {:>6}", "0"));
                        } else {
                            vals.push_str(&format!("  {:>+6.3}", v));
                        }
                    }
                    println!("{} {}", prefix, vals);
                }

                let mut max_off = 0.0_f64;
                let mut max_pair = (0usize, 0usize);
                for row in 0..5 {
                    for col in 0..5 {
                        if row != col {
                            let v = r.jacobian[row * 5 + col];
                            if v.abs() > max_off.abs() {
                                max_off = v;
                                max_pair = (row, col);
                            }
                        }
                    }
                }
                let max_total = r.jacobian.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
                println!("      dominant: ∂N_{}/∂{}={:+.4}  max|J_ij|={:.4}",
                    labels[max_pair.0], labels[max_pair.1], max_off, max_total);
            }
        }
    }
}

fn run_pipeline(lattice: &fca::FcaLattice, art_texts: &[String], params: &DynamicsParams)
    -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>, Vec<(usize, usize)>)
{
    let stats = pipeline::compute_lattice_stats(lattice);
    let results = pipeline::run_topological_iteration(lattice, &stats, params);
    let (tau_inv, rho_j, dstar) = pipeline::extract_scalars(&results, &lattice.edges);
    let _ = art_texts;
    (tau_inv, rho_j, dstar, stats.heights, lattice.edges.clone())
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

    let params = DynamicsParams::uniform();
    let stats = pipeline::compute_lattice_stats(&lattice);

    println!("  Running N-iteration on {} concepts...", n);
    let n_start = std::time::Instant::now();
    let results = pipeline::run_topological_iteration(&lattice, &stats, &params);
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
            let (b_up, rho_up) = pipeline::get_upstream(ci, &stats.feeders, &results);
            if let Some(_d) = r.verify_trajectory_conservation(&params, b_up, rho_up) {
                max_traj_dev = max_traj_dev.max(_d);
            }
        }
    }

    let d_std = pipeline::std_dev(&d_vals);
    let t_std = pipeline::std_dev(&all_tau);

    println!("  N-iteration done in {:.1}s", n_elapsed);
    println!();
    println!("  Concepts:             {}", n);
    println!("  Hasse edges:          {}", lattice.edges.len());
    println!("  Max lattice height:   {}", stats.heights.iter().max().unwrap_or(&0));
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
        pipeline::pct(fca_d_mono.0, fca_d_mono.0 + fca_d_mono.1));
    println!("  Empirical      (N-iter D*)          {}/{} = {:.1}%",
        d_star_mono.0, d_star_mono.0 + d_star_mono.1,
        pipeline::pct(d_star_mono.0, d_star_mono.0 + d_star_mono.1));
    println!("  Theorem 11.1  (τ⁻¹ monotonicity)    {}/{} = {:.1}%",
        t_mono.0, t_mono.0 + t_mono.1,
        pipeline::pct(t_mono.0, t_mono.0 + t_mono.1));
    println!("  Theorem 6.17  (trajectory)           max dev = {:.2e}", max_traj_dev);

    let mut height_stats: HashMap<usize, Vec<f64>> = HashMap::new();
    for (ci, opt_r) in results.iter().enumerate() {
        if let Some(ref r) = opt_r {
            height_stats.entry(stats.heights[ci]).or_default().push(r.tau_inv);
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
                if stats.heights[ci] == *h { v.push(ci); break; }
            }
        }
    }
    for (h, cis) in &by_h {
        for &ci in cis {
            if let Some(ref r) = results[ci] {
                let m = &r.m_star;
                let d0 = lattice.d_values[ci];
                let (b_up, _rho_up) = pipeline::get_upstream(ci, &stats.feeders, &results);
                let d0_str = if d0.is_finite() && d0 < 1e6 {
                    format!("{:.4}→{:.4}", d0 / stats.max_d_valid.min(1.0).max(0.01), m[0])
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
                let (b_up, _rho_up) = pipeline::get_upstream(ci, &stats.feeders, &results);
                let ratio = (m[3] + 0.01) / (m[3] + m[1] + b_up + 0.01);
                println!("    C{:<3} h={} R*={:.4} B*+B↑={:.4} ratio={:.4} vs D*={:.4}",
                    ci, h, m[3], m[1] + b_up, ratio, m[0]);
            }
        }
    }

    t_mono_verdict(t_mono.0, t_mono.0 + t_mono.1);

    println!("  FCA D₀=|A|/|B|:  mean={:.4}  std={:.4}  range=({:.4}, {:.4})",
        pipeline::mean(&lattice.d_values.iter().filter(|&&d| d.is_finite() && d < 1e6).copied().collect::<Vec<_>>()),
        pipeline::std_dev(&lattice.d_values.iter().filter(|&&d| d.is_finite() && d < 1e6).copied().collect::<Vec<_>>()),
        lattice.d_values.iter().filter(|&&d| d.is_finite() && d < 1e6).fold(f64::INFINITY, |a, &b| a.min(b)),
        lattice.d_values.iter().filter(|&&d| d.is_finite() && d < 1e6).fold(0.0_f64, |a, &b| a.max(b)));
    println!("  N-iter D*:        mean={:.4}  std={:.4}  range=({:.4}, {:.4})",
        pipeline::mean(&d_vals), d_std,
        d_vals.iter().cloned().fold(f64::INFINITY, f64::min),
        d_vals.iter().cloned().fold(0.0_f64, f64::max));
    println!("  B* values:        mean={:.4}  std={:.4}",
        pipeline::mean(&results.iter().filter_map(|r| r.as_ref()).map(|r| r.m_star[1]).collect::<Vec<_>>()),
        pipeline::std_dev(&results.iter().filter_map(|r| r.as_ref()).map(|r| r.m_star[1]).collect::<Vec<_>>()));

    let d0_to_dstar: Vec<f64> = results.iter().enumerate().filter_map(|(ci, r)| {
        let d0 = lattice.d_values[ci];
        if d0.is_finite() && d0 < 1e6 {
            r.as_ref().map(|rr| rr.m_star[0] / (d0 / stats.max_d_valid).max(0.01))
        } else { None }
    }).collect();
    if !d0_to_dstar.is_empty() {
        println!("  D*/D₀ ratio:      mean={:.4}  std={:.4}  range=({:.4}, {:.4})",
            pipeline::mean(&d0_to_dstar), pipeline::std_dev(&d0_to_dstar),
            d0_to_dstar.iter().cloned().fold(f64::INFINITY, f64::min),
            d0_to_dstar.iter().cloned().fold(0.0_f64, f64::max));
    }
}

fn t_mono_verdict(passes: usize, total: usize) {
    let p = pipeline::pct(passes, total);
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
