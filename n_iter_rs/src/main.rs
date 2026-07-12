use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use n_iter_rs::experiments;
use n_iter_rs::io;

fn main() {
    let args: Vec<String> = env::args().collect();
    let quick_mode = args.iter().any(|a| a == "--quick" || a == "-q");

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

    let max_concepts: usize = 5000;
    let time_limit: f64 = 1800.0;

    if quick_mode {
        println!("{}", "=".repeat(64));
        println!("  QUICK MODE: synthetic lattice experiments only");
        println!("{}", "=".repeat(64));
        run_synthetic_suite(max_concepts, time_limit, &output_dir);
        return;
    }

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

    {
        let mut merged = String::new();
        for (_, text, _) in &all_articles { merged.push_str(text); merged.push(' '); }

        let art_texts: Vec<String> = all_articles.iter().map(|(_, t, _)| t.clone()).collect();
        experiments::run_experiment(
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
        experiments::run_experiment(
            &format!("{} ({}) articles", cat, arts.len()),
            &merged,
            &art_texts,
            max_attrs,
            max_concepts,
            time_limit,
            &output_dir,
        );
    }

    experiments::run_param_scan(&all_articles, max_attrs, max_concepts, time_limit, &output_dir);
    experiments::run_lattice_richness_scan(&by_cat, max_concepts, time_limit, &output_dir);
    experiments::run_isolation_experiment(&all_articles, max_attrs, max_concepts, time_limit, &output_dir);
    experiments::run_delta1_scan(&all_articles, max_attrs, max_concepts, time_limit, &output_dir);
    let all_merged: String = all_articles.iter().map(|(_, t, _)| t.as_str()).collect::<Vec<_>>().join(" ");
    let all_texts: Vec<String> = all_articles.iter().map(|(_, t, _)| t.clone()).collect();
    experiments::run_e1_tokenization_stability(&all_merged, &all_texts, max_concepts);
    experiments::run_e2_carrier_independence(&all_articles, max_attrs, max_concepts, time_limit, &output_dir);
    experiments::run_degradation_scan(&all_articles, max_attrs, max_concepts, time_limit, &output_dir);
    experiments::run_theorem_verification(&all_articles, max_attrs, max_concepts, time_limit, &output_dir);

    run_synthetic_suite(max_concepts, time_limit, &output_dir);
}

fn run_synthetic_suite(max_concepts: usize, time_limit: f64, output_dir: &PathBuf) {
    experiments::run_synthetic_scan(max_concepts, time_limit, output_dir);
    experiments::run_stress_tests(output_dir);
    experiments::run_chain_diagnostics(output_dir);
    experiments::run_th617_verification();
    experiments::run_ode_verification();
    experiments::run_ode_stability_analysis();
    experiments::run_ode_exact_lyapunov();
    experiments::run_convergence_rate_analysis();
    experiments::run_sensitivity_analysis();
    experiments::run_cross_metric_correlation();
    experiments::run_pareto_and_linear_regression();
    experiments::run_cross_regime_pareto();
    experiments::run_beta_delta_2d_sweep();
    experiments::run_super_optimal_characterization();
    experiments::run_fine_grained_landscape();
    experiments::run_tau_mono_phase_diagram();
    experiments::run_n_pred_discrepancy_analysis();
    experiments::run_edge_tau_comparison();
    experiments::run_dstar_dvalue_analysis();
    experiments::run_analytical_dstar();
    experiments::run_bup_propagation();
    experiments::run_rho_analytical();
    experiments::run_jacobian_coupling();
    experiments::run_super_optimal_analytical();
    experiments::run_characteristic_polynomial();
    experiments::run_predictive_validation();
    experiments::run_rho_propagation();
    experiments::run_gamma_analysis();
    experiments::run_dstar_equation();
    experiments::run_closed_form_pipeline();
    experiments::run_topology_pipeline();
    experiments::run_contraction_analysis();
    experiments::run_analytical_j2d();
    experiments::run_contraction_boundary();
    experiments::run_non_contract_test();
    experiments::run_multi_fp_analysis();
    experiments::run_boundary_conditions();
    experiments::run_nonuniform_robustness();
    experiments::run_critical_behavior();
    experiments::run_multiroot_landscape();
    experiments::run_noncontract_reconciliation();
    experiments::run_bifurcation_analysis();
    experiments::run_tau_predictive_validation();
    experiments::run_propagation_map_validation();
    experiments::run_bifurcation_curve_fit();
    experiments::run_convergence_landscape();
    experiments::run_convergence_scaling();
    experiments::run_asymptotic_limits();
    experiments::run_upper_bound_analysis();
    experiments::run_root_comparison();
    experiments::run_epsilon_sensitivity();
    experiments::run_optimal_epsilon();
    experiments::run_joint_optimization();
    experiments::run_ratio_analysis();
    experiments::run_balance_condition();
    experiments::run_optimal_regime();
    experiments::run_cross_topology_optimal();
    experiments::run_max_rho_analysis();
    experiments::run_top_concept_analysis();
    experiments::run_max_rho_minimization();
    experiments::run_joint_bottleneck_optimization();
    experiments::run_lattice_validation_optimal();
    experiments::run_eps_asymptotic_analysis();
    experiments::run_asymptotic_constant_mapping();
    experiments::run_end_to_end_prediction();
    experiments::run_iteration_count_analysis();
    experiments::run_iteration_prediction_correction();
    experiments::run_effective_contraction_rate();
    experiments::run_full_iteration_prediction();
    experiments::run_root_top_rho_ratio();
    let _ = (max_concepts, time_limit, output_dir);
}