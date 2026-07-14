use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use n_iter_rs::experiments;
use n_iter_rs::io;

fn main() {
    let quick_mode = env::var("QUICK").is_ok();
    let extract_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().join("experiments").join("011-fca-n-iteration").join("extracts");
    let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().join("experiments").join("011-fca-n-iteration").join("output");
    let _ = fs::create_dir_all(&output_dir);

    println!("{}", "=".repeat(64));
    println!("  N-Iteration Experiment Suite");
    println!("{}", "=".repeat(64));

    let mut all_articles: Vec<(String, String, String)> = Vec::new();
    if extract_dir.exists() {
        let cat_map = io::build_category_map();
        for entry in fs::read_dir(&extract_dir).unwrap().flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "json") {
                if let Some((title, extract)) = io::read_wikipedia_extract(&path) {
                    let slug = path.file_stem().unwrap().to_string_lossy().to_string();
                    let cat = cat_map.get(&slug).cloned().unwrap_or_else(|| "Uncategorized".to_string());
                    all_articles.push((title, extract, cat));
                }
            }
        }
        println!("Loaded {} Wikipedia extracts", all_articles.len());
    } else {
        println!("No extracts directory found, running synthetic-only experiments");
    }

    let max_attrs = if quick_mode { 6 } else { 12 };
    let max_concepts = if quick_mode { 12 } else { 30 };
    let time_limit = if quick_mode { 5.0 } else { 30.0 };

    let merged: String = all_articles.iter().map(|(_, e, _)| e.as_str()).collect::<Vec<_>>().join(" ");
    let art_texts: Vec<String> = all_articles.iter().map(|(_, e, _)| e.clone()).collect();

    let by_cat: HashMap<String, Vec<(String, String)>> = {
        let mut m: HashMap<String, Vec<(String, String)>> = HashMap::new();
        for (title, extract, cat) in &all_articles {
            m.entry(cat.clone()).or_default().push((title.clone(), extract.clone()));
        }
        m
    };

    experiments::run_experiment("all", &merged, &art_texts, max_attrs, max_concepts, time_limit, &output_dir);
    experiments::run_param_scan(&all_articles, max_attrs, max_concepts, time_limit, &output_dir);
    experiments::run_lattice_richness_scan(&by_cat, max_concepts, time_limit, &output_dir);
    experiments::run_isolation_experiment(&all_articles, max_attrs, max_concepts, time_limit, &output_dir);
    experiments::run_delta1_scan(&all_articles, max_attrs, max_concepts, time_limit, &output_dir);
    experiments::run_e1_tokenization_stability(&merged, &art_texts, max_concepts);
    experiments::run_e2_carrier_independence(&all_articles, max_attrs, max_concepts, time_limit, &output_dir);
    experiments::run_degradation_scan(&all_articles, max_attrs, max_concepts, time_limit, &output_dir);
    experiments::run_theorem_verification(&all_articles, max_attrs, max_concepts, time_limit, &output_dir);

    run_synthetic_suite(&output_dir);
}

fn run_synthetic_suite(output_dir: &PathBuf) {
    let _ = output_dir;
    experiments::run_th617_verification();
    experiments::run_ode_verification();
    experiments::run_ode_stability_analysis();
    experiments::run_lattice_validation_optimal();
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
    experiments::run_eps_asymptotic_analysis();
    experiments::run_asymptotic_constant_mapping();
    experiments::run_end_to_end_prediction();
    experiments::run_iteration_count_analysis();
    experiments::run_iteration_prediction_correction();
    experiments::run_effective_contraction_rate();
    experiments::run_full_iteration_prediction();
    experiments::run_root_top_rho_ratio();
    experiments::run_final_formula_validation();
    experiments::run_convergence_proof();
    experiments::run_full_sensitivity_analysis();
    experiments::run_lattice_convergence_rate();
    experiments::run_k_factor_analytical();
    experiments::run_distance_correction();
    experiments::run_lattice_prediction_validation();
    experiments::run_lattice_correction_refit();
    experiments::run_universal_constant_derivation();
    experiments::run_fixed_point_analysis();
    experiments::run_jacobian_spectral_structure();
    experiments::run_eigendecomposition_dynamics();
    experiments::run_lattice_coupling_model();
    experiments::run_full_lattice_prediction();
    experiments::run_error_decomposition();
    experiments::run_edge_case_stress_test();
    experiments::run_hessian_second_order();
    experiments::run_riccati_analytical_correction();
    experiments::run_eigenvector_acceleration();
    experiments::run_convergence_phase_transition();
    experiments::run_logistic_trajectory_prediction();
    experiments::run_stepwise_contraction_dynamics();
    experiments::run_fixed_point_manifold_geometry();
    experiments::run_fixed_point_sensitivity();
    experiments::run_spectral_gap_analysis();
    experiments::run_convergence_anisotropy();
    experiments::run_nonlinear_linear_transition();
    experiments::run_trajectory_reconstruction();
    experiments::run_analytical_eigenvalues();
    experiments::run_e3_structure_analysis();
    experiments::run_jacobian_sparsity_analysis();
    experiments::run_spectral_radius_from_cycles();
    experiments::run_parameter_cycle_sensitivity();
    experiments::run_cycle_landscape();
    experiments::run_cycle_topology_invariance();
    experiments::run_full_spectrum_from_cycles();
    experiments::run_dynamic_cycle_evolution();
    experiments::run_e4_cycle_decomposition();
    experiments::run_spectral_perturbation_theory();
    experiments::run_nonnormal_spectral_analysis();
    experiments::run_transient_growth_in_trajectories();
    experiments::run_eigenvector_modal_analysis();
    experiments::run_cycle_weight_pca();
    experiments::run_analytic_cycle_constraints();
    experiments::run_spectral_gap_decomposition();
    experiments::run_spectral_radius_bounds();
    experiments::run_empirical_rho_formula();
    experiments::run_spectrally_invariant_dof();
    experiments::run_invisible_dof_parameter_dependence();
    experiments::run_cross_topology_nonnormality();
    experiments::run_gmax_cycle_weight_prediction();
    experiments::run_cycle_to_spectrum_prediction();
    experiments::run_walk_decomposition();
    experiments::run_analytical_jacobian_verification();
    experiments::run_analytical_fixed_point();
    experiments::run_top_concept_spectral_radius();
    experiments::run_full_lattice_spectral_analysis();
    experiments::run_nonuniform_param_spectral();
    experiments::run_parameter_sensitivity();
    experiments::run_optimal_parameter_search();
    experiments::run_epsilon_interaction_analysis();
    experiments::run_epsilon_zero_asymptotics();
    experiments::run_iteration_count_prediction();
    experiments::run_convergence_phase_decomposition();
    experiments::run_nonlinear_acceleration_analysis();
    experiments::run_nonlinear_iteration_model();
    experiments::run_transient_amplification_analysis();
    experiments::run_lattice_size_scaling();
    experiments::run_cascade_amplification_model();
    experiments::run_cascade_interaction_model();
    experiments::run_cascade_corrected_prediction();
    experiments::run_cross_topology_prediction();
    experiments::run_large_scale_cascade_validation();
    experiments::run_cascade_refit_expanded();
    experiments::run_per_level_cascade_decomposition();
    experiments::run_nonlinearity_cascade_separation();
    experiments::run_nonuniform_cascade_validation();
    experiments::run_adaptive_cascade_correction();
    experiments::run_cycle_weight_cascade_prediction();
    experiments::run_prediction_error_structure();
    experiments::run_d0_corrected_prediction();
    experiments::run_d0_correction_generalization();
    experiments::run_d0_coefficient_analysis();
}
