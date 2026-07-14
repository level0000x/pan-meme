use std::env;
use n_iter_rs::experiments;

fn main() {
    let args: Vec<String> = env::args().collect();
    let specific = args.get(1).map(|s| s.as_str());

    if let Some(name) = specific {
        run_single(name);
    } else {
        run_synthetic_suite();
    }
}

fn run_single(name: &str) {
    let fns: &[(&str, fn())] = &[
        ("run_th617_verification", experiments::run_th617_verification),
        ("run_ode_verification", experiments::run_ode_verification),
        ("run_ode_stability_analysis", experiments::run_ode_stability_analysis),
        ("run_lattice_validation_optimal", experiments::run_lattice_validation_optimal),
        ("run_ode_exact_lyapunov", experiments::run_ode_exact_lyapunov),
        ("run_convergence_rate_analysis", experiments::run_convergence_rate_analysis),
        ("run_sensitivity_analysis", experiments::run_sensitivity_analysis),
        ("run_cross_metric_correlation", experiments::run_cross_metric_correlation),
        ("run_pareto_and_linear_regression", experiments::run_pareto_and_linear_regression),
        ("run_cross_regime_pareto", experiments::run_cross_regime_pareto),
        ("run_beta_delta_2d_sweep", experiments::run_beta_delta_2d_sweep),
        ("run_super_optimal_characterization", experiments::run_super_optimal_characterization),
        ("run_fine_grained_landscape", experiments::run_fine_grained_landscape),
        ("run_tau_mono_phase_diagram", experiments::run_tau_mono_phase_diagram),
        ("run_n_pred_discrepancy_analysis", experiments::run_n_pred_discrepancy_analysis),
        ("run_edge_tau_comparison", experiments::run_edge_tau_comparison),
        ("run_dstar_dvalue_analysis", experiments::run_dstar_dvalue_analysis),
        ("run_analytical_dstar", experiments::run_analytical_dstar),
        ("run_bup_propagation", experiments::run_bup_propagation),
        ("run_rho_analytical", experiments::run_rho_analytical),
        ("run_jacobian_coupling", experiments::run_jacobian_coupling),
        ("run_super_optimal_analytical", experiments::run_super_optimal_analytical),
        ("run_characteristic_polynomial", experiments::run_characteristic_polynomial),
        ("run_predictive_validation", experiments::run_predictive_validation),
        ("run_rho_propagation", experiments::run_rho_propagation),
        ("run_gamma_analysis", experiments::run_gamma_analysis),
        ("run_dstar_equation", experiments::run_dstar_equation),
        ("run_closed_form_pipeline", experiments::run_closed_form_pipeline),
        ("run_topology_pipeline", experiments::run_topology_pipeline),
        ("run_contraction_analysis", experiments::run_contraction_analysis),
        ("run_analytical_j2d", experiments::run_analytical_j2d),
        ("run_contraction_boundary", experiments::run_contraction_boundary),
        ("run_non_contract_test", experiments::run_non_contract_test),
        ("run_multi_fp_analysis", experiments::run_multi_fp_analysis),
        ("run_boundary_conditions", experiments::run_boundary_conditions),
        ("run_nonuniform_robustness", experiments::run_nonuniform_robustness),
        ("run_critical_behavior", experiments::run_critical_behavior),
        ("run_multiroot_landscape", experiments::run_multiroot_landscape),
        ("run_noncontract_reconciliation", experiments::run_noncontract_reconciliation),
        ("run_bifurcation_analysis", experiments::run_bifurcation_analysis),
        ("run_tau_predictive_validation", experiments::run_tau_predictive_validation),
        ("run_propagation_map_validation", experiments::run_propagation_map_validation),
        ("run_bifurcation_curve_fit", experiments::run_bifurcation_curve_fit),
        ("run_convergence_landscape", experiments::run_convergence_landscape),
        ("run_convergence_scaling", experiments::run_convergence_scaling),
        ("run_asymptotic_limits", experiments::run_asymptotic_limits),
        ("run_upper_bound_analysis", experiments::run_upper_bound_analysis),
        ("run_root_comparison", experiments::run_root_comparison),
        ("run_epsilon_sensitivity", experiments::run_epsilon_sensitivity),
        ("run_optimal_epsilon", experiments::run_optimal_epsilon),
        ("run_joint_optimization", experiments::run_joint_optimization),
        ("run_ratio_analysis", experiments::run_ratio_analysis),
        ("run_balance_condition", experiments::run_balance_condition),
        ("run_optimal_regime", experiments::run_optimal_regime),
        ("run_cross_topology_optimal", experiments::run_cross_topology_optimal),
        ("run_max_rho_analysis", experiments::run_max_rho_analysis),
        ("run_top_concept_analysis", experiments::run_top_concept_analysis),
        ("run_max_rho_minimization", experiments::run_max_rho_minimization),
        ("run_joint_bottleneck_optimization", experiments::run_joint_bottleneck_optimization),
        ("run_eps_asymptotic_analysis", experiments::run_eps_asymptotic_analysis),
        ("run_asymptotic_constant_mapping", experiments::run_asymptotic_constant_mapping),
        ("run_end_to_end_prediction", experiments::run_end_to_end_prediction),
        ("run_iteration_count_analysis", experiments::run_iteration_count_analysis),
        ("run_iteration_prediction_correction", experiments::run_iteration_prediction_correction),
        ("run_effective_contraction_rate", experiments::run_effective_contraction_rate),
        ("run_full_iteration_prediction", experiments::run_full_iteration_prediction),
        ("run_root_top_rho_ratio", experiments::run_root_top_rho_ratio),
        ("run_final_formula_validation", experiments::run_final_formula_validation),
        ("run_convergence_proof", experiments::run_convergence_proof),
        ("run_full_sensitivity_analysis", experiments::run_full_sensitivity_analysis),
        ("run_lattice_convergence_rate", experiments::run_lattice_convergence_rate),
        ("run_k_factor_analytical", experiments::run_k_factor_analytical),
        ("run_distance_correction", experiments::run_distance_correction),
        ("run_lattice_prediction_validation", experiments::run_lattice_prediction_validation),
        ("run_lattice_correction_refit", experiments::run_lattice_correction_refit),
        ("run_universal_constant_derivation", experiments::run_universal_constant_derivation),
        ("run_fixed_point_analysis", experiments::run_fixed_point_analysis),
        ("run_jacobian_spectral_structure", experiments::run_jacobian_spectral_structure),
        ("run_eigendecomposition_dynamics", experiments::run_eigendecomposition_dynamics),
        ("run_lattice_coupling_model", experiments::run_lattice_coupling_model),
        ("run_full_lattice_prediction", experiments::run_full_lattice_prediction),
        ("run_error_decomposition", experiments::run_error_decomposition),
        ("run_edge_case_stress_test", experiments::run_edge_case_stress_test),
        ("run_hessian_second_order", experiments::run_hessian_second_order),
        ("run_riccati_analytical_correction", experiments::run_riccati_analytical_correction),
        ("run_eigenvector_acceleration", experiments::run_eigenvector_acceleration),
        ("run_convergence_phase_transition", experiments::run_convergence_phase_transition),
        ("run_logistic_trajectory_prediction", experiments::run_logistic_trajectory_prediction),
        ("run_stepwise_contraction_dynamics", experiments::run_stepwise_contraction_dynamics),
        ("run_fixed_point_manifold_geometry", experiments::run_fixed_point_manifold_geometry),
        ("run_fixed_point_sensitivity", experiments::run_fixed_point_sensitivity),
        ("run_spectral_gap_analysis", experiments::run_spectral_gap_analysis),
        ("run_convergence_anisotropy", experiments::run_convergence_anisotropy),
        ("run_nonlinear_linear_transition", experiments::run_nonlinear_linear_transition),
        ("run_trajectory_reconstruction", experiments::run_trajectory_reconstruction),
        ("run_analytical_eigenvalues", experiments::run_analytical_eigenvalues),
        ("run_e3_structure_analysis", experiments::run_e3_structure_analysis),
        ("run_jacobian_sparsity_analysis", experiments::run_jacobian_sparsity_analysis),
        ("run_spectral_radius_from_cycles", experiments::run_spectral_radius_from_cycles),
        ("run_parameter_cycle_sensitivity", experiments::run_parameter_cycle_sensitivity),
        ("run_cycle_landscape", experiments::run_cycle_landscape),
        ("run_cycle_topology_invariance", experiments::run_cycle_topology_invariance),
        ("run_full_spectrum_from_cycles", experiments::run_full_spectrum_from_cycles),
        ("run_dynamic_cycle_evolution", experiments::run_dynamic_cycle_evolution),
        ("run_e4_cycle_decomposition", experiments::run_e4_cycle_decomposition),
        ("run_spectral_perturbation_theory", experiments::run_spectral_perturbation_theory),
        ("run_nonnormal_spectral_analysis", experiments::run_nonnormal_spectral_analysis),
        ("run_transient_growth_in_trajectories", experiments::run_transient_growth_in_trajectories),
        ("run_eigenvector_modal_analysis", experiments::run_eigenvector_modal_analysis),
        ("run_cycle_weight_pca", experiments::run_cycle_weight_pca),
        ("run_analytic_cycle_constraints", experiments::run_analytic_cycle_constraints),
        ("run_spectral_gap_decomposition", experiments::run_spectral_gap_decomposition),
        ("run_spectral_radius_bounds", experiments::run_spectral_radius_bounds),
        ("run_empirical_rho_formula", experiments::run_empirical_rho_formula),
        ("run_spectrally_invariant_dof", experiments::run_spectrally_invariant_dof),
        ("run_invisible_dof_parameter_dependence", experiments::run_invisible_dof_parameter_dependence),
        ("run_cross_topology_nonnormality", experiments::run_cross_topology_nonnormality),
        ("run_gmax_cycle_weight_prediction", experiments::run_gmax_cycle_weight_prediction),
        ("run_cycle_to_spectrum_prediction", experiments::run_cycle_to_spectrum_prediction),
        ("run_walk_decomposition", experiments::run_walk_decomposition),
        ("run_analytical_jacobian_verification", experiments::run_analytical_jacobian_verification),
        ("run_analytical_fixed_point", experiments::run_analytical_fixed_point),
        ("run_top_concept_spectral_radius", experiments::run_top_concept_spectral_radius),
        ("run_full_lattice_spectral_analysis", experiments::run_full_lattice_spectral_analysis),
        ("run_nonuniform_param_spectral", experiments::run_nonuniform_param_spectral),
        ("run_parameter_sensitivity", experiments::run_parameter_sensitivity),
        ("run_optimal_parameter_search", experiments::run_optimal_parameter_search),
        ("run_epsilon_interaction_analysis", experiments::run_epsilon_interaction_analysis),
        ("run_epsilon_zero_asymptotics", experiments::run_epsilon_zero_asymptotics),
        ("run_iteration_count_prediction", experiments::run_iteration_count_prediction),
        ("run_convergence_phase_decomposition", experiments::run_convergence_phase_decomposition),
        ("run_nonlinear_acceleration_analysis", experiments::run_nonlinear_acceleration_analysis),
        ("run_nonlinear_iteration_model", experiments::run_nonlinear_iteration_model),
        ("run_transient_amplification_analysis", experiments::run_transient_amplification_analysis),
        ("run_lattice_size_scaling", experiments::run_lattice_size_scaling),
        ("run_cascade_amplification_model", experiments::run_cascade_amplification_model),
        ("run_cascade_interaction_model", experiments::run_cascade_interaction_model),
        ("run_cascade_corrected_prediction", experiments::run_cascade_corrected_prediction),
        ("run_cross_topology_prediction", experiments::run_cross_topology_prediction),
        ("run_large_scale_cascade_validation", experiments::run_large_scale_cascade_validation),
        ("run_cascade_refit_expanded", experiments::run_cascade_refit_expanded),
        ("run_per_level_cascade_decomposition", experiments::run_per_level_cascade_decomposition),
        ("run_nonlinearity_cascade_separation", experiments::run_nonlinearity_cascade_separation),
        ("run_nonuniform_cascade_validation", experiments::run_nonuniform_cascade_validation),
        ("run_adaptive_cascade_correction", experiments::run_adaptive_cascade_correction),
        ("run_cycle_weight_cascade_prediction", experiments::run_cycle_weight_cascade_prediction),
        ("run_prediction_error_structure", experiments::run_prediction_error_structure),
        ("run_d0_corrected_prediction", experiments::run_d0_corrected_prediction),
        ("run_d0_correction_generalization", experiments::run_d0_correction_generalization),
        ("run_d0_coefficient_analysis", experiments::run_d0_coefficient_analysis),
        ("run_analytical_d0_prediction", experiments::run_analytical_d0_prediction),
        ("run_rho_fp_topology_prediction", experiments::run_rho_fp_topology_prediction),
    ];
    if let Some((_, f)) = fns.iter().find(|(n, _)| *n == name) {
        f();
    } else {
        eprintln!("Unknown experiment: {}", name);
        eprintln!("Available experiments:");
        for (n, _) in fns { eprintln!("  {}", n); }
    }
}

fn run_synthetic_suite() {
    println!("{}", "=".repeat(64));
    println!("  synthetic lattice experiment suite");
    println!("{}", "=".repeat(64));

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
    experiments::run_analytical_d0_prediction();
    experiments::run_rho_fp_topology_prediction();
}
