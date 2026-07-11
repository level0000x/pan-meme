use crate::five_dim::{self, State5};
use crate::n_operator::{self, DynamicsParams};

/// Right-hand side of the ODE: dM/dt = N(M) - M
pub fn ode_rhs(m: &State5, b_up: f64, rho_up: f64, p: &DynamicsParams) -> State5 {
    n_operator::n_operator(m, b_up, rho_up, p) - m
}

/// Lyapunov function V(M) = 0.5 * [ (1-D)^2 + B^2 + (ρ-ρ*)^2 + R^2 + (1-S)^2 ]
/// where ρ* is the rho_up value (upstream energy density).
pub fn lyapunov(m: &State5, rho_up: f64) -> f64 {
    let d = five_dim::d_of(m);
    let b = five_dim::b_of(m);
    let rho = five_dim::rho_of(m);
    let r = five_dim::r_of(m);
    let s = five_dim::s_of(m);
    0.5 * ((1.0 - d).powi(2) + b.powi(2) + (rho - rho_up).powi(2) + r.powi(2) + (1.0 - s).powi(2))
}

/// Derivative of Lyapunov along the ODE flow: dV/dt = ∇V · f(M)
pub fn lyapunov_derivative(m: &State5, b_up: f64, rho_up: f64, p: &DynamicsParams) -> f64 {
    let f = ode_rhs(m, b_up, rho_up, p);
    let d = five_dim::d_of(m);
    let b = five_dim::b_of(m);
    let rho = five_dim::rho_of(m);
    let r = five_dim::r_of(m);
    let s = five_dim::s_of(m);
    // ∇V = ( -(1-D), B, (ρ-ρ_up), R, -(1-S) )
    let grad: [f64; 5] = [-(1.0 - d), b, rho - rho_up, r, -(1.0 - s)];
    grad[0] * f[0] + grad[1] * f[1] + grad[2] * f[2] + grad[3] * f[3] + grad[4] * f[4]
}

/// Lyapunov function V(M) = 0.5 * ||M - M*||^2
/// Uses the actual fixed point M* for proper monotonicity verification.
pub fn lyapunov_exact(m: &State5, m_star: &State5) -> f64 {
    0.5 * (m - m_star).norm_squared()
}

/// Derivative of exact Lyapunov along the ODE flow: dV/dt = (M - M*) · f(M)
pub fn lyapunov_exact_derivative(m: &State5, m_star: &State5, b_up: f64, rho_up: f64, p: &DynamicsParams) -> f64 {
    let f = ode_rhs(m, b_up, rho_up, p);
    let diff = m - m_star;
    diff[0] * f[0] + diff[1] * f[1] + diff[2] * f[2] + diff[3] * f[3] + diff[4] * f[4]
}

#[derive(Debug, Clone)]
pub struct OdeResult {
    pub m_steady: State5,
    pub converged: bool,
    pub n_steps: usize,
    pub t_final: f64,
    pub trajectory: Vec<[f64; 5]>,
    pub lyapunov_traj: Vec<f64>,
    pub lyapunov_monotonic: bool,
    pub domain_ok: bool,
}

/// Runge-Kutta 4th order solver for dM/dt = N(M) - M.
/// Stops when ||dM/dt|| < tol or max_steps reached.
/// If m_star is provided, uses exact Lyapunov V(M) = 0.5||M-M*||^2.
/// Otherwise uses the approximate V with rho_up as target.
pub fn solve_rk4(
    m0: &State5,
    b_up: f64,
    rho_up: f64,
    p: &DynamicsParams,
    dt: f64,
    max_steps: usize,
    tol: f64,
) -> OdeResult {
    solve_rk4_with_mstar(m0, b_up, rho_up, p, dt, max_steps, tol, None)
}

/// Extended RK4 solver with optional fixed point M* for exact Lyapunov tracking.
pub fn solve_rk4_with_mstar(
    m0: &State5,
    b_up: f64,
    rho_up: f64,
    p: &DynamicsParams,
    dt: f64,
    max_steps: usize,
    tol: f64,
    m_star: Option<&State5>,
) -> OdeResult {
    let use_exact = m_star.is_some();
    let m_star_ref = m_star.unwrap_or(m0); // fallback for approximate mode

    let mut m = *m0;
    let mut trajectory: Vec<[f64; 5]> = vec![five_dim::to_array(&m)];
    let mut lyapunov_traj: Vec<f64> = vec![
        if use_exact { lyapunov_exact(&m, m_star_ref) } else { lyapunov(&m, rho_up) }
    ];
    let mut domain_ok = five_dim::is_valid(m0);
    let mut lyap_monotonic = true;

    let mut converged = false;
    let mut n_steps = 0usize;

    for step in 0..max_steps {
        let k1 = ode_rhs(&m, b_up, rho_up, p);
        let k2 = ode_rhs(&(m + k1 * (dt / 2.0)), b_up, rho_up, p);
        let k3 = ode_rhs(&(m + k2 * (dt / 2.0)), b_up, rho_up, p);
        let k4 = ode_rhs(&(m + k3 * dt), b_up, rho_up, p);

        let m_next = m + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);

        // Clamp to Ω for stability
        let m_clamped = five_dim::clamp_to_omega(&m_next);

        trajectory.push(five_dim::to_array(&m_clamped));
        let v = if use_exact { lyapunov_exact(&m_clamped, m_star_ref) } else { lyapunov(&m_clamped, rho_up) };
        lyapunov_traj.push(v);

        if !five_dim::is_valid(&m_clamped) {
            domain_ok = false;
        }

        if step > 0 && v > lyapunov_traj[step - 1] + 1e-12 {
            lyap_monotonic = false;
        }

        n_steps = step + 1;

        // Check convergence: ||dM/dt|| ≈ ||M_{k+1} - M_k|| / dt < tol
        let change = five_dim::total_change(&m_clamped, &m);
        if change / dt < tol {
            converged = true;
            m = m_clamped;
            break;
        }

        m = m_clamped;
    }

    OdeResult {
        m_steady: m,
        converged,
        n_steps,
        t_final: n_steps as f64 * dt,
        trajectory,
        lyapunov_traj,
        lyapunov_monotonic: lyap_monotonic,
        domain_ok,
    }
}

/// Compare discrete fixed point M*_discrete with ODE steady state M*_ode.
/// Returns ||M*_ode - M*_discrete|| (Euclidean distance).
pub fn compare_fixed_points(m_discrete: &State5, m_ode: &State5) -> f64 {
    (m_ode - m_discrete).norm()
}

/// Compute the ODE Jacobian at a fixed point: J_ode = J_N - I
/// where J_N = ∂N/∂M is the discrete N-operator Jacobian.
/// Stability in continuous time requires Re(λ(J_ode)) < 0 for all λ.
pub fn ode_jacobian(m_star: &State5, b_up: f64, rho_up: f64, p: &DynamicsParams) -> n_operator::Jacobian5 {
    use nalgebra::SMatrix;
    let j_n = n_operator::compute_jacobian(m_star, b_up, rho_up, p);
    let i5 = SMatrix::<f64, 5, 5>::identity();
    j_n - i5
}

/// Classification of an ODE fixed point.
#[derive(Debug, Clone)]
pub struct OdeFixedPointClass {
    pub eigenvalues: Vec<nalgebra::Complex<f64>>,
    pub max_re: f64,           // max real part of eigenvalues
    pub min_re: f64,           // min real part of eigenvalues
    pub spectral_radius: f64,  // max |λ| (for comparison with discrete)
    pub all_negative: bool,    // Re(λ) < 0 for all λ → stable
    pub has_imaginary: bool,   // any eigenvalue has nonzero imaginary part
    pub classification: &'static str,  // "stable node", "stable spiral", "saddle", etc.
}

/// Classify an ODE fixed point by computing eigenvalues of J_ode = J_N - I.
pub fn classify_ode_fixed_point(m_star: &State5, b_up: f64, rho_up: f64, p: &DynamicsParams) -> OdeFixedPointClass {
    let j_ode = ode_jacobian(m_star, b_up, rho_up, p);
    let eigs = j_ode.complex_eigenvalues();
    let eig_vec: Vec<nalgebra::Complex<f64>> = eigs.iter().copied().collect();

    let reals: Vec<f64> = eigs.iter().map(|c| c.re).collect();
    let max_re = reals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_re = reals.iter().cloned().fold(f64::INFINITY, f64::min);
    let spectral_radius = eigs.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
    let all_negative = reals.iter().all(|&re| re < 0.0);
    let has_imaginary = eigs.iter().any(|c| c.im.abs() > 1e-12);

    let classification = if all_negative {
        if has_imaginary { "stable spiral" } else { "stable node" }
    } else if max_re > 0.0 {
        if has_imaginary { "unstable spiral" } else { "saddle" }
    } else {
        "center/degenerate"
    };

    OdeFixedPointClass {
        eigenvalues: eig_vec,
        max_re,
        min_re,
        spectral_radius,
        all_negative,
        has_imaginary,
        classification,
    }
}

/// Verify the correspondence between discrete and continuous stability:
/// λ_ode = λ_N - 1 for each eigenvalue pair.
/// Computes J_N once, then derives J_ode = J_N - I to ensure numerical consistency.
pub fn verify_stability_correspondence(m_star: &State5, b_up: f64, rho_up: f64, p: &DynamicsParams) -> bool {
    use nalgebra::SMatrix;
    let j_n = n_operator::compute_jacobian(m_star, b_up, rho_up, p);
    let i5 = SMatrix::<f64, 5, 5>::identity();
    let j_ode = j_n - i5;

    let mut eigs_n: Vec<nalgebra::Complex<f64>> = j_n.complex_eigenvalues().iter().copied().collect();
    let mut eigs_ode: Vec<nalgebra::Complex<f64>> = j_ode.complex_eigenvalues().iter().copied().collect();

    // Sort by (real, imag) for deterministic ordering
    eigs_n.sort_by(|a, b| {
        a.re.partial_cmp(&b.re).unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal))
    });
    eigs_ode.sort_by(|a, b| {
        a.re.partial_cmp(&b.re).unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal))
    });

    for (ln, lo) in eigs_n.iter().zip(eigs_ode.iter()) {
        let expected = ln - nalgebra::Complex::new(1.0, 0.0);
        let diff = (lo - expected).norm();
        if diff > 1e-8 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ode_rhs_zero_at_fixed_point() {
        let p = DynamicsParams::uniform();
        let b_up = 0.0;
        let rho_up = 0.0;

        // Find fixed point via discrete iteration
        let m0 = five_dim::make_state(0.5, 0.5, 0.5, 0.5, 0.5);
        let result = n_operator::run_iteration(&m0, b_up, rho_up, &p, 500, 1e-12);
        let m_star = result.m_star;

        // At fixed point, ODE RHS should be zero
        let rhs = ode_rhs(&m_star, b_up, rho_up, &p);
        let rhs_norm = rhs.norm();
        assert!(rhs_norm < 1e-10, "ODE RHS at fixed point = {:.2e}, expected < 1e-10", rhs_norm);
    }

    #[test]
    fn test_ode_converges_to_fixed_point() {
        let p = DynamicsParams::uniform();
        let b_up = 0.0;
        let rho_up = 0.0;

        let m0 = five_dim::make_state(0.5, 0.5, 0.5, 0.5, 0.5);
        let discrete = n_operator::run_iteration(&m0, b_up, rho_up, &p, 500, 1e-12);

        let ode = solve_rk4(&m0, b_up, rho_up, &p, 0.01, 5000, 1e-8);
        let diff = compare_fixed_points(&discrete.m_star, &ode.m_steady);

        assert!(diff < 1e-6, "|M*_ode - M*_discrete| = {:.2e}, expected < 1e-6", diff);
        assert!(ode.converged, "ODE should converge");
        assert!(ode.domain_ok, "ODE should stay in Omega");
    }

    #[test]
    fn test_lyapunov_monotonic() {
        let p = DynamicsParams::uniform();
        let b_up = 0.0;
        let rho_up = 0.0;

        let m0 = five_dim::make_state(0.3, 0.7, 0.2, 0.5, 0.4);
        let ode = solve_rk4(&m0, b_up, rho_up, &p, 0.01, 5000, 1e-8);

        // V should decrease monotonically
        for i in 1..ode.lyapunov_traj.len() {
            assert!(
                ode.lyapunov_traj[i] <= ode.lyapunov_traj[i - 1] + 1e-12,
                "V increased at step {}: {} -> {}",
                i, ode.lyapunov_traj[i - 1], ode.lyapunov_traj[i]
            );
        }
    }

    #[test]
    fn test_stability_correspondence() {
        let p = DynamicsParams::uniform();
        let b_up = 0.0;
        let rho_up = 0.0;

        let m0 = five_dim::make_state(0.5, 0.5, 0.5, 0.5, 0.5);
        let result = n_operator::run_iteration(&m0, b_up, rho_up, &p, 500, 1e-12);
        let m_star = result.m_star;

        // Verify λ_ode = λ_N - 1 at the fixed point
        assert!(verify_stability_correspondence(&m_star, b_up, rho_up, &p),
            "Stability correspondence failed: λ_ode ≠ λ_N - 1");

        // Classify the ODE fixed point
        let cls = classify_ode_fixed_point(&m_star, b_up, rho_up, &p);
        assert!(cls.all_negative, "ODE fixed point should be stable (Re(λ) < 0)");
        assert!(cls.max_re < 0.0, "max Re(λ) = {:.4}, expected < 0", cls.max_re);

        // Discrete spectral radius < 1 implies ODE stability
        assert!(result.rho_spectral < 1.0, "discrete ρ(J_N) = {:.4}, expected < 1", result.rho_spectral);
    }

    #[test]
    fn test_exact_lyapunov_monotonic() {
        let p = DynamicsParams::uniform();
        let b_up = 0.0;
        let rho_up = 0.0;

        let m0 = five_dim::make_state(0.3, 0.7, 0.2, 0.5, 0.4);
        // First find the discrete fixed point
        let discrete = n_operator::run_iteration(&m0, b_up, rho_up, &p, 500, 1e-12);
        let m_star = discrete.m_star;

        // Now run ODE with exact Lyapunov using the known fixed point
        let ode = solve_rk4_with_mstar(&m0, b_up, rho_up, &p, 0.01, 15000, 1e-8, Some(&m_star));

        assert!(ode.converged, "ODE should converge");
        assert!(ode.lyapunov_monotonic,
            "Exact V(M) = 0.5||M-M*||^2 should be strictly monotonic decreasing");

        // V should go to 0 at the fixed point
        let v_final = ode.lyapunov_traj.last().unwrap();
        assert!(v_final < 1e-12, "V at steady state = {:.2e}, expected < 1e-12", v_final);

        // dV/dt should be negative at each step
        for i in 1..ode.lyapunov_traj.len() {
            let dv = ode.lyapunov_traj[i] - ode.lyapunov_traj[i - 1];
            assert!(dv <= 1e-12, "dV/dt = {:.2e} at step {} (should be ≤ 0)", dv, i);
        }
    }
}