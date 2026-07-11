use nalgebra::{SMatrix, ComplexField};
use crate::five_dim::{self, State5};

pub type Jacobian5 = SMatrix<f64, 5, 5>;

#[derive(Debug, Clone)]
pub struct DynamicsParams {
    pub alpha1: f64,
    pub beta1: f64,
    pub gamma1: f64,
    pub delta1: f64,
    pub zeta1: f64,
    pub eta1: f64,
    pub theta1: f64,
    pub kappa1: f64,
    pub kappa2: f64,
    pub lambda1: f64,
    pub mu1: f64,
    pub eps: f64,
}

impl DynamicsParams {
    pub fn uniform() -> Self {
        Self {
            alpha1: 1.0, beta1: 1.0, gamma1: 1.0, delta1: 1.0,
            zeta1: 1.0, eta1: 1.0, theta1: 1.0, kappa1: 1.0,
            kappa2: 1.0, lambda1: 1.0, mu1: 1.0, eps: 0.01,
        }
    }

    pub fn with_beta1(&self, beta1: f64) -> Self {
        Self { beta1, ..*self }
    }

    pub fn with_delta1(&self, delta1: f64) -> Self {
        Self { delta1, ..*self }
    }

    pub fn with_kappa1(&self, kappa1: f64) -> Self {
        Self { kappa1, ..*self }
    }
}

pub fn n_operator(m: &State5, b_up: f64, rho_up: f64, p: &DynamicsParams) -> State5 {
    let d = five_dim::d_of(m);
    let b = five_dim::b_of(m);
    let rho = five_dim::rho_of(m);
    let r = five_dim::r_of(m);
    let s = five_dim::s_of(m);
    let eps = p.eps;

    let num_d = p.alpha1 * r + eps;
    let den_d = num_d + p.beta1 * (b + b_up);

    let num_b = p.gamma1 * (r + b_up) + eps;
    let den_b = num_b + p.delta1 * d;

    let num_rho = p.zeta1 * (d + rho_up) + eps;
    let den_rho = num_rho + p.eta1 * r;

    let num_r = p.theta1 * (rho + rho_up + b_up) + eps;
    let den_r = num_r + p.kappa1 * d + p.kappa2 * s;

    let num_s = p.lambda1 * d + eps;
    let den_s = num_s + p.mu1 * r;

    five_dim::make_state(
        num_d / den_d,
        num_b / den_b,
        num_rho / den_rho,
        num_r / den_r,
        num_s / den_s,
    )
}

pub fn compute_jacobian(m_star: &State5, b_up: f64, rho_up: f64, p: &DynamicsParams) -> Jacobian5 {
    let d = five_dim::d_of(m_star);
    let b = five_dim::b_of(m_star);
    let rho = five_dim::rho_of(m_star);
    let r = five_dim::r_of(m_star);
    let s = five_dim::s_of(m_star);
    let eps = p.eps;

    let den_d = p.alpha1 * r + p.beta1 * (b + b_up) + eps;
    let den_b = p.gamma1 * (r + b_up) + p.delta1 * d + eps;
    let den_rho = p.zeta1 * (d + rho_up) + p.eta1 * r + eps;
    let den_r = p.theta1 * (rho + rho_up + b_up) + p.kappa1 * d + p.kappa2 * s + eps;
    let den_s = p.lambda1 * d + p.mu1 * r + eps;

    let mut j = Jacobian5::zeros();

    j[(0, 1)] = -p.beta1 * d / den_d;
    j[(0, 3)] = p.alpha1 * (1.0 - d) / den_d;

    j[(1, 0)] = -p.delta1 * b / den_b;
    j[(1, 3)] = p.gamma1 * (1.0 - b) / den_b;

    j[(2, 0)] = p.zeta1 * (1.0 - rho) / den_rho;
    j[(2, 3)] = -p.eta1 * rho / den_rho;

    j[(3, 0)] = -p.kappa1 * r / den_r;
    j[(3, 2)] = p.theta1 * (1.0 - r) / den_r;
    j[(3, 4)] = -p.kappa2 * r / den_r;

    j[(4, 0)] = p.lambda1 * (1.0 - s) / den_s;
    j[(4, 3)] = -p.mu1 * s / den_s;

    j
}

#[derive(Debug, Clone)]
pub struct IterResult {
    pub converged: bool,
    pub m_star: State5,
    pub rho_spectral: f64,
    pub tau_inv: f64,
    pub n_iters: usize,
    pub trajectory: Vec<[f64; 5]>,
    pub jacobian: [f64; 25],
}

pub fn run_iteration(
    m0: &State5,
    b_up: f64,
    rho_up: f64,
    p: &DynamicsParams,
    max_iter: usize,
    tol: f64,
) -> IterResult {
    let mut m = *m0;
    let mut trajectory: Vec<[f64; 5]> = vec![five_dim::to_array(&m)];

    for k in 0..max_iter {
        let m_next = n_operator(&m, b_up, rho_up, p);
        trajectory.push(five_dim::to_array(&m_next));
        if (m_next - m).abs().max() < tol {
            m = m_next;
            let j = compute_jacobian(&m, b_up, rho_up, p);
            let eigs = j.complex_eigenvalues();
            let rho_val = eigs.iter().map(|c| c.modulus()).fold(0.0_f64, f64::max);
            let tau = -f64::ln(rho_val.max(1e-10));
            let mut j_arr = [0.0_f64; 25];
            for r in 0..5 { for c in 0..5 { j_arr[r * 5 + c] = j[(r, c)]; } }
            return IterResult {
                converged: true,
                m_star: m,
                rho_spectral: rho_val,
                tau_inv: tau,
                n_iters: k + 1,
                trajectory,
                jacobian: j_arr,
            };
        }
        m = m_next;
    }

    let j = compute_jacobian(&m, b_up, rho_up, p);
    let eigs = j.complex_eigenvalues();
    let rho_val = eigs.iter().map(|c| c.modulus()).fold(0.0_f64, f64::max);
    let tau = -f64::ln(rho_val.max(1e-10));
    let mut j_arr = [0.0_f64; 25];
    for r in 0..5 { for c in 0..5 { j_arr[r * 5 + c] = j[(r, c)]; } }
    IterResult {
        converged: false,
        m_star: m,
        rho_spectral: rho_val,
        tau_inv: tau,
        n_iters: max_iter,
        trajectory,
        jacobian: j_arr,
    }
}

impl IterResult {
    pub fn verify_trajectory_conservation(&self, p: &DynamicsParams, b_up: f64, rho_up: f64) -> Option<f64> {
        if self.trajectory.len() < 2 {
            return None;
        }
        let mut max_dev = 0.0_f64;
        for w in self.trajectory.windows(2) {
            let m_prev = five_dim::from_array(&w[0]);
            let m_expected = n_operator(&m_prev, b_up, rho_up, p);
            let dev = five_dim::total_change(&m_expected, &five_dim::from_array(&w[1]));
            if dev > max_dev { max_dev = dev; }
        }
        Some(max_dev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_uniform() {
        let p = DynamicsParams::uniform();
        let m0 = five_dim::make_state(0.5, 0.5, 0.3, 0.5, 0.5);
        let r = run_iteration(&m0, 0.0, 0.0, &p, 500, 1e-12);
        assert!(r.converged);
        assert!(five_dim::is_valid(&r.m_star));
        assert!(r.rho_spectral >= 0.0 && r.rho_spectral <= 1.0);
        assert!(r.tau_inv >= 0.0);
        assert!(r.trajectory.len() >= 2);
    }

    #[test]
    fn test_trajectory_conservation() {
        let p = DynamicsParams::uniform();
        let m0 = five_dim::make_state(0.5, 0.5, 0.3, 0.5, 0.5);
        let r = run_iteration(&m0, 0.0, 0.0, &p, 500, 1e-12);
        let dev = r.verify_trajectory_conservation(&p, 0.0, 0.0).unwrap();
        assert!(dev < 1e-10, "Trajectory deviation {} exceeds tolerance", dev);
    }

    #[test]
    fn test_jacobian_spectral_radius() {
        let p = DynamicsParams::uniform();
        let m_star = five_dim::make_state(0.4, 0.5, 0.3, 0.2, 0.5);
        let j = compute_jacobian(&m_star, 0.0, 0.0, &p);
        let eigs = j.complex_eigenvalues();
        let rho = eigs.iter().map(|c| c.modulus()).fold(0.0_f64, f64::max);
        assert!(rho > 0.0 && rho < 2.0);
    }

    #[test]
    fn test_n_operator_in_domain() {
        let p = DynamicsParams::uniform();
        for _ in 0..100 {
            let d = rand_f64();
            let b = rand_f64();
            let rho = rand_f64();
            let r = rand_f64();
            let s = rand_f64();
            let m = five_dim::make_state(d, b, rho, r, s);
            let m_next = n_operator(&m, 0.0, 0.0, &p);
            for i in 0..5 {
                assert!(m_next[i] > 0.0 && m_next[i] < 1.0,
                        "N_operator({})[{}]={} ∉ (0,1)", i, i, m_next[i]);
            }
        }
    }

    fn rand_f64() -> f64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            ^ 0xDEADBEEF)
            .hash(&mut h);
        (h.finish() & 0xFFFF_FFFF) as f64 / 0xFFFF_FFFFu64 as f64
    }
}
