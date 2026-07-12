use crate::fca::FcaLattice;
use crate::five_dim::{self, State5};
use crate::n_operator::{self, DynamicsParams, IterResult};

pub struct LatticeStats {
    pub heights: Vec<usize>,
    pub feeders: Vec<Vec<usize>>,
    pub total_extent: usize,
    pub total_intent: usize,
    pub max_d_valid: f64,
}

pub fn compute_lattice_stats(lattice: &FcaLattice) -> LatticeStats {
    let n = lattice.concepts.len();
    let heights = crate::fca::hasse_heights(n, &lattice.edges);
    let mut feeders: Vec<Vec<usize>> = vec![vec![]; n];
    for &(p, c) in &lattice.edges {
        feeders[p].push(c);
    }
    let total_extent: usize = lattice.concept_sizes.iter().map(|(_, b)| *b).sum();
    let total_intent: usize = lattice.concept_sizes.iter().map(|(a, _)| *a).sum();
    let valid_d: Vec<f64> = lattice
        .d_values
        .iter()
        .filter(|&&d| d.is_finite() && d < 1e6)
        .copied()
        .collect();
    let max_d_valid = valid_d.iter().cloned().fold(1.0_f64, f64::max);
    LatticeStats {
        heights,
        feeders,
        total_extent,
        total_intent,
        max_d_valid,
    }
}

pub fn init_state(ci: usize, lattice: &FcaLattice, stats: &LatticeStats) -> five_dim::State5 {
    let csize = &lattice.concept_sizes[ci];
    let raw_d = lattice.d_values[ci];
    let d_init = if raw_d.is_finite() && raw_d < 1e6 {
        (raw_d / stats.max_d_valid).min(1.0)
    } else {
        0.8
    };
    let b_init = (1.0 - csize.1 as f64 / stats.total_extent.max(1) as f64).clamp(0.0, 1.0);
    let rho_init = (csize.0 as f64 / stats.total_intent.max(1) as f64).clamp(0.0, 1.0);
    five_dim::make_state(d_init, b_init, rho_init, 0.5, 0.5)
}

pub fn get_upstream(
    ci: usize,
    feeders: &[Vec<usize>],
    results: &[Option<IterResult>],
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
        if cnt > 0 {
            b_up = sums.0 / cnt as f64;
            rho_up = sums.1 / cnt as f64;
        }
    }
    (b_up, rho_up)
}

pub fn run_topological_iteration(
    lattice: &FcaLattice,
    stats: &LatticeStats,
    params: &DynamicsParams,
) -> Vec<Option<IterResult>> {
    let n = lattice.concepts.len();
    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by_key(|&i| std::cmp::Reverse(stats.heights[i]));
    let mut results: Vec<Option<IterResult>> = vec![None; n];

    for &ci in &sorted {
        let m0 = init_state(ci, lattice, stats);
        let (b_up, rho_up) = get_upstream(ci, &stats.feeders, &results);
        let result = n_operator::run_iteration(&m0, b_up, rho_up, params, 500, 1e-12);
        results[ci] = Some(result);
    }
    results
}

pub fn extract_scalars(
    results: &[Option<IterResult>],
    edges: &[(usize, usize)],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let tau_inv: Vec<f64> = results
        .iter()
        .filter_map(|r| r.as_ref())
        .map(|r| r.tau_inv)
        .collect();
    let rho_j: Vec<f64> = results
        .iter()
        .filter_map(|r| r.as_ref())
        .map(|r| r.rho_spectral)
        .collect();
    let dstar: Vec<f64> = results
        .iter()
        .filter_map(|r| r.as_ref())
        .map(|r| r.m_star[0])
        .collect();
    let _ = edges;
    (tau_inv, rho_j, dstar)
}

pub fn compute_edge_tau(
    results: &[Option<IterResult>],
    edges: &[(usize, usize)],
    num_concepts: usize,
) -> Vec<f64> {
    let m_stars: Vec<Option<State5>> = results
        .iter()
        .map(|r| r.as_ref().map(|ir| ir.m_star))
        .collect();

    let mut out_count = vec![0usize; num_concepts];
    let mut tau_sum = vec![0.0_f64; num_concepts];

    for &(gen, spec) in edges {
        if let (Some(m_gen), Some(m_spec)) = (m_stars[gen], m_stars[spec]) {
            let tau_e = if m_spec[0] <= m_gen[0] { 1.0 } else { 0.0 };
            tau_sum[gen] += tau_e;
            out_count[gen] += 1;
        }
    }

    (0..num_concepts)
        .map(|i| {
            if out_count[i] > 0 {
                tau_sum[i] / out_count[i] as f64
            } else {
                f64::NAN
            }
        })
        .collect()
}

pub fn mean(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.iter().sum::<f64>() / vals.len() as f64
}

pub fn std_dev(vals: &[f64]) -> f64 {
    let n = vals.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let avg = mean(vals);
    let var = vals.iter().map(|x| (x - avg).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

pub fn pct(passes: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        100.0 * passes as f64 / total as f64
    }
}
