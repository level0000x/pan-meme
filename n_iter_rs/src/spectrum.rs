use nalgebra::DMatrix;
use std::collections::{HashMap, HashSet};
use crate::n_operator::DynamicsParams;

pub fn word_bigram_data(words: &[String], max_bigrams: usize)
    -> (Vec<String>, HashMap<usize, HashSet<usize>>, HashMap<usize, HashSet<usize>>)
{
    let mut unique: Vec<String> = Vec::new();
    let mut seen = HashSet::new();
    for w in words {
        if w.len() >= 4 && seen.insert(w.clone()) {
            unique.push(w.clone());
        }
    }

    let mut bg_count: HashMap<String, usize> = HashMap::new();
    let mut w_bg_sets: HashMap<usize, HashSet<String>> = HashMap::new();
    for (wi, w) in unique.iter().enumerate() {
        let mut bgs = HashSet::new();
        let chars: Vec<char> = w.chars().collect();
        for i in 0..chars.len().saturating_sub(1) {
            let bg = format!("{}{}", chars[i], chars[i + 1]);
            bgs.insert(bg);
        }
        for bg in &bgs {
            *bg_count.entry(bg.clone()).or_insert(0) += 1;
        }
        w_bg_sets.insert(wi, bgs);
    }

    let mut freq: Vec<(&String, &usize)> = bg_count.iter().collect();
    freq.sort_by(|a, b| b.1.cmp(a.1));
    let top_bigrams: Vec<String> = freq.iter()
        .take(max_bigrams)
        .map(|(s, _)| (*s).clone())
        .collect();

    let bg_to_idx: HashMap<String, usize> = top_bigrams.iter().enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();

    let mut w_bg_idxs: HashMap<usize, HashSet<usize>> = HashMap::new();
    for (wi, bgs) in &w_bg_sets {
        let idxs: HashSet<usize> = bgs.iter()
            .filter_map(|bg| bg_to_idx.get(bg).copied())
            .collect();
        if !idxs.is_empty() {
            w_bg_idxs.insert(*wi, idxs);
        }
    }

    let mut bg_to_w: HashMap<usize, HashSet<usize>> = HashMap::new();
    for (wi, idxs) in &w_bg_idxs {
        for &bi in idxs {
            bg_to_w.entry(bi).or_default().insert(*wi);
        }
    }

    (top_bigrams, bg_to_w, w_bg_idxs)
}

pub fn build_concept_subgraph(
    word_indices: &[usize],
    w_bg_idxs: &HashMap<usize, HashSet<usize>>,
) -> (Vec<f64>, Vec<f64>) {
    let m = word_indices.len();
    if m <= 1 {
        return (vec![1.0], vec![0.0]);
    }

    let wlist: Vec<usize> = word_indices.to_vec();
    let _old_to_new: HashMap<usize, usize> = wlist.iter().enumerate()
        .map(|(new, &old)| (old, new))
        .collect();

    let mut a = DMatrix::zeros(m, m);
    for i in 0..m {
        let bg_i = w_bg_idxs.get(&wlist[i]);
        if bg_i.is_none() { continue; }
        let bg_i = bg_i.unwrap();
        for j in (i + 1)..m {
            let bg_j = w_bg_idxs.get(&wlist[j]);
            if bg_j.is_none() { continue; }
            let shared = bg_i.intersection(bg_j.unwrap()).count();
            if shared > 0 {
                a[(i, j)] = shared as f64;
                a[(j, i)] = shared as f64;
            }
        }
    }

    let d_vec: Vec<f64> = (0..m).map(|i| a.row(i).sum()).collect();
    let mut d_inv_sqrt = DMatrix::zeros(m, m);
    for i in 0..m {
        if d_vec[i] > 0.0 {
            d_inv_sqrt[(i, i)] = 1.0 / d_vec[i].sqrt();
        }
    }

    let i_mat = DMatrix::identity(m, m);
    let l = i_mat - &d_inv_sqrt * &a * &d_inv_sqrt;

    let eigvals = l.symmetric_eigenvalues();
    (d_vec, eigvals.as_slice().to_vec())
}

pub fn spectral_params(
    eigvals: &[f64],
    _d: &[f64],
    n_words: usize,
) -> DynamicsParams {
    let m = eigvals.len();
    if m <= 1 {
        return DynamicsParams::uniform();
    }

    let beta0 = eigvals.iter().filter(|&&v| v < 1e-10).count().max(1);
    let lambda1_val = if beta0 < m { eigvals[beta0] } else { eigvals[m - 1] };
    let lambda_max = if m > 0 { eigvals[m - 1] } else { 2.0 };

    let lambda1_val = lambda1_val.max(1e-10);
    let lambda_max = lambda_max.max(1.1 * lambda1_val);

    let t_star = if lambda1_val > 1e-10 {
        (beta0 as f64 + 1.0).ln() / lambda1_val
    } else {
        1.0
    };

    let theta_half: f64 = eigvals.iter()
        .map(|&v| (-t_star / 2.0 * v).exp())
        .filter(|x| x.is_finite())
        .sum();

    let _theta_star: f64 = eigvals.iter()
        .map(|&v| (-t_star * v).exp())
        .filter(|x| x.is_finite())
        .sum();

    let theta_2: f64 = eigvals.iter()
        .map(|&v| (-2.0 * t_star * v).exp())
        .filter(|x| x.is_finite())
        .sum();

    let _theta_3: f64 = eigvals.iter()
        .map(|&v| (-3.0 * t_star * v).exp())
        .filter(|x| x.is_finite())
        .sum();

    let nw = n_words.max(1) as f64;

    let alpha1_val = lambda1_val;
    let beta1_val = lambda_max;
    let gamma1_val = theta_half / nw;
    let delta1_val = lambda1_val;
    let zeta1_val = theta_half / nw;
    let eta1_val = (lambda_max - lambda1_val).max(0.1);
    let theta1_val = theta_half / beta0.max(1) as f64;
    let kappa1_val = lambda_max;
    let kappa2_val = lambda1_val / lambda_max.max(0.1);
    let lambda1_p_val = theta_2 / beta0.max(1) as f64;
    let mu1_val = 1.0 - beta0 as f64 / nw;

    let raw = [
        alpha1_val, beta1_val, gamma1_val, delta1_val,
        zeta1_val, eta1_val, theta1_val,
        kappa1_val, kappa2_val, lambda1_p_val, mu1_val,
    ];

    let (lo, hi) = (0.5_f64, 3.0_f64);
    let p_min = raw.iter().cloned().fold(f64::INFINITY, f64::min);
    let p_max = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let scaled: Vec<f64> = if p_max > p_min {
        raw.iter().map(|&v| lo + (hi - lo) * (v - p_min) / (p_max - p_min)).collect()
    } else {
        vec![(lo + hi) / 2.0; 11]
    };

    DynamicsParams {
        alpha1: scaled[0], beta1: scaled[1], gamma1: scaled[2],
        delta1: scaled[3], zeta1: scaled[4], eta1: scaled[5],
        theta1: scaled[6], kappa1: scaled[7], kappa2: scaled[8],
        lambda1: scaled[9], mu1: scaled[10],
        eps: 0.01,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bigram_data() {
        let words: Vec<String> = vec!["hello", "world", "hello"]
            .into_iter().map(|s| s.to_string()).collect();
        let (bigrams, bg_to_w, _w_bg_idxs) = word_bigram_data(&words, 40);
        assert!(!bigrams.is_empty());
        assert!(!bg_to_w.is_empty());
    }

    #[test]
    fn test_spectral_params() {
        let eigvals = vec![0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0];
        let d = vec![1.0; 7];
        let p = spectral_params(&eigvals, &d, 7);
        assert!(p.alpha1 > 0.0);
        assert!(p.beta1 > 0.0);
        assert!(p.mu1 >= 0.0 && p.mu1 <= 3.0);
        assert!(p.eps == 0.01);
    }
}
