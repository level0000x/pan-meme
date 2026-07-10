use nalgebra::{DMatrix, DVector};

pub fn pca(x: &DMatrix<f64>, n_components: usize) -> (DMatrix<f64>, DVector<f64>, DMatrix<f64>) {
    let n = x.nrows();
    let mean = x.row_mean();
    let xc = DMatrix::from_fn(n, x.ncols(), |i, j| x[(i, j)] - mean[j]);
    let xc_scaled = &xc / ((n as f64 - 1.0).sqrt());
    let svd = xc_scaled.svd(true, true);
    let u = svd.u.unwrap();
    let s = svd.singular_values;
    let vt = svd.v_t.unwrap();

    let var = DVector::from_iterator(s.len(), s.iter().map(|v| v * v));
    let total_var = var.sum();
    let var_ratio = var / total_var;

    let k = n_components.min(s.len());
    let u_k = u.columns(0, k);
    let s_k = DMatrix::from_diagonal(&s.rows(0, k));
    let scores = u_k * s_k;
    let loadings = vt.rows(0, k).transpose();

    (scores, var_ratio, loadings)
}

pub fn nearest_centroid_cv(
    x: &DMatrix<f64>,
    y: &[usize],
    n_classes: usize,
    n_folds: usize,
    rng_seed: u64,
) -> (f64, Vec<usize>) {
    let n = x.nrows();
    let xc = {
        let mean = x.row_mean();
        let centered = DMatrix::from_fn(n, x.ncols(), |i, j| x[(i, j)] - mean[j]);
        let stds: Vec<f64> = (0..centered.ncols()).map(|j| {
            let col = centered.column(j);
            let s = col.norm() / (n as f64).sqrt();
            if s < 1e-12 { 1.0 } else { s }
        }).collect();
        let mut scaled = centered.clone();
        for j in 0..centered.ncols() {
            for i in 0..n {
                scaled[(i, j)] /= stds[j];
            }
        }
        scaled
    };

    let mut rng = RngState::new(rng_seed);
    let mut indices: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.next() % (i + 1);
        indices.swap(i, j);
    }

    let mut preds = vec![0usize; n];
    let mut correct = 0usize;

    for fold in 0..n_folds {
        let s = fold * n / n_folds;
        let e = (fold + 1) * n / n_folds;
        let mut tst = vec![false; n];
        for &idx in &indices[s..e] {
            tst[idx] = true;
        }

        let mut cents = vec![vec![0.0; xc.ncols()]; n_classes];
        let mut counts = vec![0usize; n_classes];
        for i in 0..n {
            if tst[i] { continue; }
            let ci = y[i];
            for j in 0..xc.ncols() {
                cents[ci][j] += xc[(i, j)];
            }
            counts[ci] += 1;
        }
        for ci in 0..n_classes {
            if counts[ci] > 0 {
                for j in 0..xc.ncols() {
                    cents[ci][j] /= counts[ci] as f64;
                }
            } else {
                for j in 0..xc.ncols() {
                    cents[ci][j] = f64::INFINITY;
                }
            }
        }

        let cent_dmat = DMatrix::from_fn(n_classes, xc.ncols(), |ci, j| cents[ci][j]);

        for &i in &indices[s..e] {
            let mut best_c = 0;
            let mut best_d2 = f64::INFINITY;
            let row = xc.row(i);
            for ci in 0..n_classes {
                if counts[ci] == 0 { continue; }
                let d2 = (0..xc.ncols()).map(|j| {
                    let diff = row[j] - cent_dmat[(ci, j)];
                    diff * diff
                }).sum::<f64>();
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_c = ci;
                }
            }
            preds[i] = best_c;
            if best_c == y[i] {
                correct += 1;
            }
        }
    }

    (correct as f64 / n as f64, preds)
}

pub fn kmeans(x: &DMatrix<f64>, k: usize, max_iter: usize, rng_seed: u64) -> Vec<usize> {
    let n = x.nrows();
    let mut rng = RngState::new(rng_seed);
    let mut assignments = vec![0usize; n];
    let mut centroids = vec![vec![0.0; x.ncols()]; k];

    let mut init_idxs = Vec::with_capacity(k);
    while init_idxs.len() < k {
        let idx = rng.next() % n;
        if !init_idxs.contains(&idx) {
            init_idxs.push(idx);
        }
    }
    for (ci, &idx) in init_idxs.iter().enumerate() {
        for j in 0..x.ncols() {
            centroids[ci][j] = x[(idx, j)];
        }
    }

    for _iter in 0..max_iter {
        let mut changed = false;
        for i in 0..n {
            let mut best_c = 0;
            let mut best_d2 = f64::INFINITY;
            for ci in 0..k {
                let d2 = (0..x.ncols()).map(|j| {
                    let diff = x[(i, j)] - centroids[ci][j];
                    diff * diff
                }).sum::<f64>();
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_c = ci;
                }
            }
            if assignments[i] != best_c {
                assignments[i] = best_c;
                changed = true;
            }
        }
        if !changed { break; }

        for ci in 0..k {
            for j in 0..x.ncols() {
                centroids[ci][j] = 0.0;
            }
        }
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let ci = assignments[i];
            for j in 0..x.ncols() {
                centroids[ci][j] += x[(i, j)];
            }
            counts[ci] += 1;
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                for j in 0..x.ncols() {
                    centroids[ci][j] /= counts[ci] as f64;
                }
            }
        }
    }

    assignments
}

pub fn cluster_purity(assignments: &[usize], y: &[usize], n_classes: usize) -> Vec<f64> {
    let k = assignments.iter().max().map(|&x| x + 1).unwrap_or(1);
    let mut purities = Vec::with_capacity(k);
    for cl in 0..k {
        let mut counts = vec![0usize; n_classes];
        let mut total = 0usize;
        for (a, &label) in assignments.iter().zip(y.iter()) {
            if *a == cl {
                counts[label] += 1;
                total += 1;
            }
        }
        if total > 0 {
            let max_c = counts.iter().max().copied().unwrap_or(0) as f64 / total as f64;
            purities.push(max_c);
        }
    }
    purities
}

struct RngState {
    state: u64,
}

impl RngState {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> usize {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.state >> 33) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_pca_dims() {
        let x = DMatrix::from_row_slice(6, 3, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            2.0, 3.0, 4.0,
            5.0, 6.0, 7.0,
            8.0, 9.0, 10.0,
        ]);
        let (scores, var, loadings) = pca(&x, 2);
        assert_eq!(scores.nrows(), 6);
        assert_eq!(scores.ncols(), 2);
        assert_eq!(loadings.nrows(), 3);
        assert!(var[0] > 0.0);
    }

    #[test]
    fn test_kmeans() {
        let x = DMatrix::from_row_slice(6, 2, &[
            0.0, 0.0,
            0.1, 0.1,
            0.2, 0.2,
            5.0, 5.0,
            5.1, 5.1,
            5.2, 5.2,
        ]);
        let labs = kmeans(&x, 2, 20, 42);
        assert_eq!(labs[0], labs[1]);
        assert_eq!(labs[0], labs[2]);
        assert_eq!(labs[3], labs[4]);
        assert_eq!(labs[3], labs[5]);
        assert_ne!(labs[0], labs[3]);
    }
}
