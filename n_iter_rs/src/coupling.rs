use crate::five_dim::State5;

pub fn compute_coupling(
    state_i: &State5,
    state_j: &State5,
    shared: usize,
    total_i: usize,
    total_j: usize,
) -> f64 {
    if total_i == 0 || total_j == 0 {
        return 0.0;
    }

    let union = total_i + total_j - shared;
    let jaccard = if union > 0 {
        shared as f64 / union as f64
    } else {
        0.0
    };

    let state_diff = ((state_i[0] - state_j[0]).powi(2)
        + (state_i[1] - state_j[1]).powi(2)
        + (state_i[2] - state_j[2]).powi(2)
        + (state_i[3] - state_j[3]).powi(2)
        + (state_i[4] - state_j[4]).powi(2))
    .sqrt();

    let state_similarity = 1.0 / (1.0 + state_diff);

    let coupling = jaccard * state_similarity;
    if coupling.is_nan() || coupling.is_infinite() {
        0.0
    } else {
        coupling
    }
}

pub fn build_coupling_matrix(
    memes: &[State5],
    meme_vertices: &[Vec<usize>],
) -> Vec<Vec<f64>> {
    let n = memes.len();
    let mut c = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let shared = meme_vertices[i]
                .iter()
                .filter(|v| meme_vertices[j].contains(v))
                .count();
            let coupling = compute_coupling(
                &memes[i],
                &memes[j],
                shared,
                meme_vertices[i].len(),
                meme_vertices[j].len(),
            );
            c[i][j] = coupling;
            c[j][i] = coupling;
        }
    }

    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::five_dim;

    #[test]
    fn test_no_shared() {
        let a = five_dim::make_state(0.0, 0.0, 0.0, 0.0, 0.0);
        let b = five_dim::make_state(0.0, 0.0, 0.0, 0.0, 0.0);
        let c = compute_coupling(&a, &b, 0, 10, 10);
        assert!((c).abs() < 1e-10);
    }

    #[test]
    fn test_perfect_shared() {
        let a = five_dim::make_state(0.5, 0.5, 1.0, 0.3, 0.8);
        let b = five_dim::make_state(0.5, 0.5, 1.0, 0.3, 0.8);
        let c = compute_coupling(&a, &b, 10, 10, 10);
        assert!(c > 0.0);
        assert!(c <= 1.0);
    }

    #[test]
    fn test_coupling_matrix() {
        let memes = vec![
            five_dim::make_state(0.5, 0.5, 1.0, 0.3, 0.8),
            five_dim::make_state(0.5, 0.5, 1.0, 0.3, 0.8),
        ];
        let vertices = vec![vec![0, 1, 2], vec![2, 3, 4]];
        let c = build_coupling_matrix(&memes, &vertices);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].len(), 2);
        assert_eq!(c[0][1], c[1][0]);
    }
}
