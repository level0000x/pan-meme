use nalgebra::SVector;

pub type State5 = SVector<f64, 5>;

/// 内禀度 D ∈ [0, 1]
pub fn d_of(m: &State5) -> f64 { m[0] }
/// 关联度 B ∈ [0, 1]
pub fn b_of(m: &State5) -> f64 { m[1] }
/// 能流密度 ρ ∈ [0, ∞)
pub fn rho_of(m: &State5) -> f64 { m[2] }
/// 演化速率 R ∈ [0, 1]
pub fn r_of(m: &State5) -> f64 { m[3] }
/// 结构韧度 S ∈ [0, 1]
pub fn s_of(m: &State5) -> f64 { m[4] }

pub fn make_state(d: f64, b: f64, rho: f64, r: f64, s: f64) -> State5 {
    State5::new(
        nan_guard(d),
        nan_guard(b),
        nan_guard(rho),
        nan_guard(r),
        nan_guard(s),
    )
}

pub fn is_valid(m: &State5) -> bool {
    m[0] >= 0.0 && m[0] <= 1.0
        && m[1] >= 0.0 && m[1] <= 1.0
        && m[2] >= 0.0
        && m[3] >= 0.0 && m[3] <= 1.0
        && m[4] >= 0.0 && m[4] <= 1.0
}

pub fn clamp_to_omega(m: &State5) -> State5 {
    State5::new(
        m[0].clamp(0.0, 1.0),
        m[1].clamp(0.0, 1.0),
        m[2].max(0.0),
        m[3].clamp(0.0, 1.0),
        m[4].clamp(0.0, 1.0),
    )
}

pub fn total_change(a: &State5, b: &State5) -> f64 {
    (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs()
        + (a[3] - b[3]).abs() + (a[4] - b[4]).abs()
}

pub fn to_array(m: &State5) -> [f64; 5] {
    [m[0], m[1], m[2], m[3], m[4]]
}

pub fn from_array(arr: &[f64; 5]) -> State5 {
    make_state(arr[0], arr[1], arr[2], arr[3], arr[4])
}

fn nan_guard(v: f64) -> f64 {
    if v.is_nan() || v.is_infinite() { 0.0 } else { v }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_state() {
        assert!(is_valid(&make_state(0.5, 0.5, 1.0, 0.3, 0.8)));
    }

    #[test]
    fn test_invalid_state() {
        assert!(!is_valid(&make_state(1.5, 0.5, 1.0, 0.3, 0.8)));
    }

    #[test]
    fn test_nan_guard() {
        let s = make_state(f64::NAN, 0.5, 1.0, 0.3, 0.8);
        assert_eq!(s[0], 0.0);
    }

    #[test]
    fn test_clamp_to_omega() {
        let s = make_state(1.5, -0.5, -1.0, 2.0, 100.0);
        let c = clamp_to_omega(&s);
        assert_eq!(c[0], 1.0);
        assert_eq!(c[1], 0.0);
        assert_eq!(c[2], 0.0);
        assert_eq!(c[3], 1.0);
        assert_eq!(c[4], 1.0);
    }
}
