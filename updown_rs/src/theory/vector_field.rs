//! 离散梯度场 — 泛模因理论 §3.3.3
//!
//! 核心构造：
//! - ScalarField: φ: K⁰ → ℝ，标量场（信息深度）
//! - VectorField: F = -∇φ，离散梯度场

use crate::theory::cw_complex::CWComplex;

/// 标量场 φ: K⁰ → ℝ — 论文 §3.3.3
///
/// 标量场值由信息深度决定：φ(v) = depth(v) / max_depth
#[derive(Debug, Clone)]
pub struct ScalarField {
    /// 每个顶点的标量值
    pub values: Vec<f64>,
}

impl ScalarField {
    /// 从信息深度创建标量场
    pub fn from_depths(depths: &[f64]) -> Self {
        let max_depth = depths.iter().cloned().fold(0.0_f64, f64::max);
        let values: Vec<f64> = if max_depth > 0.0 {
            depths.iter().map(|&d| d / max_depth).collect()
        } else {
            depths.to_vec()
        };
        ScalarField { values }
    }

    /// 获取顶点 v 的标量值
    pub fn get(&self, v: usize) -> f64 {
        self.values.get(v).copied().unwrap_or(0.0)
    }
}

/// 离散梯度场 F = -∇φ — 论文 §3.3.3
///
/// 每条边的梯度 = -(φ(v₂) - φ(v₁))
#[derive(Debug, Clone)]
pub struct VectorField {
    /// 每条边的梯度值 (边索引 → 梯度)
    pub edge_gradients: Vec<f64>,
}

impl VectorField {
    /// 从 CW 复形和标量场计算离散梯度场
    ///
    /// 对每条边 e = (v₁, v₂)，梯度 = -(φ(v₂) - φ(v₁))
    pub fn compute(complex: &CWComplex, scalar: &ScalarField) -> Self {
        let mut edge_gradients = Vec::new();
        for cell in &complex.cells {
            if cell.dim == 1 && cell.boundary.len() == 2 {
                let v1 = cell.boundary[0];
                let v2 = cell.boundary[1];
                let phi1 = scalar.get(v1);
                let phi2 = scalar.get(v2);
                let grad = -(phi2 - phi1);
                let grad = if grad.is_nan() || grad.is_infinite() { 0.0 } else { grad };
                edge_gradients.push(grad);
            }
        }
        VectorField { edge_gradients }
    }

    /// 平均梯度值
    pub fn mean_gradient(&self) -> f64 {
        if self.edge_gradients.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.edge_gradients.iter().sum();
        sum / self.edge_gradients.len() as f64
    }

    /// 梯度标准差
    pub fn std_gradient(&self) -> f64 {
        if self.edge_gradients.is_empty() {
            return 0.0;
        }
        let mean = self.mean_gradient();
        let variance: f64 = self.edge_gradients.iter()
            .map(|&g| (g - mean).powi(2))
            .sum::<f64>() / self.edge_gradients.len() as f64;
        variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::cw_complex::CWComplex;

    #[test]
    fn test_scalar_field_from_depths() {
        let depths = vec![1.0, 2.0, 3.0];
        let sf = ScalarField::from_depths(&depths);
        assert!((sf.get(0) - 1.0 / 3.0).abs() < 1e-10);
        assert!((sf.get(1) - 2.0 / 3.0).abs() < 1e-10);
        assert!((sf.get(2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_field() {
        let mut c = CWComplex::new();
        let v0 = c.add_vertex();
        let v1 = c.add_vertex();
        c.add_edge(v0, v1);

        let sf = ScalarField { values: vec![0.0, 1.0] };
        let vf = VectorField::compute(&c, &sf);
        assert_eq!(vf.edge_gradients.len(), 1);
        // 梯度 = -(1.0 - 0.0) = -1.0
        assert!((vf.edge_gradients[0] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nan_guard() {
        let mut c = CWComplex::new();
        let v0 = c.add_vertex();
        let v1 = c.add_vertex();
        c.add_edge(v0, v1);

        let sf = ScalarField { values: vec![f64::NAN, 1.0] };
        let vf = VectorField::compute(&c, &sf);
        assert_eq!(vf.edge_gradients[0], 0.0);
    }
}